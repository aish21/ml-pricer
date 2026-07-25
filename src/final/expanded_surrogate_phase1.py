"""Production-parity label experiment for the expanded surrogate models.

This phase deliberately keeps the existing balanced LightGBM architecture
fixed. It changes the teacher data: labels use the production monitoring grid,
piecewise markets, independently scrambled Sobol replications, explicit label
uncertainty, and capped inverse-variance sample weights.

Nothing in this module can approve or package a runtime artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import lightgbm
import numpy as np
from lightgbm import LGBMRegressor
from scipy.stats import t as student_t
from sklearn.metrics import mean_absolute_error, r2_score

from src.final.barrier_reverse_convertible import (
    BARRIER_REVERSE_CONVERTIBLE_V1,
    BarrierReverseConvertiblePayoff,
    BarrierReverseConvertibleV1Contract,
)
from src.final.data_generator import (
    build_simulation_time_grid,
    simulate_piecewise_gbm_paths,
)
from src.final.market import EquityMarketSegment, EquityMarketTermStructure
from src.final.payoffs import PhoenixPayoff
from src.final.phoenix_contract import (
    PHOENIX_SINGLE_V3_CONTRACT_VERSION,
    PhoenixSingleV3Contract,
)
from src.final.reference_pricer import DEFAULT_REFERENCE_STEPS
from src.final.surrogate_data import standard_normal_shocks


LOGGER = logging.getLogger(__name__)
EXPERIMENT_VERSION = "expanded-surrogate-phase1-v1"
LABEL_PROTOCOL_VERSION = "expanded-production-parity-rqmc-label-v1"
WEIGHT_POLICY_VERSION = "capped-inverse-label-variance-v1"
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data" / "expanded_surrogate_phase1"
CURVE_TIME_FRACTIONS = (0.25, 0.50, 0.75, 1.0)
PRODUCT_KEYS = ("phoenix_v3", "barrier_reverse_convertible")

PHOENIX_FEATURE_ORDER = (
    "spot_to_reference",
    "risk_free_rate",
    "dividend_yield",
    "volatility",
    "maturity_years",
    "first_autocall_barrier_frac",
    "final_autocall_barrier_frac",
    "coupon_barrier_frac",
    "coupon_rate",
    "knock_in_frac",
    "observation_count",
    "memory_coupon",
    "unpaid_coupon_count",
    "prior_knock_in_breached",
    "spot_minus_first_autocall",
    "spot_minus_final_autocall",
    "spot_minus_coupon_barrier",
    "spot_minus_knock_in",
    "autocall_stepdown",
    "coupon_including_unpaid",
)

BRC_FEATURE_ORDER = (
    "spot_to_reference",
    "risk_free_rate",
    "dividend_yield",
    "volatility",
    "maturity_years",
    "coupon_rate_per_period",
    "strike_frac",
    "knock_in_frac",
    "coupon_count",
    "prior_knock_in_breached",
    "spot_minus_strike",
    "spot_minus_knock_in",
    "strike_minus_knock_in",
    "total_coupon_rate",
)


class ExpandedSurrogatePhase1Error(ValueError):
    pass


@dataclass(frozen=True)
class Phase1Config:
    development_samples: int = 2_500
    validation_samples: int = 500
    evaluation_samples: int = 300
    paths_per_replication: int = 128
    evaluation_paths_per_replication: int = 512
    label_replications: int = 8
    production_steps: int = DEFAULT_REFERENCE_STEPS
    confidence_level: float = 0.95
    maximum_weight_multiple: float = 10.0
    seed: int = 20_260_725
    trees: int = 1_000
    barrier_focus_probability: float = 0.60

    def __post_init__(self) -> None:
        for name in (
            "development_samples",
            "validation_samples",
            "evaluation_samples",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 4:
                raise ExpandedSurrogatePhase1Error(f"{name} must be an integer >= 4")
        for name in (
            "paths_per_replication",
            "evaluation_paths_per_replication",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 8
                or value & (value - 1)
            ):
                raise ExpandedSurrogatePhase1Error(
                    f"{name} must be a power of two and at least 8"
                )
        if (
            isinstance(self.label_replications, bool)
            or not isinstance(self.label_replications, int)
            or not 2 <= self.label_replications <= 32
        ):
            raise ExpandedSurrogatePhase1Error(
                "label_replications must be between 2 and 32"
            )
        if (
            isinstance(self.production_steps, bool)
            or not isinstance(self.production_steps, int)
            or self.production_steps != DEFAULT_REFERENCE_STEPS
        ):
            raise ExpandedSurrogatePhase1Error(
                f"production_steps must equal {DEFAULT_REFERENCE_STEPS}"
            )
        if not 0.80 <= self.confidence_level < 1.0:
            raise ExpandedSurrogatePhase1Error(
                "confidence_level must be in [0.80, 1.0)"
            )
        if not 1.0 <= self.maximum_weight_multiple <= 100.0:
            raise ExpandedSurrogatePhase1Error(
                "maximum_weight_multiple must be between 1 and 100"
            )
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or not 0 <= self.seed < 2**32
        ):
            raise ExpandedSurrogatePhase1Error("seed must be a uint32")
        if (
            isinstance(self.trees, bool)
            or not isinstance(self.trees, int)
            or not 1 <= self.trees <= 10_000
        ):
            raise ExpandedSurrogatePhase1Error("trees must be between 1 and 10000")
        if not 0.0 <= self.barrier_focus_probability <= 1.0:
            raise ExpandedSurrogatePhase1Error(
                "barrier_focus_probability must be in [0, 1]"
            )


@dataclass(frozen=True)
class TrainingCase:
    product_key: str
    market: EquityMarketTermStructure
    contract: PhoenixSingleV3Contract | BarrierReverseConvertibleV1Contract
    features: np.ndarray


@dataclass(frozen=True)
class ReplicatedLabel:
    price: float
    standard_error: float
    confidence_half_width: float
    effective_steps: int
    replication_means: tuple[float, ...]


@dataclass(frozen=True)
class Phase1Dataset:
    product_key: str
    role: str
    features: np.ndarray
    labels: np.ndarray
    label_standard_errors: np.ndarray
    confidence_interval_low: np.ndarray
    confidence_interval_high: np.ndarray
    effective_steps: np.ndarray
    market_payloads: tuple[str, ...]
    contract_payloads: tuple[str, ...]
    replication_means: np.ndarray
    dataset_id: str
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        samples = len(self.labels)
        feature_order = feature_order_for(self.product_key)
        if self.role not in {"development", "validation", "evaluation"}:
            raise ExpandedSurrogatePhase1Error("dataset role is invalid")
        if self.features.shape != (samples, len(feature_order)):
            raise ExpandedSurrogatePhase1Error("dataset feature shape is invalid")
        if self.replication_means.shape[0] != samples:
            raise ExpandedSurrogatePhase1Error(
                "replication means do not align with labels"
            )
        for name, values in (
            ("label_standard_errors", self.label_standard_errors),
            ("confidence_interval_low", self.confidence_interval_low),
            ("confidence_interval_high", self.confidence_interval_high),
            ("effective_steps", self.effective_steps),
            ("market_payloads", self.market_payloads),
            ("contract_payloads", self.contract_payloads),
        ):
            if len(values) != samples:
                raise ExpandedSurrogatePhase1Error(f"{name} does not align with labels")
        if samples < 1 or not np.all(np.isfinite(self.features)):
            raise ExpandedSurrogatePhase1Error("dataset features are invalid")
        for values in (
            self.labels,
            self.label_standard_errors,
            self.confidence_interval_low,
            self.confidence_interval_high,
            self.replication_means,
        ):
            if not np.all(np.isfinite(values)):
                raise ExpandedSurrogatePhase1Error(
                    "dataset labels and uncertainty must be finite"
                )
        if np.any(self.label_standard_errors < 0.0):
            raise ExpandedSurrogatePhase1Error(
                "label standard errors must be non-negative"
            )
        if np.any(self.confidence_interval_low > self.labels) or np.any(
            self.confidence_interval_high < self.labels
        ):
            raise ExpandedSurrogatePhase1Error(
                "confidence intervals must contain their labels"
            )
        if np.any(self.effective_steps < DEFAULT_REFERENCE_STEPS):
            raise ExpandedSurrogatePhase1Error(
                "effective steps cannot be below the production grid"
            )


def feature_order_for(product_key: str) -> tuple[str, ...]:
    if product_key == "phoenix_v3":
        return PHOENIX_FEATURE_ORDER
    if product_key == "barrier_reverse_convertible":
        return BRC_FEATURE_ORDER
    raise ExpandedSurrogatePhase1Error("unknown product")


def _even_schedule(maturity: float, count: int) -> tuple[float, ...]:
    values = [maturity * index / count for index in range(1, count + 1)]
    values[-1] = maturity
    return tuple(values)


def _piecewise_market(
    random: np.random.Generator,
    *,
    spot: float,
    maturity: float,
    symbol: str,
) -> EquityMarketTermStructure:
    rate_level = random.uniform(0.0, 0.07)
    dividend_level = random.uniform(0.0, 0.04)
    volatility_level = random.uniform(0.10, 0.50)
    rate_slope = random.uniform(-0.025, 0.025)
    dividend_slope = random.uniform(-0.015, 0.015)
    volatility_slope = random.uniform(-0.15, 0.15)
    segments = []
    for fraction in CURVE_TIME_FRACTIONS:
        centered = fraction - 0.625
        segments.append(
            EquityMarketSegment(
                end_time_years=maturity * fraction,
                risk_free_rate=float(
                    np.clip(
                        rate_level + rate_slope * centered + random.normal(0.0, 0.0015),
                        0.0,
                        0.07,
                    )
                ),
                dividend_yield=float(
                    np.clip(
                        dividend_level
                        + dividend_slope * centered
                        + random.normal(0.0, 0.0008),
                        0.0,
                        0.04,
                    )
                ),
                volatility=float(
                    np.clip(
                        volatility_level
                        + volatility_slope * centered
                        + random.normal(0.0, 0.0075),
                        0.10,
                        0.50,
                    )
                ),
            )
        )
    timestamp = datetime(2026, 7, 25, 12, 0, tzinfo=timezone.utc)
    return EquityMarketTermStructure(
        symbol=symbol,
        underlier_type="equity",
        currency="USD",
        valuation_time=timestamp,
        market_data_time=timestamp,
        spot=spot,
        segments=tuple(segments),
        calendar="WEEKDAYS",
        day_count="ACT/365F",
        source=LABEL_PROTOCOL_VERSION,
    )


def _sample_phoenix_case(
    random: np.random.Generator,
    *,
    case_index: int,
    barrier_focus_probability: float,
) -> TrainingCase:
    reference = 100.0
    maturity = random.uniform(0.5, 2.0)
    observations = int(random.integers(2, 9))
    first_autocall = random.uniform(0.98, 1.20)
    coupon_barrier = random.uniform(0.65, min(1.05, first_autocall))
    final_autocall = random.uniform(coupon_barrier, first_autocall)
    autocall_schedule = tuple(
        float(value)
        for value in np.linspace(first_autocall, final_autocall, observations)
    )
    knock_in = random.uniform(0.45, min(0.85, coupon_barrier))
    if random.random() < barrier_focus_probability:
        anchor = random.choice(
            np.asarray([1.0, first_autocall, final_autocall, coupon_barrier, knock_in])
        )
        spot_ratio = float(np.clip(anchor + random.normal(0.0, 0.035), 0.55, 1.35))
    else:
        spot_ratio = random.uniform(0.55, 1.35)
    coupon_rate = random.uniform(0.005, 0.04)
    memory_coupon = bool(random.integers(0, 2))
    unpaid_coupon_count = int(random.integers(0, 4)) if memory_coupon else 0
    prior_knock_in = bool(random.random() < 0.15)
    contract = PhoenixSingleV3Contract(
        reference_level=reference,
        maturity_years=maturity,
        observation_times_years=_even_schedule(maturity, observations),
        autocall_barrier_fracs=autocall_schedule,
        coupon_barrier_frac=coupon_barrier,
        coupon_rate=coupon_rate,
        knock_in_frac=knock_in,
        prior_knock_in_breached=prior_knock_in,
        memory_coupon=memory_coupon,
        unpaid_coupon_count=unpaid_coupon_count,
    )
    market = _piecewise_market(
        random,
        spot=reference * spot_ratio,
        maturity=maturity,
        symbol=f"PHASE1-PHX-{case_index:06d}",
    )
    equivalent = market.equivalent_flat_parameters(maturity)
    features = np.asarray(
        [
            spot_ratio,
            equivalent["risk_free_rate"],
            equivalent["dividend_yield"],
            equivalent["volatility"],
            maturity,
            first_autocall,
            final_autocall,
            coupon_barrier,
            coupon_rate,
            knock_in,
            observations,
            float(memory_coupon),
            unpaid_coupon_count,
            float(prior_knock_in),
            spot_ratio - first_autocall,
            spot_ratio - final_autocall,
            spot_ratio - coupon_barrier,
            spot_ratio - knock_in,
            first_autocall - final_autocall,
            coupon_rate * (1 + unpaid_coupon_count),
        ],
        dtype=np.float64,
    )
    return TrainingCase("phoenix_v3", market, contract, features)


def _sample_brc_case(
    random: np.random.Generator,
    *,
    case_index: int,
    barrier_focus_probability: float,
) -> TrainingCase:
    reference = 100.0
    maturity = random.uniform(0.25, 2.0)
    coupon_count = int(random.integers(1, 9))
    coupon_rate = random.uniform(0.005, 0.04)
    strike = random.uniform(0.90, 1.10)
    knock_in = random.uniform(0.45, min(0.90, strike))
    if random.random() < barrier_focus_probability:
        anchor = random.choice(np.asarray([1.0, strike, knock_in]))
        spot_ratio = float(np.clip(anchor + random.normal(0.0, 0.035), 0.55, 1.35))
    else:
        spot_ratio = random.uniform(0.55, 1.35)
    prior_knock_in = bool(random.random() < 0.15)
    contract = BarrierReverseConvertibleV1Contract(
        reference_level=reference,
        maturity_years=maturity,
        coupon_times_years=_even_schedule(maturity, coupon_count),
        coupon_rate_per_period=coupon_rate,
        strike_frac=strike,
        knock_in_frac=knock_in,
        prior_knock_in_breached=prior_knock_in,
    )
    market = _piecewise_market(
        random,
        spot=reference * spot_ratio,
        maturity=maturity,
        symbol=f"PHASE1-BRC-{case_index:06d}",
    )
    equivalent = market.equivalent_flat_parameters(maturity)
    features = np.asarray(
        [
            spot_ratio,
            equivalent["risk_free_rate"],
            equivalent["dividend_yield"],
            equivalent["volatility"],
            maturity,
            coupon_rate,
            strike,
            knock_in,
            coupon_count,
            float(prior_knock_in),
            spot_ratio - strike,
            spot_ratio - knock_in,
            strike - knock_in,
            coupon_rate * coupon_count,
        ],
        dtype=np.float64,
    )
    return TrainingCase(
        "barrier_reverse_convertible",
        market,
        contract,
        features,
    )


def _sample_case(
    product_key: str,
    random: np.random.Generator,
    *,
    case_index: int,
    barrier_focus_probability: float,
) -> TrainingCase:
    if product_key == "phoenix_v3":
        return _sample_phoenix_case(
            random,
            case_index=case_index,
            barrier_focus_probability=barrier_focus_probability,
        )
    if product_key == "barrier_reverse_convertible":
        return _sample_brc_case(
            random,
            case_index=case_index,
            barrier_focus_probability=barrier_focus_probability,
        )
    raise ExpandedSurrogatePhase1Error("unknown product")


def _event_times(case: TrainingCase) -> tuple[float, ...]:
    if isinstance(case.contract, PhoenixSingleV3Contract):
        return case.contract.observation_times_years
    return case.contract.coupon_times_years


def _discounted_payoffs(
    case: TrainingCase,
    *,
    time_grid: np.ndarray,
    standard_normal_shocks: np.ndarray,
) -> np.ndarray:
    contract = case.contract
    paths = simulate_piecewise_gbm_paths(
        market=case.market,
        T=contract.maturity_years,
        n_steps=len(time_grid) - 1,
        n_paths=len(standard_normal_shocks),
        seed=None,
        standard_normal_shocks=standard_normal_shocks,
        time_grid_years=time_grid,
    )
    equivalent = case.market.equivalent_flat_parameters(contract.maturity_years)
    params = contract.to_payoff_params(
        risk_free_rate=equivalent["risk_free_rate"],
        volatility=equivalent["volatility"],
    )
    if isinstance(contract, PhoenixSingleV3Contract):
        return PhoenixPayoff().compute_payoff_with_explicit_schedule_and_discount_curve(
            paths=paths,
            params=params,
            path_times_years=time_grid,
            observation_times_years=contract.observation_times_years,
            prior_knock_in_breached=contract.prior_knock_in_breached,
            discount_factor=case.market.discount_factor,
            autocall_barrier_fracs=contract.autocall_barrier_fracs,
            memory_coupon=contract.memory_coupon,
            unpaid_coupon_count=contract.unpaid_coupon_count,
        )
    ledger = BarrierReverseConvertiblePayoff().compute_event_ledger(
        paths=paths,
        params=params,
        path_times_years=time_grid,
        coupon_times_years=contract.coupon_times_years,
        prior_knock_in_breached=contract.prior_knock_in_breached,
        discount_factor=case.market.discount_factor,
    )
    return (
        ledger["coupon_pv"]
        + ledger["protected_principal_pv"]
        + ledger["downside_redemption_pv"]
    )


def _replicated_label(
    case: TrainingCase,
    *,
    paths_per_replication: int,
    label_replications: int,
    production_steps: int,
    confidence_level: float,
    label_seed: int,
    sample_index: int,
) -> ReplicatedLabel:
    time_grid = build_simulation_time_grid(
        case.contract.maturity_years,
        production_steps,
        _event_times(case),
    )
    effective_steps = len(time_grid) - 1
    replication_means = []
    for replication in range(label_replications):
        seed_sequence = np.random.SeedSequence([label_seed, sample_index, replication])
        seed = int(seed_sequence.generate_state(1, dtype=np.uint32)[0])
        shocks = standard_normal_shocks(
            method="sobol",
            n_paths=paths_per_replication,
            n_steps=effective_steps,
            seed=seed,
        )
        payoffs = _discounted_payoffs(
            case,
            time_grid=time_grid,
            standard_normal_shocks=shocks,
        )
        replication_means.append(float(np.mean(payoffs)))
    values = np.asarray(replication_means, dtype=np.float64)
    standard_error = float(np.std(values, ddof=1) / math.sqrt(len(values)))
    multiplier = float(
        student_t.ppf(0.5 + confidence_level / 2.0, df=label_replications - 1)
    )
    return ReplicatedLabel(
        price=float(np.mean(values)),
        standard_error=standard_error,
        confidence_half_width=multiplier * standard_error,
        effective_steps=effective_steps,
        replication_means=tuple(float(value) for value in values),
    )


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _dataset(
    product_key: str,
    *,
    role: str,
    samples: int,
    paths_per_replication: int,
    config: Phase1Config,
    dataset_seed: int,
    label_seed: int,
) -> Phase1Dataset:
    started = time.perf_counter()
    LOGGER.info(
        "Generating %s %s labels (%s samples, %sx%s paths)",
        product_key,
        role,
        samples,
        config.label_replications,
        paths_per_replication,
    )
    random = np.random.default_rng(dataset_seed)
    cases = [
        _sample_case(
            product_key,
            random,
            case_index=index,
            barrier_focus_probability=config.barrier_focus_probability,
        )
        for index in range(samples)
    ]
    labels = []
    progress_interval = max(1, min(100, samples // 10))
    for index, case in enumerate(cases):
        labels.append(
            _replicated_label(
                case,
                paths_per_replication=paths_per_replication,
                label_replications=config.label_replications,
                production_steps=config.production_steps,
                confidence_level=config.confidence_level,
                label_seed=label_seed,
                sample_index=index,
            )
        )
        completed = index + 1
        if completed % progress_interval == 0 or completed == samples:
            LOGGER.info(
                "Generated %s/%s %s %s labels in %.1fs",
                completed,
                samples,
                product_key,
                role,
                time.perf_counter() - started,
            )
    features = np.stack([case.features for case in cases])
    prices = np.asarray([label.price for label in labels], dtype=np.float64)
    standard_errors = np.asarray(
        [label.standard_error for label in labels],
        dtype=np.float64,
    )
    half_widths = np.asarray(
        [label.confidence_half_width for label in labels],
        dtype=np.float64,
    )
    effective_steps = np.asarray(
        [label.effective_steps for label in labels],
        dtype=np.int64,
    )
    replication_means = np.asarray(
        [label.replication_means for label in labels],
        dtype=np.float64,
    )
    market_payloads = tuple(
        json.dumps(
            case.market.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
        )
        for case in cases
    )
    contract_payloads = tuple(
        json.dumps(
            case.contract.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
        )
        for case in cases
    )
    metadata = {
        "schema_version": EXPERIMENT_VERSION,
        "label_protocol_version": LABEL_PROTOCOL_VERSION,
        "product_key": product_key,
        "role": role,
        "samples": samples,
        "paths_per_replication": paths_per_replication,
        "label_replications": config.label_replications,
        "total_paths_per_label": (paths_per_replication * config.label_replications),
        "production_steps": config.production_steps,
        "confidence_level": config.confidence_level,
        "sampling_method": "independently_scrambled_sobol",
        "replication_seed_strategy": (
            "numpy.SeedSequence(label_seed, sample_index, replication)"
        ),
        "curve_time_fractions": list(CURVE_TIME_FRACTIONS),
        "feature_order": list(feature_order_for(product_key)),
        "dataset_seed": dataset_seed,
        "label_seed": label_seed,
    }
    digest = hashlib.sha256()
    digest.update(_canonical_bytes(metadata))
    for array in (
        features,
        prices,
        standard_errors,
        half_widths,
        effective_steps,
        replication_means,
    ):
        digest.update(np.ascontiguousarray(array).tobytes())
    for payload in (*market_payloads, *contract_payloads):
        digest.update(payload.encode("utf-8"))
    dataset_id = f"sha256:{digest.hexdigest()}"
    return Phase1Dataset(
        product_key=product_key,
        role=role,
        features=features,
        labels=prices,
        label_standard_errors=standard_errors,
        confidence_interval_low=prices - half_widths,
        confidence_interval_high=prices + half_widths,
        effective_steps=effective_steps,
        market_payloads=market_payloads,
        contract_payloads=contract_payloads,
        replication_means=replication_means,
        dataset_id=dataset_id,
        metadata=metadata,
    )


def _save_dataset(dataset: Phase1Dataset, output_root: Path) -> dict[str, Any]:
    directory = output_root / dataset.product_key / "datasets"
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"{dataset.role}-{dataset.dataset_id.removeprefix('sha256:')}"
    data_path = directory / f"{stem}.npz"
    manifest_path = directory / f"{stem}.json"
    np.savez_compressed(
        data_path,
        features=dataset.features,
        labels=dataset.labels,
        label_standard_errors=dataset.label_standard_errors,
        confidence_interval_low=dataset.confidence_interval_low,
        confidence_interval_high=dataset.confidence_interval_high,
        effective_steps=dataset.effective_steps,
        market_payloads=np.asarray(dataset.market_payloads),
        contract_payloads=np.asarray(dataset.contract_payloads),
        replication_means=dataset.replication_means,
    )
    manifest = {
        **dict(dataset.metadata),
        "dataset_id": dataset.dataset_id,
        "data_file": data_path.name,
        "data_file_sha256": (
            f"sha256:{hashlib.sha256(data_path.read_bytes()).hexdigest()}"
        ),
        "label_standard_error": {
            "mean": float(np.mean(dataset.label_standard_errors)),
            "median": float(np.median(dataset.label_standard_errors)),
            "p95": float(np.quantile(dataset.label_standard_errors, 0.95)),
        },
        "confidence_half_width": {
            "mean": float(
                np.mean(
                    (dataset.confidence_interval_high - dataset.confidence_interval_low)
                    / 2.0
                )
            ),
            "p95": float(
                np.quantile(
                    (dataset.confidence_interval_high - dataset.confidence_interval_low)
                    / 2.0,
                    0.95,
                )
            ),
        },
        "effective_steps": {
            "minimum": int(np.min(dataset.effective_steps)),
            "maximum": int(np.max(dataset.effective_steps)),
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "dataset_id": dataset.dataset_id,
        "data_path": str(data_path),
        "manifest_path": str(manifest_path),
        "summary": manifest,
    }


def uncertainty_sample_weights(
    label_standard_errors: np.ndarray,
    *,
    maximum_weight_multiple: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    standard_errors = np.asarray(label_standard_errors, dtype=np.float64)
    if (
        standard_errors.ndim != 1
        or not len(standard_errors)
        or not np.all(np.isfinite(standard_errors))
        or np.any(standard_errors < 0.0)
    ):
        raise ExpandedSurrogatePhase1Error("label standard errors are invalid")
    positive = standard_errors[standard_errors > 0.0]
    reference_se = float(np.median(positive)) if positive.size else 1e-6
    variance_floor = max((0.25 * reference_se) ** 2, 1e-12)
    raw = 1.0 / (standard_errors**2 + variance_floor)
    raw_median = float(np.median(raw))
    lower = raw_median / maximum_weight_multiple
    upper = raw_median * maximum_weight_multiple
    clipped = np.clip(raw, lower, upper)
    weights = clipped / float(np.mean(clipped))
    return weights, {
        "version": WEIGHT_POLICY_VERSION,
        "formula": "1 / (label_standard_error^2 + variance_floor)",
        "variance_floor": variance_floor,
        "maximum_weight_multiple": maximum_weight_multiple,
        "raw_median": raw_median,
        "normalized_minimum": float(np.min(weights)),
        "normalized_median": float(np.median(weights)),
        "normalized_maximum": float(np.max(weights)),
        "normalized_mean": float(np.mean(weights)),
        "capped_low_count": int(np.sum(raw < lower)),
        "capped_high_count": int(np.sum(raw > upper)),
    }


def _balanced_l1_parameters(config: Phase1Config) -> dict[str, Any]:
    return {
        "objective": "regression_l1",
        "learning_rate": 0.025,
        "num_leaves": 31,
        "min_child_samples": 15,
        "reg_lambda": 0.01,
        "n_estimators": config.trees,
        "subsample": 0.9,
        "subsample_freq": 1,
        "colsample_bytree": 0.9,
        "n_jobs": -1,
        "random_state": config.seed,
        "verbosity": -1,
    }


def _metrics(dataset: Phase1Dataset, predictions: np.ndarray) -> dict[str, float]:
    predicted = np.asarray(predictions, dtype=np.float64)
    if predicted.shape != dataset.labels.shape or not np.all(np.isfinite(predicted)):
        raise ExpandedSurrogatePhase1Error("model predictions are invalid")
    residuals = predicted - dataset.labels
    absolute = np.abs(residuals)
    half_width = (
        dataset.confidence_interval_high - dataset.confidence_interval_low
    ) / 2.0
    return {
        "samples": int(len(dataset.labels)),
        "mae": float(mean_absolute_error(dataset.labels, predicted)),
        "p95_absolute_error": float(np.quantile(absolute, 0.95)),
        "rmse": float(np.sqrt(np.mean(residuals**2))),
        "bias": float(np.mean(residuals)),
        "r2": float(r2_score(dataset.labels, predicted)),
        "within_label_confidence_interval_fraction": float(
            np.mean(absolute <= half_width)
        ),
        "within_one_percent_or_label_interval_fraction": float(
            np.mean(absolute <= np.maximum(0.01, half_width))
        ),
        "mean_label_standard_error": float(np.mean(dataset.label_standard_errors)),
        "p95_label_confidence_half_width": float(np.quantile(half_width, 0.95)),
    }


def _latency_ms(model: LGBMRegressor, features: np.ndarray) -> float:
    observations = []
    for row in features[: min(100, len(features))]:
        started = time.perf_counter()
        model.booster_.predict(row.reshape(1, -1))
        observations.append((time.perf_counter() - started) * 1_000.0)
    return float(np.median(observations))


def _incumbent_model(product_key: str) -> tuple[Any | None, str | None]:
    latest = (
        REPO_ROOT
        / "final"
        / "research_candidates"
        / product_key
        / "latest_experiment.json"
    )
    if not latest.exists():
        return None, None
    payload = json.loads(latest.read_text(encoding="utf-8"))
    package = REPO_ROOT / str(payload["package_path"])
    model_path = package / "model.joblib"
    if not model_path.exists():
        return None, None
    return joblib.load(model_path), str(payload.get("experiment_id"))


def _predict(model: Any, features: np.ndarray) -> np.ndarray:
    booster = getattr(model, "booster_", None)
    if booster is not None:
        return np.asarray(booster.predict(features), dtype=np.float64)
    return np.asarray(model.predict(features), dtype=np.float64)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_joblib(path: Path, model: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    joblib.dump(model, temporary)
    temporary.replace(path)


def _product_candidate_id(
    product_key: str,
    *,
    datasets: Sequence[Phase1Dataset],
    learner_parameters: Mapping[str, Any],
) -> str:
    payload = {
        "experiment_version": EXPERIMENT_VERSION,
        "label_protocol_version": LABEL_PROTOCOL_VERSION,
        "weight_policy_version": WEIGHT_POLICY_VERSION,
        "product_key": product_key,
        "dataset_ids": {dataset.role: dataset.dataset_id for dataset in datasets},
        "learner_parameters": dict(learner_parameters),
    }
    return f"sha256:{hashlib.sha256(_canonical_bytes(payload)).hexdigest()}"


def _run_product(
    product_key: str,
    *,
    config: Phase1Config,
    output_root: Path,
    product_offset: int,
) -> dict[str, Any]:
    LOGGER.info("Starting Phase 1 experiment for %s", product_key)
    development = _dataset(
        product_key,
        role="development",
        samples=config.development_samples,
        paths_per_replication=config.paths_per_replication,
        config=config,
        dataset_seed=config.seed + product_offset + 101,
        label_seed=config.seed + product_offset + 10_001,
    )
    stored_datasets = {development.role: _save_dataset(development, output_root)}
    validation = _dataset(
        product_key,
        role="validation",
        samples=config.validation_samples,
        paths_per_replication=config.paths_per_replication,
        config=config,
        dataset_seed=config.seed + product_offset + 20_001,
        label_seed=config.seed + product_offset + 30_001,
    )
    stored_datasets[validation.role] = _save_dataset(validation, output_root)
    evaluation = _dataset(
        product_key,
        role="evaluation",
        samples=config.evaluation_samples,
        paths_per_replication=config.evaluation_paths_per_replication,
        config=config,
        dataset_seed=config.seed + product_offset + 50_001,
        label_seed=config.seed + product_offset + 60_001,
    )
    stored_datasets[evaluation.role] = _save_dataset(evaluation, output_root)
    weights, weight_policy = uncertainty_sample_weights(
        development.label_standard_errors,
        maximum_weight_multiple=config.maximum_weight_multiple,
    )
    parameters = _balanced_l1_parameters(config)
    LOGGER.info("Training fixed LightGBM ablation for %s", product_key)
    unweighted = LGBMRegressor(**parameters)
    unweighted.fit(development.features, development.labels)
    weighted = LGBMRegressor(**parameters)
    weighted.fit(
        development.features,
        development.labels,
        sample_weight=weights,
    )
    candidate_id = _product_candidate_id(
        product_key,
        datasets=(development, validation, evaluation),
        learner_parameters=parameters,
    )
    models_dir = (
        output_root / product_key / "models" / candidate_id.removeprefix("sha256:")[:16]
    )
    models_dir.mkdir(parents=True, exist_ok=True)
    unweighted_path = models_dir / "balanced_l1_unweighted.joblib"
    weighted_path = models_dir / "balanced_l1_uncertainty_weighted.joblib"
    _atomic_joblib(unweighted_path, unweighted)
    _atomic_joblib(weighted_path, weighted)

    comparisons: dict[str, Any] = {}
    incumbent, incumbent_id = _incumbent_model(product_key)
    if incumbent is not None:
        comparisons["incumbent_v2"] = {
            "experiment_id": incumbent_id,
            "validation": _metrics(
                validation,
                _predict(incumbent, validation.features),
            ),
            "evaluation": _metrics(
                evaluation,
                _predict(incumbent, evaluation.features),
            ),
        }
    for name, model in (
        ("phase1_unweighted", unweighted),
        ("phase1_uncertainty_weighted", weighted),
    ):
        comparisons[name] = {
            "validation": _metrics(
                validation,
                _predict(model, validation.features),
            ),
            "evaluation": _metrics(
                evaluation,
                _predict(model, evaluation.features),
            ),
            "median_single_row_latency_ms": _latency_ms(
                model,
                evaluation.features,
            ),
        }
    return {
        "product_key": product_key,
        "candidate_id": candidate_id,
        "contract_version": (
            PHOENIX_SINGLE_V3_CONTRACT_VERSION
            if product_key == "phoenix_v3"
            else BARRIER_REVERSE_CONVERTIBLE_V1
        ),
        "feature_order": list(feature_order_for(product_key)),
        "datasets": stored_datasets,
        "learner": {
            "class": "lightgbm.LGBMRegressor",
            "library_version": lightgbm.__version__,
            "parameters": parameters,
            "primary_candidate": "phase1_uncertainty_weighted",
            "architecture_changed_from_v2": False,
            "unweighted_model_path": str(unweighted_path),
            "weighted_model_path": str(weighted_path),
        },
        "weight_policy": weight_policy,
        "comparisons": comparisons,
        "runtime_approved": False,
        "runtime_artifact_created": False,
        "status": "research_only",
    }


def run_phase1_experiment(
    *,
    config: Phase1Config | None = None,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    active = config or Phase1Config()
    started = time.perf_counter()
    products = [
        _run_product(
            product_key,
            config=active,
            output_root=Path(output_root),
            product_offset=index * 1_000_003,
        )
        for index, product_key in enumerate(PRODUCT_KEYS)
    ]
    report = {
        "experiment_version": EXPERIMENT_VERSION,
        "label_protocol_version": LABEL_PROTOCOL_VERSION,
        "weight_policy_version": WEIGHT_POLICY_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": time.perf_counter() - started,
        "configuration": {
            **active.__dict__,
            "curve_time_fractions": list(CURVE_TIME_FRACTIONS),
            "sampling_method": "independently_scrambled_sobol",
        },
        "products": products,
        "runtime_policy_changed": False,
        "runtime_artifact_created": False,
        "status": "research_only",
    }
    identity_payload = {
        "experiment_version": report["experiment_version"],
        "label_protocol_version": report["label_protocol_version"],
        "weight_policy_version": report["weight_policy_version"],
        "configuration": report["configuration"],
        "products": [
            {
                "product_key": product["product_key"],
                "candidate_id": product["candidate_id"],
                "contract_version": product["contract_version"],
                "dataset_ids": {
                    role: stored["dataset_id"]
                    for role, stored in product["datasets"].items()
                },
                "learner": {
                    key: value
                    for key, value in product["learner"].items()
                    if key
                    not in {
                        "unweighted_model_path",
                        "weighted_model_path",
                    }
                },
                "weight_policy": product["weight_policy"],
                "comparisons": {
                    name: {
                        key: value
                        for key, value in comparison.items()
                        if key != "median_single_row_latency_ms"
                    }
                    for name, comparison in product["comparisons"].items()
                },
            }
            for product in products
        ],
        "runtime_policy_changed": report["runtime_policy_changed"],
        "runtime_artifact_created": report["runtime_artifact_created"],
        "status": report["status"],
    }
    identity = hashlib.sha256(_canonical_bytes(identity_payload)).hexdigest()
    report["experiment_id"] = f"sha256:{identity}"
    reports_root = Path(output_root) / "reports"
    _atomic_json(reports_root / f"{identity}.json", report)
    _atomic_json(Path(output_root) / "phase1_report.json", report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-samples", type=int, default=2_500)
    parser.add_argument("--validation-samples", type=int, default=500)
    parser.add_argument("--evaluation-samples", type=int, default=300)
    parser.add_argument("--paths-per-replication", type=int, default=128)
    parser.add_argument(
        "--evaluation-paths-per-replication",
        type=int,
        default=512,
    )
    parser.add_argument("--label-replications", type=int, default=8)
    parser.add_argument("--trees", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=20_260_725)
    parser.add_argument(
        "--maximum-weight-multiple",
        type=float,
        default=10.0,
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    arguments = parser.parse_args(argv)
    result = run_phase1_experiment(
        config=Phase1Config(
            development_samples=arguments.development_samples,
            validation_samples=arguments.validation_samples,
            evaluation_samples=arguments.evaluation_samples,
            paths_per_replication=arguments.paths_per_replication,
            evaluation_paths_per_replication=(
                arguments.evaluation_paths_per_replication
            ),
            label_replications=arguments.label_replications,
            maximum_weight_multiple=arguments.maximum_weight_multiple,
            seed=arguments.seed,
            trees=arguments.trees,
        ),
        output_root=arguments.output_root,
    )
    print(
        json.dumps(
            {
                "experiment_id": result["experiment_id"],
                "status": result["status"],
                "duration_seconds": result["duration_seconds"],
                "report_path": str(arguments.output_root / "phase1_report.json"),
                "products": {
                    item["product_key"]: item["comparisons"]
                    for item in result["products"]
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
