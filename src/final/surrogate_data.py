import hashlib
import json
import math
import os
import platform
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np

from .market import EquityMarketSegment, EquityMarketTermStructure
from .payoffs import PhoenixPayoff
from .reference_pricer import phoenix_piecewise_discounted_components
from .surrogate_contract import (
    CURVE_TIME_FRACTIONS,
    DEFAULT_TRAINING_DOMAIN,
    PHOENIX_SURROGATE_FEATURE_NAMES,
    PHOENIX_PAYOFF_AWARE_TARGET_NAMES,
    PHOENIX_PRICE_COMPONENT_NAMES,
    PHOENIX_SURROGATE_FEATURE_VERSION,
    extract_phoenix_surrogate_features,
    surrogate_contract_metadata,
)


DATASET_SCHEMA_VERSION = "phoenix-surrogate-dataset-v3"
SPLIT_NAMES = ("train", "validation", "test")
MARKET_REGIMES = ("low_vol", "normal", "high_vol", "crisis")
MONEYNESS_REGIONS = ("broad", "knock_in", "coupon", "autocall")


class SurrogateDatasetError(ValueError):
    pass


@dataclass(frozen=True)
class PhoenixDatasetConfig:
    n_contracts: int = 1_024
    markets_per_contract: int = 4
    paths_per_replication: int = 1_024
    label_replications: int = 2
    n_steps: int = 252
    dataset_seed: int = 42
    label_seed: int = 7_301
    sampling_method: str = "sobol"
    validation_fraction: float = 0.15
    test_fraction: float = 0.15
    dataset_role: str = "development"
    barrier_focused_sampling: bool = True
    sampling_profile: str = "balanced"

    def __post_init__(self) -> None:
        if self.n_contracts < 6:
            raise SurrogateDatasetError("n_contracts must be at least 6")
        if self.markets_per_contract < 1 or self.markets_per_contract > 32:
            raise SurrogateDatasetError("markets_per_contract must be between 1 and 32")
        if self.paths_per_replication < 8:
            raise SurrogateDatasetError("paths_per_replication must be at least 8")
        if self.label_replications < 1 or self.label_replications > 32:
            raise SurrogateDatasetError("label_replications must be between 1 and 32")
        minimum_steps = int(DEFAULT_TRAINING_DOMAIN["obs_count"][1])
        if self.n_steps < minimum_steps or self.n_steps > 2_520:
            raise SurrogateDatasetError(
                f"n_steps must be between {minimum_steps} and 2520"
            )
        if self.sampling_method not in {"sobol", "antithetic"}:
            raise SurrogateDatasetError(
                "sampling_method must be one of: sobol, antithetic"
            )
        if self.sampling_method == "sobol" and (
            self.paths_per_replication & (self.paths_per_replication - 1)
        ):
            raise SurrogateDatasetError(
                "Sobol paths_per_replication must be a power of two"
            )
        if self.sampling_method == "antithetic" and self.paths_per_replication % 2:
            raise SurrogateDatasetError("antithetic paths_per_replication must be even")
        if not 0.0 < self.validation_fraction < 0.5:
            raise SurrogateDatasetError("validation_fraction must be in (0, 0.5)")
        if not 0.0 < self.test_fraction < 0.5:
            raise SurrogateDatasetError("test_fraction must be in (0, 0.5)")
        if self.validation_fraction + self.test_fraction >= 0.8:
            raise SurrogateDatasetError(
                "validation_fraction + test_fraction must be < 0.8"
            )
        if self.dataset_role not in {"development", "audit"}:
            raise SurrogateDatasetError(
                "dataset_role must be one of: development, audit"
            )
        if self.sampling_profile not in {"balanced", "low_vol_barrier_focus"}:
            raise SurrogateDatasetError(
                "sampling_profile must be one of: balanced, low_vol_barrier_focus"
            )
        if self.dataset_role == "audit" and self.sampling_profile != "balanced":
            raise SurrogateDatasetError("audit datasets must use balanced sampling")


@dataclass(frozen=True)
class PhoenixSurrogateDataset:
    X: np.ndarray
    y: np.ndarray
    label_standard_error: np.ndarray
    payoff_standard_deviation: np.ndarray
    auxiliary_targets: np.ndarray
    auxiliary_standard_error: np.ndarray
    group_ids: np.ndarray
    split_names: np.ndarray
    regime_names: np.ndarray
    moneyness_region_names: np.ndarray
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        n_samples = len(self.y)
        if self.X.shape != (n_samples, len(PHOENIX_SURROGATE_FEATURE_NAMES)):
            raise SurrogateDatasetError("dataset feature matrix has invalid shape")
        expected_target_shape = (
            n_samples,
            len(PHOENIX_PAYOFF_AWARE_TARGET_NAMES),
        )
        if self.auxiliary_targets.shape != expected_target_shape or (
            self.auxiliary_standard_error.shape != expected_target_shape
        ):
            raise SurrogateDatasetError("payoff-aware targets have invalid shape")
        for name, values in (
            ("label_standard_error", self.label_standard_error),
            ("payoff_standard_deviation", self.payoff_standard_deviation),
            ("group_ids", self.group_ids),
            ("split_names", self.split_names),
            ("regime_names", self.regime_names),
            ("moneyness_region_names", self.moneyness_region_names),
        ):
            if len(values) != n_samples:
                raise SurrogateDatasetError(f"{name} has invalid length")
        if n_samples < 1 or not np.all(np.isfinite(self.X)):
            raise SurrogateDatasetError("dataset features must be non-empty and finite")
        if not np.all(np.isfinite(self.y)) or not np.all(
            np.isfinite(self.label_standard_error)
        ):
            raise SurrogateDatasetError("dataset labels must be finite")
        if not np.all(np.isfinite(self.auxiliary_targets)) or not np.all(
            np.isfinite(self.auxiliary_standard_error)
        ):
            raise SurrogateDatasetError("payoff-aware targets must be finite")
        if np.any(self.auxiliary_standard_error < 0.0):
            raise SurrogateDatasetError(
                "payoff-aware target standard errors must be non-negative"
            )
        event_start = len(PHOENIX_PRICE_COMPONENT_NAMES)
        if np.any(self.auxiliary_targets[:, :event_start] < 0.0) or np.any(
            (self.auxiliary_targets[:, event_start:] < 0.0)
            | (self.auxiliary_targets[:, event_start:] > 1.0)
        ):
            raise SurrogateDatasetError("payoff-aware targets violate their bounds")
        reconstructed = np.sum(self.auxiliary_targets[:, :event_start], axis=1)
        if not np.allclose(reconstructed, self.y, rtol=0.0, atol=1e-12):
            raise SurrogateDatasetError(
                "cashflow component labels do not reconcile to price"
            )
        if np.any(self.y < 0.0) or np.any(self.y > 5.0):
            raise SurrogateDatasetError("dataset labels are outside sanity bounds")
        if np.any(self.label_standard_error < 0.0):
            raise SurrogateDatasetError("label standard errors must be non-negative")
        if not np.all(np.isfinite(self.payoff_standard_deviation)) or np.any(
            self.payoff_standard_deviation < 0.0
        ):
            raise SurrogateDatasetError(
                "payoff standard deviations must be finite and non-negative"
            )
        observed_splits = set(str(value) for value in self.split_names)
        expected_splits = (
            set(SPLIT_NAMES)
            if self.metadata.get("dataset_role") == "development"
            else {"audit"}
        )
        if observed_splits != expected_splits:
            raise SurrogateDatasetError(
                "dataset splits do not match the declared dataset role"
            )


def _group_split_map(config: PhoenixDatasetConfig) -> dict[int, str]:
    rng = np.random.RandomState(config.dataset_seed + 991)
    shuffled = list(int(value) for value in rng.permutation(config.n_contracts))
    n_test = max(1, int(round(config.n_contracts * config.test_fraction)))
    n_validation = max(1, int(round(config.n_contracts * config.validation_fraction)))
    if n_test + n_validation >= config.n_contracts:
        raise SurrogateDatasetError("not enough contract groups for all splits")
    test_groups = set(shuffled[:n_test])
    validation_groups = set(shuffled[n_test : n_test + n_validation])
    return {
        group_id: (
            "test"
            if group_id in test_groups
            else "validation" if group_id in validation_groups else "train"
        )
        for group_id in range(config.n_contracts)
    }


def _sample_terms(rng: np.random.RandomState) -> dict[str, Any]:
    domain = DEFAULT_TRAINING_DOMAIN
    knock_in = float(rng.uniform(*domain["knock_in_frac"]))
    autocall_lower = max(domain["autocall_barrier_frac"][0], knock_in + 0.05)
    autocall = float(rng.uniform(autocall_lower, domain["autocall_barrier_frac"][1]))
    coupon_lower = max(domain["coupon_barrier_frac"][0], knock_in)
    coupon_upper = min(domain["coupon_barrier_frac"][1], autocall)
    return {
        "maturity_years": float(rng.uniform(*domain["maturity_years"])),
        "autocall_barrier_frac": autocall,
        "coupon_barrier_frac": float(rng.uniform(coupon_lower, coupon_upper)),
        "coupon_rate": float(rng.uniform(*domain["coupon_rate"])),
        "knock_in_frac": knock_in,
        "obs_count": int(
            rng.randint(
                int(domain["obs_count"][0]),
                int(domain["obs_count"][1]) + 1,
            )
        ),
    }


def _regime_ranges(regime: str) -> tuple[tuple[float, float], tuple[float, float]]:
    if regime == "low_vol":
        return (-0.01, 0.04), (0.08, 0.20)
    if regime == "normal":
        return (0.015, 0.07), (0.16, 0.38)
    if regime == "high_vol":
        return (0.03, 0.10), (0.32, 0.65)
    return (0.05, 0.12), (0.50, 0.90)


def _case_sampling_labels(
    *,
    config: PhoenixDatasetConfig,
    rng: np.random.RandomState,
    group_id: int,
    market_index: int,
) -> tuple[str, str]:
    if config.sampling_profile == "low_vol_barrier_focus" and market_index >= 4:
        regime = "low_vol"
        region = ("knock_in", "coupon", "autocall")[(group_id + market_index) % 3]
        return regime, region
    regime = MARKET_REGIMES[(group_id + market_index) % len(MARKET_REGIMES)]
    region = (
        MONEYNESS_REGIONS[int(rng.randint(len(MONEYNESS_REGIONS)))]
        if config.barrier_focused_sampling
        else "broad"
    )
    return regime, region


def _sample_market(
    *,
    rng: np.random.RandomState,
    terms: dict[str, Any],
    regime: str,
    group_id: int,
    market_index: int,
    moneyness_region: str,
) -> tuple[EquityMarketTermStructure, float]:
    domain = DEFAULT_TRAINING_DOMAIN
    rate_range, volatility_range = _regime_ranges(regime)
    maturity = terms["maturity_years"]
    rate_level = float(rng.uniform(*rate_range))
    rate_slope = float(rng.uniform(-0.025, 0.025))
    dividend_level = float(rng.uniform(0.0, 0.055))
    dividend_slope = float(rng.uniform(-0.015, 0.015))
    volatility_level = float(rng.uniform(*volatility_range))
    volatility_slope = float(rng.uniform(-0.12, 0.12))
    segments = []
    for fraction in CURVE_TIME_FRACTIONS:
        centered = fraction - 0.625
        rate = np.clip(
            rate_level + rate_slope * centered + rng.normal(0.0, 0.002),
            *domain["risk_free_rate"],
        )
        dividend = np.clip(
            dividend_level + dividend_slope * centered + rng.normal(0.0, 0.001),
            *domain["dividend_yield"],
        )
        volatility = np.clip(
            volatility_level + volatility_slope * centered + rng.normal(0.0, 0.01),
            *domain["volatility"],
        )
        segments.append(
            EquityMarketSegment(
                end_time_years=maturity * fraction,
                risk_free_rate=float(rate),
                dividend_yield=float(dividend),
                volatility=float(volatility),
            )
        )
    reference_spot = 100.0
    if moneyness_region == "broad":
        spot_ratio = float(rng.uniform(*domain["spot_ratio"]))
    else:
        barrier_name = {
            "knock_in": "knock_in_frac",
            "coupon": "coupon_barrier_frac",
            "autocall": "autocall_barrier_frac",
        }[moneyness_region]
        spot_ratio = float(
            np.clip(
                terms[barrier_name] * math.exp(rng.normal(0.0, 0.08)),
                *domain["spot_ratio"],
            )
        )
    valuation_time = datetime(2026, 1, 2, 16, 0, tzinfo=timezone.utc)
    market = EquityMarketTermStructure(
        symbol=f"TRAIN-{group_id:05d}-{market_index:02d}",
        underlier_type="equity",
        currency="USD",
        valuation_time=valuation_time,
        market_data_time=valuation_time,
        spot=reference_spot * spot_ratio,
        segments=tuple(segments),
        calendar="WEEKDAYS",
        day_count="ACT/365F",
        source=PHOENIX_SURROGATE_FEATURE_VERSION,
    )
    return market, reference_spot


def _standard_normal_shocks(
    *,
    method: str,
    n_paths: int,
    n_steps: int,
    seed: int,
) -> np.ndarray:
    if method == "antithetic":
        rng = np.random.RandomState(seed)
        half = rng.standard_normal((n_paths // 2, n_steps))
        return np.vstack([half, -half])

    from scipy.stats import norm, qmc

    sampler = qmc.Sobol(d=n_steps, scramble=True, seed=seed)
    uniforms = sampler.random_base2(m=int(math.log2(n_paths)))
    epsilon = np.finfo(np.float64).eps
    return norm.ppf(np.clip(uniforms, epsilon, 1.0 - epsilon))


def _label_case(
    *,
    market: EquityMarketTermStructure,
    terms: dict[str, Any],
    reference_spot: float,
    config: PhoenixDatasetConfig,
    sample_index: int,
) -> tuple[float, float, float, np.ndarray, np.ndarray]:
    payoff = PhoenixPayoff()
    params = {
        "S0": reference_spot,
        "r": 0.0,
        "sigma": 1.0,
        "T": terms["maturity_years"],
        "autocall_barrier_frac": terms["autocall_barrier_frac"],
        "coupon_barrier_frac": terms["coupon_barrier_frac"],
        "coupon_rate": terms["coupon_rate"],
        "knock_in_frac": terms["knock_in_frac"],
        "obs_count": terms["obs_count"],
    }
    replication_means = []
    replication_targets = []
    all_payoffs = []
    all_target_paths = []
    for replication in range(config.label_replications):
        seed = (config.label_seed + sample_index * 104_729 + replication * 7_919) % (
            2**32 - 1
        )
        shocks = _standard_normal_shocks(
            method=config.sampling_method,
            n_paths=config.paths_per_replication,
            n_steps=config.n_steps,
            seed=seed,
        )
        components = phoenix_piecewise_discounted_components(
            payoff=payoff,
            params=params,
            market=market,
            n_paths=config.paths_per_replication,
            n_steps=config.n_steps,
            seed=None,
            standard_normal_shocks=shocks,
        )
        target_paths = np.column_stack(
            [components[name] for name in PHOENIX_PAYOFF_AWARE_TARGET_NAMES]
        )
        discounted = np.sum(
            target_paths[:, : len(PHOENIX_PRICE_COMPONENT_NAMES)], axis=1
        )
        replication_means.append(float(np.mean(discounted)))
        replication_targets.append(np.mean(target_paths, axis=0))
        all_payoffs.append(discounted)
        all_target_paths.append(target_paths)
    joined = np.concatenate(all_payoffs)
    price = float(np.mean(replication_means))
    payoff_std = float(np.std(joined, ddof=1)) if joined.size > 1 else 0.0
    if len(replication_means) > 1:
        label_se = float(np.std(replication_means, ddof=1)) / math.sqrt(
            len(replication_means)
        )
    else:
        label_se = payoff_std / math.sqrt(joined.size)
    target = np.mean(np.asarray(replication_targets), axis=0)
    if len(replication_targets) > 1:
        target_se = np.std(np.asarray(replication_targets), axis=0, ddof=1) / math.sqrt(
            len(replication_targets)
        )
    else:
        joined_targets = np.concatenate(all_target_paths, axis=0)
        target_se = np.std(joined_targets, axis=0, ddof=1) / math.sqrt(
            len(joined_targets)
        )
    return price, label_se, payoff_std, target, target_se


def _dataset_identity(
    *,
    X: np.ndarray,
    y: np.ndarray,
    label_standard_error: np.ndarray,
    payoff_standard_deviation: np.ndarray,
    auxiliary_targets: np.ndarray,
    auxiliary_standard_error: np.ndarray,
    group_ids: np.ndarray,
    split_names: np.ndarray,
    regime_names: np.ndarray,
    moneyness_region_names: np.ndarray,
    config: PhoenixDatasetConfig,
    generation_environment: dict[str, str],
) -> str:
    digest = hashlib.sha256()
    digest.update(DATASET_SCHEMA_VERSION.encode("utf-8"))
    digest.update(json.dumps(asdict(config), sort_keys=True).encode("utf-8"))
    digest.update(json.dumps(generation_environment, sort_keys=True).encode("utf-8"))
    digest.update(
        json.dumps(surrogate_contract_metadata(), sort_keys=True).encode("utf-8")
    )
    for values in (
        X,
        y,
        label_standard_error,
        payoff_standard_deviation,
        auxiliary_targets,
        auxiliary_standard_error,
        group_ids,
    ):
        digest.update(np.ascontiguousarray(values).tobytes())
    digest.update("|".join(str(value) for value in split_names).encode("utf-8"))
    digest.update("|".join(str(value) for value in regime_names).encode("utf-8"))
    digest.update(
        "|".join(str(value) for value in moneyness_region_names).encode("utf-8")
    )
    return f"sha256:{digest.hexdigest()}"


def _generation_environment() -> dict[str, str]:
    environment = {"python": platform.python_version()}
    for package in ("numpy", "scipy"):
        try:
            environment[package] = version(package)
        except PackageNotFoundError:
            environment[package] = "unknown"
    return environment


def generate_phoenix_surrogate_dataset(
    config: PhoenixDatasetConfig,
    *,
    verbose: bool = True,
) -> PhoenixSurrogateDataset:
    started = time.perf_counter()
    rng = np.random.RandomState(config.dataset_seed)
    group_splits = (
        _group_split_map(config)
        if config.dataset_role == "development"
        else {group_id: "audit" for group_id in range(config.n_contracts)}
    )
    n_samples = config.n_contracts * config.markets_per_contract
    X = np.zeros((n_samples, len(PHOENIX_SURROGATE_FEATURE_NAMES)), dtype=np.float64)
    y = np.zeros(n_samples, dtype=np.float64)
    label_se = np.zeros(n_samples, dtype=np.float64)
    payoff_std = np.zeros(n_samples, dtype=np.float64)
    auxiliary_targets = np.zeros(
        (n_samples, len(PHOENIX_PAYOFF_AWARE_TARGET_NAMES)), dtype=np.float64
    )
    auxiliary_se = np.zeros_like(auxiliary_targets)
    group_ids = np.zeros(n_samples, dtype=np.int64)
    split_names = np.empty(n_samples, dtype="<U10")
    regime_names = np.empty(n_samples, dtype="<U12")
    moneyness_region_names = np.empty(n_samples, dtype="<U12")

    sample_index = 0
    for group_id in range(config.n_contracts):
        terms = _sample_terms(rng)
        for market_index in range(config.markets_per_contract):
            regime, moneyness_region = _case_sampling_labels(
                config=config,
                rng=rng,
                group_id=group_id,
                market_index=market_index,
            )
            market, reference_spot = _sample_market(
                rng=rng,
                terms=terms,
                regime=regime,
                group_id=group_id,
                market_index=market_index,
                moneyness_region=moneyness_region,
            )
            X[sample_index] = extract_phoenix_surrogate_features(
                market=market,
                terms=terms,
                contract_reference_spot=reference_spot,
            )
            (
                y[sample_index],
                label_se[sample_index],
                payoff_std[sample_index],
                auxiliary_targets[sample_index],
                auxiliary_se[sample_index],
            ) = _label_case(
                market=market,
                terms=terms,
                reference_spot=reference_spot,
                config=config,
                sample_index=sample_index,
            )
            group_ids[sample_index] = group_id
            split_names[sample_index] = group_splits[group_id]
            regime_names[sample_index] = regime
            moneyness_region_names[sample_index] = moneyness_region
            sample_index += 1
            if verbose and (
                sample_index == n_samples or sample_index % max(1, n_samples // 20) == 0
            ):
                print(
                    f"[PhoenixSurrogateData] {sample_index}/{n_samples} labels complete",
                    flush=True,
                )

    generation_environment = _generation_environment()
    dataset_id = _dataset_identity(
        X=X,
        y=y,
        label_standard_error=label_se,
        payoff_standard_deviation=payoff_std,
        auxiliary_targets=auxiliary_targets,
        auxiliary_standard_error=auxiliary_se,
        group_ids=group_ids,
        split_names=split_names,
        regime_names=regime_names,
        moneyness_region_names=moneyness_region_names,
        config=config,
        generation_environment=generation_environment,
    )
    active_splits = SPLIT_NAMES if config.dataset_role == "development" else ("audit",)
    split_counts = {split: int(np.sum(split_names == split)) for split in active_splits}
    split_group_counts = {
        split: len(set(group_ids[split_names == split].tolist()))
        for split in active_splits
    }
    metadata = {
        "dataset_schema_version": DATASET_SCHEMA_VERSION,
        "dataset_id": dataset_id,
        "dataset_role": config.dataset_role,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": asdict(config),
        "n_samples": n_samples,
        "split_counts": split_counts,
        "split_group_counts": split_group_counts,
        "group_split_rule": "contract-group-disjoint-v1",
        "generation_environment": generation_environment,
        "label_uncertainty": (
            "standard error across independent scrambled replications"
            if config.label_replications > 1
            else "pathwise standard error"
        ),
        "generation_seconds": time.perf_counter() - started,
        **surrogate_contract_metadata(),
    }
    return PhoenixSurrogateDataset(
        X=X,
        y=y,
        label_standard_error=label_se,
        payoff_standard_deviation=payoff_std,
        auxiliary_targets=auxiliary_targets,
        auxiliary_standard_error=auxiliary_se,
        group_ids=group_ids,
        split_names=split_names,
        regime_names=regime_names,
        moneyness_region_names=moneyness_region_names,
        metadata=metadata,
    )


def save_phoenix_surrogate_dataset(
    dataset: PhoenixSurrogateDataset, output_path: Path
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            X=dataset.X,
            y=dataset.y,
            label_standard_error=dataset.label_standard_error,
            payoff_standard_deviation=dataset.payoff_standard_deviation,
            auxiliary_targets=dataset.auxiliary_targets,
            auxiliary_standard_error=dataset.auxiliary_standard_error,
            group_ids=dataset.group_ids,
            split_names=dataset.split_names,
            regime_names=dataset.regime_names,
            moneyness_region_names=dataset.moneyness_region_names,
            metadata_json=np.asarray(
                json.dumps(dataset.metadata, sort_keys=True), dtype=np.str_
            ),
        )
    os.replace(temporary, path)


def load_phoenix_surrogate_dataset(input_path: Path) -> PhoenixSurrogateDataset:
    path = Path(input_path)
    try:
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata_json"].item()))
            dataset = PhoenixSurrogateDataset(
                X=np.asarray(data["X"], dtype=np.float64),
                y=np.asarray(data["y"], dtype=np.float64),
                label_standard_error=np.asarray(
                    data["label_standard_error"], dtype=np.float64
                ),
                payoff_standard_deviation=np.asarray(
                    data["payoff_standard_deviation"], dtype=np.float64
                ),
                auxiliary_targets=np.asarray(
                    data["auxiliary_targets"], dtype=np.float64
                ),
                auxiliary_standard_error=np.asarray(
                    data["auxiliary_standard_error"], dtype=np.float64
                ),
                group_ids=np.asarray(data["group_ids"], dtype=np.int64),
                split_names=np.asarray(data["split_names"], dtype="<U10"),
                regime_names=np.asarray(data["regime_names"], dtype="<U12"),
                moneyness_region_names=np.asarray(
                    data["moneyness_region_names"], dtype="<U12"
                ),
                metadata=metadata,
            )
    except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
        raise SurrogateDatasetError("surrogate dataset load failed") from exc
    if metadata.get("dataset_schema_version") != DATASET_SCHEMA_VERSION:
        raise SurrogateDatasetError("surrogate dataset schema version mismatch")
    if metadata.get("feature_names") != list(PHOENIX_SURROGATE_FEATURE_NAMES):
        raise SurrogateDatasetError("surrogate dataset feature schema mismatch")
    expected_contract = surrogate_contract_metadata()
    for name, expected_value in expected_contract.items():
        if metadata.get(name) != expected_value:
            raise SurrogateDatasetError(f"surrogate dataset {name} contract mismatch")
    try:
        config_payload = metadata["config"]
        if not isinstance(config_payload, dict):
            raise TypeError
        config = PhoenixDatasetConfig(**config_payload)
    except (KeyError, TypeError) as exc:
        raise SurrogateDatasetError("surrogate dataset config is invalid") from exc
    generation_environment = metadata.get("generation_environment")
    if not isinstance(generation_environment, dict) or not all(
        isinstance(name, str) and isinstance(value, str)
        for name, value in generation_environment.items()
    ):
        raise SurrogateDatasetError(
            "surrogate dataset generation environment is invalid"
        )
    expected_id = _dataset_identity(
        X=dataset.X,
        y=dataset.y,
        label_standard_error=dataset.label_standard_error,
        payoff_standard_deviation=dataset.payoff_standard_deviation,
        auxiliary_targets=dataset.auxiliary_targets,
        auxiliary_standard_error=dataset.auxiliary_standard_error,
        group_ids=dataset.group_ids,
        split_names=dataset.split_names,
        regime_names=dataset.regime_names,
        moneyness_region_names=dataset.moneyness_region_names,
        config=config,
        generation_environment=generation_environment,
    )
    if metadata.get("dataset_id") != expected_id:
        raise SurrogateDatasetError("surrogate dataset fingerprint mismatch")
    return dataset
