"""Reproducible, non-promoting surrogate experiments for expanded products.

This module deliberately writes outside the runtime artifact directory. A
candidate is packaged only when every sealed-audit gate passes, and even then
the manifest remains ``runtime_approved: false`` until a separate governance
step integrates and monitors it.
"""

import argparse
import hashlib
import json
import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import joblib
import lightgbm
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, r2_score

from .barrier_reverse_convertible import (
    BarrierReverseConvertiblePayoff,
    BarrierReverseConvertibleV1Contract,
)
from .market import EquityMarketSegment, EquityMarketTermStructure
from .payoffs import PhoenixPayoff
from .phoenix_contract import PhoenixSingleV3Contract
from .reference_pricer import (
    price_barrier_reverse_convertible_reference,
    price_phoenix_v3_piecewise_reference,
)


EXPERIMENT_VERSION = "expanded-surrogate-experiment-v2"
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "final" / "research_candidates"


@dataclass(frozen=True)
class ExperimentConfig:
    development_samples: int = 700
    validation_samples: int = 180
    audit_samples: int = 220
    development_paths: int = 768
    audit_paths: int = 2_048
    monitoring_steps: int = 64
    seed: int = 20_260_719
    trees: int = 500
    barrier_focus_probability: float = 0.60
    maximum_mae: float = 0.015
    maximum_p95_absolute_error: float = 0.04
    minimum_r2: float = 0.90
    maximum_mean_label_standard_error: float = 0.01
    maximum_median_latency_ms: float = 5.0


@dataclass(frozen=True)
class ProductExperiment:
    key: str
    contract_version: str
    feature_order: tuple[str, ...]
    sampler: Callable[
        [np.random.Generator, int, int, float],
        tuple[np.ndarray, float, float],
    ]
    domain: dict[str, Any]


def _market(
    *,
    spot: float,
    maturity: float,
    rate: float,
    dividend: float,
    volatility: float,
) -> EquityMarketTermStructure:
    timestamp = datetime(2026, 7, 19, 12, 0, tzinfo=timezone.utc)
    return EquityMarketTermStructure(
        symbol="SYNTH",
        underlier_type="equity",
        currency="USD",
        valuation_time=timestamp,
        market_data_time=timestamp,
        spot=spot,
        segments=(
            EquityMarketSegment(
                end_time_years=maturity,
                risk_free_rate=rate,
                dividend_yield=dividend,
                volatility=volatility,
            ),
        ),
        calendar="WEEKDAYS",
        day_count="ACT/365F",
        source=EXPERIMENT_VERSION,
    )


def _even_schedule(maturity: float, count: int) -> tuple[float, ...]:
    values = [maturity * index / count for index in range(1, count + 1)]
    values[-1] = maturity
    return tuple(values)


def _sample_phoenix_v3(
    random: np.random.Generator,
    paths: int,
    monitoring_steps: int,
    barrier_focus_probability: float,
) -> tuple[np.ndarray, float, float]:
    reference = 100.0
    maturity = random.uniform(0.5, 2.0)
    rate = random.uniform(0.0, 0.07)
    dividend = random.uniform(0.0, 0.04)
    volatility = random.uniform(0.10, 0.50)
    observations = int(random.integers(2, 9))
    first_autocall = random.uniform(0.98, 1.20)
    coupon_barrier = random.uniform(0.65, min(1.05, first_autocall))
    final_autocall = random.uniform(coupon_barrier, first_autocall)
    autocall_schedule = tuple(np.linspace(first_autocall, final_autocall, observations))
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
    result = price_phoenix_v3_piecewise_reference(
        payoff=PhoenixPayoff(),
        contract=contract,
        market=_market(
            spot=reference * spot_ratio,
            maturity=maturity,
            rate=rate,
            dividend=dividend,
            volatility=volatility,
        ),
        n_paths=paths,
        n_steps=monitoring_steps,
        seed=int(random.integers(0, 2**32 - 1)),
    )
    features = np.asarray(
        [
            spot_ratio,
            rate,
            dividend,
            volatility,
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
    return features, float(result["price"]), float(result["standard_error"])


def _sample_barrier_reverse_convertible(
    random: np.random.Generator,
    paths: int,
    monitoring_steps: int,
    barrier_focus_probability: float,
) -> tuple[np.ndarray, float, float]:
    reference = 100.0
    maturity = random.uniform(0.25, 2.0)
    rate = random.uniform(0.0, 0.07)
    dividend = random.uniform(0.0, 0.04)
    volatility = random.uniform(0.10, 0.50)
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
    result = price_barrier_reverse_convertible_reference(
        payoff=BarrierReverseConvertiblePayoff(),
        contract=contract,
        market=_market(
            spot=reference * spot_ratio,
            maturity=maturity,
            rate=rate,
            dividend=dividend,
            volatility=volatility,
        ),
        n_paths=paths,
        n_steps=monitoring_steps,
        seed=int(random.integers(0, 2**32 - 1)),
    )
    features = np.asarray(
        [
            spot_ratio,
            rate,
            dividend,
            volatility,
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
    return features, float(result["price"]), float(result["standard_error"])


PRODUCTS = (
    ProductExperiment(
        key="phoenix_v3",
        contract_version="phoenix-single-v3",
        feature_order=(
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
        ),
        sampler=_sample_phoenix_v3,
        domain={
            "schedule": "even observations and linear autocall step-down",
            "underlier": "single USD equity-like synthetic GBM",
            "seasoning": "spot/reference, prior knock-in, and unpaid coupon state",
        },
    ),
    ProductExperiment(
        key="barrier_reverse_convertible",
        contract_version="barrier-reverse-convertible-v1",
        feature_order=(
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
        ),
        sampler=_sample_barrier_reverse_convertible,
        domain={
            "schedule": "even fixed-coupon observations",
            "underlier": "single USD equity-like synthetic GBM",
            "seasoning": "spot/reference and prior knock-in state",
        },
    ),
)


def _dataset(
    product: ProductExperiment,
    *,
    samples: int,
    paths: int,
    monitoring_steps: int,
    seed: int,
    barrier_focus_probability: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    random = np.random.default_rng(seed)
    rows = [
        product.sampler(
            random,
            paths,
            monitoring_steps,
            barrier_focus_probability,
        )
        for _ in range(samples)
    ]
    features = np.stack([row[0] for row in rows])
    labels = np.asarray([row[1] for row in rows], dtype=np.float64)
    standard_errors = np.asarray([row[2] for row in rows], dtype=np.float64)
    digest = hashlib.sha256()
    digest.update(features.tobytes())
    digest.update(labels.tobytes())
    digest.update(standard_errors.tobytes())
    return features, labels, standard_errors, f"sha256:{digest.hexdigest()}"


def _latency_ms(model: LGBMRegressor, features: np.ndarray) -> float:
    observations = []
    for row in features[: min(100, len(features))]:
        started = time.perf_counter()
        model.booster_.predict(row.reshape(1, -1))
        observations.append((time.perf_counter() - started) * 1_000.0)
    return float(np.median(observations))


def _candidate_specs(
    config: ExperimentConfig,
) -> tuple[tuple[str, dict[str, Any]], ...]:
    shared = {
        "n_estimators": config.trees,
        "subsample": 0.9,
        "subsample_freq": 1,
        "colsample_bytree": 0.9,
        "n_jobs": -1,
        "random_state": config.seed,
        "verbosity": -1,
    }
    return (
        (
            "balanced_l1",
            {
                **shared,
                "objective": "regression_l1",
                "learning_rate": 0.025,
                "num_leaves": 31,
                "min_child_samples": 15,
                "reg_lambda": 0.01,
            },
        ),
        (
            "boundary_l1",
            {
                **shared,
                "objective": "regression_l1",
                "learning_rate": 0.02,
                "num_leaves": 63,
                "min_child_samples": 10,
                "reg_alpha": 0.0025,
                "reg_lambda": 0.05,
            },
        ),
        (
            "smooth_l2",
            {
                **shared,
                "objective": "regression",
                "learning_rate": 0.02,
                "num_leaves": 63,
                "min_child_samples": 12,
                "reg_lambda": 0.05,
            },
        ),
    )


def _selection_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    config: ExperimentConfig,
) -> dict[str, float]:
    errors = np.abs(predictions - labels)
    mae = float(mean_absolute_error(labels, predictions))
    p95 = float(np.quantile(errors, 0.95))
    mae_scale = max(config.maximum_mae, 1e-12)
    p95_scale = max(config.maximum_p95_absolute_error, 1e-12)
    return {
        "mae": mae,
        "p95_absolute_error": p95,
        "r2": float(r2_score(labels, predictions)),
        "selection_score": mae / mae_scale + p95 / p95_scale,
    }


def _run_product(
    product: ProductExperiment,
    config: ExperimentConfig,
    output_root: Path,
) -> dict[str, Any]:
    development = _dataset(
        product,
        samples=config.development_samples,
        paths=config.development_paths,
        monitoring_steps=config.monitoring_steps,
        seed=config.seed + 101,
        barrier_focus_probability=config.barrier_focus_probability,
    )
    validation = _dataset(
        product,
        samples=config.validation_samples,
        paths=config.development_paths,
        monitoring_steps=config.monitoring_steps,
        seed=config.seed + 20_001,
        barrier_focus_probability=config.barrier_focus_probability,
    )
    audit = _dataset(
        product,
        samples=config.audit_samples,
        paths=config.audit_paths,
        monitoring_steps=config.monitoring_steps,
        seed=config.seed + 50_001,
        barrier_focus_probability=config.barrier_focus_probability,
    )
    development_x, development_y, development_se, development_id = development
    validation_x, validation_y, validation_se, validation_id = validation
    audit_x, audit_y, audit_se, audit_id = audit
    candidates: dict[str, tuple[LGBMRegressor, dict[str, float], dict[str, Any]]] = {}
    for name, parameters in _candidate_specs(config):
        candidate = LGBMRegressor(**parameters)
        candidate.fit(development_x, development_y)
        candidate_predictions = candidate.booster_.predict(validation_x)
        candidates[name] = (
            candidate,
            _selection_metrics(validation_y, candidate_predictions, config),
            parameters,
        )
    selected_name = min(
        candidates,
        key=lambda name: candidates[name][1]["selection_score"],
    )
    model, selected_validation, selected_parameters = candidates[selected_name]
    predictions = model.booster_.predict(audit_x)
    errors = np.abs(predictions - audit_y)
    metrics = {
        "mae": float(mean_absolute_error(audit_y, predictions)),
        "p95_absolute_error": float(np.quantile(errors, 0.95)),
        "r2": float(r2_score(audit_y, predictions)),
        "mean_label_standard_error": float(np.mean(audit_se)),
        "p95_label_standard_error": float(np.quantile(audit_se, 0.95)),
        "median_inference_latency_ms": _latency_ms(model, audit_x),
    }
    checks = {
        "mae": {
            "value": metrics["mae"],
            "maximum": config.maximum_mae,
            "passed": metrics["mae"] <= config.maximum_mae,
        },
        "p95_absolute_error": {
            "value": metrics["p95_absolute_error"],
            "maximum": config.maximum_p95_absolute_error,
            "passed": (
                metrics["p95_absolute_error"] <= config.maximum_p95_absolute_error
            ),
        },
        "r2": {
            "value": metrics["r2"],
            "minimum": config.minimum_r2,
            "passed": metrics["r2"] >= config.minimum_r2,
        },
        "mean_label_standard_error": {
            "value": metrics["mean_label_standard_error"],
            "maximum": config.maximum_mean_label_standard_error,
            "passed": (
                metrics["mean_label_standard_error"]
                <= config.maximum_mean_label_standard_error
            ),
        },
        "median_inference_latency_ms": {
            "value": metrics["median_inference_latency_ms"],
            "maximum": config.maximum_median_latency_ms,
            "passed": (
                metrics["median_inference_latency_ms"]
                <= config.maximum_median_latency_ms
            ),
        },
    }
    passed = all(check["passed"] for check in checks.values())
    experiment_payload = {
        "experiment_version": EXPERIMENT_VERSION,
        "product_key": product.key,
        "contract_version": product.contract_version,
        "development_dataset_id": development_id,
        "validation_dataset_id": validation_id,
        "audit_dataset_id": audit_id,
        "configuration": {
            **config.__dict__,
            "learner": {
                "class": "lightgbm.LGBMRegressor",
                "library_version": lightgbm.__version__,
                "selected_candidate": selected_name,
                "parameters": selected_parameters,
            },
            "feature_order": list(product.feature_order),
            "domain": product.domain,
        },
        "datasets": {
            "development_samples": config.development_samples,
            "development_paths_per_label": config.development_paths,
            "validation_samples": config.validation_samples,
            "validation_paths_per_label": config.development_paths,
            "audit_samples": config.audit_samples,
            "audit_paths_per_label": config.audit_paths,
            "development_mean_label_standard_error": float(np.mean(development_se)),
            "validation_mean_label_standard_error": float(np.mean(validation_se)),
        },
        "development_selection": {
            "selected_candidate": selected_name,
            "selected_metrics": selected_validation,
            "candidates": {
                name: {
                    "metrics": values[1],
                    "parameters": values[2],
                }
                for name, values in candidates.items()
            },
        },
        "sealed_audit": {
            "passed": passed,
            "metrics": metrics,
            "checks": checks,
        },
        "runtime_approved": False,
        "status": "research_candidate" if passed else "rejected",
    }
    identity_payload = json.dumps(
        experiment_payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    experiment_id = f"sha256:{hashlib.sha256(identity_payload).hexdigest()}"
    experiment_payload["experiment_id"] = experiment_id
    product_root = output_root / product.key
    product_root.mkdir(parents=True, exist_ok=True)
    latest_path = product_root / "latest_experiment.json"
    latest_path.write_text(
        json.dumps(experiment_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    if passed:
        package_dir = product_root / experiment_id.removeprefix("sha256:")
        package_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, package_dir / "model.joblib")
        (package_dir / "manifest.json").write_text(
            json.dumps(experiment_payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        experiment_payload["package_path"] = package_dir.relative_to(
            REPO_ROOT
        ).as_posix()
        latest_path.write_text(
            json.dumps(experiment_payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    else:
        experiment_payload["package_path"] = None
    return experiment_payload


def run_expanded_surrogate_experiments(
    config: ExperimentConfig | None = None,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    settings = config or ExperimentConfig()
    started = time.perf_counter()
    results = [_run_product(product, settings, output_root) for product in PRODUCTS]
    summary = {
        "experiment_version": EXPERIMENT_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "runtime_policy_changed": False,
        "products": results,
        "elapsed_seconds": time.perf_counter() - started,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "experiment_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-samples", type=int, default=700)
    parser.add_argument("--validation-samples", type=int, default=180)
    parser.add_argument("--audit-samples", type=int, default=220)
    parser.add_argument("--development-paths", type=int, default=768)
    parser.add_argument("--audit-paths", type=int, default=2_048)
    parser.add_argument("--monitoring-steps", type=int, default=64)
    parser.add_argument("--trees", type=int, default=500)
    parser.add_argument("--barrier-focus-probability", type=float, default=0.60)
    parser.add_argument("--seed", type=int, default=20_260_719)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    config = ExperimentConfig(
        development_samples=arguments.development_samples,
        validation_samples=arguments.validation_samples,
        audit_samples=arguments.audit_samples,
        development_paths=arguments.development_paths,
        audit_paths=arguments.audit_paths,
        monitoring_steps=arguments.monitoring_steps,
        trees=arguments.trees,
        seed=arguments.seed,
        barrier_focus_probability=arguments.barrier_focus_probability,
    )
    if (
        min(
            config.development_samples,
            config.validation_samples,
            config.audit_samples,
            config.development_paths,
            config.audit_paths,
            config.monitoring_steps,
            config.trees,
        )
        < 1
    ):
        raise SystemExit("all experiment sizes must be positive")
    if not 0.0 <= config.barrier_focus_probability <= 1.0:
        raise SystemExit("barrier focus probability must be between zero and one")
    summary = run_expanded_surrogate_experiments(
        config=config,
        output_root=arguments.output_root,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
