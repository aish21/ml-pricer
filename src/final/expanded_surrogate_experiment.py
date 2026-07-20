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


EXPERIMENT_VERSION = "expanded-surrogate-experiment-v1"
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "final" / "research_candidates"


@dataclass(frozen=True)
class ExperimentConfig:
    development_samples: int = 700
    audit_samples: int = 220
    development_paths: int = 768
    audit_paths: int = 2_048
    monitoring_steps: int = 64
    seed: int = 20_260_719
    trees: int = 500
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
    sampler: Callable[[np.random.Generator, int, int], tuple[np.ndarray, float, float]]
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
) -> tuple[np.ndarray, float, float]:
    reference = 100.0
    spot_ratio = random.uniform(0.55, 1.35)
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
        ],
        dtype=np.float64,
    )
    return features, float(result["price"]), float(result["standard_error"])


def _sample_barrier_reverse_convertible(
    random: np.random.Generator,
    paths: int,
    monitoring_steps: int,
) -> tuple[np.ndarray, float, float]:
    reference = 100.0
    spot_ratio = random.uniform(0.55, 1.35)
    maturity = random.uniform(0.25, 2.0)
    rate = random.uniform(0.0, 0.07)
    dividend = random.uniform(0.0, 0.04)
    volatility = random.uniform(0.10, 0.50)
    coupon_count = int(random.integers(1, 9))
    coupon_rate = random.uniform(0.005, 0.04)
    strike = random.uniform(0.90, 1.10)
    knock_in = random.uniform(0.45, min(0.90, strike))
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    random = np.random.default_rng(seed)
    rows = [product.sampler(random, paths, monitoring_steps) for _ in range(samples)]
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
    )
    audit = _dataset(
        product,
        samples=config.audit_samples,
        paths=config.audit_paths,
        monitoring_steps=config.monitoring_steps,
        seed=config.seed + 10_001,
    )
    development_x, development_y, development_se, development_id = development
    audit_x, audit_y, audit_se, audit_id = audit
    model = LGBMRegressor(
        objective="regression_l1",
        n_estimators=config.trees,
        learning_rate=0.025,
        num_leaves=31,
        min_child_samples=15,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=0.01,
        n_jobs=-1,
        random_state=config.seed,
        verbosity=-1,
    )
    model.fit(development_x, development_y)
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
        "audit_dataset_id": audit_id,
        "configuration": {
            **config.__dict__,
            "learner": {
                "class": "lightgbm.LGBMRegressor",
                "library_version": lightgbm.__version__,
                "objective": "regression_l1",
            },
            "feature_order": list(product.feature_order),
            "domain": product.domain,
        },
        "datasets": {
            "development_samples": config.development_samples,
            "development_paths_per_label": config.development_paths,
            "audit_samples": config.audit_samples,
            "audit_paths_per_label": config.audit_paths,
            "development_mean_label_standard_error": float(np.mean(development_se)),
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
        experiment_payload["package_path"] = str(package_dir.relative_to(REPO_ROOT))
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
    parser.add_argument("--audit-samples", type=int, default=220)
    parser.add_argument("--development-paths", type=int, default=768)
    parser.add_argument("--audit-paths", type=int, default=2_048)
    parser.add_argument("--monitoring-steps", type=int, default=64)
    parser.add_argument("--trees", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20_260_719)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    config = ExperimentConfig(
        development_samples=arguments.development_samples,
        audit_samples=arguments.audit_samples,
        development_paths=arguments.development_paths,
        audit_paths=arguments.audit_paths,
        monitoring_steps=arguments.monitoring_steps,
        trees=arguments.trees,
        seed=arguments.seed,
    )
    if (
        min(
            config.development_samples,
            config.audit_samples,
            config.development_paths,
            config.audit_paths,
            config.monitoring_steps,
            config.trees,
        )
        < 1
    ):
        raise SystemExit("all experiment sizes must be positive")
    summary = run_expanded_surrogate_experiments(
        config=config,
        output_root=arguments.output_root,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
