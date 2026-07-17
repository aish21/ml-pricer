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
from .reference_pricer import (
    DEFAULT_REFERENCE_STEPS,
    phoenix_piecewise_discounted_payoffs,
)
from .surrogate_contract import (
    PHOENIX_SURROGATE_ARTIFACT_VERSION,
    PHOENIX_EVENT_TARGET_NAMES,
    PHOENIX_SURROGATE_FEATURE_NAMES,
    PHOENIX_SURROGATE_MODEL_VERSION,
    PHOENIX_PAYOFF_AWARE_MODEL_OUTPUT_NAMES,
    PHOENIX_PAYOFF_AWARE_TARGET_NAMES,
    PHOENIX_PRICE_COMPONENT_NAMES,
    extract_phoenix_surrogate_features,
    reconstruct_phoenix_surrogate_case,
    surrogate_contract_metadata,
)
from .surrogate_data import DATASET_SCHEMA_VERSION, PhoenixSurrogateDataset
from .surrogate_model import NumpyMLPSurrogate, file_sha256


class SurrogateTrainingError(RuntimeError):
    pass


AUDIT_UNCERTAINTY_POLICY_VERSION = "phoenix-audit-uncertainty-v1"


def _json_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


@dataclass(frozen=True)
class PhoenixSurrogateTrainingConfig:
    hidden_layer_sizes: tuple[int, ...] = (128, 128)
    candidate_hidden_layer_sizes: tuple[tuple[int, ...], ...] = ()
    candidate_seed_offsets: tuple[int, ...] = (0,)
    alpha: float = 0.0001
    learning_rate_init: float = 0.001
    max_iter: int = 1_000
    random_state: int = 42
    train_lightgbm_baseline: bool = True
    greek_validation_cases: int = 16
    greek_validation_paths: int = 4_096
    acceptance_audit_mae: float = 0.02
    acceptance_audit_p95_absolute_error: float = 0.05
    acceptance_audit_r2: float = 0.90
    acceptance_maximum_regime_mae: float = 0.03
    acceptance_maximum_moneyness_region_mae: float = 0.03
    acceptance_maximum_regime_moneyness_mae: float = 0.04
    audit_confidence_level: float = 0.95
    minimum_audit_label_replications: int = 8
    minimum_audit_total_paths: int = 2_048
    acceptance_economic_price_tolerance: float = 0.01
    acceptance_minimum_uncertainty_or_economic_coverage: float = 0.80
    acceptance_maximum_label_confidence_half_width_p95: float = 0.015
    acceptance_minimum_greek_sign_agreement: float = 0.50
    acceptance_maximum_component_mae: float = 0.08
    acceptance_maximum_event_mae: float = 0.08
    acceptance_maximum_mean_output_boundary_violation: float = 0.005
    acceptance_maximum_cashflow_reconstruction_mae: float = 0.08
    selection_worst_regime_weight: float = 0.35
    selection_worst_moneyness_weight: float = 0.25
    selection_worst_joint_cell_weight: float = 0.25
    selection_validation_folds: int = 5
    selection_validation_repeats: int = 3
    selection_worst_fold_weight: float = 0.25
    focused_head_regime_weight: float = 2.0
    focused_head_coupon_weight: float = 2.0
    focused_head_ridge: float = 0.001
    train_focused_head_candidates: bool = True
    require_audit_dataset: bool = True

    def __post_init__(self) -> None:
        if not self.hidden_layer_sizes or any(
            width < 1 or width > 2_048 for width in self.hidden_layer_sizes
        ):
            raise SurrogateTrainingError("hidden_layer_sizes are invalid")
        if len(self.candidate_hidden_layer_sizes) > 8:
            raise SurrogateTrainingError("too many candidate hidden-layer layouts")
        for layout in self.candidate_hidden_layer_sizes:
            if not layout or any(width < 1 or width > 2_048 for width in layout):
                raise SurrogateTrainingError("candidate_hidden_layer_sizes are invalid")
        if (
            not self.candidate_seed_offsets
            or len(self.candidate_seed_offsets) > 4
            or any(
                offset < 0 or offset > 1_000_000
                for offset in self.candidate_seed_offsets
            )
            or len(set(self.candidate_seed_offsets)) != len(self.candidate_seed_offsets)
        ):
            raise SurrogateTrainingError("candidate_seed_offsets are invalid")
        if self.alpha < 0.0 or self.learning_rate_init <= 0.0:
            raise SurrogateTrainingError("MLP optimization settings are invalid")
        if self.max_iter < 10:
            raise SurrogateTrainingError("max_iter must be at least 10")
        if self.greek_validation_cases < 0 or self.greek_validation_cases > 100:
            raise SurrogateTrainingError(
                "greek_validation_cases must be between 0 and 100"
            )
        if self.greek_validation_paths < 8 or self.greek_validation_paths % 2:
            raise SurrogateTrainingError(
                "greek_validation_paths must be an even integer of at least 8"
            )
        if any(
            not math.isfinite(value) or value <= 0.0
            for value in (
                self.acceptance_audit_mae,
                self.acceptance_audit_p95_absolute_error,
                self.acceptance_maximum_regime_mae,
                self.acceptance_maximum_moneyness_region_mae,
                self.acceptance_maximum_regime_moneyness_mae,
                self.acceptance_economic_price_tolerance,
                self.acceptance_maximum_label_confidence_half_width_p95,
                self.acceptance_maximum_component_mae,
                self.acceptance_maximum_event_mae,
                self.acceptance_maximum_cashflow_reconstruction_mae,
            )
        ):
            raise SurrogateTrainingError("acceptance error thresholds are invalid")
        if any(
            not math.isfinite(value) or value < 0.0
            for value in (
                self.selection_worst_regime_weight,
                self.selection_worst_moneyness_weight,
                self.selection_worst_joint_cell_weight,
                self.selection_worst_fold_weight,
            )
        ):
            raise SurrogateTrainingError("candidate selection weights are invalid")
        if (
            self.selection_validation_folds < 2
            or self.selection_validation_folds > 10
            or self.selection_validation_repeats < 1
            or self.selection_validation_repeats > 10
        ):
            raise SurrogateTrainingError("validation fold settings are invalid")
        if (
            not math.isfinite(self.focused_head_regime_weight)
            or self.focused_head_regime_weight < 1.0
            or not math.isfinite(self.focused_head_coupon_weight)
            or self.focused_head_coupon_weight < 1.0
            or not math.isfinite(self.focused_head_ridge)
            or self.focused_head_ridge <= 0.0
        ):
            raise SurrogateTrainingError("focused output-head settings are invalid")
        if (
            not math.isfinite(self.acceptance_audit_r2)
            or self.acceptance_audit_r2 > 1.0
        ):
            raise SurrogateTrainingError("acceptance R-squared threshold is invalid")
        if not 0.80 <= self.audit_confidence_level < 1.0:
            raise SurrogateTrainingError("audit confidence level is invalid")
        if (
            self.minimum_audit_label_replications < 2
            or self.minimum_audit_label_replications > 32
            or self.minimum_audit_total_paths < 8
        ):
            raise SurrogateTrainingError("audit sampling requirements are invalid")
        if any(
            not 0.0 <= value <= 1.0
            for value in (
                self.acceptance_minimum_uncertainty_or_economic_coverage,
                self.acceptance_minimum_greek_sign_agreement,
                self.acceptance_maximum_mean_output_boundary_violation,
            )
        ):
            raise SurrogateTrainingError("acceptance fraction thresholds are invalid")


def _audit_uncertainty_policy(
    dataset: PhoenixSurrogateDataset,
    config: PhoenixSurrogateTrainingConfig,
) -> dict[str, Any]:
    dataset_config = dataset.metadata.get("config")
    uncertainty = dataset.metadata.get("label_uncertainty_protocol")
    if not isinstance(dataset_config, dict) or not isinstance(uncertainty, dict):
        raise SurrogateTrainingError("audit dataset uncertainty metadata is missing")
    try:
        replications = int(dataset_config["label_replications"])
        paths_per_replication = int(dataset_config["paths_per_replication"])
        sampling_method = str(dataset_config["sampling_method"])
    except (KeyError, TypeError, ValueError) as exc:
        raise SurrogateTrainingError(
            "audit dataset uncertainty configuration is invalid"
        ) from exc
    if (
        not isinstance(dataset_config.get("label_replications"), int)
        or isinstance(dataset_config.get("label_replications"), bool)
        or not isinstance(dataset_config.get("paths_per_replication"), int)
        or isinstance(dataset_config.get("paths_per_replication"), bool)
    ):
        raise SurrogateTrainingError(
            "audit dataset uncertainty configuration is invalid"
        )
    total_paths = replications * paths_per_replication
    expected_metadata = {
        "estimator": "between-replication-standard-error",
        "independent_randomizations": replications,
        "paths_per_randomization": paths_per_replication,
        "total_paths_per_label": total_paths,
        "sampling_method": sampling_method,
    }
    if uncertainty != expected_metadata:
        raise SurrogateTrainingError(
            "audit dataset uncertainty metadata is inconsistent"
        )
    if sampling_method != "sobol":
        raise SurrogateTrainingError(
            "audit uncertainty policy requires scrambled Sobol sampling"
        )
    if replications < config.minimum_audit_label_replications:
        raise SurrogateTrainingError(
            "audit uncertainty policy requires more independent replications"
        )
    if total_paths < config.minimum_audit_total_paths:
        raise SurrogateTrainingError(
            "audit uncertainty policy requires more paths per label"
        )

    from scipy.stats import t

    confidence_multiplier = float(
        t.ppf(
            0.5 + config.audit_confidence_level / 2.0,
            df=replications - 1,
        )
    )
    policy = {
        "policy_version": AUDIT_UNCERTAINTY_POLICY_VERSION,
        "sampling_method": sampling_method,
        "minimum_independent_randomizations": (config.minimum_audit_label_replications),
        "observed_independent_randomizations": replications,
        "minimum_total_paths_per_label": config.minimum_audit_total_paths,
        "observed_total_paths_per_label": total_paths,
        "confidence_interval": "student-t-between-randomizations",
        "confidence_level": config.audit_confidence_level,
        "confidence_multiplier": confidence_multiplier,
        "economic_price_tolerance": config.acceptance_economic_price_tolerance,
        "coverage_rule": (
            "absolute_error <= max(label_confidence_half_width, "
            "economic_price_tolerance)"
        ),
        "minimum_coverage_fraction": (
            config.acceptance_minimum_uncertainty_or_economic_coverage
        ),
        "maximum_label_confidence_half_width_p95": (
            config.acceptance_maximum_label_confidence_half_width_p95
        ),
    }
    return {**policy, "policy_id": _json_sha256(policy)}


def _regression_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    label_standard_error: np.ndarray,
) -> dict[str, Any]:
    errors = np.asarray(predicted) - np.asarray(actual)
    absolute = np.abs(errors)
    denominator = np.maximum(np.asarray(label_standard_error), 1e-8)
    total_variance = float(np.sum((actual - np.mean(actual)) ** 2))
    residual_variance = float(np.sum(errors**2))
    return {
        "n_samples": int(len(actual)),
        "mae": float(np.mean(absolute)),
        "rmse": float(math.sqrt(np.mean(errors**2))),
        "p95_absolute_error": float(np.quantile(absolute, 0.95)),
        "max_absolute_error": float(np.max(absolute)),
        "mean_error": float(np.mean(errors)),
        "r2": (
            1.0 - residual_variance / total_variance if total_variance > 0.0 else 0.0
        ),
        "within_one_label_se_fraction": float(np.mean(absolute <= denominator)),
        "within_two_label_se_fraction": float(np.mean(absolute <= 2.0 * denominator)),
        "median_error_to_label_se": float(np.median(absolute / denominator)),
    }


def _uncertainty_aware_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    label_standard_error: np.ndarray,
    policy: dict[str, Any],
) -> dict[str, Any]:
    absolute = np.abs(np.asarray(predicted) - np.asarray(actual))
    standard_error = np.asarray(label_standard_error)
    confidence_half_width = standard_error * policy["confidence_multiplier"]
    economic_tolerance = policy["economic_price_tolerance"]
    combined_tolerance = np.maximum(confidence_half_width, economic_tolerance)
    return {
        "label_confidence_level": policy["confidence_level"],
        "label_confidence_multiplier": policy["confidence_multiplier"],
        "label_standard_error_median": float(np.median(standard_error)),
        "label_standard_error_p95": float(np.quantile(standard_error, 0.95)),
        "label_standard_error_maximum": float(np.max(standard_error)),
        "label_confidence_half_width_median": float(np.median(confidence_half_width)),
        "label_confidence_half_width_p95": float(
            np.quantile(confidence_half_width, 0.95)
        ),
        "label_confidence_half_width_maximum": float(np.max(confidence_half_width)),
        "economic_price_tolerance": economic_tolerance,
        "economic_tolerance_binding_fraction": float(
            np.mean(economic_tolerance >= confidence_half_width)
        ),
        "within_label_confidence_interval_fraction": float(
            np.mean(absolute <= confidence_half_width)
        ),
        "within_uncertainty_or_economic_tolerance_fraction": float(
            np.mean(absolute <= combined_tolerance)
        ),
    }


def _split_mask(dataset: PhoenixSurrogateDataset, split: str) -> np.ndarray:
    mask = dataset.split_names == split
    if not np.any(mask):
        raise SurrogateTrainingError(f"dataset split is empty: {split}")
    return mask


def _regime_metrics(
    dataset: PhoenixSurrogateDataset, predictions: np.ndarray
) -> dict[str, dict[str, dict[str, Any]]]:
    output = {}
    for split in ("train", "validation", "test"):
        output[split] = {}
        for regime in sorted(set(str(value) for value in dataset.regime_names)):
            mask = (dataset.split_names == split) & (dataset.regime_names == regime)
            if np.any(mask):
                output[split][regime] = _regression_metrics(
                    dataset.y[mask],
                    predictions[mask],
                    dataset.label_standard_error[mask],
                )
    return output


def _moneyness_metrics(
    dataset: PhoenixSurrogateDataset, predictions: np.ndarray
) -> dict[str, dict[str, dict[str, Any]]]:
    output = {}
    for split in ("train", "validation", "test"):
        output[split] = {}
        for region in sorted(
            set(str(value) for value in dataset.moneyness_region_names)
        ):
            mask = (dataset.split_names == split) & (
                dataset.moneyness_region_names == region
            )
            if np.any(mask):
                output[split][region] = _regression_metrics(
                    dataset.y[mask],
                    predictions[mask],
                    dataset.label_standard_error[mask],
                )
    return output


def _joint_regime_moneyness_metrics(
    dataset: PhoenixSurrogateDataset,
    predictions: np.ndarray,
    *,
    split: str | None = None,
) -> dict[str, dict[str, Any]]:
    output = {}
    split_mask = (
        np.ones(len(dataset.y), dtype=bool)
        if split is None
        else dataset.split_names == split
    )
    regimes = sorted(set(str(value) for value in dataset.regime_names[split_mask]))
    regions = sorted(
        set(str(value) for value in dataset.moneyness_region_names[split_mask])
    )
    for regime in regimes:
        for region in regions:
            mask = (
                split_mask
                & (dataset.regime_names == regime)
                & (dataset.moneyness_region_names == region)
            )
            if np.any(mask):
                output[f"{regime}:{region}"] = _regression_metrics(
                    dataset.y[mask],
                    predictions[mask],
                    dataset.label_standard_error[mask],
                )
    return output


def _development_error_analysis(
    dataset: PhoenixSurrogateDataset,
    predictions: np.ndarray,
) -> dict[str, Any]:
    mask = _split_mask(dataset, "validation")
    indices = np.flatnonzero(mask)
    absolute_errors = np.abs(predictions - dataset.y)
    denominator = np.maximum(dataset.label_standard_error, 1e-8)
    ordered = indices[np.argsort(-absolute_errors[indices])]
    feature_positions = {
        name: PHOENIX_SURROGATE_FEATURE_NAMES.index(name)
        for name in (
            "spot_ratio",
            "maturity_years",
            "coupon_rate_per_year",
            "total_variance_t100",
        )
    }
    worst_cases = []
    for row_index in ordered[:20]:
        worst_cases.append(
            {
                "row_index": int(row_index),
                "group_id": int(dataset.group_ids[row_index]),
                "regime": str(dataset.regime_names[row_index]),
                "moneyness_region": str(dataset.moneyness_region_names[row_index]),
                "actual_price": float(dataset.y[row_index]),
                "predicted_price": float(predictions[row_index]),
                "absolute_error": float(absolute_errors[row_index]),
                "label_standard_error": float(dataset.label_standard_error[row_index]),
                "error_to_label_se": float(
                    absolute_errors[row_index] / denominator[row_index]
                ),
                "features": {
                    name: float(dataset.X[row_index, position])
                    for name, position in feature_positions.items()
                },
            }
        )
    cells = _joint_regime_moneyness_metrics(dataset, predictions, split="validation")
    return {
        "split": "validation",
        "worst_regime_moneyness_cells": [
            {"cell": name, **values}
            for name, values in sorted(cells.items(), key=lambda item: -item[1]["mae"])
        ],
        "worst_cases": worst_cases,
    }


def _audit_metrics(
    dataset: PhoenixSurrogateDataset,
    surrogate: NumpyMLPSurrogate,
    uncertainty_policy: dict[str, Any],
) -> dict[str, Any]:
    predictions = surrogate.predict(dataset.X)
    output_predictions = surrogate.predict_outputs(dataset.X)
    raw_output_predictions = surrogate.predict_raw_outputs(dataset.X)
    price_metrics = _regression_metrics(
        dataset.y, predictions, dataset.label_standard_error
    )
    price_metrics.update(
        _uncertainty_aware_metrics(
            dataset.y,
            predictions,
            dataset.label_standard_error,
            uncertainty_policy,
        )
    )
    regime_metrics = {}
    for regime in sorted(set(str(value) for value in dataset.regime_names)):
        mask = dataset.regime_names == regime
        regime_metrics[regime] = _regression_metrics(
            dataset.y[mask],
            predictions[mask],
            dataset.label_standard_error[mask],
        )
    moneyness_metrics = {}
    for region in sorted(set(str(value) for value in dataset.moneyness_region_names)):
        mask = dataset.moneyness_region_names == region
        moneyness_metrics[region] = _regression_metrics(
            dataset.y[mask],
            predictions[mask],
            dataset.label_standard_error[mask],
        )
    if surrogate.output_names == PHOENIX_PAYOFF_AWARE_MODEL_OUTPUT_NAMES:
        output_targets = np.column_stack([dataset.y, dataset.auxiliary_targets])
        output_standard_error = np.column_stack(
            [dataset.label_standard_error, dataset.auxiliary_standard_error]
        )
    else:
        output_targets = dataset.y[:, None]
        output_standard_error = dataset.label_standard_error[:, None]
    output_metrics = {
        name: _regression_metrics(
            output_targets[:, index],
            output_predictions[:, index],
            output_standard_error[:, index],
        )
        for index, name in enumerate(surrogate.output_names)
    }
    for component_name in PHOENIX_PRICE_COMPONENT_NAMES:
        if component_name in surrogate.output_names:
            index = surrogate.output_names.index(component_name)
            violations = np.maximum(-raw_output_predictions[:, index], 0.0)
            output_metrics[component_name].update(
                {
                    "raw_outside_bounds_fraction": float(np.mean(violations > 0.0)),
                    "raw_mean_boundary_violation": float(np.mean(violations)),
                    "raw_maximum_boundary_violation": float(np.max(violations)),
                }
            )
    for event_name in ("autocall_probability", "downside_probability"):
        if event_name in surrogate.output_names:
            index = surrogate.output_names.index(event_name)
            raw_values = raw_output_predictions[:, index]
            violations = np.maximum(-raw_values, 0.0) + np.maximum(
                raw_values - 1.0, 0.0
            )
            output_metrics[event_name].update(
                {
                    "raw_outside_bounds_fraction": float(np.mean(violations > 0.0)),
                    "raw_mean_boundary_violation": float(np.mean(violations)),
                    "raw_maximum_boundary_violation": float(np.max(violations)),
                }
            )
    if surrogate.output_names == PHOENIX_PAYOFF_AWARE_MODEL_OUTPUT_NAMES:
        component_indices = [
            surrogate.output_names.index(name)
            for name in (
                "coupon_pv",
                "autocall_principal_pv",
                "maturity_protected_pv",
                "maturity_downside_pv",
            )
        ]
        component_price = np.sum(output_predictions[:, component_indices], axis=1)
        output_metrics["cashflow_reconstruction"] = {
            "mae_to_price_head": float(np.mean(np.abs(component_price - predictions))),
            "p95_absolute_gap": float(
                np.quantile(np.abs(component_price - predictions), 0.95)
            ),
        }
    return {
        "dataset_id": dataset.metadata["dataset_id"],
        "n_samples": int(len(dataset.y)),
        "uncertainty_policy": uncertainty_policy,
        "price_metrics": price_metrics,
        "regime_metrics": regime_metrics,
        "moneyness_region_metrics": moneyness_metrics,
        "regime_moneyness_metrics": _joint_regime_moneyness_metrics(
            dataset, predictions
        ),
        "output_metrics": output_metrics,
    }


def _development_test_metrics(
    dataset: PhoenixSurrogateDataset, surrogate: NumpyMLPSurrogate
) -> dict[str, Any]:
    mask = _split_mask(dataset, "test")
    predictions = surrogate.predict(dataset.X)
    regime_metrics = {}
    for regime in sorted(set(str(value) for value in dataset.regime_names[mask])):
        slice_mask = mask & (dataset.regime_names == regime)
        regime_metrics[regime] = _regression_metrics(
            dataset.y[slice_mask],
            predictions[slice_mask],
            dataset.label_standard_error[slice_mask],
        )
    return {
        "dataset_id": dataset.metadata["dataset_id"],
        "n_samples": int(np.sum(mask)),
        "price_metrics": _regression_metrics(
            dataset.y[mask],
            predictions[mask],
            dataset.label_standard_error[mask],
        ),
        "regime_metrics": regime_metrics,
        "moneyness_region_metrics": {},
        "regime_moneyness_metrics": {},
    }


def _validate_training_dataset(
    dataset: PhoenixSurrogateDataset, *, expected_role: str
) -> None:
    if dataset.metadata.get("dataset_schema_version") != DATASET_SCHEMA_VERSION:
        raise SurrogateTrainingError("dataset schema version is incompatible")
    for name, expected_value in surrogate_contract_metadata().items():
        if dataset.metadata.get(name) != expected_value:
            raise SurrogateTrainingError(f"dataset {name} contract is incompatible")
    dataset_id = dataset.metadata.get("dataset_id")
    if not isinstance(dataset_id, str) or not dataset_id.startswith("sha256:"):
        raise SurrogateTrainingError("dataset id is invalid")
    if dataset.metadata.get("n_samples") != len(dataset.y):
        raise SurrogateTrainingError("dataset sample count is inconsistent")
    if not isinstance(dataset.metadata.get("generation_environment"), dict):
        raise SurrogateTrainingError("dataset generation environment is missing")
    if dataset.metadata.get("dataset_role") != expected_role:
        raise SurrogateTrainingError(
            f"expected a {expected_role} dataset, received another role"
        )
    group_splits: dict[int, set[str]] = {}
    for group_id, split in zip(dataset.group_ids, dataset.split_names):
        group_splits.setdefault(int(group_id), set()).add(str(split))
    if any(len(splits) != 1 for splits in group_splits.values()):
        raise SurrogateTrainingError("contract groups leak across dataset splits")


def _fit_numpy_mlp_candidate(
    dataset: PhoenixSurrogateDataset,
    config: PhoenixSurrogateTrainingConfig,
    *,
    strategy: str,
    hidden_layer_sizes: tuple[int, ...],
    random_state: int,
    candidate_name: str,
) -> tuple[NumpyMLPSurrogate, dict[str, Any]]:
    from sklearn.neural_network import MLPRegressor

    train_mask = _split_mask(dataset, "train")
    feature_mean = np.mean(dataset.X[train_mask], axis=0)
    feature_scale = np.std(dataset.X[train_mask], axis=0)
    feature_scale[feature_scale < 1e-12] = 1.0
    if strategy == "direct_price":
        raw_targets = dataset.y[:, None]
        output_names = ("price",)
        target_standard_error = dataset.label_standard_error[:, None]
    elif strategy == "payoff_aware":
        raw_targets = np.column_stack([dataset.y, dataset.auxiliary_targets])
        output_names = PHOENIX_PAYOFF_AWARE_MODEL_OUTPUT_NAMES
        target_standard_error = np.column_stack(
            [dataset.label_standard_error, dataset.auxiliary_standard_error]
        )
    else:
        raise SurrogateTrainingError(f"unknown MLP strategy: {strategy}")
    target_mean = np.mean(raw_targets[train_mask], axis=0)
    target_scale = np.std(raw_targets[train_mask], axis=0)
    target_scale[target_scale < 1e-12] = 1.0
    X_train = (dataset.X[train_mask] - feature_mean) / feature_scale
    y_train = (raw_targets[train_mask] - target_mean) / target_scale
    model = MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        activation="relu",
        solver="adam",
        alpha=config.alpha,
        learning_rate_init=config.learning_rate_init,
        max_iter=config.max_iter,
        random_state=random_state,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=30,
    )
    started = time.perf_counter()
    model.fit(X_train, y_train[:, 0] if strategy == "direct_price" else y_train)
    elapsed = time.perf_counter() - started
    surrogate = NumpyMLPSurrogate(
        feature_names=PHOENIX_SURROGATE_FEATURE_NAMES,
        output_names=tuple(output_names),
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        target_mean=target_mean,
        target_scale=target_scale,
        weights=tuple(np.asarray(values, dtype=np.float64) for values in model.coefs_),
        biases=tuple(
            np.asarray(values, dtype=np.float64) for values in model.intercepts_
        ),
    )
    predictions = surrogate.predict(dataset.X)
    output_predictions = surrogate.predict_outputs(dataset.X)
    split_metrics = {}
    for split in ("train", "validation", "test"):
        mask = _split_mask(dataset, split)
        split_metrics[split] = _regression_metrics(
            dataset.y[mask],
            predictions[mask],
            dataset.label_standard_error[mask],
        )
    output_metrics = {}
    for split in ("train", "validation", "test"):
        mask = _split_mask(dataset, split)
        output_metrics[split] = {
            name: _regression_metrics(
                raw_targets[mask, index],
                output_predictions[mask, index],
                target_standard_error[mask, index],
            )
            for index, name in enumerate(output_names)
        }
    return surrogate, {
        "model_type": f"numpy-mlp-{strategy}",
        "candidate_name": candidate_name,
        "strategy": strategy,
        "hidden_layer_sizes": list(hidden_layer_sizes),
        "random_state": random_state,
        "output_names": list(output_names),
        "fit_seconds": elapsed,
        "iterations": int(model.n_iter_),
        "loss": float(model.loss_),
        "split_metrics": split_metrics,
        "regime_metrics": _regime_metrics(dataset, predictions),
        "moneyness_metrics": _moneyness_metrics(dataset, predictions),
        "regime_moneyness_validation_metrics": _joint_regime_moneyness_metrics(
            dataset, predictions, split="validation"
        ),
        "output_metrics": output_metrics,
    }


def _refit_focused_output_head(
    *,
    dataset: PhoenixSurrogateDataset,
    base_surrogate: NumpyMLPSurrogate,
    base_metrics: dict[str, Any],
    config: PhoenixSurrogateTrainingConfig,
    candidate_name: str,
) -> tuple[NumpyMLPSurrogate, dict[str, Any]]:
    if base_surrogate.output_names != PHOENIX_PAYOFF_AWARE_MODEL_OUTPUT_NAMES:
        raise SurrogateTrainingError(
            "focused output-head refit requires payoff-aware outputs"
        )
    train_mask = _split_mask(dataset, "train")
    raw_targets = np.column_stack([dataset.y, dataset.auxiliary_targets])
    target_standard_error = np.column_stack(
        [dataset.label_standard_error, dataset.auxiliary_standard_error]
    )
    standardized_targets = (
        raw_targets[train_mask] - base_surrogate.target_mean
    ) / base_surrogate.target_scale
    hidden = (
        dataset.X[train_mask] - base_surrogate.feature_mean
    ) / base_surrogate.feature_scale
    for weight, bias in zip(
        base_surrogate.weights[:-1],
        base_surrogate.biases[:-1],
    ):
        hidden = np.maximum(hidden @ weight + bias, 0.0)

    sample_weights = np.ones(int(np.sum(train_mask)), dtype=np.float64)
    sample_weights *= np.where(
        dataset.regime_names[train_mask] == "low_vol",
        config.focused_head_regime_weight,
        1.0,
    )
    sample_weights *= np.where(
        dataset.moneyness_region_names[train_mask] == "coupon",
        config.focused_head_coupon_weight,
        1.0,
    )
    design = np.column_stack([hidden, np.ones(len(hidden), dtype=np.float64)])
    weighted_design = design * np.sqrt(sample_weights)[:, None]
    weighted_targets = standardized_targets * np.sqrt(sample_weights)[:, None]
    normalizer = float(np.sum(sample_weights))
    gram = weighted_design.T @ weighted_design / normalizer
    penalty = np.eye(gram.shape[0], dtype=np.float64) * config.focused_head_ridge
    penalty[-1, -1] = 0.0
    coefficients = np.linalg.solve(
        gram + penalty,
        weighted_design.T @ weighted_targets / normalizer,
    )
    surrogate = NumpyMLPSurrogate(
        feature_names=base_surrogate.feature_names,
        output_names=base_surrogate.output_names,
        feature_mean=base_surrogate.feature_mean,
        feature_scale=base_surrogate.feature_scale,
        target_mean=base_surrogate.target_mean,
        target_scale=base_surrogate.target_scale,
        weights=base_surrogate.weights[:-1] + (coefficients[:-1],),
        biases=base_surrogate.biases[:-1] + (coefficients[-1],),
    )
    predictions = surrogate.predict(dataset.X)
    output_predictions = surrogate.predict_outputs(dataset.X)
    split_metrics = {}
    output_metrics = {}
    for split in ("train", "validation", "test"):
        mask = _split_mask(dataset, split)
        split_metrics[split] = _regression_metrics(
            dataset.y[mask],
            predictions[mask],
            dataset.label_standard_error[mask],
        )
        output_metrics[split] = {
            name: _regression_metrics(
                raw_targets[mask, index],
                output_predictions[mask, index],
                target_standard_error[mask, index],
            )
            for index, name in enumerate(surrogate.output_names)
        }
    return surrogate, {
        **{
            name: value
            for name, value in base_metrics.items()
            if name
            not in {
                "candidate_name",
                "strategy",
                "model_type",
                "split_metrics",
                "regime_metrics",
                "moneyness_metrics",
                "regime_moneyness_validation_metrics",
                "output_metrics",
            }
        },
        "model_type": "numpy-mlp-payoff-aware-focused-head",
        "candidate_name": candidate_name,
        "strategy": "payoff_aware_focused_head",
        "base_candidate": base_metrics["candidate_name"],
        "focused_head": {
            "regime": "low_vol",
            "regime_weight": config.focused_head_regime_weight,
            "moneyness_region": "coupon",
            "moneyness_weight": config.focused_head_coupon_weight,
            "joint_weight": (
                config.focused_head_regime_weight * config.focused_head_coupon_weight
            ),
            "ridge": config.focused_head_ridge,
        },
        "split_metrics": split_metrics,
        "regime_metrics": _regime_metrics(dataset, predictions),
        "moneyness_metrics": _moneyness_metrics(dataset, predictions),
        "regime_moneyness_validation_metrics": _joint_regime_moneyness_metrics(
            dataset,
            predictions,
            split="validation",
        ),
        "output_metrics": output_metrics,
    }


def _repeated_group_validation(
    dataset: PhoenixSurrogateDataset,
    predictions: np.ndarray,
    config: PhoenixSurrogateTrainingConfig,
) -> dict[str, Any]:
    validation_mask = _split_mask(dataset, "validation")
    validation_groups = np.unique(dataset.group_ids[validation_mask])
    if len(validation_groups) < config.selection_validation_folds:
        raise SurrogateTrainingError(
            "not enough validation groups for robust selection folds"
        )
    folds = []
    for repeat in range(config.selection_validation_repeats):
        rng = np.random.RandomState(config.random_state + 50_021 + repeat * 10_007)
        shuffled = rng.permutation(validation_groups)
        for fold_index, fold_groups in enumerate(
            np.array_split(shuffled, config.selection_validation_folds)
        ):
            mask = validation_mask & np.isin(dataset.group_ids, fold_groups)
            base = _regression_metrics(
                dataset.y[mask],
                predictions[mask],
                dataset.label_standard_error[mask],
            )

            def worst_mae(labels: np.ndarray) -> float:
                return max(
                    float(
                        np.mean(
                            np.abs(
                                predictions[mask & (labels == label)]
                                - dataset.y[mask & (labels == label)]
                            )
                        )
                    )
                    for label in np.unique(labels[mask])
                )

            joint_labels = np.asarray(
                [
                    f"{regime}:{region}"
                    for regime, region in zip(
                        dataset.regime_names,
                        dataset.moneyness_region_names,
                    )
                ]
            )
            worst_regime = worst_mae(dataset.regime_names)
            worst_moneyness = worst_mae(dataset.moneyness_region_names)
            worst_joint = worst_mae(joint_labels)
            score = (
                base["mae"]
                + config.selection_worst_regime_weight * worst_regime
                + config.selection_worst_moneyness_weight * worst_moneyness
                + config.selection_worst_joint_cell_weight * worst_joint
            )
            folds.append(
                {
                    "repeat": repeat,
                    "fold": fold_index,
                    "n_groups": int(len(fold_groups)),
                    "n_samples": int(np.sum(mask)),
                    "mae": base["mae"],
                    "maximum_regime_mae": worst_regime,
                    "maximum_moneyness_mae": worst_moneyness,
                    "maximum_regime_moneyness_mae": worst_joint,
                    "score": score,
                }
            )
    scores = np.asarray([fold["score"] for fold in folds])
    mean_score = float(np.mean(scores))
    worst_score = float(np.max(scores))
    return {
        "policy": "repeated-group-held-out-validation-v1",
        "folds": config.selection_validation_folds,
        "repeats": config.selection_validation_repeats,
        "n_validation_groups": int(len(validation_groups)),
        "mean_fold_score": mean_score,
        "median_fold_score": float(np.median(scores)),
        "worst_fold_score": worst_score,
        "worst_fold_weight": config.selection_worst_fold_weight,
        "selection_score": (
            mean_score + config.selection_worst_fold_weight * worst_score
        ),
        "fold_metrics": folds,
    }


def _fit_candidate_models(
    dataset: PhoenixSurrogateDataset,
    config: PhoenixSurrogateTrainingConfig,
) -> tuple[NumpyMLPSurrogate, str, str, dict[str, Any]]:
    candidates = {}
    models = {}
    direct_name = "direct_price"
    models[direct_name], candidates[direct_name] = _fit_numpy_mlp_candidate(
        dataset,
        config,
        strategy="direct_price",
        hidden_layer_sizes=config.hidden_layer_sizes,
        random_state=config.random_state,
        candidate_name=direct_name,
    )
    layouts = tuple(
        dict.fromkeys(
            (config.hidden_layer_sizes,) + config.candidate_hidden_layer_sizes
        )
    )
    for layout in layouts:
        layout_label = "x".join(str(width) for width in layout)
        for seed_offset in config.candidate_seed_offsets:
            random_state = config.random_state + seed_offset
            name = (
                "payoff_aware"
                if len(layouts) == 1 and len(config.candidate_seed_offsets) == 1
                else f"payoff_aware__{layout_label}__seed{random_state}"
            )
            models[name], candidates[name] = _fit_numpy_mlp_candidate(
                dataset,
                config,
                strategy="payoff_aware",
                hidden_layer_sizes=layout,
                random_state=random_state,
                candidate_name=name,
            )
            if config.train_focused_head_candidates:
                focused_name = f"{name}__focused_head"
                models[focused_name], candidates[focused_name] = (
                    _refit_focused_output_head(
                        dataset=dataset,
                        base_surrogate=models[name],
                        base_metrics=candidates[name],
                        config=config,
                        candidate_name=focused_name,
                    )
                )
    for name, metrics in candidates.items():
        predictions = models[name].predict(dataset.X)
        validation_mae = metrics["split_metrics"]["validation"]["mae"]
        maximum_regime_mae = max(
            values["mae"] for values in metrics["regime_metrics"]["validation"].values()
        )
        maximum_moneyness_mae = max(
            values["mae"]
            for values in metrics["moneyness_metrics"]["validation"].values()
        )
        maximum_joint_cell_mae = max(
            values["mae"]
            for values in metrics["regime_moneyness_validation_metrics"].values()
        )
        repeated_validation = _repeated_group_validation(
            dataset,
            predictions,
            config,
        )
        metrics["selection"] = {
            "policy": "robust-validation-mae-v3",
            "validation_mae": validation_mae,
            "maximum_validation_regime_mae": maximum_regime_mae,
            "maximum_validation_moneyness_mae": maximum_moneyness_mae,
            "maximum_validation_regime_moneyness_mae": (maximum_joint_cell_mae),
            "single_split_score": (
                validation_mae
                + config.selection_worst_regime_weight * maximum_regime_mae
                + config.selection_worst_moneyness_weight * maximum_moneyness_mae
                + config.selection_worst_joint_cell_weight * maximum_joint_cell_mae
            ),
            "repeated_group_validation": repeated_validation,
            "score": repeated_validation["selection_score"],
        }
    selected_candidate = min(
        candidates,
        key=lambda name: (
            candidates[name]["selection"]["score"],
            candidates[name]["selection"]["validation_mae"],
            name,
        ),
    )
    selected_strategy = candidates[selected_candidate]["strategy"]
    return (
        models[selected_candidate],
        selected_candidate,
        selected_strategy,
        candidates,
    )


def _lightgbm_baseline(
    dataset: PhoenixSurrogateDataset,
    config: PhoenixSurrogateTrainingConfig,
) -> dict[str, Any] | None:
    if not config.train_lightgbm_baseline:
        return None
    from lightgbm import LGBMRegressor

    train_mask = _split_mask(dataset, "train")
    model = LGBMRegressor(
        n_estimators=600,
        learning_rate=0.03,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=0.001,
        random_state=config.random_state,
        n_jobs=1,
        verbosity=-1,
    )
    started = time.perf_counter()
    model.fit(dataset.X[train_mask], dataset.y[train_mask])
    predictions = np.asarray(model.booster_.predict(dataset.X), dtype=np.float64)
    split_metrics = {}
    for split in ("train", "validation", "test"):
        mask = _split_mask(dataset, split)
        split_metrics[split] = _regression_metrics(
            dataset.y[mask],
            predictions[mask],
            dataset.label_standard_error[mask],
        )
    importance = sorted(
        [
            {
                "feature": PHOENIX_SURROGATE_FEATURE_NAMES[index],
                "importance": float(value),
            }
            for index, value in enumerate(model.feature_importances_)
        ],
        key=lambda item: -item["importance"],
    )
    return {
        "model_type": "lightgbm-research-baseline",
        "fit_seconds": time.perf_counter() - started,
        "split_metrics": split_metrics,
        "regime_metrics": _regime_metrics(dataset, predictions),
        "feature_importance": importance,
        "runtime_eligible": False,
    }


def _bump_market(
    market: EquityMarketTermStructure,
    *,
    spot_relative: float = 0.0,
    volatility_absolute: float = 0.0,
    rate_absolute: float = 0.0,
    dividend_absolute: float = 0.0,
) -> EquityMarketTermStructure:
    return EquityMarketTermStructure(
        symbol=market.symbol,
        underlier_type=market.underlier_type,
        currency=market.currency,
        valuation_time=market.valuation_time,
        market_data_time=market.market_data_time,
        spot=market.spot * (1.0 + spot_relative),
        segments=tuple(
            EquityMarketSegment(
                end_time_years=segment.end_time_years,
                risk_free_rate=segment.risk_free_rate + rate_absolute,
                dividend_yield=segment.dividend_yield + dividend_absolute,
                volatility=segment.volatility + volatility_absolute,
            )
            for segment in market.segments
        ),
        calendar=market.calendar,
        day_count=market.day_count,
        source=market.source,
    )


def _surrogate_price(
    surrogate: NumpyMLPSurrogate,
    market: EquityMarketTermStructure,
    terms: dict[str, Any],
    reference_spot: float,
) -> float:
    features = extract_phoenix_surrogate_features(
        market=market,
        terms=terms,
        contract_reference_spot=reference_spot,
    )
    return float(surrogate.predict(features)[0])


def _greek_validation(
    surrogate: NumpyMLPSurrogate,
    dataset: PhoenixSurrogateDataset,
    config: PhoenixSurrogateTrainingConfig,
) -> dict[str, Any]:
    if config.greek_validation_cases == 0:
        return {"n_cases": 0, "metrics": {}}
    eligible_indices = (
        np.arange(len(dataset.y))
        if dataset.metadata.get("dataset_role") == "audit"
        else np.flatnonzero(dataset.split_names == "test")
    )
    selected = eligible_indices[: config.greek_validation_cases]
    if len(selected) == 0:
        raise SurrogateTrainingError("Greek validation requires test samples")
    measures: dict[str, list[tuple[float, float]]] = {
        name: [] for name in ("delta", "gamma", "vega", "rho", "dividend_rho")
    }
    payoff = PhoenixPayoff()
    for case_number, row_index in enumerate(selected):
        market, terms, reference_spot = reconstruct_phoenix_surrogate_case(
            dataset.X[row_index]
        )
        rng = np.random.RandomState(config.random_state + case_number * 1_009)
        half = rng.standard_normal(
            (config.greek_validation_paths // 2, DEFAULT_REFERENCE_STEPS)
        )
        common_shocks = np.vstack([half, -half])
        params = {
            "S0": reference_spot,
            "r": 0.0,
            "sigma": 1.0,
            "T": terms["maturity_years"],
            **{
                name: terms[name]
                for name in (
                    "autocall_barrier_frac",
                    "coupon_barrier_frac",
                    "coupon_rate",
                    "knock_in_frac",
                    "obs_count",
                )
            },
        }
        spot_h = market.spot * 0.01
        market_variants = {
            "base": market,
            "spot_up": _bump_market(market, spot_relative=0.01),
            "spot_down": _bump_market(market, spot_relative=-0.01),
            "vol_up": _bump_market(market, volatility_absolute=0.01),
            "vol_down": _bump_market(market, volatility_absolute=-0.01),
            "rate_up": _bump_market(market, rate_absolute=0.001),
            "rate_down": _bump_market(market, rate_absolute=-0.001),
            "dividend_up": _bump_market(market, dividend_absolute=0.001),
            "dividend_down": _bump_market(market, dividend_absolute=-0.001),
        }
        pathwise = {
            name: phoenix_piecewise_discounted_payoffs(
                payoff=payoff,
                params=params,
                market=variant,
                n_paths=config.greek_validation_paths,
                standard_normal_shocks=common_shocks,
                seed=None,
            )
            for name, variant in market_variants.items()
        }
        predictions = {
            name: _surrogate_price(surrogate, variant, terms, reference_spot)
            for name, variant in market_variants.items()
        }
        reference_values = {
            "delta": float(
                np.mean(pathwise["spot_up"] - pathwise["spot_down"]) / (2.0 * spot_h)
            ),
            "gamma": float(
                np.mean(
                    pathwise["spot_up"] - 2.0 * pathwise["base"] + pathwise["spot_down"]
                )
                / (spot_h**2)
            ),
            "vega": float(np.mean(pathwise["vol_up"] - pathwise["vol_down"]) / 2.0),
            "rho": float(np.mean(pathwise["rate_up"] - pathwise["rate_down"]) / 0.2),
            "dividend_rho": float(
                np.mean(pathwise["dividend_up"] - pathwise["dividend_down"]) / 0.2
            ),
        }
        surrogate_values = {
            "delta": (predictions["spot_up"] - predictions["spot_down"])
            / (2.0 * spot_h),
            "gamma": (
                predictions["spot_up"]
                - 2.0 * predictions["base"]
                + predictions["spot_down"]
            )
            / (spot_h**2),
            "vega": (predictions["vol_up"] - predictions["vol_down"]) / 2.0,
            "rho": (predictions["rate_up"] - predictions["rate_down"]) / 0.2,
            "dividend_rho": (predictions["dividend_up"] - predictions["dividend_down"])
            / 0.2,
        }
        for name in measures:
            measures[name].append((reference_values[name], surrogate_values[name]))

    metrics = {}
    for name, pairs in measures.items():
        reference = np.asarray([pair[0] for pair in pairs])
        predicted = np.asarray([pair[1] for pair in pairs])
        nonzero = np.abs(reference) > 1e-10
        sign_agreement = (
            float(np.mean(np.sign(reference[nonzero]) == np.sign(predicted[nonzero])))
            if np.any(nonzero)
            else 1.0
        )
        metrics[name] = {
            "mae": float(np.mean(np.abs(predicted - reference))),
            "sign_agreement": sign_agreement,
            "reference_mean_absolute": float(np.mean(np.abs(reference))),
            "surrogate_mean_absolute": float(np.mean(np.abs(predicted))),
        }
    return {"n_cases": int(len(selected)), "metrics": metrics}


def _acceptance(
    *,
    evaluation_metrics: dict[str, Any],
    greek_validation: dict[str, Any],
    config: PhoenixSurrogateTrainingConfig,
) -> dict[str, Any]:
    test = evaluation_metrics["price_metrics"]
    maximum_regime_mae = max(
        values["mae"] for values in evaluation_metrics["regime_metrics"].values()
    )
    maximum_moneyness_region_mae = max(
        values["mae"]
        for values in evaluation_metrics["moneyness_region_metrics"].values()
    )
    maximum_regime_moneyness_mae = max(
        values["mae"]
        for values in evaluation_metrics["regime_moneyness_metrics"].values()
    )
    checks = {
        "audit_mae": {
            "value": test["mae"],
            "maximum": config.acceptance_audit_mae,
            "passed": test["mae"] <= config.acceptance_audit_mae,
        },
        "audit_p95_absolute_error": {
            "value": test["p95_absolute_error"],
            "maximum": config.acceptance_audit_p95_absolute_error,
            "passed": test["p95_absolute_error"]
            <= config.acceptance_audit_p95_absolute_error,
        },
        "audit_r2": {
            "value": test["r2"],
            "minimum": config.acceptance_audit_r2,
            "passed": test["r2"] >= config.acceptance_audit_r2,
        },
        "maximum_audit_regime_mae": {
            "value": maximum_regime_mae,
            "maximum": config.acceptance_maximum_regime_mae,
            "passed": maximum_regime_mae <= config.acceptance_maximum_regime_mae,
        },
        "maximum_moneyness_region_mae": {
            "value": maximum_moneyness_region_mae,
            "maximum": config.acceptance_maximum_moneyness_region_mae,
            "passed": maximum_moneyness_region_mae
            <= config.acceptance_maximum_moneyness_region_mae,
        },
        "maximum_regime_moneyness_cell_mae": {
            "value": maximum_regime_moneyness_mae,
            "maximum": config.acceptance_maximum_regime_moneyness_mae,
            "passed": maximum_regime_moneyness_mae
            <= config.acceptance_maximum_regime_moneyness_mae,
        },
        "audit_label_confidence_half_width_p95": {
            "value": test["label_confidence_half_width_p95"],
            "maximum": config.acceptance_maximum_label_confidence_half_width_p95,
            "passed": test["label_confidence_half_width_p95"]
            <= config.acceptance_maximum_label_confidence_half_width_p95,
        },
        "uncertainty_or_economic_tolerance_coverage": {
            "value": test["within_uncertainty_or_economic_tolerance_fraction"],
            "minimum": (config.acceptance_minimum_uncertainty_or_economic_coverage),
            "passed": test["within_uncertainty_or_economic_tolerance_fraction"]
            >= config.acceptance_minimum_uncertainty_or_economic_coverage,
        },
    }
    output_metrics = evaluation_metrics.get("output_metrics", {})
    if all(name in output_metrics for name in PHOENIX_PAYOFF_AWARE_TARGET_NAMES):
        maximum_component_mae = max(
            output_metrics[name]["mae"] for name in PHOENIX_PRICE_COMPONENT_NAMES
        )
        maximum_event_mae = max(
            output_metrics[name]["mae"] for name in PHOENIX_EVENT_TARGET_NAMES
        )
        maximum_mean_boundary_violation = max(
            output_metrics[name]["raw_mean_boundary_violation"]
            for name in PHOENIX_PAYOFF_AWARE_TARGET_NAMES
        )
        reconstruction_mae = output_metrics["cashflow_reconstruction"][
            "mae_to_price_head"
        ]
        checks.update(
            {
                "maximum_cashflow_component_mae": {
                    "value": maximum_component_mae,
                    "maximum": config.acceptance_maximum_component_mae,
                    "passed": maximum_component_mae
                    <= config.acceptance_maximum_component_mae,
                },
                "maximum_event_probability_mae": {
                    "value": maximum_event_mae,
                    "maximum": config.acceptance_maximum_event_mae,
                    "passed": maximum_event_mae <= config.acceptance_maximum_event_mae,
                },
                "maximum_mean_output_boundary_violation": {
                    "value": maximum_mean_boundary_violation,
                    "maximum": (
                        config.acceptance_maximum_mean_output_boundary_violation
                    ),
                    "passed": maximum_mean_boundary_violation
                    <= config.acceptance_maximum_mean_output_boundary_violation,
                },
                "cashflow_reconstruction_mae": {
                    "value": reconstruction_mae,
                    "maximum": config.acceptance_maximum_cashflow_reconstruction_mae,
                    "passed": reconstruction_mae
                    <= config.acceptance_maximum_cashflow_reconstruction_mae,
                },
            }
        )
    if greek_validation["n_cases"]:
        for name in ("delta", "vega", "rho"):
            value = greek_validation["metrics"][name]["sign_agreement"]
            checks[f"{name}_sign_agreement"] = {
                "value": value,
                "minimum": config.acceptance_minimum_greek_sign_agreement,
                "passed": value >= config.acceptance_minimum_greek_sign_agreement,
            }
    return {
        "passed": all(check["passed"] for check in checks.values()),
        "evaluation_dataset_id": evaluation_metrics["dataset_id"],
        "checks": checks,
    }


def _write_weights(path: Path, surrogate: NumpyMLPSurrogate) -> None:
    payload: dict[str, Any] = {
        "feature_mean": surrogate.feature_mean,
        "feature_scale": surrogate.feature_scale,
        "target_mean": np.asarray(surrogate.target_mean),
        "target_scale": np.asarray(surrogate.target_scale),
        "n_layers": np.asarray(len(surrogate.weights), dtype=np.int64),
    }
    for index, (weight, bias) in enumerate(zip(surrogate.weights, surrogate.biases)):
        payload[f"weight_{index}"] = weight
        payload[f"bias_{index}"] = bias
    with path.open("wb") as handle:
        np.savez_compressed(handle, **payload)


def _training_environment(config: PhoenixSurrogateTrainingConfig) -> dict[str, str]:
    packages = ["numpy", "scikit-learn", "scipy"]
    if config.train_lightgbm_baseline:
        packages.append("lightgbm")
    environment = {"python": platform.python_version()}
    for package in packages:
        try:
            environment[package] = version(package)
        except PackageNotFoundError:
            environment[package] = "unknown"
    return environment


def train_phoenix_surrogate(
    *,
    dataset: PhoenixSurrogateDataset,
    audit_dataset: PhoenixSurrogateDataset | None = None,
    output_root: Path,
    config: PhoenixSurrogateTrainingConfig,
    verbose: bool = True,
) -> dict[str, Any]:
    _validate_training_dataset(dataset, expected_role="development")
    if audit_dataset is None and config.require_audit_dataset:
        raise SurrogateTrainingError(
            "a sealed audit dataset is required for this training configuration"
        )
    if audit_dataset is not None:
        _validate_training_dataset(audit_dataset, expected_role="audit")
        if audit_dataset.metadata["dataset_id"] == dataset.metadata["dataset_id"]:
            raise SurrogateTrainingError(
                "development and audit datasets must be independent"
            )
        development_config = dataset.metadata.get("config", {})
        audit_config = audit_dataset.metadata.get("config", {})
        if any(
            development_config.get(name) == audit_config.get(name)
            for name in ("dataset_seed", "label_seed")
        ):
            raise SurrogateTrainingError(
                "development and audit datasets must use independent seeds"
            )
    uncertainty_policy = (
        _audit_uncertainty_policy(audit_dataset, config)
        if audit_dataset is not None
        else None
    )
    started = time.perf_counter()
    (
        surrogate,
        selected_candidate,
        selected_strategy,
        candidate_metrics,
    ) = _fit_candidate_models(dataset, config)
    selected_metrics = candidate_metrics[selected_candidate]
    development_predictions = surrogate.predict(dataset.X)
    baseline_metrics = _lightgbm_baseline(dataset, config)
    evaluation_metrics = (
        _audit_metrics(audit_dataset, surrogate, uncertainty_policy)
        if audit_dataset is not None
        else _development_test_metrics(dataset, surrogate)
    )
    greek_dataset = audit_dataset if audit_dataset is not None else dataset
    greek_metrics = _greek_validation(surrogate, greek_dataset, config)
    acceptance = _acceptance(
        evaluation_metrics=evaluation_metrics,
        greek_validation=greek_metrics,
        config=config,
    )

    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    temporary = root / f".training-{os.getpid()}-{time.time_ns()}"
    temporary.mkdir()
    weights_path = temporary / "weights.npz"
    _write_weights(weights_path, surrogate)
    weights_checksum = file_sha256(weights_path)
    identity = {
        "artifact_schema_version": PHOENIX_SURROGATE_ARTIFACT_VERSION,
        "model_version": PHOENIX_SURROGATE_MODEL_VERSION,
        "dataset_id": dataset.metadata["dataset_id"],
        "audit_dataset_id": (
            audit_dataset.metadata["dataset_id"] if audit_dataset is not None else None
        ),
        "dataset_generation_environment": dataset.metadata["generation_environment"],
        "feature_schema_version": dataset.metadata["feature_schema_version"],
        "training_config": asdict(config),
        "training_environment": _training_environment(config),
        "selected_candidate": selected_candidate,
        "selected_strategy": selected_strategy,
        "output_names": list(surrogate.output_names),
        "audit_uncertainty_policy": uncertainty_policy,
        "weights_sha256": weights_checksum,
    }
    artifact_id = _json_sha256(identity)
    directory_name = artifact_id.removeprefix("sha256:")
    final_directory = root / directory_name
    manifest = {
        **identity,
        "artifact_id": artifact_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "deployment_status": (
            "shadow_approved" if acceptance["passed"] else "research_only"
        ),
        "model_type": f"numpy-mlp-{selected_strategy}",
        "feature_names": list(PHOENIX_SURROGATE_FEATURE_NAMES),
        "output_names": list(surrogate.output_names),
        "contract_version": dataset.metadata["contract_version"],
        "market_data_version": dataset.metadata["market_data_version"],
        "label_model_version": dataset.metadata["label_model_version"],
        "label_schema_version": dataset.metadata["label_schema_version"],
        "training_domain": dataset.metadata["training_domain"],
        "dataset": {
            "dataset_id": dataset.metadata["dataset_id"],
            "dataset_schema_version": dataset.metadata["dataset_schema_version"],
            "n_samples": dataset.metadata["n_samples"],
            "split_counts": dataset.metadata["split_counts"],
            "split_group_counts": dataset.metadata["split_group_counts"],
            "config": dataset.metadata["config"],
            "generation_environment": dataset.metadata["generation_environment"],
        },
        "audit_dataset": (
            {
                "dataset_id": audit_dataset.metadata["dataset_id"],
                "dataset_schema_version": audit_dataset.metadata[
                    "dataset_schema_version"
                ],
                "n_samples": audit_dataset.metadata["n_samples"],
                "config": audit_dataset.metadata["config"],
                "generation_environment": audit_dataset.metadata[
                    "generation_environment"
                ],
            }
            if audit_dataset is not None
            else None
        ),
        "selected_strategy": selected_strategy,
        "selected_candidate": selected_candidate,
        "selected_model": selected_metrics,
        "candidate_models": candidate_metrics,
        "development_error_analysis": _development_error_analysis(
            dataset, development_predictions
        ),
        "lightgbm_baseline": baseline_metrics,
        "audit_evaluation": evaluation_metrics,
        "greek_validation": greek_metrics,
        "acceptance": acceptance,
        "files": {"weights.npz": weights_checksum},
        "training_seconds": time.perf_counter() - started,
        "runtime_policy": "shadow-only",
    }
    (temporary / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    if final_directory.exists():
        for child in temporary.iterdir():
            child.unlink()
        temporary.rmdir()
    else:
        os.replace(temporary, final_directory)
    pointer_path = root / "current.json"
    pointer_temp = root / f".current-{os.getpid()}.tmp"
    pointer_temp.write_text(
        json.dumps(
            {
                "artifact_id": artifact_id,
                "directory": directory_name,
                "deployment_status": manifest["deployment_status"],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    os.replace(pointer_temp, pointer_path)
    if verbose:
        print(
            "[PhoenixSurrogateTrainer] "
            f"artifact={artifact_id} status={manifest['deployment_status']}",
            flush=True,
        )
        print(
            "[PhoenixSurrogateTrainer] "
            f"audit_mae={evaluation_metrics['price_metrics']['mae']:.6f}",
            flush=True,
        )
    return manifest
