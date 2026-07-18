import math
import time
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .surrogate_contract import (
    PHOENIX_PAYOFF_AWARE_MODEL_OUTPUT_NAMES,
    PHOENIX_SURROGATE_FEATURE_NAMES,
)
from .surrogate_hazard_data import PhoenixHazardDataset
from .surrogate_model import NumpyMLPSurrogate
from .surrogate_trainer import (
    PhoenixSurrogateTrainingConfig,
    SurrogateTrainingError,
    _joint_regime_moneyness_metrics,
    _moneyness_metrics,
    _regime_metrics,
    _regression_metrics,
    _repeated_group_validation,
)


PHOENIX_EVENT_SUMMARY_HYBRID_RESEARCH_VERSION = (
    "phoenix-event-summary-hybrid-research-v1"
)
PHOENIX_EVENT_SUMMARY_TARGET_NAMES = (
    "conditional_expected_autocall_time_fraction",
    "conditional_autocall_time_variance",
    "final_survival_probability",
    "expected_coupon_count",
    "early_coupon_mass",
    "late_coupon_mass",
)
PHOENIX_EVENT_SUMMARY_HYBRID_OUTPUT_NAMES = (
    PHOENIX_PAYOFF_AWARE_MODEL_OUTPUT_NAMES + PHOENIX_EVENT_SUMMARY_TARGET_NAMES
)


def summarize_phoenix_observation_events(
    dataset: PhoenixHazardDataset,
) -> np.ndarray:
    """Compress variable-length observation labels into fixed-width targets."""
    observation_count = np.sum(dataset.observation_mask, axis=1)
    if np.any(observation_count < 1):
        raise SurrogateTrainingError("hybrid event labels have no observations")

    observation_number = np.arange(
        1,
        dataset.observation_mask.shape[1] + 1,
        dtype=np.float64,
    )[None, :]
    observation_fraction = observation_number / observation_count[:, None]
    autocall_probability = np.sum(dataset.first_autocall_probability, axis=1)
    autocall_time_numerator = np.sum(
        dataset.first_autocall_probability * observation_fraction,
        axis=1,
    )
    expected_autocall_time = np.divide(
        autocall_time_numerator,
        autocall_probability,
        out=np.zeros_like(autocall_probability),
        where=autocall_probability > 0.0,
    )
    centered_time = observation_fraction - expected_autocall_time[:, None]
    autocall_time_variance = np.divide(
        np.sum(dataset.first_autocall_probability * centered_time**2, axis=1),
        autocall_probability,
        out=np.zeros_like(autocall_probability),
        where=autocall_probability > 0.0,
    )

    last_observation = observation_count - 1
    final_survival = dataset.survival_after_probability[
        np.arange(len(observation_count)),
        last_observation,
    ]
    expected_coupon_count = np.sum(dataset.coupon_probability, axis=1)
    early_mask = dataset.observation_mask & (observation_fraction <= 0.5)
    late_mask = dataset.observation_mask & ~early_mask
    early_coupon_mass = np.sum(
        np.where(early_mask, dataset.coupon_probability, 0.0),
        axis=1,
    )
    late_coupon_mass = np.sum(
        np.where(late_mask, dataset.coupon_probability, 0.0),
        axis=1,
    )
    summaries = np.column_stack(
        [
            expected_autocall_time,
            autocall_time_variance,
            final_survival,
            expected_coupon_count,
            early_coupon_mass,
            late_coupon_mass,
        ]
    )
    if not np.all(np.isfinite(summaries)):
        raise SurrogateTrainingError("hybrid event summaries must be finite")
    if np.any((summaries[:, :3] < -1e-12) | (summaries[:, :3] > 1.0 + 1e-12)):
        raise SurrogateTrainingError("hybrid probability summaries violate bounds")
    if np.any(summaries[:, 3:] < -1e-12):
        raise SurrogateTrainingError("hybrid coupon summaries violate bounds")
    if not np.allclose(
        early_coupon_mass + late_coupon_mass,
        expected_coupon_count,
        rtol=0.0,
        atol=1e-12,
    ):
        raise SurrogateTrainingError("hybrid coupon summaries do not reconcile")
    return summaries


@dataclass(frozen=True)
class PhoenixEventSummaryHybridSurrogate:
    """Research-only multi-task network with an independent direct-price head."""

    network: NumpyMLPSurrogate

    def __post_init__(self) -> None:
        if self.network.output_names != PHOENIX_EVENT_SUMMARY_HYBRID_OUTPUT_NAMES:
            raise SurrogateTrainingError("hybrid output contract is incompatible")

    def predict_raw_outputs(
        self,
        features: Sequence[float] | np.ndarray,
    ) -> np.ndarray:
        return self.network.predict_raw_outputs(features)

    def predict_outputs(
        self,
        features: Sequence[float] | np.ndarray,
    ) -> np.ndarray:
        outputs = self.network.predict_outputs(features)
        for name in PHOENIX_EVENT_SUMMARY_TARGET_NAMES[:3]:
            index = self.network.output_names.index(name)
            outputs[:, index] = np.clip(outputs[:, index], 0.0, 1.0)
        for name in PHOENIX_EVENT_SUMMARY_TARGET_NAMES[3:]:
            index = self.network.output_names.index(name)
            outputs[:, index] = np.maximum(outputs[:, index], 0.0)
        return outputs

    def predict(
        self,
        features: Sequence[float] | np.ndarray,
    ) -> np.ndarray:
        outputs = self.predict_outputs(features)
        return outputs[:, self.network.output_names.index("price")]


def _fit_shared_network(
    *,
    dataset: PhoenixHazardDataset,
    targets: np.ndarray,
    config: PhoenixSurrogateTrainingConfig,
    hidden_layer_sizes: tuple[int, ...],
    random_state: int,
) -> tuple[NumpyMLPSurrogate, dict[str, Any]]:
    from sklearn.neural_network import MLPRegressor

    base = dataset.base
    train_mask = base.split_names == "train"
    feature_mean = np.mean(base.X[train_mask], axis=0)
    feature_scale = np.std(base.X[train_mask], axis=0)
    feature_scale[feature_scale < 1e-12] = 1.0
    target_mean = np.mean(targets[train_mask], axis=0)
    target_scale = np.std(targets[train_mask], axis=0)
    target_scale[target_scale < 1e-12] = 1.0
    X_train = (base.X[train_mask] - feature_mean) / feature_scale
    y_train = (targets[train_mask] - target_mean) / target_scale
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
    model.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - started
    return (
        NumpyMLPSurrogate(
            feature_names=PHOENIX_SURROGATE_FEATURE_NAMES,
            output_names=PHOENIX_EVENT_SUMMARY_HYBRID_OUTPUT_NAMES,
            feature_mean=feature_mean,
            feature_scale=feature_scale,
            target_mean=target_mean,
            target_scale=target_scale,
            weights=tuple(
                np.asarray(values, dtype=np.float64) for values in model.coefs_
            ),
            biases=tuple(
                np.asarray(values, dtype=np.float64) for values in model.intercepts_
            ),
        ),
        {
            "fit_seconds": fit_seconds,
            "iterations": int(model.n_iter_),
            "loss": float(model.loss_),
        },
    )


def _refit_focused_head(
    *,
    dataset: PhoenixHazardDataset,
    targets: np.ndarray,
    network: NumpyMLPSurrogate,
    config: PhoenixSurrogateTrainingConfig,
) -> NumpyMLPSurrogate:
    base = dataset.base
    train_mask = base.split_names == "train"
    standardized_targets = (
        targets[train_mask] - network.target_mean
    ) / network.target_scale
    hidden = (base.X[train_mask] - network.feature_mean) / network.feature_scale
    for weight, bias in zip(network.weights[:-1], network.biases[:-1]):
        hidden = np.maximum(hidden @ weight + bias, 0.0)

    sample_weights = np.ones(int(np.sum(train_mask)), dtype=np.float64)
    sample_weights *= np.where(
        base.regime_names[train_mask] == "low_vol",
        config.focused_head_regime_weight,
        1.0,
    )
    sample_weights *= np.where(
        base.moneyness_region_names[train_mask] == "coupon",
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
    return NumpyMLPSurrogate(
        feature_names=network.feature_names,
        output_names=network.output_names,
        feature_mean=network.feature_mean,
        feature_scale=network.feature_scale,
        target_mean=network.target_mean,
        target_scale=network.target_scale,
        weights=network.weights[:-1] + (coefficients[:-1],),
        biases=network.biases[:-1] + (coefficients[-1],),
    )


def _plain_regression_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> dict[str, Any]:
    errors = predicted - actual
    absolute = np.abs(errors)
    return {
        "n_samples": int(len(actual)),
        "mae": float(np.mean(absolute)),
        "rmse": float(math.sqrt(np.mean(errors**2))),
        "p95_absolute_error": float(np.quantile(absolute, 0.95)),
        "mean_error": float(np.mean(errors)),
    }


def _evaluate_candidate(
    *,
    name: str,
    strategy: str,
    dataset: PhoenixHazardDataset,
    targets: np.ndarray,
    network: NumpyMLPSurrogate,
    config: PhoenixSurrogateTrainingConfig,
    fit: dict[str, Any],
) -> dict[str, Any]:
    base = dataset.base
    surrogate = PhoenixEventSummaryHybridSurrogate(network)
    predictions = surrogate.predict(base.X)
    output_predictions = surrogate.predict_outputs(base.X)
    split_metrics = {}
    output_metrics = {}
    for split in ("train", "validation", "test"):
        mask = base.split_names == split
        split_metrics[split] = _regression_metrics(
            base.y[mask],
            predictions[mask],
            base.label_standard_error[mask],
        )
        output_metrics[split] = {
            output_name: _plain_regression_metrics(
                targets[mask, output_index],
                output_predictions[mask, output_index],
            )
            for output_index, output_name in enumerate(network.output_names)
        }

    regime_metrics = _regime_metrics(base, predictions)
    moneyness_metrics = _moneyness_metrics(base, predictions)
    joint_metrics = _joint_regime_moneyness_metrics(
        base,
        predictions,
        split="validation",
    )
    validation_metrics = split_metrics["validation"]
    maximum_regime_mae = max(
        values["mae"] for values in regime_metrics["validation"].values()
    )
    maximum_moneyness_mae = max(
        values["mae"] for values in moneyness_metrics["validation"].values()
    )
    maximum_joint_mae = max(values["mae"] for values in joint_metrics.values())
    repeated_validation = _repeated_group_validation(base, predictions, config)
    return {
        "candidate_name": name,
        "model_type": "numpy-mlp-event-summary-hybrid-research",
        "strategy": strategy,
        "fit": fit,
        "split_metrics": split_metrics,
        "regime_metrics": regime_metrics,
        "moneyness_metrics": moneyness_metrics,
        "regime_moneyness_validation_metrics": joint_metrics,
        "output_metrics": output_metrics,
        "selection": {
            "policy": "event-summary-hybrid-development-comparison-v1",
            "validation_mae": validation_metrics["mae"],
            "maximum_validation_regime_mae": maximum_regime_mae,
            "maximum_validation_moneyness_mae": maximum_moneyness_mae,
            "maximum_validation_regime_moneyness_mae": maximum_joint_mae,
            "single_split_score": (
                validation_metrics["mae"]
                + config.selection_worst_regime_weight * maximum_regime_mae
                + config.selection_worst_moneyness_weight * maximum_moneyness_mae
                + config.selection_worst_joint_cell_weight * maximum_joint_mae
            ),
            "repeated_group_validation": repeated_validation,
            "score": repeated_validation["selection_score"],
        },
    }


def train_phoenix_event_summary_hybrid_candidate(
    dataset: PhoenixHazardDataset,
    config: PhoenixSurrogateTrainingConfig,
    *,
    hidden_layer_sizes: tuple[int, ...] = (256, 128, 64),
    random_state: int = 143,
) -> tuple[PhoenixEventSummaryHybridSurrogate, dict[str, Any]]:
    """Fit a development-only direct-price model with event-summary auxiliaries."""
    base = dataset.base
    if base.metadata.get("dataset_role") != "development":
        raise SurrogateTrainingError("hybrid candidate requires a development dataset")
    if not hidden_layer_sizes or any(
        width < 1 or width > 2_048 for width in hidden_layer_sizes
    ):
        raise SurrogateTrainingError("hybrid hidden_layer_sizes are invalid")
    if random_state < 0 or random_state > 1_000_000:
        raise SurrogateTrainingError("hybrid random_state is invalid")

    summaries = summarize_phoenix_observation_events(dataset)
    targets = np.column_stack([base.y, base.auxiliary_targets, summaries])
    shared_network, fit = _fit_shared_network(
        dataset=dataset,
        targets=targets,
        config=config,
        hidden_layer_sizes=hidden_layer_sizes,
        random_state=random_state,
    )
    focused_network = _refit_focused_head(
        dataset=dataset,
        targets=targets,
        network=shared_network,
        config=config,
    )
    base_name = (
        "event_summary_hybrid__"
        + "x".join(str(width) for width in hidden_layer_sizes)
        + f"__seed{random_state}"
    )
    candidates = {
        base_name: _evaluate_candidate(
            name=base_name,
            strategy="event_summary_hybrid",
            dataset=dataset,
            targets=targets,
            network=shared_network,
            config=config,
            fit=fit,
        ),
        f"{base_name}__focused_head": _evaluate_candidate(
            name=f"{base_name}__focused_head",
            strategy="event_summary_hybrid_focused_head",
            dataset=dataset,
            targets=targets,
            network=focused_network,
            config=config,
            fit={
                **fit,
                "focused_head": {
                    "regime": "low_vol",
                    "regime_weight": config.focused_head_regime_weight,
                    "moneyness_region": "coupon",
                    "moneyness_weight": config.focused_head_coupon_weight,
                    "joint_weight": (
                        config.focused_head_regime_weight
                        * config.focused_head_coupon_weight
                    ),
                    "ridge": config.focused_head_ridge,
                },
            },
        ),
    }
    models = {
        base_name: shared_network,
        f"{base_name}__focused_head": focused_network,
    }
    selected_name = min(
        candidates,
        key=lambda name: (
            candidates[name]["selection"]["score"],
            candidates[name]["selection"]["validation_mae"],
            name,
        ),
    )
    report = {
        "research_version": PHOENIX_EVENT_SUMMARY_HYBRID_RESEARCH_VERSION,
        "dataset_id": dataset.metadata["dataset_id"],
        "base_dataset_id": base.metadata["dataset_id"],
        "runtime_eligible": False,
        "audit_evaluated": False,
        "deployment_status": "research_only",
        "exclusion_reason": (
            "development-only event-summary architecture experiment; not part "
            "of the production artifact contract"
        ),
        "training_split": "train",
        "selection_split": "validation",
        "hidden_layer_sizes": list(hidden_layer_sizes),
        "random_state": random_state,
        "validation_seed": config.random_state,
        "output_names": list(PHOENIX_EVENT_SUMMARY_HYBRID_OUTPUT_NAMES),
        "event_summary_target_names": list(PHOENIX_EVENT_SUMMARY_TARGET_NAMES),
        "loss_policy": "equal-weight-standardized-multi-task-squared-error-v1",
        "price_inference": "independent direct-price output",
        "candidate_models": candidates,
        "selected_candidate": selected_name,
        "selection": candidates[selected_name]["selection"],
        "split_metrics": candidates[selected_name]["split_metrics"],
        "regime_metrics": candidates[selected_name]["regime_metrics"],
        "moneyness_metrics": candidates[selected_name]["moneyness_metrics"],
        "regime_moneyness_validation_metrics": candidates[selected_name][
            "regime_moneyness_validation_metrics"
        ],
        "output_metrics": candidates[selected_name]["output_metrics"],
    }
    return PhoenixEventSummaryHybridSurrogate(models[selected_name]), report
