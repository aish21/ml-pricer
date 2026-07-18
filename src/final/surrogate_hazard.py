import time
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np

from .surrogate_contract import PHOENIX_SURROGATE_FEATURE_NAMES
from .surrogate_hazard_data import (
    PHOENIX_HAZARD_MAX_OBSERVATIONS,
    PhoenixHazardDataset,
    phoenix_observation_schedule,
    reconstruct_hazard_prices,
)
from .surrogate_trainer import (
    _joint_regime_moneyness_metrics,
    _moneyness_metrics,
    _regime_metrics,
    _regression_metrics,
    _repeated_group_validation,
)


PHOENIX_HAZARD_RESEARCH_MODEL_VERSION = "phoenix-observation-hazard-research-v1"


class PhoenixHazardTrainingError(ValueError):
    pass


@dataclass(frozen=True)
class PhoenixHazardTrainingConfig:
    max_iter: int = 800
    learning_rate: float = 0.025
    max_leaf_nodes: int = 15
    min_samples_leaf: int = 10
    l2_regularization: float = 0.001
    random_state: int = 143
    selection_worst_regime_weight: float = 0.35
    selection_worst_moneyness_weight: float = 0.25
    selection_worst_joint_cell_weight: float = 0.25
    selection_validation_folds: int = 5
    selection_validation_repeats: int = 3
    selection_worst_fold_weight: float = 0.25

    def __post_init__(self) -> None:
        if self.max_iter < 10 or self.max_iter > 5_000:
            raise PhoenixHazardTrainingError("hazard max_iter is invalid")
        if not 0.0 < self.learning_rate <= 1.0:
            raise PhoenixHazardTrainingError("hazard learning_rate is invalid")
        if self.max_leaf_nodes < 2 or self.max_leaf_nodes > 255:
            raise PhoenixHazardTrainingError("hazard max_leaf_nodes is invalid")
        if self.min_samples_leaf < 2:
            raise PhoenixHazardTrainingError("hazard min_samples_leaf is invalid")
        if self.l2_regularization < 0.0:
            raise PhoenixHazardTrainingError("hazard l2_regularization is invalid")
        if self.random_state < 0 or self.random_state > 1_000_000:
            raise PhoenixHazardTrainingError("hazard random_state is invalid")
        if (
            self.selection_validation_folds < 2
            or self.selection_validation_folds > 10
            or self.selection_validation_repeats < 1
            or self.selection_validation_repeats > 10
        ):
            raise PhoenixHazardTrainingError(
                "hazard validation fold settings are invalid"
            )


def _observation_rows(
    features: np.ndarray,
    observation_mask: np.ndarray,
    *,
    n_steps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = []
    case_indices = []
    observation_indices = []
    for case_index, (feature_row, mask_row) in enumerate(
        zip(features, observation_mask)
    ):
        observation_count = int(np.sum(mask_row))
        path_indices = np.linspace(0, n_steps, observation_count + 1, dtype=int)[1:]
        for observation_index, path_index in enumerate(path_indices):
            ordinal = (observation_index + 1) / observation_count
            time_fraction = path_index / n_steps
            remaining_fraction = (
                observation_count - observation_index - 1
            ) / observation_count
            rows.append(
                np.concatenate(
                    [
                        feature_row,
                        np.asarray(
                            [ordinal, time_fraction, remaining_fraction],
                            dtype=np.float64,
                        ),
                    ]
                )
            )
            case_indices.append(case_index)
            observation_indices.append(observation_index)
    return (
        np.asarray(rows, dtype=np.float64),
        np.asarray(case_indices, dtype=np.int64),
        np.asarray(observation_indices, dtype=np.int64),
    )


def _fit_soft_binary_model(
    features: np.ndarray,
    probabilities: np.ndarray,
    config: PhoenixHazardTrainingConfig,
    *,
    random_state: int,
):
    from lightgbm import LGBMRegressor

    targets = np.asarray(probabilities, dtype=np.float64)
    if np.any((targets < 0.0) | (targets > 1.0)):
        raise PhoenixHazardTrainingError("soft binary targets violate bounds")
    if float(np.min(targets)) == float(np.max(targets)):
        raise PhoenixHazardTrainingError(
            "soft binary target does not contain probability variation"
        )
    model = LGBMRegressor(
        objective="cross_entropy",
        n_estimators=config.max_iter,
        learning_rate=config.learning_rate,
        num_leaves=config.max_leaf_nodes,
        min_child_samples=config.min_samples_leaf,
        reg_lambda=config.l2_regularization,
        random_state=random_state,
        n_jobs=1,
        verbosity=-1,
    )
    model.fit(features, targets)
    return model


def _soft_log_loss(actual: np.ndarray, predicted: np.ndarray) -> float:
    clipped = np.clip(predicted, 1e-12, 1.0 - 1e-12)
    return float(
        -np.mean(actual * np.log(clipped) + (1.0 - actual) * np.log(1.0 - clipped))
    )


def _predict_lightgbm(model: Any, features: np.ndarray) -> np.ndarray:
    # LightGBM 4.2 with newer scikit-learn releases reports a harmless
    # generated-column-name mismatch for NumPy inputs.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=(
                "X does not have valid feature names, but LGBMRegressor "
                "was fitted with feature names"
            ),
            category=UserWarning,
        )
        return np.asarray(
            model.predict(features, validate_features=False),
            dtype=np.float64,
        )


@dataclass(frozen=True)
class PhoenixObservationHazardSurrogate:
    autocall_hazard_model: Any
    coupon_hazard_model: Any
    terminal_downside_hazard_model: Any
    downside_recovery_model: Any
    n_steps: int

    def predict_event_probabilities(
        self,
        features: np.ndarray,
    ) -> dict[str, np.ndarray]:
        feature_matrix = np.asarray(features, dtype=np.float64)
        single = feature_matrix.ndim == 1
        if single:
            feature_matrix = feature_matrix.reshape(1, -1)
        if feature_matrix.ndim != 2 or feature_matrix.shape[1] != len(
            PHOENIX_SURROGATE_FEATURE_NAMES
        ):
            raise PhoenixHazardTrainingError(
                "hazard prediction features have invalid shape"
            )
        observation_mask, _, _ = phoenix_observation_schedule(
            feature_matrix,
            n_steps=self.n_steps,
        )
        rows, case_indices, observation_indices = _observation_rows(
            feature_matrix,
            observation_mask,
            n_steps=self.n_steps,
        )
        autocall_hazard_rows = np.clip(
            _predict_lightgbm(self.autocall_hazard_model, rows),
            0.0,
            1.0,
        )
        coupon_hazard_rows = np.clip(
            _predict_lightgbm(self.coupon_hazard_model, rows),
            0.0,
            1.0,
        )
        coupon_hazard_rows = np.maximum(coupon_hazard_rows, autocall_hazard_rows)

        shape = (len(feature_matrix), PHOENIX_HAZARD_MAX_OBSERVATIONS)
        autocall_hazard = np.zeros(shape, dtype=np.float64)
        coupon_hazard = np.zeros(shape, dtype=np.float64)
        autocall_hazard[case_indices, observation_indices] = autocall_hazard_rows
        coupon_hazard[case_indices, observation_indices] = coupon_hazard_rows
        first_autocall_probability = np.zeros(shape, dtype=np.float64)
        coupon_probability = np.zeros(shape, dtype=np.float64)
        survival_after_probability = np.zeros(shape, dtype=np.float64)
        survival = np.ones(len(feature_matrix), dtype=np.float64)
        for observation_index in range(PHOENIX_HAZARD_MAX_OBSERVATIONS):
            active = observation_mask[:, observation_index]
            coupon_probability[active, observation_index] = (
                survival[active] * coupon_hazard[active, observation_index]
            )
            first_autocall_probability[active, observation_index] = (
                survival[active] * autocall_hazard[active, observation_index]
            )
            survival[active] -= first_autocall_probability[active, observation_index]
            survival_after_probability[active, observation_index] = survival[active]

        downside_hazard = np.clip(
            _predict_lightgbm(self.terminal_downside_hazard_model, feature_matrix),
            0.0,
            1.0,
        )
        downside_probability = survival * downside_hazard
        protected_probability = survival - downside_probability
        downside_conditional_recovery = np.clip(
            _predict_lightgbm(self.downside_recovery_model, feature_matrix),
            0.0,
            1.0,
        )
        output = {
            "observation_mask": observation_mask,
            "coupon_probability": coupon_probability,
            "first_autocall_probability": first_autocall_probability,
            "survival_after_probability": survival_after_probability,
            "protected_probability": protected_probability,
            "downside_probability": downside_probability,
            "downside_conditional_recovery": downside_conditional_recovery,
            "autocall_hazard": autocall_hazard,
            "coupon_hazard": coupon_hazard,
            "terminal_downside_hazard": downside_hazard,
        }
        if single:
            return {
                name: values[0] if values.ndim > 1 else values[0:1]
                for name, values in output.items()
            }
        return output

    def predict(self, features: np.ndarray) -> np.ndarray:
        feature_matrix = np.asarray(features, dtype=np.float64)
        if feature_matrix.ndim == 1:
            feature_matrix = feature_matrix.reshape(1, -1)
        events = self.predict_event_probabilities(feature_matrix)
        return reconstruct_hazard_prices(
            features=feature_matrix,
            observation_mask=events["observation_mask"],
            coupon_probability=events["coupon_probability"],
            first_autocall_probability=events["first_autocall_probability"],
            protected_probability=events["protected_probability"],
            downside_probability=events["downside_probability"],
            downside_conditional_recovery=events["downside_conditional_recovery"],
            n_steps=self.n_steps,
        )


def train_phoenix_observation_hazard_candidate(
    dataset: PhoenixHazardDataset,
    config: PhoenixHazardTrainingConfig,
) -> tuple[PhoenixObservationHazardSurrogate, dict[str, Any]]:
    """Fit a development-only sequential event-hazard candidate."""
    from lightgbm import LGBMRegressor

    base = dataset.base
    if base.metadata.get("dataset_role") != "development":
        raise PhoenixHazardTrainingError(
            "hazard candidate requires a development dataset"
        )
    train_case_mask = base.split_names == "train"
    n_steps = int(dataset.metadata["n_steps"])
    observation_rows, case_indices, observation_indices = _observation_rows(
        base.X,
        dataset.observation_mask,
        n_steps=n_steps,
    )
    survival_before = np.ones_like(dataset.survival_after_probability)
    survival_before[:, 1:] = dataset.survival_after_probability[:, :-1]
    valid_observation = (
        dataset.observation_mask & (survival_before > 0.0) & train_case_mask[:, None]
    )
    row_training_mask = valid_observation[case_indices, observation_indices]
    selected_cases = case_indices[row_training_mask]
    selected_observations = observation_indices[row_training_mask]
    selected_survival = survival_before[selected_cases, selected_observations]
    autocall_hazard_target = (
        dataset.first_autocall_probability[selected_cases, selected_observations]
        / selected_survival
    )
    coupon_hazard_target = (
        dataset.coupon_probability[selected_cases, selected_observations]
        / selected_survival
    )

    started = time.perf_counter()
    autocall_model = _fit_soft_binary_model(
        observation_rows[row_training_mask],
        autocall_hazard_target,
        config,
        random_state=config.random_state,
    )
    coupon_model = _fit_soft_binary_model(
        observation_rows[row_training_mask],
        coupon_hazard_target,
        config,
        random_state=config.random_state + 101,
    )

    last_observation_indices = np.sum(dataset.observation_mask, axis=1) - 1
    final_survival = dataset.survival_after_probability[
        np.arange(len(base.y)), last_observation_indices
    ]
    terminal_training_mask = train_case_mask & (final_survival > 0.0)
    terminal_downside_hazard = (
        dataset.downside_probability[terminal_training_mask]
        / final_survival[terminal_training_mask]
    )
    terminal_downside_model = _fit_soft_binary_model(
        base.X[terminal_training_mask],
        terminal_downside_hazard,
        config,
        random_state=config.random_state + 202,
    )

    recovery_training_mask = train_case_mask & (dataset.downside_probability > 0.0)
    if int(np.sum(recovery_training_mask)) < config.min_samples_leaf:
        raise PhoenixHazardTrainingError(
            "too few downside cases for recovery regression"
        )
    downside_recovery_model = LGBMRegressor(
        objective="regression_l2",
        n_estimators=config.max_iter,
        learning_rate=config.learning_rate,
        num_leaves=config.max_leaf_nodes,
        min_child_samples=config.min_samples_leaf,
        reg_lambda=config.l2_regularization,
        random_state=config.random_state + 303,
        n_jobs=1,
        verbosity=-1,
    )
    downside_recovery_model.fit(
        base.X[recovery_training_mask],
        dataset.downside_conditional_recovery[recovery_training_mask],
    )
    fit_seconds = time.perf_counter() - started
    surrogate = PhoenixObservationHazardSurrogate(
        autocall_hazard_model=autocall_model,
        coupon_hazard_model=coupon_model,
        terminal_downside_hazard_model=terminal_downside_model,
        downside_recovery_model=downside_recovery_model,
        n_steps=n_steps,
    )

    predictions = surrogate.predict(base.X)
    predicted_events = surrogate.predict_event_probabilities(base.X)
    split_metrics = {}
    for split in ("train", "validation", "test"):
        mask = base.split_names == split
        split_metrics[split] = _regression_metrics(
            base.y[mask],
            predictions[mask],
            base.label_standard_error[mask],
        )

    regime_metrics = _regime_metrics(base, predictions)
    moneyness_metrics = _moneyness_metrics(base, predictions)
    joint_metrics = _joint_regime_moneyness_metrics(
        base, predictions, split="validation"
    )
    validation = base.split_names == "validation"
    validation_observations = dataset.observation_mask & validation[:, None]
    valid_hazard_validation = validation_observations & (survival_before > 0.0)
    actual_autocall_hazard = (
        dataset.first_autocall_probability[valid_hazard_validation]
        / survival_before[valid_hazard_validation]
    )
    actual_coupon_hazard = (
        dataset.coupon_probability[valid_hazard_validation]
        / survival_before[valid_hazard_validation]
    )
    predicted_autocall_hazard = predicted_events["autocall_hazard"][
        valid_hazard_validation
    ]
    predicted_coupon_hazard = predicted_events["coupon_hazard"][valid_hazard_validation]
    validation_terminal = validation & (final_survival > 0.0)
    actual_terminal_hazard = (
        dataset.downside_probability[validation_terminal]
        / final_survival[validation_terminal]
    )
    predicted_terminal_hazard = predicted_events["terminal_downside_hazard"][
        validation_terminal
    ]
    repeated_validation = _repeated_group_validation(base, predictions, config)
    validation_metrics = split_metrics["validation"]
    maximum_regime_mae = max(
        values["mae"] for values in regime_metrics["validation"].values()
    )
    maximum_moneyness_mae = max(
        values["mae"] for values in moneyness_metrics["validation"].values()
    )
    maximum_joint_mae = max(values["mae"] for values in joint_metrics.values())
    target_reconstruction = reconstruct_hazard_prices(
        features=base.X,
        observation_mask=dataset.observation_mask,
        coupon_probability=dataset.coupon_probability,
        first_autocall_probability=dataset.first_autocall_probability,
        protected_probability=dataset.protected_probability,
        downside_probability=dataset.downside_probability,
        downside_conditional_recovery=dataset.downside_conditional_recovery,
        n_steps=n_steps,
    )
    report = {
        "research_version": PHOENIX_HAZARD_RESEARCH_MODEL_VERSION,
        "candidate_name": "observation_hazard_lightgbm",
        "model_type": "sequential-soft-label-lightgbm-hazard-mixture",
        "dataset_id": dataset.metadata["dataset_id"],
        "base_dataset_id": base.metadata["dataset_id"],
        "runtime_eligible": False,
        "audit_evaluated": False,
        "deployment_status": "research_only",
        "exclusion_reason": (
            "development-only observation-hazard experiment; not part of "
            "the production artifact contract"
        ),
        "loss_policy": "soft-binomial-weighted-log-loss-v1",
        "fit_seconds": fit_seconds,
        "n_training_observations": int(np.sum(row_training_mask)),
        "n_terminal_training_cases": int(np.sum(terminal_training_mask)),
        "n_recovery_training_cases": int(np.sum(recovery_training_mask)),
        "target_reconstruction_maximum_error": float(
            np.max(np.abs(target_reconstruction - base.y))
        ),
        "split_metrics": split_metrics,
        "regime_metrics": regime_metrics,
        "moneyness_metrics": moneyness_metrics,
        "regime_moneyness_validation_metrics": joint_metrics,
        "hazard_validation_metrics": {
            "autocall_soft_log_loss": _soft_log_loss(
                actual_autocall_hazard, predicted_autocall_hazard
            ),
            "coupon_soft_log_loss": _soft_log_loss(
                actual_coupon_hazard, predicted_coupon_hazard
            ),
            "terminal_downside_soft_log_loss": _soft_log_loss(
                actual_terminal_hazard, predicted_terminal_hazard
            ),
            "autocall_hazard_mae": float(
                np.mean(np.abs(actual_autocall_hazard - predicted_autocall_hazard))
            ),
            "coupon_hazard_mae": float(
                np.mean(np.abs(actual_coupon_hazard - predicted_coupon_hazard))
            ),
            "terminal_downside_hazard_mae": float(
                np.mean(np.abs(actual_terminal_hazard - predicted_terminal_hazard))
            ),
            "aggregate_autocall_probability_mae": float(
                np.mean(
                    np.abs(
                        np.sum(
                            dataset.first_autocall_probability[validation],
                            axis=1,
                        )
                        - np.sum(
                            predicted_events["first_autocall_probability"][validation],
                            axis=1,
                        )
                    )
                )
            ),
            "downside_probability_mae": float(
                np.mean(
                    np.abs(
                        dataset.downside_probability[validation]
                        - predicted_events["downside_probability"][validation]
                    )
                )
            ),
            "downside_recovery_mae_on_positive_events": float(
                np.mean(
                    np.abs(
                        dataset.downside_conditional_recovery[
                            validation & (dataset.downside_probability > 0.0)
                        ]
                        - predicted_events["downside_conditional_recovery"][
                            validation & (dataset.downside_probability > 0.0)
                        ]
                    )
                )
            ),
        },
        "selection": {
            "policy": "observation-hazard-development-comparison-v1",
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
    return surrogate, report
