import math
import time
from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np

from .surrogate_contract import (
    PHOENIX_EVENT_TARGET_NAMES,
    PHOENIX_PAYOFF_AWARE_TARGET_NAMES,
    PHOENIX_PRICE_COMPONENT_NAMES,
    PHOENIX_SURROGATE_FEATURE_NAMES,
)
from .surrogate_hazard_data import PhoenixHazardDataset
from .surrogate_hybrid import summarize_phoenix_observation_events
from .surrogate_price_first_contract import (
    PHOENIX_PRICE_FIRST_AUDIT_VERSION,
    PHOENIX_PRICE_FIRST_EVENT_TARGET_NAMES,
    PHOENIX_PRICE_FIRST_FROZEN_AUXILIARY_WEIGHT,
    PHOENIX_PRICE_FIRST_FROZEN_SPECIFICATION_COMMIT,
    PHOENIX_PRICE_FIRST_OUTPUT_NAMES,
    PHOENIX_PRICE_FIRST_RESEARCH_VERSION,
)
from .surrogate_trainer import (
    PhoenixSurrogateTrainingConfig,
    SurrogateTrainingError,
    _acceptance,
    _audit_uncertainty_policy,
    _greek_validation,
    _joint_regime_moneyness_metrics,
    _moneyness_metrics,
    _regime_metrics,
    _regression_metrics,
    _repeated_group_validation,
    _uncertainty_aware_metrics,
    _validate_training_dataset,
)


PHOENIX_PRICE_FIRST_FROZEN_DEVELOPMENT_METRICS = {
    "validation_mae": 0.006644006053484162,
    "repeated_selection_score": 0.018586225678601333,
    "development_test_mae": 0.006971920441342174,
}


@dataclass(frozen=True)
class PhoenixPriceFirstTrainingConfig:
    hidden_layer_sizes: tuple[int, ...] = (256, 128, 64)
    auxiliary_head_width: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    batch_size: int = 256
    epochs: int = 200
    model_random_state: int = 143
    validation_random_state: int = 42
    internal_selection_folds: int = 3
    auxiliary_loss_weights: tuple[float, ...] = (0.0, 0.03, 0.1)
    payoff_loss_weight: float = 0.25
    selection_worst_regime_weight: float = 0.35
    selection_worst_moneyness_weight: float = 0.25
    selection_worst_joint_cell_weight: float = 0.25
    selection_validation_folds: int = 5
    selection_validation_repeats: int = 3
    selection_worst_fold_weight: float = 0.25
    focused_head_regime_weight: float = 2.0
    focused_head_coupon_weight: float = 2.0
    focused_head_ridge: float = 0.001
    torch_threads: int = 1

    @property
    def random_state(self) -> int:
        """Compatibility seed for the established repeated-validation helper."""
        return self.validation_random_state

    def __post_init__(self) -> None:
        if not self.hidden_layer_sizes or any(
            width < 1 or width > 2_048 for width in self.hidden_layer_sizes
        ):
            raise SurrogateTrainingError("price-first hidden_layer_sizes are invalid")
        if self.auxiliary_head_width < 1 or self.auxiliary_head_width > 1_024:
            raise SurrogateTrainingError("price-first auxiliary_head_width is invalid")
        if (
            not math.isfinite(self.learning_rate)
            or self.learning_rate <= 0.0
            or not math.isfinite(self.weight_decay)
            or self.weight_decay < 0.0
        ):
            raise SurrogateTrainingError("price-first optimizer settings are invalid")
        if self.batch_size < 8 or self.epochs < 1:
            raise SurrogateTrainingError("price-first training budget is invalid")
        if (
            self.model_random_state < 0
            or self.validation_random_state < 0
            or self.model_random_state > 1_000_000
            or self.validation_random_state > 1_000_000
        ):
            raise SurrogateTrainingError("price-first random state is invalid")
        if self.internal_selection_folds < 2 or self.internal_selection_folds > 5:
            raise SurrogateTrainingError("price-first internal fold count is invalid")
        if (
            not self.auxiliary_loss_weights
            or len(self.auxiliary_loss_weights) > 6
            or len(set(self.auxiliary_loss_weights)) != len(self.auxiliary_loss_weights)
            or any(
                not math.isfinite(weight) or weight < 0.0 or weight > 1.0
                for weight in self.auxiliary_loss_weights
            )
        ):
            raise SurrogateTrainingError(
                "price-first auxiliary loss weights are invalid"
            )
        if (
            not math.isfinite(self.payoff_loss_weight)
            or self.payoff_loss_weight < 0.0
            or self.payoff_loss_weight > 1.0
        ):
            raise SurrogateTrainingError("price-first payoff loss weight is invalid")
        if (
            self.selection_validation_folds < 2
            or self.selection_validation_folds > 10
            or self.selection_validation_repeats < 1
            or self.selection_validation_repeats > 10
        ):
            raise SurrogateTrainingError(
                "price-first validation fold settings are invalid"
            )
        if any(
            not math.isfinite(value) or value < 0.0
            for value in (
                self.selection_worst_regime_weight,
                self.selection_worst_moneyness_weight,
                self.selection_worst_joint_cell_weight,
                self.selection_worst_fold_weight,
            )
        ):
            raise SurrogateTrainingError("price-first selection weights are invalid")
        if (
            not math.isfinite(self.focused_head_regime_weight)
            or self.focused_head_regime_weight < 1.0
            or not math.isfinite(self.focused_head_coupon_weight)
            or self.focused_head_coupon_weight < 1.0
            or not math.isfinite(self.focused_head_ridge)
            or self.focused_head_ridge <= 0.0
        ):
            raise SurrogateTrainingError(
                "price-first focused head settings are invalid"
            )
        if self.torch_threads < 1 or self.torch_threads > 32:
            raise SurrogateTrainingError("price-first torch_threads is invalid")


def price_first_event_targets(
    dataset: PhoenixHazardDataset,
) -> tuple[np.ndarray, np.ndarray]:
    """Return non-redundant event targets and their case-level loss weights."""
    summaries = summarize_phoenix_observation_events(dataset)
    targets = summaries[:, (0, 1, 2, 4, 5)]
    autocall_index = PHOENIX_PAYOFF_AWARE_TARGET_NAMES.index("autocall_probability")
    autocall_probability = dataset.base.auxiliary_targets[:, autocall_index]
    loss_weights = np.ones_like(targets)
    loss_weights[:, :2] = autocall_probability[:, None]
    if not np.all(np.isfinite(targets)) or not np.all(np.isfinite(loss_weights)):
        raise SurrogateTrainingError(
            "price-first event targets and weights must be finite"
        )
    if np.any((loss_weights < 0.0) | (loss_weights > 1.0)):
        raise SurrogateTrainingError("price-first event loss weights violate bounds")
    return targets, loss_weights


def _import_torch():
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise SurrogateTrainingError(
            "price-first research training requires the 'research' extra"
        ) from exc
    return torch, nn


def _build_network(
    *,
    torch,
    nn,
    feature_count: int,
    hidden_layer_sizes: tuple[int, ...],
    auxiliary_head_width: int,
):
    class PriceFirstNetwork(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            layers = []
            previous_width = feature_count
            for width in hidden_layer_sizes:
                layers.extend([nn.Linear(previous_width, width), nn.ReLU()])
                previous_width = width
            self.trunk = nn.Sequential(*layers)
            self.price_head = nn.Linear(previous_width, 1)
            self.payoff_head = nn.Sequential(
                nn.Linear(previous_width, auxiliary_head_width),
                nn.ReLU(),
                nn.Linear(
                    auxiliary_head_width,
                    len(PHOENIX_PAYOFF_AWARE_TARGET_NAMES),
                ),
            )
            self.event_head = nn.Sequential(
                nn.Linear(previous_width, auxiliary_head_width),
                nn.ReLU(),
                nn.Linear(
                    auxiliary_head_width,
                    len(PHOENIX_PRICE_FIRST_EVENT_TARGET_NAMES),
                ),
            )

        def forward(self, features):
            hidden = self.trunk(features)
            return (
                self.price_head(hidden),
                self.payoff_head(hidden),
                self.event_head(hidden),
            )

    return PriceFirstNetwork()


@dataclass(frozen=True)
class PhoenixPriceFirstSurrogate:
    network: Any
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    target_mean: np.ndarray
    target_scale: np.ndarray

    @property
    def output_names(self) -> tuple[str, ...]:
        return PHOENIX_PRICE_FIRST_OUTPUT_NAMES

    def predict_raw_outputs(
        self,
        features: Sequence[float] | np.ndarray,
    ) -> np.ndarray:
        torch, _ = _import_torch()
        values = np.asarray(features, dtype=np.float64)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        if values.ndim != 2 or values.shape[1] != len(PHOENIX_SURROGATE_FEATURE_NAMES):
            raise SurrogateTrainingError(
                "price-first prediction features have invalid shape"
            )
        if not np.all(np.isfinite(values)):
            raise SurrogateTrainingError(
                "price-first prediction features must be finite"
            )
        standardized = ((values - self.feature_mean) / self.feature_scale).astype(
            np.float32
        )
        self.network.eval()
        with torch.no_grad():
            price, payoff, events = self.network(torch.from_numpy(standardized))
        normalized = np.column_stack(
            [
                price.detach().cpu().numpy(),
                payoff.detach().cpu().numpy(),
                events.detach().cpu().numpy(),
            ]
        )
        return normalized * self.target_scale + self.target_mean

    def predict_outputs(
        self,
        features: Sequence[float] | np.ndarray,
    ) -> np.ndarray:
        outputs = self.predict_raw_outputs(features)
        for name in (
            "coupon_pv",
            "autocall_principal_pv",
            "maturity_protected_pv",
            "maturity_downside_pv",
            "expected_coupon_count",
            "early_coupon_mass",
            "late_coupon_mass",
        ):
            if name in PHOENIX_PRICE_FIRST_OUTPUT_NAMES:
                index = PHOENIX_PRICE_FIRST_OUTPUT_NAMES.index(name)
                outputs[:, index] = np.maximum(outputs[:, index], 0.0)
        for name in (
            "autocall_probability",
            "downside_probability",
            "conditional_expected_autocall_time_fraction",
            "conditional_autocall_time_variance",
            "final_survival_probability",
        ):
            index = PHOENIX_PRICE_FIRST_OUTPUT_NAMES.index(name)
            outputs[:, index] = np.clip(outputs[:, index], 0.0, 1.0)
        return outputs

    def predict(
        self,
        features: Sequence[float] | np.ndarray,
    ) -> np.ndarray:
        return self.predict_outputs(features)[:, 0]


def _standardization(
    dataset: PhoenixHazardDataset,
    *,
    fit_mask: np.ndarray,
    targets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    base = dataset.base
    feature_mean = np.mean(base.X[fit_mask], axis=0)
    feature_scale = np.std(base.X[fit_mask], axis=0)
    feature_scale[feature_scale < 1e-12] = 1.0
    target_mean = np.mean(targets[fit_mask], axis=0)
    target_scale = np.std(targets[fit_mask], axis=0)
    target_scale[target_scale < 1e-12] = 1.0
    return feature_mean, feature_scale, target_mean, target_scale


def _fit_network(
    *,
    dataset: PhoenixHazardDataset,
    targets: np.ndarray,
    event_loss_weights: np.ndarray,
    fit_mask: np.ndarray,
    auxiliary_loss_weight: float,
    config: PhoenixPriceFirstTrainingConfig,
    random_state: int,
) -> tuple[PhoenixPriceFirstSurrogate, dict[str, Any]]:
    torch, nn = _import_torch()
    (
        feature_mean,
        feature_scale,
        target_mean,
        target_scale,
    ) = _standardization(dataset, fit_mask=fit_mask, targets=targets)
    standardized_features = ((dataset.base.X - feature_mean) / feature_scale).astype(
        np.float32
    )
    standardized_targets = ((targets - target_mean) / target_scale).astype(np.float32)
    event_loss_weights = event_loss_weights.astype(np.float32)
    fit_indices = np.flatnonzero(fit_mask)
    if len(fit_indices) < config.batch_size:
        batch_size = max(8, min(config.batch_size, len(fit_indices)))
    else:
        batch_size = config.batch_size

    previous_threads = torch.get_num_threads()
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    started = time.perf_counter()
    try:
        torch.set_num_threads(config.torch_threads)
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(random_state)
        network = _build_network(
            torch=torch,
            nn=nn,
            feature_count=standardized_features.shape[1],
            hidden_layer_sizes=config.hidden_layer_sizes,
            auxiliary_head_width=config.auxiliary_head_width,
        )
        optimizer = torch.optim.AdamW(
            network.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        rng = np.random.RandomState(random_state)
        last_losses = {}
        for _epoch in range(config.epochs):
            shuffled = rng.permutation(fit_indices)
            epoch_totals = {
                "total": 0.0,
                "price": 0.0,
                "payoff": 0.0,
                "event": 0.0,
            }
            observations = 0
            network.train()
            for start in range(0, len(shuffled), batch_size):
                batch = shuffled[start : start + batch_size]
                features = torch.from_numpy(standardized_features[batch])
                observed = torch.from_numpy(standardized_targets[batch])
                observed_event_weights = torch.from_numpy(event_loss_weights[batch])
                predicted_price, predicted_payoff, predicted_event = network(features)
                price_loss = torch.mean((predicted_price - observed[:, :1]) ** 2)
                payoff_end = 1 + len(PHOENIX_PAYOFF_AWARE_TARGET_NAMES)
                payoff_loss = torch.mean(
                    (predicted_payoff - observed[:, 1:payoff_end]) ** 2
                )
                event_squared_error = (predicted_event - observed[:, payoff_end:]) ** 2
                per_target_event_loss = torch.sum(
                    event_squared_error * observed_event_weights,
                    dim=0,
                ) / torch.clamp(
                    torch.sum(observed_event_weights, dim=0),
                    min=1e-12,
                )
                event_loss = torch.mean(per_target_event_loss)
                loss = (
                    price_loss
                    + config.payoff_loss_weight * payoff_loss
                    + auxiliary_loss_weight * event_loss
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=5.0)
                optimizer.step()
                observed_count = len(batch)
                observations += observed_count
                for name, value in (
                    ("total", loss),
                    ("price", price_loss),
                    ("payoff", payoff_loss),
                    ("event", event_loss),
                ):
                    epoch_totals[name] += float(value.detach()) * observed_count
            last_losses = {
                name: total / observations for name, total in epoch_totals.items()
            }
    finally:
        torch.set_num_threads(previous_threads)
        torch.use_deterministic_algorithms(previous_deterministic)

    surrogate = PhoenixPriceFirstSurrogate(
        network=network,
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        target_mean=target_mean,
        target_scale=target_scale,
    )
    return surrogate, {
        "fit_seconds": time.perf_counter() - started,
        "epochs": config.epochs,
        "batch_size": batch_size,
        "random_state": random_state,
        "final_epoch_losses": last_losses,
    }


def _refit_focused_price_head(
    *,
    dataset: PhoenixHazardDataset,
    surrogate: PhoenixPriceFirstSurrogate,
    fit_mask: np.ndarray,
    config: PhoenixPriceFirstTrainingConfig,
) -> None:
    torch, _ = _import_torch()
    standardized_features = (
        (dataset.base.X[fit_mask] - surrogate.feature_mean) / surrogate.feature_scale
    ).astype(np.float32)
    surrogate.network.eval()
    with torch.no_grad():
        hidden = (
            surrogate.network.trunk(torch.from_numpy(standardized_features))
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )
    standardized_price = (
        dataset.base.y[fit_mask] - surrogate.target_mean[0]
    ) / surrogate.target_scale[0]
    sample_weights = np.ones(int(np.sum(fit_mask)), dtype=np.float64)
    sample_weights *= np.where(
        dataset.base.regime_names[fit_mask] == "low_vol",
        config.focused_head_regime_weight,
        1.0,
    )
    sample_weights *= np.where(
        dataset.base.moneyness_region_names[fit_mask] == "coupon",
        config.focused_head_coupon_weight,
        1.0,
    )
    design = np.column_stack([hidden, np.ones(len(hidden), dtype=np.float64)])
    weighted_design = design * np.sqrt(sample_weights)[:, None]
    weighted_target = standardized_price * np.sqrt(sample_weights)
    normalizer = float(np.sum(sample_weights))
    gram = weighted_design.T @ weighted_design / normalizer
    penalty = np.eye(gram.shape[0], dtype=np.float64) * config.focused_head_ridge
    penalty[-1, -1] = 0.0
    coefficients = np.linalg.solve(
        gram + penalty,
        weighted_design.T @ weighted_target / normalizer,
    )
    with torch.no_grad():
        surrogate.network.price_head.weight.copy_(
            torch.from_numpy(coefficients[:-1].astype(np.float32)[None, :])
        )
        surrogate.network.price_head.bias.copy_(
            torch.from_numpy(coefficients[-1:].astype(np.float32))
        )


def _masked_price_score(
    dataset: PhoenixHazardDataset,
    predictions: np.ndarray,
    mask: np.ndarray,
    config: PhoenixPriceFirstTrainingConfig,
) -> dict[str, float]:
    actual = dataset.base.y
    base_mae = float(np.mean(np.abs(predictions[mask] - actual[mask])))

    def worst_mae(labels: np.ndarray) -> float:
        return max(
            float(
                np.mean(
                    np.abs(
                        predictions[mask & (labels == label)]
                        - actual[mask & (labels == label)]
                    )
                )
            )
            for label in np.unique(labels[mask])
        )

    joint_labels = np.asarray(
        [
            f"{regime}:{region}"
            for regime, region in zip(
                dataset.base.regime_names,
                dataset.base.moneyness_region_names,
            )
        ]
    )
    worst_regime = worst_mae(dataset.base.regime_names)
    worst_moneyness = worst_mae(dataset.base.moneyness_region_names)
    worst_joint = worst_mae(joint_labels)
    return {
        "mae": base_mae,
        "maximum_regime_mae": worst_regime,
        "maximum_moneyness_mae": worst_moneyness,
        "maximum_regime_moneyness_mae": worst_joint,
        "score": (
            base_mae
            + config.selection_worst_regime_weight * worst_regime
            + config.selection_worst_moneyness_weight * worst_moneyness
            + config.selection_worst_joint_cell_weight * worst_joint
        ),
    }


def _select_auxiliary_weight(
    *,
    dataset: PhoenixHazardDataset,
    targets: np.ndarray,
    event_loss_weights: np.ndarray,
    config: PhoenixPriceFirstTrainingConfig,
    verbose: bool,
) -> tuple[float, dict[str, Any]]:
    training_mask = dataset.base.split_names == "train"
    training_groups = np.unique(dataset.base.group_ids[training_mask])
    if len(training_groups) < config.internal_selection_folds:
        raise SurrogateTrainingError(
            "too few training groups for price-first internal selection"
        )
    rng = np.random.RandomState(config.validation_random_state + 91_117)
    folds = np.array_split(
        rng.permutation(training_groups),
        config.internal_selection_folds,
    )
    candidate_reports = {}
    for auxiliary_weight in config.auxiliary_loss_weights:
        fold_metrics = []
        for fold_index, held_out_groups in enumerate(folds):
            held_out_mask = training_mask & np.isin(
                dataset.base.group_ids,
                held_out_groups,
            )
            fit_mask = training_mask & ~held_out_mask
            surrogate, fit = _fit_network(
                dataset=dataset,
                targets=targets,
                event_loss_weights=event_loss_weights,
                fit_mask=fit_mask,
                auxiliary_loss_weight=auxiliary_weight,
                config=config,
                random_state=config.model_random_state + fold_index * 101,
            )
            _refit_focused_price_head(
                dataset=dataset,
                surrogate=surrogate,
                fit_mask=fit_mask,
                config=config,
            )
            predictions = surrogate.predict(dataset.base.X)
            metrics = _masked_price_score(
                dataset,
                predictions,
                held_out_mask,
                config,
            )
            fold_metrics.append(
                {
                    "fold": fold_index,
                    "n_groups": int(len(held_out_groups)),
                    "n_samples": int(np.sum(held_out_mask)),
                    "fit_seconds": fit["fit_seconds"],
                    **metrics,
                }
            )
            if verbose:
                print(
                    "[PhoenixPriceFirst] "
                    f"aux={auxiliary_weight:g} fold={fold_index + 1}/"
                    f"{config.internal_selection_folds} score={metrics['score']:.6f}",
                    flush=True,
                )
        scores = np.asarray([fold["score"] for fold in fold_metrics])
        candidate_reports[f"{auxiliary_weight:g}"] = {
            "auxiliary_loss_weight": auxiliary_weight,
            "mean_score": float(np.mean(scores)),
            "worst_score": float(np.max(scores)),
            "selection_score": float(
                np.mean(scores) + config.selection_worst_fold_weight * np.max(scores)
            ),
            "fold_metrics": fold_metrics,
        }
    selected_key = min(
        candidate_reports,
        key=lambda key: (
            candidate_reports[key]["selection_score"],
            candidate_reports[key]["mean_score"],
            float(key),
        ),
    )
    return candidate_reports[selected_key]["auxiliary_loss_weight"], {
        "policy": "training-only-group-fold-auxiliary-weight-selection-v1",
        "split": "train",
        "folds": config.internal_selection_folds,
        "validation_or_test_rows_used": False,
        "candidates": candidate_reports,
        "selected_auxiliary_loss_weight": candidate_reports[selected_key][
            "auxiliary_loss_weight"
        ],
    }


def _plain_regression_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    *,
    weights: np.ndarray | None = None,
) -> dict[str, Any]:
    active = np.ones(len(actual), dtype=bool)
    if weights is not None:
        active = weights > 0.0
    errors = predicted[active] - actual[active]
    absolute = np.abs(errors)
    return {
        "n_samples": int(np.sum(active)),
        "mae": float(np.mean(absolute)),
        "rmse": float(math.sqrt(np.mean(errors**2))),
        "p95_absolute_error": float(np.quantile(absolute, 0.95)),
        "mean_error": float(np.mean(errors)),
    }


def train_phoenix_price_first_candidate(
    dataset: PhoenixHazardDataset,
    config: PhoenixPriceFirstTrainingConfig = PhoenixPriceFirstTrainingConfig(),
    *,
    verbose: bool = True,
) -> tuple[PhoenixPriceFirstSurrogate, dict[str, Any]]:
    """Fit a price-dominant, masked multi-task candidate on development data."""
    base = dataset.base
    if base.metadata.get("dataset_role") != "development":
        raise SurrogateTrainingError(
            "price-first candidate requires a development dataset"
        )
    event_targets, event_loss_weights = price_first_event_targets(dataset)
    targets = np.column_stack([base.y, base.auxiliary_targets, event_targets])
    selected_weight, internal_selection = _select_auxiliary_weight(
        dataset=dataset,
        targets=targets,
        event_loss_weights=event_loss_weights,
        config=config,
        verbose=verbose,
    )
    training_mask = base.split_names == "train"
    surrogate, fit = _fit_network(
        dataset=dataset,
        targets=targets,
        event_loss_weights=event_loss_weights,
        fit_mask=training_mask,
        auxiliary_loss_weight=selected_weight,
        config=config,
        random_state=config.model_random_state,
    )
    _refit_focused_price_head(
        dataset=dataset,
        surrogate=surrogate,
        fit_mask=training_mask,
        config=config,
    )
    predictions = surrogate.predict(base.X)
    output_predictions = surrogate.predict_outputs(base.X)
    split_metrics = {}
    output_metrics = {}
    payoff_end = 1 + len(PHOENIX_PAYOFF_AWARE_TARGET_NAMES)
    for split in ("train", "validation", "test"):
        mask = base.split_names == split
        split_metrics[split] = _regression_metrics(
            base.y[mask],
            predictions[mask],
            base.label_standard_error[mask],
        )
        output_metrics[split] = {}
        for index, name in enumerate(PHOENIX_PRICE_FIRST_OUTPUT_NAMES):
            weights = None
            if index >= payoff_end:
                weights = event_loss_weights[mask, index - payoff_end]
            output_metrics[split][name] = _plain_regression_metrics(
                targets[mask, index],
                output_predictions[mask, index],
                weights=weights,
            )

    regime_metrics = _regime_metrics(base, predictions)
    moneyness_metrics = _moneyness_metrics(base, predictions)
    joint_metrics = _joint_regime_moneyness_metrics(
        base,
        predictions,
        split="validation",
    )
    repeated_validation = _repeated_group_validation(base, predictions, config)
    validation = split_metrics["validation"]
    maximum_regime_mae = max(
        values["mae"] for values in regime_metrics["validation"].values()
    )
    maximum_moneyness_mae = max(
        values["mae"] for values in moneyness_metrics["validation"].values()
    )
    maximum_joint_mae = max(values["mae"] for values in joint_metrics.values())
    report = {
        "research_version": PHOENIX_PRICE_FIRST_RESEARCH_VERSION,
        "candidate_name": "price_first_masked_multitask__focused_head",
        "model_type": "pytorch-shared-trunk-separate-heads-research",
        "dataset_id": dataset.metadata["dataset_id"],
        "base_dataset_id": base.metadata["dataset_id"],
        "runtime_eligible": False,
        "audit_evaluated": False,
        "deployment_status": "research_only",
        "exclusion_reason": (
            "development-only price-first multi-task experiment; not part of "
            "the production artifact contract"
        ),
        "training_split": "train",
        "selection_split": "validation",
        "output_names": list(PHOENIX_PRICE_FIRST_OUTPUT_NAMES),
        "event_target_names": list(PHOENIX_PRICE_FIRST_EVENT_TARGET_NAMES),
        "price_inference": "independent focused direct-price head",
        "loss_policy": "price-dominant-masked-multi-head-squared-error-v1",
        "masked_event_targets": list(PHOENIX_PRICE_FIRST_EVENT_TARGET_NAMES[:2]),
        "redundant_coupon_count_target_removed": True,
        "config": {
            "hidden_layer_sizes": list(config.hidden_layer_sizes),
            "auxiliary_head_width": config.auxiliary_head_width,
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
            "batch_size": config.batch_size,
            "epochs": config.epochs,
            "model_random_state": config.model_random_state,
            "validation_random_state": config.validation_random_state,
            "payoff_loss_weight": config.payoff_loss_weight,
            "auxiliary_loss_weights": list(config.auxiliary_loss_weights),
        },
        "internal_auxiliary_weight_selection": internal_selection,
        "selected_auxiliary_loss_weight": selected_weight,
        "fit": {
            **fit,
            "focused_price_head": {
                "regime": "low_vol",
                "regime_weight": config.focused_head_regime_weight,
                "moneyness_region": "coupon",
                "moneyness_weight": config.focused_head_coupon_weight,
                "ridge": config.focused_head_ridge,
            },
        },
        "split_metrics": split_metrics,
        "regime_metrics": regime_metrics,
        "moneyness_metrics": moneyness_metrics,
        "regime_moneyness_validation_metrics": joint_metrics,
        "output_metrics": output_metrics,
        "selection": {
            "policy": "price-first-development-comparison-v1",
            "validation_mae": validation["mae"],
            "maximum_validation_regime_mae": maximum_regime_mae,
            "maximum_validation_moneyness_mae": maximum_moneyness_mae,
            "maximum_validation_regime_moneyness_mae": maximum_joint_mae,
            "single_split_score": (
                validation["mae"]
                + config.selection_worst_regime_weight * maximum_regime_mae
                + config.selection_worst_moneyness_weight * maximum_moneyness_mae
                + config.selection_worst_joint_cell_weight * maximum_joint_mae
            ),
            "repeated_group_validation": repeated_validation,
            "score": repeated_validation["selection_score"],
        },
    }
    return surrogate, report


def _fit_frozen_phoenix_price_first_candidate(
    dataset: PhoenixHazardDataset,
) -> tuple[PhoenixPriceFirstSurrogate, dict[str, Any]]:
    config = PhoenixPriceFirstTrainingConfig()
    event_targets, event_loss_weights = price_first_event_targets(dataset)
    targets = np.column_stack(
        [dataset.base.y, dataset.base.auxiliary_targets, event_targets]
    )
    training_mask = dataset.base.split_names == "train"
    surrogate, fit = _fit_network(
        dataset=dataset,
        targets=targets,
        event_loss_weights=event_loss_weights,
        fit_mask=training_mask,
        auxiliary_loss_weight=PHOENIX_PRICE_FIRST_FROZEN_AUXILIARY_WEIGHT,
        config=config,
        random_state=config.model_random_state,
    )
    _refit_focused_price_head(
        dataset=dataset,
        surrogate=surrogate,
        fit_mask=training_mask,
        config=config,
    )
    predictions = surrogate.predict(dataset.base.X)
    validation_mask = dataset.base.split_names == "validation"
    test_mask = dataset.base.split_names == "test"
    observed = {
        "validation_mae": float(
            np.mean(
                np.abs(predictions[validation_mask] - dataset.base.y[validation_mask])
            )
        ),
        "repeated_selection_score": _repeated_group_validation(
            dataset.base,
            predictions,
            config,
        )["selection_score"],
        "development_test_mae": float(
            np.mean(np.abs(predictions[test_mask] - dataset.base.y[test_mask]))
        ),
    }
    mismatches = {
        name: {
            "expected": expected,
            "observed": observed[name],
        }
        for name, expected in PHOENIX_PRICE_FIRST_FROZEN_DEVELOPMENT_METRICS.items()
        if not math.isclose(
            observed[name],
            expected,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    }
    if mismatches:
        raise SurrogateTrainingError(
            "frozen price-first development fingerprint does not reproduce"
        )
    return surrogate, {
        "specification_commit": PHOENIX_PRICE_FIRST_FROZEN_SPECIFICATION_COMMIT,
        "selected_auxiliary_loss_weight": (PHOENIX_PRICE_FIRST_FROZEN_AUXILIARY_WEIGHT),
        "development_metrics": observed,
        "fit": fit,
        "training_config": asdict(config),
    }


def _price_first_audit_metrics(
    *,
    dataset,
    surrogate: PhoenixPriceFirstSurrogate,
    uncertainty_policy: dict[str, Any],
) -> dict[str, Any]:
    predictions = surrogate.predict(dataset.X)
    output_predictions = surrogate.predict_outputs(dataset.X)
    raw_output_predictions = surrogate.predict_raw_outputs(dataset.X)
    price_metrics = _regression_metrics(
        dataset.y,
        predictions,
        dataset.label_standard_error,
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

    audited_output_names = ("price",) + PHOENIX_PAYOFF_AWARE_TARGET_NAMES
    output_targets = np.column_stack([dataset.y, dataset.auxiliary_targets])
    output_standard_error = np.column_stack(
        [dataset.label_standard_error, dataset.auxiliary_standard_error]
    )
    output_metrics = {}
    for name in audited_output_names:
        index = PHOENIX_PRICE_FIRST_OUTPUT_NAMES.index(name)
        target_index = audited_output_names.index(name)
        output_metrics[name] = _regression_metrics(
            output_targets[:, target_index],
            output_predictions[:, index],
            output_standard_error[:, target_index],
        )
    for component_name in PHOENIX_PRICE_COMPONENT_NAMES:
        index = PHOENIX_PRICE_FIRST_OUTPUT_NAMES.index(component_name)
        violations = np.maximum(-raw_output_predictions[:, index], 0.0)
        output_metrics[component_name].update(
            {
                "raw_outside_bounds_fraction": float(np.mean(violations > 0.0)),
                "raw_mean_boundary_violation": float(np.mean(violations)),
                "raw_maximum_boundary_violation": float(np.max(violations)),
            }
        )
    for event_name in PHOENIX_EVENT_TARGET_NAMES:
        index = PHOENIX_PRICE_FIRST_OUTPUT_NAMES.index(event_name)
        raw_values = raw_output_predictions[:, index]
        violations = np.maximum(-raw_values, 0.0) + np.maximum(
            raw_values - 1.0,
            0.0,
        )
        output_metrics[event_name].update(
            {
                "raw_outside_bounds_fraction": float(np.mean(violations > 0.0)),
                "raw_mean_boundary_violation": float(np.mean(violations)),
                "raw_maximum_boundary_violation": float(np.max(violations)),
            }
        )
    component_indices = [
        PHOENIX_PRICE_FIRST_OUTPUT_NAMES.index(name)
        for name in PHOENIX_PRICE_COMPONENT_NAMES
    ]
    component_price = np.sum(
        output_predictions[:, component_indices],
        axis=1,
    )
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
            dataset,
            predictions,
        ),
        "output_metrics": output_metrics,
        "outputs_without_audit_labels": list(PHOENIX_PRICE_FIRST_EVENT_TARGET_NAMES),
    }


def audit_frozen_phoenix_price_first_candidate(
    *,
    development_dataset: PhoenixHazardDataset,
    audit_dataset,
    verbose: bool = True,
) -> dict[str, Any]:
    """Evaluate the frozen development winner once on an independent audit."""
    _validate_training_dataset(
        development_dataset.base,
        expected_role="development",
    )
    _validate_training_dataset(audit_dataset, expected_role="audit")
    if (
        audit_dataset.metadata["dataset_id"]
        == development_dataset.base.metadata["dataset_id"]
    ):
        raise SurrogateTrainingError(
            "price-first development and audit datasets must be independent"
        )
    development_config = development_dataset.base.metadata.get("config", {})
    audit_generation_config = audit_dataset.metadata.get("config", {})
    if any(
        development_config.get(name) == audit_generation_config.get(name)
        for name in ("dataset_seed", "label_seed")
    ):
        raise SurrogateTrainingError(
            "price-first development and audit seeds must be independent"
        )

    audit_config = PhoenixSurrogateTrainingConfig(
        train_lightgbm_baseline=False,
    )
    uncertainty_policy = _audit_uncertainty_policy(
        audit_dataset,
        audit_config,
    )
    started = time.perf_counter()
    surrogate, frozen_model = _fit_frozen_phoenix_price_first_candidate(
        development_dataset
    )
    evaluation = _price_first_audit_metrics(
        dataset=audit_dataset,
        surrogate=surrogate,
        uncertainty_policy=uncertainty_policy,
    )
    greek_validation = _greek_validation(
        surrogate,
        audit_dataset,
        audit_config,
    )
    acceptance = _acceptance(
        evaluation_metrics=evaluation,
        greek_validation=greek_validation,
        config=audit_config,
    )
    report = {
        "audit_version": PHOENIX_PRICE_FIRST_AUDIT_VERSION,
        "model_research_version": PHOENIX_PRICE_FIRST_RESEARCH_VERSION,
        "model_specification_commit": (PHOENIX_PRICE_FIRST_FROZEN_SPECIFICATION_COMMIT),
        "development_dataset_id": (development_dataset.base.metadata["dataset_id"]),
        "observation_dataset_id": development_dataset.metadata["dataset_id"],
        "audit_dataset_id": audit_dataset.metadata["dataset_id"],
        "runtime_eligible": False,
        "artifact_written": False,
        "deployment_status": "research_only",
        "audit_decision": "passed" if acceptance["passed"] else "failed",
        "frozen_model": frozen_model,
        "audit_generation_config": audit_generation_config,
        "audit_uncertainty_policy": uncertainty_policy,
        "audit_gate_config": asdict(audit_config),
        "audit_evaluation": evaluation,
        "greek_validation": greek_validation,
        "acceptance": acceptance,
        "evaluation_seconds": time.perf_counter() - started,
    }
    if verbose:
        print(
            "[PhoenixPriceFirstAudit] "
            f"decision={report['audit_decision']} "
            f"mae={evaluation['price_metrics']['mae']:.6f}",
            flush=True,
        )
    return report
