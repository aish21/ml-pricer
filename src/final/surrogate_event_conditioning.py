from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .surrogate_contract import (
    PHOENIX_EVENT_TARGET_NAMES,
    PHOENIX_PAYOFF_AWARE_TARGET_NAMES,
    PHOENIX_PRICE_COMPONENT_NAMES,
)
from .surrogate_model import NumpyMLPSurrogate, SurrogateModelError


PHOENIX_EVENT_CONDITIONED_RESEARCH_VERSION = "phoenix-event-conditioned-research-v1"
PHOENIX_EVENT_CONDITIONED_OUTPUT_NAMES = (
    "coupon_pv",
    "autocall_probability",
    "protected_maturity_probability",
    "downside_probability",
    "autocall_conditional_principal_pv",
    "protected_maturity_conditional_principal_pv",
    "downside_conditional_principal_pv",
)


class EventConditioningError(ValueError):
    pass


def _as_output_matrix(
    values: Sequence[float] | np.ndarray,
    *,
    width: int,
    name: str,
) -> tuple[np.ndarray, bool]:
    matrix = np.asarray(values, dtype=np.float64)
    single = matrix.ndim == 1
    if single:
        matrix = matrix.reshape(1, -1)
    if matrix.ndim != 2 or matrix.shape[1] != width:
        raise EventConditioningError(f"{name} has invalid shape")
    if not np.all(np.isfinite(matrix)):
        raise EventConditioningError(f"{name} must be finite")
    return matrix, single


def decompose_event_conditioned_targets(
    payoff_aware_targets: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Convert unconditional Phoenix labels into event-conditioned targets.

    The terminal branches are mutually exclusive and exhaustive:
    autocall, protected maturity, or downside maturity. Dividing each
    unconditional branch PV by its event probability gives the corresponding
    conditional value. A branch with zero probability receives a zero
    conditional target because its contribution to price is exactly zero.
    """
    targets, single = _as_output_matrix(
        payoff_aware_targets,
        width=len(PHOENIX_PAYOFF_AWARE_TARGET_NAMES),
        name="payoff-aware targets",
    )
    component_count = len(PHOENIX_PRICE_COMPONENT_NAMES)
    components = targets[:, :component_count]
    event_probabilities = targets[
        :,
        component_count : component_count + len(PHOENIX_EVENT_TARGET_NAMES),
    ]
    if np.any(components < 0.0) or np.any(
        (event_probabilities < 0.0) | (event_probabilities > 1.0)
    ):
        raise EventConditioningError("payoff-aware targets violate their bounds")

    autocall_probability = event_probabilities[:, 0]
    downside_probability = event_probabilities[:, 1]
    protected_probability = 1.0 - autocall_probability - downside_probability
    if np.any(protected_probability < -1e-12):
        raise EventConditioningError(
            "terminal event probabilities exceed one in aggregate"
        )
    protected_probability = np.maximum(protected_probability, 0.0)

    def conditional_value(component: np.ndarray, probability: np.ndarray) -> np.ndarray:
        inconsistent = (probability == 0.0) & (component > 1e-12)
        if np.any(inconsistent):
            raise EventConditioningError(
                "a zero-probability event has a non-zero cashflow label"
            )
        return np.divide(
            component,
            probability,
            out=np.zeros_like(component),
            where=probability > 0.0,
        )

    output = np.column_stack(
        [
            components[:, 0],
            autocall_probability,
            protected_probability,
            downside_probability,
            conditional_value(components[:, 1], autocall_probability),
            conditional_value(components[:, 2], protected_probability),
            conditional_value(components[:, 3], downside_probability),
        ]
    )
    return output[0] if single else output


def _project_probability_simplex(values: np.ndarray) -> np.ndarray:
    """Return the Euclidean projection of each row onto the probability simplex."""
    ordered = np.sort(values, axis=1)[:, ::-1]
    cumulative = np.cumsum(ordered, axis=1) - 1.0
    divisors = np.arange(1, values.shape[1] + 1, dtype=np.float64)
    positive = ordered - cumulative / divisors > 0.0
    active_count = np.maximum(np.sum(positive, axis=1), 1)
    row_indices = np.arange(len(values))
    threshold = cumulative[row_indices, active_count - 1] / active_count
    return np.maximum(values - threshold[:, None], 0.0)


def project_event_conditioned_outputs(
    outputs: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Enforce non-negative values and exhaustive terminal probabilities."""
    values, single = _as_output_matrix(
        outputs,
        width=len(PHOENIX_EVENT_CONDITIONED_OUTPUT_NAMES),
        name="event-conditioned outputs",
    )
    projected = values.copy()
    probability_indices = (1, 2, 3)
    projected[:, probability_indices] = _project_probability_simplex(
        projected[:, probability_indices]
    )
    value_indices = (0, 4, 5, 6)
    projected[:, value_indices] = np.maximum(projected[:, value_indices], 0.0)
    return projected[0] if single else projected


def reconstruct_event_conditioned_price(
    outputs: Sequence[float] | np.ndarray,
    *,
    project: bool = False,
) -> np.ndarray:
    """Recombine coupon PV and probability-weighted terminal branch values."""
    values = (
        project_event_conditioned_outputs(outputs)
        if project
        else np.asarray(outputs, dtype=np.float64)
    )
    matrix, single = _as_output_matrix(
        values,
        width=len(PHOENIX_EVENT_CONDITIONED_OUTPUT_NAMES),
        name="event-conditioned outputs",
    )
    price = matrix[:, 0] + np.sum(matrix[:, 1:4] * matrix[:, 4:7], axis=1)
    return price[0] if single else price


@dataclass(frozen=True)
class EventConditionedResearchSurrogate:
    """Offline-only probability network plus conditional-value expert networks."""

    event_network: NumpyMLPSurrogate
    branch_networks: tuple[NumpyMLPSurrogate, ...]

    def __post_init__(self) -> None:
        if (
            self.event_network.output_names
            != PHOENIX_EVENT_CONDITIONED_OUTPUT_NAMES[:4]
        ):
            raise SurrogateModelError("event network output contract is incompatible")
        expected_branch_names = PHOENIX_EVENT_CONDITIONED_OUTPUT_NAMES[4:]
        if len(self.branch_networks) != len(expected_branch_names) or any(
            network.output_names != (name,)
            for network, name in zip(self.branch_networks, expected_branch_names)
        ):
            raise SurrogateModelError("branch network output contract is incompatible")

    def predict_raw_outputs(self, features: Sequence[float] | np.ndarray) -> np.ndarray:
        event_outputs = self.event_network.predict_raw_outputs(features)
        branch_outputs = [
            network.predict_raw_outputs(features) for network in self.branch_networks
        ]
        return np.column_stack([event_outputs, *branch_outputs])

    def predict_outputs(self, features: Sequence[float] | np.ndarray) -> np.ndarray:
        return project_event_conditioned_outputs(self.predict_raw_outputs(features))

    def predict(self, features: Sequence[float] | np.ndarray) -> np.ndarray:
        return reconstruct_event_conditioned_price(self.predict_outputs(features))
