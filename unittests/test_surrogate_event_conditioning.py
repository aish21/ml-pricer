import numpy as np
import pytest

from src.final.surrogate_event_conditioning import (
    EventConditioningError,
    decompose_event_conditioned_targets,
    project_event_conditioned_outputs,
    reconstruct_event_conditioned_price,
)


def test_event_conditioned_targets_reconstruct_the_original_price():
    payoff_aware_targets = np.asarray(
        [
            [0.08, 0.35, 0.40, 0.09, 0.40, 0.15],
            [0.03, 0.00, 0.85, 0.00, 0.00, 0.00],
            [0.12, 0.72, 0.00, 0.08, 0.80, 0.20],
        ],
        dtype=np.float64,
    )

    conditioned = decompose_event_conditioned_targets(payoff_aware_targets)
    reconstructed = reconstruct_event_conditioned_price(conditioned)

    expected = np.sum(payoff_aware_targets[:, :4], axis=1)
    assert np.allclose(reconstructed, expected, rtol=0.0, atol=1e-15)
    assert np.allclose(np.sum(conditioned[:, 1:4], axis=1), 1.0)
    assert conditioned[1, 4] == 0.0
    assert conditioned[1, 6] == 0.0


def test_event_conditioned_projection_enforces_financial_constraints():
    raw = np.asarray(
        [
            [-0.1, 1.2, -0.2, 0.4, -0.5, 0.9, 0.7],
            [0.2, 0.1, 0.1, 0.1, 0.8, -0.3, 0.6],
        ],
        dtype=np.float64,
    )

    projected = project_event_conditioned_outputs(raw)
    prices = reconstruct_event_conditioned_price(projected)

    assert np.all(projected[:, (0, 4, 5, 6)] >= 0.0)
    assert np.allclose(np.sum(projected[:, 1:4], axis=1), 1.0)
    assert np.all(np.isfinite(prices))
    assert np.all(prices >= 0.0)


def test_event_conditioning_rejects_impossible_terminal_probabilities():
    invalid = np.asarray([0.1, 0.7, 0.0, 0.1, 0.8, 0.3])

    with pytest.raises(
        EventConditioningError,
        match="terminal event probabilities exceed one",
    ):
        decompose_event_conditioned_targets(invalid)
