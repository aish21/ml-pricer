import numpy as np
import pytest
from numpy.testing import assert_allclose

from src.gbm import simulate_gbm_paths


def test_positive_paths_shape():
    paths, grid = simulate_gbm_paths(
        spot=100,
        risk_free_rate=0.01,
        dividend_yield=0.0,
        volatility=0.2,
        time_to_maturity=1.0,
        num_time_steps=12,
        num_paths=10,
        seed=42,
        antithetic=False,  # disable doubling
    )
    assert paths.shape == (10, 13)


def test_antithetic_doubles_paths():
    paths_no, _ = simulate_gbm_paths(
        100, 0.01, 0.0, 0.2, 1.0, 12, 10, seed=42, antithetic=False
    )
    paths_yes, _ = simulate_gbm_paths(
        100, 0.01, 0.0, 0.2, 1.0, 12, 10, seed=42, antithetic=True
    )
    assert paths_yes.shape[0] == 2 * paths_no.shape[0]


@pytest.mark.parametrize(
    "bad_params",
    [
        {"spot": -100},  # negative spot
        {"volatility": -0.1},  # negative vol
        {"time_to_maturity": 0},  # zero tenor
        {"num_time_steps": 0},  # zero steps
        {"num_paths": 0},  # zero paths
    ],
)
def test_invalid_inputs_raise(bad_params):
    kwargs = dict(
        spot=100,
        risk_free_rate=0.01,
        dividend_yield=0.0,
        volatility=0.2,
        time_to_maturity=1.0,
        num_time_steps=12,
        num_paths=10,
    )
    kwargs.update(bad_params)
    with pytest.raises(ValueError):
        simulate_gbm_paths(**kwargs)  # type: ignore
