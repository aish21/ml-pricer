import pytest

from app.services.scenario_service import (
    InvalidScenarioInputError,
    apply_shocks_to_params,
    normalize_shocks,
    run_scenario,
)
from app.services.product_registry import get_bb_product_definitions


BASE_PARAMS = {
    "S0": 100.0,
    "r": 0.03,
    "sigma": 0.2,
    "T": 1.0,
    "autocall_barrier_frac": 1.05,
    "coupon_barrier_frac": 1.0,
    "coupon_rate": 0.02,
    "knock_in_frac": 0.7,
    "obs_count": 6,
}


def test_scenario_applies_spot_percentage_shock():
    shocked, shocks = apply_shocks_to_params(BASE_PARAMS, {"spot_pct": "-10"})

    assert shocks["spot_pct"] == -10.0
    assert shocked["S0"] == pytest.approx(90.0)


def test_scenario_applies_volatility_absolute_shock():
    shocked, shocks = apply_shocks_to_params(BASE_PARAMS, {"vol_abs": "0.05"})

    assert shocks["vol_abs"] == 0.05
    assert shocked["sigma"] == pytest.approx(0.25)


def test_scenario_applies_rate_basis_point_shock():
    shocked, shocks = apply_shocks_to_params(BASE_PARAMS, {"rate_bps": "50"})

    assert shocks["rate_bps"] == 50.0
    assert shocked["r"] == pytest.approx(0.035)


def test_scenario_allows_negative_rates_within_product_bounds():
    shocked, _ = apply_shocks_to_params(BASE_PARAMS, {"rate_bps": "-400"})

    assert shocked["r"] == pytest.approx(-0.01)


def test_scenario_rejects_no_shocks():
    with pytest.raises(InvalidScenarioInputError):
        normalize_shocks({"spot_pct": "", "vol_abs": "", "rate_bps": ""})


def test_scenario_rejects_shocked_spot_not_positive():
    with pytest.raises(InvalidScenarioInputError):
        apply_shocks_to_params(BASE_PARAMS, {"spot_pct": "-100"})


def test_scenario_rejects_shocked_volatility_not_positive():
    with pytest.raises(InvalidScenarioInputError):
        apply_shocks_to_params(BASE_PARAMS, {"vol_abs": "-0.2"})


def test_scenario_runs_for_each_bb_enabled_product(monkeypatch):
    def fake_price_product(product_key, params, n_paths=500, seed=42):
        return {
            "product_key": product_key,
            "params": params,
            "n_paths": n_paths,
            "price": 90.0,
            "model": "LightGBM surrogate",
            "latency_ms": 1,
        }

    monkeypatch.setattr(
        "app.services.scenario_service.price_product",
        fake_price_product,
    )

    for product in get_bb_product_definitions():
        params = {field.name: field.default for field in product.bb_fields}
        base_request = {
            "product_key": product.key,
            "params": params,
            "n_paths": 5,
            "seed": 42,
            "contract_version": "phoenix-single-v1",
        }
        base_result = {"price": 100.0}

        result = run_scenario(base_request, base_result, {"spot_pct": "-10"})

        assert result["product_key"] == product.key
        assert result["base_price"] == 100.0
        assert result["shocked_price"] == 90.0
        assert result["price_change"] == -10.0
        assert result["shocked_request"]["params"]["S0"] == pytest.approx(90.0)
        assert product.display_name in result["summary"]
