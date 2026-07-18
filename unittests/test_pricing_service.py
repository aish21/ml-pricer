from datetime import datetime, timezone

import pytest

from src.final.market import (
    EquityMarketSegment,
    EquityMarketSnapshot,
    EquityMarketTermStructure,
)
from src.final.phoenix_contract import PhoenixSingleV2Contract
from app.services.model_cache import clear_model_cache
from app.services.pricing_service import (
    InvalidPricingInputError,
    UnsupportedProductError,
    normalize_pricing_params,
    price_phoenix_v2_with_term_structure,
    price_phoenix_with_term_structure,
    price_phoenix_with_market_snapshot,
    price_product,
)
from app.services.product_registry import get_bb_product_definitions


VALID_PHOENIX_PARAMS = {
    "S0": "100.0",
    "r": "0.03",
    "sigma": "0.2",
    "T": "1.0",
    "autocall_barrier_frac": "1.05",
    "coupon_barrier_frac": "1.0",
    "coupon_rate": "0.02",
    "knock_in_frac": "0.7",
    "obs_count": "6",
}

VALID_PHOENIX_TERMS = {
    "maturity_years": 1.0,
    "autocall_barrier_frac": 1.05,
    "coupon_barrier_frac": 1.0,
    "coupon_rate": 0.02,
    "knock_in_frac": 0.7,
    "obs_count": 6,
}


def make_market_snapshot(dividend_yield=0.0, symbol="SPY"):
    timestamp = datetime(2026, 7, 13, 12, 0, tzinfo=timezone.utc)
    return EquityMarketSnapshot(
        symbol=symbol,
        underlier_type="etf",
        currency="USD",
        valuation_time=timestamp,
        market_data_time=timestamp,
        spot=100.0,
        risk_free_rate=0.03,
        dividend_yield=dividend_yield,
        volatility=0.2,
        calendar="XNYS",
        day_count="ACT/365F",
        source="test-fixture",
    )


def make_term_structure(segments=None, spot=100.0):
    timestamp = datetime(2026, 7, 13, 12, 0, tzinfo=timezone.utc)
    return EquityMarketTermStructure(
        symbol="SPY",
        underlier_type="etf",
        currency="USD",
        valuation_time=timestamp,
        market_data_time=timestamp,
        spot=spot,
        segments=tuple(
            segments
            or [
                EquityMarketSegment(
                    end_time_years=1.0,
                    risk_free_rate=0.03,
                    dividend_yield=0.0,
                    volatility=0.2,
                )
            ]
        ),
        calendar="XNYS",
        day_count="ACT/365F",
        source="test-fixture",
    )


@pytest.fixture(autouse=True)
def reset_model_cache():
    clear_model_cache()
    yield
    clear_model_cache()


def default_params_for(product):
    return {field.name: str(field.default) for field in product.bb_fields}


def make_v2_contract(**overrides):
    values = {
        "reference_level": 100.0,
        "maturity_years": 1.0,
        "observation_times_years": (0.2, 0.45, 0.7, 1.0),
        "autocall_barrier_frac": 1.05,
        "coupon_barrier_frac": 1.0,
        "coupon_rate": 0.02,
        "knock_in_frac": 0.7,
        "prior_knock_in_breached": False,
    }
    values.update(overrides)
    return PhoenixSingleV2Contract(**values)


def test_pricing_service_prices_valid_phoenix_request():
    result = price_product("phoenix", VALID_PHOENIX_PARAMS, n_paths=5)

    assert result["product_key"] == "phoenix"
    assert result["model"] == "Monte Carlo reference"
    assert result["contract_version"] == "phoenix-single-v1"
    assert result["pricing_method"] == "monte_carlo_reference"
    assert isinstance(result["price"], float)
    assert result["mc_price"] == result["price"]
    assert result["standard_error"] >= 0
    assert len(result["confidence_interval"]) == 2
    assert result["latency_ms"] >= 0


def test_pricing_service_prices_each_bb_enabled_product():
    for product in get_bb_product_definitions():
        result = price_product(product.key, default_params_for(product), n_paths=5)

        assert result["product_key"] == product.key
        assert result["product_name"] == product.display_name
        assert result["model"] == "Monte Carlo reference"
        assert isinstance(result["price"], float)
        assert isinstance(result["mc_price"], float)
        assert result["latency_ms"] >= 0


def test_pricing_service_rejects_invalid_product():
    with pytest.raises(UnsupportedProductError):
        price_product("not_real", VALID_PHOENIX_PARAMS, n_paths=5)


def test_pricing_service_rejects_invalid_numeric_input():
    params = dict(VALID_PHOENIX_PARAMS)
    params["sigma"] = "not-a-number"

    with pytest.raises(InvalidPricingInputError):
        normalize_pricing_params("phoenix", params)


def test_pricing_service_rejects_missing_required_input():
    params = dict(VALID_PHOENIX_PARAMS)
    del params["sigma"]

    with pytest.raises(InvalidPricingInputError):
        normalize_pricing_params("phoenix", params)


def test_pricing_service_is_deterministic_without_surrogate_artifacts(tmp_path):
    first = price_product(
        "phoenix", VALID_PHOENIX_PARAMS, n_paths=100, results_dir=tmp_path
    )
    second = price_product(
        "phoenix", VALID_PHOENIX_PARAMS, n_paths=100, results_dir=tmp_path
    )

    assert first["price"] == second["price"]
    assert first["confidence_interval"] == second["confidence_interval"]


def test_pricing_service_rejects_invalid_barrier_order():
    params = dict(VALID_PHOENIX_PARAMS)
    params["coupon_barrier_frac"] = "1.10"

    with pytest.raises(InvalidPricingInputError, match="barriers must satisfy"):
        normalize_pricing_params("phoenix", params)


def test_pricing_service_rejects_non_finite_and_excessive_inputs():
    non_finite = dict(VALID_PHOENIX_PARAMS)
    non_finite["sigma"] = "nan"
    with pytest.raises(InvalidPricingInputError, match="must be finite"):
        normalize_pricing_params("phoenix", non_finite)

    too_many_observations = dict(VALID_PHOENIX_PARAMS)
    too_many_observations["obs_count"] = "253"
    with pytest.raises(InvalidPricingInputError, match="must be <= 252"):
        normalize_pricing_params("phoenix", too_many_observations)


def test_snapshot_pricing_separates_market_data_from_phoenix_terms():
    result = price_phoenix_with_market_snapshot(
        make_market_snapshot(symbol="7203.T"), VALID_PHOENIX_TERMS, n_paths=100
    )

    assert result["underlier"] == {
        "symbol": "7203.T",
        "type": "etf",
        "currency": "USD",
    }
    assert result["terms"] == VALID_PHOENIX_TERMS
    assert result["market_snapshot_version"] == "equity-market-snapshot-v1"
    assert result["model_version"] == "equity-gbm-flat-v2"
    assert result["market_snapshot"]["snapshot_id"].startswith("sha256:")


def test_zero_dividend_snapshot_matches_legacy_numerics():
    snapshot_result = price_phoenix_with_market_snapshot(
        make_market_snapshot(), VALID_PHOENIX_TERMS, n_paths=100
    )
    legacy_result = price_product("phoenix", VALID_PHOENIX_PARAMS, n_paths=100)

    assert snapshot_result["price"] == legacy_result["price"]
    assert (
        snapshot_result["confidence_interval"] == legacy_result["confidence_interval"]
    )


def test_snapshot_pricing_uses_dividend_yield():
    no_dividend = price_phoenix_with_market_snapshot(
        make_market_snapshot(dividend_yield=0.0), VALID_PHOENIX_TERMS, n_paths=500
    )
    with_dividend = price_phoenix_with_market_snapshot(
        make_market_snapshot(dividend_yield=0.05), VALID_PHOENIX_TERMS, n_paths=500
    )

    assert with_dividend["price"] != no_dividend["price"]


def test_snapshot_pricing_rejects_unknown_terms():
    terms = {**VALID_PHOENIX_TERMS, "memory_coupon": True}

    with pytest.raises(InvalidPricingInputError, match="unknown Phoenix terms"):
        price_phoenix_with_market_snapshot(make_market_snapshot(), terms, n_paths=5)


def test_term_structure_pricing_is_versioned_and_attributable():
    result = price_phoenix_with_term_structure(
        make_term_structure(), VALID_PHOENIX_TERMS, n_paths=100
    )

    assert result["model_version"] == "equity-gbm-piecewise-v1"
    assert result["market_data_version"] == "equity-market-term-structure-v1"
    assert result["market_term_structure"]["term_structure_id"].startswith("sha256:")
    assert result["underlier"] == {
        "symbol": "SPY",
        "type": "etf",
        "currency": "USD",
    }


def test_v2_pricing_separates_live_spot_from_contract_reference_level():
    market = make_term_structure(spot=82.0)
    contract = make_v2_contract(reference_level=100.0)

    result = price_phoenix_v2_with_term_structure(
        market,
        contract,
        n_paths=100,
    )

    assert result["contract_version"] == "phoenix-single-v2"
    assert result["market_term_structure"]["spot"] == 82.0
    assert result["params"]["S0"] == 100.0
    assert result["contract"]["reference_level"] == 100.0
    assert result["contract"]["contract_id"] == contract.contract_id
    assert result["surrogate_shadow"]["status"] == "not_applicable"


def test_v2_pricing_carries_prior_knock_in_into_the_remaining_value():
    market = make_term_structure(
        segments=[
            EquityMarketSegment(
                end_time_years=1.0,
                risk_free_rate=0.0,
                dividend_yield=0.0,
                volatility=0.000001,
            )
        ],
        spot=80.0,
    )
    protected = make_v2_contract(
        observation_times_years=(1.0,),
        autocall_barrier_frac=2.0,
        coupon_barrier_frac=2.0,
        prior_knock_in_breached=False,
    )
    breached = make_v2_contract(
        observation_times_years=(1.0,),
        autocall_barrier_frac=2.0,
        coupon_barrier_frac=2.0,
        prior_knock_in_breached=True,
    )

    protected_result = price_phoenix_v2_with_term_structure(
        market, protected, n_paths=100
    )
    breached_result = price_phoenix_v2_with_term_structure(
        market, breached, n_paths=100
    )

    assert protected_result["price"] == pytest.approx(1.0)
    assert breached_result["price"] == pytest.approx(0.8, rel=1e-5)


def test_term_structure_shadow_result_never_replaces_reference_price(monkeypatch):
    monkeypatch.setattr(
        "app.services.surrogate_service.evaluate_surrogate_shadow",
        lambda **kwargs: {
            "status": "success",
            "used_for_price": False,
            "surrogate_price": kwargs["reference_price"] + 0.5,
        },
    )

    result = price_phoenix_with_term_structure(
        make_term_structure(), VALID_PHOENIX_TERMS, n_paths=100
    )

    assert result["price"] == result["mc_price"]
    assert result["surrogate_shadow"]["surrogate_price"] == pytest.approx(
        result["price"] + 0.5
    )
    assert result["surrogate_shadow"]["used_for_price"] is False


def test_term_structure_shadow_failure_is_isolated(monkeypatch):
    def fail_shadow(**_kwargs):
        raise RuntimeError("artifact details must not leak")

    monkeypatch.setattr(
        "app.services.surrogate_service.evaluate_surrogate_shadow", fail_shadow
    )

    result = price_phoenix_with_term_structure(
        make_term_structure(), VALID_PHOENIX_TERMS, n_paths=100
    )

    assert result["price"] == result["mc_price"]
    assert result["surrogate_shadow"] == {
        "status": "error",
        "mode": "shadow-only",
        "used_for_price": False,
        "reason": "surrogate shadow evaluation failed",
    }


def test_one_segment_term_structure_matches_flat_snapshot_numerics():
    flat = price_phoenix_with_market_snapshot(
        make_market_snapshot(), VALID_PHOENIX_TERMS, n_paths=500
    )
    piecewise = price_phoenix_with_term_structure(
        make_term_structure(), VALID_PHOENIX_TERMS, n_paths=500
    )

    assert piecewise["price"] == pytest.approx(flat["price"], abs=1e-14)
    assert piecewise["standard_error"] == pytest.approx(
        flat["standard_error"], abs=1e-14
    )


def test_term_structure_pricing_uses_curve_and_volatility_shape():
    shaped_market = make_term_structure(
        [
            EquityMarketSegment(0.5, 0.01, 0.0, 0.1),
            EquityMarketSegment(1.0, 0.05, 0.02, 0.3),
        ]
    )

    flat = price_phoenix_with_term_structure(
        make_term_structure(), VALID_PHOENIX_TERMS, n_paths=500
    )
    shaped = price_phoenix_with_term_structure(
        shaped_market, VALID_PHOENIX_TERMS, n_paths=500
    )

    assert shaped["price"] != flat["price"]


def test_term_structure_pricing_rejects_insufficient_coverage():
    short_market = make_term_structure([EquityMarketSegment(0.5, 0.03, 0.0, 0.2)])

    with pytest.raises(InvalidPricingInputError, match="does not cover"):
        price_phoenix_with_term_structure(short_market, VALID_PHOENIX_TERMS, n_paths=5)
