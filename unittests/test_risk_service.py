from datetime import datetime, timezone

import numpy as np
import pytest

from app.services.risk_service import (
    InvalidRiskInputError,
    apply_term_structure_shock,
    calculate_phoenix_term_structure_risk,
    normalize_risk_bumps,
    normalize_term_structure_shock,
    run_phoenix_term_structure_scenario,
)
from src.final.data_generator import simulate_piecewise_gbm_paths
from src.final.market import EquityMarketSegment, EquityMarketTermStructure


TERMS = {
    "maturity_years": 1.0,
    "autocall_barrier_frac": 1.05,
    "coupon_barrier_frac": 0.7,
    "coupon_rate": 0.08,
    "knock_in_frac": 0.6,
    "obs_count": 4,
}


def make_market() -> EquityMarketTermStructure:
    valuation_time = datetime(2026, 7, 15, 12, 0, 2, tzinfo=timezone.utc)
    return EquityMarketTermStructure(
        symbol="SPY",
        underlier_type="etf",
        currency="USD",
        valuation_time=valuation_time,
        market_data_time=datetime(2026, 7, 15, 12, 0, tzinfo=timezone.utc),
        spot=100.0,
        segments=(
            EquityMarketSegment(0.5, 0.03, 0.01, 0.20),
            EquityMarketSegment(1.0, 0.04, 0.012, 0.25),
        ),
        calendar="XNYS",
        day_count="ACT/365F",
        source="risk-test",
    )


def test_term_structure_shock_combines_parallel_and_segment_changes():
    market = make_market()
    shocked, normalized = apply_term_structure_shock(
        market,
        {
            "spot_pct": -10,
            "rate_parallel_bps": 25,
            "volatility_parallel_abs": 0.01,
            "segment_shocks": [
                {
                    "segment_index": 1,
                    "rate_bps": 10,
                    "dividend_bps": -5,
                    "volatility_abs": 0.02,
                }
            ],
        },
    )

    assert market.spot == 100.0
    assert shocked.spot == pytest.approx(90.0)
    assert shocked.segments[0].risk_free_rate == pytest.approx(0.0325)
    assert shocked.segments[1].risk_free_rate == pytest.approx(0.0435)
    assert shocked.segments[1].dividend_yield == pytest.approx(0.0115)
    assert shocked.segments[1].volatility == pytest.approx(0.28)
    assert normalized["segment_shocks"][0]["segment_index"] == 1
    assert shocked.term_structure_id != market.term_structure_id


@pytest.mark.parametrize(
    "shock, message",
    [
        ({}, "at least one"),
        ({"spot_pct": 0}, "at least one"),
        (
            {
                "segment_shocks": [
                    {"segment_index": 0, "rate_bps": 1},
                    {"segment_index": 0, "rate_bps": 2},
                ]
            },
            "duplicate",
        ),
        (
            {"segment_shocks": [{"segment_index": 2, "rate_bps": 1}]},
            "outside",
        ),
        ({"volatility_parallel_abs": -0.2}, "invalid"),
    ],
)
def test_term_structure_shock_rejects_invalid_requests(shock, message):
    with pytest.raises(InvalidRiskInputError, match=message):
        apply_term_structure_shock(make_market(), shock)


def test_shock_normalization_rejects_unknown_fields_and_zero_bucket():
    with pytest.raises(InvalidRiskInputError, match="unknown shock"):
        normalize_term_structure_shock({"credit_spread_bps": 100})
    with pytest.raises(InvalidRiskInputError, match="non-zero"):
        normalize_term_structure_shock(
            {"segment_shocks": [{"segment_index": 0, "rate_bps": 0}]}
        )


def test_scenario_freezes_contract_reference_spot_and_is_reproducible(monkeypatch):
    calls = []

    def fake_pathwise_payoffs(*, params, market, n_paths, **kwargs):
        calls.append((params["S0"], market.spot))
        return np.full(n_paths, market.spot / params["S0"])

    monkeypatch.setattr(
        "app.services.risk_service.phoenix_piecewise_discounted_payoffs",
        fake_pathwise_payoffs,
    )
    first = run_phoenix_term_structure_scenario(
        market=make_market(),
        terms=TERMS,
        shock={"spot_pct": 10},
        n_paths=8,
        seed=7,
        market_calibration={"calibration_id": "sha256:test"},
    )
    second = run_phoenix_term_structure_scenario(
        market=make_market(),
        terms=TERMS,
        shock={"spot_pct": 10},
        n_paths=8,
        seed=7,
        market_calibration={"calibration_id": "sha256:test"},
    )

    assert [reference for reference, _ in calls] == [100.0] * 4
    assert [spot for _, spot in calls] == pytest.approx([100.0, 110.0] * 2)
    assert first["pnl"]["value"] == pytest.approx(0.1)
    assert first["pnl"]["standard_error"] == 0.0
    assert first["scenario_id"] == second["scenario_id"]
    assert first["provenance"]["market_calibration_id"] == "sha256:test"
    assert first["provenance"]["common_random_numbers"] is True


def test_risk_finite_differences_have_explicit_units_and_paired_noise(monkeypatch):
    def analytical_payoffs(*, market, n_paths, standard_normal_shocks, **kwargs):
        segment = market.segments[0]
        deterministic = (
            market.spot**2
            + 3.0 * segment.volatility
            + 5.0 * segment.risk_free_rate
            + 7.0 * segment.dividend_yield
        )
        return deterministic + standard_normal_shocks[:, 0] * 0.1

    monkeypatch.setattr(
        "app.services.risk_service.phoenix_piecewise_discounted_payoffs",
        analytical_payoffs,
    )
    result = calculate_phoenix_term_structure_risk(
        market=make_market(), terms=TERMS, n_paths=32, seed=11
    )

    sensitivities = result["sensitivities"]
    assert sensitivities["delta"]["value"] == pytest.approx(200.0)
    assert sensitivities["gamma"]["value"] == pytest.approx(2.0)
    assert sensitivities["vega"]["value"] == pytest.approx(0.03)
    assert sensitivities["rho"]["value"] == pytest.approx(0.05)
    assert sensitivities["dividend_rho"]["value"] == pytest.approx(0.07)
    assert all(item["standard_error"] < 1e-12 for item in sensitivities.values())
    assert sensitivities["vega"]["units"] == "price change per 1 volatility point"
    assert result["provenance"]["common_random_numbers"] is True
    assert len(result["bump_valuations"]) == 8


def test_risk_rejects_bump_that_makes_market_invalid():
    with pytest.raises(InvalidRiskInputError, match="invalid"):
        calculate_phoenix_term_structure_risk(
            market=make_market(),
            terms=TERMS,
            n_paths=8,
            bumps={"volatility_absolute": 0.2},
        )


def test_risk_bump_normalization_is_bounded_and_uses_defaults():
    assert normalize_risk_bumps(None)["spot_relative"] == 0.01
    with pytest.raises(InvalidRiskInputError, match="between"):
        normalize_risk_bumps({"spot_relative": 0.0})


def test_explicit_standard_normal_shocks_are_reused_and_validated():
    market = make_market()
    shocks = np.zeros((3, 4))
    first = simulate_piecewise_gbm_paths(
        market, T=1.0, n_steps=4, n_paths=3, standard_normal_shocks=shocks
    )
    second = simulate_piecewise_gbm_paths(
        market,
        T=1.0,
        n_steps=4,
        n_paths=3,
        seed=999,
        standard_normal_shocks=shocks,
    )
    assert np.array_equal(first, second)
    with pytest.raises(ValueError, match="shape"):
        simulate_piecewise_gbm_paths(
            market,
            T=1.0,
            n_steps=4,
            n_paths=3,
            standard_normal_shocks=np.zeros((2, 4)),
        )
