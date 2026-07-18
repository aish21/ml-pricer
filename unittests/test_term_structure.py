from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from src.final.data_generator import (
    build_simulation_time_grid,
    simulate_gbm_paths,
    simulate_piecewise_gbm_paths,
)
from src.final.market import (
    EquityMarketSegment,
    EquityMarketTermStructure,
    MarketDataValidationError,
)


VALUATION_TIME = datetime(2026, 7, 14, 12, 0, tzinfo=timezone.utc)


def segment(end, rate=0.03, dividend=0.01, volatility=0.2):
    return EquityMarketSegment(
        end_time_years=end,
        risk_free_rate=rate,
        dividend_yield=dividend,
        volatility=volatility,
    )


def make_term_structure(segments=None, **overrides):
    values = {
        "symbol": "SPY",
        "underlier_type": "etf",
        "currency": "usd",
        "valuation_time": VALUATION_TIME,
        "market_data_time": VALUATION_TIME - timedelta(seconds=2),
        "spot": 100.0,
        "segments": tuple([segment(1.0)] if segments is None else segments),
        "calendar": "xnys",
        "day_count": "act/365f",
        "source": "test-fixture",
    }
    values.update(overrides)
    return EquityMarketTermStructure(**values)


def test_term_structure_is_normalized_dated_and_fingerprinted():
    first = make_term_structure()
    second = make_term_structure()

    assert first.currency == "USD"
    assert first.calendar == "XNYS"
    assert first.age_seconds == 2.0
    assert first.max_time_years == 1.0
    assert first.term_structure_id == second.term_structure_id
    payload = first.to_dict()
    assert payload["schema_version"] == "equity-market-term-structure-v1"
    assert payload["term_structure_id"].startswith("sha256:")


def test_term_structure_fingerprint_changes_with_market_inputs():
    first = make_term_structure([segment(1.0, volatility=0.2)])
    second = make_term_structure([segment(1.0, volatility=0.21)])

    assert first.term_structure_id != second.term_structure_id


def test_term_structure_rejects_lookahead_market_data():
    with pytest.raises(
        MarketDataValidationError,
        match="market_data_time cannot be after valuation_time",
    ):
        make_term_structure(market_data_time=VALUATION_TIME + timedelta(seconds=1))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_market_segment_rejects_non_finite_values(value):
    with pytest.raises(MarketDataValidationError, match="volatility must be finite"):
        segment(1.0, volatility=value)


def test_term_structure_integrates_segments_exactly():
    market = make_term_structure(
        [
            segment(0.5, rate=0.02, dividend=0.01, volatility=0.1),
            segment(1.0, rate=0.04, dividend=0.02, volatility=0.3),
        ]
    )

    assert market.integrated_risk_free_rate(0.0, 1.0) == pytest.approx(0.03)
    assert market.integrated_dividend_yield(0.0, 1.0) == pytest.approx(0.015)
    assert market.integrated_variance(0.0, 1.0) == pytest.approx(0.05)
    assert market.discount_factor(1.0) == pytest.approx(np.exp(-0.03))
    equivalent = market.equivalent_flat_parameters(1.0)
    assert equivalent == pytest.approx(
        {
            "risk_free_rate": 0.03,
            "dividend_yield": 0.015,
            "volatility": np.sqrt(0.05),
        }
    )


@pytest.mark.parametrize(
    ("segments", "message"),
    [
        ([], "between 1 and 252"),
        ([segment(1.0), segment(0.5)], "strictly increasing"),
        ([segment(1.0), segment(1.0)], "strictly increasing"),
    ],
)
def test_term_structure_rejects_invalid_tenors(segments, message):
    with pytest.raises(MarketDataValidationError, match=message):
        make_term_structure(segments)


def test_term_structure_refuses_silent_extrapolation():
    market = make_term_structure([segment(0.5)])

    with pytest.raises(MarketDataValidationError, match="does not cover"):
        market.discount_factor(1.0)
    with pytest.raises(MarketDataValidationError, match="does not cover"):
        market.equivalent_flat_parameters(1.0)


def test_one_segment_piecewise_paths_match_flat_gbm():
    market = make_term_structure(
        [segment(1.0, rate=0.03, dividend=0.01, volatility=0.2)]
    )

    piecewise = simulate_piecewise_gbm_paths(
        market=market,
        T=1.0,
        n_steps=12,
        n_paths=10,
        seed=42,
    )
    flat = simulate_gbm_paths(
        s0=100.0,
        r=0.03,
        sigma=0.2,
        T=1.0,
        n_steps=12,
        n_paths=10,
        seed=42,
        dividend_yield=0.01,
    )

    assert piecewise == pytest.approx(flat, rel=1e-14, abs=1e-14)


def test_piecewise_simulation_is_deterministic_and_uses_term_shape():
    flat_market = make_term_structure([segment(1.0, volatility=0.2)])
    shaped_market = make_term_structure(
        [segment(0.5, volatility=0.1), segment(1.0, volatility=0.3)]
    )

    first = simulate_piecewise_gbm_paths(shaped_market, 1.0, 12, 10, seed=7)
    second = simulate_piecewise_gbm_paths(shaped_market, 1.0, 12, 10, seed=7)
    flat = simulate_piecewise_gbm_paths(flat_market, 1.0, 12, 10, seed=7)

    assert np.array_equal(first, second)
    assert not np.array_equal(first, flat)


def test_contract_event_times_are_inserted_exactly_into_simulation_grid():
    time_grid = build_simulation_time_grid(
        1.0,
        4,
        required_times_years=(0.4, 0.9, 1.0),
    )
    market = make_term_structure()

    paths = simulate_piecewise_gbm_paths(
        market,
        1.0,
        len(time_grid) - 1,
        3,
        seed=7,
        time_grid_years=time_grid,
    )

    assert 0.4 in time_grid
    assert 0.9 in time_grid
    assert paths.shape == (3, len(time_grid))
