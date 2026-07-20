from datetime import datetime, timezone

import pytest

from app.ui.payloads import (
    FrontendInputError,
    barrier_levels,
    build_flat_term_structure,
    build_barrier_reverse_convertible_contract,
    build_v3_contract,
    diagnostic_grids,
    even_observation_schedule,
    parse_observation_schedule,
    stepped_autocall_schedule,
)


def test_even_schedule_includes_maturity_exactly():
    schedule = even_observation_schedule(1.0, 4)

    assert schedule == (0.25, 0.5, 0.75, 1.0)


def test_explicit_schedule_requires_order_and_maturity():
    assert parse_observation_schedule("0.2, 0.55, 1.0", 1.0) == (
        0.2,
        0.55,
        1.0,
    )

    with pytest.raises(FrontendInputError, match="strictly increasing"):
        parse_observation_schedule("0.5, 0.4, 1.0", 1.0)
    with pytest.raises(FrontendInputError, match="final observation"):
        parse_observation_schedule("0.25, 0.75", 1.0)


def test_flat_market_payload_is_dated_and_normalized():
    timestamp = datetime(2026, 7, 18, 12, 0, tzinfo=timezone.utc)

    market = build_flat_term_structure(
        symbol=" spy ",
        underlier_type="ETF",
        currency="usd",
        spot=620.0,
        risk_free_rate=0.04,
        dividend_yield=0.01,
        volatility=0.2,
        maturity_years=1.0,
        valuation_time=timestamp,
    )

    assert market["symbol"] == "SPY"
    assert market["underlier_type"] == "etf"
    assert market["currency"] == "USD"
    assert market["valuation_time"] == "2026-07-18T12:00:00+00:00"
    assert market["segments"][0]["end_time_years"] == 1.0


def test_barrier_ladder_uses_reference_not_live_spot():
    levels = barrier_levels(
        live_spot=80.0,
        reference_level=100.0,
        terms={
            "knock_in_frac": 0.7,
            "coupon_barrier_frac": 1.0,
            "autocall_barrier_frac": 1.05,
        },
    )
    by_name = {item["name"]: item["level"] for item in levels}

    assert by_name["Live spot"] == 80.0
    assert by_name["Knock-in barrier"] == 70.0
    assert by_name["Autocall barrier"] == 105.0


def test_diagnostic_volatility_grid_cannot_make_base_volatility_non_positive():
    grids = diagnostic_grids(
        {
            "segments": [
                {"volatility": 0.02},
                {"volatility": 0.03},
            ]
        }
    )

    assert min(grids["volatility_shocks_abs"]) == -0.01


def test_richer_contract_payload_aligns_memory_and_stepdown_schedules():
    observations = even_observation_schedule(1.0, 3)
    barriers = stepped_autocall_schedule(
        initial_barrier_frac=1.10,
        final_barrier_frac=1.0,
        observation_count=3,
    )
    contract = build_v3_contract(
        reference_level=100.0,
        terms={
            "maturity_years": 1.0,
            "coupon_barrier_frac": 0.8,
            "coupon_rate": 0.02,
            "knock_in_frac": 0.6,
        },
        observation_times_years=observations,
        autocall_barrier_fracs=barriers,
        prior_knock_in_breached=True,
        memory_coupon=True,
        unpaid_coupon_count=2,
    )

    assert barriers == pytest.approx((1.10, 1.05, 1.0))
    assert contract["contract_version"] == "phoenix-single-v3"
    assert contract["autocall_barrier_fracs"] == pytest.approx(barriers)
    assert contract["unpaid_coupon_count"] == 2


def test_reverse_convertible_payload_keeps_coupon_and_downside_rules_explicit():
    contract = build_barrier_reverse_convertible_contract(
        reference_level=100.0,
        maturity_years=1.0,
        coupon_times_years=(0.25, 0.5, 0.75, 1.0),
        coupon_rate_per_period=0.02,
        strike_frac=1.0,
        knock_in_frac=0.7,
        prior_knock_in_breached=False,
    )

    assert contract["contract_version"] == "barrier-reverse-convertible-v1"
    assert contract["coupon_times_years"][-1] == 1.0
    assert contract["knock_in_frac"] == 0.7
