from datetime import datetime, timezone

import pytest

from app.services.diagnostics_service import (
    InvalidDiagnosticsInputError,
    get_phoenix_v1_diagnostics,
    get_phoenix_v2_diagnostics,
)
from src.final.market import EquityMarketSegment, EquityMarketTermStructure
from src.final.phoenix_contract import PhoenixSingleV2Contract


TERMS = {
    "maturity_years": 1.0,
    "autocall_barrier_frac": 1.05,
    "coupon_barrier_frac": 1.0,
    "coupon_rate": 0.02,
    "knock_in_frac": 0.7,
    "obs_count": 4,
}


def make_market(spot=100.0):
    timestamp = datetime(2026, 7, 18, 12, 0, tzinfo=timezone.utc)
    return EquityMarketTermStructure(
        symbol="SPY",
        underlier_type="etf",
        currency="USD",
        valuation_time=timestamp,
        market_data_time=timestamp,
        spot=spot,
        segments=(
            EquityMarketSegment(
                end_time_years=1.0,
                risk_free_rate=0.03,
                dividend_yield=0.01,
                volatility=0.2,
            ),
        ),
        calendar="XNYS",
        day_count="ACT/365F",
        source="test",
    )


def make_contract():
    return PhoenixSingleV2Contract(
        reference_level=110.0,
        maturity_years=1.0,
        observation_times_years=(0.18, 0.43, 0.68, 1.0),
        autocall_barrier_frac=1.05,
        coupon_barrier_frac=1.0,
        coupon_rate=0.02,
        knock_in_frac=0.7,
        prior_knock_in_breached=True,
    )


def test_v1_diagnostics_reconcile_cashflows_and_use_nested_convergence():
    report = get_phoenix_v1_diagnostics(
        market=make_market(),
        terms=TERMS,
        n_paths=200,
        seed=7,
        convergence_path_counts=(50, 100, 200),
        spot_shocks_pct=(-10.0, 0.0, 10.0),
        volatility_shocks_abs=(0.0,),
    )

    component_total = sum(
        row["expected_pv"] for row in report["cashflows"]["components"]
    )
    assert report["contract_version"] == "phoenix-single-v1"
    assert component_total == pytest.approx(report["base"]["price"])
    assert [point["n_paths"] for point in report["convergence"]] == [50, 100, 200]
    assert len(report["surface"]["cells"]) == 3
    assert report["provenance"]["raw_paths_returned"] is False
    assert report["diagnostic_id"].startswith("sha256:")


def test_v2_diagnostics_preserve_contract_state_and_exact_schedule():
    contract = make_contract()

    report = get_phoenix_v2_diagnostics(
        market=make_market(spot=90.0),
        contract=contract,
        n_paths=100,
        seed=11,
        spot_shocks_pct=(0.0,),
        volatility_shocks_abs=(0.0,),
    )

    assert report["contract_version"] == "phoenix-single-v2"
    assert report["provenance"]["contract"]["contract_id"] == contract.contract_id
    assert report["provenance"]["effective_simulation_steps"] > 252
    assert report["cashflows"]["downside_probability"] >= 0.0
    assert len(report["distribution"]["histogram"]["counts"]) == 24


def test_diagnostics_are_deterministic_for_the_same_request():
    kwargs = {
        "market": make_market(),
        "terms": TERMS,
        "n_paths": 100,
        "seed": 19,
        "spot_shocks_pct": (0.0,),
        "volatility_shocks_abs": (0.0,),
    }

    first = get_phoenix_v1_diagnostics(**kwargs)
    second = get_phoenix_v1_diagnostics(**kwargs)

    assert first["diagnostic_id"] == second["diagnostic_id"]
    assert first["base"] == second["base"]
    assert first["distribution"] == second["distribution"]


def test_diagnostics_reject_excessive_surface_work():
    with pytest.raises(InvalidDiagnosticsInputError, match="work limit"):
        get_phoenix_v1_diagnostics(
            market=make_market(),
            terms=TERMS,
            n_paths=5_000,
            spot_shocks_pct=tuple(range(-50, 51, 10)),
            volatility_shocks_abs=tuple(value / 100 for value in range(-5, 6)),
        )
