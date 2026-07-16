from app.frontend_support import (
    compact_nonzero_shock,
    frozen_term_structure_from_pricing_result,
)


def test_frontend_reuses_existing_term_structure_unchanged():
    market = {
        "schema_version": "equity-market-term-structure-v1",
        "symbol": "SPY",
        "underlier_type": "etf",
        "currency": "USD",
        "valuation_time": "2026-07-15T12:00:02Z",
        "market_data_time": "2026-07-15T12:00:00Z",
        "spot": 100,
        "segments": [],
        "calendar": "XNYS",
        "day_count": "ACT/365F",
        "source": "test",
        "term_structure_id": "sha256:computed-response-field",
        "age_seconds": 2.0,
        "max_time_years": 1.0,
    }
    frozen = frozen_term_structure_from_pricing_result(
        {"market_term_structure": market}, 1.0
    )
    assert frozen["spot"] == 100
    assert "term_structure_id" not in frozen
    assert "age_seconds" not in frozen


def test_frontend_lifts_flat_snapshot_into_frozen_one_segment_curve():
    snapshot = {
        "symbol": "SPY",
        "underlier_type": "etf",
        "currency": "USD",
        "valuation_time": "2026-07-15T12:00:02Z",
        "market_data_time": "2026-07-15T12:00:00Z",
        "spot": 620.0,
        "risk_free_rate": 0.04,
        "dividend_yield": 0.01,
        "volatility": 0.2,
        "calendar": "XNYS",
        "day_count": "ACT/365F",
        "source": "test",
    }
    market = frozen_term_structure_from_pricing_result(
        {"market_snapshot": snapshot}, 1.5
    )
    assert market["spot"] == 620.0
    assert market["segments"] == [
        {
            "end_time_years": 1.5,
            "risk_free_rate": 0.04,
            "dividend_yield": 0.01,
            "volatility": 0.2,
        }
    ]


def test_frontend_compacts_parallel_and_bucket_shocks():
    shock = compact_nonzero_shock(
        spot_pct=-10,
        rate_parallel_bps=0,
        dividend_parallel_bps=5,
        volatility_parallel_abs=0,
        segment_shock={
            "segment_index": 1,
            "rate_bps": 10,
            "dividend_bps": 0,
            "volatility_abs": 0.02,
        },
    )
    assert shock == {
        "spot_pct": -10.0,
        "dividend_parallel_bps": 5.0,
        "segment_shocks": [
            {"segment_index": 1, "rate_bps": 10.0, "volatility_abs": 0.02}
        ],
    }
