from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from src.final.data_generator import simulate_gbm_paths
from src.final.market import EquityMarketSnapshot, MarketDataValidationError


VALUATION_TIME = datetime(2026, 7, 13, 12, 0, tzinfo=timezone.utc)


def make_snapshot(**overrides):
    values = {
        "symbol": "SPY",
        "underlier_type": "etf",
        "currency": "usd",
        "valuation_time": VALUATION_TIME,
        "market_data_time": VALUATION_TIME - timedelta(seconds=2),
        "spot": 620.0,
        "risk_free_rate": 0.04,
        "dividend_yield": 0.012,
        "volatility": 0.2,
        "calendar": "xnys",
        "day_count": "act/365f",
        "source": "test-fixture",
    }
    values.update(overrides)
    return EquityMarketSnapshot(**values)


def test_market_snapshot_is_normalized_dated_and_fingerprinted():
    first = make_snapshot()
    second = make_snapshot()

    assert first.currency == "USD"
    assert first.underlier_type == "etf"
    assert first.calendar == "XNYS"
    assert first.age_seconds == 2.0
    assert first.snapshot_id == second.snapshot_id
    assert first.to_dict()["schema_version"] == "equity-market-snapshot-v1"


def test_market_snapshot_fingerprint_changes_with_market_data():
    first = make_snapshot()
    second = make_snapshot(spot=621.0)

    assert first.snapshot_id != second.snapshot_id


def test_market_snapshot_normalizes_equivalent_timestamps_to_utc():
    singapore_time = timezone(timedelta(hours=8))
    first = make_snapshot()
    second = make_snapshot(
        valuation_time=VALUATION_TIME.astimezone(singapore_time),
        market_data_time=(VALUATION_TIME - timedelta(seconds=2)).astimezone(
            singapore_time
        ),
    )

    assert second.to_dict()["valuation_time"].endswith("Z")
    assert first.snapshot_id == second.snapshot_id


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"underlier_type": "crypto"}, "underlier_type"),
        ({"valuation_time": datetime(2026, 7, 13, 12, 0)}, "UTC offset"),
        (
            {"market_data_time": VALUATION_TIME + timedelta(seconds=1)},
            "cannot be after",
        ),
        ({"spot": float("nan")}, "spot must be finite"),
    ],
)
def test_market_snapshot_rejects_invalid_or_lookahead_data(overrides, message):
    with pytest.raises(MarketDataValidationError, match=message):
        make_snapshot(**overrides)


def test_gbm_dividend_yield_reduces_risk_neutral_forward():
    no_dividend = simulate_gbm_paths(
        s0=100.0,
        r=0.05,
        sigma=0.0,
        T=1.0,
        n_steps=4,
        n_paths=1,
        seed=7,
    )
    with_dividend = simulate_gbm_paths(
        s0=100.0,
        r=0.05,
        sigma=0.0,
        T=1.0,
        n_steps=4,
        n_paths=1,
        seed=7,
        dividend_yield=0.02,
    )

    assert no_dividend[0, -1] == pytest.approx(100.0 * np.exp(0.05))
    assert with_dividend[0, -1] == pytest.approx(100.0 * np.exp(0.03))
