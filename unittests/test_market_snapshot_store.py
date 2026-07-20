from datetime import datetime, timezone

from app.services.market_snapshot_store import (
    get_market_snapshot_store_status,
    get_research_market_snapshot,
    list_research_market_snapshots,
    save_research_market_snapshot,
)


SNAPSHOT_ID = f"sha256:{'1' * 64}"
TERM_STRUCTURE_ID = f"sha256:{'2' * 64}"


def _market():
    return {
        "symbol": "SPY",
        "underlier_type": "etf",
        "currency": "USD",
        "market_data_time": "2026-07-19T10:00:00+00:00",
        "term_structure_id": TERM_STRUCTURE_ID,
        "segments": [
            {
                "end_time_years": 1.0,
                "risk_free_rate": 0.04,
                "dividend_yield": 0.01,
                "volatility": 0.2,
            }
        ],
    }


def test_market_snapshot_store_is_immutable_and_returns_bounded_metadata(tmp_path):
    database = tmp_path / "snapshots.sqlite3"
    created = datetime(2026, 7, 19, 10, 1, tzinfo=timezone.utc)
    first = save_research_market_snapshot(
        market=_market(),
        calibration={"calibration_id": SNAPSHOT_ID, "quality": {"status": "first"}},
        db_path=database,
        now=created,
    )
    save_research_market_snapshot(
        market=_market(),
        calibration={"calibration_id": SNAPSHOT_ID, "quality": {"status": "changed"}},
        db_path=database,
        now=datetime(2026, 7, 19, 11, 0, tzinfo=timezone.utc),
    )

    loaded = get_research_market_snapshot(SNAPSHOT_ID, db_path=database)
    recent = list_research_market_snapshots(limit=500, db_path=database)
    status = get_market_snapshot_store_status(
        db_path=database,
        now=datetime(2026, 7, 19, 12, 0, tzinfo=timezone.utc),
    )

    assert first["snapshot_id"] == SNAPSHOT_ID
    assert first["immutable"] is True
    assert loaded["created_at"] == created.isoformat()
    assert loaded["market_calibration"]["quality"]["status"] == "first"
    assert recent == [first]
    assert status == {
        "available": True,
        "snapshot_count": 1,
        "latest_created_at": created.isoformat(),
        "latest_market_data_time": "2026-07-19T10:00:00+00:00",
        "latest_market_data_age_hours": 2.0,
    }
