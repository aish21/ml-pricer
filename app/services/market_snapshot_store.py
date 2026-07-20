import json
import os
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from app.services.product_registry import REPO_ROOT


DEFAULT_MARKET_SNAPSHOT_STORE = REPO_ROOT / "data" / "market_snapshots.sqlite3"
SNAPSHOT_ID_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")


class MarketSnapshotStoreError(RuntimeError):
    pass


def get_market_snapshot_store_path() -> Path:
    configured = os.getenv("MARKET_SNAPSHOT_STORE_FILE")
    return Path(configured) if configured else DEFAULT_MARKET_SNAPSHOT_STORE


def _connect(db_path: Path | None = None) -> sqlite3.Connection:
    path = Path(db_path) if db_path else get_market_snapshot_store_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(path)
        connection.row_factory = sqlite3.Row
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS research_market_snapshots (
                snapshot_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                symbol TEXT NOT NULL,
                underlier_type TEXT NOT NULL,
                currency TEXT NOT NULL,
                maturity_years REAL NOT NULL,
                market_data_time TEXT NOT NULL,
                term_structure_id TEXT NOT NULL,
                market_payload TEXT NOT NULL,
                calibration_payload TEXT NOT NULL
            )
            """
        )
        connection.commit()
    except (OSError, sqlite3.Error) as exc:
        raise MarketSnapshotStoreError("market snapshot store is unavailable") from exc
    return connection


def _snapshot_metadata(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "snapshot_id": row["snapshot_id"],
        "created_at": row["created_at"],
        "symbol": row["symbol"],
        "underlier_type": row["underlier_type"],
        "currency": row["currency"],
        "maturity_years": row["maturity_years"],
        "market_data_time": row["market_data_time"],
        "term_structure_id": row["term_structure_id"],
        "immutable": True,
    }


def save_research_market_snapshot(
    *,
    market: Mapping[str, Any],
    calibration: Mapping[str, Any],
    db_path: Path | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    snapshot_id = str(calibration.get("calibration_id", ""))
    if not SNAPSHOT_ID_PATTERN.fullmatch(snapshot_id):
        raise MarketSnapshotStoreError("calibration_id is not a valid snapshot ID")
    term_structure_id = str(market.get("term_structure_id", ""))
    if not SNAPSHOT_ID_PATTERN.fullmatch(term_structure_id):
        raise MarketSnapshotStoreError("term_structure_id is invalid")
    segments = market.get("segments")
    if not isinstance(segments, list) or not segments:
        raise MarketSnapshotStoreError("market snapshot has no term-structure segments")
    maturity_years = float(segments[-1]["end_time_years"])
    created_at = (
        (now or datetime.now(timezone.utc)).astimezone(timezone.utc).isoformat()
    )
    market_payload = json.dumps(dict(market), sort_keys=True, allow_nan=False)
    calibration_payload = json.dumps(dict(calibration), sort_keys=True, allow_nan=False)

    connection = _connect(db_path)
    try:
        connection.execute(
            """
            INSERT OR IGNORE INTO research_market_snapshots (
                snapshot_id,
                created_at,
                symbol,
                underlier_type,
                currency,
                maturity_years,
                market_data_time,
                term_structure_id,
                market_payload,
                calibration_payload
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                snapshot_id,
                created_at,
                str(market["symbol"]),
                str(market["underlier_type"]),
                str(market["currency"]),
                maturity_years,
                str(market["market_data_time"]),
                term_structure_id,
                market_payload,
                calibration_payload,
            ),
        )
        connection.commit()
        row = connection.execute(
            "SELECT * FROM research_market_snapshots WHERE snapshot_id = ?",
            (snapshot_id,),
        ).fetchone()
    except sqlite3.Error as exc:
        raise MarketSnapshotStoreError(
            "market snapshot could not be persisted"
        ) from exc
    finally:
        connection.close()
    if row is None:
        raise MarketSnapshotStoreError("market snapshot could not be persisted")
    return _snapshot_metadata(row)


def get_research_market_snapshot(
    snapshot_id: str,
    *,
    db_path: Path | None = None,
) -> dict[str, Any] | None:
    if not SNAPSHOT_ID_PATTERN.fullmatch(str(snapshot_id)):
        return None
    connection = _connect(db_path)
    try:
        row = connection.execute(
            "SELECT * FROM research_market_snapshots WHERE snapshot_id = ?",
            (str(snapshot_id),),
        ).fetchone()
    except sqlite3.Error as exc:
        raise MarketSnapshotStoreError("market snapshot store is unavailable") from exc
    finally:
        connection.close()
    if row is None:
        return None
    return {
        **_snapshot_metadata(row),
        "market_term_structure": json.loads(row["market_payload"]),
        "market_calibration": json.loads(row["calibration_payload"]),
    }


def list_research_market_snapshots(
    *,
    limit: int = 20,
    db_path: Path | None = None,
) -> list[dict[str, Any]]:
    bounded_limit = max(1, min(int(limit), 100))
    connection = _connect(db_path)
    try:
        rows = connection.execute(
            """
            SELECT *
            FROM research_market_snapshots
            ORDER BY created_at DESC, snapshot_id DESC
            LIMIT ?
            """,
            (bounded_limit,),
        ).fetchall()
    except sqlite3.Error as exc:
        raise MarketSnapshotStoreError("market snapshot store is unavailable") from exc
    finally:
        connection.close()
    return [_snapshot_metadata(row) for row in rows]
