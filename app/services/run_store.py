import json
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.services.product_registry import REPO_ROOT


DEFAULT_RUN_STORE = REPO_ROOT / "data" / "pricing_runs.sqlite3"


def get_run_store_path() -> Path:
    configured = os.getenv("MODEL_RUN_STORE_FILE")
    return Path(configured) if configured else DEFAULT_RUN_STORE


def _connect(db_path: Optional[Path] = None) -> sqlite3.Connection:
    path = Path(db_path) if db_path else get_run_store_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    init_db(conn)
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS pricing_runs (
            run_id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            product_key TEXT NOT NULL,
            request_payload TEXT NOT NULL,
            result_payload TEXT NOT NULL,
            model TEXT,
            latency_ms INTEGER,
            run_type TEXT NOT NULL DEFAULT 'price',
            parent_run_id TEXT
        )
        """
    )
    columns = {
        row[1]
        for row in conn.execute("PRAGMA table_info(pricing_runs)").fetchall()
    }
    if "run_type" not in columns:
        conn.execute(
            "ALTER TABLE pricing_runs ADD COLUMN run_type TEXT NOT NULL DEFAULT 'price'"
        )
    if "parent_run_id" not in columns:
        conn.execute("ALTER TABLE pricing_runs ADD COLUMN parent_run_id TEXT")
    conn.commit()


def _new_run_id(now: Optional[datetime] = None) -> str:
    current = now or datetime.now(timezone.utc)
    stamp = current.strftime("%Y%m%d-%H%M%S")
    suffix = uuid.uuid4().hex[:6]
    return f"run_{stamp}_{suffix}"


def save_run(
    product_key: str,
    request_payload: Dict[str, Any],
    result_payload: Dict[str, Any],
    db_path: Optional[Path] = None,
    run_type: str = "price",
    parent_run_id: Optional[str] = None,
) -> str:
    created_at = datetime.now(timezone.utc).isoformat()
    run_id = _new_run_id()
    model = result_payload.get("model")
    latency_ms = result_payload.get("latency_ms")

    conn = _connect(db_path)
    try:
        conn.execute(
            """
            INSERT INTO pricing_runs (
                run_id,
                created_at,
                product_key,
                request_payload,
                result_payload,
                model,
                latency_ms,
                run_type,
                parent_run_id
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                created_at,
                product_key,
                json.dumps(request_payload, sort_keys=True),
                json.dumps(result_payload, sort_keys=True),
                model,
                latency_ms,
                run_type,
                parent_run_id,
            ),
        )
        conn.commit()
    finally:
        conn.close()
    return run_id


def _row_to_run(row: sqlite3.Row) -> Dict[str, Any]:
    return {
        "run_id": row["run_id"],
        "created_at": row["created_at"],
        "product_key": row["product_key"],
        "request_payload": json.loads(row["request_payload"]),
        "result_payload": json.loads(row["result_payload"]),
        "model": row["model"],
        "latency_ms": row["latency_ms"],
        "run_type": row["run_type"],
        "parent_run_id": row["parent_run_id"],
    }


def get_run(run_id: str, db_path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    conn = _connect(db_path)
    try:
        row = conn.execute(
            "SELECT * FROM pricing_runs WHERE run_id = ?",
            (run_id,),
        ).fetchone()
    finally:
        conn.close()
    return _row_to_run(row) if row else None


def list_recent_runs(
    limit: int = 10, db_path: Optional[Path] = None
) -> List[Dict[str, Any]]:
    conn = _connect(db_path)
    try:
        rows = conn.execute(
            """
            SELECT * FROM pricing_runs
            ORDER BY created_at DESC, run_id DESC
            LIMIT ?
            """,
            (int(limit),),
        ).fetchall()
    finally:
        conn.close()
    return [_row_to_run(row) for row in rows]
