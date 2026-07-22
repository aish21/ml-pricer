"""Bounded telemetry and readiness evidence for expanded-product shadows."""

from __future__ import annotations

import json
import math
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from app.services.product_registry import REPO_ROOT
from src.final.barrier_reverse_convertible import BarrierReverseConvertibleV1Contract
from src.final.market import EquityMarketSegment, EquityMarketTermStructure
from src.final.phoenix_contract import PhoenixSingleV3Contract


SCHEMA_VERSION = "expanded-shadow-observation-v1"
DEFAULT_DB = REPO_ROOT / "data" / "expanded_shadow_observations.sqlite3"
PRODUCTS = ("phoenix_v3", "barrier_reverse_convertible")


class ExpandedShadowMonitoringError(RuntimeError):
    pass


def _enabled() -> bool:
    return os.getenv(
        "EXPANDED_SURROGATE_TELEMETRY_ENABLED", "false"
    ).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _db_path() -> Path:
    return Path(os.getenv("EXPANDED_SURROGATE_TELEMETRY_DB", str(DEFAULT_DB)))


def _max_rows() -> int:
    try:
        value = int(os.getenv("EXPANDED_SURROGATE_TELEMETRY_MAX_ROWS", "100000"))
    except ValueError as exc:
        raise ExpandedShadowMonitoringError(
            "telemetry max rows must be an integer"
        ) from exc
    if not 100 <= value <= 10_000_000:
        raise ExpandedShadowMonitoringError("telemetry max rows is invalid")
    return value


def _connect() -> sqlite3.Connection:
    try:
        path = _db_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(path, timeout=5.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=5000")
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS expanded_shadow_observations (
                observation_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                schema_version TEXT NOT NULL,
                product_key TEXT NOT NULL,
                contract_version TEXT NOT NULL,
                symbol TEXT NOT NULL,
                market_date TEXT NOT NULL,
                artifact_id TEXT,
                status TEXT NOT NULL,
                reference_price REAL NOT NULL,
                reference_standard_error REAL NOT NULL,
                surrogate_price REAL,
                absolute_error REAL,
                relative_error REAL,
                latency_ms REAL,
                reference_latency_ms REAL NOT NULL,
                domain_utilization REAL,
                market_regime TEXT NOT NULL,
                payoff_region TEXT NOT NULL,
                market_payload TEXT NOT NULL,
                contract_payload TEXT NOT NULL,
                shadow_payload TEXT NOT NULL
            )
            """
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_expanded_shadow_product_created "
            "ON expanded_shadow_observations(product_key, created_at DESC)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_expanded_shadow_artifact "
            "ON expanded_shadow_observations(artifact_id, created_at DESC)"
        )
        connection.commit()
        return connection
    except (OSError, sqlite3.Error) as exc:
        raise ExpandedShadowMonitoringError(
            "expanded shadow database is unavailable"
        ) from exc


def _finite(value: Any, *, optional: bool = False) -> float | None:
    if value is None and optional:
        return None
    if isinstance(value, bool):
        raise ExpandedShadowMonitoringError("telemetry value must be numeric")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ExpandedShadowMonitoringError("telemetry value must be finite")
    return parsed


def _regime(market: EquityMarketTermStructure, maturity: float) -> str:
    volatility = market.equivalent_flat_parameters(maturity)["volatility"]
    if volatility <= 0.20:
        return "low_vol"
    if volatility <= 0.38:
        return "normal_vol"
    return "high_vol"


def _region(
    product_key: str,
    market: EquityMarketTermStructure,
    contract: PhoenixSingleV3Contract | BarrierReverseConvertibleV1Contract,
) -> str:
    spot = market.spot / contract.reference_level
    if product_key == "phoenix_v3" and isinstance(contract, PhoenixSingleV3Contract):
        levels = {
            "first_autocall": contract.autocall_barrier_fracs[0],
            "final_autocall": contract.autocall_barrier_fracs[-1],
            "coupon": contract.coupon_barrier_frac,
            "knock_in": contract.knock_in_frac,
        }
    elif isinstance(contract, BarrierReverseConvertibleV1Contract):
        levels = {"strike": contract.strike_frac, "knock_in": contract.knock_in_frac}
    else:
        return "unknown"
    name, distance = min(
        ((name, abs(spot - level)) for name, level in levels.items()),
        key=lambda item: item[1],
    )
    return name if distance <= 0.05 else "broad"


def record_expanded_shadow_observation(
    *,
    product_key: str,
    market: EquityMarketTermStructure,
    contract: PhoenixSingleV3Contract | BarrierReverseConvertibleV1Contract,
    reference_price: float,
    reference_standard_error: float,
    reference_latency_ms: float,
    shadow_result: Mapping[str, Any],
) -> bool:
    if not _enabled() or shadow_result.get("status") in {"disabled", "not_sampled"}:
        return False
    try:
        created = datetime.now(timezone.utc)
        row = (
            uuid.uuid4().hex,
            created.isoformat(),
            SCHEMA_VERSION,
            product_key,
            contract.contract_version,
            market.symbol,
            market.market_data_time.date().isoformat(),
            shadow_result.get("artifact_id"),
            str(shadow_result.get("status", "unknown")),
            _finite(reference_price),
            _finite(reference_standard_error),
            _finite(shadow_result.get("surrogate_price"), optional=True),
            _finite(shadow_result.get("absolute_error"), optional=True),
            _finite(shadow_result.get("relative_error"), optional=True),
            _finite(shadow_result.get("latency_ms"), optional=True),
            _finite(reference_latency_ms),
            _finite(shadow_result.get("maximum_domain_utilization"), optional=True),
            _regime(market, contract.maturity_years),
            _region(product_key, market, contract),
            json.dumps(market.to_dict(), sort_keys=True, separators=(",", ":")),
            json.dumps(contract.to_dict(), sort_keys=True, separators=(",", ":")),
            json.dumps(dict(shadow_result), sort_keys=True, separators=(",", ":")),
        )
        with _connect() as connection:
            connection.execute(
                """
                INSERT INTO expanded_shadow_observations VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
                """,
                row,
            )
            connection.execute(
                """
                DELETE FROM expanded_shadow_observations WHERE observation_id IN (
                    SELECT observation_id FROM expanded_shadow_observations
                    ORDER BY created_at DESC LIMIT -1 OFFSET ?
                )
                """,
                (_max_rows(),),
            )
        return True
    except Exception:
        return False


def _stats(rows: list[sqlite3.Row]) -> dict[str, Any]:
    successes = [row for row in rows if row["status"] == "success"]
    errors = np.asarray(
        [
            float(row["absolute_error"])
            for row in successes
            if row["absolute_error"] is not None
        ],
        dtype=np.float64,
    )
    latencies = np.asarray(
        [
            float(row["latency_ms"])
            for row in successes
            if row["latency_ms"] is not None
        ],
        dtype=np.float64,
    )
    statuses: dict[str, int] = {}
    for row in rows:
        statuses[str(row["status"])] = statuses.get(str(row["status"]), 0) + 1
    span_days = None
    if rows:
        first = datetime.fromisoformat(str(rows[-1]["created_at"]))
        last = datetime.fromisoformat(str(rows[0]["created_at"]))
        span_days = max(0.0, (last - first).total_seconds() / 86_400.0)
    return {
        "n_observations": len(rows),
        "n_success": len(successes),
        "success_rate": len(successes) / len(rows) if rows else None,
        "statuses": statuses,
        "mae": float(np.mean(errors)) if errors.size else None,
        "p95_absolute_error": float(np.quantile(errors, 0.95)) if errors.size else None,
        "median_latency_ms": float(np.median(latencies)) if latencies.size else None,
        "p95_latency_ms": (
            float(np.quantile(latencies, 0.95)) if latencies.size else None
        ),
        "symbols": len({str(row["symbol"]) for row in rows}),
        "market_dates": len({str(row["market_date"]) for row in rows}),
        "first_observation_at": rows[-1]["created_at"] if rows else None,
        "last_observation_at": rows[0]["created_at"] if rows else None,
        "observation_span_days": span_days,
    }


def _slice_counts(rows: list[sqlite3.Row], field: str) -> dict[str, int]:
    values: dict[str, int] = {}
    for row in rows:
        name = str(row[field])
        values[name] = values.get(name, 0) + 1
    return values


def get_expanded_shadow_summary(limit: int = 5_000) -> dict[str, Any]:
    limit = max(1, min(int(limit), 100_000))
    if not _db_path().exists():
        return {
            "available": False,
            "reason": "no expanded shadow observations yet",
            "products": {},
        }
    try:
        with _connect() as connection:
            rows = connection.execute(
                "SELECT * FROM expanded_shadow_observations ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
    except ExpandedShadowMonitoringError:
        raise
    products: dict[str, Any] = {}
    for key in PRODUCTS:
        product_rows = [row for row in rows if row["product_key"] == key]
        products[key] = {
            **_stats(product_rows),
            "market_regimes": _slice_counts(product_rows, "market_regime"),
            "payoff_regions": _slice_counts(product_rows, "payoff_region"),
            "artifact_ids": sorted(
                {str(row["artifact_id"]) for row in product_rows if row["artifact_id"]}
            ),
        }
    return {
        "available": bool(rows),
        "schema_version": SCHEMA_VERSION,
        "limit": limit,
        "products": products,
    }


def _check(
    value: float | int | None,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> dict[str, Any]:
    passed = value is not None
    if passed and minimum is not None:
        passed = float(value) >= minimum
    if passed and maximum is not None:
        passed = float(value) <= maximum
    result: dict[str, Any] = {"value": value, "passed": bool(passed)}
    if minimum is not None:
        result["minimum"] = minimum
    if maximum is not None:
        result["maximum"] = maximum
    return result


def get_expanded_shadow_readiness(limit: int = 100_000) -> dict[str, Any]:
    from app.services.expanded_shadow_service import get_expanded_shadow_status

    summary = get_expanded_shadow_summary(limit)
    runtime_products = get_expanded_shadow_status().get("products") or {}
    decisions: dict[str, Any] = {}
    required_regions = {
        "phoenix_v3": {
            "broad",
            "coupon",
            "knock_in",
            "first_autocall",
            "final_autocall",
        },
        "barrier_reverse_convertible": {"broad", "strike", "knock_in"},
    }
    for key in PRODUCTS:
        observed = summary.get("products", {}).get(key, {})
        regions = observed.get("payoff_regions") or {}
        regimes = observed.get("market_regimes") or {}
        observation_count = int(observed.get("n_observations") or 0)
        non_success = observation_count - int(observed.get("n_success") or 0)
        failure_rate = non_success / observation_count if observation_count else None
        artifact_ids = observed.get("artifact_ids") or []
        current_artifact_id = (runtime_products.get(key) or {}).get("artifact_id")
        checks = {
            "observations": _check(observed.get("n_observations"), minimum=2_000),
            "successful_observations": _check(observed.get("n_success"), minimum=1_800),
            "symbols": _check(observed.get("symbols"), minimum=10),
            "market_dates": _check(observed.get("market_dates"), minimum=10),
            "observation_span_days": _check(
                observed.get("observation_span_days"), minimum=14
            ),
            "pinned_artifact_integrity": _check(
                int(artifact_ids == [current_artifact_id]), minimum=1
            ),
            "mae": _check(observed.get("mae"), maximum=0.015),
            "p95_absolute_error": _check(
                observed.get("p95_absolute_error"), maximum=0.04
            ),
            "p95_latency_ms": _check(observed.get("p95_latency_ms"), maximum=5.0),
            "success_rate": _check(observed.get("success_rate"), minimum=0.995),
            "failure_rate": _check(failure_rate, maximum=0.005),
            "market_regime_coverage": _check(
                min(
                    (
                        regimes.get(name, 0)
                        for name in ("low_vol", "normal_vol", "high_vol")
                    ),
                    default=0,
                ),
                minimum=100,
            ),
            "payoff_region_coverage": _check(
                min(
                    (regions.get(name, 0) for name in required_regions[key]), default=0
                ),
                minimum=100,
            ),
        }
        ready = all(check["passed"] for check in checks.values())
        decisions[key] = {
            "decision": "ready_for_human_review" if ready else "insufficient_evidence",
            "ready_for_human_review": ready,
            "runtime_eligible": False,
            "automatic_promotion_permitted": False,
            "checks": checks,
        }
    return {
        "policy_version": "expanded-shadow-readiness-v1",
        "automatic_promotion_permitted": False,
        "products": decisions,
    }


def get_expanded_shadow_series(limit: int = 250) -> dict[str, Any]:
    if not _db_path().exists():
        return {"available": False, "observations": []}
    with _connect() as connection:
        rows = connection.execute(
            """
            SELECT created_at, product_key, symbol, artifact_id, status,
                   reference_price, surrogate_price, absolute_error, latency_ms,
                   market_regime, payoff_region
            FROM expanded_shadow_observations ORDER BY created_at DESC LIMIT ?
            """,
            (max(1, min(int(limit), 5_000)),),
        ).fetchall()
    return {
        "available": bool(rows),
        "observations": [dict(row) for row in reversed(rows)],
    }


def _market(payload: Mapping[str, Any]) -> EquityMarketTermStructure:
    def timestamp(name: str) -> datetime:
        return datetime.fromisoformat(str(payload[name]).replace("Z", "+00:00"))

    return EquityMarketTermStructure(
        symbol=str(payload["symbol"]),
        underlier_type=str(payload["underlier_type"]),
        currency=str(payload["currency"]),
        valuation_time=timestamp("valuation_time"),
        market_data_time=timestamp("market_data_time"),
        spot=float(payload["spot"]),
        segments=tuple(
            EquityMarketSegment(
                end_time_years=float(item["end_time_years"]),
                risk_free_rate=float(item["risk_free_rate"]),
                dividend_yield=float(item["dividend_yield"]),
                volatility=float(item["volatility"]),
            )
            for item in payload["segments"]
        ),
        calendar=str(payload["calendar"]),
        day_count=str(payload["day_count"]),
        source=str(payload["source"]),
    )


def _contract(
    product_key: str, payload: Mapping[str, Any]
) -> PhoenixSingleV3Contract | BarrierReverseConvertibleV1Contract:
    if product_key == "phoenix_v3":
        return PhoenixSingleV3Contract(
            reference_level=float(payload["reference_level"]),
            maturity_years=float(payload["maturity_years"]),
            observation_times_years=tuple(payload["observation_times_years"]),
            autocall_barrier_fracs=tuple(payload["autocall_barrier_fracs"]),
            coupon_barrier_frac=float(payload["coupon_barrier_frac"]),
            coupon_rate=float(payload["coupon_rate"]),
            knock_in_frac=float(payload["knock_in_frac"]),
            prior_knock_in_breached=bool(payload["prior_knock_in_breached"]),
            memory_coupon=bool(payload["memory_coupon"]),
            unpaid_coupon_count=int(payload["unpaid_coupon_count"]),
        )
    return BarrierReverseConvertibleV1Contract(
        reference_level=float(payload["reference_level"]),
        maturity_years=float(payload["maturity_years"]),
        coupon_times_years=tuple(payload["coupon_times_years"]),
        coupon_rate_per_period=float(payload["coupon_rate_per_period"]),
        strike_frac=float(payload["strike_frac"]),
        knock_in_frac=float(payload["knock_in_frac"]),
        prior_knock_in_breached=bool(payload["prior_knock_in_breached"]),
    )


def replay_expanded_shadow_observations(
    product_key: str, limit: int = 100
) -> dict[str, Any]:
    from app.services.expanded_shadow_service import evaluate_expanded_shadow

    if product_key not in PRODUCTS:
        raise ExpandedShadowMonitoringError("unknown expanded shadow product")
    if not _db_path().exists():
        return {
            "product_key": product_key,
            "requested": 0,
            "replayed": 0,
            "results": [],
        }
    with _connect() as connection:
        rows = connection.execute(
            """
            SELECT * FROM expanded_shadow_observations
            WHERE product_key = ? AND status = 'success'
            ORDER BY created_at DESC LIMIT ?
            """,
            (product_key, max(1, min(int(limit), 1_000))),
        ).fetchall()
    results = []
    for row in rows:
        market = _market(json.loads(row["market_payload"]))
        contract = _contract(product_key, json.loads(row["contract_payload"]))
        replay = evaluate_expanded_shadow(
            product_key=product_key,
            market=market,
            contract=contract,
            reference_price=float(row["reference_price"]),
            reference_standard_error=float(row["reference_standard_error"]),
            reference_latency_ms=float(row["reference_latency_ms"]),
            force=True,
        )
        results.append(
            {
                "observation_id": row["observation_id"],
                "stored_artifact_id": row["artifact_id"],
                "replay_artifact_id": replay.get("artifact_id"),
                "status": replay.get("status"),
                "stored_surrogate_price": row["surrogate_price"],
                "replay_surrogate_price": replay.get("surrogate_price"),
            }
        )
    return {
        "product_key": product_key,
        "requested": len(rows),
        "replayed": len(results),
        "results": results,
    }


def get_expanded_shadow_monitoring_status() -> dict[str, Any]:
    path = _db_path()
    row_count = 0
    if path.exists():
        try:
            with _connect() as connection:
                row_count = int(
                    connection.execute(
                        "SELECT COUNT(*) FROM expanded_shadow_observations"
                    ).fetchone()[0]
                )
        except ExpandedShadowMonitoringError:
            pass
    return {
        "version": SCHEMA_VERSION,
        "enabled": _enabled(),
        "database_available": path.exists(),
        "row_count": row_count,
        "max_rows": _max_rows(),
    }
