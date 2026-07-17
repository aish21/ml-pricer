import json
import math
import os
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src.final.market import EquityMarketSegment, EquityMarketTermStructure

from app.services.product_registry import REPO_ROOT


SHADOW_OBSERVATION_SCHEMA_VERSION = "phoenix-shadow-observation-v1"
DEFAULT_SURROGATE_MONITORING_DB = (
    REPO_ROOT / "data" / "surrogate_shadow_observations.sqlite3"
)


class SurrogateMonitoringError(RuntimeError):
    pass


@dataclass(frozen=True)
class SurrogateMonitoringSettings:
    enabled: bool
    db_path: Path
    max_rows: int = 100_000

    def __post_init__(self) -> None:
        if self.max_rows < 100 or self.max_rows > 10_000_000:
            raise SurrogateMonitoringError("surrogate monitoring max_rows is invalid")

    @classmethod
    def from_env(cls) -> "SurrogateMonitoringSettings":
        enabled = (
            os.getenv("PHOENIX_SURROGATE_TELEMETRY_ENABLED", "false").strip().lower()
        )
        raw_max_rows = os.getenv("PHOENIX_SURROGATE_TELEMETRY_MAX_ROWS", "100000")
        try:
            max_rows = int(raw_max_rows)
        except ValueError as exc:
            raise SurrogateMonitoringError(
                "PHOENIX_SURROGATE_TELEMETRY_MAX_ROWS must be an integer"
            ) from exc
        return cls(
            enabled=enabled in {"1", "true", "yes", "on"},
            db_path=Path(
                os.getenv(
                    "PHOENIX_SURROGATE_TELEMETRY_DB",
                    str(DEFAULT_SURROGATE_MONITORING_DB),
                )
            ),
            max_rows=max_rows,
        )


def _connect(path: Path) -> sqlite3.Connection:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(path, timeout=5.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 5000")
        _initialize(connection)
        return connection
    except (OSError, sqlite3.Error) as exc:
        raise SurrogateMonitoringError(
            "surrogate monitoring database is unavailable"
        ) from exc


def _initialize(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS surrogate_shadow_observations (
            observation_id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            schema_version TEXT NOT NULL,
            symbol TEXT NOT NULL,
            market_data_time TEXT NOT NULL,
            artifact_id TEXT,
            model_version TEXT NOT NULL,
            status TEXT NOT NULL,
            reference_price REAL NOT NULL,
            reference_standard_error REAL NOT NULL,
            surrogate_price REAL,
            absolute_error REAL,
            error_to_reference_standard_error REAL,
            latency_ms INTEGER,
            maximum_standardized_feature_distance REAL,
            market_regime TEXT NOT NULL,
            moneyness_region TEXT NOT NULL,
            contract_reference_spot REAL NOT NULL,
            market_payload TEXT NOT NULL,
            terms_payload TEXT NOT NULL
        )
        """
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_surrogate_shadow_created
        ON surrogate_shadow_observations(created_at DESC)
        """
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_surrogate_shadow_artifact
        ON surrogate_shadow_observations(artifact_id, created_at DESC)
        """
    )
    connection.commit()


def _finite_number(value: Any, name: str, *, allow_none: bool = False) -> float | None:
    if value is None and allow_none:
        return None
    if isinstance(value, bool):
        raise SurrogateMonitoringError(f"{name} must be numeric")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise SurrogateMonitoringError(f"{name} must be numeric") from exc
    if not math.isfinite(parsed):
        raise SurrogateMonitoringError(f"{name} must be finite")
    return parsed


def _classify_market_regime(
    market: EquityMarketTermStructure, maturity_years: float
) -> str:
    average_volatility = math.sqrt(
        market.integrated_variance(0.0, maturity_years) / maturity_years
    )
    if average_volatility <= 0.20:
        return "low_vol"
    if average_volatility <= 0.38:
        return "normal"
    if average_volatility <= 0.65:
        return "high_vol"
    return "crisis"


def _classify_moneyness_region(
    *,
    market: EquityMarketTermStructure,
    terms: Mapping[str, Any],
    reference_spot: float,
) -> str:
    distances = {
        "knock_in": abs(
            math.log(market.spot / (reference_spot * float(terms["knock_in_frac"])))
        ),
        "coupon": abs(
            math.log(
                market.spot / (reference_spot * float(terms["coupon_barrier_frac"]))
            )
        ),
        "autocall": abs(
            math.log(
                market.spot / (reference_spot * float(terms["autocall_barrier_frac"]))
            )
        ),
    }
    region, distance = min(distances.items(), key=lambda item: item[1])
    return region if distance <= 0.12 else "broad"


def record_surrogate_shadow_observation(
    *,
    market: EquityMarketTermStructure,
    terms: Mapping[str, Any],
    contract_reference_spot: float,
    reference_price: float,
    reference_standard_error: float,
    shadow_result: Mapping[str, Any],
    settings: SurrogateMonitoringSettings | None = None,
) -> bool:
    active = settings or SurrogateMonitoringSettings.from_env()
    if not active.enabled:
        return False
    if not isinstance(market, EquityMarketTermStructure):
        raise SurrogateMonitoringError("monitoring market is invalid")
    if not isinstance(shadow_result, Mapping):
        raise SurrogateMonitoringError("shadow result must be an object")
    reference_spot = _finite_number(contract_reference_spot, "contract_reference_spot")
    reference_value = _finite_number(reference_price, "reference_price")
    reference_se = _finite_number(reference_standard_error, "reference_standard_error")
    if reference_spot is None or reference_spot <= 0.0:
        raise SurrogateMonitoringError("contract_reference_spot must be positive")
    if reference_value is None or reference_se is None or reference_se < 0.0:
        raise SurrogateMonitoringError("reference values are invalid")
    maturity = _finite_number(terms.get("maturity_years"), "maturity_years")
    if maturity is None or maturity <= 0.0:
        raise SurrogateMonitoringError("maturity_years must be positive")

    status = str(shadow_result.get("status", "unknown"))
    if status not in {"success", "out_of_domain", "unavailable", "error"}:
        raise SurrogateMonitoringError("shadow status is invalid")
    surrogate_price = _finite_number(
        shadow_result.get("surrogate_price"), "surrogate_price", allow_none=True
    )
    absolute_error = _finite_number(
        shadow_result.get("absolute_error"), "absolute_error", allow_none=True
    )
    error_to_se = _finite_number(
        shadow_result.get("error_to_reference_standard_error"),
        "error_to_reference_standard_error",
        allow_none=True,
    )
    latency = shadow_result.get("latency_ms")
    latency_ms = int(latency) if latency is not None else None
    if latency_ms is not None and latency_ms < 0:
        raise SurrogateMonitoringError("latency_ms must be non-negative")
    diagnostics = shadow_result.get("input_diagnostics")
    maximum_distance = None
    if isinstance(diagnostics, Mapping):
        maximum_distance = _finite_number(
            diagnostics.get("maximum_standardized_feature_distance"),
            "maximum_standardized_feature_distance",
            allow_none=True,
        )

    observation_id = f"shadow_{uuid.uuid4().hex}"
    created_at = datetime.now(timezone.utc).isoformat()
    market_payload = market.to_dict()
    terms_payload = dict(terms)
    connection = _connect(active.db_path)
    try:
        connection.execute(
            """
            INSERT INTO surrogate_shadow_observations (
                observation_id, created_at, schema_version, symbol,
                market_data_time, artifact_id, model_version, status,
                reference_price, reference_standard_error, surrogate_price,
                absolute_error, error_to_reference_standard_error, latency_ms,
                maximum_standardized_feature_distance, market_regime,
                moneyness_region, contract_reference_spot, market_payload,
                terms_payload
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                observation_id,
                created_at,
                SHADOW_OBSERVATION_SCHEMA_VERSION,
                market.symbol,
                market.market_data_time.isoformat(),
                shadow_result.get("artifact_id"),
                str(shadow_result.get("model_version", "unknown")),
                status,
                reference_value,
                reference_se,
                surrogate_price,
                absolute_error,
                error_to_se,
                latency_ms,
                maximum_distance,
                _classify_market_regime(market, maturity),
                _classify_moneyness_region(
                    market=market,
                    terms=terms,
                    reference_spot=reference_spot,
                ),
                reference_spot,
                json.dumps(market_payload, sort_keys=True),
                json.dumps(terms_payload, sort_keys=True),
            ),
        )
        connection.execute(
            """
            DELETE FROM surrogate_shadow_observations
            WHERE observation_id NOT IN (
                SELECT observation_id
                FROM surrogate_shadow_observations
                ORDER BY created_at DESC
                LIMIT ?
            )
            """,
            (active.max_rows,),
        )
        connection.commit()
    except (TypeError, ValueError, sqlite3.Error) as exc:
        raise SurrogateMonitoringError(
            "surrogate shadow observation could not be stored"
        ) from exc
    finally:
        connection.close()
    return True


def _quantile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _error_summary(rows: list[sqlite3.Row]) -> dict[str, Any]:
    successful = [row for row in rows if row["status"] == "success"]
    absolute_errors = [
        float(row["absolute_error"])
        for row in successful
        if row["absolute_error"] is not None
    ]
    error_to_se = [
        float(row["error_to_reference_standard_error"])
        for row in successful
        if row["error_to_reference_standard_error"] is not None
    ]
    latencies = [
        float(row["latency_ms"]) for row in successful if row["latency_ms"] is not None
    ]
    return {
        "n_observations": len(rows),
        "n_successful": len(successful),
        "mae": (
            sum(absolute_errors) / len(absolute_errors) if absolute_errors else None
        ),
        "p95_absolute_error": _quantile(absolute_errors, 0.95),
        "within_two_reference_se_fraction": (
            sum(value <= 2.0 for value in error_to_se) / len(error_to_se)
            if error_to_se
            else None
        ),
        "mean_latency_ms": sum(latencies) / len(latencies) if latencies else None,
    }


def get_surrogate_monitoring_summary(
    *,
    limit: int = 1_000,
    settings: SurrogateMonitoringSettings | None = None,
) -> dict[str, Any]:
    active = settings or SurrogateMonitoringSettings.from_env()
    if not active.enabled:
        return {"enabled": False, "available": False, "reason": "disabled"}
    if limit < 1 or limit > 100_000:
        raise SurrogateMonitoringError("monitoring summary limit is invalid")
    connection = _connect(active.db_path)
    try:
        rows = connection.execute(
            """
            SELECT * FROM surrogate_shadow_observations
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    except sqlite3.Error as exc:
        raise SurrogateMonitoringError("monitoring summary query failed") from exc
    finally:
        connection.close()
    status_counts: dict[str, int] = {}
    for row in rows:
        status_counts[row["status"]] = status_counts.get(row["status"], 0) + 1
    by_regime = {
        regime: _error_summary([row for row in rows if row["market_regime"] == regime])
        for regime in sorted(set(row["market_regime"] for row in rows))
    }
    by_region = {
        region: _error_summary(
            [row for row in rows if row["moneyness_region"] == region]
        )
        for region in sorted(set(row["moneyness_region"] for row in rows))
    }
    distances = [
        float(row["maximum_standardized_feature_distance"])
        for row in rows
        if row["maximum_standardized_feature_distance"] is not None
    ]
    return {
        "enabled": True,
        "available": True,
        "schema_version": SHADOW_OBSERVATION_SCHEMA_VERSION,
        "limit": limit,
        "status_counts": status_counts,
        "overall": _error_summary(rows),
        "by_market_regime": by_regime,
        "by_moneyness_region": by_region,
        "feature_drift": {
            "maximum_standardized_feature_distance": (
                max(distances) if distances else None
            ),
            "above_four_sigma_fraction": (
                sum(value > 4.0 for value in distances) / len(distances)
                if distances
                else None
            ),
        },
        "symbol_count": len(set(row["symbol"] for row in rows)),
        "newest_observation_at": rows[0]["created_at"] if rows else None,
        "oldest_observation_at": rows[-1]["created_at"] if rows else None,
    }


def get_surrogate_monitoring_status(
    settings: SurrogateMonitoringSettings | None = None,
) -> dict[str, Any]:
    active = settings or SurrogateMonitoringSettings.from_env()
    if not active.enabled:
        return {"enabled": False, "available": False, "reason": "disabled"}
    try:
        connection = _connect(active.db_path)
        try:
            count = int(
                connection.execute(
                    "SELECT COUNT(*) FROM surrogate_shadow_observations"
                ).fetchone()[0]
            )
        finally:
            connection.close()
    except SurrogateMonitoringError as exc:
        return {"enabled": True, "available": False, "reason": str(exc)}
    return {
        "enabled": True,
        "available": True,
        "schema_version": SHADOW_OBSERVATION_SCHEMA_VERSION,
        "observation_count": count,
    }


def _market_from_payload(payload: Mapping[str, Any]) -> EquityMarketTermStructure:
    try:
        return EquityMarketTermStructure(
            symbol=str(payload["symbol"]),
            underlier_type=str(payload["underlier_type"]),
            currency=str(payload["currency"]),
            valuation_time=datetime.fromisoformat(
                str(payload["valuation_time"]).replace("Z", "+00:00")
            ),
            market_data_time=datetime.fromisoformat(
                str(payload["market_data_time"]).replace("Z", "+00:00")
            ),
            spot=float(payload["spot"]),
            segments=tuple(
                EquityMarketSegment(
                    end_time_years=float(segment["end_time_years"]),
                    risk_free_rate=float(segment["risk_free_rate"]),
                    dividend_yield=float(segment["dividend_yield"]),
                    volatility=float(segment["volatility"]),
                )
                for segment in payload["segments"]
            ),
            calendar=str(payload["calendar"]),
            day_count=str(payload["day_count"]),
            source=str(payload["source"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise SurrogateMonitoringError("stored market payload is invalid") from exc


def replay_surrogate_shadow_observations(
    *,
    limit: int = 100,
    monitoring_settings: SurrogateMonitoringSettings | None = None,
    surrogate_settings: Any = None,
) -> dict[str, Any]:
    active = monitoring_settings or SurrogateMonitoringSettings.from_env()
    if not active.enabled:
        raise SurrogateMonitoringError("surrogate monitoring is disabled")
    if limit < 1 or limit > 10_000:
        raise SurrogateMonitoringError("replay limit is invalid")
    connection = _connect(active.db_path)
    try:
        rows = connection.execute(
            """
            SELECT * FROM surrogate_shadow_observations
            WHERE status = 'success'
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    finally:
        connection.close()

    from app.services.surrogate_service import evaluate_surrogate_shadow

    replayed = []
    for row in rows:
        try:
            market = _market_from_payload(json.loads(row["market_payload"]))
            terms = json.loads(row["terms_payload"])
            result = evaluate_surrogate_shadow(
                market=market,
                terms=terms,
                contract_reference_spot=float(row["contract_reference_spot"]),
                reference_price=float(row["reference_price"]),
                reference_standard_error=float(row["reference_standard_error"]),
                settings=surrogate_settings,
            )
        except (json.JSONDecodeError, SurrogateMonitoringError, TypeError, ValueError):
            result = {"status": "error", "reason": "stored observation is invalid"}
        replayed.append(
            {
                "observation_id": row["observation_id"],
                "created_at": row["created_at"],
                "symbol": row["symbol"],
                "original_artifact_id": row["artifact_id"],
                "result": result,
            }
        )
    successful_results = [
        item["result"]
        for item in replayed
        if isinstance(item["result"], Mapping)
        and item["result"].get("status") == "success"
    ]
    absolute_errors = [float(result["absolute_error"]) for result in successful_results]
    return {
        "schema_version": SHADOW_OBSERVATION_SCHEMA_VERSION,
        "requested_limit": limit,
        "n_replayed": len(replayed),
        "n_successful": len(successful_results),
        "artifact_id": (
            successful_results[0].get("artifact_id") if successful_results else None
        ),
        "mae": (
            sum(absolute_errors) / len(absolute_errors) if absolute_errors else None
        ),
        "p95_absolute_error": _quantile(absolute_errors, 0.95),
        "observations": replayed,
    }
