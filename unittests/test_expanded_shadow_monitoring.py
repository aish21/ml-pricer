import sqlite3

from app.services.expanded_shadow_monitoring import (
    get_expanded_shadow_monitoring_status,
    get_expanded_shadow_readiness,
    get_expanded_shadow_series,
    get_expanded_shadow_summary,
    record_expanded_shadow_observation,
    replay_expanded_shadow_observations,
)
from app.services.expanded_shadow_service import evaluate_expanded_shadow
from unittests.test_expanded_shadow_service import make_market, phoenix_contract


def test_expanded_shadow_telemetry_summary_readiness_and_replay(monkeypatch, tmp_path):
    database = tmp_path / "expanded.sqlite3"
    monkeypatch.setenv("EXPANDED_SURROGATE_TELEMETRY_ENABLED", "true")
    monkeypatch.setenv("EXPANDED_SURROGATE_TELEMETRY_DB", str(database))
    market = make_market()
    contract = phoenix_contract()
    shadow = evaluate_expanded_shadow(
        product_key="phoenix_v3",
        market=market,
        contract=contract,
        reference_price=0.97,
        reference_standard_error=0.003,
        reference_latency_ms=30.0,
        force=True,
    )

    assert record_expanded_shadow_observation(
        product_key="phoenix_v3",
        market=market,
        contract=contract,
        reference_price=0.97,
        reference_standard_error=0.003,
        reference_latency_ms=30.0,
        shadow_result=shadow,
        reference_paths=4_096,
        reference_seed=42,
    )
    summary = get_expanded_shadow_summary()
    assert summary["available"] is True
    assert summary["products"]["phoenix_v3"]["n_observations"] == 1
    assert summary["products"]["phoenix_v3"]["n_success"] == 1
    phoenix = summary["products"]["phoenix_v3"]
    assert phoenix["reliable_reference"]["n_observations"] == 0
    assert phoenix["campaign_evidence"]["n_observations"] == 0
    assert summary["products"]["phoenix_v3"]["observation_sources"] == {
        "interactive": 1
    }
    assert (
        get_expanded_shadow_series()["observations"][0]["product_key"] == "phoenix_v3"
    )
    readiness = get_expanded_shadow_readiness()
    assert readiness["products"]["phoenix_v3"]["decision"] == "insufficient_evidence"
    assert readiness["products"]["phoenix_v3"]["runtime_eligible"] is False
    replay = replay_expanded_shadow_observations("phoenix_v3")
    assert replay["replayed"] == 1
    assert replay["results"][0]["status"] == "success"
    assert replay["results"][0]["replay_surrogate_price"] == shadow["surrogate_price"]


def test_v1_monitoring_database_is_migrated_without_losing_rows(tmp_path):
    database = tmp_path / "legacy.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute(
            """
            CREATE TABLE expanded_shadow_observations (
                observation_id TEXT PRIMARY KEY, created_at TEXT NOT NULL,
                schema_version TEXT NOT NULL, product_key TEXT NOT NULL,
                contract_version TEXT NOT NULL, symbol TEXT NOT NULL,
                market_date TEXT NOT NULL, artifact_id TEXT, status TEXT NOT NULL,
                reference_price REAL NOT NULL,
                reference_standard_error REAL NOT NULL, surrogate_price REAL,
                absolute_error REAL, relative_error REAL, latency_ms REAL,
                reference_latency_ms REAL NOT NULL, domain_utilization REAL,
                market_regime TEXT NOT NULL, payoff_region TEXT NOT NULL,
                market_payload TEXT NOT NULL, contract_payload TEXT NOT NULL,
                shadow_payload TEXT NOT NULL
            )
            """
        )

    status = get_expanded_shadow_monitoring_status(db_path=database)
    with sqlite3.connect(database) as connection:
        columns = {
            row[1]
            for row in connection.execute(
                "PRAGMA table_info(expanded_shadow_observations)"
            ).fetchall()
        }

    assert status["database_available"] is True
    assert {
        "observation_source",
        "campaign_id",
        "case_id",
        "reference_paths",
        "reference_seed",
        "market_snapshot_id",
    } <= columns
