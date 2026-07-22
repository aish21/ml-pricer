from app.services.expanded_shadow_monitoring import (
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
    )
    summary = get_expanded_shadow_summary()
    assert summary["available"] is True
    assert summary["products"]["phoenix_v3"]["n_observations"] == 1
    assert summary["products"]["phoenix_v3"]["n_success"] == 1
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
