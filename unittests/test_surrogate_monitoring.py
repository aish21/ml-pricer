from datetime import datetime, timezone

import pytest

from app.services.surrogate_monitoring import (
    SurrogateMonitoringSettings,
    get_surrogate_monitoring_status,
    get_surrogate_monitoring_summary,
    record_surrogate_shadow_observation,
    replay_surrogate_shadow_observations,
)
from src.final.market import EquityMarketSegment, EquityMarketTermStructure


TERMS = {
    "maturity_years": 1.0,
    "autocall_barrier_frac": 1.05,
    "coupon_barrier_frac": 0.8,
    "coupon_rate": 0.02,
    "knock_in_frac": 0.6,
    "obs_count": 6,
}


def make_market() -> EquityMarketTermStructure:
    timestamp = datetime(2026, 1, 2, 16, 0, tzinfo=timezone.utc)
    return EquityMarketTermStructure(
        symbol="SPY",
        underlier_type="etf",
        currency="USD",
        valuation_time=timestamp,
        market_data_time=timestamp,
        spot=100.0,
        segments=(EquityMarketSegment(1.0, 0.03, 0.01, 0.18),),
        calendar="XNYS",
        day_count="ACT/365F",
        source="monitoring-test",
    )


def enabled_settings(tmp_path) -> SurrogateMonitoringSettings:
    return SurrogateMonitoringSettings(
        enabled=True,
        db_path=tmp_path / "shadow.sqlite3",
    )


def shadow_result(*, absolute_error=0.01):
    return {
        "status": "success",
        "artifact_id": f"sha256:{'a' * 64}",
        "model_version": "phoenix-surrogate-payoff-aware-v7",
        "surrogate_price": 1.0 + absolute_error,
        "absolute_error": absolute_error,
        "error_to_reference_standard_error": absolute_error / 0.01,
        "latency_ms": 3,
        "input_diagnostics": {
            "maximum_standardized_feature_distance": 2.5,
            "features_above_four_sigma": [],
        },
    }


def test_monitoring_records_and_summarizes_shadow_observation(tmp_path):
    settings = enabled_settings(tmp_path)

    recorded = record_surrogate_shadow_observation(
        market=make_market(),
        terms=TERMS,
        contract_reference_spot=100.0,
        reference_price=1.0,
        reference_standard_error=0.01,
        shadow_result=shadow_result(),
        settings=settings,
    )
    summary = get_surrogate_monitoring_summary(settings=settings)

    assert recorded is True
    assert summary["overall"]["n_observations"] == 1
    assert summary["overall"]["mae"] == pytest.approx(0.01)
    assert summary["overall"]["within_two_reference_se_fraction"] == 1.0
    assert summary["by_market_regime"]["low_vol"]["n_successful"] == 1
    assert summary["by_moneyness_region"]["autocall"]["n_successful"] == 1
    assert summary["feature_drift"]["above_four_sigma_fraction"] == 0.0
    assert get_surrogate_monitoring_status(settings)["observation_count"] == 1


def test_monitoring_is_opt_in_and_does_not_create_database(tmp_path):
    settings = SurrogateMonitoringSettings(
        enabled=False,
        db_path=tmp_path / "disabled.sqlite3",
    )

    assert (
        record_surrogate_shadow_observation(
            market=make_market(),
            terms=TERMS,
            contract_reference_spot=100.0,
            reference_price=1.0,
            reference_standard_error=0.01,
            shadow_result=shadow_result(),
            settings=settings,
        )
        is False
    )
    assert not settings.db_path.exists()


def test_replay_uses_stored_market_and_current_surrogate(monkeypatch, tmp_path):
    settings = enabled_settings(tmp_path)
    record_surrogate_shadow_observation(
        market=make_market(),
        terms=TERMS,
        contract_reference_spot=100.0,
        reference_price=1.0,
        reference_standard_error=0.01,
        shadow_result=shadow_result(absolute_error=0.02),
        settings=settings,
    )

    def evaluate(**kwargs):
        assert kwargs["market"].symbol == "SPY"
        assert kwargs["terms"] == TERMS
        return {
            "status": "success",
            "artifact_id": f"sha256:{'b' * 64}",
            "absolute_error": 0.005,
        }

    monkeypatch.setattr(
        "app.services.surrogate_service.evaluate_surrogate_shadow", evaluate
    )

    replay = replay_surrogate_shadow_observations(
        limit=10,
        monitoring_settings=settings,
        surrogate_settings=object(),
    )

    assert replay["n_replayed"] == 1
    assert replay["n_successful"] == 1
    assert replay["artifact_id"] == f"sha256:{'b' * 64}"
    assert replay["mae"] == pytest.approx(0.005)
    assert replay["observations"][0]["original_artifact_id"] == (f"sha256:{'a' * 64}")
