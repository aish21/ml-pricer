import sqlite3
from dataclasses import asdict
from datetime import datetime, timedelta, timezone

from app.services.surrogate_monitoring import (
    SurrogateMonitoringSettings,
    record_surrogate_shadow_observation,
)
from app.services.surrogate_promotion import get_surrogate_promotion_readiness
from app.services.surrogate_promotion_policy import (
    DEFAULT_SHADOW_PROMOTION_POLICY,
    ShadowPromotionPolicy,
)
from src.final.market import EquityMarketSegment, EquityMarketTermStructure
from src.final import surrogate_shadow_readiness


ARTIFACT_ID = f"sha256:{'a' * 64}"
REGIME_VOLS = {
    "low_vol": 0.18,
    "normal": 0.30,
    "high_vol": 0.50,
}
REGION_TERMS = {
    "broad": {
        "autocall_barrier_frac": 1.15,
        "coupon_barrier_frac": 0.70,
        "knock_in_frac": 0.50,
    },
    "knock_in": {
        "autocall_barrier_frac": 1.15,
        "coupon_barrier_frac": 1.05,
        "knock_in_frac": 0.95,
    },
    "coupon": {
        "autocall_barrier_frac": 1.15,
        "coupon_barrier_frac": 1.00,
        "knock_in_frac": 0.60,
    },
    "autocall": {
        "autocall_barrier_frac": 1.05,
        "coupon_barrier_frac": 0.80,
        "knock_in_frac": 0.60,
    },
}


def _settings(tmp_path) -> SurrogateMonitoringSettings:
    return SurrogateMonitoringSettings(
        enabled=True,
        db_path=tmp_path / "shadow.sqlite3",
    )


def _small_policy(**overrides) -> ShadowPromotionPolicy:
    values = asdict(DEFAULT_SHADOW_PROMOTION_POLICY)
    values.update(
        {
            "artifact_id": ARTIFACT_ID,
            "evidence_start_at": "2026-01-01T00:00:00+00:00",
            "minimum_observations": 12,
            "minimum_successful_observations": 12,
            "minimum_distinct_cases": 12,
            "minimum_unique_symbols": 5,
            "minimum_distinct_market_dates": 10,
            "minimum_observation_span_days": 10.0,
            "minimum_successful_per_slice": 1,
            "minimum_successful_per_joint_slice": 1,
            "minimum_success_fraction": 0.90,
            "maximum_out_of_domain_fraction": 0.10,
            "maximum_unavailable_fraction": 0.01,
            "maximum_error_fraction": 0.01,
            "maximum_mae": 0.01,
            "maximum_p95_absolute_error": 0.02,
            "minimum_within_two_reference_se_fraction": 0.80,
            "maximum_regime_mae": 0.01,
            "maximum_moneyness_region_mae": 0.01,
            "maximum_joint_slice_mae": 0.01,
            "maximum_p95_latency_ms": 10.0,
            "maximum_above_four_sigma_fraction": 0.01,
        }
    )
    values.update(overrides)
    return ShadowPromotionPolicy(**values)


def _market(
    *,
    symbol: str,
    volatility: float,
    timestamp: datetime,
) -> EquityMarketTermStructure:
    return EquityMarketTermStructure(
        symbol=symbol,
        underlier_type="etf",
        currency="USD",
        valuation_time=timestamp,
        market_data_time=timestamp,
        spot=100.0,
        segments=(EquityMarketSegment(1.0, 0.03, 0.01, volatility),),
        calendar="XNYS",
        day_count="ACT/365F",
        source="promotion-test",
    )


def _seed_balanced_evidence(
    settings: SurrogateMonitoringSettings,
    *,
    absolute_error: float,
) -> None:
    start = datetime(2026, 2, 1, 16, 0, tzinfo=timezone.utc)
    position = 0
    for volatility in REGIME_VOLS.values():
        for region, region_terms in REGION_TERMS.items():
            timestamp = start + timedelta(days=position)
            terms = {
                "maturity_years": 1.0,
                "coupon_rate": 0.02,
                "obs_count": 6,
                **region_terms,
            }
            record_surrogate_shadow_observation(
                market=_market(
                    symbol=f"TEST{position % 5}",
                    volatility=volatility,
                    timestamp=timestamp,
                ),
                terms=terms,
                contract_reference_spot=(
                    100.0 / region_terms["knock_in_frac"]
                    if region == "knock_in"
                    else 100.0
                ),
                reference_price=1.0,
                reference_standard_error=0.01,
                shadow_result={
                    "status": "success",
                    "artifact_id": ARTIFACT_ID,
                    "target_artifact_id": ARTIFACT_ID,
                    "model_version": "test-model",
                    "surrogate_price": 1.0 + absolute_error,
                    "absolute_error": absolute_error,
                    "error_to_reference_standard_error": (absolute_error / 0.01),
                    "latency_ms": 3,
                    "input_diagnostics": {"maximum_standardized_feature_distance": 2.0},
                },
                settings=settings,
            )
            position += 1

    connection = sqlite3.connect(settings.db_path)
    try:
        observation_ids = [
            row[0]
            for row in connection.execute(
                """
                SELECT observation_id
                FROM surrogate_shadow_observations
                ORDER BY rowid
                """
            ).fetchall()
        ]
        for position, observation_id in enumerate(observation_ids):
            created_at = start + timedelta(days=position)
            connection.execute(
                """
                UPDATE surrogate_shadow_observations
                SET created_at = ?
                WHERE observation_id = ?
                """,
                (created_at.isoformat(), observation_id),
            )
        connection.commit()
    finally:
        connection.close()


def test_disabled_readiness_exposes_policy_without_creating_evidence_store(tmp_path):
    settings = SurrogateMonitoringSettings(
        enabled=False,
        db_path=tmp_path / "disabled.sqlite3",
    )

    report = get_surrogate_promotion_readiness(
        settings=settings,
        policy=_small_policy(),
    )

    assert report["decision"] == "insufficient_evidence"
    assert report["ready_for_review"] is False
    assert report["runtime_eligible"] is False
    assert report["automatic_promotion_permitted"] is False
    assert report["policy"]["policy_id"].startswith("sha256:")
    assert not settings.db_path.exists()


def test_balanced_evidence_can_only_become_ready_for_human_review(tmp_path):
    settings = _settings(tmp_path)
    policy = _small_policy()
    _seed_balanced_evidence(settings, absolute_error=0.005)

    report = get_surrogate_promotion_readiness(
        settings=settings,
        policy=policy,
    )

    assert report["decision"] == "ready_for_review"
    assert report["ready_for_review"] is True
    assert report["runtime_eligible"] is False
    assert report["automatic_promotion_permitted"] is False
    assert report["evidence"]["n_distinct_cases"] == 12
    assert report["checks"]["joint_slice_coverage"]["passed"] is True
    assert all(check["passed"] for check in report["checks"].values())


def test_sufficient_but_inaccurate_evidence_is_not_ready(tmp_path):
    settings = _settings(tmp_path)
    _seed_balanced_evidence(settings, absolute_error=0.03)

    report = get_surrogate_promotion_readiness(
        settings=settings,
        policy=_small_policy(),
    )

    assert report["decision"] == "not_ready"
    assert report["ready_for_review"] is False
    assert report["checks"]["minimum_observations"]["passed"] is True
    assert report["checks"]["mae"]["passed"] is False
    assert report["checks"]["p95_absolute_error"]["passed"] is False


def test_target_artifact_filter_prevents_old_models_from_counting(tmp_path):
    settings = _settings(tmp_path)
    _seed_balanced_evidence(settings, absolute_error=0.005)
    connection = sqlite3.connect(settings.db_path)
    try:
        connection.execute(
            """
            UPDATE surrogate_shadow_observations
            SET target_artifact_id = ?
            WHERE rowid = 1
            """,
            (f"sha256:{'b' * 64}",),
        )
        connection.commit()
    finally:
        connection.close()

    report = get_surrogate_promotion_readiness(
        settings=settings,
        policy=_small_policy(),
    )

    assert report["decision"] == "insufficient_evidence"
    assert report["evidence"]["n_observations"] == 11
    assert report["checks"]["minimum_observations"]["passed"] is False


def test_evidence_id_binds_observation_contents(tmp_path):
    settings = _settings(tmp_path)
    policy = _small_policy()
    _seed_balanced_evidence(settings, absolute_error=0.005)
    before = get_surrogate_promotion_readiness(
        settings=settings,
        policy=policy,
    )["evidence_id"]

    connection = sqlite3.connect(settings.db_path)
    try:
        connection.execute(
            """
            UPDATE surrogate_shadow_observations
            SET latency_ms = latency_ms + 1
            WHERE rowid = 1
            """
        )
        connection.commit()
    finally:
        connection.close()
    after = get_surrogate_promotion_readiness(
        settings=settings,
        policy=policy,
    )["evidence_id"]

    assert after != before


def test_readiness_cli_reports_without_promoting(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(
        surrogate_shadow_readiness,
        "get_surrogate_promotion_readiness",
        lambda **_kwargs: {
            "decision": "insufficient_evidence",
            "runtime_eligible": False,
        },
    )

    result = surrogate_shadow_readiness.main(
        ["--monitoring-db", str(tmp_path / "shadow.sqlite3")]
    )

    assert result == 0
    output = capsys.readouterr().out
    assert '"decision": "insufficient_evidence"' in output
    assert '"runtime_eligible": false' in output
