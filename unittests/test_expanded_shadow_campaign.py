from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from app.services.expanded_shadow_monitoring import (
    _region,
    get_expanded_shadow_summary,
)
from src.final.expanded_shadow_campaign import (
    CONTRACT_GRID_VERSION,
    CampaignConfig,
    CampaignUnderlier,
    build_campaign_cases,
    campaign_plan,
    run_campaign,
)
from src.final.market import EquityMarketSegment, EquityMarketTermStructure


def make_market(symbol="SPY"):
    timestamp = datetime.now(timezone.utc)
    return EquityMarketTermStructure(
        symbol=symbol,
        underlier_type="etf",
        currency="USD",
        valuation_time=timestamp,
        market_data_time=timestamp,
        spot=100.0,
        segments=(EquityMarketSegment(2.0, 0.03, 0.01, 0.20),),
        calendar="XNYS",
        day_count="ACT/365F",
        source="campaign-test",
    )


def test_campaign_grid_covers_each_required_region_with_twelve_cases_per_product(
    tmp_path,
):
    market = make_market()
    cases = build_campaign_cases(market)
    config = CampaignConfig(
        campaign_date=market.market_data_time.date(),
        underliers=(CampaignUnderlier("SPY", "etf"),),
        output_dir=tmp_path / "reports",
        monitoring_db=tmp_path / "monitoring.sqlite3",
        snapshot_db=tmp_path / "snapshots.sqlite3",
    )
    plan = campaign_plan(config)

    assert len(cases) == 24
    assert plan["contract_grid_version"] == CONTRACT_GRID_VERSION
    assert plan["total_cases"] == 24
    assert plan["products"]["phoenix_v3"]["regions"] == {
        "broad": 4,
        "coupon": 2,
        "final_autocall": 2,
        "first_autocall": 2,
        "knock_in": 2,
    }
    assert plan["products"]["barrier_reverse_convertible"]["regions"] == {
        "broad": 4,
        "knock_in": 4,
        "strike": 4,
    }
    assert all(
        _region(case.product_key, market, case.contract) == case.expected_region
        for case in cases
    )


def test_campaign_freezes_market_records_reliable_evidence_and_is_idempotent(
    monkeypatch, tmp_path
):
    market = make_market()

    class FakeMarketService:
        calls = 0

        def build_term_structure(self, **_kwargs):
            self.calls += 1
            return SimpleNamespace(
                market=market,
                calibration={
                    "calibration_id": f"sha256:{'a' * 64}",
                    "quality": {"status": "research_ready"},
                },
            )

    service = FakeMarketService()
    config = CampaignConfig(
        campaign_date=market.market_data_time.date(),
        underliers=(CampaignUnderlier("SPY", "etf"),),
        output_dir=tmp_path / "reports",
        monitoring_db=tmp_path / "monitoring.sqlite3",
        snapshot_db=tmp_path / "snapshots.sqlite3",
    )
    monkeypatch.setattr(
        "src.final.expanded_shadow_campaign._price_case",
        lambda *_args, **_kwargs: {
            "price": 0.97,
            "standard_error": 0.002,
            "latency_ms": 30.0,
        },
    )

    first = run_campaign(config, market_service=service)
    second = run_campaign(config, market_service=service)
    summary = get_expanded_shadow_summary(limit=100_000, db_path=config.monitoring_db)

    assert first["status"] == "completed"
    assert first["results"]["recorded"] == 24
    assert second["results"]["already_recorded"] == 24
    assert service.calls == 1
    assert len(first["markets"]) == 1
    assert first["markets"]["SPY"]["snapshot_id"] == f"sha256:{'a' * 64}"
    for product in ("phoenix_v3", "barrier_reverse_convertible"):
        observed = summary["products"][product]
        assert observed["n_observations"] == 12
        assert observed["reliable_reference"]["n_observations"] == 12
        assert observed["reliable_reference"][
            "mean_reference_standard_error"
        ] == pytest.approx(0.002)
        assert observed["observation_sources"] == {"out_of_time_campaign": 12}
        assert observed["campaigns"] == 1
