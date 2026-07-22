from datetime import datetime, timezone

import pytest

from app.services.expanded_shadow_service import (
    evaluate_expanded_shadow,
    get_expanded_shadow_status,
)
from app.services.pricing_service import (
    price_barrier_reverse_convertible_with_term_structure,
    price_phoenix_v3_with_term_structure,
)
from src.final.barrier_reverse_convertible import BarrierReverseConvertibleV1Contract
from src.final.market import EquityMarketSegment, EquityMarketTermStructure
from src.final.phoenix_contract import PhoenixSingleV3Contract


def make_market(*, spot=100.0, currency="USD", volatility=0.2):
    timestamp = datetime(2026, 7, 22, 12, 0, tzinfo=timezone.utc)
    return EquityMarketTermStructure(
        symbol="SPY",
        underlier_type="etf",
        currency=currency,
        valuation_time=timestamp,
        market_data_time=timestamp,
        spot=spot,
        segments=(EquityMarketSegment(1.0, 0.03, 0.01, volatility),),
        calendar="XNYS",
        day_count="ACT/365F",
        source="expanded-shadow-test",
    )


def phoenix_contract(*, schedule=(0.25, 0.5, 0.75, 1.0)):
    return PhoenixSingleV3Contract(
        reference_level=100.0,
        maturity_years=1.0,
        observation_times_years=schedule,
        autocall_barrier_fracs=(1.10, 1.05, 1.00, 0.95),
        coupon_barrier_frac=0.80,
        coupon_rate=0.02,
        knock_in_frac=0.60,
        prior_knock_in_breached=False,
        memory_coupon=True,
        unpaid_coupon_count=1,
    )


def brc_contract():
    return BarrierReverseConvertibleV1Contract(
        reference_level=100.0,
        maturity_years=1.0,
        coupon_times_years=(0.25, 0.5, 0.75, 1.0),
        coupon_rate_per_period=0.02,
        strike_frac=1.0,
        knock_in_frac=0.65,
        prior_knock_in_breached=False,
    )


def evaluate(product_key, contract, market=None, **kwargs):
    return evaluate_expanded_shadow(
        product_key=product_key,
        market=market or make_market(),
        contract=contract,
        reference_price=0.97,
        reference_standard_error=0.003,
        reference_latency_ms=30.0,
        force=True,
        **kwargs,
    )


def test_pinned_artifacts_validate_but_start_disabled():
    status = get_expanded_shadow_status()

    assert status["automatic_promotion_permitted"] is False
    assert set(status["products"]) == {"phoenix_v3", "barrier_reverse_convertible"}
    assert all(item["artifact_available"] for item in status["products"].values())
    assert all(item["enabled"] is False for item in status["products"].values())
    assert all(
        item["runtime_approved"] is False for item in status["products"].values()
    )


@pytest.mark.parametrize(
    ("product_key", "contract"),
    [
        ("phoenix_v3", phoenix_contract()),
        ("barrier_reverse_convertible", brc_contract()),
    ],
)
def test_safe_tree_runtime_evaluates_both_pinned_models(product_key, contract):
    result = evaluate(product_key, contract)

    assert result["status"] == "success"
    assert result["mode"] == "shadow-only"
    assert result["used_for_price"] is False
    assert result["runtime_approved"] is False
    assert 0.0 < result["surrogate_price"] < 2.0
    assert result["artifact_id"].startswith("sha256:")
    assert result["latency_ms"] >= 0.0


def test_runtime_rejects_untrained_schedule_without_affecting_reference():
    result = evaluate(
        "phoenix_v3",
        phoenix_contract(schedule=(0.20, 0.50, 0.75, 1.0)),
    )

    assert result["status"] == "out_of_domain"
    assert result["used_for_price"] is False
    assert "evenly spaced" in result["reason"]


def test_runtime_fails_closed_when_artifact_root_is_missing(monkeypatch, tmp_path):
    monkeypatch.setenv("EXPANDED_SURROGATE_ARTIFACT_ROOT", str(tmp_path / "missing"))

    result = evaluate("phoenix_v3", phoenix_contract())

    assert result["status"] == "unavailable"
    assert result["used_for_price"] is False
    assert "artifact" in result["reason"]


def test_disabled_and_unsampled_controls_are_distinct(monkeypatch):
    disabled = evaluate_expanded_shadow(
        product_key="phoenix_v3",
        market=make_market(),
        contract=phoenix_contract(),
        reference_price=1.0,
        reference_standard_error=0.01,
        reference_latency_ms=20.0,
    )
    assert disabled["status"] == "disabled"

    monkeypatch.setenv("PHOENIX_V3_SHADOW_ENABLED", "true")
    monkeypatch.setenv("PHOENIX_V3_SHADOW_SAMPLE_RATE", "0")
    unsampled = evaluate_expanded_shadow(
        product_key="phoenix_v3",
        market=make_market(),
        contract=phoenix_contract(),
        reference_price=1.0,
        reference_standard_error=0.01,
        reference_latency_ms=20.0,
    )
    assert unsampled["status"] == "not_sampled"


@pytest.mark.parametrize(
    ("product_key", "enabled_env", "price_function", "contract"),
    [
        (
            "phoenix_v3",
            "PHOENIX_V3_SHADOW_ENABLED",
            price_phoenix_v3_with_term_structure,
            phoenix_contract(),
        ),
        (
            "barrier_reverse_convertible",
            "BRC_V1_SHADOW_ENABLED",
            price_barrier_reverse_convertible_with_term_structure,
            brc_contract(),
        ),
    ],
)
def test_pricing_integration_keeps_monte_carlo_authoritative(
    monkeypatch, product_key, enabled_env, price_function, contract
):
    monkeypatch.setenv(enabled_env, "true")
    sample_env = (
        "PHOENIX_V3_SHADOW_SAMPLE_RATE"
        if product_key == "phoenix_v3"
        else "BRC_V1_SHADOW_SAMPLE_RATE"
    )
    monkeypatch.setenv(sample_env, "1")

    result = price_function(make_market(), contract, n_paths=100)

    assert result["surrogate_shadow"]["status"] == "success"
    assert result["surrogate_shadow"]["used_for_price"] is False
    assert result["price"] == result["mc_price"]
    assert result["pricing_method"] == "monte_carlo_reference"
