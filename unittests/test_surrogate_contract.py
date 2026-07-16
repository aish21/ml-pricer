from datetime import datetime, timezone

import numpy as np
import pytest

from src.final.market import EquityMarketSegment, EquityMarketTermStructure
from src.final.surrogate_contract import (
    PHOENIX_SURROGATE_FEATURE_NAMES,
    SurrogateContractError,
    domain_violations,
    extract_phoenix_surrogate_features,
    reconstruct_phoenix_surrogate_case,
)


TERMS = {
    "maturity_years": 1.0,
    "autocall_barrier_frac": 1.05,
    "coupon_barrier_frac": 0.75,
    "coupon_rate": 0.03,
    "knock_in_frac": 0.6,
    "obs_count": 12,
}


def make_market(spot=110.0):
    now = datetime(2026, 1, 2, 16, 0, tzinfo=timezone.utc)
    return EquityMarketTermStructure(
        symbol="SPY",
        underlier_type="etf",
        currency="USD",
        valuation_time=now,
        market_data_time=now,
        spot=spot,
        segments=(
            EquityMarketSegment(0.25, 0.02, 0.01, 0.15),
            EquityMarketSegment(0.50, 0.03, 0.012, 0.20),
            EquityMarketSegment(0.75, 0.04, 0.014, 0.25),
            EquityMarketSegment(1.00, 0.05, 0.016, 0.30),
        ),
        calendar="XNYS",
        day_count="ACT/365F",
        source="surrogate-contract-test",
    )


def test_feature_schema_encodes_moneyness_terms_and_cumulative_curves():
    features = extract_phoenix_surrogate_features(
        market=make_market(), terms=TERMS, contract_reference_spot=100.0
    )

    assert features.shape == (len(PHOENIX_SURROGATE_FEATURE_NAMES),)
    assert features[0] == pytest.approx(1.1)
    assert features[1:4].tolist() == pytest.approx(
        [np.log(1.1 / 1.05), np.log(1.1 / 0.75), np.log(1.1 / 0.6)]
    )
    assert features[4:10].tolist() == pytest.approx([1.0, 1.05, 0.75, 0.03, 0.6, 12.0])
    assert features[10] == pytest.approx(0.36)
    assert features[11] == pytest.approx(0.36)
    assert features[12] == pytest.approx(0.02)
    assert features[13] == pytest.approx(0.01)
    assert features[14] == pytest.approx(0.15**2 * 0.25)
    assert features[-3] == pytest.approx((0.02 + 0.03 + 0.04 + 0.05) / 4)


def test_feature_reconstruction_round_trips_training_case():
    original = extract_phoenix_surrogate_features(
        market=make_market(), terms=TERMS, contract_reference_spot=100.0
    )
    market, terms, reference_spot = reconstruct_phoenix_surrogate_case(original)
    rebuilt = extract_phoenix_surrogate_features(
        market=market, terms=terms, contract_reference_spot=reference_spot
    )

    assert np.allclose(rebuilt, original)
    assert market.spot == pytest.approx(110.0)
    assert reference_spot == 100.0


def test_feature_contract_rejects_bad_barriers_and_short_curve():
    bad_terms = {**TERMS, "coupon_barrier_frac": 0.5}
    with pytest.raises(SurrogateContractError, match="barriers"):
        extract_phoenix_surrogate_features(
            market=make_market(), terms=bad_terms, contract_reference_spot=100.0
        )

    short_terms = {**TERMS, "maturity_years": 1.5}
    with pytest.raises(SurrogateContractError, match="does not cover"):
        extract_phoenix_surrogate_features(
            market=make_market(), terms=short_terms, contract_reference_spot=100.0
        )


def test_domain_check_reports_semantic_out_of_distribution_fields():
    violations = domain_violations(
        market=make_market(spot=200.0),
        terms=TERMS,
        contract_reference_spot=100.0,
    )
    assert any("spot_ratio" in violation for violation in violations)
