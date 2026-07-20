import numpy as np
import pytest

from src.final.barrier_reverse_convertible import (
    BarrierReverseConvertiblePayoff,
    BarrierReverseConvertibleV1Contract,
    BarrierReverseConvertibleValidationError,
)


def make_contract(**overrides):
    values = {
        "reference_level": 100.0,
        "maturity_years": 1.0,
        "coupon_times_years": (0.5, 1.0),
        "coupon_rate_per_period": 0.02,
        "strike_frac": 1.0,
        "knock_in_frac": 0.7,
        "prior_knock_in_breached": False,
    }
    values.update(overrides)
    return BarrierReverseConvertibleV1Contract(**values)


def test_contract_has_stable_identity_and_exact_coupon_schedule():
    contract = make_contract()

    assert contract.contract_version == "barrier-reverse-convertible-v1"
    assert contract.contract_id == make_contract().contract_id
    assert contract.to_dict()["remaining_coupon_count"] == 2


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"coupon_times_years": (0.25, 0.75)}, "final coupon time"),
        ({"knock_in_frac": 1.1}, "knock_in_frac"),
        ({"prior_knock_in_breached": 1}, "must be boolean"),
    ],
)
def test_contract_rejects_ambiguous_downside_state(overrides, message):
    with pytest.raises(BarrierReverseConvertibleValidationError, match=message):
        make_contract(**overrides)


def test_payoff_pays_fixed_coupons_and_conditional_downside_redemption():
    paths = np.array(
        [
            [100.0, 60.0, 80.0],
            [100.0, 60.0, 110.0],
            [100.0, 90.0, 80.0],
        ]
    )
    contract = make_contract()
    ledger = BarrierReverseConvertiblePayoff().compute_event_ledger(
        paths=paths,
        params=contract.to_payoff_params(risk_free_rate=0.0, volatility=0.2),
        path_times_years=np.array([0.0, 0.5, 1.0]),
        coupon_times_years=contract.coupon_times_years,
        prior_knock_in_breached=False,
        discount_factor=lambda _time: 1.0,
    )
    values = (
        ledger["coupon_pv"]
        + ledger["protected_principal_pv"]
        + ledger["downside_redemption_pv"]
    )

    assert values.tolist() == pytest.approx([0.84, 1.04, 1.04])
    assert ledger["knock_in_probability"].tolist() == [1.0, 1.0, 0.0]
    assert ledger["downside_probability"].tolist() == [1.0, 0.0, 0.0]


def test_prior_knock_in_state_is_carried_into_maturity_rule():
    paths = np.array([[100.0, 90.0, 80.0]])
    contract = make_contract(prior_knock_in_breached=True)
    ledger = BarrierReverseConvertiblePayoff().compute_event_ledger(
        paths=paths,
        params=contract.to_payoff_params(risk_free_rate=0.0, volatility=0.2),
        path_times_years=np.array([0.0, 0.5, 1.0]),
        coupon_times_years=contract.coupon_times_years,
        prior_knock_in_breached=True,
        discount_factor=lambda _time: 1.0,
    )

    assert ledger["downside_redemption_pv"].tolist() == [0.8]
