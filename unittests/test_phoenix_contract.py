import pytest

from src.final.phoenix_contract import (
    PhoenixContractValidationError,
    PhoenixSingleV2Contract,
)


def make_contract(**overrides):
    values = {
        "reference_level": 100.0,
        "maturity_years": 1.0,
        "observation_times_years": (0.2, 0.45, 0.7, 1.0),
        "autocall_barrier_frac": 1.05,
        "coupon_barrier_frac": 1.0,
        "coupon_rate": 0.02,
        "knock_in_frac": 0.7,
        "prior_knock_in_breached": False,
    }
    values.update(overrides)
    return PhoenixSingleV2Contract(**values)


def test_v2_contract_has_stable_content_identity():
    first = make_contract()
    second = make_contract()

    assert first.contract_version == "phoenix-single-v2"
    assert first.contract_id == second.contract_id
    assert first.contract_id.startswith("sha256:")
    assert first.to_dict()["remaining_observation_count"] == 4


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"reference_level": 0.0}, "reference_level"),
        ({"observation_times_years": ()}, "must not be empty"),
        (
            {"observation_times_years": (0.5, 0.4, 1.0)},
            "strictly increasing",
        ),
        (
            {"observation_times_years": (0.25, 0.75)},
            "final observation time",
        ),
        (
            {"prior_knock_in_breached": 1},
            "must be boolean",
        ),
    ],
)
def test_v2_contract_rejects_incomplete_or_ambiguous_state(overrides, message):
    with pytest.raises(PhoenixContractValidationError, match=message):
        make_contract(**overrides)


def test_contract_identity_changes_with_historical_knock_in_state():
    protected = make_contract(prior_knock_in_breached=False)
    breached = make_contract(prior_knock_in_breached=True)

    assert protected.contract_id != breached.contract_id
