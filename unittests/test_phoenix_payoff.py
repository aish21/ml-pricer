import json
from pathlib import Path

import numpy as np
import pytest

from src.final.payoffs import PhoenixPayoff
from src.final.reference_pricer import price_reference


def phoenix_params(**overrides):
    params = {
        "S0": 100.0,
        "r": 0.0,
        "sigma": 0.2,
        "T": 1.0,
        "autocall_barrier_frac": 1.10,
        "coupon_barrier_frac": 1.00,
        "coupon_rate": 0.02,
        "knock_in_frac": 0.70,
        "obs_count": 2,
    }
    params.update(overrides)
    return params


def test_feature_order_contains_every_price_sensitive_term():
    payoff = PhoenixPayoff()

    assert payoff.contract_version == "phoenix-single-v1"
    assert payoff.get_feature_order() == payoff.get_parameter_names()


def test_periodic_non_memory_coupons_are_paid_while_note_is_active():
    paths = np.array([[100.0, 100.0, 100.0]])

    value = PhoenixPayoff().compute_payoff(paths, phoenix_params(), 0.0, 1.0)

    assert value[0] == pytest.approx(1.04)


def test_autocall_pays_current_coupon_and_stops_future_cashflows():
    paths = np.array([[100.0, 106.0, 150.0]])
    params = phoenix_params(autocall_barrier_frac=1.05)

    value = PhoenixPayoff().compute_payoff(paths, params, 0.0, 1.0)

    assert value[0] == pytest.approx(1.02)


def test_coupon_barrier_changes_coupon_cashflows():
    paths = np.array([[100.0, 95.0, 95.0]])
    low_barrier = phoenix_params(coupon_barrier_frac=0.90)
    high_barrier = phoenix_params(coupon_barrier_frac=1.00)

    low_value = PhoenixPayoff().compute_payoff(paths, low_barrier, 0.0, 1.0)
    high_value = PhoenixPayoff().compute_payoff(paths, high_barrier, 0.0, 1.0)

    assert low_value[0] == pytest.approx(1.04)
    assert high_value[0] == pytest.approx(1.00)


def test_knock_in_causes_loss_only_when_final_level_is_below_initial():
    paths = np.array(
        [
            [100.0, 60.0, 80.0],
            [100.0, 60.0, 110.0],
        ]
    )
    params = phoenix_params(
        autocall_barrier_frac=2.0,
        coupon_barrier_frac=2.0,
    )

    values = PhoenixPayoff().compute_payoff(paths, params, 0.0, 1.0)

    assert values[0] == pytest.approx(0.80)
    assert values[1] == pytest.approx(1.00)


def test_payoff_aware_components_reconcile_to_contract_value_and_events():
    paths = np.array(
        [
            [100.0, 110.0, 120.0],
            [100.0, 90.0, 104.0],
            [100.0, 60.0, 80.0],
        ]
    )
    params = phoenix_params(autocall_barrier_frac=1.05)
    payoff = PhoenixPayoff()

    values = payoff.compute_payoff(paths, params, 0.0, 1.0)
    components = payoff.compute_cashflow_components_with_discount_curve(
        paths, params, 1.0, lambda _time: 1.0
    )
    reconstructed = sum(
        components[name]
        for name in (
            "coupon_pv",
            "autocall_principal_pv",
            "maturity_protected_pv",
            "maturity_downside_pv",
        )
    )

    assert reconstructed == pytest.approx(values)
    assert components["autocall_probability"].tolist() == [1.0, 0.0, 0.0]
    assert components["downside_probability"].tolist() == [0.0, 0.0, 1.0]


def test_observation_event_ledger_reconciles_sequential_events():
    paths = np.array(
        [
            [100.0, 110.0, 120.0],
            [100.0, 90.0, 104.0],
            [100.0, 60.0, 80.0],
        ]
    )
    params = phoenix_params(autocall_barrier_frac=1.05)

    ledger = PhoenixPayoff().compute_observation_event_ledger_with_discount_curve(
        paths,
        params,
        1.0,
        lambda _time: 1.0,
    )

    assert ledger["first_autocall_event"].tolist() == [
        [1.0, 0.0],
        [0.0, 0.0],
        [0.0, 0.0],
    ]
    assert ledger["coupon_event"].tolist() == [
        [1.0, 0.0],
        [0.0, 1.0],
        [0.0, 0.0],
    ]
    assert ledger["survival_after_observation"].tolist() == [
        [0.0, 0.0],
        [1.0, 1.0],
        [1.0, 1.0],
    ]
    assert ledger["protected_maturity_event"].tolist() == [0.0, 1.0, 0.0]
    assert ledger["downside_maturity_event"].tolist() == [0.0, 0.0, 1.0]
    assert ledger["downside_recovery_ratio"].tolist() == [0.0, 0.0, 0.8]


def test_explicit_schedule_uses_the_contractual_event_time():
    path_times = np.array([0.0, 0.25, 0.4, 0.75, 1.0])
    paths = np.array([[80.0, 80.0, 110.0, 80.0, 80.0]])
    params = phoenix_params(
        autocall_barrier_frac=1.05,
        coupon_barrier_frac=1.0,
    )

    value = PhoenixPayoff().compute_payoff_with_explicit_schedule_and_discount_curve(
        paths=paths,
        params=params,
        path_times_years=path_times,
        observation_times_years=(0.4, 1.0),
        prior_knock_in_breached=False,
        discount_factor=lambda _time: 1.0,
    )

    assert value[0] == pytest.approx(1.02)


def test_explicit_schedule_carries_historical_knock_in_state_forward():
    path_times = np.array([0.0, 0.4, 1.0])
    paths = np.array([[80.0, 80.0, 80.0]])
    params = phoenix_params(
        autocall_barrier_frac=2.0,
        coupon_barrier_frac=2.0,
    )
    payoff = PhoenixPayoff()

    protected = payoff.compute_payoff_with_explicit_schedule_and_discount_curve(
        paths=paths,
        params=params,
        path_times_years=path_times,
        observation_times_years=(0.4, 1.0),
        prior_knock_in_breached=False,
        discount_factor=lambda _time: 1.0,
    )
    knocked_in = payoff.compute_payoff_with_explicit_schedule_and_discount_curve(
        paths=paths,
        params=params,
        path_times_years=path_times,
        observation_times_years=(0.4, 1.0),
        prior_knock_in_breached=True,
        discount_factor=lambda _time: 1.0,
    )

    assert protected[0] == pytest.approx(1.0)
    assert knocked_in[0] == pytest.approx(0.8)


def test_explicit_schedule_recovers_carried_and_newly_missed_memory_coupons():
    path_times = np.array([0.0, 0.5, 1.0])
    paths = np.array([[100.0, 90.0, 100.0]])
    params = phoenix_params(
        autocall_barrier_frac=1.2,
        coupon_barrier_frac=1.0,
    )

    ledger = PhoenixPayoff().compute_observation_event_ledger_with_explicit_schedule(
        paths=paths,
        params=params,
        path_times_years=path_times,
        observation_times_years=(0.5, 1.0),
        prior_knock_in_breached=False,
        discount_factor=lambda _time: 1.0,
        autocall_barrier_fracs=(1.2, 1.1),
        memory_coupon=True,
        unpaid_coupon_count=1,
    )
    value = sum(
        ledger[name]
        for name in (
            "coupon_pv",
            "autocall_principal_pv",
            "maturity_protected_pv",
            "maturity_downside_pv",
        )
    )

    assert value[0] == pytest.approx(1.06)
    assert ledger["coupon_amount_event"].tolist() == [[0.0, 0.06]]
    assert ledger["coupon_memory_balance"].tolist() == [0.0]


def test_explicit_schedule_uses_the_stepdown_barrier_at_each_observation():
    path_times = np.array([0.0, 0.5, 1.0])
    paths = np.array([[100.0, 104.0, 104.0]])

    ledger = PhoenixPayoff().compute_observation_event_ledger_with_explicit_schedule(
        paths=paths,
        params=phoenix_params(),
        path_times_years=path_times,
        observation_times_years=(0.5, 1.0),
        prior_knock_in_breached=False,
        discount_factor=lambda _time: 1.0,
        autocall_barrier_fracs=(1.1, 1.03),
    )

    assert ledger["first_autocall_event"].tolist() == [[0.0, 1.0]]
    assert ledger["autocall_principal_pv"].tolist() == [1.0]


def test_payoff_is_invariant_to_underlier_price_scale():
    paths = np.array([[100.0, 95.0, 80.0], [100.0, 106.0, 120.0]])
    params = phoenix_params(autocall_barrier_frac=1.05)
    scaled_params = {**params, "S0": 250.0}

    base = PhoenixPayoff().compute_payoff(paths, params, 0.0, 1.0)
    scaled = PhoenixPayoff().compute_payoff(paths * 2.5, scaled_params, 0.0, 1.0)

    assert scaled == pytest.approx(base)


def test_reference_pricing_is_seeded_and_reports_uncertainty():
    params = phoenix_params(obs_count=6)

    first = price_reference(PhoenixPayoff(), params, n_paths=500, seed=123)
    second = price_reference(PhoenixPayoff(), params, n_paths=500, seed=123)

    assert first["price"] == second["price"]
    assert first["standard_error"] == second["standard_error"]
    assert first["confidence_interval"][0] <= first["price"]
    assert first["confidence_interval"][1] >= first["price"]


def test_reference_pricing_matches_frozen_golden_case():
    fixture_path = Path(__file__).parent / "golden" / "phoenix-single-v1.json"
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    case = fixture["cases"][0]

    result = price_reference(
        PhoenixPayoff(),
        case["params"],
        n_paths=case["n_paths"],
        n_steps=case["n_steps"],
        seed=case["seed"],
    )

    assert PhoenixPayoff.contract_version == fixture["contract_version"]
    assert result["price"] == pytest.approx(case["expected"]["price"], abs=1e-12)
    assert result["payoff_std"] == pytest.approx(
        case["expected"]["payoff_std"], abs=1e-12
    )
    assert result["standard_error"] == pytest.approx(
        case["expected"]["standard_error"], abs=1e-12
    )
