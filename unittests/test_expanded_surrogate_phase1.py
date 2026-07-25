import json
from pathlib import Path

import numpy as np
import pytest

from src.final.data_generator import build_simulation_time_grid
from src.final.expanded_surrogate_phase1 import (
    DEFAULT_REFERENCE_STEPS,
    ExpandedSurrogatePhase1Error,
    Phase1Config,
    _dataset,
    _sample_phoenix_case,
    run_phase1_experiment,
    uncertainty_sample_weights,
)


def test_phase1_configuration_requires_the_production_grid_and_sobol_sizes():
    with pytest.raises(
        ExpandedSurrogatePhase1Error,
        match="production_steps",
    ):
        Phase1Config(production_steps=64)

    with pytest.raises(
        ExpandedSurrogatePhase1Error,
        match="power of two",
    ):
        Phase1Config(paths_per_replication=100)


def test_phase1_dataset_is_reproducible_and_keeps_label_uncertainty():
    config = Phase1Config(
        development_samples=4,
        validation_samples=4,
        evaluation_samples=4,
        paths_per_replication=8,
        evaluation_paths_per_replication=8,
        label_replications=2,
        trees=5,
    )
    first = _dataset(
        "phoenix_v3",
        role="development",
        samples=4,
        paths_per_replication=8,
        config=config,
        dataset_seed=101,
        label_seed=202,
    )
    second = _dataset(
        "phoenix_v3",
        role="development",
        samples=4,
        paths_per_replication=8,
        config=config,
        dataset_seed=101,
        label_seed=202,
    )

    assert first.dataset_id == second.dataset_id
    np.testing.assert_array_equal(first.features, second.features)
    np.testing.assert_array_equal(first.labels, second.labels)
    np.testing.assert_array_equal(
        first.label_standard_errors,
        second.label_standard_errors,
    )
    assert first.replication_means.shape == (4, 2)
    assert np.all(first.confidence_interval_low <= first.labels)
    assert np.all(first.labels <= first.confidence_interval_high)
    assert np.all(first.effective_steps >= DEFAULT_REFERENCE_STEPS)
    assert all(
        len(json.loads(market)["segments"]) == 4 for market in first.market_payloads
    )


def test_phase1_grid_contains_every_contractual_observation():
    case = _sample_phoenix_case(
        np.random.default_rng(123),
        case_index=0,
        barrier_focus_probability=0.6,
    )
    grid = build_simulation_time_grid(
        case.contract.maturity_years,
        DEFAULT_REFERENCE_STEPS,
        case.contract.observation_times_years,
    )

    for observation_time in case.contract.observation_times_years:
        assert np.any(np.isclose(grid, observation_time, atol=1e-12, rtol=0.0))


def test_uncertainty_weights_downweight_noisy_labels_and_remain_bounded():
    weights, policy = uncertainty_sample_weights(
        np.asarray([0.001, 0.01, 0.1], dtype=np.float64),
        maximum_weight_multiple=10.0,
    )

    assert weights[0] > weights[1] > weights[2]
    assert np.mean(weights) == pytest.approx(1.0)
    assert np.max(weights) / np.min(weights) <= 100.0 + 1e-12
    assert policy["version"] == "capped-inverse-label-variance-v1"


def test_phase1_experiment_is_research_only_and_keeps_the_learner_fixed(
    tmp_path,
):
    output = tmp_path / "phase1"
    config = Phase1Config(
        development_samples=12,
        validation_samples=6,
        evaluation_samples=8,
        paths_per_replication=8,
        evaluation_paths_per_replication=16,
        label_replications=2,
        trees=5,
    )
    report = run_phase1_experiment(
        config=config,
        output_root=output,
    )

    assert report["status"] == "research_only"
    assert report["runtime_policy_changed"] is False
    assert report["runtime_artifact_created"] is False
    assert {product["product_key"] for product in report["products"]} == {
        "phoenix_v3",
        "barrier_reverse_convertible",
    }
    for product in report["products"]:
        assert product["status"] == "research_only"
        assert product["runtime_approved"] is False
        assert product["learner"]["architecture_changed_from_v2"] is False
        assert product["learner"]["primary_candidate"] == "phase1_uncertainty_weighted"
        assert {
            "phase1_unweighted",
            "phase1_uncertainty_weighted",
        }.issubset(product["comparisons"])
        assert Path(product["learner"]["weighted_model_path"]).is_relative_to(output)

    stored = json.loads((output / "phase1_report.json").read_text())
    assert stored["experiment_id"] == report["experiment_id"]
    assert len(list((output / "reports").glob("*.json"))) == 1

    repeated = run_phase1_experiment(
        config=config,
        output_root=tmp_path / "phase1-copy",
    )
    assert repeated["experiment_id"] == report["experiment_id"]
    assert [product["candidate_id"] for product in repeated["products"]] == [
        product["candidate_id"] for product in report["products"]
    ]
