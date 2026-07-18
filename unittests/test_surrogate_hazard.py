from functools import lru_cache

import numpy as np
import pytest

from src.final.surrogate_data import (
    PhoenixDatasetConfig,
    generate_phoenix_surrogate_dataset,
)
from src.final.surrogate_hazard import (
    PhoenixHazardTrainingConfig,
    train_phoenix_observation_hazard_candidate,
)
from src.final.surrogate_hazard_data import (
    generate_phoenix_hazard_dataset,
    load_phoenix_hazard_dataset,
    reconstruct_hazard_prices,
    save_phoenix_hazard_dataset,
)


@lru_cache(maxsize=1)
def small_hazard_dataset():
    base = generate_phoenix_surrogate_dataset(
        PhoenixDatasetConfig(
            n_contracts=40,
            markets_per_contract=2,
            paths_per_replication=16,
            label_replications=1,
            n_steps=12,
            dataset_seed=123,
            label_seed=456,
            sampling_method="antithetic",
            dataset_role="development",
            sampling_profile="balanced",
        ),
        verbose=False,
    )
    return generate_phoenix_hazard_dataset(base, verbose=False)


def test_hazard_labels_reconstruct_the_base_monte_carlo_prices(tmp_path):
    dataset = small_hazard_dataset()

    reconstructed = reconstruct_hazard_prices(
        features=dataset.base.X,
        observation_mask=dataset.observation_mask,
        coupon_probability=dataset.coupon_probability,
        first_autocall_probability=dataset.first_autocall_probability,
        protected_probability=dataset.protected_probability,
        downside_probability=dataset.downside_probability,
        downside_conditional_recovery=dataset.downside_conditional_recovery,
        n_steps=int(dataset.metadata["n_steps"]),
    )
    output_path = tmp_path / "hazard-dataset.npz"
    save_phoenix_hazard_dataset(dataset, output_path)
    loaded = load_phoenix_hazard_dataset(output_path, base=dataset.base)

    assert reconstructed == pytest.approx(dataset.base.y, abs=1e-12)
    assert loaded.metadata["dataset_id"] == dataset.metadata["dataset_id"]
    assert np.array_equal(loaded.observation_mask, dataset.observation_mask)


def test_hazard_label_identity_is_independent_of_worker_count():
    sequential = small_hazard_dataset()
    parallel = generate_phoenix_hazard_dataset(
        sequential.base,
        verbose=False,
        workers=2,
    )

    assert parallel.metadata["dataset_id"] == sequential.metadata["dataset_id"]
    assert parallel.metadata["generation_workers"] == 2


def test_hazard_candidate_uses_soft_events_and_stays_research_only():
    dataset = small_hazard_dataset()

    model, report = train_phoenix_observation_hazard_candidate(
        dataset,
        PhoenixHazardTrainingConfig(
            max_iter=30,
            min_samples_leaf=2,
            selection_validation_folds=2,
            selection_validation_repeats=2,
        ),
    )
    predictions = model.predict(dataset.base.X[:3])

    assert np.all(np.isfinite(predictions))
    assert report["loss_policy"] == "soft-binomial-weighted-log-loss-v1"
    assert report["runtime_eligible"] is False
    assert report["audit_evaluated"] is False
    assert report["deployment_status"] == "research_only"
    assert report["target_reconstruction_maximum_error"] < 1e-12
    assert report["selection"]["policy"] == (
        "observation-hazard-development-comparison-v1"
    )


def test_hazard_generation_rejects_an_audit_dataset():
    development = small_hazard_dataset().base
    audit_metadata = {**development.metadata, "dataset_role": "audit"}
    audit = type(development)(
        X=development.X,
        y=development.y,
        label_standard_error=development.label_standard_error,
        payoff_standard_deviation=development.payoff_standard_deviation,
        auxiliary_targets=development.auxiliary_targets,
        auxiliary_standard_error=development.auxiliary_standard_error,
        group_ids=development.group_ids,
        split_names=np.asarray(["audit"] * len(development.y)),
        regime_names=development.regime_names,
        moneyness_region_names=development.moneyness_region_names,
        metadata=audit_metadata,
    )

    with pytest.raises(ValueError, match="require a development dataset"):
        generate_phoenix_hazard_dataset(audit, verbose=False)
