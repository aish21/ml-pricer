import numpy as np
import pytest

from src.final.surrogate_data import (
    PhoenixDatasetConfig,
    SurrogateDatasetError,
    generate_phoenix_surrogate_dataset,
    load_phoenix_surrogate_dataset,
    save_phoenix_surrogate_dataset,
)


def tiny_config(**changes):
    values = {
        "n_contracts": 6,
        "markets_per_contract": 2,
        "paths_per_replication": 8,
        "label_replications": 2,
        "n_steps": 36,
        "dataset_seed": 17,
        "label_seed": 23,
        "sampling_method": "sobol",
    }
    values.update(changes)
    return PhoenixDatasetConfig(**values)


def test_dataset_generation_is_deterministic_and_group_disjoint():
    first = generate_phoenix_surrogate_dataset(tiny_config(), verbose=False)
    second = generate_phoenix_surrogate_dataset(tiny_config(), verbose=False)

    assert first.metadata["dataset_id"] == second.metadata["dataset_id"]
    assert np.array_equal(first.X, second.X)
    assert np.array_equal(first.y, second.y)
    assert first.X.shape == (12, 24)
    assert first.auxiliary_targets.shape == (12, 6)
    assert np.allclose(first.auxiliary_targets[:, :4].sum(axis=1), first.y)
    assert first.metadata["label_uncertainty"] == (
        "standard error across independent scrambled replications"
    )
    assert first.metadata["label_uncertainty_protocol"] == {
        "estimator": "between-replication-standard-error",
        "independent_randomizations": 2,
        "paths_per_randomization": 8,
        "total_paths_per_label": 16,
        "sampling_method": "sobol",
    }
    assert set(first.split_names) == {"train", "validation", "test"}
    for group_id in set(first.group_ids):
        group_splits = set(first.split_names[first.group_ids == group_id])
        assert len(group_splits) == 1
    split_groups = {
        split: set(first.group_ids[first.split_names == split])
        for split in ("train", "validation", "test")
    }
    assert split_groups["train"].isdisjoint(split_groups["validation"])
    assert split_groups["train"].isdisjoint(split_groups["test"])
    assert split_groups["validation"].isdisjoint(split_groups["test"])


def test_dataset_round_trip_uses_pickle_free_fingerprinted_npz(tmp_path):
    dataset = generate_phoenix_surrogate_dataset(tiny_config(), verbose=False)
    path = tmp_path / "dataset.npz"
    save_phoenix_surrogate_dataset(dataset, path)
    loaded = load_phoenix_surrogate_dataset(path)

    assert loaded.metadata["dataset_id"] == dataset.metadata["dataset_id"]
    assert np.array_equal(loaded.X, dataset.X)
    assert np.array_equal(loaded.label_standard_error, dataset.label_standard_error)
    assert np.array_equal(loaded.auxiliary_targets, dataset.auxiliary_targets)


def test_audit_dataset_is_mechanically_separate_from_development_splits():
    audit = generate_phoenix_surrogate_dataset(
        tiny_config(dataset_role="audit"), verbose=False
    )

    assert set(audit.split_names) == {"audit"}
    assert audit.metadata["dataset_role"] == "audit"
    assert audit.metadata["split_counts"] == {"audit": 12}


def test_dataset_round_trip_rejects_tampered_uncertainty(tmp_path):
    dataset = generate_phoenix_surrogate_dataset(tiny_config(), verbose=False)
    path = tmp_path / "dataset.npz"
    save_phoenix_surrogate_dataset(dataset, path)
    with np.load(path, allow_pickle=False) as data:
        payload = {name: np.array(data[name]) for name in data.files}
    payload["label_standard_error"][0] += 0.001
    with path.open("wb") as handle:
        np.savez_compressed(handle, **payload)

    with pytest.raises(SurrogateDatasetError, match="fingerprint mismatch"):
        load_phoenix_surrogate_dataset(path)


def test_dataset_configuration_rejects_weak_or_incompatible_sampling():
    with pytest.raises(SurrogateDatasetError, match="power of two"):
        tiny_config(paths_per_replication=10)
    with pytest.raises(SurrogateDatasetError, match="n_steps"):
        tiny_config(n_steps=11)
    with pytest.raises(SurrogateDatasetError, match="audit datasets"):
        tiny_config(dataset_role="audit", sampling_profile="low_vol_barrier_focus")


def test_low_vol_barrier_profile_adds_focused_development_cases():
    dataset = generate_phoenix_surrogate_dataset(
        tiny_config(
            markets_per_contract=6,
            sampling_profile="low_vol_barrier_focus",
        ),
        verbose=False,
    )

    low_volatility_count = int(np.sum(dataset.regime_names == "low_vol"))
    assert low_volatility_count > int(np.sum(dataset.regime_names == "normal"))
    focused = dataset.moneyness_region_names[np.arange(len(dataset.y)) % 6 >= 4]
    assert set(focused).issubset({"knock_in", "coupon", "autocall"})
