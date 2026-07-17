import json
from dataclasses import replace

import numpy as np
import pytest

from src.final.surrogate_contract import (
    DEFAULT_TRAINING_DOMAIN,
    PHOENIX_SURROGATE_FEATURE_NAMES,
    surrogate_contract_metadata,
)
from src.final.surrogate_data import PhoenixSurrogateDataset
from src.final.surrogate_model import load_numpy_mlp_artifact
from src.final.surrogate_trainer import (
    PhoenixSurrogateTrainingConfig,
    train_phoenix_surrogate,
)


def synthetic_dataset(*, role="development", seed=5):
    rng = np.random.RandomState(seed)
    X = rng.normal(size=(120, len(PHOENIX_SURROGATE_FEATURE_NAMES)))
    y = 1.0 + 0.02 * X[:, 0] - 0.01 * X[:, 1] + 0.005 * X[:, 2] ** 2
    split_names = (
        np.asarray(["train"] * 80 + ["validation"] * 20 + ["test"] * 20)
        if role == "development"
        else np.asarray(["audit"] * 120)
    )
    auxiliary_targets = np.column_stack(
        [
            0.1 * y,
            0.5 * y,
            0.4 * y,
            np.zeros_like(y),
            np.clip(0.5 + 0.1 * X[:, 0], 0.0, 1.0),
            np.clip(0.2 - 0.05 * X[:, 0], 0.0, 1.0),
        ]
    )
    metadata = {
        "dataset_schema_version": "phoenix-surrogate-dataset-v3",
        "dataset_id": f"sha256:synthetic-{role}-{seed}",
        "dataset_role": role,
        "n_samples": 120,
        "split_counts": (
            {"train": 80, "validation": 20, "test": 20}
            if role == "development"
            else {"audit": 120}
        ),
        "split_group_counts": (
            {"train": 80, "validation": 20, "test": 20}
            if role == "development"
            else {"audit": 120}
        ),
        "config": {
            "synthetic": True,
            "dataset_seed": seed,
            "label_seed": seed + 100,
            "label_replications": 8,
            "paths_per_replication": 256,
            "sampling_method": "sobol",
        },
        "label_uncertainty_protocol": {
            "estimator": "between-replication-standard-error",
            "independent_randomizations": 8,
            "paths_per_randomization": 256,
            "total_paths_per_label": 2048,
            "sampling_method": "sobol",
        },
        "generation_environment": {
            "python": "test",
            "numpy": "test",
            "scipy": "test",
        },
        **surrogate_contract_metadata(),
    }
    metadata["training_domain"] = {
        name: list(bounds) for name, bounds in DEFAULT_TRAINING_DOMAIN.items()
    }
    return PhoenixSurrogateDataset(
        X=X,
        y=y,
        label_standard_error=np.full(120, 0.01),
        payoff_standard_deviation=np.full(120, 0.1),
        auxiliary_targets=auxiliary_targets,
        auxiliary_standard_error=np.full((120, 6), 0.01),
        group_ids=np.arange(120),
        split_names=split_names,
        regime_names=np.asarray(["normal"] * 120),
        moneyness_region_names=np.asarray(["broad"] * 120),
        metadata=metadata,
    )


def test_training_exports_versioned_checksum_numpy_artifact(tmp_path):
    manifest = train_phoenix_surrogate(
        dataset=synthetic_dataset(),
        audit_dataset=synthetic_dataset(role="audit", seed=6),
        output_root=tmp_path,
        config=PhoenixSurrogateTrainingConfig(
            hidden_layer_sizes=(16,),
            max_iter=150,
            train_lightgbm_baseline=False,
            greek_validation_cases=0,
            acceptance_audit_mae=10.0,
            acceptance_audit_p95_absolute_error=10.0,
            acceptance_audit_r2=-100.0,
            acceptance_maximum_regime_mae=10.0,
            acceptance_maximum_moneyness_region_mae=10.0,
            acceptance_minimum_uncertainty_or_economic_coverage=0.0,
            acceptance_maximum_label_confidence_half_width_p95=10.0,
            acceptance_maximum_component_mae=10.0,
            acceptance_maximum_event_mae=10.0,
            acceptance_maximum_mean_output_boundary_violation=1.0,
            acceptance_maximum_cashflow_reconstruction_mae=10.0,
        ),
        verbose=False,
    )

    pointer = json.loads((tmp_path / "current.json").read_text())
    artifact_dir = tmp_path / pointer["directory"]
    model = load_numpy_mlp_artifact(
        artifact_dir / "weights.npz",
        manifest["feature_names"],
        manifest["output_names"],
    )
    predictions = model.predict(synthetic_dataset().X[:3])
    assert manifest["deployment_status"] == "shadow_approved"
    assert manifest["runtime_policy"] == "shadow-only"
    assert manifest["dataset_id"] == "sha256:synthetic-development-5"
    assert manifest["audit_dataset_id"] == "sha256:synthetic-audit-6"
    assert manifest["selected_strategy"] in {
        "direct_price",
        "payoff_aware",
        "payoff_aware_focused_head",
    }
    assert "scipy" in manifest["training_environment"]
    assert manifest["acceptance"]["evaluation_dataset_id"] == (
        "sha256:synthetic-audit-6"
    )
    assert (
        manifest["audit_uncertainty_policy"]["policy_version"]
        == "phoenix-audit-uncertainty-v1"
    )
    assert (
        "uncertainty_or_economic_tolerance_coverage" in manifest["acceptance"]["checks"]
    )
    assert "within_two_label_se_fraction" not in manifest["acceptance"]["checks"]
    assert predictions.shape == (3,)
    assert np.all(np.isfinite(predictions))


def test_candidate_search_is_selected_only_from_development_validation(tmp_path):
    manifest = train_phoenix_surrogate(
        dataset=synthetic_dataset(),
        audit_dataset=synthetic_dataset(role="audit", seed=8),
        output_root=tmp_path,
        config=PhoenixSurrogateTrainingConfig(
            hidden_layer_sizes=(8,),
            candidate_hidden_layer_sizes=((12,),),
            candidate_seed_offsets=(0, 1),
            max_iter=50,
            train_lightgbm_baseline=False,
            greek_validation_cases=0,
            acceptance_audit_mae=10.0,
            acceptance_audit_p95_absolute_error=10.0,
            acceptance_audit_r2=-100.0,
            acceptance_maximum_regime_mae=10.0,
            acceptance_maximum_moneyness_region_mae=10.0,
            acceptance_maximum_regime_moneyness_mae=10.0,
            acceptance_minimum_uncertainty_or_economic_coverage=0.0,
            acceptance_maximum_label_confidence_half_width_p95=10.0,
            acceptance_maximum_component_mae=10.0,
            acceptance_maximum_event_mae=10.0,
            acceptance_maximum_mean_output_boundary_violation=1.0,
            acceptance_maximum_cashflow_reconstruction_mae=10.0,
        ),
        verbose=False,
    )

    assert len(manifest["candidate_models"]) == 9
    assert manifest["selected_candidate"] in manifest["candidate_models"]
    selected = manifest["candidate_models"][manifest["selected_candidate"]]
    assert selected["selection"]["policy"] == "robust-validation-mae-v3"
    assert "maximum_validation_regime_moneyness_mae" in selected["selection"]
    repeated = selected["selection"]["repeated_group_validation"]
    assert repeated["policy"] == "repeated-group-held-out-validation-v1"
    assert repeated["folds"] == 5
    assert repeated["repeats"] == 3
    assert len(repeated["fold_metrics"]) == 15
    assert all(fold["n_groups"] > 0 for fold in repeated["fold_metrics"])
    assert manifest["development_error_analysis"]["split"] == "validation"
    assert manifest["acceptance"]["evaluation_dataset_id"] == (
        "sha256:synthetic-audit-8"
    )


def test_training_rejects_an_under_replicated_audit_before_fitting(tmp_path):
    audit = synthetic_dataset(role="audit", seed=9)
    weak_metadata = {
        **audit.metadata,
        "config": {
            **audit.metadata["config"],
            "label_replications": 2,
            "paths_per_replication": 1024,
        },
        "label_uncertainty_protocol": {
            "estimator": "between-replication-standard-error",
            "independent_randomizations": 2,
            "paths_per_randomization": 1024,
            "total_paths_per_label": 2048,
            "sampling_method": "sobol",
        },
    }

    with pytest.raises(
        RuntimeError,
        match="requires more independent replications",
    ):
        train_phoenix_surrogate(
            dataset=synthetic_dataset(),
            audit_dataset=replace(audit, metadata=weak_metadata),
            output_root=tmp_path,
            config=PhoenixSurrogateTrainingConfig(
                hidden_layer_sizes=(8,),
                max_iter=10,
                train_lightgbm_baseline=False,
                greek_validation_cases=0,
            ),
            verbose=False,
        )
