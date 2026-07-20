import json

from src.final.expanded_surrogate_experiment import (
    ExperimentConfig,
    run_expanded_surrogate_experiments,
)
from app.services.surrogate_service import get_expanded_surrogate_evidence


def test_expanded_experiment_never_promotes_and_does_not_package_failed_models(
    tmp_path,
):
    output = tmp_path / "candidates"
    summary = run_expanded_surrogate_experiments(
        config=ExperimentConfig(
            development_samples=12,
            audit_samples=8,
            development_paths=16,
            audit_paths=32,
            monitoring_steps=8,
            trees=5,
            maximum_mae=0.0,
        ),
        output_root=output,
    )

    assert summary["runtime_policy_changed"] is False
    assert {item["product_key"] for item in summary["products"]} == {
        "phoenix_v3",
        "barrier_reverse_convertible",
    }
    assert all(item["status"] == "rejected" for item in summary["products"])
    assert not list(output.rglob("model.joblib"))
    stored = json.loads((output / "experiment_summary.json").read_text())
    evidence = get_expanded_surrogate_evidence(output / "experiment_summary.json")
    assert stored["runtime_policy_changed"] is False
    assert evidence["available"] is True
    assert all(item["runtime_approved"] is False for item in evidence["products"])
