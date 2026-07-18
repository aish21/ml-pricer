import json
from datetime import datetime, timezone

import numpy as np
import pytest

import app.services.surrogate_service as surrogate_service
from app.services.surrogate_service import (
    SurrogateSettings,
    clear_surrogate_cache,
    evaluate_surrogate_shadow,
    get_surrogate_status,
)
from src.final.market import (
    EQUITY_GBM_PIECEWISE_MODEL_VERSION,
    EQUITY_MARKET_TERM_STRUCTURE_VERSION,
    EquityMarketSegment,
    EquityMarketTermStructure,
)
from src.final.payoffs import PhoenixPayoff
from src.final.surrogate_contract import (
    DEFAULT_TRAINING_DOMAIN,
    PHOENIX_SURROGATE_ARTIFACT_VERSION,
    PHOENIX_SURROGATE_FEATURE_NAMES,
    PHOENIX_SURROGATE_FEATURE_VERSION,
    PHOENIX_SURROGATE_LABEL_VERSION,
    PHOENIX_SURROGATE_MODEL_VERSION,
)
from src.final.surrogate_model import file_sha256, json_sha256
from src.final.surrogate_price_first_contract import (
    PHOENIX_PRICE_FIRST_ARTIFACT_VERSION,
    PHOENIX_PRICE_FIRST_APPROVED_ARTIFACT_ID,
    PHOENIX_PRICE_FIRST_AUDIT_DATASET_ID,
    PHOENIX_PRICE_FIRST_AUDIT_POLICY_ID,
    PHOENIX_PRICE_FIRST_AUDIT_REPORT_SHA256,
    PHOENIX_PRICE_FIRST_AUDIT_VERSION,
    PHOENIX_PRICE_FIRST_DEVELOPMENT_DATASET_ID,
    PHOENIX_PRICE_FIRST_FROZEN_AUXILIARY_WEIGHT,
    PHOENIX_PRICE_FIRST_FROZEN_SPECIFICATION_COMMIT,
    PHOENIX_PRICE_FIRST_MODEL_VERSION,
    PHOENIX_PRICE_FIRST_NUMPY_PARITY_ABSOLUTE_TOLERANCE,
    PHOENIX_PRICE_FIRST_OBSERVATION_DATASET_ID,
    PHOENIX_PRICE_FIRST_OUTPUT_NAMES,
)


TERMS = {
    "maturity_years": 1.0,
    "autocall_barrier_frac": 1.05,
    "coupon_barrier_frac": 0.8,
    "coupon_rate": 0.02,
    "knock_in_frac": 0.6,
    "obs_count": 6,
}


@pytest.fixture(autouse=True)
def reset_surrogate_cache():
    clear_surrogate_cache()
    yield
    clear_surrogate_cache()


def make_market(spot=100.0):
    timestamp = datetime(2026, 1, 2, 16, 0, tzinfo=timezone.utc)
    return EquityMarketTermStructure(
        symbol="SPY",
        underlier_type="etf",
        currency="USD",
        valuation_time=timestamp,
        market_data_time=timestamp,
        spot=spot,
        segments=(EquityMarketSegment(1.0, 0.03, 0.01, 0.2),),
        calendar="XNYS",
        day_count="ACT/365F",
        source="test-fixture",
    )


def write_artifact(root, *, deployment_status="shadow_approved"):
    directory = "a" * 64
    artifact_id = f"sha256:{directory}"
    artifact_directory = root / directory
    artifact_directory.mkdir(parents=True)
    weights_path = artifact_directory / "weights.npz"
    feature_count = len(PHOENIX_SURROGATE_FEATURE_NAMES)
    with weights_path.open("wb") as handle:
        np.savez_compressed(
            handle,
            feature_mean=np.zeros(feature_count),
            feature_scale=np.ones(feature_count),
            target_mean=np.asarray([1.0]),
            target_scale=np.asarray([1.0]),
            n_layers=np.asarray(1, dtype=np.int64),
            weight_0=np.zeros((feature_count, 1)),
            bias_0=np.zeros(1),
        )
    manifest = {
        "artifact_schema_version": PHOENIX_SURROGATE_ARTIFACT_VERSION,
        "artifact_id": artifact_id,
        "model_version": PHOENIX_SURROGATE_MODEL_VERSION,
        "feature_schema_version": PHOENIX_SURROGATE_FEATURE_VERSION,
        "market_data_version": EQUITY_MARKET_TERM_STRUCTURE_VERSION,
        "label_model_version": EQUITY_GBM_PIECEWISE_MODEL_VERSION,
        "label_schema_version": PHOENIX_SURROGATE_LABEL_VERSION,
        "contract_version": PhoenixPayoff.contract_version,
        "feature_names": list(PHOENIX_SURROGATE_FEATURE_NAMES),
        "output_names": ["price"],
        "runtime_policy": "shadow-only",
        "deployment_status": deployment_status,
        "dataset_id": "sha256:test-dataset",
        "training_domain": {
            name: list(bounds) for name, bounds in DEFAULT_TRAINING_DOMAIN.items()
        },
        "files": {"weights.npz": file_sha256(weights_path)},
    }
    (artifact_directory / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    (root / "current.json").write_text(
        json.dumps({"artifact_id": artifact_id, "directory": directory}),
        encoding="utf-8",
    )
    return weights_path


def write_price_first_artifact(root, monkeypatch, *, audit_passed=True):
    feature_count = len(PHOENIX_SURROGATE_FEATURE_NAMES)
    output_count = len(PHOENIX_PRICE_FIRST_OUTPUT_NAMES)
    temporary_weights = root / "price-first-weights.npz"
    with temporary_weights.open("wb") as handle:
        np.savez_compressed(
            handle,
            feature_mean=np.zeros(feature_count),
            feature_scale=np.ones(feature_count),
            target_mean=np.asarray([1.0] + [0.0] * (output_count - 1)),
            target_scale=np.ones(output_count),
            trunk_layers=np.asarray(1, dtype=np.int64),
            payoff_layers=np.asarray(1, dtype=np.int64),
            event_layers=np.asarray(1, dtype=np.int64),
            trunk_weight_0=np.zeros((feature_count, 2), dtype=np.float32),
            trunk_bias_0=np.zeros(2, dtype=np.float32),
            price_weight=np.zeros((2, 1), dtype=np.float32),
            price_bias=np.zeros(1, dtype=np.float32),
            payoff_weight_0=np.zeros((2, 6), dtype=np.float32),
            payoff_bias_0=np.zeros(6, dtype=np.float32),
            event_weight_0=np.zeros((2, 5), dtype=np.float32),
            event_bias_0=np.zeros(5, dtype=np.float32),
        )
    checksum = file_sha256(temporary_weights)
    identity = {
        "artifact_schema_version": PHOENIX_PRICE_FIRST_ARTIFACT_VERSION,
        "model_version": PHOENIX_PRICE_FIRST_MODEL_VERSION,
        "model_type": "numpy-branched-mlp",
        "feature_schema_version": PHOENIX_SURROGATE_FEATURE_VERSION,
        "market_data_version": EQUITY_MARKET_TERM_STRUCTURE_VERSION,
        "label_model_version": EQUITY_GBM_PIECEWISE_MODEL_VERSION,
        "label_schema_version": PHOENIX_SURROGATE_LABEL_VERSION,
        "contract_version": PhoenixPayoff.contract_version,
        "feature_names": list(PHOENIX_SURROGATE_FEATURE_NAMES),
        "output_names": list(PHOENIX_PRICE_FIRST_OUTPUT_NAMES),
        "runtime_policy": "shadow-only",
        "deployment_status": "shadow_approved",
        "dataset_id": PHOENIX_PRICE_FIRST_DEVELOPMENT_DATASET_ID,
        "development_dataset_id": PHOENIX_PRICE_FIRST_DEVELOPMENT_DATASET_ID,
        "observation_dataset_id": PHOENIX_PRICE_FIRST_OBSERVATION_DATASET_ID,
        "audit_dataset_id": PHOENIX_PRICE_FIRST_AUDIT_DATASET_ID,
        "audit_version": PHOENIX_PRICE_FIRST_AUDIT_VERSION,
        "audit_decision": "passed",
        "model_specification_commit": (PHOENIX_PRICE_FIRST_FROZEN_SPECIFICATION_COMMIT),
        "selected_auxiliary_loss_weight": (PHOENIX_PRICE_FIRST_FROZEN_AUXILIARY_WEIGHT),
        "audit_report_sha256": PHOENIX_PRICE_FIRST_AUDIT_REPORT_SHA256,
        "audit_uncertainty_policy": {"policy_id": PHOENIX_PRICE_FIRST_AUDIT_POLICY_ID},
        "audit_acceptance": {
            "passed": audit_passed,
            "evaluation_dataset_id": PHOENIX_PRICE_FIRST_AUDIT_DATASET_ID,
            "checks": {"audit_mae": {"passed": audit_passed}},
        },
        "training_config": {"model_random_state": 143},
        "training_domain": {
            name: list(bounds) for name, bounds in DEFAULT_TRAINING_DOMAIN.items()
        },
        "weights_sha256": checksum,
        "numpy_parity": {
            "passed": True,
            "maximum_absolute_output_difference": 1e-7,
            "absolute_tolerance": (PHOENIX_PRICE_FIRST_NUMPY_PARITY_ABSOLUTE_TOLERANCE),
        },
    }
    artifact_id = json_sha256(identity)
    directory = artifact_id.removeprefix("sha256:")
    artifact_directory = root / directory
    artifact_directory.mkdir(parents=True)
    temporary_weights.replace(artifact_directory / "weights.npz")
    manifest = {
        **identity,
        "artifact_id": artifact_id,
        "artifact_identity": identity,
        "files": {"weights.npz": checksum},
    }
    (artifact_directory / "manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    (root / "current.json").write_text(
        json.dumps({"artifact_id": artifact_id, "directory": directory}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        surrogate_service,
        "PHOENIX_PRICE_FIRST_APPROVED_ARTIFACT_ID",
        artifact_id,
    )
    return artifact_directory / "manifest.json"


def enabled_settings(root, *, allow_unapproved=False):
    return SurrogateSettings(
        enabled=True,
        artifact_root=root,
        allow_unapproved=allow_unapproved,
    )


def test_status_and_shadow_evaluation_load_approved_checksum_artifact(tmp_path):
    write_artifact(tmp_path)
    settings = enabled_settings(tmp_path)

    status = get_surrogate_status(settings)
    shadow = evaluate_surrogate_shadow(
        market=make_market(),
        terms=TERMS,
        contract_reference_spot=100.0,
        reference_price=1.02,
        reference_standard_error=0.01,
        settings=settings,
    )

    assert status["available"] is True
    assert status["deployment_status"] == "shadow_approved"
    assert shadow["status"] == "success"
    assert shadow["used_for_price"] is False
    assert shadow["surrogate_price"] == pytest.approx(1.0)
    assert shadow["absolute_error"] == pytest.approx(0.02)
    assert shadow["error_to_reference_standard_error"] == pytest.approx(2.0)
    assert shadow["input_diagnostics"]["maximum_standardized_feature_distance"] >= 0.0


def test_research_only_artifact_requires_explicit_override(tmp_path):
    write_artifact(tmp_path, deployment_status="research_only")

    allowed = get_surrogate_status(enabled_settings(tmp_path, allow_unapproved=True))
    rejected_after_override_load = get_surrogate_status(enabled_settings(tmp_path))

    assert allowed["available"] is True
    assert allowed["deployment_status"] == "research_only"
    assert rejected_after_override_load["available"] is False
    assert "did not pass" in rejected_after_override_load["reason"]


def test_checksum_mismatch_is_rejected(tmp_path):
    weights_path = write_artifact(tmp_path)
    weights_path.write_bytes(weights_path.read_bytes() + b"corrupt")

    status = get_surrogate_status(enabled_settings(tmp_path))

    assert status["available"] is False
    assert status["reason"] == "surrogate weights checksum mismatch"


def test_malformed_training_domain_is_rejected(tmp_path):
    write_artifact(tmp_path)
    manifest_path = tmp_path / ("a" * 64) / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["training_domain"]["spot_ratio"] = ["bad", 1.5]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    status = get_surrogate_status(enabled_settings(tmp_path))

    assert status["available"] is False
    assert status["reason"] == "surrogate manifest training domain is invalid"


def test_shadow_refuses_out_of_domain_inputs(tmp_path):
    write_artifact(tmp_path)

    shadow = evaluate_surrogate_shadow(
        market=make_market(spot=200.0),
        terms=TERMS,
        contract_reference_spot=100.0,
        reference_price=1.0,
        reference_standard_error=0.01,
        settings=enabled_settings(tmp_path),
    )

    assert shadow["status"] == "out_of_domain"
    assert any("spot_ratio" in violation for violation in shadow["violations"])


def test_disabled_shadow_does_not_touch_artifacts(tmp_path):
    settings = SurrogateSettings(
        enabled=False,
        artifact_root=tmp_path,
        allow_unapproved=False,
    )

    assert get_surrogate_status(settings)["reason"] == "disabled"
    assert (
        evaluate_surrogate_shadow(
            market=make_market(),
            terms=TERMS,
            contract_reference_spot=100.0,
            reference_price=1.0,
            reference_standard_error=0.01,
            settings=settings,
        )
        is None
    )


def test_audit_approved_price_first_artifact_loads_in_shadow_only(
    tmp_path,
    monkeypatch,
):
    write_price_first_artifact(tmp_path, monkeypatch)

    status = get_surrogate_status(enabled_settings(tmp_path))
    shadow = evaluate_surrogate_shadow(
        market=make_market(),
        terms=TERMS,
        contract_reference_spot=100.0,
        reference_price=1.02,
        reference_standard_error=0.01,
        settings=enabled_settings(tmp_path),
    )

    assert status["available"] is True
    assert status["model_version"] == PHOENIX_PRICE_FIRST_MODEL_VERSION
    assert shadow["status"] == "success"
    assert shadow["used_for_price"] is False
    assert shadow["surrogate_price"] == pytest.approx(1.0)
    assert set(shadow["cashflow_components"]) == {
        "coupon_pv",
        "autocall_principal_pv",
        "maturity_protected_pv",
        "maturity_downside_pv",
    }


def test_price_first_artifact_rejects_forged_audit_acceptance(
    tmp_path,
    monkeypatch,
):
    write_price_first_artifact(tmp_path, monkeypatch, audit_passed=False)

    status = get_surrogate_status(enabled_settings(tmp_path, allow_unapproved=True))

    assert status["available"] is False
    assert "audit acceptance is invalid" in status["reason"]


def test_price_first_runtime_rejects_an_unapproved_artifact_id(
    tmp_path,
    monkeypatch,
):
    write_price_first_artifact(tmp_path, monkeypatch)
    monkeypatch.setattr(
        surrogate_service,
        "PHOENIX_PRICE_FIRST_APPROVED_ARTIFACT_ID",
        PHOENIX_PRICE_FIRST_APPROVED_ARTIFACT_ID,
    )

    status = get_surrogate_status(enabled_settings(tmp_path))

    assert status["available"] is False
    assert "artifact_id is not audit-approved" in status["reason"]
