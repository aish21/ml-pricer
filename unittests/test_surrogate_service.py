import json
from datetime import datetime, timezone

import numpy as np
import pytest

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
from src.final.surrogate_model import file_sha256


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
