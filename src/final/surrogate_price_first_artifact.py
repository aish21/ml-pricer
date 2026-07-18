import json
import os
import platform
import time
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .surrogate_contract import (
    PHOENIX_SURROGATE_FEATURE_NAMES,
    PHOENIX_SURROGATE_FEATURE_VERSION,
    PHOENIX_SURROGATE_LABEL_VERSION,
)
from .surrogate_hazard_data import PhoenixHazardDataset
from .surrogate_model import (
    file_sha256,
    json_sha256,
    load_numpy_branched_mlp_artifact,
)
from .surrogate_price_first import (
    _fit_frozen_phoenix_price_first_candidate,
)
from .surrogate_price_first_contract import (
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
    PHOENIX_PRICE_FIRST_RESEARCH_VERSION,
)
from .surrogate_trainer import SurrogateTrainingError


DEFAULT_PRICE_FIRST_ARTIFACT_ROOT = (
    Path("data") / "surrogates" / "phoenix-price-first-v1" / "artifacts"
)


def _require_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 71
        or not value.startswith("sha256:")
        or any(character not in "0123456789abcdef" for character in value[7:])
    ):
        raise SurrogateTrainingError(f"{label} is invalid")
    return value


def _load_audit_report(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SurrogateTrainingError(
            "sealed price-first audit report is unreadable"
        ) from exc
    if not isinstance(payload, dict):
        raise SurrogateTrainingError(
            "sealed price-first audit report must be an object"
        )
    return payload


def _validate_sealed_audit_report(
    report: Mapping[str, Any],
    development_dataset: PhoenixHazardDataset,
) -> None:
    base_id = development_dataset.base.metadata.get("dataset_id")
    observation_id = development_dataset.metadata.get("dataset_id")
    expected = {
        "audit_version": PHOENIX_PRICE_FIRST_AUDIT_VERSION,
        "model_research_version": PHOENIX_PRICE_FIRST_RESEARCH_VERSION,
        "model_specification_commit": (PHOENIX_PRICE_FIRST_FROZEN_SPECIFICATION_COMMIT),
        "development_dataset_id": PHOENIX_PRICE_FIRST_DEVELOPMENT_DATASET_ID,
        "observation_dataset_id": PHOENIX_PRICE_FIRST_OBSERVATION_DATASET_ID,
        "audit_decision": "passed",
        "runtime_eligible": False,
        "artifact_written": False,
        "deployment_status": "research_only",
    }
    if (
        base_id != PHOENIX_PRICE_FIRST_DEVELOPMENT_DATASET_ID
        or observation_id != PHOENIX_PRICE_FIRST_OBSERVATION_DATASET_ID
    ):
        raise SurrogateTrainingError(
            "price-first packaging datasets do not match the audited model"
        )
    for name, expected_value in expected.items():
        if report.get(name) != expected_value:
            raise SurrogateTrainingError(
                f"sealed price-first audit report {name} is incompatible"
            )
    _require_sha256(base_id, "development dataset id")
    _require_sha256(observation_id, "observation dataset id")
    audit_id = _require_sha256(report.get("audit_dataset_id"), "audit dataset id")
    if audit_id != PHOENIX_PRICE_FIRST_AUDIT_DATASET_ID:
        raise SurrogateTrainingError("sealed price-first audit dataset is incompatible")
    acceptance = report.get("acceptance")
    if (
        not isinstance(acceptance, Mapping)
        or acceptance.get("passed") is not True
        or acceptance.get("evaluation_dataset_id") != audit_id
    ):
        raise SurrogateTrainingError(
            "sealed price-first audit acceptance is incompatible"
        )
    checks = acceptance.get("checks")
    if (
        not isinstance(checks, Mapping)
        or not checks
        or any(
            not isinstance(check, Mapping) or check.get("passed") is not True
            for check in checks.values()
        )
    ):
        raise SurrogateTrainingError("sealed price-first audit checks did not all pass")
    policy = report.get("audit_uncertainty_policy")
    if not isinstance(policy, Mapping):
        raise SurrogateTrainingError(
            "sealed price-first audit uncertainty policy is missing"
        )
    policy_id = _require_sha256(
        policy.get("policy_id"),
        "audit uncertainty policy id",
    )
    if policy_id != PHOENIX_PRICE_FIRST_AUDIT_POLICY_ID:
        raise SurrogateTrainingError(
            "sealed price-first audit uncertainty policy is incompatible"
        )
    evaluation = report.get("audit_evaluation")
    acceptance_policy = (
        evaluation.get("uncertainty_policy")
        if isinstance(evaluation, Mapping)
        else None
    )
    if (
        not isinstance(acceptance_policy, Mapping)
        or acceptance_policy.get("policy_id") != policy_id
    ):
        raise SurrogateTrainingError(
            "sealed price-first audit uncertainty policies do not match"
        )
    frozen_model = report.get("frozen_model")
    if (
        not isinstance(frozen_model, Mapping)
        or frozen_model.get("specification_commit")
        != PHOENIX_PRICE_FIRST_FROZEN_SPECIFICATION_COMMIT
        or frozen_model.get("selected_auxiliary_loss_weight")
        != PHOENIX_PRICE_FIRST_FROZEN_AUXILIARY_WEIGHT
        or not isinstance(frozen_model.get("training_config"), Mapping)
    ):
        raise SurrogateTrainingError("sealed price-first frozen model is incompatible")


def _state_linear_arrays(
    state: Mapping[str, Any],
    prefix: str,
) -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
    indices = sorted(
        {
            int(name.split(".")[1])
            for name in state
            if name.startswith(f"{prefix}.") and name.endswith(".weight")
        }
    )
    if not indices:
        raise SurrogateTrainingError(f"price-first {prefix} layers are missing")
    weights = tuple(
        state[f"{prefix}.{index}.weight"]
        .detach()
        .cpu()
        .numpy()
        .T.astype(np.float32, copy=True)
        for index in indices
    )
    biases = tuple(
        state[f"{prefix}.{index}.bias"]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32, copy=True)
        for index in indices
    )
    return weights, biases


def _write_branched_weights(path: Path, surrogate) -> None:
    state = surrogate.network.state_dict()
    trunk_weights, trunk_biases = _state_linear_arrays(state, "trunk")
    payoff_weights, payoff_biases = _state_linear_arrays(state, "payoff_head")
    event_weights, event_biases = _state_linear_arrays(state, "event_head")
    arrays: dict[str, np.ndarray] = {
        "feature_mean": np.asarray(surrogate.feature_mean, dtype=np.float64),
        "feature_scale": np.asarray(surrogate.feature_scale, dtype=np.float64),
        "target_mean": np.asarray(surrogate.target_mean, dtype=np.float64),
        "target_scale": np.asarray(surrogate.target_scale, dtype=np.float64),
        "trunk_layers": np.asarray(len(trunk_weights), dtype=np.int64),
        "payoff_layers": np.asarray(len(payoff_weights), dtype=np.int64),
        "event_layers": np.asarray(len(event_weights), dtype=np.int64),
        "price_weight": (
            state["price_head.weight"]
            .detach()
            .cpu()
            .numpy()
            .T.astype(np.float32, copy=True)
        ),
        "price_bias": (
            state["price_head.bias"]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32, copy=True)
        ),
    }
    for prefix, weights, biases in (
        ("trunk", trunk_weights, trunk_biases),
        ("payoff", payoff_weights, payoff_biases),
        ("event", event_weights, event_biases),
    ):
        for index, (weight, bias) in enumerate(zip(weights, biases)):
            arrays[f"{prefix}_weight_{index}"] = weight
            arrays[f"{prefix}_bias_{index}"] = bias
    with Path(path).open("wb") as handle:
        np.savez_compressed(handle, **arrays)


def _generation_environment() -> dict[str, str]:
    environment = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    for package in ("numpy", "torch"):
        try:
            environment[package] = version(package)
        except PackageNotFoundError:
            environment[package] = "unknown"
    return environment


def _discard_temporary(directory: Path) -> None:
    if not directory.exists():
        return
    for child in directory.iterdir():
        child.unlink()
    directory.rmdir()


def _stable_frozen_model(value: Any) -> Any:
    stable = json.loads(json.dumps(value))
    if isinstance(stable, dict) and isinstance(stable.get("fit"), dict):
        stable["fit"].pop("fit_seconds", None)
    return stable


def package_audit_approved_price_first_artifact(
    *,
    development_dataset: PhoenixHazardDataset,
    audit_report_path: Path,
    output_root: Path = DEFAULT_PRICE_FIRST_ARTIFACT_ROOT,
    verbose: bool = True,
) -> dict[str, Any]:
    """Reproduce, export, verify, and publish the sealed-audit model to shadow."""
    report_path = Path(audit_report_path)
    report = _load_audit_report(report_path)
    if file_sha256(report_path) != PHOENIX_PRICE_FIRST_AUDIT_REPORT_SHA256:
        raise SurrogateTrainingError(
            "sealed price-first audit report checksum is incompatible"
        )
    _validate_sealed_audit_report(report, development_dataset)
    started = time.perf_counter()
    surrogate, reproduced_frozen_model = _fit_frozen_phoenix_price_first_candidate(
        development_dataset
    )
    if _stable_frozen_model(reproduced_frozen_model) != _stable_frozen_model(
        report.get("frozen_model")
    ):
        raise SurrogateTrainingError(
            "sealed price-first frozen model does not reproduce during packaging"
        )

    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    temporary = root / f".packaging-{os.getpid()}-{time.time_ns()}"
    temporary.mkdir()
    try:
        weights_path = temporary / "weights.npz"
        _write_branched_weights(weights_path, surrogate)
        runtime_model = load_numpy_branched_mlp_artifact(
            weights_path,
            PHOENIX_SURROGATE_FEATURE_NAMES,
            PHOENIX_PRICE_FIRST_OUTPUT_NAMES,
        )
        source_outputs = surrogate.predict_raw_outputs(development_dataset.base.X)
        runtime_outputs = runtime_model.predict_raw_outputs(development_dataset.base.X)
        maximum_difference = float(np.max(np.abs(source_outputs - runtime_outputs)))
        if (
            not np.isfinite(maximum_difference)
            or maximum_difference > PHOENIX_PRICE_FIRST_NUMPY_PARITY_ABSOLUTE_TOLERANCE
        ):
            raise SurrogateTrainingError(
                "price-first NumPy export failed prediction parity: "
                f"maximum absolute difference {maximum_difference:.3e} exceeds "
                f"{PHOENIX_PRICE_FIRST_NUMPY_PARITY_ABSOLUTE_TOLERANCE:.3e}"
            )
        parity = {
            "cases": int(len(development_dataset.base.X)),
            "maximum_absolute_output_difference": maximum_difference,
            "absolute_tolerance": (PHOENIX_PRICE_FIRST_NUMPY_PARITY_ABSOLUTE_TOLERANCE),
            "passed": True,
        }
        base_metadata = development_dataset.base.metadata
        audit_id = report["audit_dataset_id"]
        audit_policy = report["audit_uncertainty_policy"]
        weights_checksum = file_sha256(weights_path)
        identity = {
            "artifact_schema_version": PHOENIX_PRICE_FIRST_ARTIFACT_VERSION,
            "model_version": PHOENIX_PRICE_FIRST_MODEL_VERSION,
            "model_type": "numpy-branched-mlp",
            "feature_schema_version": PHOENIX_SURROGATE_FEATURE_VERSION,
            "market_data_version": base_metadata["market_data_version"],
            "label_model_version": base_metadata["label_model_version"],
            "label_schema_version": PHOENIX_SURROGATE_LABEL_VERSION,
            "contract_version": base_metadata["contract_version"],
            "feature_names": list(PHOENIX_SURROGATE_FEATURE_NAMES),
            "output_names": list(PHOENIX_PRICE_FIRST_OUTPUT_NAMES),
            "runtime_policy": "shadow-only",
            "deployment_status": "shadow_approved",
            "dataset_id": base_metadata["dataset_id"],
            "development_dataset_id": base_metadata["dataset_id"],
            "observation_dataset_id": development_dataset.metadata["dataset_id"],
            "audit_dataset_id": audit_id,
            "model_research_version": PHOENIX_PRICE_FIRST_RESEARCH_VERSION,
            "audit_version": PHOENIX_PRICE_FIRST_AUDIT_VERSION,
            "audit_decision": "passed",
            "model_specification_commit": (
                PHOENIX_PRICE_FIRST_FROZEN_SPECIFICATION_COMMIT
            ),
            "selected_auxiliary_loss_weight": (
                PHOENIX_PRICE_FIRST_FROZEN_AUXILIARY_WEIGHT
            ),
            "training_config": reproduced_frozen_model["training_config"],
            "audit_uncertainty_policy": audit_policy,
            "audit_gate_config": report["audit_gate_config"],
            "audit_acceptance": report["acceptance"],
            "audit_report_sha256": file_sha256(report_path),
            "training_domain": base_metadata["training_domain"],
            "weights_sha256": weights_checksum,
            "numpy_parity": parity,
        }
        artifact_id = json_sha256(identity)
        if artifact_id != PHOENIX_PRICE_FIRST_APPROVED_ARTIFACT_ID:
            raise SurrogateTrainingError(
                "price-first export does not reproduce the approved artifact id"
            )
        directory_name = artifact_id.removeprefix("sha256:")
        manifest = {
            **identity,
            "artifact_id": artifact_id,
            "artifact_identity": identity,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "artifact_generation_environment": _generation_environment(),
            "development_dataset": {
                "dataset_id": base_metadata["dataset_id"],
                "dataset_schema_version": base_metadata["dataset_schema_version"],
                "n_samples": base_metadata["n_samples"],
                "split_counts": base_metadata["split_counts"],
                "config": base_metadata["config"],
                "generation_environment": base_metadata["generation_environment"],
            },
            "observation_dataset": {
                "dataset_id": development_dataset.metadata["dataset_id"],
                "n_samples": development_dataset.metadata["n_samples"],
            },
            "audit_evaluation": report["audit_evaluation"],
            "greek_validation": report["greek_validation"],
            "files": {"weights.npz": weights_checksum},
            "packaging_seconds": time.perf_counter() - started,
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        final_directory = root / directory_name
        if final_directory.exists():
            existing_manifest = _load_audit_report(final_directory / "manifest.json")
            existing_weights = final_directory / "weights.npz"
            existing_identity = existing_manifest.get("artifact_identity")
            if (
                existing_manifest.get("artifact_id") != artifact_id
                or not isinstance(existing_identity, dict)
                or json_sha256(existing_identity) != artifact_id
                or existing_manifest.get("files") != {"weights.npz": weights_checksum}
                or not existing_weights.is_file()
                or file_sha256(existing_weights) != weights_checksum
            ):
                raise SurrogateTrainingError(
                    "existing price-first artifact does not match the export"
                )
            _discard_temporary(temporary)
        else:
            os.replace(temporary, final_directory)
        pointer_temp = root / f".current-{os.getpid()}-{time.time_ns()}.tmp"
        pointer_temp.write_text(
            json.dumps(
                {
                    "artifact_id": artifact_id,
                    "directory": directory_name,
                    "deployment_status": "shadow_approved",
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(pointer_temp, root / "current.json")
    except Exception:
        _discard_temporary(temporary)
        raise

    if verbose:
        print(
            "[PhoenixPriceFirstArtifact] "
            f"artifact={artifact_id} "
            f"parity_max_abs={maximum_difference:.3e} "
            "status=shadow_approved",
            flush=True,
        )
    return manifest
