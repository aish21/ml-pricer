import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Mapping

import numpy as np

from src.final.market import (
    EQUITY_GBM_PIECEWISE_MODEL_VERSION,
    EQUITY_MARKET_TERM_STRUCTURE_VERSION,
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
    PHOENIX_PAYOFF_AWARE_MODEL_OUTPUT_NAMES,
    PHOENIX_EVENT_TARGET_NAMES,
    PHOENIX_PRICE_COMPONENT_NAMES,
    SurrogateContractError,
    domain_violations,
    extract_phoenix_surrogate_features,
)
from src.final.surrogate_model import (
    NumpyBranchedMLPSurrogate,
    NumpyMLPSurrogate,
    SurrogateModelError,
    file_sha256,
    json_sha256,
    load_numpy_branched_mlp_artifact,
    load_numpy_mlp_artifact,
)
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

from app.services.product_registry import REPO_ROOT


DEFAULT_SURROGATE_ROOT = (
    REPO_ROOT / "data" / "surrogates" / "phoenix-price-first-v1" / "artifacts"
)
DEFAULT_EXPANDED_EXPERIMENT_SUMMARY = (
    REPO_ROOT / "final" / "research_candidates" / "experiment_summary.json"
)
MAX_POINTER_BYTES = 64 * 1024
MAX_MANIFEST_BYTES = 2 * 1024 * 1024
MAX_WEIGHTS_BYTES = 32 * 1024 * 1024
MAX_CACHED_ARTIFACT_ROOTS = 16


class SurrogateServiceError(Exception):
    pass


class SurrogateArtifactUnavailableError(SurrogateServiceError):
    pass


class SurrogateArtifactInvalidError(SurrogateServiceError):
    pass


@dataclass(frozen=True)
class SurrogateSettings:
    enabled: bool
    artifact_root: Path
    allow_unapproved: bool

    @classmethod
    def from_env(cls) -> "SurrogateSettings":
        enabled = os.getenv("PHOENIX_SURROGATE_SHADOW_ENABLED", "false").strip().lower()
        allow_unapproved = (
            os.getenv("PHOENIX_SURROGATE_ALLOW_UNAPPROVED", "false").strip().lower()
        )
        return cls(
            enabled=enabled in {"1", "true", "yes", "on"},
            artifact_root=Path(
                os.getenv("PHOENIX_SURROGATE_DIR", str(DEFAULT_SURROGATE_ROOT))
            ),
            allow_unapproved=allow_unapproved in {"1", "true", "yes", "on"},
        )


@dataclass(frozen=True)
class SurrogateBundle:
    artifact_id: str
    deployment_status: str
    manifest: dict[str, Any]
    model: NumpyMLPSurrogate | NumpyBranchedMLPSurrogate
    pointer_mtime_ns: int


_CACHE_LOCK = Lock()
_CACHED_BUNDLES: dict[str, SurrogateBundle] = {}


def _read_json(path: Path, *, max_bytes: int, label: str) -> dict[str, Any]:
    try:
        size = path.stat().st_size
        if size < 2 or size > max_bytes:
            raise SurrogateArtifactInvalidError(f"surrogate {label} size is invalid")
        payload = json.loads(path.read_text(encoding="utf-8"))
    except SurrogateArtifactInvalidError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SurrogateArtifactInvalidError(f"surrogate {label} is unreadable") from exc
    if not isinstance(payload, dict):
        raise SurrogateArtifactInvalidError(f"surrogate {label} must be an object")
    return payload


def _validate_price_first_audit_binding(
    manifest: Mapping[str, Any],
    artifact_id: str,
) -> None:
    identity = manifest.get("artifact_identity")
    if not isinstance(identity, dict) or json_sha256(identity) != artifact_id:
        raise SurrogateArtifactInvalidError(
            "surrogate artifact identity checksum mismatch"
        )
    if any(manifest.get(name) != value for name, value in identity.items()):
        raise SurrogateArtifactInvalidError(
            "surrogate artifact identity fields do not match"
        )
    expected = {
        "model_type": "numpy-branched-mlp",
        "artifact_id": PHOENIX_PRICE_FIRST_APPROVED_ARTIFACT_ID,
        "deployment_status": "shadow_approved",
        "development_dataset_id": PHOENIX_PRICE_FIRST_DEVELOPMENT_DATASET_ID,
        "dataset_id": PHOENIX_PRICE_FIRST_DEVELOPMENT_DATASET_ID,
        "observation_dataset_id": PHOENIX_PRICE_FIRST_OBSERVATION_DATASET_ID,
        "audit_dataset_id": PHOENIX_PRICE_FIRST_AUDIT_DATASET_ID,
        "audit_version": PHOENIX_PRICE_FIRST_AUDIT_VERSION,
        "audit_decision": "passed",
        "model_specification_commit": (PHOENIX_PRICE_FIRST_FROZEN_SPECIFICATION_COMMIT),
        "selected_auxiliary_loss_weight": (PHOENIX_PRICE_FIRST_FROZEN_AUXILIARY_WEIGHT),
        "audit_report_sha256": PHOENIX_PRICE_FIRST_AUDIT_REPORT_SHA256,
    }
    for name, expected_value in expected.items():
        if manifest.get(name) != expected_value:
            raise SurrogateArtifactInvalidError(
                f"surrogate manifest {name} is not audit-approved"
            )
    policy = manifest.get("audit_uncertainty_policy")
    if (
        not isinstance(policy, dict)
        or policy.get("policy_id") != PHOENIX_PRICE_FIRST_AUDIT_POLICY_ID
    ):
        raise SurrogateArtifactInvalidError(
            "surrogate manifest audit policy is not approved"
        )
    acceptance = manifest.get("audit_acceptance")
    checks = acceptance.get("checks") if isinstance(acceptance, dict) else None
    if (
        not isinstance(acceptance, dict)
        or acceptance.get("passed") is not True
        or acceptance.get("evaluation_dataset_id")
        != PHOENIX_PRICE_FIRST_AUDIT_DATASET_ID
        or not isinstance(checks, dict)
        or not checks
        or any(
            not isinstance(check, dict) or check.get("passed") is not True
            for check in checks.values()
        )
    ):
        raise SurrogateArtifactInvalidError(
            "surrogate manifest audit acceptance is invalid"
        )
    parity = manifest.get("numpy_parity")
    if (
        not isinstance(parity, dict)
        or parity.get("passed") is not True
        or parity.get("absolute_tolerance")
        != PHOENIX_PRICE_FIRST_NUMPY_PARITY_ABSOLUTE_TOLERANCE
    ):
        raise SurrogateArtifactInvalidError(
            "surrogate manifest NumPy parity is invalid"
        )
    try:
        maximum_difference = float(parity["maximum_absolute_output_difference"])
    except (KeyError, TypeError, ValueError) as exc:
        raise SurrogateArtifactInvalidError(
            "surrogate manifest NumPy parity is invalid"
        ) from exc
    if (
        not math.isfinite(maximum_difference)
        or maximum_difference < 0.0
        or maximum_difference > PHOENIX_PRICE_FIRST_NUMPY_PARITY_ABSOLUTE_TOLERANCE
    ):
        raise SurrogateArtifactInvalidError(
            "surrogate manifest NumPy parity is invalid"
        )
    if not isinstance(manifest.get("training_config"), dict):
        raise SurrogateArtifactInvalidError(
            "surrogate manifest frozen training config is missing"
        )


def _load_bundle(settings: SurrogateSettings) -> SurrogateBundle:
    root = settings.artifact_root
    pointer_path = root / "current.json"
    if not pointer_path.is_file():
        raise SurrogateArtifactUnavailableError("surrogate current pointer is missing")
    try:
        pointer_mtime_ns = pointer_path.stat().st_mtime_ns
    except OSError as exc:
        raise SurrogateArtifactUnavailableError(
            "surrogate current pointer is unavailable"
        ) from exc
    cache_key = str(root.resolve())
    with _CACHE_LOCK:
        cached = _CACHED_BUNDLES.get(cache_key)
        if (
            cached is not None
            and cached.pointer_mtime_ns == pointer_mtime_ns
            and (
                cached.deployment_status == "shadow_approved"
                or settings.allow_unapproved
            )
        ):
            return cached

    pointer = _read_json(
        pointer_path, max_bytes=MAX_POINTER_BYTES, label="current pointer"
    )
    artifact_id = pointer.get("artifact_id")
    directory = pointer.get("directory")
    if not isinstance(artifact_id, str) or not artifact_id.startswith("sha256:"):
        raise SurrogateArtifactInvalidError("surrogate artifact id is invalid")
    if (
        not isinstance(directory, str)
        or len(directory) != 64
        or any(character not in "0123456789abcdef" for character in directory)
    ):
        raise SurrogateArtifactInvalidError("surrogate artifact directory is invalid")
    if artifact_id != f"sha256:{directory}":
        raise SurrogateArtifactInvalidError(
            "surrogate artifact directory does not match its id"
        )
    artifact_directory = root / directory
    try:
        resolved_directory = artifact_directory.resolve()
        if resolved_directory.parent != root.resolve():
            raise SurrogateArtifactInvalidError(
                "surrogate artifact directory escapes its root"
            )
    except OSError as exc:
        raise SurrogateArtifactInvalidError(
            "surrogate artifact directory is invalid"
        ) from exc
    manifest = _read_json(
        artifact_directory / "manifest.json",
        max_bytes=MAX_MANIFEST_BYTES,
        label="manifest",
    )
    artifact_schema_version = manifest.get("artifact_schema_version")
    if artifact_schema_version == PHOENIX_SURROGATE_ARTIFACT_VERSION:
        expected_model_version = PHOENIX_SURROGATE_MODEL_VERSION
        valid_output_names = (
            ["price"],
            list(PHOENIX_PAYOFF_AWARE_MODEL_OUTPUT_NAMES),
        )
        branched_model = False
    elif artifact_schema_version == PHOENIX_PRICE_FIRST_ARTIFACT_VERSION:
        expected_model_version = PHOENIX_PRICE_FIRST_MODEL_VERSION
        valid_output_names = (list(PHOENIX_PRICE_FIRST_OUTPUT_NAMES),)
        branched_model = True
    else:
        raise SurrogateArtifactInvalidError(
            "surrogate manifest artifact_schema_version is incompatible"
        )
    expected = {
        "model_version": expected_model_version,
        "feature_schema_version": PHOENIX_SURROGATE_FEATURE_VERSION,
        "market_data_version": EQUITY_MARKET_TERM_STRUCTURE_VERSION,
        "label_model_version": EQUITY_GBM_PIECEWISE_MODEL_VERSION,
        "label_schema_version": PHOENIX_SURROGATE_LABEL_VERSION,
        "contract_version": PhoenixPayoff.contract_version,
        "feature_names": list(PHOENIX_SURROGATE_FEATURE_NAMES),
        "runtime_policy": "shadow-only",
    }
    for name, expected_value in expected.items():
        if manifest.get(name) != expected_value:
            raise SurrogateArtifactInvalidError(
                f"surrogate manifest {name} is incompatible"
            )
    output_names = manifest.get("output_names")
    if output_names not in valid_output_names:
        raise SurrogateArtifactInvalidError(
            "surrogate manifest output names are incompatible"
        )
    if manifest.get("artifact_id") != artifact_id:
        raise SurrogateArtifactInvalidError("surrogate artifact id mismatch")
    deployment_status = manifest.get("deployment_status")
    if deployment_status not in {"shadow_approved", "research_only"}:
        raise SurrogateArtifactInvalidError("surrogate deployment status is invalid")
    if branched_model:
        _validate_price_first_audit_binding(manifest, artifact_id)
    if deployment_status != "shadow_approved" and (
        branched_model or not settings.allow_unapproved
    ):
        raise SurrogateArtifactUnavailableError(
            "surrogate artifact did not pass shadow acceptance"
        )
    training_domain = manifest.get("training_domain")
    if not isinstance(training_domain, dict) or set(training_domain) != set(
        DEFAULT_TRAINING_DOMAIN
    ):
        raise SurrogateArtifactInvalidError(
            "surrogate manifest training domain is invalid"
        )
    for bounds in training_domain.values():
        if (
            not isinstance(bounds, (list, tuple))
            or len(bounds) != 2
            or isinstance(bounds[0], bool)
            or isinstance(bounds[1], bool)
        ):
            raise SurrogateArtifactInvalidError(
                "surrogate manifest training domain is invalid"
            )
        try:
            lower, upper = float(bounds[0]), float(bounds[1])
        except (TypeError, ValueError) as exc:
            raise SurrogateArtifactInvalidError(
                "surrogate manifest training domain is invalid"
            ) from exc
        if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
            raise SurrogateArtifactInvalidError(
                "surrogate manifest training domain is invalid"
            )
    files = manifest.get("files")
    expected_checksum = files.get("weights.npz") if isinstance(files, dict) else None
    if not isinstance(expected_checksum, str):
        raise SurrogateArtifactInvalidError("surrogate weights checksum is missing")
    weights_path = artifact_directory / "weights.npz"
    try:
        if weights_path.stat().st_size > MAX_WEIGHTS_BYTES:
            raise SurrogateArtifactInvalidError(
                "surrogate weights exceed the runtime size limit"
            )
        actual_checksum = file_sha256(weights_path)
    except SurrogateArtifactInvalidError:
        raise
    except OSError as exc:
        raise SurrogateArtifactInvalidError("surrogate weights are missing") from exc
    if actual_checksum != expected_checksum:
        raise SurrogateArtifactInvalidError("surrogate weights checksum mismatch")
    if branched_model and manifest.get("weights_sha256") != expected_checksum:
        raise SurrogateArtifactInvalidError(
            "surrogate identity weights checksum mismatch"
        )
    try:
        if branched_model:
            model = load_numpy_branched_mlp_artifact(
                weights_path,
                manifest["feature_names"],
                output_names,
            )
        else:
            model = load_numpy_mlp_artifact(
                weights_path,
                manifest["feature_names"],
                output_names,
            )
    except SurrogateModelError as exc:
        raise SurrogateArtifactInvalidError(str(exc)) from exc
    bundle = SurrogateBundle(
        artifact_id=artifact_id,
        deployment_status=deployment_status,
        manifest=manifest,
        model=model,
        pointer_mtime_ns=pointer_mtime_ns,
    )
    with _CACHE_LOCK:
        if cache_key not in _CACHED_BUNDLES and (
            len(_CACHED_BUNDLES) >= MAX_CACHED_ARTIFACT_ROOTS
        ):
            _CACHED_BUNDLES.pop(next(iter(_CACHED_BUNDLES)))
        _CACHED_BUNDLES[cache_key] = bundle
    return bundle


def clear_surrogate_cache() -> None:
    with _CACHE_LOCK:
        _CACHED_BUNDLES.clear()


def get_surrogate_status(
    settings: SurrogateSettings | None = None,
) -> dict[str, Any]:
    active = settings or SurrogateSettings.from_env()
    if not active.enabled:
        return {
            "enabled": False,
            "mode": "shadow-only",
            "available": False,
            "model_version": PHOENIX_PRICE_FIRST_MODEL_VERSION,
            "reason": "disabled",
        }
    try:
        bundle = _load_bundle(active)
    except SurrogateServiceError as exc:
        return {
            "enabled": True,
            "mode": "shadow-only",
            "available": False,
            "model_version": PHOENIX_PRICE_FIRST_MODEL_VERSION,
            "reason": str(exc),
        }
    return {
        "enabled": True,
        "mode": "shadow-only",
        "available": True,
        "model_version": bundle.manifest["model_version"],
        "artifact_id": bundle.artifact_id,
        "deployment_status": bundle.deployment_status,
        "dataset_id": bundle.manifest.get("dataset_id"),
    }


def _metric_projection(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    projected: dict[str, Any] = {}
    for name in (
        "n_samples",
        "mae",
        "p95_absolute_error",
        "max_absolute_error",
        "rmse",
        "r2",
        "mean_error",
        "within_two_label_se_fraction",
        "within_uncertainty_or_economic_tolerance_fraction",
    ):
        item = value.get(name)
        if isinstance(item, bool):
            continue
        if isinstance(item, (int, float)) and math.isfinite(float(item)):
            projected[name] = item
    return projected


def get_surrogate_audit_evidence(
    settings: SurrogateSettings | None = None,
) -> dict[str, Any]:
    """Return bounded audit evidence without exposing model weights or internals."""
    active = settings or SurrogateSettings.from_env()
    try:
        bundle = _load_bundle(active)
    except SurrogateServiceError as exc:
        return {
            "available": False,
            "reason": str(exc),
            "model_version": PHOENIX_PRICE_FIRST_MODEL_VERSION,
        }

    manifest = bundle.manifest
    evaluation = manifest.get("audit_evaluation")
    evaluation = evaluation if isinstance(evaluation, Mapping) else {}
    acceptance = manifest.get("audit_acceptance")
    acceptance = acceptance if isinstance(acceptance, Mapping) else {}
    checks = acceptance.get("checks")
    checks = checks if isinstance(checks, Mapping) else {}

    acceptance_checks = {}
    for name, check in checks.items():
        if not isinstance(check, Mapping):
            continue
        acceptance_checks[str(name)] = {
            key: check[key]
            for key in ("value", "minimum", "maximum", "passed")
            if key in check
        }

    def project_slices(name: str) -> dict[str, dict[str, Any]]:
        slices = evaluation.get(name)
        if not isinstance(slices, Mapping):
            return {}
        return {
            str(slice_name): _metric_projection(metrics)
            for slice_name, metrics in slices.items()
            if isinstance(metrics, Mapping)
        }

    output_metrics = evaluation.get("output_metrics")
    output_metrics = output_metrics if isinstance(output_metrics, Mapping) else {}
    projected_outputs = {
        str(name): _metric_projection(metrics)
        for name, metrics in output_metrics.items()
        if isinstance(metrics, Mapping) and _metric_projection(metrics)
    }
    price_metrics = _metric_projection(evaluation.get("price_metrics"))
    if not price_metrics:
        for metric_name, check_name in (
            ("mae", "audit_mae"),
            ("p95_absolute_error", "audit_p95_absolute_error"),
            ("r2", "audit_r2"),
        ):
            check = checks.get(check_name)
            if isinstance(check, Mapping):
                value = check.get("value")
                if (
                    not isinstance(value, bool)
                    and isinstance(value, (int, float))
                    and math.isfinite(float(value))
                ):
                    price_metrics[metric_name] = value
    development = manifest.get("development_dataset")
    development = development if isinstance(development, Mapping) else {}
    return {
        "available": True,
        "artifact": {
            "artifact_id": bundle.artifact_id,
            "model_version": manifest.get("model_version"),
            "deployment_status": bundle.deployment_status,
            "runtime_policy": manifest.get("runtime_policy"),
            "created_at": manifest.get("created_at"),
            "contract_version": manifest.get("contract_version"),
            "feature_schema_version": manifest.get("feature_schema_version"),
            "label_model_version": manifest.get("label_model_version"),
        },
        "datasets": {
            "development_dataset_id": manifest.get("development_dataset_id"),
            "development_samples": development.get("n_samples"),
            "audit_dataset_id": manifest.get("audit_dataset_id"),
            "audit_samples": evaluation.get("n_samples"),
            "observation_dataset_id": manifest.get("observation_dataset_id"),
        },
        "sealed_audit": {
            "passed": acceptance.get("passed") is True,
            "audit_version": manifest.get("audit_version"),
            "evaluation_dataset_id": acceptance.get("evaluation_dataset_id"),
            "price_metrics": price_metrics,
            "by_market_regime": project_slices("regime_metrics"),
            "by_moneyness_region": project_slices("moneyness_region_metrics"),
            "by_regime_and_moneyness": project_slices("regime_moneyness_metrics"),
            "output_metrics": projected_outputs,
            "acceptance_checks": acceptance_checks,
        },
        "training_domain": manifest.get("training_domain"),
    }


def get_expanded_surrogate_evidence(
    summary_path: Path | None = None,
) -> dict[str, Any]:
    """Return bounded evidence for expanded products without loading models."""
    path = Path(summary_path) if summary_path else DEFAULT_EXPANDED_EXPERIMENT_SUMMARY
    if not path.is_file():
        return {
            "available": False,
            "reason": "expanded-product experiments have not been run",
        }
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {
            "available": False,
            "reason": "expanded-product experiment summary is unreadable",
        }
    products = []
    for item in payload.get("products") or []:
        if not isinstance(item, Mapping):
            continue
        audit = item.get("sealed_audit")
        audit = audit if isinstance(audit, Mapping) else {}
        raw_checks = audit.get("checks")
        raw_checks = raw_checks if isinstance(raw_checks, Mapping) else {}
        checks = {
            str(name): {
                key: check[key]
                for key in ("value", "minimum", "maximum", "passed")
                if key in check
            }
            for name, check in raw_checks.items()
            if isinstance(check, Mapping)
        }
        products.append(
            {
                "product_key": item.get("product_key"),
                "contract_version": item.get("contract_version"),
                "experiment_id": item.get("experiment_id"),
                "status": item.get("status"),
                "runtime_approved": item.get("runtime_approved") is True,
                "development_dataset_id": item.get("development_dataset_id"),
                "audit_dataset_id": item.get("audit_dataset_id"),
                "datasets": item.get("datasets"),
                "sealed_audit": {
                    "passed": audit.get("passed") is True,
                    "metrics": _metric_projection(audit.get("metrics")),
                    "checks": checks,
                },
            }
        )
    return {
        "available": bool(products),
        "experiment_version": payload.get("experiment_version"),
        "generated_at": payload.get("generated_at"),
        "runtime_policy_changed": payload.get("runtime_policy_changed") is True,
        "products": products,
    }


def _validation_metrics(manifest: Mapping[str, Any]) -> dict[str, Any] | None:
    """Expose a small, presentation-safe subset of the frozen audit evidence."""
    acceptance = manifest.get("audit_acceptance")
    if not isinstance(acceptance, Mapping) or acceptance.get("passed") is not True:
        return None
    checks = acceptance.get("checks")
    if not isinstance(checks, Mapping):
        return None

    values: dict[str, float] = {}
    for public_name, check_name in (
        ("mean_absolute_error", "audit_mae"),
        ("p95_absolute_error", "audit_p95_absolute_error"),
        ("r_squared", "audit_r2"),
    ):
        check = checks.get(check_name)
        if not isinstance(check, Mapping) or check.get("passed") is not True:
            continue
        try:
            value = float(check["value"])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(value):
            values[public_name] = value

    if not values:
        return None
    return {
        "passed": True,
        "evaluation_dataset_id": acceptance.get("evaluation_dataset_id"),
        **values,
    }


def evaluate_surrogate_shadow(
    *,
    market: EquityMarketTermStructure,
    terms: Mapping[str, Any],
    contract_reference_spot: float,
    reference_price: float,
    reference_standard_error: float,
    settings: SurrogateSettings | None = None,
) -> dict[str, Any] | None:
    active = settings or SurrogateSettings.from_env()
    if not active.enabled:
        return None
    started = time.perf_counter()
    try:
        bundle = _load_bundle(active)
    except SurrogateServiceError as exc:
        return {
            "status": "unavailable",
            "mode": "shadow-only",
            "reason": str(exc),
            "model_version": PHOENIX_PRICE_FIRST_MODEL_VERSION,
            "target_artifact_id": PHOENIX_PRICE_FIRST_APPROVED_ARTIFACT_ID,
        }
    violations = domain_violations(
        market=market,
        terms=terms,
        contract_reference_spot=contract_reference_spot,
        domain=bundle.manifest.get("training_domain"),
    )
    if violations:
        return {
            "status": "out_of_domain",
            "mode": "shadow-only",
            "artifact_id": bundle.artifact_id,
            "target_artifact_id": bundle.artifact_id,
            "model_version": bundle.manifest["model_version"],
            "violations": violations,
        }
    try:
        features = extract_phoenix_surrogate_features(
            market=market,
            terms=terms,
            contract_reference_spot=contract_reference_spot,
        )
        prediction = float(bundle.model.predict(features)[0])
        output_values = bundle.model.predict_outputs(features)[0]
        standardized_distances = np.abs(
            (features - bundle.model.feature_mean) / bundle.model.feature_scale
        )
    except (SurrogateContractError, SurrogateModelError) as exc:
        return {
            "status": "error",
            "mode": "shadow-only",
            "artifact_id": bundle.artifact_id,
            "target_artifact_id": bundle.artifact_id,
            "model_version": bundle.manifest["model_version"],
            "reason": str(exc),
        }
    if not math.isfinite(prediction) or prediction < 0.0 or prediction > 5.0:
        return {
            "status": "error",
            "mode": "shadow-only",
            "artifact_id": bundle.artifact_id,
            "target_artifact_id": bundle.artifact_id,
            "model_version": bundle.manifest["model_version"],
            "reason": "surrogate prediction failed output validation",
        }
    output_map = dict(zip(bundle.model.output_names, output_values.tolist()))
    has_payoff_outputs = all(
        name in bundle.model.output_names
        for name in PHOENIX_PAYOFF_AWARE_MODEL_OUTPUT_NAMES
    )
    if has_payoff_outputs:
        if any(
            output_map[name] < -0.05 for name in PHOENIX_PRICE_COMPONENT_NAMES
        ) or any(
            output_map[name] < -0.05 or output_map[name] > 1.05
            for name in PHOENIX_EVENT_TARGET_NAMES
        ):
            return {
                "status": "error",
                "mode": "shadow-only",
                "artifact_id": bundle.artifact_id,
                "target_artifact_id": bundle.artifact_id,
                "model_version": bundle.manifest["model_version"],
                "reason": "surrogate payoff-aware outputs failed validation",
            }
    absolute_error = abs(prediction - float(reference_price))
    standard_error = float(reference_standard_error)
    result = {
        "status": "success",
        "mode": "shadow-only",
        "used_for_price": False,
        "artifact_id": bundle.artifact_id,
        "target_artifact_id": bundle.artifact_id,
        "model_version": bundle.manifest["model_version"],
        "dataset_id": bundle.manifest.get("dataset_id"),
        "surrogate_price": prediction,
        "reference_price": float(reference_price),
        "absolute_error": absolute_error,
        "error_to_reference_standard_error": (
            absolute_error / standard_error if standard_error > 0.0 else None
        ),
        "latency_ms": round((time.perf_counter() - started) * 1_000, 3),
        "input_diagnostics": {
            "maximum_standardized_feature_distance": float(
                np.max(standardized_distances)
            ),
            "features_above_four_sigma": [
                {
                    "feature": bundle.model.feature_names[index],
                    "standardized_distance": float(standardized_distances[index]),
                }
                for index in np.argsort(-standardized_distances)
                if standardized_distances[index] > 4.0
            ][:5],
        },
    }
    validation_metrics = _validation_metrics(bundle.manifest)
    if validation_metrics is not None:
        result["validation_metrics"] = validation_metrics
    if has_payoff_outputs:
        result["cashflow_components"] = {
            name: output_map[name] for name in PHOENIX_PRICE_COMPONENT_NAMES
        }
        result["event_probabilities"] = {
            name: output_map[name] for name in PHOENIX_EVENT_TARGET_NAMES
        }
        result["cashflow_reconstruction_gap"] = prediction - sum(
            output_map[name] for name in PHOENIX_PRICE_COMPONENT_NAMES
        )
    return result
