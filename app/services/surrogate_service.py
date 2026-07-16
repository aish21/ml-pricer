import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Mapping

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
    NumpyMLPSurrogate,
    SurrogateModelError,
    file_sha256,
    load_numpy_mlp_artifact,
)

from app.services.product_registry import REPO_ROOT


DEFAULT_SURROGATE_ROOT = REPO_ROOT / "data" / "surrogates" / "phoenix-v3" / "artifacts"
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
    model: NumpyMLPSurrogate
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
    expected = {
        "artifact_schema_version": PHOENIX_SURROGATE_ARTIFACT_VERSION,
        "model_version": PHOENIX_SURROGATE_MODEL_VERSION,
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
    if output_names not in (
        ["price"],
        list(PHOENIX_PAYOFF_AWARE_MODEL_OUTPUT_NAMES),
    ):
        raise SurrogateArtifactInvalidError(
            "surrogate manifest output names are incompatible"
        )
    if manifest.get("artifact_id") != artifact_id:
        raise SurrogateArtifactInvalidError("surrogate artifact id mismatch")
    deployment_status = manifest.get("deployment_status")
    if deployment_status not in {"shadow_approved", "research_only"}:
        raise SurrogateArtifactInvalidError("surrogate deployment status is invalid")
    if deployment_status != "shadow_approved" and not settings.allow_unapproved:
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
    try:
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
            "model_version": PHOENIX_SURROGATE_MODEL_VERSION,
            "reason": "disabled",
        }
    try:
        bundle = _load_bundle(active)
    except SurrogateServiceError as exc:
        return {
            "enabled": True,
            "mode": "shadow-only",
            "available": False,
            "model_version": PHOENIX_SURROGATE_MODEL_VERSION,
            "reason": str(exc),
        }
    return {
        "enabled": True,
        "mode": "shadow-only",
        "available": True,
        "model_version": PHOENIX_SURROGATE_MODEL_VERSION,
        "artifact_id": bundle.artifact_id,
        "deployment_status": bundle.deployment_status,
        "dataset_id": bundle.manifest.get("dataset_id"),
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
            "model_version": PHOENIX_SURROGATE_MODEL_VERSION,
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
            "model_version": PHOENIX_SURROGATE_MODEL_VERSION,
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
    except (SurrogateContractError, SurrogateModelError) as exc:
        return {
            "status": "error",
            "mode": "shadow-only",
            "artifact_id": bundle.artifact_id,
            "model_version": PHOENIX_SURROGATE_MODEL_VERSION,
            "reason": str(exc),
        }
    if not math.isfinite(prediction) or prediction < 0.0 or prediction > 5.0:
        return {
            "status": "error",
            "mode": "shadow-only",
            "artifact_id": bundle.artifact_id,
            "model_version": PHOENIX_SURROGATE_MODEL_VERSION,
            "reason": "surrogate prediction failed output validation",
        }
    output_map = dict(zip(bundle.model.output_names, output_values.tolist()))
    if bundle.model.output_names == PHOENIX_PAYOFF_AWARE_MODEL_OUTPUT_NAMES:
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
                "model_version": PHOENIX_SURROGATE_MODEL_VERSION,
                "reason": "surrogate payoff-aware outputs failed validation",
            }
    absolute_error = abs(prediction - float(reference_price))
    standard_error = float(reference_standard_error)
    result = {
        "status": "success",
        "mode": "shadow-only",
        "used_for_price": False,
        "artifact_id": bundle.artifact_id,
        "model_version": PHOENIX_SURROGATE_MODEL_VERSION,
        "dataset_id": bundle.manifest.get("dataset_id"),
        "surrogate_price": prediction,
        "reference_price": float(reference_price),
        "absolute_error": absolute_error,
        "error_to_reference_standard_error": (
            absolute_error / standard_error if standard_error > 0.0 else None
        ),
        "latency_ms": int(round((time.perf_counter() - started) * 1_000)),
    }
    if bundle.model.output_names == PHOENIX_PAYOFF_AWARE_MODEL_OUTPUT_NAMES:
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
