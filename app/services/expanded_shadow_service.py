"""Fail-closed shadow inference for the expanded structured-product models."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Mapping

import numpy as np

from app.services.product_registry import REPO_ROOT
from src.final.barrier_reverse_convertible import BarrierReverseConvertibleV1Contract
from src.final.market import EquityMarketTermStructure
from src.final.phoenix_contract import PhoenixSingleV3Contract


ARTIFACT_SCHEMA = "expanded-shadow-artifact-v1"
REGISTRY_SCHEMA = "expanded-shadow-registry-v1"
DEFAULT_ARTIFACT_ROOT = REPO_ROOT / "final" / "shadow_artifacts"
SUPPORTED_PRODUCTS = {
    "phoenix_v3": "phoenix-single-v3",
    "barrier_reverse_convertible": "barrier-reverse-convertible-v1",
}
_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="expanded-shadow")
_CACHE: dict[tuple[str, int, int], "SafeTreeModel"] = {}
_CACHE_LOCK = Lock()


class ExpandedShadowError(RuntimeError):
    pass


class ExpandedShadowArtifactError(ExpandedShadowError):
    pass


class ExpandedShadowDomainError(ExpandedShadowError):
    pass


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name, "true" if default else "false").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _product_prefix(product_key: str) -> str:
    return "PHOENIX_V3" if product_key == "phoenix_v3" else "BRC_V1"


@dataclass(frozen=True)
class ExpandedShadowSettings:
    artifact_root: Path
    enabled: bool
    sample_rate: float
    timeout_ms: float

    @classmethod
    def from_env(cls, product_key: str) -> "ExpandedShadowSettings":
        prefix = _product_prefix(product_key)
        try:
            sample_rate = float(os.getenv(f"{prefix}_SHADOW_SAMPLE_RATE", "0"))
            timeout_ms = float(os.getenv("EXPANDED_SURROGATE_TIMEOUT_MS", "25"))
        except ValueError as exc:
            raise ExpandedShadowError(
                "expanded shadow settings must be numeric"
            ) from exc
        if not 0.0 <= sample_rate <= 1.0:
            raise ExpandedShadowError(f"{prefix}_SHADOW_SAMPLE_RATE must be in [0, 1]")
        if not 1.0 <= timeout_ms <= 1_000.0:
            raise ExpandedShadowError(
                "EXPANDED_SURROGATE_TIMEOUT_MS must be in [1, 1000]"
            )
        return cls(
            artifact_root=Path(
                os.getenv(
                    "EXPANDED_SURROGATE_ARTIFACT_ROOT", str(DEFAULT_ARTIFACT_ROOT)
                )
            ),
            enabled=_env_bool(f"{prefix}_SHADOW_ENABLED"),
            sample_rate=sample_rate,
            timeout_ms=timeout_ms,
        )


def _sha256(data: bytes) -> str:
    return f"sha256:{hashlib.sha256(data).hexdigest()}"


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _load_json(path: Path, *, max_bytes: int = 2_000_000) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size > max_bytes:
        raise ExpandedShadowArtifactError(f"invalid artifact metadata: {path.name}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ExpandedShadowArtifactError(
            f"unreadable artifact metadata: {path.name}"
        ) from exc
    if not isinstance(value, dict):
        raise ExpandedShadowArtifactError(
            f"artifact metadata must be an object: {path.name}"
        )
    return value


def _validate_node(node: Any, feature_count: int, depth: int = 0) -> int:
    if not isinstance(node, dict) or depth > 64:
        raise ExpandedShadowArtifactError("invalid model tree")
    if set(node) == {"leaf"}:
        value = node["leaf"]
        if isinstance(value, bool) or not math.isfinite(float(value)):
            raise ExpandedShadowArtifactError("invalid model leaf")
        return 1
    if set(node) != {"feature", "threshold", "default_left", "left", "right"}:
        raise ExpandedShadowArtifactError("invalid model split")
    feature = node["feature"]
    threshold = node["threshold"]
    if (
        isinstance(feature, bool)
        or not isinstance(feature, int)
        or not 0 <= feature < feature_count
    ):
        raise ExpandedShadowArtifactError("invalid model feature index")
    if isinstance(threshold, bool) or not math.isfinite(float(threshold)):
        raise ExpandedShadowArtifactError("invalid model threshold")
    if not isinstance(node["default_left"], bool):
        raise ExpandedShadowArtifactError("invalid model missing-value branch")
    return (
        1
        + _validate_node(node["left"], feature_count, depth + 1)
        + _validate_node(node["right"], feature_count, depth + 1)
    )


@dataclass(frozen=True)
class SafeTreeModel:
    product_key: str
    contract_version: str
    artifact_id: str
    feature_order: tuple[str, ...]
    feature_domain: Mapping[str, tuple[float, float]]
    schedule_policy: str
    trees: tuple[dict[str, Any], ...]
    manifest: Mapping[str, Any]

    def predict(self, features: np.ndarray) -> float:
        if features.shape != (len(self.feature_order),) or not np.all(
            np.isfinite(features)
        ):
            raise ExpandedShadowError(
                "shadow features must be finite and match the schema"
            )
        total = 0.0
        for root in self.trees:
            node = root
            while "leaf" not in node:
                value = float(features[node["feature"]])
                go_left = (
                    node["default_left"]
                    if math.isnan(value)
                    else value <= node["threshold"]
                )
                node = node["left"] if go_left else node["right"]
            total += float(node["leaf"])
        if not math.isfinite(total):
            raise ExpandedShadowError("shadow prediction is not finite")
        return total


def _load_model(product_key: str, settings: ExpandedShadowSettings) -> SafeTreeModel:
    registry_path = settings.artifact_root / "registry.json"
    registry = _load_json(registry_path)
    if registry.get("schema_version") != REGISTRY_SCHEMA:
        raise ExpandedShadowArtifactError("unsupported shadow registry schema")
    if registry.get("runtime_policy") != "shadow-only":
        raise ExpandedShadowArtifactError("registry is not shadow-only")
    pinned = (registry.get("artifacts") or {}).get(product_key)
    if not isinstance(pinned, dict):
        raise ExpandedShadowArtifactError(f"no pinned artifact for {product_key}")
    artifact_id = pinned.get("artifact_id")
    if not isinstance(artifact_id, str) or not artifact_id.startswith("sha256:"):
        raise ExpandedShadowArtifactError("invalid pinned artifact ID")
    directory = (
        settings.artifact_root / product_key / artifact_id.removeprefix("sha256:")
    )
    manifest_path = directory / "manifest.json"
    model_path = directory / "model.json.gz"
    cache_key = (
        str(directory.resolve()),
        manifest_path.stat().st_mtime_ns if manifest_path.exists() else -1,
        model_path.stat().st_mtime_ns if model_path.exists() else -1,
    )
    with _CACHE_LOCK:
        cached = _CACHE.get(cache_key)
    if cached is not None:
        return cached

    manifest = _load_json(manifest_path)
    required = {
        "schema_version",
        "artifact_id",
        "product_key",
        "contract_version",
        "source_experiment_id",
        "model_checksum",
        "feature_order",
        "feature_domain",
        "schedule_policy",
        "tree_count",
        "shadow_eligible",
        "runtime_approved",
        "runtime_policy",
    }
    if not required.issubset(manifest):
        raise ExpandedShadowArtifactError("shadow manifest is incomplete")
    if (
        manifest["schema_version"] != ARTIFACT_SCHEMA
        or manifest["artifact_id"] != artifact_id
        or manifest["product_key"] != product_key
        or manifest["contract_version"] != SUPPORTED_PRODUCTS[product_key]
        or pinned.get("contract_version") != SUPPORTED_PRODUCTS[product_key]
        or manifest["shadow_eligible"] is not True
        or manifest["runtime_approved"] is not False
        or manifest["runtime_policy"] != "shadow-only"
    ):
        raise ExpandedShadowArtifactError("shadow manifest policy or identity mismatch")
    expected_artifact_id = _sha256(
        _canonical_bytes(
            {
                "schema_version": ARTIFACT_SCHEMA,
                "experiment_id": manifest["source_experiment_id"],
                "model_checksum": manifest["model_checksum"],
                "contract_version": manifest["contract_version"],
                "feature_order": manifest["feature_order"],
            }
        )
    )
    if expected_artifact_id != artifact_id:
        raise ExpandedShadowArtifactError("shadow artifact identity checksum mismatch")
    try:
        with gzip.open(model_path, "rb") as handle:
            model_bytes = handle.read(32_000_001)
    except (OSError, EOFError) as exc:
        raise ExpandedShadowArtifactError("shadow model is unreadable") from exc
    if (
        len(model_bytes) > 32_000_000
        or _sha256(model_bytes) != manifest["model_checksum"]
    ):
        raise ExpandedShadowArtifactError("shadow model checksum mismatch")
    try:
        payload = json.loads(model_bytes)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ExpandedShadowArtifactError("shadow model JSON is invalid") from exc
    feature_order = tuple(manifest["feature_order"])
    if (
        not feature_order
        or len(feature_order) > 128
        or len(set(feature_order)) != len(feature_order)
        or payload.get("schema_version") != ARTIFACT_SCHEMA
        or payload.get("product_key") != product_key
        or payload.get("contract_version") != manifest["contract_version"]
        or tuple(payload.get("feature_order") or ()) != feature_order
    ):
        raise ExpandedShadowArtifactError("shadow model schema mismatch")
    trees = tuple(payload.get("trees") or ())
    if not trees or len(trees) != manifest["tree_count"] or len(trees) > 2_000:
        raise ExpandedShadowArtifactError("shadow model tree count mismatch")
    nodes = sum(_validate_node(tree, len(feature_order)) for tree in trees)
    if nodes > 200_000:
        raise ExpandedShadowArtifactError("shadow model is too large")
    domain: dict[str, tuple[float, float]] = {}
    for name, bounds in manifest["feature_domain"].items():
        if not isinstance(bounds, list) or len(bounds) != 2:
            raise ExpandedShadowArtifactError("invalid feature domain")
        lower, upper = float(bounds[0]), float(bounds[1])
        if not math.isfinite(lower) or not math.isfinite(upper) or lower > upper:
            raise ExpandedShadowArtifactError("invalid feature bounds")
        domain[str(name)] = (lower, upper)
    model = SafeTreeModel(
        product_key=product_key,
        contract_version=manifest["contract_version"],
        artifact_id=artifact_id,
        feature_order=feature_order,
        feature_domain=domain,
        schedule_policy=str(manifest["schedule_policy"]),
        trees=trees,
        manifest=manifest,
    )
    with _CACHE_LOCK:
        _CACHE.clear()
        _CACHE[cache_key] = model
    return model


def _is_even_schedule(times: tuple[float, ...], maturity: float) -> bool:
    count = len(times)
    return all(
        math.isclose(value, maturity * index / count, rel_tol=0.0, abs_tol=1e-9)
        for index, value in enumerate(times, start=1)
    )


def _is_linear(values: tuple[float, ...]) -> bool:
    if len(values) <= 2:
        return True
    expected = np.linspace(values[0], values[-1], len(values))
    return bool(np.allclose(np.asarray(values), expected, rtol=0.0, atol=1e-9))


def _features(
    product_key: str,
    market: EquityMarketTermStructure,
    contract: PhoenixSingleV3Contract | BarrierReverseConvertibleV1Contract,
) -> dict[str, float]:
    if market.currency != "USD" or market.underlier_type not in {
        "equity",
        "etf",
        "index",
    }:
        raise ExpandedShadowDomainError("model domain is USD equity-like underliers")
    equivalent = market.equivalent_flat_parameters(contract.maturity_years)
    spot_ratio = market.spot / contract.reference_level
    common = {
        "spot_to_reference": spot_ratio,
        "risk_free_rate": equivalent["risk_free_rate"],
        "dividend_yield": equivalent["dividend_yield"],
        "volatility": equivalent["volatility"],
        "maturity_years": contract.maturity_years,
    }
    if product_key == "phoenix_v3":
        if not isinstance(contract, PhoenixSingleV3Contract):
            raise ExpandedShadowDomainError("contract type does not match Phoenix v3")
        if not _is_even_schedule(
            contract.observation_times_years, contract.maturity_years
        ):
            raise ExpandedShadowDomainError("model requires evenly spaced observations")
        if not _is_linear(contract.autocall_barrier_fracs):
            raise ExpandedShadowDomainError(
                "model requires a linear autocall step-down"
            )
        first, final = (
            contract.autocall_barrier_fracs[0],
            contract.autocall_barrier_fracs[-1],
        )
        return {
            **common,
            "first_autocall_barrier_frac": first,
            "final_autocall_barrier_frac": final,
            "coupon_barrier_frac": contract.coupon_barrier_frac,
            "coupon_rate": contract.coupon_rate,
            "knock_in_frac": contract.knock_in_frac,
            "observation_count": float(len(contract.observation_times_years)),
            "memory_coupon": float(contract.memory_coupon),
            "unpaid_coupon_count": float(contract.unpaid_coupon_count),
            "prior_knock_in_breached": float(contract.prior_knock_in_breached),
            "spot_minus_first_autocall": spot_ratio - first,
            "spot_minus_final_autocall": spot_ratio - final,
            "spot_minus_coupon_barrier": spot_ratio - contract.coupon_barrier_frac,
            "spot_minus_knock_in": spot_ratio - contract.knock_in_frac,
            "autocall_stepdown": first - final,
            "coupon_including_unpaid": contract.coupon_rate
            * (1 + contract.unpaid_coupon_count),
        }
    if not isinstance(contract, BarrierReverseConvertibleV1Contract):
        raise ExpandedShadowDomainError(
            "contract type does not match reverse convertible"
        )
    if not _is_even_schedule(contract.coupon_times_years, contract.maturity_years):
        raise ExpandedShadowDomainError(
            "model requires evenly spaced coupon observations"
        )
    count = len(contract.coupon_times_years)
    return {
        **common,
        "coupon_rate_per_period": contract.coupon_rate_per_period,
        "strike_frac": contract.strike_frac,
        "knock_in_frac": contract.knock_in_frac,
        "coupon_count": float(count),
        "prior_knock_in_breached": float(contract.prior_knock_in_breached),
        "spot_minus_strike": spot_ratio - contract.strike_frac,
        "spot_minus_knock_in": spot_ratio - contract.knock_in_frac,
        "strike_minus_knock_in": contract.strike_frac - contract.knock_in_frac,
        "total_coupon_rate": contract.coupon_rate_per_period * count,
    }


def _check_domain(model: SafeTreeModel, values: Mapping[str, float]) -> float:
    utilization = 0.0
    violations = []
    for name, (lower, upper) in model.feature_domain.items():
        value = float(values[name])
        if value < lower - 1e-12 or value > upper + 1e-12:
            violations.append(name)
        half_range = (upper - lower) / 2.0
        if half_range > 0:
            utilization = max(
                utilization, abs(value - (lower + upper) / 2.0) / half_range
            )
    if violations:
        raise ExpandedShadowDomainError(
            "outside trained domain: " + ", ".join(sorted(violations))
        )
    return utilization


def _sampled(
    product_key: str, market: EquityMarketTermStructure, contract_id: str, rate: float
) -> bool:
    digest = hashlib.sha256(
        f"{product_key}|{market.term_structure_id}|{contract_id}".encode("utf-8")
    ).digest()
    bucket = int.from_bytes(digest[:8], "big") / float(2**64)
    return bucket < rate


def _base(
    status: str, reason: str, *, artifact_id: str | None = None
) -> dict[str, Any]:
    return {
        "status": status,
        "mode": "shadow-only",
        "used_for_price": False,
        "reason": reason,
        "artifact_id": artifact_id,
        "runtime_approved": False,
    }


def evaluate_expanded_shadow(
    *,
    product_key: str,
    market: EquityMarketTermStructure,
    contract: PhoenixSingleV3Contract | BarrierReverseConvertibleV1Contract,
    reference_price: float,
    reference_standard_error: float,
    reference_latency_ms: float,
    force: bool = False,
) -> dict[str, Any]:
    """Evaluate a candidate without ever changing or delaying the reference result."""
    if product_key not in SUPPORTED_PRODUCTS:
        return _base("not_applicable", "no expanded shadow adapter for this product")
    try:
        settings = ExpandedShadowSettings.from_env(product_key)
        if not force and not settings.enabled:
            return _base("disabled", "shadow collection is disabled for this product")
        model = _load_model(product_key, settings)
        if contract.contract_version != model.contract_version:
            return _base(
                "not_applicable",
                "contract version does not match the pinned artifact",
                artifact_id=model.artifact_id,
            )
        if not force and not _sampled(
            product_key, market, contract.contract_id, settings.sample_rate
        ):
            return _base(
                "not_sampled",
                "request was outside the configured shadow sample",
                artifact_id=model.artifact_id,
            )
        values = _features(product_key, market, contract)
        utilization = _check_domain(model, values)
        vector = np.asarray(
            [values[name] for name in model.feature_order], dtype=np.float64
        )
        started = time.perf_counter()
        future = _EXECUTOR.submit(model.predict, vector)
        try:
            prediction = float(future.result(timeout=settings.timeout_ms / 1_000.0))
        except FutureTimeout:
            future.cancel()
            return _base(
                "timeout",
                "shadow inference exceeded its latency budget",
                artifact_id=model.artifact_id,
            )
        latency_ms = (time.perf_counter() - started) * 1_000.0
        absolute_error = abs(prediction - float(reference_price))
        result = {
            **_base(
                "success",
                "shadow estimate recorded; Monte Carlo remains authoritative",
                artifact_id=model.artifact_id,
            ),
            "surrogate_price": prediction,
            "reference_price": float(reference_price),
            "absolute_error": absolute_error,
            "relative_error": absolute_error / max(abs(float(reference_price)), 1e-12),
            "error_to_reference_standard_error": absolute_error
            / max(float(reference_standard_error), 1e-12),
            "latency_ms": latency_ms,
            "reference_latency_ms": float(reference_latency_ms),
            "speedup": float(reference_latency_ms) / max(latency_ms, 1e-12),
            "maximum_domain_utilization": utilization,
            "contract_version": model.contract_version,
            "feature_schema": list(model.feature_order),
            "model_version": (
                f"{product_key}-{model.manifest.get('selected_candidate', 'candidate')}"
                "-expanded-v2"
            ),
            "validation_metrics": dict(
                (model.manifest.get("sealed_audit") or {}).get("metrics") or {}
            ),
        }
        return result
    except ExpandedShadowDomainError as exc:
        return _base("out_of_domain", str(exc))
    except ExpandedShadowArtifactError:
        return _base("unavailable", "pinned shadow artifact failed validation")
    except Exception:
        return _base("error", "shadow inference failed safely")


def get_expanded_shadow_status() -> dict[str, Any]:
    products: dict[str, Any] = {}
    for product_key in SUPPORTED_PRODUCTS:
        try:
            settings = ExpandedShadowSettings.from_env(product_key)
            model = _load_model(product_key, settings)
            products[product_key] = {
                "enabled": settings.enabled,
                "sample_rate": settings.sample_rate,
                "timeout_ms": settings.timeout_ms,
                "artifact_available": True,
                "artifact_id": model.artifact_id,
                "contract_version": model.contract_version,
                "runtime_policy": "shadow-only",
                "runtime_approved": False,
            }
        except Exception:
            products[product_key] = {
                "enabled": False,
                "sample_rate": 0.0,
                "artifact_available": False,
                "runtime_policy": "shadow-only",
                "runtime_approved": False,
            }
    return {
        "version": ARTIFACT_SCHEMA,
        "automatic_promotion_permitted": False,
        "products": products,
    }
