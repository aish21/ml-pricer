from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Dict, Optional, Tuple

from app.services.product_registry import (
    build_artifact_status,
    get_product_definition,
    get_results_dir,
)


class ModelCacheError(Exception):
    pass


class ModelCacheProductError(ModelCacheError):
    pass


class ModelCacheArtifactError(ModelCacheError):
    pass


class ModelCacheLoadError(ModelCacheError):
    pass


@dataclass(frozen=True)
class ModelBundle:
    product_key: str
    model: object
    scaler: object
    model_path: Path
    scaler_path: Path


_MODEL_CACHE: Dict[Tuple[str, str], ModelBundle] = {}
_CACHE_LOCK = Lock()


def _load_model_artifacts(model_path: Path, scaler_path: Path) -> tuple[object, object]:
    """Load optional surrogate artifacts only when a compatible model is requested."""
    from src.final.model_trainer import ModelTrainer

    return ModelTrainer.load(model_path, scaler_path)


def _cache_key(product_key: str, results_dir: Optional[Path]) -> Tuple[str, str]:
    base_dir = Path(results_dir) if results_dir else get_results_dir()
    return product_key, str(base_dir.resolve())


def get_model_bundle(
    product_key: str, results_dir: Optional[Path] = None
) -> ModelBundle:
    product = get_product_definition(product_key)
    if product is None:
        raise ModelCacheProductError(f"unknown product: {product_key}")

    key = _cache_key(product.key, results_dir)
    with _CACHE_LOCK:
        cached = _MODEL_CACHE.get(key)
        if cached is not None:
            return cached

    base_dir = Path(results_dir) if results_dir else get_results_dir()
    artifact_status = build_artifact_status(product, base_dir)
    if not artifact_status["ready_for_surrogate"]:
        raise ModelCacheArtifactError(f"model artifacts missing for {product.key}")

    model_path = base_dir / product.artifact_dir / "model.joblib"
    scaler_path = base_dir / product.artifact_dir / "scaler.joblib"
    try:
        model, scaler = _load_model_artifacts(model_path, scaler_path)
    except Exception as exc:
        raise ModelCacheLoadError(f"model load failed for {product.key}") from exc
    bundle = ModelBundle(
        product_key=product.key,
        model=model,
        scaler=scaler,
        model_path=model_path,
        scaler_path=scaler_path,
    )

    with _CACHE_LOCK:
        _MODEL_CACHE[key] = bundle
    return bundle


def clear_model_cache() -> None:
    with _CACHE_LOCK:
        _MODEL_CACHE.clear()


def get_model_cache_status(results_dir: Optional[Path] = None) -> dict[str, bool]:
    base_dir = Path(results_dir) if results_dir else get_results_dir()
    resolved = str(base_dir.resolve())
    status: dict[str, bool] = {}
    with _CACHE_LOCK:
        for product_key, cache_dir in _MODEL_CACHE.keys():
            status[product_key] = (
                status.get(product_key, False) or cache_dir == resolved
            )
    return status
