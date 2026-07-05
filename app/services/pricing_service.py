import time
from pathlib import Path
from typing import Any, Dict, Optional

from src.final.evaluator import Evaluator

from app.services.model_cache import (
    ModelCacheArtifactError,
    ModelCacheLoadError,
    get_model_bundle,
)
from app.services.product_registry import (
    ProductField,
    get_bb_product_definitions,
    get_product_definition,
)


class PricingServiceError(Exception):
    pass


class UnsupportedProductError(PricingServiceError):
    pass


class InvalidPricingInputError(PricingServiceError):
    pass


class PricingArtifactError(PricingServiceError):
    pass


def get_bb_pricing_products() -> list[dict[str, str]]:
    return [
        {
            "key": product.key,
            "display_name": product.display_name,
            "terminal_label": product.terminal_label,
        }
        for product in get_bb_product_definitions()
    ]


def _raw_value(params: Dict[str, Any], field: ProductField) -> Any:
    raw_value = params.get(field.name)
    if raw_value is None or raw_value == "":
        raise InvalidPricingInputError(f"missing required parameter: {field.name}")
    return raw_value


def _parse_float(params: Dict[str, Any], field: ProductField) -> float:
    raw_value = _raw_value(params, field)
    try:
        value = float(raw_value)
    except (TypeError, ValueError) as exc:
        raise InvalidPricingInputError(
            f"invalid numeric parameter: {field.name}"
        ) from exc
    if field.min_value is not None and value < field.min_value:
        raise InvalidPricingInputError(f"{field.name} must be >= {field.min_value}")
    return value


def _parse_int(params: Dict[str, Any], field: ProductField) -> int:
    raw_value = _raw_value(params, field)
    try:
        value = int(raw_value)
    except (TypeError, ValueError) as exc:
        raise InvalidPricingInputError(
            f"invalid integer parameter: {field.name}"
        ) from exc
    if field.min_value is not None and value < field.min_value:
        raise InvalidPricingInputError(f"{field.name} must be >= {field.min_value}")
    return value


def _parse_choice(params: Dict[str, Any], field: ProductField) -> float:
    raw_value = str(_raw_value(params, field))
    valid_values = {value for value, _ in field.choices}
    if raw_value not in valid_values:
        raise InvalidPricingInputError(f"invalid choice parameter: {field.name}")
    return float(raw_value)


def _parse_field(params: Dict[str, Any], field: ProductField) -> Any:
    if field.field_type == "int":
        return _parse_int(params, field)
    if field.field_type == "choice":
        return _parse_choice(params, field)
    return _parse_float(params, field)


def normalize_pricing_params(product_key: str, params: Dict[str, Any]) -> Dict[str, Any]:
    product = get_product_definition(product_key)
    if product is None:
        raise UnsupportedProductError(f"unknown product: {product_key}")
    if not product.enabled_for_bb:
        raise UnsupportedProductError(f"unsupported product: {product_key}")

    normalized: Dict[str, Any] = {}
    for field in product.bb_fields:
        normalized[field.name] = _parse_field(params, field)

    return normalized


def price_product(
    product_key: str,
    params: Dict[str, Any],
    n_paths: int = 500,
    results_dir: Optional[Path] = None,
    use_log_target: bool = True,
) -> Dict[str, Any]:
    product = get_product_definition(product_key)
    if product is None:
        raise UnsupportedProductError(f"unknown product: {product_key}")
    if not product.enabled_for_bb:
        raise UnsupportedProductError(f"unsupported product: {product_key}")

    try:
        n_paths_int = int(n_paths)
    except (TypeError, ValueError) as exc:
        raise InvalidPricingInputError("invalid Monte Carlo path count") from exc
    if n_paths_int < 1:
        raise InvalidPricingInputError("Monte Carlo path count must be positive")

    normalized_params = normalize_pricing_params(product.key, params)

    start = time.perf_counter()
    try:
        bundle = get_model_bundle(product.key, results_dir=results_dir)
    except (ModelCacheArtifactError, ModelCacheLoadError) as exc:
        raise PricingArtifactError(str(exc)) from exc

    evaluator = Evaluator(product.payoff_class(), verbose=False)
    raw_result = evaluator.evaluate_case(
        params=normalized_params,
        model=bundle.model,
        scaler=bundle.scaler,
        n_paths_list=[n_paths_int],
        use_log_target=use_log_target,
    )
    latency_ms = int(round((time.perf_counter() - start) * 1000))

    entry = raw_result["per_npaths"][str(n_paths_int)]
    model_result = entry["Model"]
    mc_result = entry["MC"]

    return {
        "product_key": product.key,
        "product_name": product.display_name,
        "params": normalized_params,
        "n_paths": n_paths_int,
        "price": model_result.get("price"),
        "mc_price": mc_result.get("price"),
        "abs_error": model_result.get("abs_error"),
        "rel_error": model_result.get("rel_error"),
        "speedup": model_result.get("speedup"),
        "model_time_s": model_result.get("time"),
        "mc_time_s": mc_result.get("time"),
        "latency_ms": latency_ms,
        "model": "LightGBM surrogate",
        "model_version": product.artifact_dir,
        "raw_result": raw_result,
    }
