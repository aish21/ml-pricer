import math
import time
from pathlib import Path
from typing import Any, Dict, Optional

from src.final.reference_pricer import DEFAULT_REFERENCE_SEED, price_reference
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


MAX_REFERENCE_PATHS = 20_000


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
    if not math.isfinite(value):
        raise InvalidPricingInputError(f"{field.name} must be finite")
    if field.min_value is not None and value < field.min_value:
        raise InvalidPricingInputError(f"{field.name} must be >= {field.min_value}")
    if field.max_value is not None and value > field.max_value:
        raise InvalidPricingInputError(f"{field.name} must be <= {field.max_value}")
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
    if field.max_value is not None and value > field.max_value:
        raise InvalidPricingInputError(f"{field.name} must be <= {field.max_value}")
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


def normalize_pricing_params(
    product_key: str, params: Dict[str, Any]
) -> Dict[str, Any]:
    product = get_product_definition(product_key)
    if product is None:
        raise UnsupportedProductError(f"unknown product: {product_key}")
    if not product.validated_for_pricing:
        raise UnsupportedProductError(f"unsupported product: {product_key}")

    normalized: Dict[str, Any] = {}
    for field in product.bb_fields:
        normalized[field.name] = _parse_field(params, field)

    if product.key == "phoenix":
        knock_in = normalized["knock_in_frac"]
        coupon_barrier = normalized["coupon_barrier_frac"]
        autocall_barrier = normalized["autocall_barrier_frac"]
        if not knock_in <= coupon_barrier <= autocall_barrier:
            raise InvalidPricingInputError(
                "barriers must satisfy knock_in_frac <= coupon_barrier_frac "
                "<= autocall_barrier_frac"
            )

    return normalized


def price_product(
    product_key: str,
    params: Dict[str, Any],
    n_paths: int = 500,
    results_dir: Optional[Path] = None,
    seed: int = DEFAULT_REFERENCE_SEED,
) -> Dict[str, Any]:
    # Kept for call compatibility during the Phase 1 migration. Reference
    # pricing deliberately does not depend on a surrogate artifact directory.
    _ = results_dir
    product = get_product_definition(product_key)
    if product is None:
        raise UnsupportedProductError(f"unknown product: {product_key}")
    if not product.validated_for_pricing or not product.reference_pricing_enabled:
        raise UnsupportedProductError(f"unsupported product: {product_key}")

    try:
        n_paths_int = int(n_paths)
    except (TypeError, ValueError) as exc:
        raise InvalidPricingInputError("invalid Monte Carlo path count") from exc
    if n_paths_int < 1 or n_paths_int > MAX_REFERENCE_PATHS:
        raise InvalidPricingInputError(
            f"Monte Carlo path count must be between 1 and {MAX_REFERENCE_PATHS}"
        )

    normalized_params = normalize_pricing_params(product.key, params)

    start = time.perf_counter()
    reference = price_reference(
        payoff=product.payoff_class(),
        params=normalized_params,
        n_paths=n_paths_int,
        seed=int(seed),
    )
    latency_ms = int(round((time.perf_counter() - start) * 1000))
    raw_result = {
        "params": normalized_params,
        "per_npaths": {str(n_paths_int): {"Reference": reference}},
    }

    return {
        "product_key": product.key,
        "product_name": product.display_name,
        "params": normalized_params,
        "n_paths": n_paths_int,
        "price": reference["price"],
        "mc_price": reference["price"],
        "abs_error": None,
        "rel_error": None,
        "speedup": None,
        "model_time_s": None,
        "mc_time_s": reference["time_s"],
        "standard_error": reference["standard_error"],
        "confidence_interval": reference["confidence_interval"],
        "seed": reference["seed"],
        "latency_ms": latency_ms,
        "model": "Monte Carlo reference",
        "pricing_method": "monte_carlo_reference",
        "contract_version": product.contract_version,
        "model_version": "gbm-flat-v1",
        "raw_result": raw_result,
    }
