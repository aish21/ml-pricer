import math
import time
from pathlib import Path
from typing import Any, Dict, Optional

from src.final.market import (
    EQUITY_GBM_FLAT_MODEL_VERSION,
    EquityMarketSnapshot,
)
from src.final.reference_pricer import DEFAULT_REFERENCE_SEED, price_reference
from app.services.product_registry import (
    ProductField,
    ProductDefinition,
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
LEGACY_MARKET_MODEL_VERSION = "gbm-flat-v1"
PHOENIX_SNAPSHOT_TERM_NAMES = frozenset(
    {
        "maturity_years",
        "autocall_barrier_frac",
        "coupon_barrier_frac",
        "coupon_rate",
        "knock_in_frac",
        "obs_count",
    }
)


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


def _validate_path_count(n_paths: Any) -> int:
    try:
        n_paths_int = int(n_paths)
    except (TypeError, ValueError) as exc:
        raise InvalidPricingInputError("invalid Monte Carlo path count") from exc
    if n_paths_int < 1 or n_paths_int > MAX_REFERENCE_PATHS:
        raise InvalidPricingInputError(
            f"Monte Carlo path count must be between 1 and {MAX_REFERENCE_PATHS}"
        )
    return n_paths_int


def _price_normalized_product(
    product: ProductDefinition,
    normalized_params: Dict[str, Any],
    n_paths: int,
    seed: int,
    model_version: str,
    dividend_yield: float = 0.0,
    market_snapshot: Optional[EquityMarketSnapshot] = None,
    terms: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    start = time.perf_counter()
    reference = price_reference(
        payoff=product.payoff_class(),
        params=normalized_params,
        n_paths=n_paths,
        seed=int(seed),
        dividend_yield=dividend_yield,
    )
    latency_ms = int(round((time.perf_counter() - start) * 1000))
    raw_result: Dict[str, Any] = {
        "params": normalized_params,
        "per_npaths": {str(n_paths): {"Reference": reference}},
    }
    if market_snapshot is not None:
        raw_result["market_snapshot"] = market_snapshot.to_dict()

    result = {
        "product_key": product.key,
        "product_name": product.display_name,
        "params": normalized_params,
        "n_paths": n_paths,
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
        "model_version": model_version,
        "raw_result": raw_result,
    }
    if market_snapshot is not None:
        result.update(
            {
                "underlier": {
                    "symbol": market_snapshot.symbol,
                    "type": market_snapshot.underlier_type,
                    "currency": market_snapshot.currency,
                },
                "market_snapshot": market_snapshot.to_dict(),
                "market_snapshot_version": market_snapshot.schema_version,
                "terms": terms or {},
            }
        )
    return result


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

    n_paths_int = _validate_path_count(n_paths)
    normalized_params = normalize_pricing_params(product.key, params)
    return _price_normalized_product(
        product=product,
        normalized_params=normalized_params,
        n_paths=n_paths_int,
        seed=seed,
        model_version=LEGACY_MARKET_MODEL_VERSION,
    )


def price_phoenix_with_market_snapshot(
    market_snapshot: EquityMarketSnapshot,
    terms: Dict[str, Any],
    n_paths: int = 500,
    seed: int = DEFAULT_REFERENCE_SEED,
) -> Dict[str, Any]:
    """Price Phoenix Single v1 from distinct product terms and market data."""
    if not isinstance(market_snapshot, EquityMarketSnapshot):
        raise InvalidPricingInputError("invalid equity market snapshot")
    if not isinstance(terms, dict):
        raise InvalidPricingInputError("Phoenix terms must be an object")

    supplied_names = set(terms)
    missing = sorted(PHOENIX_SNAPSHOT_TERM_NAMES - supplied_names)
    extra = sorted(supplied_names - PHOENIX_SNAPSHOT_TERM_NAMES)
    if missing:
        raise InvalidPricingInputError(
            f"missing required Phoenix terms: {', '.join(missing)}"
        )
    if extra:
        raise InvalidPricingInputError(f"unknown Phoenix terms: {', '.join(extra)}")

    raw_params = {
        "S0": market_snapshot.spot,
        "r": market_snapshot.risk_free_rate,
        "sigma": market_snapshot.volatility,
        "T": terms["maturity_years"],
        "autocall_barrier_frac": terms["autocall_barrier_frac"],
        "coupon_barrier_frac": terms["coupon_barrier_frac"],
        "coupon_rate": terms["coupon_rate"],
        "knock_in_frac": terms["knock_in_frac"],
        "obs_count": terms["obs_count"],
    }
    normalized_params = normalize_pricing_params("phoenix", raw_params)
    normalized_terms = {
        "maturity_years": normalized_params["T"],
        "autocall_barrier_frac": normalized_params["autocall_barrier_frac"],
        "coupon_barrier_frac": normalized_params["coupon_barrier_frac"],
        "coupon_rate": normalized_params["coupon_rate"],
        "knock_in_frac": normalized_params["knock_in_frac"],
        "obs_count": normalized_params["obs_count"],
    }
    product = get_product_definition("phoenix")
    if product is None or not product.reference_pricing_enabled:
        raise UnsupportedProductError("unsupported product: phoenix")

    return _price_normalized_product(
        product=product,
        normalized_params=normalized_params,
        n_paths=_validate_path_count(n_paths),
        seed=seed,
        model_version=EQUITY_GBM_FLAT_MODEL_VERSION,
        dividend_yield=market_snapshot.dividend_yield,
        market_snapshot=market_snapshot,
        terms=normalized_terms,
    )
