from typing import Any, Dict, Optional

from src.final.reference_pricer import DEFAULT_REFERENCE_SEED

from app.services.pricing_service import (
    InvalidPricingInputError,
    PricingServiceError,
    UnsupportedProductError,
    price_product,
)
from app.services.product_registry import get_product_definition


class ScenarioServiceError(Exception):
    pass


class InvalidScenarioInputError(ScenarioServiceError):
    pass


def _parse_optional_float(shocks: Dict[str, Any], name: str) -> Optional[float]:
    raw_value = shocks.get(name)
    if raw_value is None or raw_value == "":
        return None
    try:
        return float(raw_value)
    except (TypeError, ValueError) as exc:
        raise InvalidScenarioInputError(f"invalid shock: {name}") from exc


def normalize_shocks(shocks: Dict[str, Any]) -> Dict[str, float]:
    normalized = {}
    for name in ("spot_pct", "vol_abs", "rate_bps"):
        value = _parse_optional_float(shocks, name)
        if value is not None:
            normalized[name] = value

    if not normalized:
        raise InvalidScenarioInputError("at least one shock is required")

    return normalized


def apply_shocks_to_params(
    base_params: Dict[str, Any], shocks: Dict[str, Any]
) -> tuple[Dict[str, Any], Dict[str, float]]:
    normalized_shocks = normalize_shocks(shocks)
    shocked_params = dict(base_params)

    if "spot_pct" in normalized_shocks:
        shocked_params["S0"] = float(shocked_params["S0"]) * (
            1.0 + normalized_shocks["spot_pct"] / 100.0
        )
        if shocked_params["S0"] <= 0:
            raise InvalidScenarioInputError("shocked spot must be positive")

    if "vol_abs" in normalized_shocks:
        shocked_params["sigma"] = (
            float(shocked_params["sigma"]) + normalized_shocks["vol_abs"]
        )
        if shocked_params["sigma"] <= 0:
            raise InvalidScenarioInputError("shocked volatility must be positive")

    if "rate_bps" in normalized_shocks:
        shocked_params["r"] = float(shocked_params["r"]) + (
            normalized_shocks["rate_bps"] / 10000.0
        )

    return shocked_params, normalized_shocks


def _float_or_none(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _pct_change(
    base_price: Optional[float], price_change: Optional[float]
) -> Optional[float]:
    if base_price in (None, 0) or price_change is None:
        return None
    return (price_change / abs(base_price)) * 100.0


def summarize_scenario(shocks: Dict[str, float], product_name: str) -> str:
    parts = []
    if shocks.get("spot_pct", 0) < 0:
        parts.append("spot down")
    elif shocks.get("spot_pct", 0) > 0:
        parts.append("spot up")

    if shocks.get("vol_abs", 0) < 0:
        parts.append("vol down")
    elif shocks.get("vol_abs", 0) > 0:
        parts.append("vol up")

    if shocks.get("rate_bps", 0) < 0:
        parts.append("rates down")
    elif shocks.get("rate_bps", 0) > 0:
        parts.append("rates up")

    if not parts:
        return "Scenario repriced with supplied shocks."
    return " / ".join(parts).capitalize() + f" changed the {product_name} value."


def run_scenario(
    base_request: Dict[str, Any],
    base_result: Dict[str, Any],
    shocks: Dict[str, Any],
) -> Dict[str, Any]:
    if not base_request:
        raise InvalidScenarioInputError("base request missing")

    product_key = base_request.get("product_key")
    product = get_product_definition(product_key or "")
    if product is None:
        raise UnsupportedProductError(f"unknown product: {product_key}")
    if not product.enabled_for_bb:
        raise UnsupportedProductError(f"unsupported product: {product_key}")

    base_params = base_request.get("params")
    if not isinstance(base_params, dict):
        raise InvalidScenarioInputError("base request params missing")

    shocked_params, normalized_shocks = apply_shocks_to_params(base_params, shocks)
    n_paths = base_request.get("n_paths", 500)
    try:
        shocked_result = price_product(
            product_key=product_key,
            params=shocked_params,
            n_paths=n_paths,
            seed=int(base_request.get("seed", DEFAULT_REFERENCE_SEED)),
        )
    except (PricingServiceError, InvalidPricingInputError) as exc:
        raise InvalidScenarioInputError(str(exc)) from exc

    base_price = _float_or_none(base_result.get("price"))
    shocked_price = _float_or_none(shocked_result.get("price"))
    price_change = (
        shocked_price - base_price
        if shocked_price is not None and base_price is not None
        else None
    )

    shocked_request = dict(base_request)
    shocked_request["params"] = shocked_params

    return {
        "product_key": product_key,
        "base_price": base_price,
        "shocked_price": shocked_price,
        "price_change": price_change,
        "price_change_pct": _pct_change(base_price, price_change),
        "base_request": base_request,
        "base_result": base_result,
        "shocked_request": shocked_request,
        "shocked_result": shocked_result,
        "shocks": normalized_shocks,
        "summary": summarize_scenario(normalized_shocks, product.display_name),
        "model": shocked_result.get("model"),
        "latency_ms": shocked_result.get("latency_ms"),
    }
