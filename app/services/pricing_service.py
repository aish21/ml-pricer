import time
from pathlib import Path
from typing import Any, Dict, Optional

from src.final.evaluator import Evaluator
from src.final.model_trainer import ModelTrainer

from app.services.product_registry import (
    build_artifact_status,
    get_product_definition,
    get_results_dir,
)


BB_PRICING_PRODUCT_KEYS = ("phoenix",)


class PricingServiceError(Exception):
    pass


class UnsupportedProductError(PricingServiceError):
    pass


class InvalidPricingInputError(PricingServiceError):
    pass


class PricingArtifactError(PricingServiceError):
    pass


def get_bb_pricing_products() -> list[dict[str, str]]:
    products = []
    for key in BB_PRICING_PRODUCT_KEYS:
        product = get_product_definition(key)
        if product is not None:
            products.append({"key": product.key, "display_name": product.display_name})
    return products


def _parse_float(params: Dict[str, Any], name: str) -> float:
    raw_value = params.get(name)
    if raw_value is None or raw_value == "":
        raise InvalidPricingInputError(f"missing required parameter: {name}")
    try:
        return float(raw_value)
    except (TypeError, ValueError) as exc:
        raise InvalidPricingInputError(f"invalid numeric parameter: {name}") from exc


def _parse_int(params: Dict[str, Any], name: str) -> int:
    raw_value = params.get(name)
    if raw_value is None or raw_value == "":
        raise InvalidPricingInputError(f"missing required parameter: {name}")
    try:
        return int(raw_value)
    except (TypeError, ValueError) as exc:
        raise InvalidPricingInputError(f"invalid integer parameter: {name}") from exc


def normalize_pricing_params(product_key: str, params: Dict[str, Any]) -> Dict[str, Any]:
    product = get_product_definition(product_key)
    if product is None or product.key not in BB_PRICING_PRODUCT_KEYS:
        raise UnsupportedProductError(f"unsupported product: {product_key}")

    normalized: Dict[str, Any] = {}
    payoff = product.payoff_class()
    for name in payoff.get_parameter_names():
        if name == "obs_count":
            normalized[name] = _parse_int(params, name)
        else:
            normalized[name] = _parse_float(params, name)

    if normalized["S0"] <= 0:
        raise InvalidPricingInputError("spot must be positive")
    if normalized["sigma"] < 0:
        raise InvalidPricingInputError("volatility cannot be negative")
    if normalized["T"] <= 0:
        raise InvalidPricingInputError("maturity must be positive")
    if normalized["obs_count"] < 1:
        raise InvalidPricingInputError("observation count must be positive")

    return normalized


def price_product(
    product_key: str,
    params: Dict[str, Any],
    n_paths: int = 500,
    results_dir: Optional[Path] = None,
    use_log_target: bool = True,
) -> Dict[str, Any]:
    product = get_product_definition(product_key)
    if product is None or product.key not in BB_PRICING_PRODUCT_KEYS:
        raise UnsupportedProductError(f"unsupported product: {product_key}")

    try:
        n_paths_int = int(n_paths)
    except (TypeError, ValueError) as exc:
        raise InvalidPricingInputError("invalid Monte Carlo path count") from exc
    if n_paths_int < 1:
        raise InvalidPricingInputError("Monte Carlo path count must be positive")

    normalized_params = normalize_pricing_params(product.key, params)

    base_dir = Path(results_dir) if results_dir else get_results_dir()
    artifact_status = build_artifact_status(product, base_dir)
    if not artifact_status["ready_for_surrogate"]:
        raise PricingArtifactError(f"model artifacts missing for {product.key}")

    model_path = base_dir / product.artifact_dir / "model.joblib"
    scaler_path = base_dir / product.artifact_dir / "scaler.joblib"

    start = time.perf_counter()
    model, scaler = ModelTrainer.load(model_path, scaler_path)
    evaluator = Evaluator(product.payoff_class(), verbose=False)
    raw_result = evaluator.evaluate_case(
        params=normalized_params,
        model=model,
        scaler=scaler,
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
        "model_time_s": model_result.get("time"),
        "mc_time_s": mc_result.get("time"),
        "latency_ms": latency_ms,
        "model": "LightGBM surrogate",
        "model_version": product.artifact_dir,
        "raw_result": raw_result,
    }
