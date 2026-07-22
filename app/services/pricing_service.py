import math
import time
from pathlib import Path
from typing import Any, Dict, Optional

from src.final.barrier_reverse_convertible import (
    BARRIER_REVERSE_CONVERTIBLE_V1,
    BarrierReverseConvertibleV1Contract,
)
from src.final.market import (
    EQUITY_GBM_FLAT_MODEL_VERSION,
    EQUITY_GBM_PIECEWISE_MODEL_VERSION,
    EquityMarketSnapshot,
    EquityMarketTermStructure,
    MarketDataValidationError,
)
from src.final.reference_pricer import (
    DEFAULT_REFERENCE_SEED,
    price_barrier_reverse_convertible_reference,
    price_phoenix_piecewise_reference,
    price_phoenix_v2_piecewise_reference,
    price_phoenix_v3_piecewise_reference,
    price_reference,
)
from src.final.phoenix_contract import (
    PHOENIX_SINGLE_V2_CONTRACT_VERSION,
    PHOENIX_SINGLE_V3_CONTRACT_VERSION,
    PhoenixSingleV2Contract,
    PhoenixSingleV3Contract,
)
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


def _attach_expanded_shadow(
    *,
    result: Dict[str, Any],
    product_key: str,
    market: EquityMarketTermStructure,
    contract: PhoenixSingleV3Contract | BarrierReverseConvertibleV1Contract,
) -> None:
    """Attach best-effort ML evidence without changing the reference result."""
    try:
        from app.services.expanded_shadow_monitoring import (
            record_expanded_shadow_observation,
        )
        from app.services.expanded_shadow_service import evaluate_expanded_shadow

        shadow = evaluate_expanded_shadow(
            product_key=product_key,
            market=market,
            contract=contract,
            reference_price=result["price"],
            reference_standard_error=result["standard_error"],
            reference_latency_ms=result["latency_ms"],
        )
        shadow["telemetry_recorded"] = record_expanded_shadow_observation(
            product_key=product_key,
            market=market,
            contract=contract,
            reference_price=result["price"],
            reference_standard_error=result["standard_error"],
            reference_latency_ms=result["latency_ms"],
            shadow_result=shadow,
        )
        result["surrogate_shadow"] = shadow
    except Exception:
        result["surrogate_shadow"] = {
            "status": "error",
            "mode": "shadow-only",
            "used_for_price": False,
            "runtime_approved": False,
            "reason": "expanded shadow integration failed safely",
            "telemetry_recorded": False,
        }


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


def validate_reference_path_count(n_paths: Any) -> int:
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
    return _build_pricing_result(
        product=product,
        normalized_params=normalized_params,
        n_paths=n_paths,
        model_version=model_version,
        reference=reference,
        latency_ms=latency_ms,
        market_snapshot=market_snapshot,
        terms=terms,
    )


def _build_pricing_result(
    product: ProductDefinition,
    normalized_params: Dict[str, Any],
    n_paths: int,
    model_version: str,
    reference: Dict[str, Any],
    latency_ms: int,
    market_snapshot: Optional[EquityMarketSnapshot] = None,
    market_term_structure: Optional[EquityMarketTermStructure] = None,
    terms: Optional[Dict[str, Any]] = None,
    contract_version: Optional[str] = None,
    contract_details: Optional[Dict[str, Any]] = None,
    warnings: Optional[list[str]] = None,
) -> Dict[str, Any]:
    if market_snapshot is not None and market_term_structure is not None:
        raise PricingServiceError("pricing result received multiple market inputs")
    raw_result: Dict[str, Any] = {
        "params": normalized_params,
        "per_npaths": {str(n_paths): {"Reference": reference}},
    }
    if market_snapshot is not None:
        raw_result["market_snapshot"] = market_snapshot.to_dict()
    if market_term_structure is not None:
        raw_result["market_term_structure"] = market_term_structure.to_dict()
    if contract_details is not None:
        raw_result["contract"] = contract_details

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
        "contract_version": contract_version or product.contract_version,
        "model_version": model_version,
        "raw_result": raw_result,
    }
    if contract_details is not None:
        result["contract"] = contract_details
    if warnings:
        result["warnings"] = list(warnings)
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
    if market_term_structure is not None:
        result.update(
            {
                "underlier": {
                    "symbol": market_term_structure.symbol,
                    "type": market_term_structure.underlier_type,
                    "currency": market_term_structure.currency,
                },
                "market_term_structure": market_term_structure.to_dict(),
                "market_data_version": market_term_structure.schema_version,
                "terms": terms or {},
            }
        )
    return result


def normalize_phoenix_market_terms(
    spot: float,
    risk_free_rate: float,
    volatility: float,
    terms: Dict[str, Any],
) -> tuple[Dict[str, Any], Dict[str, Any]]:
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
        "S0": spot,
        "r": risk_free_rate,
        "sigma": volatility,
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
    return normalized_params, normalized_terms


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

    n_paths_int = validate_reference_path_count(n_paths)
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
    normalized_params, normalized_terms = normalize_phoenix_market_terms(
        spot=market_snapshot.spot,
        risk_free_rate=market_snapshot.risk_free_rate,
        volatility=market_snapshot.volatility,
        terms=terms,
    )
    product = get_product_definition("phoenix")
    if product is None or not product.reference_pricing_enabled:
        raise UnsupportedProductError("unsupported product: phoenix")

    return _price_normalized_product(
        product=product,
        normalized_params=normalized_params,
        n_paths=validate_reference_path_count(n_paths),
        seed=seed,
        model_version=EQUITY_GBM_FLAT_MODEL_VERSION,
        dividend_yield=market_snapshot.dividend_yield,
        market_snapshot=market_snapshot,
        terms=normalized_terms,
    )


def price_phoenix_with_term_structure(
    market: EquityMarketTermStructure,
    terms: Dict[str, Any],
    n_paths: int = 500,
    seed: int = DEFAULT_REFERENCE_SEED,
) -> Dict[str, Any]:
    """Price Phoenix Single v1 with deterministic piecewise market inputs."""
    if not isinstance(market, EquityMarketTermStructure):
        raise InvalidPricingInputError("invalid equity market term structure")
    normalized_params, normalized_terms = normalize_phoenix_market_terms(
        spot=market.spot,
        risk_free_rate=0.0,
        volatility=1.0,
        terms=terms,
    )
    try:
        maturity = normalized_params["T"]
        equivalent = market.equivalent_flat_parameters(maturity)
    except MarketDataValidationError as exc:
        raise InvalidPricingInputError(str(exc)) from exc
    normalized_params["r"] = equivalent["risk_free_rate"]
    normalized_params["sigma"] = equivalent["volatility"]
    product = get_product_definition("phoenix")
    if product is None or not product.reference_pricing_enabled:
        raise UnsupportedProductError("unsupported product: phoenix")

    validated_paths = validate_reference_path_count(n_paths)
    started = time.perf_counter()
    reference = price_phoenix_piecewise_reference(
        payoff=product.payoff_class(),
        params=normalized_params,
        market=market,
        n_paths=validated_paths,
        seed=int(seed),
    )
    latency_ms = int(round((time.perf_counter() - started) * 1000))
    result = _build_pricing_result(
        product=product,
        normalized_params=normalized_params,
        n_paths=validated_paths,
        model_version=EQUITY_GBM_PIECEWISE_MODEL_VERSION,
        reference=reference,
        latency_ms=latency_ms,
        market_term_structure=market,
        terms=normalized_terms,
    )
    try:
        from app.services.surrogate_service import evaluate_surrogate_shadow

        shadow = evaluate_surrogate_shadow(
            market=market,
            terms=normalized_terms,
            contract_reference_spot=market.spot,
            reference_price=result["price"],
            reference_standard_error=result["standard_error"],
        )
        if shadow is not None:
            try:
                from app.services.surrogate_monitoring import (
                    record_surrogate_shadow_observation,
                )

                shadow["telemetry_recorded"] = record_surrogate_shadow_observation(
                    market=market,
                    terms=normalized_terms,
                    contract_reference_spot=market.spot,
                    reference_price=result["price"],
                    reference_standard_error=result["standard_error"],
                    shadow_result=shadow,
                    reference_latency_ms=result["latency_ms"],
                )
            except Exception:
                shadow["telemetry_recorded"] = False
            result["surrogate_shadow"] = shadow
    except Exception:
        result["surrogate_shadow"] = {
            "status": "error",
            "mode": "shadow-only",
            "used_for_price": False,
            "reason": "surrogate shadow evaluation failed",
        }
    return result


def price_phoenix_v2_with_term_structure(
    market: EquityMarketTermStructure,
    contract: PhoenixSingleV2Contract,
    n_paths: int = 500,
    seed: int = DEFAULT_REFERENCE_SEED,
) -> Dict[str, Any]:
    """Price explicit active-trade Phoenix state under piecewise market data."""
    if not isinstance(market, EquityMarketTermStructure):
        raise InvalidPricingInputError("invalid equity market term structure")
    if not isinstance(contract, PhoenixSingleV2Contract):
        raise InvalidPricingInputError("invalid Phoenix Single v2 contract")
    try:
        equivalent = market.equivalent_flat_parameters(contract.maturity_years)
    except MarketDataValidationError as exc:
        raise InvalidPricingInputError(str(exc)) from exc
    product = get_product_definition("phoenix")
    if product is None or not product.reference_pricing_enabled:
        raise UnsupportedProductError("unsupported product: phoenix")

    validated_paths = validate_reference_path_count(n_paths)
    started = time.perf_counter()
    reference = price_phoenix_v2_piecewise_reference(
        payoff=product.payoff_class(),
        contract=contract,
        market=market,
        n_paths=validated_paths,
        seed=int(seed),
    )
    latency_ms = int(round((time.perf_counter() - started) * 1000))
    normalized_params = contract.to_payoff_params(
        risk_free_rate=equivalent["risk_free_rate"],
        volatility=equivalent["volatility"],
    )
    normalized_terms = {
        "maturity_years": contract.maturity_years,
        "autocall_barrier_frac": contract.autocall_barrier_frac,
        "coupon_barrier_frac": contract.coupon_barrier_frac,
        "coupon_rate": contract.coupon_rate,
        "knock_in_frac": contract.knock_in_frac,
        "obs_count": len(contract.observation_times_years),
    }
    result = _build_pricing_result(
        product=product,
        normalized_params=normalized_params,
        n_paths=validated_paths,
        model_version=EQUITY_GBM_PIECEWISE_MODEL_VERSION,
        reference=reference,
        latency_ms=latency_ms,
        market_term_structure=market,
        terms=normalized_terms,
        contract_version=PHOENIX_SINGLE_V2_CONTRACT_VERSION,
        contract_details=contract.to_dict(),
        warnings=[
            "The request must describe an active note; already-autocalled "
            "contracts have no remaining optionality.",
            "Historical knock-in state is caller-supplied and is not inferred "
            "from market data.",
            "Knock-in monitoring from valuation onward remains discrete on the "
            "simulation grid.",
        ],
    )
    result["surrogate_shadow"] = {
        "status": "not_applicable",
        "mode": "shadow-only",
        "used_for_price": False,
        "reason": (
            "the approved surrogate is governed only for phoenix-single-v1 "
            "new-issue contracts"
        ),
    }
    return result


def price_phoenix_v3_with_term_structure(
    market: EquityMarketTermStructure,
    contract: PhoenixSingleV3Contract,
    n_paths: int = 500,
    seed: int = DEFAULT_REFERENCE_SEED,
) -> Dict[str, Any]:
    """Price a seasoned memory/step-down Phoenix under piecewise market data."""
    if not isinstance(market, EquityMarketTermStructure):
        raise InvalidPricingInputError("invalid equity market term structure")
    if not isinstance(contract, PhoenixSingleV3Contract):
        raise InvalidPricingInputError("invalid Phoenix Single v3 contract")
    try:
        equivalent = market.equivalent_flat_parameters(contract.maturity_years)
    except MarketDataValidationError as exc:
        raise InvalidPricingInputError(str(exc)) from exc
    product = get_product_definition("phoenix")
    if product is None or not product.reference_pricing_enabled:
        raise UnsupportedProductError("unsupported product: phoenix")

    validated_paths = validate_reference_path_count(n_paths)
    started = time.perf_counter()
    reference = price_phoenix_v3_piecewise_reference(
        payoff=product.payoff_class(),
        contract=contract,
        market=market,
        n_paths=validated_paths,
        seed=int(seed),
    )
    latency_ms = int(round((time.perf_counter() - started) * 1000))
    normalized_params = contract.to_payoff_params(
        risk_free_rate=equivalent["risk_free_rate"],
        volatility=equivalent["volatility"],
    )
    normalized_terms = {
        "maturity_years": contract.maturity_years,
        "autocall_barrier_fracs": list(contract.autocall_barrier_fracs),
        "coupon_barrier_frac": contract.coupon_barrier_frac,
        "coupon_rate": contract.coupon_rate,
        "knock_in_frac": contract.knock_in_frac,
        "obs_count": len(contract.observation_times_years),
        "memory_coupon": contract.memory_coupon,
        "unpaid_coupon_count": contract.unpaid_coupon_count,
    }
    result = _build_pricing_result(
        product=product,
        normalized_params=normalized_params,
        n_paths=validated_paths,
        model_version=EQUITY_GBM_PIECEWISE_MODEL_VERSION,
        reference=reference,
        latency_ms=latency_ms,
        market_term_structure=market,
        terms=normalized_terms,
        contract_version=PHOENIX_SINGLE_V3_CONTRACT_VERSION,
        contract_details=contract.to_dict(),
        warnings=[
            "Missed memory coupons are paid only when a later observation "
            "satisfies the coupon condition; unpaid coupons are not guaranteed.",
            "Historical knock-in and unpaid-coupon state are caller-supplied "
            "and are not inferred from market data.",
            "Knock-in monitoring from valuation onward remains discrete on the "
            "simulation grid.",
        ],
    )
    _attach_expanded_shadow(
        result=result,
        product_key="phoenix_v3",
        market=market,
        contract=contract,
    )
    return result


def price_barrier_reverse_convertible_with_term_structure(
    market: EquityMarketTermStructure,
    contract: BarrierReverseConvertibleV1Contract,
    n_paths: int = 500,
    seed: int = DEFAULT_REFERENCE_SEED,
) -> Dict[str, Any]:
    """Price a focused barrier reverse convertible reference contract."""
    if not isinstance(market, EquityMarketTermStructure):
        raise InvalidPricingInputError("invalid equity market term structure")
    if not isinstance(contract, BarrierReverseConvertibleV1Contract):
        raise InvalidPricingInputError("invalid barrier reverse convertible contract")
    try:
        equivalent = market.equivalent_flat_parameters(contract.maturity_years)
    except MarketDataValidationError as exc:
        raise InvalidPricingInputError(str(exc)) from exc
    product = get_product_definition("barrier_reverse_convertible")
    if product is None or not product.reference_pricing_enabled:
        raise UnsupportedProductError(
            "unsupported product: barrier_reverse_convertible"
        )

    validated_paths = validate_reference_path_count(n_paths)
    started = time.perf_counter()
    reference = price_barrier_reverse_convertible_reference(
        payoff=product.payoff_class(),
        contract=contract,
        market=market,
        n_paths=validated_paths,
        seed=int(seed),
    )
    latency_ms = int(round((time.perf_counter() - started) * 1000))
    normalized_params = contract.to_payoff_params(
        risk_free_rate=equivalent["risk_free_rate"],
        volatility=equivalent["volatility"],
    )
    result = _build_pricing_result(
        product=product,
        normalized_params=normalized_params,
        n_paths=validated_paths,
        model_version=EQUITY_GBM_PIECEWISE_MODEL_VERSION,
        reference=reference,
        latency_ms=latency_ms,
        market_term_structure=market,
        terms={
            "maturity_years": contract.maturity_years,
            "coupon_rate_per_period": contract.coupon_rate_per_period,
            "strike_frac": contract.strike_frac,
            "knock_in_frac": contract.knock_in_frac,
            "obs_count": len(contract.coupon_times_years),
        },
        contract_version=BARRIER_REVERSE_CONVERTIBLE_V1,
        contract_details=contract.to_dict(),
        warnings=[
            "Coupons are contractual in this simplified research payoff and "
            "issuer credit/default risk is not modelled.",
            "Knock-in monitoring from valuation onward is discrete on the "
            "simulation grid.",
            "Historical knock-in state is caller-supplied.",
        ],
    )
    _attach_expanded_shadow(
        result=result,
        product_key="barrier_reverse_convertible",
        market=market,
        contract=contract,
    )
    return result
