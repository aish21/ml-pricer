import hashlib
import json
import math
import time
from typing import Any, Mapping

import numpy as np

from src.final.market import (
    EQUITY_GBM_PIECEWISE_MODEL_VERSION,
    EQUITY_MARKET_SCENARIO_VERSION,
    EQUITY_RISK_ANALYTICS_VERSION,
    EquityMarketSegment,
    EquityMarketTermStructure,
    MarketDataValidationError,
)
from src.final.payoffs import PhoenixPayoff
from src.final.reference_pricer import (
    DEFAULT_REFERENCE_SEED,
    DEFAULT_REFERENCE_STEPS,
    phoenix_piecewise_discounted_payoffs,
)

from app.services.pricing_service import (
    InvalidPricingInputError,
    normalize_phoenix_market_terms,
    validate_reference_path_count,
)


class RiskAnalyticsError(Exception):
    pass


class InvalidRiskInputError(RiskAnalyticsError):
    pass


_PARALLEL_SHOCK_FIELDS = {
    "spot_pct",
    "rate_parallel_bps",
    "dividend_parallel_bps",
    "volatility_parallel_abs",
    "segment_shocks",
}
_SEGMENT_SHOCK_FIELDS = {
    "segment_index",
    "rate_bps",
    "dividend_bps",
    "volatility_abs",
}
_RISK_BUMP_FIELDS = {
    "spot_relative",
    "volatility_absolute",
    "rate_bps",
    "dividend_bps",
}
DEFAULT_RISK_BUMPS = {
    "spot_relative": 0.01,
    "volatility_absolute": 0.01,
    "rate_bps": 10.0,
    "dividend_bps": 10.0,
}


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise InvalidRiskInputError(f"{name} must be numeric")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise InvalidRiskInputError(f"{name} must be numeric") from exc
    if not math.isfinite(parsed):
        raise InvalidRiskInputError(f"{name} must be finite")
    return parsed


def _bounded_shock(value: Any, name: str, *, lower: float, upper: float) -> float:
    parsed = _finite_float(value, name)
    if parsed < lower or parsed > upper:
        raise InvalidRiskInputError(f"{name} must be between {lower} and {upper}")
    return parsed


def normalize_term_structure_shock(shock: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(shock, Mapping):
        raise InvalidRiskInputError("shock must be an object")
    extra = sorted(set(shock) - _PARALLEL_SHOCK_FIELDS)
    if extra:
        raise InvalidRiskInputError(f"unknown shock fields: {', '.join(extra)}")

    normalized: dict[str, Any] = {}
    bounds = {
        "spot_pct": (-99.999999, 1_000.0),
        "rate_parallel_bps": (-10_000.0, 10_000.0),
        "dividend_parallel_bps": (-10_000.0, 10_000.0),
        "volatility_parallel_abs": (-5.0, 5.0),
    }
    for name, (lower, upper) in bounds.items():
        value = shock.get(name)
        if value is not None:
            normalized[name] = _bounded_shock(value, name, lower=lower, upper=upper)

    raw_segments = shock.get("segment_shocks")
    if raw_segments is not None:
        if not isinstance(raw_segments, (list, tuple)):
            raise InvalidRiskInputError("segment_shocks must be an array")
        segment_shocks: list[dict[str, Any]] = []
        seen_indices: set[int] = set()
        for position, raw_segment in enumerate(raw_segments):
            if not isinstance(raw_segment, Mapping):
                raise InvalidRiskInputError(
                    f"segment_shocks[{position}] must be an object"
                )
            extra_segment = sorted(set(raw_segment) - _SEGMENT_SHOCK_FIELDS)
            if extra_segment:
                raise InvalidRiskInputError(
                    "unknown segment shock fields: " + ", ".join(extra_segment)
                )
            raw_index = raw_segment.get("segment_index")
            if isinstance(raw_index, bool):
                raise InvalidRiskInputError("segment_index must be an integer")
            try:
                index = int(raw_index)
            except (TypeError, ValueError) as exc:
                raise InvalidRiskInputError("segment_index must be an integer") from exc
            if raw_index is None or index < 0 or index > 251:
                raise InvalidRiskInputError("segment_index must be between 0 and 251")
            if isinstance(raw_index, float) and raw_index != index:
                raise InvalidRiskInputError("segment_index must be an integer")
            if index in seen_indices:
                raise InvalidRiskInputError(f"duplicate segment shock index: {index}")
            seen_indices.add(index)

            item: dict[str, Any] = {"segment_index": index}
            for name, (lower, upper) in {
                "rate_bps": (-10_000.0, 10_000.0),
                "dividend_bps": (-10_000.0, 10_000.0),
                "volatility_abs": (-5.0, 5.0),
            }.items():
                value = raw_segment.get(name)
                if value is not None:
                    item[name] = _bounded_shock(
                        value,
                        f"segment_shocks[{position}].{name}",
                        lower=lower,
                        upper=upper,
                    )
            if len(item) == 1 or all(value == 0.0 for value in list(item.values())[1:]):
                raise InvalidRiskInputError(
                    f"segment_shocks[{position}] requires a non-zero market shock"
                )
            segment_shocks.append(item)
        if segment_shocks:
            normalized["segment_shocks"] = sorted(
                segment_shocks, key=lambda item: item["segment_index"]
            )

    scalar_values = [
        value for name, value in normalized.items() if name != "segment_shocks"
    ]
    if not normalized or (
        not normalized.get("segment_shocks")
        and all(value == 0.0 for value in scalar_values)
    ):
        raise InvalidRiskInputError("at least one non-zero market shock is required")
    return normalized


def apply_term_structure_shock(
    market: EquityMarketTermStructure,
    shock: Mapping[str, Any],
) -> tuple[EquityMarketTermStructure, dict[str, Any]]:
    if not isinstance(market, EquityMarketTermStructure):
        raise InvalidRiskInputError("invalid equity market term structure")
    normalized = normalize_term_structure_shock(shock)
    segment_shocks = {
        item["segment_index"]: item for item in normalized.get("segment_shocks", [])
    }
    for index in segment_shocks:
        if index >= len(market.segments):
            raise InvalidRiskInputError(
                f"segment_index {index} is outside the {len(market.segments)}-segment curve"
            )

    shocked_segments = []
    for index, segment in enumerate(market.segments):
        local = segment_shocks.get(index, {})
        rate = (
            segment.risk_free_rate
            + (normalized.get("rate_parallel_bps", 0.0) + local.get("rate_bps", 0.0))
            / 10_000.0
        )
        dividend = (
            segment.dividend_yield
            + (
                normalized.get("dividend_parallel_bps", 0.0)
                + local.get("dividend_bps", 0.0)
            )
            / 10_000.0
        )
        volatility = (
            segment.volatility
            + normalized.get("volatility_parallel_abs", 0.0)
            + local.get("volatility_abs", 0.0)
        )
        try:
            shocked_segments.append(
                EquityMarketSegment(
                    end_time_years=segment.end_time_years,
                    risk_free_rate=rate,
                    dividend_yield=dividend,
                    volatility=volatility,
                )
            )
        except MarketDataValidationError as exc:
            raise InvalidRiskInputError(
                f"shock makes segment {index} invalid: {exc}"
            ) from exc

    shocked_spot = market.spot * (1.0 + normalized.get("spot_pct", 0.0) / 100.0)
    try:
        shocked_market = EquityMarketTermStructure(
            symbol=market.symbol,
            underlier_type=market.underlier_type,
            currency=market.currency,
            valuation_time=market.valuation_time,
            market_data_time=market.market_data_time,
            spot=shocked_spot,
            segments=tuple(shocked_segments),
            calendar=market.calendar,
            day_count=market.day_count,
            source=market.source,
        )
    except MarketDataValidationError as exc:
        raise InvalidRiskInputError(f"shock makes market invalid: {exc}") from exc
    return shocked_market, normalized


def normalize_risk_bumps(bumps: Mapping[str, Any] | None) -> dict[str, float]:
    supplied = {} if bumps is None else bumps
    if not isinstance(supplied, Mapping):
        raise InvalidRiskInputError("bumps must be an object")
    extra = sorted(set(supplied) - _RISK_BUMP_FIELDS)
    if extra:
        raise InvalidRiskInputError(f"unknown bump fields: {', '.join(extra)}")
    normalized = dict(DEFAULT_RISK_BUMPS)
    limits = {
        "spot_relative": (0.000001, 0.5),
        "volatility_absolute": (0.000001, 1.0),
        "rate_bps": (0.000001, 5_000.0),
        "dividend_bps": (0.000001, 5_000.0),
    }
    for name, value in supplied.items():
        if value is not None:
            lower, upper = limits[name]
            normalized[name] = _bounded_shock(value, name, lower=lower, upper=upper)
    return normalized


def _validate_seed(seed: Any) -> int:
    if isinstance(seed, bool):
        raise InvalidRiskInputError("seed must be an integer")
    try:
        parsed = int(seed)
    except (TypeError, ValueError) as exc:
        raise InvalidRiskInputError("seed must be an integer") from exc
    if parsed < 0 or parsed > 2**32 - 1:
        raise InvalidRiskInputError("seed must be between 0 and 4294967295")
    return parsed


def _sample_statistics(samples: np.ndarray) -> dict[str, Any]:
    values = np.asarray(samples, dtype=np.float64)
    if values.ndim != 1 or values.size < 1 or not np.all(np.isfinite(values)):
        raise RiskAnalyticsError("risk calculation produced invalid pathwise values")
    estimate = float(np.mean(values))
    sample_std = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
    standard_error = sample_std / math.sqrt(values.size)
    half_width = 1.96 * standard_error
    signal_to_noise = abs(estimate) / standard_error if standard_error > 0.0 else None
    return {
        "value": estimate,
        "sample_std": sample_std,
        "standard_error": standard_error,
        "confidence_level": 0.95,
        "confidence_interval": [estimate - half_width, estimate + half_width],
        "statistically_resolved_95pct": bool(
            estimate - half_width > 0.0 or estimate + half_width < 0.0
        ),
        "signal_to_noise": signal_to_noise,
    }


def _valuation_statistics(
    payoffs: np.ndarray, market: EquityMarketTermStructure
) -> dict[str, Any]:
    result = _sample_statistics(payoffs)
    result["price"] = result.pop("value")
    result["term_structure_id"] = market.term_structure_id
    return result


def _fingerprint(version: str, payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        {"version": version, **payload},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _prepare_evaluation(
    market: EquityMarketTermStructure,
    terms: Mapping[str, Any],
    n_paths: Any,
    seed: Any,
) -> tuple[dict[str, Any], dict[str, Any], int, int, np.ndarray]:
    if not isinstance(market, EquityMarketTermStructure):
        raise InvalidRiskInputError("invalid equity market term structure")
    try:
        params, normalized_terms = normalize_phoenix_market_terms(
            spot=market.spot,
            risk_free_rate=0.0,
            volatility=1.0,
            terms=dict(terms),
        )
        market.equivalent_flat_parameters(params["T"])
        validated_paths = validate_reference_path_count(n_paths)
    except (
        InvalidPricingInputError,
        MarketDataValidationError,
        TypeError,
        ValueError,
    ) as exc:
        raise InvalidRiskInputError(str(exc)) from exc
    validated_seed = _validate_seed(seed)
    random_state = np.random.RandomState(validated_seed)
    common_shocks = random_state.randn(validated_paths, DEFAULT_REFERENCE_STEPS)
    return params, normalized_terms, validated_paths, validated_seed, common_shocks


def _evaluate(
    *,
    payoff: PhoenixPayoff,
    params: dict[str, Any],
    market: EquityMarketTermStructure,
    n_paths: int,
    common_shocks: np.ndarray,
) -> np.ndarray:
    try:
        return phoenix_piecewise_discounted_payoffs(
            payoff=payoff,
            params=params,
            market=market,
            n_paths=n_paths,
            standard_normal_shocks=common_shocks,
            seed=None,
        )
    except (MarketDataValidationError, ValueError) as exc:
        raise InvalidRiskInputError(str(exc)) from exc


def _provenance(
    *,
    market: EquityMarketTermStructure,
    n_paths: int,
    seed: int,
    market_calibration: Mapping[str, Any] | None,
) -> dict[str, Any]:
    calibration_id = (
        market_calibration.get("calibration_id")
        if isinstance(market_calibration, Mapping)
        else None
    )
    return {
        "contract_version": PhoenixPayoff.contract_version,
        "model_version": EQUITY_GBM_PIECEWISE_MODEL_VERSION,
        "market_data_version": market.schema_version,
        "base_term_structure_id": market.term_structure_id,
        "market_calibration_id": calibration_id,
        "n_paths": n_paths,
        "n_steps": DEFAULT_REFERENCE_STEPS,
        "seed": seed,
        "common_random_numbers": True,
        "contract_reference_spot": market.spot,
    }


def run_phoenix_term_structure_scenario(
    *,
    market: EquityMarketTermStructure,
    terms: Mapping[str, Any],
    shock: Mapping[str, Any],
    n_paths: int = 2_000,
    seed: int = DEFAULT_REFERENCE_SEED,
    market_calibration: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    shocked_market, normalized_shock = apply_term_structure_shock(market, shock)
    params, normalized_terms, paths, validated_seed, common_shocks = (
        _prepare_evaluation(market, terms, n_paths, seed)
    )
    payoff = PhoenixPayoff()
    base_payoffs = _evaluate(
        payoff=payoff,
        params=params,
        market=market,
        n_paths=paths,
        common_shocks=common_shocks,
    )
    shocked_payoffs = _evaluate(
        payoff=payoff,
        params=params,
        market=shocked_market,
        n_paths=paths,
        common_shocks=common_shocks,
    )
    pnl = _sample_statistics(shocked_payoffs - base_payoffs)
    provenance = _provenance(
        market=market,
        n_paths=paths,
        seed=validated_seed,
        market_calibration=market_calibration,
    )
    provenance["shocked_term_structure_id"] = shocked_market.term_structure_id
    identity_payload = {
        "base_term_structure_id": market.term_structure_id,
        "shocked_term_structure_id": shocked_market.term_structure_id,
        "terms": normalized_terms,
        "shock": normalized_shock,
        "n_paths": paths,
        "seed": validated_seed,
        "market_calibration_id": provenance["market_calibration_id"],
    }
    return {
        "scenario_version": EQUITY_MARKET_SCENARIO_VERSION,
        "scenario_id": _fingerprint(EQUITY_MARKET_SCENARIO_VERSION, identity_payload),
        "product_key": "phoenix",
        "model": "Monte Carlo reference",
        "pricing_method": "paired_monte_carlo_reference",
        "terms": normalized_terms,
        "shock": normalized_shock,
        "base_market": market.to_dict(),
        "shocked_market": shocked_market.to_dict(),
        "base_valuation": _valuation_statistics(base_payoffs, market),
        "shocked_valuation": _valuation_statistics(shocked_payoffs, shocked_market),
        "pnl": pnl,
        "provenance": provenance,
        "market_calibration": (
            dict(market_calibration) if market_calibration is not None else None
        ),
        "warnings": [
            "Contractual reference spot and barriers are frozen at the base market.",
            "Scenario P&L uses paired Monte Carlo paths and remains a statistical estimate.",
        ],
        "latency_ms": int(round((time.perf_counter() - started) * 1_000)),
    }


def _sensitivity(
    samples: np.ndarray,
    *,
    units: str,
    bump: dict[str, Any],
    up_price: float,
    down_price: float,
) -> dict[str, Any]:
    result = _sample_statistics(samples)
    result.update(
        {
            "units": units,
            "bump": bump,
            "up_price": up_price,
            "down_price": down_price,
        }
    )
    return result


def calculate_phoenix_term_structure_risk(
    *,
    market: EquityMarketTermStructure,
    terms: Mapping[str, Any],
    n_paths: int = 2_000,
    seed: int = DEFAULT_REFERENCE_SEED,
    bumps: Mapping[str, Any] | None = None,
    market_calibration: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    normalized_bumps = normalize_risk_bumps(bumps)
    params, normalized_terms, paths, validated_seed, common_shocks = (
        _prepare_evaluation(market, terms, n_paths, seed)
    )

    spot_pct = normalized_bumps["spot_relative"] * 100.0
    vol_bump = normalized_bumps["volatility_absolute"]
    rate_bump = normalized_bumps["rate_bps"]
    dividend_bump = normalized_bumps["dividend_bps"]
    markets = {"base": market}
    shock_specs = {
        "spot_up": {"spot_pct": spot_pct},
        "spot_down": {"spot_pct": -spot_pct},
        "volatility_up": {"volatility_parallel_abs": vol_bump},
        "volatility_down": {"volatility_parallel_abs": -vol_bump},
        "rate_up": {"rate_parallel_bps": rate_bump},
        "rate_down": {"rate_parallel_bps": -rate_bump},
        "dividend_up": {"dividend_parallel_bps": dividend_bump},
        "dividend_down": {"dividend_parallel_bps": -dividend_bump},
    }
    for name, shock_spec in shock_specs.items():
        markets[name], _ = apply_term_structure_shock(market, shock_spec)

    payoff = PhoenixPayoff()
    payoffs = {
        name: _evaluate(
            payoff=payoff,
            params=params,
            market=scenario_market,
            n_paths=paths,
            common_shocks=common_shocks,
        )
        for name, scenario_market in markets.items()
    }
    prices = {name: float(np.mean(values)) for name, values in payoffs.items()}

    spot_h = market.spot * normalized_bumps["spot_relative"]
    rate_h = rate_bump / 10_000.0
    dividend_h = dividend_bump / 10_000.0
    delta_samples = (payoffs["spot_up"] - payoffs["spot_down"]) / (2.0 * spot_h)
    gamma_samples = (
        payoffs["spot_up"] - 2.0 * payoffs["base"] + payoffs["spot_down"]
    ) / (spot_h**2)
    vega_samples = (
        (payoffs["volatility_up"] - payoffs["volatility_down"])
        / (2.0 * vol_bump)
        * 0.01
    )
    rho_samples = (payoffs["rate_up"] - payoffs["rate_down"]) / (2.0 * rate_h) * 0.01
    dividend_rho_samples = (
        (payoffs["dividend_up"] - payoffs["dividend_down"]) / (2.0 * dividend_h) * 0.01
    )

    sensitivities = {
        "delta": _sensitivity(
            delta_samples,
            units="price per 1 currency unit of spot",
            bump={"spot_relative": normalized_bumps["spot_relative"]},
            up_price=prices["spot_up"],
            down_price=prices["spot_down"],
        ),
        "gamma": _sensitivity(
            gamma_samples,
            units="price per squared currency unit of spot",
            bump={"spot_relative": normalized_bumps["spot_relative"]},
            up_price=prices["spot_up"],
            down_price=prices["spot_down"],
        ),
        "vega": _sensitivity(
            vega_samples,
            units="price change per 1 volatility point",
            bump={"volatility_absolute": vol_bump},
            up_price=prices["volatility_up"],
            down_price=prices["volatility_down"],
        ),
        "rho": _sensitivity(
            rho_samples,
            units="price change per 100 basis points of rates",
            bump={"rate_bps": rate_bump},
            up_price=prices["rate_up"],
            down_price=prices["rate_down"],
        ),
        "dividend_rho": _sensitivity(
            dividend_rho_samples,
            units="price change per 100 basis points of dividend yield",
            bump={"dividend_bps": dividend_bump},
            up_price=prices["dividend_up"],
            down_price=prices["dividend_down"],
        ),
    }
    provenance = _provenance(
        market=market,
        n_paths=paths,
        seed=validated_seed,
        market_calibration=market_calibration,
    )
    identity_payload = {
        "base_term_structure_id": market.term_structure_id,
        "terms": normalized_terms,
        "bumps": normalized_bumps,
        "n_paths": paths,
        "seed": validated_seed,
        "market_calibration_id": provenance["market_calibration_id"],
    }
    valuations = {
        name: _valuation_statistics(values, markets[name])
        for name, values in payoffs.items()
    }
    return {
        "risk_version": EQUITY_RISK_ANALYTICS_VERSION,
        "risk_id": _fingerprint(EQUITY_RISK_ANALYTICS_VERSION, identity_payload),
        "product_key": "phoenix",
        "model": "Monte Carlo reference",
        "pricing_method": "common_random_number_finite_difference",
        "terms": normalized_terms,
        "bumps": normalized_bumps,
        "base_market": market.to_dict(),
        "base_valuation": valuations["base"],
        "sensitivities": sensitivities,
        "bump_valuations": {
            name: value for name, value in valuations.items() if name != "base"
        },
        "provenance": provenance,
        "market_calibration": (
            dict(market_calibration) if market_calibration is not None else None
        ),
        "warnings": [
            "Finite-difference Greeks freeze the calibrated curve except for the named bump.",
            "Phoenix barriers are discontinuous; inspect confidence intervals and bump stability.",
            "Contractual reference spot and barriers are frozen at the base market.",
        ],
        "latency_ms": int(round((time.perf_counter() - started) * 1_000)),
    }
