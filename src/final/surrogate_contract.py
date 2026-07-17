import math
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

import numpy as np

from .market import (
    EQUITY_GBM_PIECEWISE_MODEL_VERSION,
    EQUITY_MARKET_TERM_STRUCTURE_VERSION,
    EquityMarketSegment,
    EquityMarketTermStructure,
    MarketDataValidationError,
)
from .payoffs import PhoenixPayoff


PHOENIX_SURROGATE_FEATURE_VERSION = "phoenix-surrogate-features-v3"
PHOENIX_SURROGATE_LABEL_VERSION = "phoenix-piecewise-payoff-aware-label-v2"
PHOENIX_SURROGATE_MODEL_VERSION = "phoenix-surrogate-payoff-aware-v5"
PHOENIX_SURROGATE_ARTIFACT_VERSION = "phoenix-surrogate-artifact-v3"
CURVE_TIME_FRACTIONS = (0.25, 0.5, 0.75, 1.0)
TERM_FEATURE_NAMES = (
    "spot_ratio",
    "log_spot_to_autocall_barrier",
    "log_spot_to_coupon_barrier",
    "log_spot_to_knock_in_barrier",
    "maturity_years",
    "autocall_barrier_frac",
    "coupon_barrier_frac",
    "coupon_rate",
    "knock_in_frac",
    "obs_count",
    "maximum_coupon_amount",
    "coupon_rate_per_year",
)
CURVE_FEATURE_NAMES = tuple(
    name
    for fraction_label in ("25", "50", "75", "100")
    for name in (
        f"zero_rate_t{fraction_label}",
        f"dividend_yield_t{fraction_label}",
        f"total_variance_t{fraction_label}",
    )
)
PHOENIX_SURROGATE_FEATURE_NAMES = TERM_FEATURE_NAMES + CURVE_FEATURE_NAMES
PHOENIX_PRICE_COMPONENT_NAMES = (
    "coupon_pv",
    "autocall_principal_pv",
    "maturity_protected_pv",
    "maturity_downside_pv",
)
PHOENIX_EVENT_TARGET_NAMES = (
    "autocall_probability",
    "downside_probability",
)
PHOENIX_PAYOFF_AWARE_TARGET_NAMES = (
    PHOENIX_PRICE_COMPONENT_NAMES + PHOENIX_EVENT_TARGET_NAMES
)
PHOENIX_PAYOFF_AWARE_MODEL_OUTPUT_NAMES = ("price",) + PHOENIX_PAYOFF_AWARE_TARGET_NAMES

DEFAULT_TRAINING_DOMAIN: dict[str, tuple[float, float]] = {
    "spot_ratio": (0.5, 1.5),
    "maturity_years": (0.5, 2.5),
    "autocall_barrier_frac": (0.95, 1.15),
    "coupon_barrier_frac": (0.7, 1.05),
    "coupon_rate": (0.005, 0.05),
    "knock_in_frac": (0.5, 0.95),
    "obs_count": (4.0, 12.0),
    "risk_free_rate": (-0.02, 0.12),
    "dividend_yield": (0.0, 0.08),
    "volatility": (0.08, 0.9),
}


class SurrogateContractError(ValueError):
    pass


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise SurrogateContractError(f"{name} must be numeric")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise SurrogateContractError(f"{name} must be numeric") from exc
    if not math.isfinite(parsed):
        raise SurrogateContractError(f"{name} must be finite")
    return parsed


def normalize_phoenix_surrogate_terms(terms: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(terms, Mapping):
        raise SurrogateContractError("Phoenix terms must be an object")
    required = {
        "maturity_years",
        "autocall_barrier_frac",
        "coupon_barrier_frac",
        "coupon_rate",
        "knock_in_frac",
        "obs_count",
    }
    missing = sorted(required - set(terms))
    extra = sorted(set(terms) - required)
    if missing:
        raise SurrogateContractError(
            f"missing required Phoenix terms: {', '.join(missing)}"
        )
    if extra:
        raise SurrogateContractError(f"unknown Phoenix terms: {', '.join(extra)}")
    normalized = {
        name: _finite_float(terms[name], name) for name in required - {"obs_count"}
    }
    raw_obs = terms["obs_count"]
    if isinstance(raw_obs, bool):
        raise SurrogateContractError("obs_count must be an integer")
    try:
        obs_count = int(raw_obs)
    except (TypeError, ValueError) as exc:
        raise SurrogateContractError("obs_count must be an integer") from exc
    if isinstance(raw_obs, float) and raw_obs != obs_count:
        raise SurrogateContractError("obs_count must be an integer")
    normalized["obs_count"] = obs_count
    if normalized["maturity_years"] <= 0.0:
        raise SurrogateContractError("maturity_years must be > 0")
    if normalized["coupon_rate"] < 0.0:
        raise SurrogateContractError("coupon_rate must be >= 0")
    if obs_count < 1 or obs_count > 252:
        raise SurrogateContractError("obs_count must be between 1 and 252")
    if not (
        0.0
        < normalized["knock_in_frac"]
        <= normalized["coupon_barrier_frac"]
        <= normalized["autocall_barrier_frac"]
    ):
        raise SurrogateContractError(
            "barriers must satisfy knock_in_frac <= coupon_barrier_frac "
            "<= autocall_barrier_frac"
        )
    return normalized


def extract_phoenix_surrogate_features(
    *,
    market: EquityMarketTermStructure,
    terms: Mapping[str, Any],
    contract_reference_spot: float,
) -> np.ndarray:
    if not isinstance(market, EquityMarketTermStructure):
        raise SurrogateContractError("invalid equity market term structure")
    reference_spot = _finite_float(contract_reference_spot, "contract_reference_spot")
    if reference_spot <= 0.0:
        raise SurrogateContractError("contract_reference_spot must be > 0")
    normalized_terms = normalize_phoenix_surrogate_terms(terms)
    maturity = normalized_terms["maturity_years"]
    if maturity > market.max_time_years + 1e-12:
        raise SurrogateContractError(
            "term structure does not cover the product maturity"
        )
    values = [
        market.spot / reference_spot,
        math.log(
            market.spot / (reference_spot * normalized_terms["autocall_barrier_frac"])
        ),
        math.log(
            market.spot / (reference_spot * normalized_terms["coupon_barrier_frac"])
        ),
        math.log(market.spot / (reference_spot * normalized_terms["knock_in_frac"])),
        maturity,
        normalized_terms["autocall_barrier_frac"],
        normalized_terms["coupon_barrier_frac"],
        normalized_terms["coupon_rate"],
        normalized_terms["knock_in_frac"],
        float(normalized_terms["obs_count"]),
        normalized_terms["coupon_rate"] * normalized_terms["obs_count"],
        normalized_terms["coupon_rate"] * normalized_terms["obs_count"] / maturity,
    ]
    try:
        for fraction in CURVE_TIME_FRACTIONS:
            time_years = maturity * fraction
            values.extend(
                [
                    market.integrated_risk_free_rate(0.0, time_years) / time_years,
                    market.integrated_dividend_yield(0.0, time_years) / time_years,
                    market.integrated_variance(0.0, time_years),
                ]
            )
    except MarketDataValidationError as exc:
        raise SurrogateContractError(str(exc)) from exc
    features = np.asarray(values, dtype=np.float64)
    if features.shape != (len(PHOENIX_SURROGATE_FEATURE_NAMES),):
        raise SurrogateContractError("surrogate feature vector has invalid shape")
    if not np.all(np.isfinite(features)):
        raise SurrogateContractError("surrogate features must be finite")
    return features


def domain_violations(
    *,
    market: EquityMarketTermStructure,
    terms: Mapping[str, Any],
    contract_reference_spot: float,
    domain: Mapping[str, Sequence[float]] | None = None,
) -> list[str]:
    active_domain = domain or DEFAULT_TRAINING_DOMAIN
    try:
        normalized_terms = normalize_phoenix_surrogate_terms(terms)
        reference_spot = _finite_float(
            contract_reference_spot, "contract_reference_spot"
        )
    except SurrogateContractError as exc:
        return [str(exc)]
    values: dict[str, list[float]] = {
        "spot_ratio": (
            [market.spot / reference_spot] if reference_spot > 0.0 else [math.inf]
        ),
        "maturity_years": [normalized_terms["maturity_years"]],
        "autocall_barrier_frac": [normalized_terms["autocall_barrier_frac"]],
        "coupon_barrier_frac": [normalized_terms["coupon_barrier_frac"]],
        "coupon_rate": [normalized_terms["coupon_rate"]],
        "knock_in_frac": [normalized_terms["knock_in_frac"]],
        "obs_count": [float(normalized_terms["obs_count"])],
        "risk_free_rate": [segment.risk_free_rate for segment in market.segments],
        "dividend_yield": [segment.dividend_yield for segment in market.segments],
        "volatility": [segment.volatility for segment in market.segments],
    }
    violations = []
    for name, field_values in values.items():
        bounds = active_domain.get(name)
        if not isinstance(bounds, Sequence) or len(bounds) != 2:
            violations.append(f"training domain missing valid bounds for {name}")
            continue
        lower, upper = float(bounds[0]), float(bounds[1])
        for value in field_values:
            if not math.isfinite(value) or value < lower or value > upper:
                violations.append(
                    f"{name}={value:.8g} is outside [{lower:.8g}, {upper:.8g}]"
                )
                break
    return violations


def reconstruct_phoenix_surrogate_case(
    features: Sequence[float],
) -> tuple[EquityMarketTermStructure, dict[str, Any], float]:
    vector = np.asarray(features, dtype=np.float64)
    expected_shape = (len(PHOENIX_SURROGATE_FEATURE_NAMES),)
    if vector.shape != expected_shape or not np.all(np.isfinite(vector)):
        raise SurrogateContractError(
            f"features must be a finite vector with shape {expected_shape}"
        )
    reference_spot = 100.0
    positions = {
        name: index for index, name in enumerate(PHOENIX_SURROGATE_FEATURE_NAMES)
    }
    maturity = float(vector[positions["maturity_years"]])
    terms = {
        "maturity_years": maturity,
        "autocall_barrier_frac": float(vector[positions["autocall_barrier_frac"]]),
        "coupon_barrier_frac": float(vector[positions["coupon_barrier_frac"]]),
        "coupon_rate": float(vector[positions["coupon_rate"]]),
        "knock_in_frac": float(vector[positions["knock_in_frac"]]),
        "obs_count": int(round(float(vector[positions["obs_count"]]))),
    }
    normalize_phoenix_surrogate_terms(terms)
    spot_ratio = float(vector[positions["spot_ratio"]])
    expected_log_moneyness = {
        "log_spot_to_autocall_barrier": math.log(
            spot_ratio / terms["autocall_barrier_frac"]
        ),
        "log_spot_to_coupon_barrier": math.log(
            spot_ratio / terms["coupon_barrier_frac"]
        ),
        "log_spot_to_knock_in_barrier": math.log(spot_ratio / terms["knock_in_frac"]),
    }
    for name, expected_value in expected_log_moneyness.items():
        if not math.isclose(
            float(vector[positions[name]]),
            expected_value,
            rel_tol=1e-10,
            abs_tol=1e-10,
        ):
            raise SurrogateContractError(
                f"engineered surrogate feature is inconsistent: {name}"
            )
    expected_coupon_features = {
        "maximum_coupon_amount": terms["coupon_rate"] * terms["obs_count"],
        "coupon_rate_per_year": terms["coupon_rate"] * terms["obs_count"] / maturity,
    }
    for name, expected_value in expected_coupon_features.items():
        if not math.isclose(
            float(vector[positions[name]]),
            expected_value,
            rel_tol=1e-10,
            abs_tol=1e-10,
        ):
            raise SurrogateContractError(
                f"engineered surrogate feature is inconsistent: {name}"
            )

    cumulative_rates = []
    cumulative_dividends = []
    cumulative_variances = []
    offset = len(TERM_FEATURE_NAMES)
    for index, fraction in enumerate(CURVE_TIME_FRACTIONS):
        time_years = maturity * fraction
        zero_rate, dividend_yield, total_variance = vector[
            offset + index * 3 : offset + index * 3 + 3
        ]
        cumulative_rates.append(float(zero_rate) * time_years)
        cumulative_dividends.append(float(dividend_yield) * time_years)
        cumulative_variances.append(float(total_variance))

    segments = []
    previous_time = 0.0
    previous_rate = 0.0
    previous_dividend = 0.0
    previous_variance = 0.0
    for index, fraction in enumerate(CURVE_TIME_FRACTIONS):
        end_time = maturity * fraction
        interval = end_time - previous_time
        variance_increment = cumulative_variances[index] - previous_variance
        if variance_increment <= 0.0:
            raise SurrogateContractError(
                "cumulative variance features must be strictly increasing"
            )
        segments.append(
            EquityMarketSegment(
                end_time_years=end_time,
                risk_free_rate=(cumulative_rates[index] - previous_rate) / interval,
                dividend_yield=(cumulative_dividends[index] - previous_dividend)
                / interval,
                volatility=math.sqrt(variance_increment / interval),
            )
        )
        previous_time = end_time
        previous_rate = cumulative_rates[index]
        previous_dividend = cumulative_dividends[index]
        previous_variance = cumulative_variances[index]

    valuation_time = datetime(2026, 1, 2, 16, 0, tzinfo=timezone.utc)
    market = EquityMarketTermStructure(
        symbol="PHOENIX-TRAINING",
        underlier_type="equity",
        currency="USD",
        valuation_time=valuation_time,
        market_data_time=valuation_time,
        spot=reference_spot * spot_ratio,
        segments=tuple(segments),
        calendar="WEEKDAYS",
        day_count="ACT/365F",
        source=PHOENIX_SURROGATE_FEATURE_VERSION,
    )
    return market, terms, reference_spot


def surrogate_contract_metadata() -> dict[str, Any]:
    return {
        "feature_schema_version": PHOENIX_SURROGATE_FEATURE_VERSION,
        "feature_names": list(PHOENIX_SURROGATE_FEATURE_NAMES),
        "payoff_aware_target_names": list(PHOENIX_PAYOFF_AWARE_TARGET_NAMES),
        "payoff_aware_model_output_names": list(
            PHOENIX_PAYOFF_AWARE_MODEL_OUTPUT_NAMES
        ),
        "price_component_names": list(PHOENIX_PRICE_COMPONENT_NAMES),
        "curve_time_fractions": list(CURVE_TIME_FRACTIONS),
        "contract_version": PhoenixPayoff.contract_version,
        "market_data_version": EQUITY_MARKET_TERM_STRUCTURE_VERSION,
        "label_model_version": EQUITY_GBM_PIECEWISE_MODEL_VERSION,
        "label_schema_version": PHOENIX_SURROGATE_LABEL_VERSION,
        "training_domain": {
            name: list(bounds) for name, bounds in DEFAULT_TRAINING_DOMAIN.items()
        },
    }
