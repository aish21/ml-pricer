import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping


class FrontendInputError(ValueError):
    pass


@dataclass(frozen=True)
class PricingConfiguration:
    experience_mode: str
    trade_stage: str
    market_source: str
    symbol: str
    underlier_type: str
    currency: str
    maturity_years: float
    display_notional: float
    n_paths: int
    seed: int
    terms: Mapping[str, Any]
    contract: Mapping[str, Any] | None
    manual_market: Mapping[str, Any] | None
    product_key: str = "phoenix"

    @property
    def is_seasoned(self) -> bool:
        return self.trade_stage == "Seasoned trade"


def even_observation_schedule(
    maturity_years: float,
    observation_count: int,
) -> tuple[float, ...]:
    maturity = float(maturity_years)
    count = int(observation_count)
    if not math.isfinite(maturity) or maturity <= 0.0:
        raise FrontendInputError("Maturity must be positive.")
    if count < 1 or count > 252:
        raise FrontendInputError("Observation count must be between 1 and 252.")
    return tuple((index / count) * maturity for index in range(1, count + 1))


def parse_observation_schedule(
    raw_schedule: str,
    maturity_years: float,
) -> tuple[float, ...]:
    try:
        values = tuple(
            float(item.strip()) for item in raw_schedule.split(",") if item.strip()
        )
    except ValueError as exc:
        raise FrontendInputError(
            "Observation times must be comma-separated numbers."
        ) from exc
    if not values:
        raise FrontendInputError("Enter at least one remaining observation time.")
    if any(not math.isfinite(value) for value in values):
        raise FrontendInputError("Observation times must be finite.")
    if any(value <= 0.0 or value > maturity_years for value in values):
        raise FrontendInputError(
            "Observation times must be after valuation and no later than maturity."
        )
    if any(current <= previous for previous, current in zip(values[:-1], values[1:])):
        raise FrontendInputError("Observation times must be strictly increasing.")
    if not math.isclose(values[-1], maturity_years, abs_tol=1e-12):
        raise FrontendInputError("The final observation must equal maturity.")
    return values


def build_phoenix_terms(
    *,
    maturity_years: float,
    autocall_barrier_frac: float,
    coupon_barrier_frac: float,
    coupon_rate: float,
    knock_in_frac: float,
    observation_count: int,
) -> dict[str, Any]:
    if not knock_in_frac <= coupon_barrier_frac <= autocall_barrier_frac:
        raise FrontendInputError("Barriers must satisfy knock-in ≤ coupon ≤ autocall.")
    return {
        "maturity_years": float(maturity_years),
        "autocall_barrier_frac": float(autocall_barrier_frac),
        "coupon_barrier_frac": float(coupon_barrier_frac),
        "coupon_rate": float(coupon_rate),
        "knock_in_frac": float(knock_in_frac),
        "obs_count": int(observation_count),
    }


def build_v2_contract(
    *,
    reference_level: float,
    terms: Mapping[str, Any],
    observation_times_years: tuple[float, ...],
    prior_knock_in_breached: bool,
) -> dict[str, Any]:
    reference = float(reference_level)
    if not math.isfinite(reference) or reference <= 0.0:
        raise FrontendInputError("The contractual reference level must be positive.")
    return {
        "contract_version": "phoenix-single-v2",
        "reference_level": reference,
        "maturity_years": float(terms["maturity_years"]),
        "observation_times_years": list(observation_times_years),
        "autocall_barrier_frac": float(terms["autocall_barrier_frac"]),
        "coupon_barrier_frac": float(terms["coupon_barrier_frac"]),
        "coupon_rate": float(terms["coupon_rate"]),
        "knock_in_frac": float(terms["knock_in_frac"]),
        "prior_knock_in_breached": bool(prior_knock_in_breached),
    }


def stepped_autocall_schedule(
    *,
    initial_barrier_frac: float,
    final_barrier_frac: float,
    observation_count: int,
) -> tuple[float, ...]:
    initial = float(initial_barrier_frac)
    final = float(final_barrier_frac)
    count = int(observation_count)
    if count < 1 or count > 252:
        raise FrontendInputError("Observation count must be between 1 and 252.")
    if not all(math.isfinite(value) for value in (initial, final)):
        raise FrontendInputError("Autocall barriers must be finite.")
    if initial <= 0.0 or final <= 0.0 or final > initial:
        raise FrontendInputError(
            "The final autocall barrier must be positive and no higher than the first."
        )
    if count == 1:
        return (initial,)
    step = (initial - final) / (count - 1)
    return tuple(initial - index * step for index in range(count))


def build_v3_contract(
    *,
    reference_level: float,
    terms: Mapping[str, Any],
    observation_times_years: tuple[float, ...],
    autocall_barrier_fracs: tuple[float, ...],
    prior_knock_in_breached: bool,
    memory_coupon: bool,
    unpaid_coupon_count: int,
) -> dict[str, Any]:
    reference = float(reference_level)
    if not math.isfinite(reference) or reference <= 0.0:
        raise FrontendInputError("The contractual reference level must be positive.")
    if len(autocall_barrier_fracs) != len(observation_times_years):
        raise FrontendInputError(
            "Enter one autocall barrier for every remaining observation."
        )
    coupon_barrier = float(terms["coupon_barrier_frac"])
    if any(barrier < coupon_barrier for barrier in autocall_barrier_fracs):
        raise FrontendInputError(
            "Every autocall barrier must be at or above the coupon barrier."
        )
    unpaid = int(unpaid_coupon_count)
    if unpaid < 0 or unpaid > 252:
        raise FrontendInputError("Missed memory coupons must be between 0 and 252.")
    if not memory_coupon and unpaid:
        raise FrontendInputError(
            "Missed coupon history requires the memory-coupon feature."
        )
    return {
        "contract_version": "phoenix-single-v3",
        "reference_level": reference,
        "maturity_years": float(terms["maturity_years"]),
        "observation_times_years": list(observation_times_years),
        "autocall_barrier_fracs": list(autocall_barrier_fracs),
        "coupon_barrier_frac": coupon_barrier,
        "coupon_rate": float(terms["coupon_rate"]),
        "knock_in_frac": float(terms["knock_in_frac"]),
        "prior_knock_in_breached": bool(prior_knock_in_breached),
        "memory_coupon": bool(memory_coupon),
        "unpaid_coupon_count": unpaid,
    }


def build_barrier_reverse_convertible_contract(
    *,
    reference_level: float,
    maturity_years: float,
    coupon_times_years: tuple[float, ...],
    coupon_rate_per_period: float,
    strike_frac: float,
    knock_in_frac: float,
    prior_knock_in_breached: bool,
) -> dict[str, Any]:
    reference = float(reference_level)
    strike = float(strike_frac)
    knock_in = float(knock_in_frac)
    if not math.isfinite(reference) or reference <= 0.0:
        raise FrontendInputError("The contractual reference level must be positive.")
    if not 0.0 < knock_in <= strike:
        raise FrontendInputError(
            "The knock-in barrier must be positive and no higher than the strike."
        )
    coupon_rate = float(coupon_rate_per_period)
    if not 0.0 <= coupon_rate <= 1.0:
        raise FrontendInputError(
            "The coupon per payment date must be between 0% and 100%."
        )
    return {
        "contract_version": "barrier-reverse-convertible-v1",
        "reference_level": reference,
        "maturity_years": float(maturity_years),
        "coupon_times_years": list(coupon_times_years),
        "coupon_rate_per_period": coupon_rate,
        "strike_frac": strike,
        "knock_in_frac": knock_in,
        "prior_knock_in_breached": bool(prior_knock_in_breached),
    }


def build_flat_term_structure(
    *,
    symbol: str,
    underlier_type: str,
    currency: str,
    spot: float,
    risk_free_rate: float,
    dividend_yield: float,
    volatility: float,
    maturity_years: float,
    source: str = "streamlit-manual",
    valuation_time: datetime | None = None,
) -> dict[str, Any]:
    timestamp = valuation_time or datetime.now(timezone.utc)
    timestamp_text = timestamp.astimezone(timezone.utc).isoformat()
    return {
        "schema_version": "equity-market-term-structure-v1",
        "symbol": symbol.strip().upper(),
        "underlier_type": underlier_type.strip().lower(),
        "currency": currency.strip().upper(),
        "valuation_time": timestamp_text,
        "market_data_time": timestamp_text,
        "spot": float(spot),
        "segments": [
            {
                "end_time_years": float(maturity_years),
                "risk_free_rate": float(risk_free_rate),
                "dividend_yield": float(dividend_yield),
                "volatility": float(volatility),
            }
        ],
        "calendar": "XNYS",
        "day_count": "ACT/365F",
        "source": source,
    }


def barrier_levels(
    *,
    live_spot: float,
    reference_level: float,
    terms: Mapping[str, Any],
) -> list[dict[str, Any]]:
    reference = float(reference_level)
    return [
        {
            "name": "Knock-in barrier",
            "level": reference * float(terms["knock_in_frac"]),
            "kind": "risk",
        },
        {
            "name": "Live spot",
            "level": float(live_spot),
            "kind": "market",
        },
        {
            "name": "Coupon barrier",
            "level": reference * float(terms["coupon_barrier_frac"]),
            "kind": "coupon",
        },
        {
            "name": "Reference level",
            "level": reference,
            "kind": "reference",
        },
        {
            "name": "Autocall barrier",
            "level": reference * float(terms["autocall_barrier_frac"]),
            "kind": "autocall",
        },
    ]


def diagnostic_grids(market: Mapping[str, Any]) -> dict[str, list[float]]:
    segments = market.get("segments") or []
    volatilities = [
        float(segment["volatility"])
        for segment in segments
        if isinstance(segment, Mapping) and "volatility" in segment
    ]
    minimum_volatility = min(volatilities) if volatilities else 0.2
    downward = -min(0.05, minimum_volatility * 0.5)
    return {
        "spot_shocks_pct": [-20.0, -10.0, 0.0, 10.0, 20.0],
        "volatility_shocks_abs": [downward, 0.0, 0.05],
    }
