import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from app.services.pricing_service import (
    InvalidPricingInputError,
    normalize_phoenix_market_terms,
    validate_reference_path_count,
)
from app.services.risk_service import (
    InvalidRiskInputError,
    apply_term_structure_shock,
)
from src.final.data_generator import (
    build_simulation_time_grid,
    simulate_piecewise_gbm_paths,
)
from src.final.market import (
    EQUITY_GBM_PIECEWISE_MODEL_VERSION,
    EquityMarketTermStructure,
    MarketDataValidationError,
)
from src.final.payoffs import PhoenixPayoff
from src.final.phoenix_contract import (
    PHOENIX_SINGLE_V2_CONTRACT_VERSION,
    PhoenixSingleV2Contract,
)
from src.final.reference_pricer import DEFAULT_REFERENCE_SEED, DEFAULT_REFERENCE_STEPS


PHOENIX_DIAGNOSTICS_VERSION = "phoenix-reference-diagnostics-v1"
MAX_DIAGNOSTIC_PATHS = 5_000
MAX_SURFACE_AXIS_POINTS = 11
MAX_SURFACE_PATH_EVALUATIONS = 200_000
DEFAULT_SPOT_SHOCKS_PCT = (-20.0, -10.0, 0.0, 10.0, 20.0)
COMPONENT_NAMES = (
    "coupon_pv",
    "autocall_principal_pv",
    "maturity_protected_pv",
    "maturity_downside_pv",
)


class DiagnosticsServiceError(Exception):
    pass


class InvalidDiagnosticsInputError(DiagnosticsServiceError):
    pass


@dataclass(frozen=True)
class _DiagnosticCase:
    contract_version: str
    contract_payload: Mapping[str, Any]
    params: Mapping[str, Any]
    time_grid: np.ndarray
    observation_times: tuple[float, ...]
    prior_knock_in_breached: bool
    explicit_schedule: bool


def _finite_sequence(
    values: Sequence[Any] | None,
    *,
    name: str,
    default: Sequence[float],
    minimum: float,
    maximum: float,
    max_length: int,
) -> tuple[float, ...]:
    supplied = default if values is None or len(values) == 0 else values
    if len(supplied) > max_length:
        raise InvalidDiagnosticsInputError(
            f"{name} must contain at most {max_length} values"
        )
    normalized: list[float] = []
    for raw_value in supplied:
        if isinstance(raw_value, bool):
            raise InvalidDiagnosticsInputError(f"{name} must be numeric")
        try:
            value = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise InvalidDiagnosticsInputError(f"{name} must be numeric") from exc
        if not math.isfinite(value) or value < minimum or value > maximum:
            raise InvalidDiagnosticsInputError(
                f"{name} values must be between {minimum} and {maximum}"
            )
        normalized.append(value)
    if len(set(normalized)) != len(normalized):
        raise InvalidDiagnosticsInputError(f"{name} must not contain duplicates")
    return tuple(sorted(normalized))


def _validated_path_count(value: Any) -> int:
    try:
        normalized = validate_reference_path_count(value)
    except InvalidPricingInputError as exc:
        raise InvalidDiagnosticsInputError(str(exc)) from exc
    if normalized < 100 or normalized > MAX_DIAGNOSTIC_PATHS:
        raise InvalidDiagnosticsInputError(
            f"diagnostic path count must be between 100 and {MAX_DIAGNOSTIC_PATHS}"
        )
    return normalized


def _convergence_counts(
    values: Sequence[Any] | None,
    *,
    n_paths: int,
) -> tuple[int, ...]:
    if values is None or len(values) == 0:
        candidates = (100, 250, 500, 1_000, 2_000, n_paths)
    else:
        candidates = tuple(values)
    if len(candidates) > 8:
        raise InvalidDiagnosticsInputError(
            "convergence_path_counts must contain at most 8 values"
        )
    normalized: list[int] = []
    for raw_value in candidates:
        if isinstance(raw_value, bool):
            raise InvalidDiagnosticsInputError(
                "convergence path counts must be integers"
            )
        try:
            value = int(raw_value)
        except (TypeError, ValueError) as exc:
            raise InvalidDiagnosticsInputError(
                "convergence path counts must be integers"
            ) from exc
        if value < 2 or value > n_paths:
            if values is not None and len(values) > 0:
                raise InvalidDiagnosticsInputError(
                    "convergence path counts must satisfy 2 <= count <= n_paths"
                )
            continue
        normalized.append(value)
    normalized.append(n_paths)
    return tuple(sorted(set(normalized)))


def _v1_case(
    market: EquityMarketTermStructure,
    terms: Mapping[str, Any],
) -> _DiagnosticCase:
    try:
        params, normalized_terms = normalize_phoenix_market_terms(
            spot=market.spot,
            risk_free_rate=0.0,
            volatility=1.0,
            terms=dict(terms),
        )
        equivalent = market.equivalent_flat_parameters(params["T"])
    except (
        InvalidPricingInputError,
        MarketDataValidationError,
        TypeError,
        ValueError,
    ) as exc:
        raise InvalidDiagnosticsInputError(str(exc)) from exc
    params["r"] = equivalent["risk_free_rate"]
    params["sigma"] = equivalent["volatility"]
    time_grid = np.linspace(
        0.0,
        float(params["T"]),
        DEFAULT_REFERENCE_STEPS + 1,
        dtype=np.float64,
    )
    observation_indices = np.linspace(
        0,
        DEFAULT_REFERENCE_STEPS,
        int(params["obs_count"]) + 1,
        dtype=int,
    )[1:]
    observation_times = tuple(float(time_grid[index]) for index in observation_indices)
    return _DiagnosticCase(
        contract_version=PhoenixPayoff.contract_version,
        contract_payload={
            "contract_version": PhoenixPayoff.contract_version,
            "reference_level": market.spot,
            **normalized_terms,
        },
        params=params,
        time_grid=time_grid,
        observation_times=observation_times,
        prior_knock_in_breached=False,
        explicit_schedule=False,
    )


def _v2_case(
    market: EquityMarketTermStructure,
    contract: PhoenixSingleV2Contract,
) -> _DiagnosticCase:
    try:
        equivalent = market.equivalent_flat_parameters(contract.maturity_years)
        time_grid = build_simulation_time_grid(
            contract.maturity_years,
            DEFAULT_REFERENCE_STEPS,
            contract.observation_times_years,
        )
    except MarketDataValidationError as exc:
        raise InvalidDiagnosticsInputError(str(exc)) from exc
    return _DiagnosticCase(
        contract_version=PHOENIX_SINGLE_V2_CONTRACT_VERSION,
        contract_payload=contract.to_dict(),
        params=contract.to_payoff_params(
            risk_free_rate=equivalent["risk_free_rate"],
            volatility=equivalent["volatility"],
        ),
        time_grid=time_grid,
        observation_times=contract.observation_times_years,
        prior_knock_in_breached=contract.prior_knock_in_breached,
        explicit_schedule=True,
    )


def _evaluate_case(
    *,
    case: _DiagnosticCase,
    market: EquityMarketTermStructure,
    shocks: np.ndarray,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    payoff = PhoenixPayoff()
    n_paths, effective_steps = shocks.shape
    try:
        paths = simulate_piecewise_gbm_paths(
            market=market,
            T=float(case.params["T"]),
            n_steps=effective_steps,
            n_paths=n_paths,
            seed=None,
            standard_normal_shocks=shocks,
            time_grid_years=case.time_grid,
        )
        if case.explicit_schedule:
            ledger = payoff.compute_observation_event_ledger_with_explicit_schedule(
                paths=paths,
                params=dict(case.params),
                path_times_years=case.time_grid,
                observation_times_years=case.observation_times,
                prior_knock_in_breached=case.prior_knock_in_breached,
                discount_factor=market.discount_factor,
            )
        else:
            ledger = payoff.compute_observation_event_ledger_with_discount_curve(
                paths=paths,
                params=dict(case.params),
                T=float(case.params["T"]),
                discount_factor=market.discount_factor,
            )
    except (MarketDataValidationError, ValueError) as exc:
        raise InvalidDiagnosticsInputError(str(exc)) from exc
    discounted_payoffs = sum(ledger[name] for name in COMPONENT_NAMES)
    return discounted_payoffs, ledger


def _sample_statistics(values: np.ndarray) -> dict[str, float]:
    count = len(values)
    mean = float(np.mean(values))
    sample_std = float(np.std(values, ddof=1)) if count > 1 else 0.0
    standard_error = sample_std / math.sqrt(count)
    half_width = 1.96 * standard_error
    return {
        "price": mean,
        "payoff_std": sample_std,
        "standard_error": standard_error,
        "confidence_interval_low": mean - half_width,
        "confidence_interval_high": mean + half_width,
    }


def _convergence_report(
    discounted_payoffs: np.ndarray,
    counts: Sequence[int],
) -> list[dict[str, Any]]:
    return [
        {
            "n_paths": count,
            **_sample_statistics(discounted_payoffs[:count]),
        }
        for count in counts
    ]


def _cashflow_report(ledger: Mapping[str, np.ndarray]) -> dict[str, Any]:
    component_rows = []
    for name in COMPONENT_NAMES:
        values = np.asarray(ledger[name], dtype=np.float64)
        statistics = _sample_statistics(values)
        component_rows.append(
            {
                "component": name,
                "expected_pv": statistics["price"],
                "standard_error": statistics["standard_error"],
            }
        )
    coupon_events = np.asarray(ledger["coupon_event"], dtype=np.float64)
    return {
        "components": component_rows,
        "autocall_probability": float(
            np.mean(np.asarray(ledger["autocall_probability"], dtype=np.float64))
        ),
        "downside_probability": float(
            np.mean(np.asarray(ledger["downside_probability"], dtype=np.float64))
        ),
        "expected_coupon_count": float(np.mean(np.sum(coupon_events, axis=1))),
    }


def _distribution_report(discounted_payoffs: np.ndarray) -> dict[str, Any]:
    quantile_levels = (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)
    quantile_values = np.quantile(discounted_payoffs, quantile_levels)
    counts, edges = np.histogram(discounted_payoffs, bins=24)
    return {
        "quantiles": [
            {"probability": probability, "value": float(value)}
            for probability, value in zip(quantile_levels, quantile_values)
        ],
        "histogram": {
            "bin_edges": [float(value) for value in edges],
            "counts": [int(value) for value in counts],
        },
    }


def _surface_report(
    *,
    case: _DiagnosticCase,
    market: EquityMarketTermStructure,
    shocks: np.ndarray,
    base_price: float,
    spot_shocks_pct: Sequence[float],
    volatility_shocks_abs: Sequence[float],
) -> dict[str, Any]:
    cells: list[dict[str, Any]] = []
    for volatility_shock in volatility_shocks_abs:
        for spot_shock in spot_shocks_pct:
            if spot_shock == 0.0 and volatility_shock == 0.0:
                shocked_market = market
            else:
                try:
                    shocked_market, _ = apply_term_structure_shock(
                        market,
                        {
                            "spot_pct": spot_shock,
                            "volatility_parallel_abs": volatility_shock,
                        },
                    )
                except InvalidRiskInputError as exc:
                    raise InvalidDiagnosticsInputError(str(exc)) from exc
            payoffs, _ = _evaluate_case(
                case=case,
                market=shocked_market,
                shocks=shocks,
            )
            statistics = _sample_statistics(payoffs)
            cells.append(
                {
                    "spot_shock_pct": spot_shock,
                    "volatility_shock_abs": volatility_shock,
                    "spot": shocked_market.spot,
                    "price": statistics["price"],
                    "price_change": statistics["price"] - base_price,
                    "standard_error": statistics["standard_error"],
                }
            )
    return {
        "spot_shocks_pct": list(spot_shocks_pct),
        "volatility_shocks_abs": list(volatility_shocks_abs),
        "cells": cells,
    }


def _fingerprint(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _build_diagnostics(
    *,
    market: EquityMarketTermStructure,
    case: _DiagnosticCase,
    n_paths: Any,
    seed: Any,
    convergence_path_counts: Sequence[Any] | None,
    spot_shocks_pct: Sequence[Any] | None,
    volatility_shocks_abs: Sequence[Any] | None,
) -> dict[str, Any]:
    if not isinstance(market, EquityMarketTermStructure):
        raise InvalidDiagnosticsInputError("invalid equity market term structure")
    validated_paths = _validated_path_count(n_paths)
    if isinstance(seed, bool):
        raise InvalidDiagnosticsInputError("seed must be an integer")
    try:
        validated_seed = int(seed)
    except (TypeError, ValueError) as exc:
        raise InvalidDiagnosticsInputError("seed must be an integer") from exc
    if validated_seed < 0 or validated_seed > 4_294_967_295:
        raise InvalidDiagnosticsInputError("seed must be between 0 and 4294967295")
    counts = _convergence_counts(
        convergence_path_counts,
        n_paths=validated_paths,
    )
    spot_grid = _finite_sequence(
        spot_shocks_pct,
        name="spot_shocks_pct",
        default=DEFAULT_SPOT_SHOCKS_PCT,
        minimum=-90.0,
        maximum=100.0,
        max_length=MAX_SURFACE_AXIS_POINTS,
    )
    minimum_market_volatility = min(segment.volatility for segment in market.segments)
    default_volatility_grid = (
        -min(0.05, minimum_market_volatility * 0.5),
        0.0,
        0.05,
    )
    volatility_grid = _finite_sequence(
        volatility_shocks_abs,
        name="volatility_shocks_abs",
        default=default_volatility_grid,
        minimum=-1.0,
        maximum=1.0,
        max_length=MAX_SURFACE_AXIS_POINTS,
    )
    surface_work = validated_paths * len(spot_grid) * len(volatility_grid)
    if surface_work > MAX_SURFACE_PATH_EVALUATIONS:
        raise InvalidDiagnosticsInputError(
            "surface grid and path count exceed the diagnostic work limit"
        )

    effective_steps = len(case.time_grid) - 1
    random_state = np.random.RandomState(validated_seed)
    shocks = random_state.randn(validated_paths, effective_steps)
    discounted_payoffs, ledger = _evaluate_case(
        case=case,
        market=market,
        shocks=shocks,
    )
    base_statistics = _sample_statistics(discounted_payoffs)
    surface = _surface_report(
        case=case,
        market=market,
        shocks=shocks,
        base_price=base_statistics["price"],
        spot_shocks_pct=spot_grid,
        volatility_shocks_abs=volatility_grid,
    )
    identity_payload = {
        "diagnostic_version": PHOENIX_DIAGNOSTICS_VERSION,
        "market_term_structure_id": market.term_structure_id,
        "contract": dict(case.contract_payload),
        "n_paths": validated_paths,
        "seed": validated_seed,
        "convergence_path_counts": list(counts),
        "spot_shocks_pct": list(spot_grid),
        "volatility_shocks_abs": list(volatility_grid),
    }
    return {
        "diagnostic_version": PHOENIX_DIAGNOSTICS_VERSION,
        "diagnostic_id": _fingerprint(identity_payload),
        "contract_version": case.contract_version,
        "model_version": EQUITY_GBM_PIECEWISE_MODEL_VERSION,
        "base": base_statistics,
        "convergence": _convergence_report(discounted_payoffs, counts),
        "cashflows": _cashflow_report(ledger),
        "distribution": _distribution_report(discounted_payoffs),
        "surface": surface,
        "provenance": {
            "market_term_structure_id": market.term_structure_id,
            "contract": dict(case.contract_payload),
            "n_paths": validated_paths,
            "base_monitoring_steps": DEFAULT_REFERENCE_STEPS,
            "effective_simulation_steps": effective_steps,
            "seed": validated_seed,
            "common_random_numbers": True,
            "raw_paths_returned": False,
            "surface_path_evaluations": surface_work,
        },
    }


def get_phoenix_v1_diagnostics(
    *,
    market: EquityMarketTermStructure,
    terms: Mapping[str, Any],
    n_paths: Any = 2_000,
    seed: Any = DEFAULT_REFERENCE_SEED,
    convergence_path_counts: Sequence[Any] | None = None,
    spot_shocks_pct: Sequence[Any] | None = None,
    volatility_shocks_abs: Sequence[Any] | None = None,
) -> dict[str, Any]:
    return _build_diagnostics(
        market=market,
        case=_v1_case(market, terms),
        n_paths=n_paths,
        seed=seed,
        convergence_path_counts=convergence_path_counts,
        spot_shocks_pct=spot_shocks_pct,
        volatility_shocks_abs=volatility_shocks_abs,
    )


def get_phoenix_v2_diagnostics(
    *,
    market: EquityMarketTermStructure,
    contract: PhoenixSingleV2Contract,
    n_paths: Any = 2_000,
    seed: Any = DEFAULT_REFERENCE_SEED,
    convergence_path_counts: Sequence[Any] | None = None,
    spot_shocks_pct: Sequence[Any] | None = None,
    volatility_shocks_abs: Sequence[Any] | None = None,
) -> dict[str, Any]:
    if not isinstance(contract, PhoenixSingleV2Contract):
        raise InvalidDiagnosticsInputError("invalid Phoenix Single v2 contract")
    return _build_diagnostics(
        market=market,
        case=_v2_case(market, contract),
        n_paths=n_paths,
        seed=seed,
        convergence_path_counts=convergence_path_counts,
        spot_shocks_pct=spot_shocks_pct,
        volatility_shocks_abs=volatility_shocks_abs,
    )
