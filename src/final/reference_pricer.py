import time
from typing import Any, Dict, Optional

import numpy as np

from .data_generator import (
    build_simulation_time_grid,
    simulate_gbm_paths,
    simulate_piecewise_gbm_paths,
)
from .market import EquityMarketTermStructure
from .payoffs import BasePayoff, PhoenixPayoff
from .phoenix_contract import PhoenixSingleV2Contract


DEFAULT_REFERENCE_SEED = 42
DEFAULT_REFERENCE_STEPS = 252


def _summarize_discounted_payoffs(
    discounted_payoffs: np.ndarray,
    *,
    n_paths: int,
    n_steps: int,
    seed: Optional[int],
    elapsed: float,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build one stable Monte Carlo result shape for every reference model."""
    price = float(np.mean(discounted_payoffs))
    payoff_std = float(np.std(discounted_payoffs, ddof=1)) if n_paths > 1 else 0.0
    standard_error = payoff_std / float(np.sqrt(n_paths))
    ci_half_width = 1.96 * standard_error
    return {
        "price": price,
        "payoff_std": payoff_std,
        "standard_error": standard_error,
        "confidence_level": 0.95,
        "confidence_interval": [price - ci_half_width, price + ci_half_width],
        "n_paths": n_paths,
        "n_steps": n_steps,
        "seed": seed,
        **(metadata or {}),
        "time_s": elapsed,
    }


def price_reference(
    payoff: BasePayoff,
    params: Dict[str, Any],
    n_paths: int,
    n_steps: int = DEFAULT_REFERENCE_STEPS,
    seed: Optional[int] = DEFAULT_REFERENCE_SEED,
    dividend_yield: float = 0.0,
) -> Dict[str, Any]:
    """Price one contract with deterministic Monte Carlo and uncertainty data."""
    started = time.perf_counter()
    paths = simulate_gbm_paths(
        s0=float(params["S0"]),
        r=float(params["r"]),
        sigma=float(params["sigma"]),
        T=float(params["T"]),
        n_steps=n_steps,
        n_paths=n_paths,
        seed=seed,
        dividend_yield=dividend_yield,
    )
    discounted_payoffs = payoff.compute_payoff(
        paths,
        params,
        float(params["r"]),
        float(params["T"]),
    )
    elapsed = time.perf_counter() - started

    return _summarize_discounted_payoffs(
        discounted_payoffs,
        n_paths=n_paths,
        n_steps=n_steps,
        seed=seed,
        elapsed=elapsed,
        metadata={"dividend_yield": dividend_yield},
    )


def price_phoenix_piecewise_reference(
    payoff: PhoenixPayoff,
    params: Dict[str, Any],
    market: EquityMarketTermStructure,
    n_paths: int,
    n_steps: int = DEFAULT_REFERENCE_STEPS,
    seed: Optional[int] = DEFAULT_REFERENCE_SEED,
) -> Dict[str, Any]:
    """Price Phoenix under deterministic piecewise carry and volatility."""
    started = time.perf_counter()
    maturity = float(params["T"])
    paths = simulate_piecewise_gbm_paths(
        market=market,
        T=maturity,
        n_steps=n_steps,
        n_paths=n_paths,
        seed=seed,
    )
    discounted_payoffs = payoff.compute_payoff_with_discount_curve(
        paths=paths,
        params=params,
        T=maturity,
        discount_factor=market.discount_factor,
    )
    elapsed = time.perf_counter() - started

    return _summarize_discounted_payoffs(
        discounted_payoffs,
        n_paths=n_paths,
        n_steps=n_steps,
        seed=seed,
        elapsed=elapsed,
        metadata={"term_structure_id": market.term_structure_id},
    )


def price_phoenix_v2_piecewise_reference(
    payoff: PhoenixPayoff,
    contract: PhoenixSingleV2Contract,
    market: EquityMarketTermStructure,
    n_paths: int,
    n_steps: int = DEFAULT_REFERENCE_STEPS,
    seed: Optional[int] = DEFAULT_REFERENCE_SEED,
) -> Dict[str, Any]:
    """Price active Phoenix state on a grid containing every event time."""
    started = time.perf_counter()
    equivalent = market.equivalent_flat_parameters(contract.maturity_years)
    params = contract.to_payoff_params(
        risk_free_rate=equivalent["risk_free_rate"],
        volatility=equivalent["volatility"],
    )
    time_grid = build_simulation_time_grid(
        contract.maturity_years,
        n_steps,
        contract.observation_times_years,
    )
    effective_steps = len(time_grid) - 1
    paths = simulate_piecewise_gbm_paths(
        market=market,
        T=contract.maturity_years,
        n_steps=effective_steps,
        n_paths=n_paths,
        seed=seed,
        time_grid_years=time_grid,
    )
    discounted_payoffs = (
        payoff.compute_payoff_with_explicit_schedule_and_discount_curve(
            paths=paths,
            params=params,
            path_times_years=time_grid,
            observation_times_years=contract.observation_times_years,
            prior_knock_in_breached=contract.prior_knock_in_breached,
            discount_factor=market.discount_factor,
        )
    )
    elapsed = time.perf_counter() - started

    return _summarize_discounted_payoffs(
        discounted_payoffs,
        n_paths=n_paths,
        n_steps=effective_steps,
        seed=seed,
        elapsed=elapsed,
        metadata={
            "term_structure_id": market.term_structure_id,
            "contract_id": contract.contract_id,
            "base_monitoring_steps": n_steps,
            "contract_event_times_inserted": effective_steps - n_steps,
        },
    )


def phoenix_piecewise_discounted_payoffs(
    payoff: PhoenixPayoff,
    params: Dict[str, Any],
    market: EquityMarketTermStructure,
    n_paths: int,
    n_steps: int = DEFAULT_REFERENCE_STEPS,
    seed: Optional[int] = DEFAULT_REFERENCE_SEED,
    standard_normal_shocks: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Return pathwise Phoenix PVs for paired scenario and risk estimates.

    When ``standard_normal_shocks`` is supplied, callers can value several
    markets with exactly the same random draws.  ``params['S0']`` remains the
    frozen contractual reference level while ``market.spot`` is the simulated
    market spot.
    """
    maturity = float(params["T"])
    paths = simulate_piecewise_gbm_paths(
        market=market,
        T=maturity,
        n_steps=n_steps,
        n_paths=n_paths,
        seed=seed,
        standard_normal_shocks=standard_normal_shocks,
    )
    return payoff.compute_payoff_with_discount_curve(
        paths=paths,
        params=params,
        T=maturity,
        discount_factor=market.discount_factor,
    )


def phoenix_piecewise_discounted_components(
    payoff: PhoenixPayoff,
    params: Dict[str, Any],
    market: EquityMarketTermStructure,
    n_paths: int,
    n_steps: int = DEFAULT_REFERENCE_STEPS,
    seed: Optional[int] = DEFAULT_REFERENCE_SEED,
    standard_normal_shocks: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """Return pathwise Phoenix cashflow components and event indicators."""
    maturity = float(params["T"])
    paths = simulate_piecewise_gbm_paths(
        market=market,
        T=maturity,
        n_steps=n_steps,
        n_paths=n_paths,
        seed=seed,
        standard_normal_shocks=standard_normal_shocks,
    )
    return payoff.compute_cashflow_components_with_discount_curve(
        paths=paths,
        params=params,
        T=maturity,
        discount_factor=market.discount_factor,
    )


def phoenix_piecewise_observation_event_ledger(
    payoff: PhoenixPayoff,
    params: Dict[str, Any],
    market: EquityMarketTermStructure,
    n_paths: int,
    n_steps: int = DEFAULT_REFERENCE_STEPS,
    seed: Optional[int] = DEFAULT_REFERENCE_SEED,
    standard_normal_shocks: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """Return pathwise Phoenix events at every contractual observation."""
    maturity = float(params["T"])
    paths = simulate_piecewise_gbm_paths(
        market=market,
        T=maturity,
        n_steps=n_steps,
        n_paths=n_paths,
        seed=seed,
        standard_normal_shocks=standard_normal_shocks,
    )
    return payoff.compute_observation_event_ledger_with_discount_curve(
        paths=paths,
        params=params,
        T=maturity,
        discount_factor=market.discount_factor,
    )
