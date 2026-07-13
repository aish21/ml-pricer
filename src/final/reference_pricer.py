import time
from typing import Any, Dict, Optional

import numpy as np

from .data_generator import simulate_gbm_paths
from .payoffs import BasePayoff


DEFAULT_REFERENCE_SEED = 42
DEFAULT_REFERENCE_STEPS = 252


def price_reference(
    payoff: BasePayoff,
    params: Dict[str, Any],
    n_paths: int,
    n_steps: int = DEFAULT_REFERENCE_STEPS,
    seed: Optional[int] = DEFAULT_REFERENCE_SEED,
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
    )
    discounted_payoffs = payoff.compute_payoff(
        paths,
        params,
        float(params["r"]),
        float(params["T"]),
    )
    elapsed = time.perf_counter() - started

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
        "time_s": elapsed,
    }
