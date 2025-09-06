from __future__ import annotations

"""
GBM path simulator under risk-neutral dynamics

This module provides a numerically stable, vectorized GBM simulator that:
- Accepts an explicit random seed for reproducibility.
- supports antithetic variates
- uses log-space updates for speed
- returns simulated price paths for the time grid
"""

import math
import logging
import numpy as np
from typing import Tuple, Optional

logger = logging.getLogger("src.gbm")


def simulate_gbm_paths(
    spot: float,
    risk_free_rate: float,
    dividend_yield: float,
    volatility: float,
    time_to_maturity: float,
    num_time_steps: int,
    num_paths: int,
    seed: Optional[int] = None,
    antithetic: bool = True,
    batch_size: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate GBM asset price paths under the risk neutral measure

    The discrete-log update used (vectorized) is:
        increments = (r - q - 0.5 sigma^2)dt + sigma * sqrt(dt) * Z
        log_paths = log(S0) + cumsum(increments, axis = 1) with a prepended zero column
        path = exp(log_paths)

    Parameters
        spot: float
            Initial price S0
        risk_free_rate: float
            Continuously compounded risk-free rate r
        dividend_yield: float
            Continuous dividend yield q
        volatility: float
            Annualized volatility sigma
        time_to_maturity: float
            Total horizon in years T
        num_time_steps: int
            Number of time steps to discretize [0, T] - The simulator will return num_time_steps + 1 times including t=0 and t=T
        num_paths: int
            Number of Monte Carlo paths to generate (pre-antithetic). If `antithetic=True`, the effective number of simulated paths returned is num_paths * 2
        seed : int, optional
            If rng is None, seed is used to construct a local np.random.Generator(seed)
        antithetic : bool, default True
            If True, returns antithetic pairs [-Z] appended to the base draws
        batch_size : int or None, optional
            If set, simulation is done in batches of this size to reduce memory usage.

    Returns
        paths: np.ndarray
            Array of shape (n_simulated_paths, num_time_steps + 1) with S values.
        time_grid: np.ndarray
            Array of shape (num_time_steps + 1,) containing the times in years.

    Raises
        ValueError: If any of the input parameters are invalid.
    """

    # Input validation
    if spot <= 0:
        raise ValueError("spot must be positive.")
    if volatility < 0:
        raise ValueError("volatility must be non-negative.")
    if time_to_maturity <= 0:
        raise ValueError("time_to_maturity must be positive.")
    if num_time_steps <= 0:
        raise ValueError("num_time_steps must be positive.")
    if num_paths <= 0:
        raise ValueError("num_paths must be positive.")
    if batch_size is not None and batch_size <= 0:
        raise ValueError("batch_size must be positive or None.")

    rng = np.random.default_rng(seed)
    num_steps = int(num_time_steps)
    T = float(time_to_maturity)
    dt = T / num_steps
    mu_dt = (risk_free_rate - dividend_yield - 0.5 * (volatility**2)) * dt
    sig_sqrt_dt = volatility * math.sqrt(dt)

    # Effective number of paths after applying antithetic
    n_effective = num_paths * (2 if antithetic else 1)

    # Pre-allocate full output
    paths = np.empty((n_effective, num_steps + 1), dtype=np.float64)
    paths[:, 0] = spot

    # Time grid
    time_grid = np.linspace(0.0, T, num_steps + 1, dtype=np.float64)

    # Batch size setup
    if batch_size is None:
        batch_size = num_paths

    out_idx = 0
    for batch_start in range(0, num_paths, batch_size):
        batch_n = min(batch_size, num_paths - batch_start)
        Z = rng.standard_normal(size=(batch_n, num_steps), dtype=np.float64)

        if antithetic:
            Z = np.concatenate([Z, -Z], axis=0)

        log_S = np.full((Z.shape[0],), math.log(spot), dtype=np.float64)
        paths_batch = np.empty((Z.shape[0], num_steps + 1), dtype=np.float64)
        paths_batch[:, 0] = spot

        for t in range(num_steps):
            log_S += mu_dt + sig_sqrt_dt * Z[:, t]
            paths_batch[:, t + 1] = np.exp(log_S)

        # Copy into the full array
        paths[out_idx : out_idx + Z.shape[0], :] = paths_batch
        out_idx += Z.shape[0]

    return paths, time_grid


__all__ = ["simulate_gbm_paths"]
