from __future__ import annotations

"""
GBM path simulator under risk-neutral dynamics

This module provides a numerically stable, vectorized GBM simulator that:
- Accepts an explicit random seed for reproducibility.
- supports antithetic variates
- uses log-space updates for speed
- returns simulated price paths for the time grid
"""

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
    rng: Optional[np.random.Generator] = None,
    seed: Optional[int] = None,
    antithetic: bool = True,
    dtype: type[np.floating] | np.dtype = np.float64,
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
        rng : numpy.random.Generator, optional
            A NumPy random generator to draw standard normals from
        seed : int, optional
            If rng is None, seed is used to construct a local np.random.Generator(seed)
        antithetic : bool, default True
            If True, returns antithetic pairs [-Z] appended to the base draws
        dtype : np.dtype, default np.float64
            Numeric dtype for simulation arrays

    Returns
        paths: np.ndarray
            Array of shape (n_simulated_paths, num_time_steps + 1) with S values.
        time_grid: np.ndarray
            Array of shape (num_time_steps + 1,) containing the times in years.

    Raises
        ValueError: If any of the input parameters are invalid.
    """

    # Input validation
    if spot <= 0.0:
        raise ValueError("Initial spot must be strictly positive.")
    if time_to_maturity <= 0.0:
        raise ValueError("Time to maturity must be strictly positive.")
    if num_time_steps < 1:
        raise ValueError("Number of time steps must be at least 1.")
    if num_paths < 1:
        raise ValueError("Number of paths must be at least 1.")
    if volatility < 0.0:
        raise ValueError("Volatility must be non-negative.")

    if rng is None:
        rng = np.random.default_rng(seed)
        logger.debug("No RNG provided, using local generator with seed %s", seed)
    else:
        logger.debug("Using provided RNG")

    num_steps = int(num_time_steps)
    T = float(time_to_maturity)
    dt = T / num_steps
    mu_dt = (risk_free_rate - dividend_yield - 0.5 * (volatility**2)) * dt
    sig_sqrt_dt = volatility * np.sqrt(dt, dtype=dtype)

    # Draw base standard normals (shape: num_paths x n_steps)
    Z_base = rng.standard_normal(size=(num_paths, num_steps), dtype=dtype)

    if antithetic:
        Z = np.vstack([Z_base, -Z_base])
        logger.debug("Using antithetic variates, total paths: %d", Z.shape[0])
    else:
        Z = Z_base
        logger.debug("Not using antithetic variates, total paths: %d", Z.shape[0])

    # Vectorized log-increment accumulation - increments: shape (n_simulated_paths, n_steps)
    increments = mu_dt + sig_sqrt_dt * Z
    cum_log_increments = np.concatenate(
        [
            np.zeros((increments.shape[0], 1), dtype=dtype),
            np.cumsum(increments, axis=1),
        ],
        axis=1,
    )

    log_S0 = np.log(spot)
    log_paths = log_S0 + cum_log_increments
    paths = np.exp(log_paths, dtype=dtype)

    # Time grid
    time_grid = np.linspace(0.0, T, num_steps + 1, dtype=dtype)

    logger.debug(
        "simulate_gbm_paths: spot=%.4f, steps=%d, num_paths=%d, dt=%.6f",
        spot,
        num_steps,
        Z.shape[0],
        dt,
    )

    return paths, time_grid


__all__ = ["simulate_gbm_paths"]
