import time
import math
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np

from .market import EquityMarketTermStructure, MarketDataValidationError
from .payoffs import BasePayoff


def simulate_gbm_paths(
    s0: float,
    r: float,
    sigma: float,
    T: float,
    n_steps: int,
    n_paths: int,
    seed: Optional[int] = None,
    dividend_yield: float = 0.0,
) -> np.ndarray:
    """Simulate log-GBM paths with a continuous flat dividend yield."""
    if seed is not None:
        rng = np.random.RandomState(seed)
        Z = rng.randn(n_paths, n_steps)
    else:
        Z = np.random.randn(n_paths, n_steps)

    dt = T / n_steps
    drift = (r - dividend_yield - 0.5 * sigma**2) * dt
    diffusion = sigma * math.sqrt(dt)
    increments = drift + diffusion * Z

    log_paths = np.zeros((n_paths, n_steps + 1), dtype=np.float64)
    log_paths[:, 0] = np.log(s0)
    log_paths[:, 1:] = np.log(s0) + np.cumsum(increments, axis=1)
    paths = np.exp(log_paths)
    return paths


def simulate_piecewise_gbm_paths(
    market: EquityMarketTermStructure,
    T: float,
    n_steps: int,
    n_paths: int,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Simulate GBM using exactly integrated piecewise carry and variance."""
    if not isinstance(market, EquityMarketTermStructure):
        raise MarketDataValidationError("invalid equity market term structure")
    if isinstance(T, bool):
        raise MarketDataValidationError("T must be numeric")
    try:
        maturity = float(T)
    except (TypeError, ValueError) as exc:
        raise MarketDataValidationError("T must be numeric") from exc
    if not math.isfinite(maturity) or maturity <= 0:
        raise MarketDataValidationError("T must be finite and > 0")
    if maturity > market.max_time_years + 1e-12:
        raise MarketDataValidationError(
            "term structure does not cover the product maturity"
        )
    if not isinstance(n_steps, int) or isinstance(n_steps, bool) or n_steps < 1:
        raise MarketDataValidationError("n_steps must be a positive integer")
    if not isinstance(n_paths, int) or isinstance(n_paths, bool) or n_paths < 1:
        raise MarketDataValidationError("n_paths must be a positive integer")

    if seed is not None:
        rng = np.random.RandomState(seed)
        shocks = rng.randn(n_paths, n_steps)
    else:
        shocks = np.random.randn(n_paths, n_steps)

    times = np.linspace(0.0, maturity, n_steps + 1)
    integrated_rates = np.array(
        [
            market.integrated_risk_free_rate(start, end)
            for start, end in zip(times[:-1], times[1:])
        ]
    )
    integrated_dividends = np.array(
        [
            market.integrated_dividend_yield(start, end)
            for start, end in zip(times[:-1], times[1:])
        ]
    )
    integrated_variances = np.array(
        [
            market.integrated_variance(start, end)
            for start, end in zip(times[:-1], times[1:])
        ]
    )
    drift = integrated_rates - integrated_dividends - 0.5 * integrated_variances
    increments = drift + np.sqrt(integrated_variances) * shocks

    log_paths = np.zeros((n_paths, n_steps + 1), dtype=np.float64)
    log_paths[:, 0] = np.log(market.spot)
    log_paths[:, 1:] = np.log(market.spot) + np.cumsum(increments, axis=1)
    return np.exp(log_paths)


def sample_parameters(
    n_samples: int, payoff: BasePayoff, seed: Optional[int] = None
) -> list:
    """Sample random parameter sets."""
    rng = np.random.RandomState(seed)
    samples = []

    for _ in range(n_samples):
        s = {}
        for param_name, (lo, hi) in payoff.param_ranges.items():
            if param_name in ["obs_count"]:
                s[param_name] = int(rng.randint(int(lo), int(hi) + 1))
            else:
                s[param_name] = float(rng.uniform(lo, hi))
        samples.append(s)

    return samples


class DataGenerator:
    """Generate training data using Monte Carlo simulation."""

    def __init__(self, payoff: BasePayoff, n_steps: int = 252, verbose: bool = True):
        self.payoff = payoff
        self.n_steps = n_steps
        self.verbose = verbose

    def generate(
        self, n_samples: int, n_paths_per_sample: int, seed: Optional[int] = None
    ) -> tuple:
        """Generate training data."""
        start_time = time.time()
        if self.verbose:
            print(
                f"[DataGenerator] Generating {n_samples} samples with "
                f"{n_paths_per_sample} paths each..."
            )

        params_list = sample_parameters(n_samples, self.payoff, seed=seed)

        feature_order = self.payoff.get_feature_order()
        n_features = len(feature_order)
        X = np.zeros((n_samples, n_features), dtype=np.float64)
        y = np.zeros(n_samples, dtype=np.float64)

        for i, params in enumerate(params_list):
            if self.verbose and (i + 1) % 1000 == 0:
                print(f"  Generated {i+1}/{n_samples} samples")

            S0 = params["S0"]
            r = params["r"]
            sigma = params["sigma"]
            T = params["T"]

            path_seed = None if seed is None else seed + i
            paths = simulate_gbm_paths(
                S0, r, sigma, T, self.n_steps, n_paths_per_sample, seed=path_seed
            )

            payoffs = self.payoff.compute_payoff(paths, params, r, T)

            price = float(np.mean(payoffs))

            X[i, :] = [params[f] for f in feature_order]
            y[i] = price

        elapsed = time.time() - start_time
        if self.verbose:
            print(f"[DataGenerator] Completed in {elapsed:.1f}s")

        return X, y

    def save(self, X: np.ndarray, y: np.ndarray, output_path: Path):
        """Save training data to file."""
        import json

        meta = {
            "n_samples": len(y),
            "n_features": X.shape[1],
            "feature_order": self.payoff.get_feature_order(),
            "generated_at": time.time(),
            "payoff_type": self.payoff.__class__.__name__,
            "contract_version": self.payoff.contract_version,
        }

        np.savez_compressed(output_path, X=X, y=y, meta=meta)

        if self.verbose:
            print(f"[DataGenerator] Saved data to {output_path}")

    @staticmethod
    def load(input_path: Path) -> tuple:
        """Load training data from file."""
        data = np.load(input_path, allow_pickle=True)
        X = data["X"]
        y = data["y"]
        meta = dict(data["meta"].item()) if "meta" in data else {}
        return X, y, meta
