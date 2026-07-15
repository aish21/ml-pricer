import math
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Tuple
import numpy as np


class BasePayoff(ABC):
    """Base class for payoff functions."""

    contract_version = "unversioned"

    def __init__(self, param_ranges: Dict[str, Tuple[float, float]]):
        self.param_ranges = param_ranges

    @abstractmethod
    def get_parameter_names(self) -> List[str]:
        """Return list of parameter names."""
        pass

    @abstractmethod
    def compute_payoff(
        self, paths: np.ndarray, params: Dict[str, Any], r: float, T: float
    ) -> np.ndarray:
        """Compute payoff for each path."""
        pass

    @abstractmethod
    def get_feature_order(self) -> List[str]:
        """Return ordered list of feature names."""
        pass


class PhoenixPayoff(BasePayoff):
    """Single-underlier Phoenix with non-memory periodic coupons."""

    contract_version = "phoenix-single-v1"

    def __init__(self):
        param_ranges = {
            "S0": (80.0, 120.0),
            "r": (0.0, 0.05),
            "sigma": (0.05, 0.45),
            "T": (0.5, 2.5),
            "autocall_barrier_frac": (0.95, 1.15),
            "coupon_barrier_frac": (0.7, 1.05),
            "coupon_rate": (0.005, 0.05),
            "knock_in_frac": (0.5, 0.95),
            "obs_count": (4, 12),
        }
        super().__init__(param_ranges)

    def get_parameter_names(self) -> List[str]:
        return [
            "S0",
            "r",
            "sigma",
            "T",
            "autocall_barrier_frac",
            "coupon_barrier_frac",
            "coupon_rate",
            "knock_in_frac",
            "obs_count",
        ]

    def get_feature_order(self) -> List[str]:
        return [
            "S0",
            "r",
            "sigma",
            "T",
            "autocall_barrier_frac",
            "coupon_barrier_frac",
            "coupon_rate",
            "knock_in_frac",
            "obs_count",
        ]

    def compute_payoff(
        self, paths: np.ndarray, params: Dict[str, Any], r: float, T: float
    ) -> np.ndarray:
        """Return discounted cashflows per unit notional for every path.

        Coupons are non-memory and paid on each observation where the underlier
        is at or above the coupon barrier.  An autocall redeems principal on the
        first observation at or above the autocall barrier.  If the note is not
        called, maturity redemption is one unless the knock-in was touched and
        the final level is below the initial level, in which case redemption is
        ``S_T / S0``.
        """
        return self.compute_payoff_with_discount_curve(
            paths=paths,
            params=params,
            T=T,
            discount_factor=lambda time_years: math.exp(-r * time_years),
        )

    def compute_payoff_with_discount_curve(
        self,
        paths: np.ndarray,
        params: Dict[str, Any],
        T: float,
        discount_factor: Callable[[float], float],
    ) -> np.ndarray:
        """Return Phoenix PVs using a caller-supplied deterministic curve."""
        n_paths, n_points = paths.shape
        n_steps = n_points - 1
        obs_count = int(params.get("obs_count", 6))
        if obs_count < 1 or obs_count > n_steps:
            raise ValueError("obs_count must be between 1 and the path step count")
        obs_idx = np.linspace(0, n_steps, obs_count + 1, dtype=int)[1:]

        s0 = float(params["S0"])
        autocall_b = s0 * float(params["autocall_barrier_frac"])
        coupon_b = s0 * float(params["coupon_barrier_frac"])
        coupon_rate = float(params["coupon_rate"])
        knockin_b = s0 * float(params["knock_in_frac"])

        present_values = np.zeros(n_paths, dtype=np.float64)
        active = np.ones(n_paths, dtype=bool)

        for idx in obs_idx:
            observation_time = (idx / n_steps) * T
            observation_discount = float(discount_factor(observation_time))
            levels = paths[:, idx]

            coupon_due = active & (levels >= coupon_b)
            present_values[coupon_due] += coupon_rate * observation_discount

            called = active & (levels >= autocall_b)
            present_values[called] += observation_discount
            active[called] = False

        if np.any(active):
            final_levels = paths[:, -1]
            knocked_in = np.any(paths <= knockin_b, axis=1)
            capital_loss = knocked_in & (final_levels < s0)
            redemption = np.ones(n_paths, dtype=np.float64)
            redemption[capital_loss] = final_levels[capital_loss] / s0
            present_values[active] += redemption[active] * float(discount_factor(T))

        return present_values


class AccumulatorPayoff(BasePayoff):
    """Accumulator - accumulates shares when in range."""

    def __init__(self):
        param_ranges = {
            "S0": (80.0, 120.0),
            "r": (0.0, 0.05),
            "sigma": (0.05, 0.45),
            "T": (0.5, 2.5),
            "upper_barrier_frac": (1.02, 1.10),
            "lower_barrier_frac": (0.90, 0.98),
            "participation_rate": (1.5, 3.0),
            "obs_frequency": (0.1, 1.0),
        }
        super().__init__(param_ranges)

    def get_parameter_names(self) -> List[str]:
        return [
            "S0",
            "r",
            "sigma",
            "T",
            "upper_barrier_frac",
            "lower_barrier_frac",
            "participation_rate",
            "obs_frequency",
        ]

    def get_feature_order(self) -> List[str]:
        return [
            "S0",
            "r",
            "sigma",
            "T",
            "upper_barrier_frac",
            "lower_barrier_frac",
            "participation_rate",
            "obs_frequency",
        ]

    def compute_payoff(
        self, paths: np.ndarray, params: Dict[str, Any], r: float, T: float
    ) -> np.ndarray:
        """Accumulate shares at discount when price is within barriers."""
        n_paths, n_points = paths.shape
        n_steps = n_points - 1

        S0 = params["S0"]
        upper_barrier = params["S0"] * params["upper_barrier_frac"]
        lower_barrier = params["S0"] * params["lower_barrier_frac"]
        participation_rate = params["participation_rate"]
        obs_frequency = params.get("obs_frequency", 0.25)

        n_obs = max(2, int(T / obs_frequency))
        obs_idx = np.linspace(0, n_steps, n_obs + 1, dtype=int)[1:]

        payoffs = np.zeros(n_paths, dtype=np.float64)

        for i in range(n_paths):
            path = paths[i]
            accumulated_value = 0.0
            count = 0

            for idx in obs_idx:
                S_t = path[idx]
                if lower_barrier < S_t < upper_barrier:
                    discounted_price = S_t / (1 + participation_rate)
                    accumulated_value += discounted_price
                    count += 1

            if count > 0:
                avg_price = accumulated_value / count
                payoffs[i] = avg_price * math.exp(-r * T) * count / n_obs
            else:
                payoffs[i] = 0.0

        return payoffs


class BarrierOptionPayoff(BasePayoff):
    """Down-and-out barrier option."""

    def __init__(self):
        param_ranges = {
            "S0": (80.0, 120.0),
            "r": (0.0, 0.05),
            "sigma": (0.05, 0.45),
            "T": (0.5, 2.5),
            "K": (80.0, 120.0),
            "barrier_frac": (0.60, 0.95),
            "option_type": (0.0, 1.0),
        }
        super().__init__(param_ranges)

    def get_parameter_names(self) -> List[str]:
        return ["S0", "r", "sigma", "T", "K", "barrier_frac", "option_type"]

    def get_feature_order(self) -> List[str]:
        return ["S0", "r", "sigma", "T", "K", "barrier_frac", "option_type"]

    def compute_payoff(
        self, paths: np.ndarray, params: Dict[str, Any], r: float, T: float
    ) -> np.ndarray:
        """Barrier option payoff with improved stability."""
        n_paths = paths.shape[0]
        K = params["K"]
        barrier = params["S0"] * params["barrier_frac"]
        is_call = params["option_type"] >= 0.5

        payoffs = np.zeros(n_paths, dtype=np.float64)

        for i in range(n_paths):
            path = paths[i]

            # Check if barrier was breached
            hit_barrier = np.any(path <= (barrier + 1e-10))

            if hit_barrier:
                payoffs[i] = 0.0
            else:
                S_T = path[-1]
                # Add small epsilon for numerical stability
                if is_call:
                    payoff = max(S_T - K, 0.0)
                else:
                    payoff = max(K - S_T, 0.0)

                # Discount to present value
                payoffs[i] = payoff * math.exp(-r * T) if payoff > 1e-10 else 0.0

        return payoffs


class DecumulatorPayoff(BasePayoff):
    """Decumulator - sells shares when price is outside barriers."""

    def __init__(self):
        param_ranges = {
            "S0": (80.0, 120.0),
            "r": (0.0, 0.05),
            "sigma": (0.05, 0.45),
            "T": (0.5, 2.5),
            "upper_barrier_frac": (1.02, 1.10),
            "lower_barrier_frac": (0.90, 0.98),
            "participation_rate": (1.5, 3.0),
            "obs_frequency": (0.1, 1.0),
        }
        super().__init__(param_ranges)

    def get_parameter_names(self) -> List[str]:
        return [
            "S0",
            "r",
            "sigma",
            "T",
            "upper_barrier_frac",
            "lower_barrier_frac",
            "participation_rate",
            "obs_frequency",
        ]

    def get_feature_order(self) -> List[str]:
        return [
            "S0",
            "r",
            "sigma",
            "T",
            "upper_barrier_frac",
            "lower_barrier_frac",
            "participation_rate",
            "obs_frequency",
        ]

    def compute_payoff(
        self, paths: np.ndarray, params: Dict[str, Any], r: float, T: float
    ) -> np.ndarray:
        """Sell shares when price is outside barriers."""
        n_paths, n_points = paths.shape
        n_steps = n_points - 1

        S0 = params["S0"]
        upper_barrier = params["S0"] * params["upper_barrier_frac"]
        lower_barrier = params["S0"] * params["lower_barrier_frac"]
        participation_rate = params["participation_rate"]
        obs_frequency = params.get("obs_frequency", 0.25)

        n_obs = max(2, int(T / obs_frequency))
        obs_idx = np.linspace(0, n_steps, n_obs + 1, dtype=int)[1:]

        payoffs = np.zeros(n_paths, dtype=np.float64)

        for i in range(n_paths):
            path = paths[i]
            accumulated_value = 0.0
            count = 0

            for idx in obs_idx:
                S_t = path[idx]
                if S_t >= upper_barrier or S_t <= lower_barrier:
                    discounted_price = S_t * (1 + participation_rate)
                    accumulated_value += discounted_price
                    count += 1

            if count > 0:
                avg_price = accumulated_value / count
                payoffs[i] = avg_price * math.exp(-r * T) * count / n_obs
            else:
                payoffs[i] = 0.0

        return payoffs
