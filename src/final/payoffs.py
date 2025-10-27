import math
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple
import numpy as np


class BasePayoff(ABC):
    """Base class for payoff functions."""

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
    """Phoenix/autocallable structured product payoff."""

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
        ]

    def compute_payoff(
        self, paths: np.ndarray, params: Dict[str, Any], r: float, T: float
    ) -> np.ndarray:
        """Phoenix payoff: autocall or knockout depending on barriers."""
        n_paths, n_points = paths.shape
        n_steps = n_points - 1
        obs_count = int(params.get("obs_count", 6))
        obs_idx = np.linspace(0, n_steps, obs_count + 1, dtype=int)[1:]

        autocall_b = params["S0"] * params["autocall_barrier_frac"]
        coupon_rate = params["coupon_rate"]
        knockin_b = params["S0"] * params["knock_in_frac"]

        payoffs = np.zeros(n_paths, dtype=np.float64)

        for i in range(n_paths):
            path = paths[i]
            knocked_in = np.any(path < knockin_b)
            call_idx = None

            for idx in obs_idx:
                if path[idx] >= autocall_b:
                    call_idx = idx
                    break

            if call_idx is not None:
                t_call = (call_idx / n_steps) * T
                payoff = 1.0 + coupon_rate
                payoffs[i] = payoff * math.exp(-r * t_call)
            else:
                t_maturity = T
                if knocked_in:
                    payoff = path[-1] / path[0]
                else:
                    payoff = 1.0 + coupon_rate
                payoffs[i] = payoff * math.exp(-r * t_maturity)

        return payoffs


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
