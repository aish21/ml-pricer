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

    def compute_payoff_with_explicit_schedule_and_discount_curve(
        self,
        paths: np.ndarray,
        params: Dict[str, Any],
        path_times_years: np.ndarray,
        observation_times_years: Tuple[float, ...],
        prior_knock_in_breached: bool,
        discount_factor: Callable[[float], float],
        autocall_barrier_fracs: Tuple[float, ...] | None = None,
        memory_coupon: bool = False,
        unpaid_coupon_count: int = 0,
    ) -> np.ndarray:
        """Return active-trade PVs using exact remaining observation times."""
        ledger = self.compute_observation_event_ledger_with_explicit_schedule(
            paths=paths,
            params=params,
            path_times_years=path_times_years,
            observation_times_years=observation_times_years,
            prior_knock_in_breached=prior_knock_in_breached,
            discount_factor=discount_factor,
            autocall_barrier_fracs=autocall_barrier_fracs,
            memory_coupon=memory_coupon,
            unpaid_coupon_count=unpaid_coupon_count,
        )
        return (
            ledger["coupon_pv"]
            + ledger["autocall_principal_pv"]
            + ledger["maturity_protected_pv"]
            + ledger["maturity_downside_pv"]
        )

    def compute_observation_event_ledger_with_explicit_schedule(
        self,
        paths: np.ndarray,
        params: Dict[str, Any],
        path_times_years: np.ndarray,
        observation_times_years: Tuple[float, ...],
        prior_knock_in_breached: bool,
        discount_factor: Callable[[float], float],
        autocall_barrier_fracs: Tuple[float, ...] | None = None,
        memory_coupon: bool = False,
        unpaid_coupon_count: int = 0,
    ) -> Dict[str, np.ndarray]:
        """Return cashflows for an active note with explicit remaining state."""
        path_array = np.asarray(paths, dtype=np.float64)
        path_times = np.asarray(path_times_years, dtype=np.float64)
        observations = np.asarray(observation_times_years, dtype=np.float64)
        if path_array.ndim != 2 or path_array.shape[1] < 2:
            raise ValueError("paths must be a two-dimensional path matrix")
        if not np.all(np.isfinite(path_array)) or np.any(path_array <= 0.0):
            raise ValueError("paths must contain finite positive levels")
        if (
            path_times.ndim != 1
            or len(path_times) != path_array.shape[1]
            or not np.all(np.isfinite(path_times))
            or not math.isclose(float(path_times[0]), 0.0, abs_tol=1e-12)
            or np.any(np.diff(path_times) <= 0.0)
        ):
            raise ValueError(
                "path_times_years must align with paths and be strictly increasing"
            )
        if (
            observations.ndim != 1
            or len(observations) < 1
            or not np.all(np.isfinite(observations))
            or np.any(observations <= 0.0)
            or np.any(np.diff(observations) <= 0.0)
            or observations[-1] > path_times[-1] + 1e-12
        ):
            raise ValueError(
                "observation_times_years must be finite, positive, strictly "
                "increasing, and covered by the path grid"
            )
        if not isinstance(prior_knock_in_breached, bool):
            raise ValueError("prior_knock_in_breached must be boolean")
        if not isinstance(memory_coupon, bool):
            raise ValueError("memory_coupon must be boolean")
        if isinstance(unpaid_coupon_count, bool) or not isinstance(
            unpaid_coupon_count, int
        ):
            raise ValueError("unpaid_coupon_count must be an integer")
        if unpaid_coupon_count < 0 or unpaid_coupon_count > 252:
            raise ValueError("unpaid_coupon_count must be between 0 and 252")
        if not memory_coupon and unpaid_coupon_count:
            raise ValueError(
                "unpaid_coupon_count must be zero when memory_coupon is false"
            )

        observation_indices: list[int] = []
        for observation_time in observations:
            index = int(np.searchsorted(path_times, observation_time))
            if index >= len(path_times) or not math.isclose(
                float(path_times[index]),
                float(observation_time),
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    "every observation time must be present in the path grid"
                )
            observation_indices.append(index)

        n_paths = path_array.shape[0]
        obs_count = len(observation_indices)
        reference_level = float(params["S0"])
        if autocall_barrier_fracs is None:
            autocall_barrier_schedule = np.full(
                obs_count,
                float(params["autocall_barrier_frac"]),
                dtype=np.float64,
            )
        else:
            autocall_barrier_schedule = np.asarray(
                autocall_barrier_fracs, dtype=np.float64
            )
            if (
                autocall_barrier_schedule.ndim != 1
                or len(autocall_barrier_schedule) != obs_count
                or not np.all(np.isfinite(autocall_barrier_schedule))
                or np.any(autocall_barrier_schedule <= 0.0)
            ):
                raise ValueError(
                    "autocall_barrier_fracs must contain one finite positive "
                    "barrier per observation"
                )
        coupon_barrier = reference_level * float(params["coupon_barrier_frac"])
        coupon_rate = float(params["coupon_rate"])
        knock_in_barrier = reference_level * float(params["knock_in_frac"])

        coupon_pv = np.zeros(n_paths, dtype=np.float64)
        autocall_principal_pv = np.zeros(n_paths, dtype=np.float64)
        maturity_protected_pv = np.zeros(n_paths, dtype=np.float64)
        maturity_downside_pv = np.zeros(n_paths, dtype=np.float64)
        autocalled = np.zeros(n_paths, dtype=bool)
        downside_redemption = np.zeros(n_paths, dtype=bool)
        coupon_event = np.zeros((n_paths, obs_count), dtype=np.float64)
        coupon_amount_event = np.zeros((n_paths, obs_count), dtype=np.float64)
        first_autocall_event = np.zeros((n_paths, obs_count), dtype=np.float64)
        survival_after_observation = np.zeros((n_paths, obs_count), dtype=np.float64)
        observation_discounts = np.zeros(obs_count, dtype=np.float64)
        active = np.ones(n_paths, dtype=bool)
        coupon_memory_balance = np.full(n_paths, unpaid_coupon_count, dtype=np.int64)

        for observation_index, (time_years, path_index) in enumerate(
            zip(observations, observation_indices)
        ):
            observation_discount = float(discount_factor(float(time_years)))
            levels = path_array[:, path_index]
            observation_discounts[observation_index] = observation_discount

            coupon_due = active & (levels >= coupon_barrier)
            if memory_coupon:
                coupon_amount = coupon_rate * (coupon_memory_balance + 1)
                coupon_pv[coupon_due] += (
                    coupon_amount[coupon_due] * observation_discount
                )
                coupon_amount_event[coupon_due, observation_index] = coupon_amount[
                    coupon_due
                ]
                coupon_memory_balance[coupon_due] = 0
                coupon_missed = active & ~coupon_due
                coupon_memory_balance[coupon_missed] += 1
            else:
                coupon_pv[coupon_due] += coupon_rate * observation_discount
                coupon_amount_event[coupon_due, observation_index] = coupon_rate
            coupon_event[:, observation_index] = coupon_due

            autocall_barrier = (
                reference_level * autocall_barrier_schedule[observation_index]
            )
            called = active & (levels >= autocall_barrier)
            autocall_principal_pv[called] = observation_discount
            autocalled[called] = True
            first_autocall_event[:, observation_index] = called
            active[called] = False
            survival_after_observation[:, observation_index] = active

        protected = np.zeros(n_paths, dtype=bool)
        downside_recovery_ratio = np.zeros(n_paths, dtype=np.float64)
        if np.any(active):
            final_levels = path_array[:, -1]
            knocked_in = np.any(path_array <= knock_in_barrier, axis=1)
            if prior_knock_in_breached:
                knocked_in = np.ones(n_paths, dtype=bool)
            capital_loss = active & knocked_in & (final_levels < reference_level)
            protected = active & ~capital_loss
            maturity_discount = float(discount_factor(float(path_times[-1])))
            maturity_protected_pv[protected] = maturity_discount
            maturity_downside_pv[capital_loss] = (
                final_levels[capital_loss] / reference_level
            ) * maturity_discount
            downside_redemption[capital_loss] = True
            downside_recovery_ratio[capital_loss] = (
                final_levels[capital_loss] / reference_level
            )

        return {
            "coupon_pv": coupon_pv,
            "autocall_principal_pv": autocall_principal_pv,
            "maturity_protected_pv": maturity_protected_pv,
            "maturity_downside_pv": maturity_downside_pv,
            "autocall_probability": autocalled.astype(np.float64),
            "downside_probability": downside_redemption.astype(np.float64),
            "coupon_event": coupon_event,
            "coupon_amount_event": coupon_amount_event,
            "coupon_memory_balance": coupon_memory_balance.astype(np.float64),
            "first_autocall_event": first_autocall_event,
            "survival_after_observation": survival_after_observation,
            "protected_maturity_event": protected.astype(np.float64),
            "downside_maturity_event": downside_redemption.astype(np.float64),
            "downside_recovery_ratio": downside_recovery_ratio,
            "observation_times": observations.copy(),
            "observation_discounts": observation_discounts,
            "autocall_barrier_fracs": autocall_barrier_schedule.copy(),
        }

    def compute_observation_event_ledger_with_discount_curve(
        self,
        paths: np.ndarray,
        params: Dict[str, Any],
        T: float,
        discount_factor: Callable[[float], float],
    ) -> Dict[str, np.ndarray]:
        """Return pathwise cashflows and the observation-level event ledger.

        The ledger is the canonical source for aggregate cashflow labels and
        research-only hazard labels. Keeping both decompositions beside the
        payoff rules prevents data generators from implementing another
        version of the product.
        """
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

        coupon_pv = np.zeros(n_paths, dtype=np.float64)
        autocall_principal_pv = np.zeros(n_paths, dtype=np.float64)
        maturity_protected_pv = np.zeros(n_paths, dtype=np.float64)
        maturity_downside_pv = np.zeros(n_paths, dtype=np.float64)
        autocalled = np.zeros(n_paths, dtype=bool)
        downside_redemption = np.zeros(n_paths, dtype=bool)
        coupon_event = np.zeros((n_paths, obs_count), dtype=np.float64)
        first_autocall_event = np.zeros((n_paths, obs_count), dtype=np.float64)
        survival_after_observation = np.zeros((n_paths, obs_count), dtype=np.float64)
        observation_times = np.zeros(obs_count, dtype=np.float64)
        observation_discounts = np.zeros(obs_count, dtype=np.float64)
        active = np.ones(n_paths, dtype=bool)

        for observation_index, idx in enumerate(obs_idx):
            observation_time = (idx / n_steps) * T
            observation_discount = float(discount_factor(observation_time))
            levels = paths[:, idx]
            observation_times[observation_index] = observation_time
            observation_discounts[observation_index] = observation_discount

            coupon_due = active & (levels >= coupon_b)
            coupon_pv[coupon_due] += coupon_rate * observation_discount
            coupon_event[:, observation_index] = coupon_due

            called = active & (levels >= autocall_b)
            autocall_principal_pv[called] = observation_discount
            autocalled[called] = True
            first_autocall_event[:, observation_index] = called
            active[called] = False
            survival_after_observation[:, observation_index] = active

        protected = np.zeros(n_paths, dtype=bool)
        downside_recovery_ratio = np.zeros(n_paths, dtype=np.float64)
        if np.any(active):
            final_levels = paths[:, -1]
            knocked_in = np.any(paths <= knockin_b, axis=1)
            capital_loss = active & knocked_in & (final_levels < s0)
            protected = active & ~capital_loss
            maturity_discount = float(discount_factor(T))
            maturity_protected_pv[protected] = maturity_discount
            maturity_downside_pv[capital_loss] = (
                final_levels[capital_loss] / s0
            ) * maturity_discount
            downside_redemption[capital_loss] = True
            downside_recovery_ratio[capital_loss] = final_levels[capital_loss] / s0

        return {
            "coupon_pv": coupon_pv,
            "autocall_principal_pv": autocall_principal_pv,
            "maturity_protected_pv": maturity_protected_pv,
            "maturity_downside_pv": maturity_downside_pv,
            "autocall_probability": autocalled.astype(np.float64),
            "downside_probability": downside_redemption.astype(np.float64),
            "coupon_event": coupon_event,
            "first_autocall_event": first_autocall_event,
            "survival_after_observation": survival_after_observation,
            "protected_maturity_event": protected.astype(np.float64),
            "downside_maturity_event": downside_redemption.astype(np.float64),
            "downside_recovery_ratio": downside_recovery_ratio,
            "observation_times": observation_times,
            "observation_discounts": observation_discounts,
        }

    def compute_cashflow_components_with_discount_curve(
        self,
        paths: np.ndarray,
        params: Dict[str, Any],
        T: float,
        discount_factor: Callable[[float], float],
    ) -> Dict[str, np.ndarray]:
        """Return the stable aggregate cashflow and event-label contract."""
        ledger = self.compute_observation_event_ledger_with_discount_curve(
            paths=paths,
            params=params,
            T=T,
            discount_factor=discount_factor,
        )
        return {
            name: ledger[name]
            for name in (
                "coupon_pv",
                "autocall_principal_pv",
                "maturity_protected_pv",
                "maturity_downside_pv",
                "autocall_probability",
                "downside_probability",
            )
        }


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
