import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Callable, List

import numpy as np

from .payoffs import BasePayoff


BARRIER_REVERSE_CONVERTIBLE_V1 = "barrier-reverse-convertible-v1"


class BarrierReverseConvertibleValidationError(ValueError):
    pass


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise BarrierReverseConvertibleValidationError(f"{name} must be numeric")
    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise BarrierReverseConvertibleValidationError(
            f"{name} must be numeric"
        ) from exc
    if not math.isfinite(normalized):
        raise BarrierReverseConvertibleValidationError(f"{name} must be finite")
    return normalized


@dataclass(frozen=True)
class BarrierReverseConvertibleV1Contract:
    """Exact remaining state for a single-underlier barrier reverse convertible."""

    reference_level: float
    maturity_years: float
    coupon_times_years: tuple[float, ...]
    coupon_rate_per_period: float
    strike_frac: float
    knock_in_frac: float
    prior_knock_in_breached: bool = False
    contract_version: str = BARRIER_REVERSE_CONVERTIBLE_V1

    def __post_init__(self) -> None:
        if self.contract_version != BARRIER_REVERSE_CONVERTIBLE_V1:
            raise BarrierReverseConvertibleValidationError(
                "contract_version must be barrier-reverse-convertible-v1"
            )
        reference = _finite(self.reference_level, "reference_level")
        maturity = _finite(self.maturity_years, "maturity_years")
        coupon_rate = _finite(self.coupon_rate_per_period, "coupon_rate_per_period")
        strike = _finite(self.strike_frac, "strike_frac")
        knock_in = _finite(self.knock_in_frac, "knock_in_frac")
        if reference <= 0.0:
            raise BarrierReverseConvertibleValidationError(
                "reference_level must be > 0"
            )
        if maturity <= 0.0 or maturity > 30.0:
            raise BarrierReverseConvertibleValidationError(
                "maturity_years must satisfy 0 < value <= 30"
            )
        if coupon_rate < 0.0 or coupon_rate > 1.0:
            raise BarrierReverseConvertibleValidationError(
                "coupon_rate_per_period must satisfy 0 <= value <= 1"
            )
        if strike <= 0.0 or strike > 3.0:
            raise BarrierReverseConvertibleValidationError(
                "strike_frac must satisfy 0 < value <= 3"
            )
        if knock_in <= 0.0 or knock_in > strike:
            raise BarrierReverseConvertibleValidationError(
                "barriers must satisfy 0 < knock_in_frac <= strike_frac"
            )
        if not isinstance(self.prior_knock_in_breached, bool):
            raise BarrierReverseConvertibleValidationError(
                "prior_knock_in_breached must be boolean"
            )
        try:
            coupon_times = tuple(
                _finite(value, "coupon time") for value in self.coupon_times_years
            )
        except TypeError as exc:
            raise BarrierReverseConvertibleValidationError(
                "coupon_times_years must be a sequence"
            ) from exc
        if not coupon_times or len(coupon_times) > 252:
            raise BarrierReverseConvertibleValidationError(
                "coupon_times_years must contain between 1 and 252 entries"
            )
        if any(value <= 0.0 or value > maturity for value in coupon_times):
            raise BarrierReverseConvertibleValidationError(
                "coupon times must satisfy 0 < time <= maturity_years"
            )
        if any(
            current <= previous
            for previous, current in zip(coupon_times[:-1], coupon_times[1:])
        ):
            raise BarrierReverseConvertibleValidationError(
                "coupon_times_years must be strictly increasing"
            )
        if not math.isclose(coupon_times[-1], maturity, abs_tol=1e-12):
            raise BarrierReverseConvertibleValidationError(
                "the final coupon time must equal maturity_years"
            )
        object.__setattr__(self, "reference_level", reference)
        object.__setattr__(self, "maturity_years", maturity)
        object.__setattr__(self, "coupon_times_years", coupon_times)
        object.__setattr__(self, "coupon_rate_per_period", coupon_rate)
        object.__setattr__(self, "strike_frac", strike)
        object.__setattr__(self, "knock_in_frac", knock_in)

    def _canonical_payload(self) -> dict[str, Any]:
        return {
            "contract_version": self.contract_version,
            "reference_level": self.reference_level,
            "maturity_years": self.maturity_years,
            "coupon_times_years": list(self.coupon_times_years),
            "coupon_rate_per_period": self.coupon_rate_per_period,
            "strike_frac": self.strike_frac,
            "knock_in_frac": self.knock_in_frac,
            "prior_knock_in_breached": self.prior_knock_in_breached,
        }

    @property
    def contract_id(self) -> str:
        encoded = json.dumps(
            self._canonical_payload(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return f"sha256:{hashlib.sha256(encoded).hexdigest()}"

    def to_dict(self) -> dict[str, Any]:
        return {
            **self._canonical_payload(),
            "contract_id": self.contract_id,
            "remaining_coupon_count": len(self.coupon_times_years),
        }

    def to_payoff_params(
        self, *, risk_free_rate: float, volatility: float
    ) -> dict[str, Any]:
        return {
            "S0": self.reference_level,
            "r": float(risk_free_rate),
            "sigma": float(volatility),
            "T": self.maturity_years,
            "coupon_rate_per_period": self.coupon_rate_per_period,
            "strike_frac": self.strike_frac,
            "knock_in_frac": self.knock_in_frac,
            "obs_count": len(self.coupon_times_years),
        }


class BarrierReverseConvertiblePayoff(BasePayoff):
    contract_version = BARRIER_REVERSE_CONVERTIBLE_V1

    def __init__(self):
        super().__init__(
            {
                "S0": (80.0, 120.0),
                "r": (0.0, 0.05),
                "sigma": (0.05, 0.45),
                "T": (0.25, 2.5),
                "coupon_rate_per_period": (0.0025, 0.05),
                "strike_frac": (0.8, 1.2),
                "knock_in_frac": (0.5, 0.9),
                "obs_count": (1, 12),
            }
        )

    def get_parameter_names(self) -> List[str]:
        return [
            "S0",
            "r",
            "sigma",
            "T",
            "coupon_rate_per_period",
            "strike_frac",
            "knock_in_frac",
            "obs_count",
        ]

    def get_feature_order(self) -> List[str]:
        return self.get_parameter_names()

    def compute_payoff(
        self,
        paths: np.ndarray,
        params: dict[str, Any],
        r: float,
        T: float,
    ) -> np.ndarray:
        n_steps = paths.shape[1] - 1
        path_times = np.linspace(0.0, T, n_steps + 1)
        count = int(params.get("obs_count", 1))
        coupon_times = tuple(float(T) * index / count for index in range(1, count + 1))
        ledger = self.compute_event_ledger(
            paths=paths,
            params=params,
            path_times_years=path_times,
            coupon_times_years=coupon_times,
            prior_knock_in_breached=False,
            discount_factor=lambda time_years: math.exp(-r * time_years),
        )
        return (
            ledger["coupon_pv"]
            + ledger["protected_principal_pv"]
            + ledger["downside_redemption_pv"]
        )

    def compute_event_ledger(
        self,
        *,
        paths: np.ndarray,
        params: dict[str, Any],
        path_times_years: np.ndarray,
        coupon_times_years: tuple[float, ...],
        prior_knock_in_breached: bool,
        discount_factor: Callable[[float], float],
    ) -> dict[str, np.ndarray]:
        path_array = np.asarray(paths, dtype=np.float64)
        path_times = np.asarray(path_times_years, dtype=np.float64)
        coupon_times = np.asarray(coupon_times_years, dtype=np.float64)
        if path_array.ndim != 2 or path_array.shape[1] != len(path_times):
            raise ValueError("paths and path_times_years must align")
        if not np.all(np.isfinite(path_array)) or np.any(path_array <= 0.0):
            raise ValueError("paths must contain finite positive levels")
        if (
            len(path_times) < 2
            or not math.isclose(float(path_times[0]), 0.0, abs_tol=1e-12)
            or np.any(np.diff(path_times) <= 0.0)
        ):
            raise ValueError("path_times_years must start at zero and increase")
        coupon_indices = []
        for coupon_time in coupon_times:
            index = int(np.searchsorted(path_times, coupon_time))
            if index >= len(path_times) or not math.isclose(
                float(path_times[index]), float(coupon_time), abs_tol=1e-12
            ):
                raise ValueError("every coupon time must be present in the path grid")
            coupon_indices.append(index)
        if not isinstance(prior_knock_in_breached, bool):
            raise ValueError("prior_knock_in_breached must be boolean")

        n_paths = path_array.shape[0]
        reference = float(params["S0"])
        coupon_rate = float(params["coupon_rate_per_period"])
        strike_level = reference * float(params["strike_frac"])
        knock_in_level = reference * float(params["knock_in_frac"])
        coupon_pv = np.zeros(n_paths, dtype=np.float64)
        coupon_event_pv = np.zeros((n_paths, len(coupon_times)), dtype=np.float64)
        for event_index, coupon_time in enumerate(coupon_times):
            cashflow_pv = coupon_rate * float(discount_factor(float(coupon_time)))
            coupon_pv += cashflow_pv
            coupon_event_pv[:, event_index] = cashflow_pv

        knocked_in = np.any(path_array <= knock_in_level, axis=1)
        if prior_knock_in_breached:
            knocked_in = np.ones(n_paths, dtype=bool)
        final_levels = path_array[:, -1]
        downside = knocked_in & (final_levels < strike_level)
        maturity_discount = float(discount_factor(float(path_times[-1])))
        protected_principal_pv = (~downside).astype(np.float64) * maturity_discount
        downside_redemption_pv = np.zeros(n_paths, dtype=np.float64)
        downside_redemption_pv[downside] = (
            final_levels[downside] / strike_level
        ) * maturity_discount
        return {
            "coupon_pv": coupon_pv,
            "protected_principal_pv": protected_principal_pv,
            "downside_redemption_pv": downside_redemption_pv,
            "coupon_event_pv": coupon_event_pv,
            "knock_in_probability": knocked_in.astype(np.float64),
            "downside_probability": downside.astype(np.float64),
            "downside_recovery_ratio": np.where(
                downside, final_levels / strike_level, 0.0
            ),
            "coupon_times": coupon_times.copy(),
            "coupon_path_indices": np.asarray(coupon_indices, dtype=np.int64),
        }
