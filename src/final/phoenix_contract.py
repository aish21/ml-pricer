import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Iterable


PHOENIX_SINGLE_V2_CONTRACT_VERSION = "phoenix-single-v2"
MAX_PHOENIX_OBSERVATIONS = 252


class PhoenixContractValidationError(ValueError):
    """Raised when contractual Phoenix state is incomplete or inconsistent."""


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise PhoenixContractValidationError(f"{name} must be numeric")
    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise PhoenixContractValidationError(f"{name} must be numeric") from exc
    if not math.isfinite(normalized):
        raise PhoenixContractValidationError(f"{name} must be finite")
    return normalized


def _observation_schedule(values: Iterable[Any], maturity: float) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)):
        raise PhoenixContractValidationError(
            "observation_times_years must be a sequence"
        )
    try:
        normalized = tuple(_finite_float(value, "observation time") for value in values)
    except TypeError as exc:
        raise PhoenixContractValidationError(
            "observation_times_years must be a sequence"
        ) from exc
    if not normalized:
        raise PhoenixContractValidationError(
            "observation_times_years must not be empty"
        )
    if len(normalized) > MAX_PHOENIX_OBSERVATIONS:
        raise PhoenixContractValidationError(
            f"observation_times_years must contain at most "
            f"{MAX_PHOENIX_OBSERVATIONS} entries"
        )
    if any(value <= 0.0 or value > maturity for value in normalized):
        raise PhoenixContractValidationError(
            "observation times must satisfy 0 < time <= maturity_years"
        )
    if any(
        current <= previous
        for previous, current in zip(normalized[:-1], normalized[1:])
    ):
        raise PhoenixContractValidationError(
            "observation_times_years must be strictly increasing"
        )
    if not math.isclose(normalized[-1], maturity, rel_tol=0.0, abs_tol=1e-12):
        raise PhoenixContractValidationError(
            "the final observation time must equal maturity_years"
        )
    return normalized


@dataclass(frozen=True)
class PhoenixSingleV2Contract:
    """Remaining contractual state for an active single-underlier Phoenix.

    Times are ACT/365F-like year fractions measured forward from the valuation
    time. Historical coupon and autocall events need not be stored for an
    active non-memory note, but a historical knock-in remains economically
    relevant and is therefore explicit.
    """

    reference_level: float
    maturity_years: float
    observation_times_years: tuple[float, ...]
    autocall_barrier_frac: float
    coupon_barrier_frac: float
    coupon_rate: float
    knock_in_frac: float
    prior_knock_in_breached: bool
    contract_version: str = PHOENIX_SINGLE_V2_CONTRACT_VERSION

    def __post_init__(self) -> None:
        if self.contract_version != PHOENIX_SINGLE_V2_CONTRACT_VERSION:
            raise PhoenixContractValidationError(
                "contract_version must be phoenix-single-v2"
            )
        reference_level = _finite_float(self.reference_level, "reference_level")
        maturity = _finite_float(self.maturity_years, "maturity_years")
        autocall = _finite_float(self.autocall_barrier_frac, "autocall_barrier_frac")
        coupon_barrier = _finite_float(self.coupon_barrier_frac, "coupon_barrier_frac")
        coupon_rate = _finite_float(self.coupon_rate, "coupon_rate")
        knock_in = _finite_float(self.knock_in_frac, "knock_in_frac")
        if reference_level <= 0.0:
            raise PhoenixContractValidationError("reference_level must be > 0")
        if maturity <= 0.0 or maturity > 30.0:
            raise PhoenixContractValidationError(
                "maturity_years must satisfy 0 < maturity_years <= 30"
            )
        if not 0.0 < knock_in <= 1.0:
            raise PhoenixContractValidationError(
                "knock_in_frac must satisfy 0 < knock_in_frac <= 1"
            )
        if not knock_in <= coupon_barrier <= autocall:
            raise PhoenixContractValidationError(
                "barriers must satisfy knock_in_frac <= coupon_barrier_frac "
                "<= autocall_barrier_frac"
            )
        if autocall > 3.0:
            raise PhoenixContractValidationError("autocall_barrier_frac must be <= 3")
        if coupon_rate < 0.0 or coupon_rate > 1.0:
            raise PhoenixContractValidationError(
                "coupon_rate must satisfy 0 <= coupon_rate <= 1"
            )
        if not isinstance(self.prior_knock_in_breached, bool):
            raise PhoenixContractValidationError(
                "prior_knock_in_breached must be boolean"
            )
        observations = _observation_schedule(self.observation_times_years, maturity)

        object.__setattr__(self, "reference_level", reference_level)
        object.__setattr__(self, "maturity_years", maturity)
        object.__setattr__(self, "observation_times_years", observations)
        object.__setattr__(self, "autocall_barrier_frac", autocall)
        object.__setattr__(self, "coupon_barrier_frac", coupon_barrier)
        object.__setattr__(self, "coupon_rate", coupon_rate)
        object.__setattr__(self, "knock_in_frac", knock_in)

    def _canonical_payload(self) -> dict[str, Any]:
        return {
            "contract_version": self.contract_version,
            "reference_level": self.reference_level,
            "maturity_years": self.maturity_years,
            "observation_times_years": list(self.observation_times_years),
            "autocall_barrier_frac": self.autocall_barrier_frac,
            "coupon_barrier_frac": self.coupon_barrier_frac,
            "coupon_rate": self.coupon_rate,
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
            "remaining_observation_count": len(self.observation_times_years),
        }

    def to_payoff_params(
        self, *, risk_free_rate: float, volatility: float
    ) -> dict[str, Any]:
        return {
            "S0": self.reference_level,
            "r": float(risk_free_rate),
            "sigma": float(volatility),
            "T": self.maturity_years,
            "autocall_barrier_frac": self.autocall_barrier_frac,
            "coupon_barrier_frac": self.coupon_barrier_frac,
            "coupon_rate": self.coupon_rate,
            "knock_in_frac": self.knock_in_frac,
            "obs_count": len(self.observation_times_years),
        }
