import math
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

from src.final.surrogate_model import json_sha256
from src.final.surrogate_price_first_contract import (
    PHOENIX_PRICE_FIRST_APPROVED_ARTIFACT_ID,
)


SHADOW_PROMOTION_POLICY_VERSION = "phoenix-shadow-promotion-readiness-v1"
SHADOW_PROMOTION_POLICY_ID = (
    "sha256:efbbe2e882555ba8720962090abdbb5d82807eeb258627a95e6058cbccb4dda5"
)


@dataclass(frozen=True)
class ShadowPromotionPolicy:
    """Frozen evidence gates for a human review of the shadow model."""

    policy_version: str = SHADOW_PROMOTION_POLICY_VERSION
    artifact_id: str = PHOENIX_PRICE_FIRST_APPROVED_ARTIFACT_ID
    evidence_start_at: str = "2026-07-18T05:10:00+00:00"
    evaluation_scope: str = (
        "new-issue single-underlier Phoenix requests inside the frozen "
        "training domain; seasoned trades are excluded"
    )
    minimum_observations: int = 1_000
    minimum_successful_observations: int = 800
    minimum_distinct_cases: int = 500
    minimum_unique_symbols: int = 5
    minimum_distinct_market_dates: int = 10
    minimum_observation_span_days: float = 14.0
    required_market_regimes: tuple[str, ...] = (
        "low_vol",
        "normal",
        "high_vol",
    )
    required_moneyness_regions: tuple[str, ...] = (
        "broad",
        "coupon",
        "autocall",
    )
    minimum_successful_per_slice: int = 25
    minimum_successful_per_joint_slice: int = 10
    minimum_success_fraction: float = 0.90
    minimum_artifact_match_fraction: float = 1.0
    minimum_current_schema_fraction: float = 1.0
    maximum_out_of_domain_fraction: float = 0.10
    maximum_unavailable_fraction: float = 0.01
    maximum_error_fraction: float = 0.01
    maximum_mae: float = 0.015
    maximum_p95_absolute_error: float = 0.04
    minimum_within_two_reference_se_fraction: float = 0.80
    maximum_regime_mae: float = 0.02
    maximum_moneyness_region_mae: float = 0.02
    maximum_joint_slice_mae: float = 0.025
    maximum_p95_latency_ms: float = 25.0
    maximum_above_four_sigma_fraction: float = 0.05

    def __post_init__(self) -> None:
        if self.policy_version != SHADOW_PROMOTION_POLICY_VERSION:
            raise ValueError("shadow promotion policy version is invalid")
        if not self.evaluation_scope.strip():
            raise ValueError("shadow promotion evaluation scope is invalid")
        if not self.artifact_id.startswith("sha256:") or len(self.artifact_id) != 71:
            raise ValueError("shadow promotion artifact id is invalid")
        try:
            parsed_start = datetime.fromisoformat(
                self.evidence_start_at.replace("Z", "+00:00")
            )
        except ValueError as exc:
            raise ValueError("shadow promotion evidence start is invalid") from exc
        if parsed_start.tzinfo is None or parsed_start.utcoffset() is None:
            raise ValueError("shadow promotion evidence start must include UTC offset")
        positive_counts = (
            self.minimum_observations,
            self.minimum_successful_observations,
            self.minimum_distinct_cases,
            self.minimum_unique_symbols,
            self.minimum_distinct_market_dates,
            self.minimum_successful_per_slice,
            self.minimum_successful_per_joint_slice,
        )
        if any(value < 1 for value in positive_counts):
            raise ValueError("shadow promotion sample requirements are invalid")
        if (
            not math.isfinite(self.minimum_observation_span_days)
            or self.minimum_observation_span_days <= 0.0
        ):
            raise ValueError("shadow promotion time-span requirement is invalid")
        fractions = (
            self.minimum_success_fraction,
            self.minimum_artifact_match_fraction,
            self.minimum_current_schema_fraction,
            self.maximum_out_of_domain_fraction,
            self.maximum_unavailable_fraction,
            self.maximum_error_fraction,
            self.minimum_within_two_reference_se_fraction,
            self.maximum_above_four_sigma_fraction,
        )
        if any(
            not math.isfinite(value) or value < 0.0 or value > 1.0
            for value in fractions
        ):
            raise ValueError("shadow promotion fraction requirements are invalid")
        maxima = (
            self.maximum_mae,
            self.maximum_p95_absolute_error,
            self.maximum_regime_mae,
            self.maximum_moneyness_region_mae,
            self.maximum_joint_slice_mae,
            self.maximum_p95_latency_ms,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in maxima):
            raise ValueError("shadow promotion metric requirements are invalid")
        if (
            not self.required_market_regimes
            or len(set(self.required_market_regimes))
            != len(self.required_market_regimes)
            or not self.required_moneyness_regions
            or len(set(self.required_moneyness_regions))
            != len(self.required_moneyness_regions)
        ):
            raise ValueError("shadow promotion slice requirements are invalid")

    @property
    def policy_id(self) -> str:
        return json_sha256(asdict(self))

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["required_market_regimes"] = list(self.required_market_regimes)
        payload["required_moneyness_regions"] = list(self.required_moneyness_regions)
        return {**payload, "policy_id": self.policy_id}


DEFAULT_SHADOW_PROMOTION_POLICY = ShadowPromotionPolicy()
if DEFAULT_SHADOW_PROMOTION_POLICY.policy_id != SHADOW_PROMOTION_POLICY_ID:
    raise RuntimeError(
        "default shadow promotion policy changed without a new frozen policy id"
    )
