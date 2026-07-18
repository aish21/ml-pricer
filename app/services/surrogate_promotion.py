import hashlib
import math
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping

from src.final.surrogate_model import json_sha256

from app.services.surrogate_monitoring import (
    SHADOW_OBSERVATION_SCHEMA_VERSION,
    SurrogateMonitoringError,
    SurrogateMonitoringSettings,
    load_surrogate_shadow_observations,
)
from app.services.surrogate_promotion_policy import (
    DEFAULT_SHADOW_PROMOTION_POLICY,
    ShadowPromotionPolicy,
)


def _quantile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _parse_timestamp(value: Any, label: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise SurrogateMonitoringError(f"stored shadow {label} is invalid") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise SurrogateMonitoringError(f"stored shadow {label} must include UTC offset")
    return parsed.astimezone(timezone.utc)


def _case_id(row: Mapping[str, Any]) -> str:
    encoded = "\x1f".join(
        (
            str(row["market_payload"]),
            str(row["terms_payload"]),
            str(row["contract_reference_spot"]),
        )
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _error_summary(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    selected = list(rows)
    errors = [
        float(row["absolute_error"])
        for row in selected
        if row.get("absolute_error") is not None
    ]
    error_to_se = [
        float(row["error_to_reference_standard_error"])
        for row in selected
        if row.get("error_to_reference_standard_error") is not None
    ]
    latencies = [
        float(row["latency_ms"])
        for row in selected
        if row.get("latency_ms") is not None
    ]
    return {
        "n_successful": len(selected),
        "mae": sum(errors) / len(errors) if errors else None,
        "p95_absolute_error": _quantile(errors, 0.95),
        "within_two_reference_se_fraction": (
            sum(value <= 2.0 for value in error_to_se) / len(error_to_se)
            if error_to_se
            else None
        ),
        "mean_latency_ms": sum(latencies) / len(latencies) if latencies else None,
        "p95_latency_ms": _quantile(latencies, 0.95),
    }


def _minimum_check(
    value: int | float | None,
    minimum: int | float,
    *,
    kind: str,
    details: Any = None,
) -> dict[str, Any]:
    return {
        "kind": kind,
        "value": value,
        "minimum": minimum,
        "passed": value is not None and value >= minimum,
        **({"details": details} if details is not None else {}),
    }


def _maximum_check(
    value: int | float | None,
    maximum: int | float,
    *,
    kind: str,
    details: Any = None,
) -> dict[str, Any]:
    return {
        "kind": kind,
        "value": value,
        "maximum": maximum,
        "passed": value is not None and value <= maximum,
        **({"details": details} if details is not None else {}),
    }


def _worst_metric(
    summaries: Mapping[str, Mapping[str, Any]],
    required_names: Iterable[str],
    metric: str,
) -> float | None:
    values = [
        summaries[name].get(metric)
        for name in required_names
        if name in summaries and summaries[name].get(metric) is not None
    ]
    return max(float(value) for value in values) if values else None


def _disabled_report(policy: ShadowPromotionPolicy) -> dict[str, Any]:
    return {
        "enabled": False,
        "available": False,
        "policy": policy.to_dict(),
        "decision": "insufficient_evidence",
        "ready_for_review": False,
        "runtime_eligible": False,
        "automatic_promotion_permitted": False,
        "reason": "surrogate monitoring is disabled",
    }


def get_surrogate_promotion_readiness(
    *,
    limit: int = 100_000,
    settings: SurrogateMonitoringSettings | None = None,
    policy: ShadowPromotionPolicy = DEFAULT_SHADOW_PROMOTION_POLICY,
) -> dict[str, Any]:
    """Evaluate frozen shadow gates without changing the runtime policy."""
    active = settings or SurrogateMonitoringSettings.from_env()
    if not active.enabled:
        return _disabled_report(policy)
    rows = load_surrogate_shadow_observations(
        target_artifact_id=policy.artifact_id,
        evidence_start_at=policy.evidence_start_at,
        limit=limit,
        settings=active,
    )
    successful = [row for row in rows if row["status"] == "success"]
    status_counts = {
        status: sum(row["status"] == status for row in rows)
        for status in ("success", "out_of_domain", "unavailable", "error")
    }
    n_observations = len(rows)
    status_fractions = {
        status: (status_counts[status] / n_observations if n_observations else None)
        for status in status_counts
    }
    artifact_match_fraction = (
        sum(row["artifact_id"] == policy.artifact_id for row in successful)
        / len(successful)
        if successful
        else None
    )
    current_schema_fraction = (
        sum(row["schema_version"] == SHADOW_OBSERVATION_SCHEMA_VERSION for row in rows)
        / n_observations
        if n_observations
        else None
    )

    created_times = [
        _parse_timestamp(row["created_at"], "created_at") for row in successful
    ]
    market_times = [
        _parse_timestamp(row["market_data_time"], "market_data_time")
        for row in successful
    ]
    observation_span_days = (
        (max(created_times) - min(created_times)).total_seconds() / 86_400.0
        if created_times
        else None
    )
    distinct_case_ids = {_case_id(row) for row in successful}
    distinct_market_dates = {timestamp.date().isoformat() for timestamp in market_times}

    regimes = {
        name: _error_summary(row for row in successful if row["market_regime"] == name)
        for name in policy.required_market_regimes
    }
    regions = {
        name: _error_summary(
            row for row in successful if row["moneyness_region"] == name
        )
        for name in policy.required_moneyness_regions
    }
    required_joint_names = tuple(
        f"{regime}:{region}"
        for regime in policy.required_market_regimes
        for region in policy.required_moneyness_regions
    )
    joint_slices = {
        name: _error_summary(
            row
            for row in successful
            if f"{row['market_regime']}:{row['moneyness_region']}" == name
        )
        for name in required_joint_names
    }
    overall = _error_summary(successful)
    distances = [
        float(row["maximum_standardized_feature_distance"])
        for row in successful
        if row.get("maximum_standardized_feature_distance") is not None
    ]
    above_four_sigma_fraction = (
        sum(value > 4.0 for value in distances) / len(distances) if distances else None
    )

    regime_counts = {
        name: int(summary["n_successful"]) for name, summary in regimes.items()
    }
    region_counts = {
        name: int(summary["n_successful"]) for name, summary in regions.items()
    }
    joint_counts = {
        name: int(summary["n_successful"]) for name, summary in joint_slices.items()
    }
    checks = {
        "minimum_observations": _minimum_check(
            n_observations,
            policy.minimum_observations,
            kind="evidence",
        ),
        "minimum_successful_observations": _minimum_check(
            len(successful),
            policy.minimum_successful_observations,
            kind="evidence",
        ),
        "minimum_distinct_cases": _minimum_check(
            len(distinct_case_ids),
            policy.minimum_distinct_cases,
            kind="evidence",
        ),
        "minimum_unique_symbols": _minimum_check(
            len({str(row["symbol"]) for row in successful}),
            policy.minimum_unique_symbols,
            kind="evidence",
        ),
        "minimum_distinct_market_dates": _minimum_check(
            len(distinct_market_dates),
            policy.minimum_distinct_market_dates,
            kind="evidence",
        ),
        "minimum_observation_span_days": _minimum_check(
            observation_span_days,
            policy.minimum_observation_span_days,
            kind="evidence",
        ),
        "market_regime_coverage": _minimum_check(
            min(regime_counts.values()) if regime_counts else 0,
            policy.minimum_successful_per_slice,
            kind="evidence",
            details=regime_counts,
        ),
        "moneyness_region_coverage": _minimum_check(
            min(region_counts.values()) if region_counts else 0,
            policy.minimum_successful_per_slice,
            kind="evidence",
            details=region_counts,
        ),
        "joint_slice_coverage": _minimum_check(
            min(joint_counts.values()) if joint_counts else 0,
            policy.minimum_successful_per_joint_slice,
            kind="evidence",
            details=joint_counts,
        ),
        "success_fraction": _minimum_check(
            status_fractions["success"],
            policy.minimum_success_fraction,
            kind="operations",
        ),
        "artifact_match_fraction": _minimum_check(
            artifact_match_fraction,
            policy.minimum_artifact_match_fraction,
            kind="integrity",
        ),
        "current_schema_fraction": _minimum_check(
            current_schema_fraction,
            policy.minimum_current_schema_fraction,
            kind="integrity",
        ),
        "out_of_domain_fraction": _maximum_check(
            status_fractions["out_of_domain"],
            policy.maximum_out_of_domain_fraction,
            kind="operations",
        ),
        "unavailable_fraction": _maximum_check(
            status_fractions["unavailable"],
            policy.maximum_unavailable_fraction,
            kind="operations",
        ),
        "error_fraction": _maximum_check(
            status_fractions["error"],
            policy.maximum_error_fraction,
            kind="operations",
        ),
        "mae": _maximum_check(
            overall["mae"],
            policy.maximum_mae,
            kind="quality",
        ),
        "p95_absolute_error": _maximum_check(
            overall["p95_absolute_error"],
            policy.maximum_p95_absolute_error,
            kind="quality",
        ),
        "within_two_reference_se_fraction": _minimum_check(
            overall["within_two_reference_se_fraction"],
            policy.minimum_within_two_reference_se_fraction,
            kind="quality",
        ),
        "maximum_regime_mae": _maximum_check(
            _worst_metric(regimes, policy.required_market_regimes, "mae"),
            policy.maximum_regime_mae,
            kind="quality",
        ),
        "maximum_moneyness_region_mae": _maximum_check(
            _worst_metric(regions, policy.required_moneyness_regions, "mae"),
            policy.maximum_moneyness_region_mae,
            kind="quality",
        ),
        "maximum_joint_slice_mae": _maximum_check(
            _worst_metric(joint_slices, required_joint_names, "mae"),
            policy.maximum_joint_slice_mae,
            kind="quality",
        ),
        "p95_latency_ms": _maximum_check(
            overall["p95_latency_ms"],
            policy.maximum_p95_latency_ms,
            kind="operations",
        ),
        "above_four_sigma_fraction": _maximum_check(
            above_four_sigma_fraction,
            policy.maximum_above_four_sigma_fraction,
            kind="drift",
        ),
    }
    evidence_sufficient = all(
        check["passed"] for check in checks.values() if check["kind"] == "evidence"
    )
    all_checks_passed = all(check["passed"] for check in checks.values())
    ready_for_review = evidence_sufficient and all_checks_passed
    if not evidence_sufficient:
        decision = "insufficient_evidence"
        next_action = "Collect broader, independent shadow observations."
    elif not all_checks_passed:
        decision = "not_ready"
        next_action = "Investigate failed gates; do not promote or tune on this report."
    else:
        decision = "ready_for_review"
        next_action = (
            "Freeze this evidence snapshot for independent human review. "
            "A separate artifact and runtime change is still required."
        )
    evidence_fields = (
        "observation_id",
        "created_at",
        "schema_version",
        "symbol",
        "market_data_time",
        "artifact_id",
        "target_artifact_id",
        "model_version",
        "status",
        "reference_price",
        "reference_standard_error",
        "surrogate_price",
        "absolute_error",
        "error_to_reference_standard_error",
        "latency_ms",
        "maximum_standardized_feature_distance",
        "market_regime",
        "moneyness_region",
        "contract_reference_spot",
        "market_payload",
        "terms_payload",
    )
    evidence_id = json_sha256(
        {
            "policy_id": policy.policy_id,
            "observations": sorted(
                ({name: row.get(name) for name in evidence_fields} for row in rows),
                key=lambda row: str(row["observation_id"]),
            ),
        }
    )
    return {
        "enabled": True,
        "available": True,
        "policy": policy.to_dict(),
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "evidence_id": evidence_id,
        "evidence": {
            "limit": limit,
            "n_observations": n_observations,
            "n_successful": len(successful),
            "n_distinct_cases": len(distinct_case_ids),
            "n_unique_symbols": len({str(row["symbol"]) for row in successful}),
            "n_distinct_market_dates": len(distinct_market_dates),
            "observation_span_days": observation_span_days,
            "oldest_observation_at": (
                min(created_times).isoformat() if created_times else None
            ),
            "newest_observation_at": (
                max(created_times).isoformat() if created_times else None
            ),
            "status_counts": status_counts,
            "status_fractions": status_fractions,
            "artifact_match_fraction": artifact_match_fraction,
            "current_schema_fraction": current_schema_fraction,
        },
        "metrics": {
            "overall": overall,
            "by_market_regime": regimes,
            "by_moneyness_region": regions,
            "by_joint_slice": joint_slices,
            "feature_drift": {
                "maximum_standardized_feature_distance": (
                    max(distances) if distances else None
                ),
                "above_four_sigma_fraction": above_four_sigma_fraction,
            },
        },
        "checks": checks,
        "decision": decision,
        "ready_for_review": ready_for_review,
        "runtime_eligible": False,
        "automatic_promotion_permitted": False,
        "next_action": next_action,
    }
