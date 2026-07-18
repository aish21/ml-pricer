import hashlib
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .payoffs import PhoenixPayoff
from .reference_pricer import phoenix_piecewise_observation_event_ledger
from .surrogate_contract import (
    DEFAULT_TRAINING_DOMAIN,
    PHOENIX_SURROGATE_FEATURE_NAMES,
    reconstruct_phoenix_surrogate_case,
)
from .surrogate_data import PhoenixSurrogateDataset, _standard_normal_shocks


PHOENIX_HAZARD_DATASET_SCHEMA_VERSION = "phoenix-hazard-dataset-v1"
PHOENIX_HAZARD_LABEL_VERSION = "phoenix-observation-hazard-label-v1"
PHOENIX_HAZARD_MAX_OBSERVATIONS = int(DEFAULT_TRAINING_DOMAIN["obs_count"][1])


class PhoenixHazardDatasetError(ValueError):
    pass


def phoenix_observation_schedule(
    features: np.ndarray,
    *,
    n_steps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reconstruct observation masks, discounts, and maturity discounts."""
    feature_matrix = np.asarray(features, dtype=np.float64)
    single = feature_matrix.ndim == 1
    if single:
        feature_matrix = feature_matrix.reshape(1, -1)
    if feature_matrix.ndim != 2 or feature_matrix.shape[1] != len(
        PHOENIX_SURROGATE_FEATURE_NAMES
    ):
        raise PhoenixHazardDatasetError("hazard features have invalid shape")
    if not np.all(np.isfinite(feature_matrix)):
        raise PhoenixHazardDatasetError("hazard features must be finite")
    if n_steps < PHOENIX_HAZARD_MAX_OBSERVATIONS:
        raise PhoenixHazardDatasetError("hazard path grid has too few steps")

    n_samples = len(feature_matrix)
    mask = np.zeros((n_samples, PHOENIX_HAZARD_MAX_OBSERVATIONS), dtype=bool)
    discounts = np.zeros_like(mask, dtype=np.float64)
    maturity_discounts = np.zeros(n_samples, dtype=np.float64)
    for row_index, row in enumerate(feature_matrix):
        market, terms, _ = reconstruct_phoenix_surrogate_case(row)
        observation_count = int(terms["obs_count"])
        observation_indices = np.linspace(0, n_steps, observation_count + 1, dtype=int)[
            1:
        ]
        observation_times = (
            observation_indices / float(n_steps) * terms["maturity_years"]
        )
        mask[row_index, :observation_count] = True
        discounts[row_index, :observation_count] = [
            market.discount_factor(float(time_years))
            for time_years in observation_times
        ]
        maturity_discounts[row_index] = market.discount_factor(terms["maturity_years"])
    if single:
        return mask[0], discounts[0], maturity_discounts[0:1]
    return mask, discounts, maturity_discounts


def reconstruct_hazard_prices(
    *,
    features: np.ndarray,
    observation_mask: np.ndarray,
    coupon_probability: np.ndarray,
    first_autocall_probability: np.ndarray,
    protected_probability: np.ndarray,
    downside_probability: np.ndarray,
    downside_conditional_recovery: np.ndarray,
    n_steps: int,
) -> np.ndarray:
    """Reconstruct price from observation events and terminal recovery."""
    feature_matrix = np.asarray(features, dtype=np.float64)
    if feature_matrix.ndim == 1:
        feature_matrix = feature_matrix.reshape(1, -1)
    expected_event_shape = (
        len(feature_matrix),
        PHOENIX_HAZARD_MAX_OBSERVATIONS,
    )
    mask = np.asarray(observation_mask, dtype=bool)
    coupon = np.asarray(coupon_probability, dtype=np.float64)
    autocall = np.asarray(first_autocall_probability, dtype=np.float64)
    if (
        mask.shape != expected_event_shape
        or coupon.shape != expected_event_shape
        or autocall.shape != expected_event_shape
    ):
        raise PhoenixHazardDatasetError("hazard observation arrays have invalid shape")
    terminal_arrays = [
        np.asarray(values, dtype=np.float64)
        for values in (
            protected_probability,
            downside_probability,
            downside_conditional_recovery,
        )
    ]
    if any(values.shape != (len(feature_matrix),) for values in terminal_arrays):
        raise PhoenixHazardDatasetError("hazard terminal arrays have invalid shape")
    if not all(
        np.all(np.isfinite(values)) for values in (coupon, autocall, *terminal_arrays)
    ):
        raise PhoenixHazardDatasetError("hazard reconstruction inputs must be finite")

    schedule_mask, discounts, maturity_discounts = phoenix_observation_schedule(
        feature_matrix,
        n_steps=n_steps,
    )
    if not np.array_equal(mask, schedule_mask):
        raise PhoenixHazardDatasetError(
            "hazard observation mask disagrees with contract terms"
        )
    coupon_rate_index = PHOENIX_SURROGATE_FEATURE_NAMES.index("coupon_rate")
    coupon_rates = feature_matrix[:, coupon_rate_index]
    coupon_pv = coupon_rates * np.sum(coupon * discounts, axis=1)
    autocall_pv = np.sum(autocall * discounts, axis=1)
    protected_pv = terminal_arrays[0] * maturity_discounts
    downside_pv = terminal_arrays[1] * terminal_arrays[2] * maturity_discounts
    return coupon_pv + autocall_pv + protected_pv + downside_pv


@dataclass(frozen=True)
class PhoenixHazardDataset:
    base: PhoenixSurrogateDataset
    observation_mask: np.ndarray
    coupon_probability: np.ndarray
    first_autocall_probability: np.ndarray
    survival_after_probability: np.ndarray
    protected_probability: np.ndarray
    downside_probability: np.ndarray
    downside_conditional_recovery: np.ndarray
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        n_samples = len(self.base.y)
        observation_shape = (n_samples, PHOENIX_HAZARD_MAX_OBSERVATIONS)
        for name, values in (
            ("observation_mask", self.observation_mask),
            ("coupon_probability", self.coupon_probability),
            ("first_autocall_probability", self.first_autocall_probability),
            ("survival_after_probability", self.survival_after_probability),
        ):
            if values.shape != observation_shape:
                raise PhoenixHazardDatasetError(f"{name} has invalid shape")
        for name, values in (
            ("protected_probability", self.protected_probability),
            ("downside_probability", self.downside_probability),
            ("downside_conditional_recovery", self.downside_conditional_recovery),
        ):
            if values.shape != (n_samples,):
                raise PhoenixHazardDatasetError(f"{name} has invalid shape")
        numeric_arrays = (
            self.coupon_probability,
            self.first_autocall_probability,
            self.survival_after_probability,
            self.protected_probability,
            self.downside_probability,
            self.downside_conditional_recovery,
        )
        if not all(np.all(np.isfinite(values)) for values in numeric_arrays):
            raise PhoenixHazardDatasetError("hazard targets must be finite")
        if any(np.any((values < 0.0) | (values > 1.0)) for values in numeric_arrays):
            raise PhoenixHazardDatasetError("hazard targets violate probability bounds")
        inactive = ~self.observation_mask
        if any(
            np.any(values[inactive] != 0.0)
            for values in (
                self.coupon_probability,
                self.first_autocall_probability,
                self.survival_after_probability,
            )
        ):
            raise PhoenixHazardDatasetError(
                "inactive observations must have zero targets"
            )
        expected_survival = 1.0 - np.cumsum(self.first_autocall_probability, axis=1)
        if not np.allclose(
            self.survival_after_probability[self.observation_mask],
            expected_survival[self.observation_mask],
            rtol=0.0,
            atol=1e-12,
        ):
            raise PhoenixHazardDatasetError(
                "survival labels do not reconcile to first-autocall labels"
            )
        if np.any(self.coupon_probability + 1e-12 < self.first_autocall_probability):
            raise PhoenixHazardDatasetError(
                "autocall events must also receive the current coupon"
            )
        last_indices = np.sum(self.observation_mask, axis=1) - 1
        final_survival = self.survival_after_probability[
            np.arange(n_samples), last_indices
        ]
        if not np.allclose(
            final_survival,
            self.protected_probability + self.downside_probability,
            rtol=0.0,
            atol=1e-12,
        ):
            raise PhoenixHazardDatasetError(
                "terminal labels do not reconcile to final survival"
            )
        expected_metadata = {
            "dataset_schema_version": PHOENIX_HAZARD_DATASET_SCHEMA_VERSION,
            "label_schema_version": PHOENIX_HAZARD_LABEL_VERSION,
            "base_dataset_id": self.base.metadata["dataset_id"],
            "n_samples": n_samples,
        }
        for name, expected in expected_metadata.items():
            if self.metadata.get(name) != expected:
                raise PhoenixHazardDatasetError(
                    f"hazard metadata is inconsistent: {name}"
                )
        reconstructed = reconstruct_hazard_prices(
            features=self.base.X,
            observation_mask=self.observation_mask,
            coupon_probability=self.coupon_probability,
            first_autocall_probability=self.first_autocall_probability,
            protected_probability=self.protected_probability,
            downside_probability=self.downside_probability,
            downside_conditional_recovery=self.downside_conditional_recovery,
            n_steps=int(self.metadata["n_steps"]),
        )
        if not np.allclose(reconstructed, self.base.y, rtol=0.0, atol=1e-10):
            raise PhoenixHazardDatasetError(
                "hazard labels do not reconstruct base prices"
            )


def _hazard_dataset_id(
    *,
    base_dataset_id: str,
    observation_mask: np.ndarray,
    coupon_probability: np.ndarray,
    first_autocall_probability: np.ndarray,
    survival_after_probability: np.ndarray,
    protected_probability: np.ndarray,
    downside_probability: np.ndarray,
    downside_conditional_recovery: np.ndarray,
) -> str:
    digest = hashlib.sha256()
    digest.update(PHOENIX_HAZARD_DATASET_SCHEMA_VERSION.encode("utf-8"))
    digest.update(PHOENIX_HAZARD_LABEL_VERSION.encode("utf-8"))
    digest.update(base_dataset_id.encode("utf-8"))
    for values in (
        observation_mask,
        coupon_probability,
        first_autocall_probability,
        survival_after_probability,
        protected_probability,
        downside_probability,
        downside_conditional_recovery,
    ):
        digest.update(np.ascontiguousarray(values).tobytes())
    return f"sha256:{digest.hexdigest()}"


def _label_hazard_sample(
    payload: tuple[int, np.ndarray, int, int, int, int, str],
) -> tuple[
    int,
    int,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
    float,
    float,
]:
    (
        sample_index,
        features,
        n_steps,
        paths_per_replication,
        label_replications,
        label_seed,
        sampling_method,
    ) = payload
    payoff = PhoenixPayoff()
    market, terms, reference_spot = reconstruct_phoenix_surrogate_case(features)
    observation_count = int(terms["obs_count"])
    params = {
        "S0": reference_spot,
        "r": 0.0,
        "sigma": 1.0,
        "T": terms["maturity_years"],
        "autocall_barrier_frac": terms["autocall_barrier_frac"],
        "coupon_barrier_frac": terms["coupon_barrier_frac"],
        "coupon_rate": terms["coupon_rate"],
        "knock_in_frac": terms["knock_in_frac"],
        "obs_count": observation_count,
    }
    replication_coupon = []
    replication_autocall = []
    replication_survival = []
    replication_protected = []
    replication_downside = []
    replication_recovery_numerator = []
    for replication in range(label_replications):
        seed = (label_seed + sample_index * 104_729 + replication * 7_919) % (2**32 - 1)
        shocks = _standard_normal_shocks(
            method=sampling_method,
            n_paths=paths_per_replication,
            n_steps=n_steps,
            seed=seed,
        )
        ledger = phoenix_piecewise_observation_event_ledger(
            payoff=payoff,
            params=params,
            market=market,
            n_paths=paths_per_replication,
            n_steps=n_steps,
            seed=None,
            standard_normal_shocks=shocks,
        )
        replication_coupon.append(np.mean(ledger["coupon_event"], axis=0))
        replication_autocall.append(np.mean(ledger["first_autocall_event"], axis=0))
        replication_survival.append(
            np.mean(ledger["survival_after_observation"], axis=0)
        )
        replication_protected.append(float(np.mean(ledger["protected_maturity_event"])))
        replication_downside.append(float(np.mean(ledger["downside_maturity_event"])))
        replication_recovery_numerator.append(
            float(np.mean(ledger["downside_recovery_ratio"]))
        )
    observed_downside = float(np.mean(replication_downside))
    recovery_numerator = float(np.mean(replication_recovery_numerator))
    return (
        sample_index,
        observation_count,
        np.mean(replication_coupon, axis=0),
        np.mean(replication_autocall, axis=0),
        np.mean(replication_survival, axis=0),
        float(np.mean(replication_protected)),
        observed_downside,
        recovery_numerator / observed_downside if observed_downside > 0.0 else 0.0,
    )


def generate_phoenix_hazard_dataset(
    base: PhoenixSurrogateDataset,
    *,
    verbose: bool = True,
    workers: int = 1,
) -> PhoenixHazardDataset:
    """Replay a development dataset's paths into observation-level labels."""
    if base.metadata.get("dataset_role") != "development":
        raise PhoenixHazardDatasetError(
            "hazard research labels require a development dataset"
        )
    config = base.metadata.get("config")
    if not isinstance(config, dict):
        raise PhoenixHazardDatasetError("base dataset generation config is missing")
    try:
        n_steps = int(config["n_steps"])
        paths_per_replication = int(config["paths_per_replication"])
        label_replications = int(config["label_replications"])
        label_seed = int(config["label_seed"])
        sampling_method = str(config["sampling_method"])
    except (KeyError, TypeError, ValueError) as exc:
        raise PhoenixHazardDatasetError(
            "base dataset generation config is invalid"
        ) from exc
    if workers < 1 or workers > 32:
        raise PhoenixHazardDatasetError("hazard generation workers are invalid")

    n_samples = len(base.y)
    shape = (n_samples, PHOENIX_HAZARD_MAX_OBSERVATIONS)
    observation_mask = np.zeros(shape, dtype=bool)
    coupon_probability = np.zeros(shape, dtype=np.float64)
    first_autocall_probability = np.zeros(shape, dtype=np.float64)
    survival_after_probability = np.zeros(shape, dtype=np.float64)
    protected_probability = np.zeros(n_samples, dtype=np.float64)
    downside_probability = np.zeros(n_samples, dtype=np.float64)
    downside_conditional_recovery = np.zeros(n_samples, dtype=np.float64)
    started = time.perf_counter()

    payloads = (
        (
            sample_index,
            features,
            n_steps,
            paths_per_replication,
            label_replications,
            label_seed,
            sampling_method,
        )
        for sample_index, features in enumerate(base.X)
    )
    executor = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
    results = (
        executor.map(_label_hazard_sample, payloads, chunksize=4)
        if executor is not None
        else map(_label_hazard_sample, payloads)
    )
    try:
        for completed, result in enumerate(results, start=1):
            (
                sample_index,
                observation_count,
                observed_coupon,
                observed_autocall,
                observed_survival,
                observed_protected,
                observed_downside,
                observed_recovery,
            ) = result
            observation_mask[sample_index, :observation_count] = True
            coupon_probability[sample_index, :observation_count] = observed_coupon
            first_autocall_probability[sample_index, :observation_count] = (
                observed_autocall
            )
            survival_after_probability[sample_index, :observation_count] = (
                observed_survival
            )
            protected_probability[sample_index] = observed_protected
            downside_probability[sample_index] = observed_downside
            downside_conditional_recovery[sample_index] = observed_recovery
            if verbose and (
                completed == n_samples or completed % max(1, n_samples // 20) == 0
            ):
                print(
                    f"[PhoenixHazardData] {completed}/{n_samples} labels complete",
                    flush=True,
                )
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)

    dataset_id = _hazard_dataset_id(
        base_dataset_id=base.metadata["dataset_id"],
        observation_mask=observation_mask,
        coupon_probability=coupon_probability,
        first_autocall_probability=first_autocall_probability,
        survival_after_probability=survival_after_probability,
        protected_probability=protected_probability,
        downside_probability=downside_probability,
        downside_conditional_recovery=downside_conditional_recovery,
    )
    metadata = {
        "dataset_schema_version": PHOENIX_HAZARD_DATASET_SCHEMA_VERSION,
        "label_schema_version": PHOENIX_HAZARD_LABEL_VERSION,
        "dataset_id": dataset_id,
        "base_dataset_id": base.metadata["dataset_id"],
        "n_samples": n_samples,
        "n_steps": n_steps,
        "maximum_observations": PHOENIX_HAZARD_MAX_OBSERVATIONS,
        "generation_workers": workers,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "generation_seconds": time.perf_counter() - started,
        "source_randomization": {
            "label_seed": label_seed,
            "label_replications": label_replications,
            "paths_per_replication": paths_per_replication,
            "sampling_method": sampling_method,
        },
    }
    return PhoenixHazardDataset(
        base=base,
        observation_mask=observation_mask,
        coupon_probability=coupon_probability,
        first_autocall_probability=first_autocall_probability,
        survival_after_probability=survival_after_probability,
        protected_probability=protected_probability,
        downside_probability=downside_probability,
        downside_conditional_recovery=downside_conditional_recovery,
        metadata=metadata,
    )


def save_phoenix_hazard_dataset(
    dataset: PhoenixHazardDataset,
    output_path: Path,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            observation_mask=dataset.observation_mask,
            coupon_probability=dataset.coupon_probability,
            first_autocall_probability=dataset.first_autocall_probability,
            survival_after_probability=dataset.survival_after_probability,
            protected_probability=dataset.protected_probability,
            downside_probability=dataset.downside_probability,
            downside_conditional_recovery=dataset.downside_conditional_recovery,
            metadata_json=np.asarray(
                json.dumps(dataset.metadata, sort_keys=True), dtype=np.str_
            ),
        )
    os.replace(temporary, path)


def load_phoenix_hazard_dataset(
    input_path: Path,
    *,
    base: PhoenixSurrogateDataset,
) -> PhoenixHazardDataset:
    try:
        with np.load(Path(input_path), allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata_json"].item()))
            dataset = PhoenixHazardDataset(
                base=base,
                observation_mask=np.asarray(data["observation_mask"], dtype=bool),
                coupon_probability=np.asarray(
                    data["coupon_probability"], dtype=np.float64
                ),
                first_autocall_probability=np.asarray(
                    data["first_autocall_probability"], dtype=np.float64
                ),
                survival_after_probability=np.asarray(
                    data["survival_after_probability"], dtype=np.float64
                ),
                protected_probability=np.asarray(
                    data["protected_probability"], dtype=np.float64
                ),
                downside_probability=np.asarray(
                    data["downside_probability"], dtype=np.float64
                ),
                downside_conditional_recovery=np.asarray(
                    data["downside_conditional_recovery"], dtype=np.float64
                ),
                metadata=metadata,
            )
    except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
        raise PhoenixHazardDatasetError("hazard dataset load failed") from exc
    observed_id = _hazard_dataset_id(
        base_dataset_id=dataset.metadata["base_dataset_id"],
        observation_mask=dataset.observation_mask,
        coupon_probability=dataset.coupon_probability,
        first_autocall_probability=dataset.first_autocall_probability,
        survival_after_probability=dataset.survival_after_probability,
        protected_probability=dataset.protected_probability,
        downside_probability=dataset.downside_probability,
        downside_conditional_recovery=dataset.downside_conditional_recovery,
    )
    if observed_id != dataset.metadata.get("dataset_id"):
        raise PhoenixHazardDatasetError("hazard dataset fingerprint mismatch")
    return dataset
