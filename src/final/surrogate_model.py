import hashlib
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


class SurrogateModelError(ValueError):
    pass


MAX_ARTIFACT_ARRAY_BYTES = 64 * 1024 * 1024
MAX_ARTIFACT_MEMBERS = 40


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def json_sha256(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


@dataclass(frozen=True)
class NumpyMLPSurrogate:
    feature_names: tuple[str, ...]
    output_names: tuple[str, ...]
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    target_mean: np.ndarray
    target_scale: np.ndarray
    weights: tuple[np.ndarray, ...]
    biases: tuple[np.ndarray, ...]

    def __post_init__(self) -> None:
        n_features = len(self.feature_names)
        if n_features < 1:
            raise SurrogateModelError("surrogate feature names cannot be empty")
        n_outputs = len(self.output_names)
        if n_outputs < 1 or len(set(self.output_names)) != n_outputs:
            raise SurrogateModelError("surrogate output names are invalid")
        if self.feature_mean.shape != (n_features,) or self.feature_scale.shape != (
            n_features,
        ):
            raise SurrogateModelError("surrogate feature scaling shape mismatch")
        if not np.all(np.isfinite(self.feature_mean)) or not np.all(
            np.isfinite(self.feature_scale)
        ):
            raise SurrogateModelError("surrogate feature scaling must be finite")
        if np.any(self.feature_scale <= 0.0):
            raise SurrogateModelError("surrogate feature scales must be positive")
        if self.target_mean.shape != (n_outputs,) or self.target_scale.shape != (
            n_outputs,
        ):
            raise SurrogateModelError("surrogate target scaling shape mismatch")
        if not np.all(np.isfinite(self.target_mean)) or not np.all(
            np.isfinite(self.target_scale)
        ):
            raise SurrogateModelError("surrogate target scaling must be finite")
        if np.any(self.target_scale <= 0.0):
            raise SurrogateModelError("surrogate target scale must be positive")
        if not self.weights or len(self.weights) != len(self.biases):
            raise SurrogateModelError("surrogate layers are missing")
        previous_width = n_features
        for layer_index, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            if weight.ndim != 2 or bias.ndim != 1:
                raise SurrogateModelError("surrogate layer arrays have invalid rank")
            if weight.shape[0] != previous_width or weight.shape[1] != bias.shape[0]:
                raise SurrogateModelError(
                    f"surrogate layer {layer_index} shape mismatch"
                )
            if not np.all(np.isfinite(weight)) or not np.all(np.isfinite(bias)):
                raise SurrogateModelError("surrogate layer values must be finite")
            previous_width = weight.shape[1]
        if previous_width != n_outputs:
            raise SurrogateModelError("surrogate output layer width is invalid")

    def predict_raw_outputs(self, features: Sequence[float] | np.ndarray) -> np.ndarray:
        """Return unconstrained linear-head outputs for offline diagnostics."""
        values = np.asarray(features, dtype=np.float64)
        single = values.ndim == 1
        if single:
            values = values.reshape(1, -1)
        if values.ndim != 2 or values.shape[1] != len(self.feature_names):
            raise SurrogateModelError("surrogate prediction feature shape mismatch")
        if not np.all(np.isfinite(values)):
            raise SurrogateModelError("surrogate prediction features must be finite")
        hidden = (values - self.feature_mean) / self.feature_scale
        for layer_index, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            hidden = hidden @ weight + bias
            if layer_index < len(self.weights) - 1:
                hidden = np.maximum(hidden, 0.0)
        predictions = hidden * self.target_scale + self.target_mean
        return predictions[0:1] if single else predictions

    def predict_outputs(self, features: Sequence[float] | np.ndarray) -> np.ndarray:
        """Return outputs projected onto their known financial domains.

        Projection is the least-squares solution over each feasible interval.
        It cannot increase error when the true component is non-negative and
        the true event probability lies in [0, 1]. The raw values remain
        available so audit reports can detect a network that relies too heavily
        on this safety constraint.
        """
        predictions = self.predict_raw_outputs(features).copy()
        for name in (
            "coupon_pv",
            "autocall_principal_pv",
            "maturity_protected_pv",
            "maturity_downside_pv",
        ):
            if name in self.output_names:
                predictions[:, self.output_names.index(name)] = np.maximum(
                    predictions[:, self.output_names.index(name)], 0.0
                )
        for name in ("autocall_probability", "downside_probability"):
            if name in self.output_names:
                predictions[:, self.output_names.index(name)] = np.clip(
                    predictions[:, self.output_names.index(name)], 0.0, 1.0
                )
        return predictions

    def predict(self, features: Sequence[float] | np.ndarray) -> np.ndarray:
        """Return price predictions from a direct or payoff-aware output head."""
        outputs = self.predict_outputs(features)
        if "price" in self.output_names:
            return outputs[:, self.output_names.index("price")]
        component_names = (
            "coupon_pv",
            "autocall_principal_pv",
            "maturity_protected_pv",
            "maturity_downside_pv",
        )
        if not all(name in self.output_names for name in component_names):
            raise SurrogateModelError(
                "surrogate outputs do not define a price reconstruction"
            )
        indices = [self.output_names.index(name) for name in component_names]
        return np.sum(outputs[:, indices], axis=1)


@dataclass(frozen=True)
class NumpyBranchedMLPSurrogate:
    """Pure-NumPy inference for the price-first trunk and its three heads."""

    feature_names: tuple[str, ...]
    output_names: tuple[str, ...]
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    target_mean: np.ndarray
    target_scale: np.ndarray
    trunk_weights: tuple[np.ndarray, ...]
    trunk_biases: tuple[np.ndarray, ...]
    price_weight: np.ndarray
    price_bias: np.ndarray
    payoff_weights: tuple[np.ndarray, ...]
    payoff_biases: tuple[np.ndarray, ...]
    event_weights: tuple[np.ndarray, ...]
    event_biases: tuple[np.ndarray, ...]

    def __post_init__(self) -> None:
        n_features = len(self.feature_names)
        n_outputs = len(self.output_names)
        if n_features < 1:
            raise SurrogateModelError("surrogate feature names cannot be empty")
        if n_outputs < 3 or len(set(self.output_names)) != n_outputs:
            raise SurrogateModelError("surrogate output names are invalid")
        if self.output_names[0] != "price":
            raise SurrogateModelError("branched surrogate must expose price first")
        self._validate_scale(
            self.feature_mean,
            self.feature_scale,
            n_features,
            "feature",
        )
        self._validate_scale(
            self.target_mean,
            self.target_scale,
            n_outputs,
            "target",
        )
        trunk_width = self._validate_layers(
            self.trunk_weights,
            self.trunk_biases,
            n_features,
            "trunk",
        )
        if self.price_weight.shape != (trunk_width, 1) or self.price_bias.shape != (1,):
            raise SurrogateModelError("branched surrogate price head shape mismatch")
        if not np.all(np.isfinite(self.price_weight)) or not np.all(
            np.isfinite(self.price_bias)
        ):
            raise SurrogateModelError("branched surrogate price head must be finite")
        payoff_width = self._validate_layers(
            self.payoff_weights,
            self.payoff_biases,
            trunk_width,
            "payoff head",
        )
        event_width = self._validate_layers(
            self.event_weights,
            self.event_biases,
            trunk_width,
            "event head",
        )
        if 1 + payoff_width + event_width != n_outputs:
            raise SurrogateModelError("branched surrogate output width is invalid")

    @staticmethod
    def _validate_scale(
        mean: np.ndarray,
        scale: np.ndarray,
        width: int,
        label: str,
    ) -> None:
        if mean.shape != (width,) or scale.shape != (width,):
            raise SurrogateModelError(
                f"branched surrogate {label} scaling shape mismatch"
            )
        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(scale)):
            raise SurrogateModelError(
                f"branched surrogate {label} scaling must be finite"
            )
        if np.any(scale <= 0.0):
            raise SurrogateModelError(
                f"branched surrogate {label} scales must be positive"
            )

    @staticmethod
    def _validate_layers(
        weights: tuple[np.ndarray, ...],
        biases: tuple[np.ndarray, ...],
        input_width: int,
        label: str,
    ) -> int:
        if not weights or len(weights) != len(biases):
            raise SurrogateModelError(f"branched surrogate {label} is missing")
        previous_width = input_width
        for layer_index, (weight, bias) in enumerate(zip(weights, biases)):
            if (
                weight.ndim != 2
                or bias.ndim != 1
                or weight.shape[0] != previous_width
                or weight.shape[1] != bias.shape[0]
            ):
                raise SurrogateModelError(
                    f"branched surrogate {label} layer {layer_index} shape mismatch"
                )
            if not np.all(np.isfinite(weight)) or not np.all(np.isfinite(bias)):
                raise SurrogateModelError(
                    f"branched surrogate {label} values must be finite"
                )
            previous_width = weight.shape[1]
        return previous_width

    @staticmethod
    def _forward_layers(
        values: np.ndarray,
        weights: tuple[np.ndarray, ...],
        biases: tuple[np.ndarray, ...],
        *,
        final_relu: bool,
    ) -> np.ndarray:
        hidden = values
        for layer_index, (weight, bias) in enumerate(zip(weights, biases)):
            hidden = hidden @ weight + bias
            if final_relu or layer_index < len(weights) - 1:
                hidden = np.maximum(hidden, 0.0)
        return hidden

    def predict_raw_outputs(self, features: Sequence[float] | np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=np.float64)
        single = values.ndim == 1
        if single:
            values = values.reshape(1, -1)
        if values.ndim != 2 or values.shape[1] != len(self.feature_names):
            raise SurrogateModelError("surrogate prediction feature shape mismatch")
        if not np.all(np.isfinite(values)):
            raise SurrogateModelError("surrogate prediction features must be finite")
        standardized = ((values - self.feature_mean) / self.feature_scale).astype(
            np.float32
        )
        trunk = self._forward_layers(
            standardized,
            self.trunk_weights,
            self.trunk_biases,
            final_relu=True,
        )
        price = trunk @ self.price_weight + self.price_bias
        payoff = self._forward_layers(
            trunk,
            self.payoff_weights,
            self.payoff_biases,
            final_relu=False,
        )
        events = self._forward_layers(
            trunk,
            self.event_weights,
            self.event_biases,
            final_relu=False,
        )
        normalized = np.column_stack((price, payoff, events))
        predictions = normalized * self.target_scale + self.target_mean
        return predictions[0:1] if single else predictions

    def predict_outputs(self, features: Sequence[float] | np.ndarray) -> np.ndarray:
        predictions = self.predict_raw_outputs(features).copy()
        nonnegative = (
            "coupon_pv",
            "autocall_principal_pv",
            "maturity_protected_pv",
            "maturity_downside_pv",
            "expected_coupon_count",
            "early_coupon_mass",
            "late_coupon_mass",
        )
        bounded = (
            "autocall_probability",
            "downside_probability",
            "conditional_expected_autocall_time_fraction",
            "conditional_autocall_time_variance",
            "final_survival_probability",
        )
        for name in nonnegative:
            if name in self.output_names:
                index = self.output_names.index(name)
                predictions[:, index] = np.maximum(predictions[:, index], 0.0)
        for name in bounded:
            if name in self.output_names:
                index = self.output_names.index(name)
                predictions[:, index] = np.clip(
                    predictions[:, index],
                    0.0,
                    1.0,
                )
        return predictions

    def predict(self, features: Sequence[float] | np.ndarray) -> np.ndarray:
        return self.predict_outputs(features)[:, 0]


def load_numpy_mlp_artifact(
    weights_path: Path,
    feature_names: Sequence[str],
    output_names: Sequence[str],
) -> NumpyMLPSurrogate:
    try:
        with zipfile.ZipFile(Path(weights_path)) as archive:
            members = archive.infolist()
            if (
                len(members) > MAX_ARTIFACT_MEMBERS
                or sum(member.file_size for member in members)
                > MAX_ARTIFACT_ARRAY_BYTES
            ):
                raise SurrogateModelError(
                    "surrogate weights exceed the expanded size limit"
                )
        with np.load(Path(weights_path), allow_pickle=False) as data:
            n_layers = int(data["n_layers"].item())
            if n_layers < 1 or n_layers > 16:
                raise SurrogateModelError("surrogate layer count is invalid")
            return NumpyMLPSurrogate(
                feature_names=tuple(str(name) for name in feature_names),
                output_names=tuple(str(name) for name in output_names),
                feature_mean=np.asarray(data["feature_mean"], dtype=np.float64),
                feature_scale=np.asarray(data["feature_scale"], dtype=np.float64),
                target_mean=np.asarray(data["target_mean"], dtype=np.float64),
                target_scale=np.asarray(data["target_scale"], dtype=np.float64),
                weights=tuple(
                    np.asarray(data[f"weight_{index}"], dtype=np.float64)
                    for index in range(n_layers)
                ),
                biases=tuple(
                    np.asarray(data[f"bias_{index}"], dtype=np.float64)
                    for index in range(n_layers)
                ),
            )
    except (OSError, KeyError, ValueError, zipfile.BadZipFile) as exc:
        if isinstance(exc, SurrogateModelError):
            raise
        raise SurrogateModelError("surrogate weights load failed") from exc


def load_numpy_branched_mlp_artifact(
    weights_path: Path,
    feature_names: Sequence[str],
    output_names: Sequence[str],
) -> NumpyBranchedMLPSurrogate:
    try:
        with zipfile.ZipFile(Path(weights_path)) as archive:
            members = archive.infolist()
            if (
                len(members) > MAX_ARTIFACT_MEMBERS
                or sum(member.file_size for member in members)
                > MAX_ARTIFACT_ARRAY_BYTES
            ):
                raise SurrogateModelError(
                    "surrogate weights exceed the expanded size limit"
                )
        with np.load(Path(weights_path), allow_pickle=False) as data:
            trunk_layers = int(data["trunk_layers"].item())
            payoff_layers = int(data["payoff_layers"].item())
            event_layers = int(data["event_layers"].item())
            if (
                trunk_layers < 1
                or trunk_layers > 16
                or payoff_layers < 1
                or payoff_layers > 8
                or event_layers < 1
                or event_layers > 8
            ):
                raise SurrogateModelError("branched surrogate layer count is invalid")
            return NumpyBranchedMLPSurrogate(
                feature_names=tuple(str(name) for name in feature_names),
                output_names=tuple(str(name) for name in output_names),
                feature_mean=np.asarray(data["feature_mean"], dtype=np.float64),
                feature_scale=np.asarray(data["feature_scale"], dtype=np.float64),
                target_mean=np.asarray(data["target_mean"], dtype=np.float64),
                target_scale=np.asarray(data["target_scale"], dtype=np.float64),
                trunk_weights=tuple(
                    np.asarray(data[f"trunk_weight_{index}"], dtype=np.float32)
                    for index in range(trunk_layers)
                ),
                trunk_biases=tuple(
                    np.asarray(data[f"trunk_bias_{index}"], dtype=np.float32)
                    for index in range(trunk_layers)
                ),
                price_weight=np.asarray(data["price_weight"], dtype=np.float32),
                price_bias=np.asarray(data["price_bias"], dtype=np.float32),
                payoff_weights=tuple(
                    np.asarray(data[f"payoff_weight_{index}"], dtype=np.float32)
                    for index in range(payoff_layers)
                ),
                payoff_biases=tuple(
                    np.asarray(data[f"payoff_bias_{index}"], dtype=np.float32)
                    for index in range(payoff_layers)
                ),
                event_weights=tuple(
                    np.asarray(data[f"event_weight_{index}"], dtype=np.float32)
                    for index in range(event_layers)
                ),
                event_biases=tuple(
                    np.asarray(data[f"event_bias_{index}"], dtype=np.float32)
                    for index in range(event_layers)
                ),
            )
    except (OSError, KeyError, ValueError, zipfile.BadZipFile) as exc:
        if isinstance(exc, SurrogateModelError):
            raise
        raise SurrogateModelError("surrogate weights load failed") from exc
