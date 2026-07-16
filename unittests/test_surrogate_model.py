import numpy as np
import pytest

from src.final.surrogate_model import (
    NumpyMLPSurrogate,
    SurrogateModelError,
    load_numpy_mlp_artifact,
)


def test_numpy_mlp_predicts_without_training_runtime_dependencies(tmp_path):
    path = tmp_path / "weights.npz"
    np.savez_compressed(
        path,
        feature_mean=np.zeros(2),
        feature_scale=np.ones(2),
        target_mean=np.asarray([1.0]),
        target_scale=np.asarray([2.0]),
        n_layers=np.asarray(1),
        weight_0=np.asarray([[1.0], [2.0]]),
        bias_0=np.asarray([0.0]),
    )
    model = load_numpy_mlp_artifact(path, ["a", "b"], ["price"])
    assert model.predict([3.0, 4.0])[0] == pytest.approx(23.0)


def test_numpy_mlp_rejects_bad_layer_shapes():
    with pytest.raises(SurrogateModelError, match="shape mismatch"):
        NumpyMLPSurrogate(
            feature_names=("a", "b"),
            output_names=("price",),
            feature_mean=np.zeros(2),
            feature_scale=np.ones(2),
            target_mean=np.zeros(1),
            target_scale=np.ones(1),
            weights=(np.zeros((3, 1)),),
            biases=(np.zeros(1),),
        )


def test_payoff_aware_outputs_are_projected_and_reconstruct_price():
    model = NumpyMLPSurrogate(
        feature_names=("x",),
        output_names=(
            "coupon_pv",
            "autocall_principal_pv",
            "maturity_protected_pv",
            "maturity_downside_pv",
            "autocall_probability",
            "downside_probability",
        ),
        feature_mean=np.zeros(1),
        feature_scale=np.ones(1),
        target_mean=np.zeros(6),
        target_scale=np.ones(6),
        weights=(np.zeros((1, 6)),),
        biases=(np.asarray([0.1, 0.5, -0.3, 0.05, 1.7, -0.2]),),
    )

    assert model.predict([1.0])[0] == pytest.approx(0.65)
    assert model.predict_outputs([1.0])[0].tolist() == pytest.approx(
        [0.1, 0.5, 0.0, 0.05, 1.0, 0.0]
    )
    assert model.predict_raw_outputs([1.0])[0].tolist() == pytest.approx(
        [0.1, 0.5, -0.3, 0.05, 1.7, -0.2]
    )
