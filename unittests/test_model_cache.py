import json

import pytest

from app.services.model_cache import (
    ModelCacheArtifactError,
    ModelCacheProductError,
    clear_model_cache,
    get_model_bundle,
    get_model_cache_status,
)


@pytest.fixture(autouse=True)
def reset_model_cache():
    clear_model_cache()
    yield
    clear_model_cache()


def make_artifact_dir(tmp_path, product_key="phoenix"):
    product_dir = tmp_path / product_key
    product_dir.mkdir()
    (product_dir / "model.joblib").write_text("model", encoding="utf-8")
    (product_dir / "scaler.joblib").write_text("scaler", encoding="utf-8")
    (product_dir / "results.json").write_text(
        json.dumps(
            {
                "config": {
                    "contract_version": "phoenix-single-v1",
                    "feature_order": [
                        "S0",
                        "r",
                        "sigma",
                        "T",
                        "autocall_barrier_frac",
                        "coupon_barrier_frac",
                        "coupon_rate",
                        "knock_in_frac",
                        "obs_count",
                    ],
                }
            }
        ),
        encoding="utf-8",
    )
    return product_dir


def test_model_cache_loads_once_and_reuses_bundle(monkeypatch, tmp_path):
    make_artifact_dir(tmp_path)
    calls = []
    model = object()
    scaler = object()

    def fake_load(model_path, scaler_path):
        calls.append((model_path, scaler_path))
        return model, scaler

    monkeypatch.setattr("app.services.model_cache._load_model_artifacts", fake_load)

    first = get_model_bundle("phoenix", results_dir=tmp_path)
    second = get_model_bundle("phoenix", results_dir=tmp_path)

    assert first is second
    assert first.model is model
    assert first.scaler is scaler
    assert len(calls) == 1
    assert get_model_cache_status(results_dir=tmp_path)["phoenix"] is True


def test_clear_model_cache_removes_cached_status(monkeypatch, tmp_path):
    make_artifact_dir(tmp_path)
    monkeypatch.setattr(
        "app.services.model_cache._load_model_artifacts",
        lambda model_path, scaler_path: (object(), object()),
    )

    get_model_bundle("phoenix", results_dir=tmp_path)
    assert get_model_cache_status(results_dir=tmp_path)["phoenix"] is True

    clear_model_cache()
    assert get_model_cache_status(results_dir=tmp_path) == {}


def test_model_cache_rejects_unknown_product():
    with pytest.raises(ModelCacheProductError):
        get_model_bundle("not_real")


def test_model_cache_rejects_missing_artifacts(tmp_path):
    with pytest.raises(ModelCacheArtifactError):
        get_model_bundle("phoenix", results_dir=tmp_path)
