from app.services.product_registry import (
    build_artifact_status,
    get_product_definition,
    list_products,
)


EXPECTED_PRODUCT_KEYS = {
    "phoenix",
    "accumulator",
    "barrier",
    "decumulator",
    "phoenix_stepdown",
    "reverse_accumulator",
}


def test_product_registry_returns_expected_supported_keys():
    keys = {product["key"] for product in list_products()}
    assert EXPECTED_PRODUCT_KEYS.issubset(keys)


def test_model_status_detects_artifact_availability(tmp_path):
    product = get_product_definition("phoenix")
    product_dir = tmp_path / "phoenix"
    product_dir.mkdir()

    missing = build_artifact_status(product, tmp_path)
    assert missing["ready_for_surrogate"] is False
    assert missing["model_available"] is False
    assert missing["scaler_available"] is False

    (product_dir / "model.joblib").write_text("model", encoding="utf-8")
    (product_dir / "scaler.joblib").write_text("scaler", encoding="utf-8")
    (product_dir / "results.json").write_text("{}", encoding="utf-8")

    available = build_artifact_status(product, tmp_path)
    assert available["ready_for_surrogate"] is True
    assert available["model_available"] is True
    assert available["scaler_available"] is True
    assert available["training_metadata_available"] is True
