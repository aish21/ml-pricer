from app.services.product_registry import (
    build_artifact_status,
    get_bb_product_definitions,
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
EXPECTED_VALIDATED_KEYS = {"phoenix"}


def test_product_registry_returns_expected_supported_keys():
    products = list_products()
    keys = {product["key"] for product in products}
    assert EXPECTED_PRODUCT_KEYS.issubset(keys)

    phoenix = next(product for product in products if product["key"] == "phoenix")
    assert phoenix["market_snapshot_versions"] == ["equity-market-snapshot-v1"]
    assert phoenix["market_term_structure_versions"] == [
        "equity-market-term-structure-v1"
    ]
    assert "equity-gbm-flat-v2" in phoenix["market_model_versions"]
    assert "equity-gbm-piecewise-v1" in phoenix["market_model_versions"]


def test_bb_enabled_products_have_terminal_fields():
    products = get_bb_product_definitions()
    assert {product.key for product in products} == EXPECTED_VALIDATED_KEYS

    for product in products:
        assert product.terminal_label
        assert product.bb_fields
        assert all(
            field.name and field.label and field.field_type
            for field in product.bb_fields
        )


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

    incompatible = build_artifact_status(product, tmp_path)
    assert incompatible["ready_for_surrogate"] is False
    assert incompatible["artifact_compatible"] is False

    (product_dir / "results.json").write_text(
        """{
          "config": {
            "contract_version": "phoenix-single-v1",
            "feature_order": [
              "S0", "r", "sigma", "T", "autocall_barrier_frac",
              "coupon_barrier_frac", "coupon_rate", "knock_in_frac",
              "obs_count"
            ]
          }
        }""",
        encoding="utf-8",
    )

    available = build_artifact_status(product, tmp_path)
    assert available["ready_for_surrogate"] is True
    assert available["artifact_compatible"] is True
