import pytest

from app.services.model_cache import clear_model_cache
from app.services.pricing_service import (
    InvalidPricingInputError,
    PricingArtifactError,
    UnsupportedProductError,
    normalize_pricing_params,
    price_product,
)
from app.services.product_registry import get_bb_product_definitions


VALID_PHOENIX_PARAMS = {
    "S0": "100.0",
    "r": "0.03",
    "sigma": "0.2",
    "T": "1.0",
    "autocall_barrier_frac": "1.05",
    "coupon_barrier_frac": "1.0",
    "coupon_rate": "0.02",
    "knock_in_frac": "0.7",
    "obs_count": "6",
}


@pytest.fixture(autouse=True)
def reset_model_cache():
    clear_model_cache()
    yield
    clear_model_cache()


def default_params_for(product):
    return {field.name: str(field.default) for field in product.bb_fields}


def test_pricing_service_prices_valid_phoenix_request():
    result = price_product("phoenix", VALID_PHOENIX_PARAMS, n_paths=5)

    assert result["product_key"] == "phoenix"
    assert result["model"] == "LightGBM surrogate"
    assert isinstance(result["price"], float)
    assert isinstance(result["mc_price"], float)
    assert result["latency_ms"] >= 0


def test_pricing_service_prices_each_bb_enabled_product():
    for product in get_bb_product_definitions():
        result = price_product(product.key, default_params_for(product), n_paths=5)

        assert result["product_key"] == product.key
        assert result["product_name"] == product.display_name
        assert result["model"] == "LightGBM surrogate"
        assert isinstance(result["price"], float)
        assert isinstance(result["mc_price"], float)
        assert result["latency_ms"] >= 0


def test_pricing_service_rejects_invalid_product():
    with pytest.raises(UnsupportedProductError):
        price_product("not_real", VALID_PHOENIX_PARAMS, n_paths=5)


def test_pricing_service_rejects_invalid_numeric_input():
    params = dict(VALID_PHOENIX_PARAMS)
    params["sigma"] = "not-a-number"

    with pytest.raises(InvalidPricingInputError):
        normalize_pricing_params("phoenix", params)


def test_pricing_service_rejects_missing_required_input():
    params = dict(VALID_PHOENIX_PARAMS)
    del params["sigma"]

    with pytest.raises(InvalidPricingInputError):
        normalize_pricing_params("phoenix", params)


def test_pricing_service_rejects_missing_model_artifacts(tmp_path):
    with pytest.raises(PricingArtifactError):
        price_product("phoenix", VALID_PHOENIX_PARAMS, n_paths=5, results_dir=tmp_path)
