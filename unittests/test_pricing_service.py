import pytest

from app.services.model_cache import clear_model_cache
from app.services.pricing_service import (
    InvalidPricingInputError,
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
    assert result["model"] == "Monte Carlo reference"
    assert result["contract_version"] == "phoenix-single-v1"
    assert result["pricing_method"] == "monte_carlo_reference"
    assert isinstance(result["price"], float)
    assert result["mc_price"] == result["price"]
    assert result["standard_error"] >= 0
    assert len(result["confidence_interval"]) == 2
    assert result["latency_ms"] >= 0


def test_pricing_service_prices_each_bb_enabled_product():
    for product in get_bb_product_definitions():
        result = price_product(product.key, default_params_for(product), n_paths=5)

        assert result["product_key"] == product.key
        assert result["product_name"] == product.display_name
        assert result["model"] == "Monte Carlo reference"
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


def test_pricing_service_is_deterministic_without_surrogate_artifacts(tmp_path):
    first = price_product(
        "phoenix", VALID_PHOENIX_PARAMS, n_paths=100, results_dir=tmp_path
    )
    second = price_product(
        "phoenix", VALID_PHOENIX_PARAMS, n_paths=100, results_dir=tmp_path
    )

    assert first["price"] == second["price"]
    assert first["confidence_interval"] == second["confidence_interval"]


def test_pricing_service_rejects_invalid_barrier_order():
    params = dict(VALID_PHOENIX_PARAMS)
    params["coupon_barrier_frac"] = "1.10"

    with pytest.raises(InvalidPricingInputError, match="barriers must satisfy"):
        normalize_pricing_params("phoenix", params)


def test_pricing_service_rejects_non_finite_and_excessive_inputs():
    non_finite = dict(VALID_PHOENIX_PARAMS)
    non_finite["sigma"] = "nan"
    with pytest.raises(InvalidPricingInputError, match="must be finite"):
        normalize_pricing_params("phoenix", non_finite)

    too_many_observations = dict(VALID_PHOENIX_PARAMS)
    too_many_observations["obs_count"] = "253"
    with pytest.raises(InvalidPricingInputError, match="must be <= 252"):
        normalize_pricing_params("phoenix", too_many_observations)
