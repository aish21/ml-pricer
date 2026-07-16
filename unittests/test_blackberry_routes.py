import os
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

RUN_DB_PATH = Path(tempfile.gettempdir()) / "ml_pricer_test_runs.sqlite3"
if RUN_DB_PATH.exists():
    RUN_DB_PATH.unlink()

os.environ.setdefault(
    "MODEL_HISTORY_FILE",
    str(Path(tempfile.gettempdir()) / "ml_pricer_test_history.csv"),
)
os.environ.setdefault("MODEL_RUN_STORE_FILE", str(RUN_DB_PATH))

from app.backend import app
from app.services.model_cache import clear_model_cache
from app.services.product_registry import (
    get_bb_product_definitions,
    get_product_definition,
)
from app.services.run_store import save_run


client = TestClient(app)

VALID_FORM_DATA = {
    "product_key": "phoenix",
    "S0": "100.0",
    "r": "0.03",
    "sigma": "0.2",
    "T": "1.0",
    "autocall_barrier_frac": "1.05",
    "coupon_barrier_frac": "1.0",
    "coupon_rate": "0.02",
    "knock_in_frac": "0.7",
    "obs_count": "6",
    "n_paths": "5",
}

BASE_REQUEST = {
    "product_key": "phoenix",
    "params": {
        "S0": 100.0,
        "r": 0.03,
        "sigma": 0.2,
        "T": 1.0,
        "autocall_barrier_frac": 1.05,
        "coupon_barrier_frac": 1.0,
        "coupon_rate": 0.02,
        "knock_in_frac": 0.7,
        "obs_count": 6,
    },
    "n_paths": 5,
    "seed": 42,
    "contract_version": "phoenix-single-v1",
}

BASE_RESULT = {
    "product_key": "phoenix",
    "price": 0.984945,
    "model": "Monte Carlo reference",
    "latency_ms": 1,
}


@pytest.fixture(autouse=True)
def reset_run_db():
    clear_model_cache()
    if RUN_DB_PATH.exists():
        RUN_DB_PATH.unlink()
    yield
    clear_model_cache()
    if RUN_DB_PATH.exists():
        RUN_DB_PATH.unlink()


def default_form_data(product, n_paths="5"):
    form_data = {"product_key": product.key, "n_paths": n_paths}
    form_data.update({field.name: str(field.default) for field in product.bb_fields})
    return form_data


def save_base_run_for_route_tests(product_key="phoenix", params=None, price=0.984945):
    request_payload = BASE_REQUEST
    result_payload = BASE_RESULT
    if product_key != "phoenix":
        request_payload = {
            "product_key": product_key,
            "params": params or {},
            "n_paths": 5,
            "seed": 42,
        }
        result_payload = {
            "product_key": product_key,
            "price": price,
            "model": "Monte Carlo reference",
            "latency_ms": 1,
        }
    return save_run(
        product_key,
        request_payload=request_payload,
        result_payload=result_payload,
    )


def test_api_v1_products_returns_json():
    response = client.get("/api/v1/products")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")

    payload = response.json()
    assert payload["status"] == "success"
    assert any(product["key"] == "phoenix" for product in payload["products"])


def test_api_v1_model_info_returns_json():
    response = client.get("/api/v1/model-info")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")

    payload = response.json()
    assert payload["status"] == "success"
    assert payload["model_info"]["api"] == "online"
    assert "phoenix" in payload["model_info"]["supported_product_keys"]
    assert payload["model_info"]["surrogate_shadow"]["reason"] == "disabled"


def test_blackberry_home_returns_terminal_style_html():
    response = client.get("/bb")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")

    body = response.text
    assert "ML-PRICER TERMINAL" in body
    assert "BB-9780 CLIENT" in body
    assert "RECENT RUNS" in body
    assert "LOCAL TERMINAL" in body
    assert "<script" not in body.lower()


def test_blackberry_model_status_returns_terminal_style_html():
    response = client.get("/bb/model-status")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")

    body = response.text
    assert "MODEL STATUS" in body
    assert "PRODUCT" in body
    assert "PHOENIX" in body
    assert "REF" in body
    assert "[0] HOME" in body
    assert "<script" not in body.lower()


def test_blackberry_reference_pricing_does_not_cache_legacy_model():
    client.post("/bb/price", data=VALID_FORM_DATA, follow_redirects=False)
    response = client.get("/bb/model-status")
    assert response.status_code == 200

    body = response.text
    assert "PHOENIX" in body
    assert "REF" in body
    assert "CACHED" not in body


def test_blackberry_price_form_returns_html():
    response = client.get("/bb/price")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")

    body = response.text
    assert "PRICE NOTE" in body
    assert "PHOENIX" in body
    assert "ACCUM" not in body
    assert "BARRIER" not in body
    assert "<script" not in body.lower()


def test_blackberry_price_product_forms_render_for_each_enabled_product():
    for product in get_bb_product_definitions():
        response = client.get(f"/bb/price?product={product.key}")
        assert response.status_code == 200

        body = response.text
        assert f"PRICE {product.terminal_label}" in body
        assert '<form method="post" action="/bb/price">' in body
        for field in product.bb_fields:
            assert field.label in body
        assert "<script" not in body.lower()


def test_blackberry_price_unknown_product_returns_error_html():
    response = client.get("/bb/price?product=not_real")
    assert response.status_code == 200

    body = response.text
    assert "ERROR" in body
    assert "product not enabled" in body
    assert "Traceback" not in body


def test_blackberry_price_post_redirects_to_result():
    response = client.post("/bb/price", data=VALID_FORM_DATA, follow_redirects=False)
    assert response.status_code == 303
    assert response.headers["location"].startswith("/bb/result/")


def test_blackberry_price_post_redirects_for_each_enabled_product(monkeypatch):
    def fake_price_product(product_key, params, n_paths):
        return {
            "product_key": product_key,
            "product_name": product_key,
            "params": params,
            "n_paths": int(n_paths),
            "price": 1.23,
            "mc_price": 1.25,
            "abs_error": 0.02,
            "model": "LightGBM surrogate",
            "latency_ms": 1,
        }

    monkeypatch.setattr("app.bb.routes.price_product", fake_price_product)

    for product in get_bb_product_definitions():
        response = client.post(
            "/bb/price",
            data=default_form_data(product),
            follow_redirects=False,
        )
        assert response.status_code == 303

        result_response = client.get(response.headers["location"])
        assert result_response.status_code == 200
        assert product.terminal_label in result_response.text


def test_blackberry_result_returns_compact_html_after_pricing():
    post_response = client.post(
        "/bb/price", data=VALID_FORM_DATA, follow_redirects=False
    )
    result_response = client.get(post_response.headers["location"])

    assert result_response.status_code == 200
    body = result_response.text
    assert "RUN:" in body
    assert "PHOENIX" in body
    assert "Price:" in body
    assert "Latency:" in body
    assert "[1] SCENARIO SHOCK" in body
    assert "[2] NEW PRICE" in body
    assert "[3] RECENT RUNS" in body


def test_blackberry_result_invalid_run_returns_error_html():
    response = client.get("/bb/result/not-a-run")
    assert response.status_code == 200

    body = response.text
    assert "ERROR" in body
    assert "run not found" in body
    assert "[1] RECENT RUNS" in body
    assert "[2] HOME" in body


def test_blackberry_price_invalid_numeric_input_returns_error_html():
    form_data = dict(VALID_FORM_DATA)
    form_data["sigma"] = "bad-vol"

    response = client.post("/bb/price", data=form_data)
    assert response.status_code == 200

    body = response.text
    assert "ERROR" in body
    assert "invalid numeric parameter: sigma" in body


def test_blackberry_scenario_form_returns_html_for_valid_run():
    run_id = save_base_run_for_route_tests()
    response = client.get(f"/bb/scenario/{run_id}")
    assert response.status_code == 200

    body = response.text
    assert "SCENARIO SHOCK" in body
    assert "Spot %" in body
    assert "PRICE SHOCK" in body


def test_blackberry_scenario_form_missing_run_returns_error_html():
    response = client.get("/bb/scenario/missing")
    assert response.status_code == 200

    body = response.text
    assert "ERROR" in body
    assert "base run not found" in body


def test_blackberry_scenario_post_valid_shocks_returns_result():
    run_id = save_base_run_for_route_tests()
    response = client.post(
        f"/bb/scenario/{run_id}",
        data={"spot_pct": "-10", "vol_abs": "0.05", "rate_bps": "50"},
    )
    assert response.status_code == 200

    body = response.text
    assert "SCENARIO RESULT" in body
    assert "Base:" in body
    assert "Shock:" in body
    assert "Move:" in body
    assert "Spot: -10%" in body
    assert "[1] BASE RUN" in body
    assert "[2] NEW SHOCK" in body


def test_blackberry_scenario_post_works_for_each_enabled_product(monkeypatch):
    def fake_run_scenario(base_request, base_result, shocks):
        product_key = base_request["product_key"]
        shocked_request = dict(base_request)
        shocked_request["params"] = dict(base_request["params"])
        return {
            "product_key": product_key,
            "base_price": base_result.get("price"),
            "shocked_price": 0.9,
            "price_change": -0.1,
            "price_change_pct": -10.0,
            "base_request": base_request,
            "base_result": base_result,
            "shocked_request": shocked_request,
            "shocked_result": {
                "product_key": product_key,
                "price": 0.9,
                "model": "LightGBM surrogate",
                "latency_ms": 1,
            },
            "shocks": {"spot_pct": -10.0},
            "summary": "Spot down changed the product value.",
            "model": "LightGBM surrogate",
            "latency_ms": 1,
        }

    monkeypatch.setattr("app.bb.routes.run_scenario", fake_run_scenario)

    for product in get_bb_product_definitions():
        params = {field.name: field.default for field in product.bb_fields}
        run_id = save_base_run_for_route_tests(product.key, params=params, price=1.0)
        response = client.post(f"/bb/scenario/{run_id}", data={"spot_pct": "-10"})
        assert response.status_code == 200

        body = response.text
        assert "SCENARIO RESULT" in body
        assert product.terminal_label in body
        assert "Spot: -10%" in body


def test_blackberry_scenario_post_no_shocks_returns_error_html():
    run_id = save_base_run_for_route_tests()
    response = client.post(
        f"/bb/scenario/{run_id}",
        data={"spot_pct": "", "vol_abs": "", "rate_bps": ""},
    )
    assert response.status_code == 200

    body = response.text
    assert "ERROR" in body
    assert "at least one shock is required" in body
    assert "Traceback" not in body


def test_blackberry_recent_runs_empty_state_returns_html():
    response = client.get("/bb/recent-runs")
    assert response.status_code == 200

    body = response.text
    assert "NO RUNS YET" in body
    assert "[1] PRICE NOTE" in body
    assert "[H] HOME" in body


def test_blackberry_recent_runs_shows_price_run_with_result_link():
    run_id = save_base_run_for_route_tests()
    response = client.get("/bb/recent-runs")
    assert response.status_code == 200

    body = response.text
    assert "RECENT RUNS" in body
    assert "PRICE PHOENIX" in body
    assert "/bb/result/" + run_id in body
    assert "0.984945" in body


def test_blackberry_recent_runs_shows_multiple_product_labels():
    save_base_run_for_route_tests()
    accumulator = get_product_definition("accumulator")
    save_base_run_for_route_tests(
        accumulator.key,
        params={field.name: field.default for field in accumulator.bb_fields},
        price=8.5,
    )

    response = client.get("/bb/recent-runs")
    assert response.status_code == 200

    body = response.text
    assert "PRICE PHOENIX" in body
    assert "PRICE ACCUM" in body


def test_blackberry_recent_runs_shows_scenario_run_with_base():
    base_id = save_base_run_for_route_tests()
    scenario_id = save_run(
        "phoenix",
        request_payload={
            "product_key": "phoenix",
            "base_run_id": base_id,
            "params": BASE_REQUEST["params"],
            "shocks": {"spot_pct": -10},
        },
        result_payload={
            "product_key": "phoenix",
            "base_price": 0.984945,
            "shocked_price": 0.9,
            "price_change": -0.084945,
            "price_change_pct": -8.62,
            "shocks": {"spot_pct": -10},
            "summary": "Spot down changed the Phoenix value.",
            "model": "LightGBM surrogate",
        },
        run_type="scenario",
        parent_run_id=base_id,
    )

    response = client.get("/bb/recent-runs")
    assert response.status_code == 200

    body = response.text
    assert "SCENARIO PHOENIX" in body
    assert "/bb/result/" + scenario_id in body
    assert "base:" in body
    assert "0.900000" in body


def test_blackberry_result_for_saved_scenario_has_base_navigation():
    base_id = save_base_run_for_route_tests()
    scenario_id = save_run(
        "phoenix",
        request_payload={"product_key": "phoenix", "base_run_id": base_id},
        result_payload={
            "product_key": "phoenix",
            "base_price": 0.984945,
            "shocked_price": 0.9,
            "price_change": -0.084945,
            "price_change_pct": -8.62,
            "shocks": {"spot_pct": -10},
            "summary": "Spot down changed the Phoenix value.",
            "model": "LightGBM surrogate",
        },
        run_type="scenario",
        parent_run_id=base_id,
    )

    response = client.get(f"/bb/result/{scenario_id}")
    assert response.status_code == 200

    body = response.text
    assert "SCENARIO RESULT" in body
    assert f"/bb/result/{base_id}" in body
    assert f"/bb/scenario/{base_id}" in body


def test_existing_price_route_still_rejects_invalid_product():
    response = client.post("/price/", json={"payoff_type": "not_real", "params": {}})
    assert response.status_code == 400
    assert response.json()["status"] == "error"
