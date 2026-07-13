import os
import tempfile
from pathlib import Path

from fastapi.testclient import TestClient

os.environ.setdefault(
    "MODEL_HISTORY_FILE",
    str(Path(tempfile.gettempdir()) / "ml_pricer_test_api_history.csv"),
)

from app.backend import app


client = TestClient(app)

VALID_REQUEST = {
    "payoff_type": "phoenix",
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
    "n_paths": 20,
}

SNAPSHOT_REQUEST = {
    "market": {
        "schema_version": "equity-market-snapshot-v1",
        "symbol": "^SPX",
        "underlier_type": "index",
        "currency": "USD",
        "valuation_time": "2026-07-13T12:00:00Z",
        "market_data_time": "2026-07-13T11:59:58Z",
        "spot": 6300.0,
        "risk_free_rate": 0.04,
        "dividend_yield": 0.013,
        "volatility": 0.19,
        "calendar": "XNYS",
        "day_count": "ACT/365F",
        "source": "api-test-fixture",
    },
    "terms": {
        "maturity_years": 1.0,
        "autocall_barrier_frac": 1.05,
        "coupon_barrier_frac": 1.0,
        "coupon_rate": 0.02,
        "knock_in_frac": 0.7,
        "obs_count": 6,
    },
    "n_paths": 20,
}


def test_pricing_api_returns_versioned_reference_result():
    response = client.post("/price/", json=VALID_REQUEST)

    assert response.status_code == 200
    result = response.json()["result"]
    assert result["contract_version"] == "phoenix-single-v1"
    assert result["pricing_method"] == "monte_carlo_reference"
    assert result["seed"] == 42
    assert len(result["confidence_interval"]) == 2


def test_v1_pricing_api_returns_versioned_reference_result():
    response = client.post("/api/v1/price", json=VALID_REQUEST)

    assert response.status_code == 200
    result = response.json()["result"]
    assert result["contract_version"] == "phoenix-single-v1"
    assert result["pricing_method"] == "monte_carlo_reference"


def test_health_endpoints_are_available():
    live = client.get("/health/live")
    ready = client.get("/health/ready")

    assert live.status_code == 200
    assert live.json() == {"status": "alive"}
    assert ready.status_code == 200
    assert ready.json()["contract_version"] == "phoenix-single-v1"


def test_product_focused_phoenix_api_uses_dated_market_snapshot():
    response = client.post("/api/v1/products/phoenix/price", json=SNAPSHOT_REQUEST)

    assert response.status_code == 200
    result = response.json()["result"]
    assert result["underlier"]["symbol"] == "^SPX"
    assert result["underlier"]["type"] == "index"
    assert result["market_snapshot"]["source"] == "api-test-fixture"
    assert result["market_snapshot"]["age_seconds"] == 2.0
    assert result["model_version"] == "equity-gbm-flat-v2"
    assert result["contract_version"] == "phoenix-single-v1"


def test_product_focused_api_rejects_lookahead_market_data():
    payload = {
        **SNAPSHOT_REQUEST,
        "market": {
            **SNAPSHOT_REQUEST["market"],
            "market_data_time": "2026-07-13T12:00:01Z",
        },
    }

    response = client.post("/api/v1/products/phoenix/price", json=payload)

    assert response.status_code == 422
    assert "cannot be after valuation_time" in response.json()["message"]


def test_product_focused_api_rejects_non_equity_like_underlier():
    payload = {
        **SNAPSHOT_REQUEST,
        "market": {**SNAPSHOT_REQUEST["market"], "underlier_type": "crypto"},
    }

    response = client.post("/api/v1/products/phoenix/price", json=payload)

    assert response.status_code == 422


def test_pricing_api_rejects_client_controlled_target_transform():
    payload = {**VALID_REQUEST, "use_log_target": False}

    response = client.post("/price/", json=payload)

    assert response.status_code == 422


def test_pricing_api_rejects_unvalidated_research_product():
    payload = {**VALID_REQUEST, "payoff_type": "accumulator"}

    response = client.post("/price/", json=payload)

    assert response.status_code == 400
    assert "trace" not in response.json()
