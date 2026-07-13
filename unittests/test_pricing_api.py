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


def test_pricing_api_rejects_client_controlled_target_transform():
    payload = {**VALID_REQUEST, "use_log_target": False}

    response = client.post("/price/", json=payload)

    assert response.status_code == 422


def test_pricing_api_rejects_unvalidated_research_product():
    payload = {**VALID_REQUEST, "payoff_type": "accumulator"}

    response = client.post("/price/", json=payload)

    assert response.status_code == 400
    assert "trace" not in response.json()
