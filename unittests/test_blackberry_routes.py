import os
import tempfile
from pathlib import Path

from fastapi.testclient import TestClient

os.environ.setdefault(
    "MODEL_HISTORY_FILE", str(Path(tempfile.gettempdir()) / "ml_pricer_test_history.csv")
)

from app.backend import app


client = TestClient(app)


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


def test_blackberry_home_returns_terminal_style_html():
    response = client.get("/bb")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")

    body = response.text
    assert "ML-PRICER TERMINAL" in body
    assert "BB-9780 CLIENT" in body
    assert "READ-ONLY SHELL" in body
    assert "<script" not in body.lower()


def test_blackberry_model_status_returns_terminal_style_html():
    response = client.get("/bb/model-status")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")

    body = response.text
    assert "MODEL STATUS" in body
    assert "PRODUCT" in body
    assert "phoenix" in body
    assert "[0] HOME" in body
    assert "<script" not in body.lower()
