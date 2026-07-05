import os
import tempfile
from pathlib import Path

from fastapi.testclient import TestClient

os.environ.setdefault(
    "MODEL_HISTORY_FILE", str(Path(tempfile.gettempdir()) / "ml_pricer_test_history.csv")
)
os.environ.setdefault(
    "MODEL_RUN_STORE_FILE",
    str(Path(tempfile.gettempdir()) / "ml_pricer_test_runs.sqlite3"),
)

from app.backend import app


client = TestClient(app)


def test_api_bb_ping_returns_plain_text():
    response = client.get("/api/bb/ping")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert response.text == "OK\nSERVICE=ASHBERRY\n"


def test_api_bb_model_status_returns_compact_plain_text():
    response = client.get("/api/bb/model-status")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")

    body = response.text
    assert body.startswith("OK\n")
    assert "PHOENIX=READY," in body
    assert "ACCUM=READY," in body
    assert "BARRIER=READY," in body


def test_api_bb_products_returns_compact_plain_text():
    response = client.get("/api/bb/products")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")

    lines = response.text.strip().splitlines()
    assert lines[0] == "OK"
    assert "phoenix|Phoenix Autocallable|PHOENIX" in lines
    assert "accumulator|Accumulator|ACCUM" in lines
    assert "reverse_accumulator|Reverse Accumulator|REV-ACC" in lines
