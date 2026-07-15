import os
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi.testclient import TestClient

from src.final.market import EquityMarketSnapshot
from app.services.live_market_data import (
    LiveEquityQuote,
    LiveSnapshotResult,
    MarketDataRateLimitError,
    QuoteFetchResult,
)

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

TERM_STRUCTURE_REQUEST = {
    "market": {
        "schema_version": "equity-market-term-structure-v1",
        "symbol": "SPY",
        "underlier_type": "etf",
        "currency": "USD",
        "valuation_time": "2026-07-13T12:00:00Z",
        "market_data_time": "2026-07-13T11:59:58Z",
        "spot": 620.0,
        "segments": [
            {
                "end_time_years": 0.5,
                "risk_free_rate": 0.035,
                "dividend_yield": 0.012,
                "volatility": 0.18,
            },
            {
                "end_time_years": 1.0,
                "risk_free_rate": 0.04,
                "dividend_yield": 0.013,
                "volatility": 0.21,
            },
        ],
        "calendar": "XNYS",
        "day_count": "ACT/365F",
        "source": "test-term-structure",
    },
    "terms": SNAPSHOT_REQUEST["terms"],
    "n_paths": 20,
}

MARKET_REQUEST = {
    "market": {
        "symbol": "SPY",
        "underlier_type": "etf",
        "risk_free_rate": 0.04,
        "dividend_yield": 0.012,
        "volatility": 0.2,
        "day_count": "ACT/365F",
    },
    "terms": SNAPSHOT_REQUEST["terms"],
    "n_paths": 20,
}


def make_market_results():
    valuation_time = datetime(2026, 7, 13, 12, 0, 2, tzinfo=timezone.utc)
    market_data_time = valuation_time - timedelta(seconds=2)
    quote = LiveEquityQuote(
        provider_name="yfinance",
        symbol="SPY",
        spot=620.25,
        currency="USD",
        market_data_time=market_data_time,
        exchange="NYSE ARCA",
        mic_code="ARCX",
        instrument_type="ETF",
        underlier_type="etf",
        provider_spot=620.25,
        provider_currency="USD",
        unit_conversion_factor=1.0,
        bar_interval="1m",
        data_delay_seconds=0,
    )
    snapshot = EquityMarketSnapshot(
        symbol="SPY",
        underlier_type="etf",
        currency="USD",
        valuation_time=valuation_time,
        market_data_time=market_data_time,
        spot=620.25,
        risk_free_rate=0.04,
        dividend_yield=0.012,
        volatility=0.2,
        calendar="ARCX",
        day_count="ACT/365F",
        source="yfinance:1m-close+request-model-inputs",
    )
    return quote, LiveSnapshotResult(
        snapshot=snapshot,
        quote=quote,
        cache_hit=False,
        quote_age_seconds=2.0,
    )


class FakeMarketDataService:
    def __init__(self):
        self.quote, self.snapshot = make_market_results()

    def get_quote(self, symbol):
        assert symbol == "SPY"
        return QuoteFetchResult(self.quote, cache_hit=True), 2.0

    def get_snapshot(self, **kwargs):
        assert kwargs["symbol"] == "SPY"
        assert kwargs["underlier_type"] == "etf"
        return self.snapshot


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


def test_term_structure_api_returns_versioned_piecewise_result():
    response = client.post(
        "/api/v1/products/phoenix/price/term-structure",
        json=TERM_STRUCTURE_REQUEST,
    )

    assert response.status_code == 200
    result = response.json()["result"]
    assert result["model_version"] == "equity-gbm-piecewise-v1"
    assert result["market_data_version"] == "equity-market-term-structure-v1"
    assert result["market_term_structure"]["segments"][1]["volatility"] == 0.21


def test_term_structure_api_rejects_unsorted_or_short_segments():
    unsorted = {
        **TERM_STRUCTURE_REQUEST,
        "market": {
            **TERM_STRUCTURE_REQUEST["market"],
            "segments": list(reversed(TERM_STRUCTURE_REQUEST["market"]["segments"])),
        },
    }
    response = client.post(
        "/api/v1/products/phoenix/price/term-structure", json=unsorted
    )
    assert response.status_code == 422
    assert "strictly increasing" in response.json()["message"]

    short = {
        **TERM_STRUCTURE_REQUEST,
        "market": {
            **TERM_STRUCTURE_REQUEST["market"],
            "segments": TERM_STRUCTURE_REQUEST["market"]["segments"][:1],
        },
    }
    response = client.post("/api/v1/products/phoenix/price/term-structure", json=short)
    assert response.status_code == 422
    assert "does not cover" in response.json()["message"]


def test_market_quote_endpoint_returns_normalized_cached_quote(monkeypatch):
    monkeypatch.setattr(
        "app.api.v1.get_live_market_data_service", FakeMarketDataService
    )

    response = client.get("/api/v1/market-data/quote", params={"symbol": "SPY"})

    assert response.status_code == 200
    quote = response.json()["quote"]
    assert quote["spot"] == 620.25
    assert quote["currency"] == "USD"
    assert quote["cache_hit"] is True
    assert quote["quote_age_seconds"] == 2.0
    assert quote["research_only"] is True


def test_market_phoenix_endpoint_prices_provider_snapshot(monkeypatch):
    monkeypatch.setattr(
        "app.api.v1.get_live_market_data_service", FakeMarketDataService
    )

    response = client.post("/api/v1/products/phoenix/price/market", json=MARKET_REQUEST)

    assert response.status_code == 200
    result = response.json()["result"]
    assert result["market_snapshot"]["spot"] == 620.25
    assert result["market_snapshot"]["source"].startswith("yfinance")
    assert result["market_data"]["provider"] == "yfinance"
    assert result["market_data"]["research_only"] is True
    assert result["market_data"]["input_sources"]["volatility"] == "request"
    assert result["model_version"] == "equity-gbm-flat-v2"


def test_market_data_failure_is_sanitized(monkeypatch):
    def rate_limited():
        raise MarketDataRateLimitError("market data provider rate limit reached")

    monkeypatch.setattr("app.api.v1.get_live_market_data_service", rate_limited)

    response = client.get("/api/v1/market-data/quote", params={"symbol": "SPY"})

    assert response.status_code == 503
    assert response.json() == {
        "status": "error",
        "message": "market data provider rate limit reached",
    }


def test_market_data_status_reports_credential_free_research_provider():
    response = client.get("/api/v1/market-data/status")

    assert response.status_code == 200
    payload = response.json()
    assert payload["market_data"] == {
        "enabled": True,
        "configured": True,
        "provider": "yfinance",
        "credentials_required": False,
        "research_only": True,
    }


def test_pricing_api_rejects_client_controlled_target_transform():
    payload = {**VALID_REQUEST, "use_log_target": False}

    response = client.post("/price/", json=payload)

    assert response.status_code == 422


def test_pricing_api_rejects_unvalidated_research_product():
    payload = {**VALID_REQUEST, "payoff_type": "accumulator"}

    response = client.post("/price/", json=payload)

    assert response.status_code == 400
    assert "trace" not in response.json()
