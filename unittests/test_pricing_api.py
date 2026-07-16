import os
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi.testclient import TestClient

from src.final.market import (
    EquityMarketSegment,
    EquityMarketSnapshot,
    EquityMarketTermStructure,
)
from app.services.live_market_data import (
    LiveEquityQuote,
    LiveSnapshotResult,
    MarketDataRateLimitError,
    QuoteFetchResult,
)
from app.services.research_market_data import (
    ResearchMarketBuildResult,
    ResearchMarketUnsupportedError,
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

RESEARCH_MARKET_REQUEST = {
    "market": {
        "symbol": "SPY",
        "underlier_type": "etf",
        "currency": "USD",
    },
    "terms": SNAPSHOT_REQUEST["terms"],
    "n_paths": 20,
}

SCENARIO_REQUEST = {
    **TERM_STRUCTURE_REQUEST,
    "shock": {
        "spot_pct": -10.0,
        "rate_parallel_bps": 25.0,
        "segment_shocks": [{"segment_index": 1, "volatility_abs": 0.02}],
    },
    "seed": 42,
}

RISK_REQUEST = {
    **TERM_STRUCTURE_REQUEST,
    "bumps": {
        "spot_relative": 0.01,
        "volatility_absolute": 0.01,
        "rate_bps": 10.0,
        "dividend_bps": 10.0,
    },
    "seed": 42,
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


def make_research_market_result():
    market = EquityMarketTermStructure(
        symbol="SPY",
        underlier_type="etf",
        currency="USD",
        valuation_time=datetime(2026, 7, 13, 12, 0, 2, tzinfo=timezone.utc),
        market_data_time=datetime(2026, 7, 13, 12, 0, tzinfo=timezone.utc),
        spot=620.25,
        segments=(
            EquityMarketSegment(0.5, 0.035, 0.012, 0.18),
            EquityMarketSegment(1.0, 0.04, 0.013, 0.21),
        ),
        calendar="ARCX",
        day_count="ACT/365F",
        source="research-api-test",
    )
    return ResearchMarketBuildResult(
        market=market,
        calibration={
            "calibration_version": "equity-research-market-v1",
            "calibration_id": "sha256:test-calibration",
            "term_structure_id": market.term_structure_id,
            "research_only": True,
            "methods": {
                "risk_free_rate": "treasury-cmt-continuous-zero-proxy-v1",
                "dividend_yield": "yfinance-trailing-cash-distribution-yield-v1",
                "volatility": "yfinance-atm-forward-variance-v1",
            },
            "warnings": ["test research warning"],
        },
    )


class FakeResearchMarketDataService:
    def __init__(self):
        self.result = make_research_market_result()

    def build_term_structure(self, symbol, underlier_type, maturity_years):
        assert symbol == "SPY"
        assert underlier_type == "etf"
        assert maturity_years == 1.0
        return self.result


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
    assert ready.json()["surrogate_shadow"] == {
        "enabled": False,
        "mode": "shadow-only",
        "available": False,
        "model_version": "phoenix-surrogate-payoff-aware-v3",
        "reason": "disabled",
    }


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


def test_research_market_build_endpoint_returns_calibrated_structure(monkeypatch):
    monkeypatch.setattr(
        "app.api.v1.get_research_market_data_service",
        FakeResearchMarketDataService,
    )
    payload = {
        "market": RESEARCH_MARKET_REQUEST["market"],
        "maturity_years": 1.0,
    }

    response = client.post("/api/v1/market-data/research-term-structure", json=payload)

    assert response.status_code == 200
    result = response.json()
    assert result["market_term_structure"]["schema_version"] == (
        "equity-market-term-structure-v1"
    )
    assert result["market_calibration"]["calibration_version"] == (
        "equity-research-market-v1"
    )
    assert result["market_calibration"]["research_only"] is True


def test_research_market_phoenix_endpoint_prices_calibrated_structure(monkeypatch):
    monkeypatch.setattr(
        "app.api.v1.get_research_market_data_service",
        FakeResearchMarketDataService,
    )

    response = client.post(
        "/api/v1/products/phoenix/price/research-market",
        json=RESEARCH_MARKET_REQUEST,
    )

    assert response.status_code == 200
    result = response.json()["result"]
    assert result["model_version"] == "equity-gbm-piecewise-v1"
    assert result["market_calibration_version"] == "equity-research-market-v1"
    assert result["market_calibration"]["calibration_id"] == ("sha256:test-calibration")
    assert result["market_term_structure"]["spot"] == 620.25


def test_research_market_endpoint_rejects_unsupported_calibration(monkeypatch):
    class UnsupportedResearchMarketDataService:
        def build_term_structure(self, **kwargs):
            raise ResearchMarketUnsupportedError(
                "research calibration currently supports USD underliers only"
            )

    monkeypatch.setattr(
        "app.api.v1.get_research_market_data_service",
        UnsupportedResearchMarketDataService,
    )

    response = client.post(
        "/api/v1/products/phoenix/price/research-market",
        json=RESEARCH_MARKET_REQUEST,
    )

    assert response.status_code == 422
    assert response.json()["message"] == (
        "research calibration currently supports USD underliers only"
    )


def test_term_structure_scenario_api_persists_paired_result(monkeypatch):
    monkeypatch.setattr("app.api.v1.save_run", lambda **kwargs: "run_scenario_test")

    response = client.post(
        "/api/v1/products/phoenix/scenario/term-structure",
        json=SCENARIO_REQUEST,
    )

    assert response.status_code == 200
    payload = response.json()
    result = payload["result"]
    assert payload["run_id"] == "run_scenario_test"
    assert result["scenario_version"] == "equity-market-scenario-v1"
    assert result["provenance"]["common_random_numbers"] is True
    assert result["provenance"]["contract_reference_spot"] == 620.0
    assert result["shocked_market"]["spot"] == 558.0
    assert result["shock"]["segment_shocks"][0]["segment_index"] == 1
    assert len(result["pnl"]["confidence_interval"]) == 2


def test_term_structure_risk_api_returns_greeks_and_provenance(monkeypatch):
    monkeypatch.setattr("app.api.v1.save_run", lambda **kwargs: "run_risk_test")

    response = client.post(
        "/api/v1/products/phoenix/risk/term-structure", json=RISK_REQUEST
    )

    assert response.status_code == 200
    payload = response.json()
    result = payload["result"]
    assert payload["run_id"] == "run_risk_test"
    assert result["risk_version"] == "equity-risk-analytics-v1"
    assert set(result["sensitivities"]) == {
        "delta",
        "gamma",
        "vega",
        "rho",
        "dividend_rho",
    }
    assert result["provenance"]["model_version"] == "equity-gbm-piecewise-v1"
    assert result["provenance"]["seed"] == 42


def test_research_scenario_and_risk_apis_preserve_calibration(monkeypatch):
    monkeypatch.setattr(
        "app.api.v1.get_research_market_data_service",
        FakeResearchMarketDataService,
    )
    monkeypatch.setattr("app.api.v1.save_run", lambda **kwargs: "run_research_test")
    scenario_payload = {
        **RESEARCH_MARKET_REQUEST,
        "shock": {"volatility_parallel_abs": 0.02},
        "seed": 7,
    }
    risk_payload = {
        **RESEARCH_MARKET_REQUEST,
        "bumps": RISK_REQUEST["bumps"],
        "seed": 7,
    }

    scenario = client.post(
        "/api/v1/products/phoenix/scenario/research-market",
        json=scenario_payload,
    )
    risk = client.post(
        "/api/v1/products/phoenix/risk/research-market", json=risk_payload
    )

    assert scenario.status_code == 200
    assert risk.status_code == 200
    assert scenario.json()["result"]["provenance"]["market_calibration_id"] == (
        "sha256:test-calibration"
    )
    assert risk.json()["result"]["provenance"]["market_calibration_id"] == (
        "sha256:test-calibration"
    )


def test_scenario_api_rejects_empty_or_invalid_shocks(monkeypatch):
    monkeypatch.setattr("app.api.v1.save_run", lambda **kwargs: "not-saved")
    empty = {**SCENARIO_REQUEST, "shock": {}}
    invalid = {
        **SCENARIO_REQUEST,
        "shock": {"segment_shocks": [{"segment_index": 9, "rate_bps": 1}]},
    }

    empty_response = client.post(
        "/api/v1/products/phoenix/scenario/term-structure", json=empty
    )
    invalid_response = client.post(
        "/api/v1/products/phoenix/scenario/term-structure", json=invalid
    )

    assert empty_response.status_code == 422
    assert "at least one" in empty_response.json()["message"]
    assert invalid_response.status_code == 422
    assert "outside" in invalid_response.json()["message"]


def test_analysis_run_json_endpoints_round_trip(tmp_path, monkeypatch):
    store = tmp_path / "api-runs.sqlite3"
    monkeypatch.setenv("MODEL_RUN_STORE_FILE", str(store))

    response = client.post(
        "/api/v1/products/phoenix/scenario/term-structure",
        json=SCENARIO_REQUEST,
    )
    assert response.status_code == 200
    run_id = response.json()["run_id"]

    fetched = client.get(f"/api/v1/runs/{run_id}")
    recent = client.get("/api/v1/runs", params={"limit": 1})

    assert fetched.status_code == 200
    assert fetched.json()["run"]["run_type"] == "scenario"
    assert fetched.json()["run"]["result_payload"]["scenario_version"] == (
        "equity-market-scenario-v1"
    )
    assert recent.status_code == 200
    assert recent.json()["runs"][0]["run_id"] == run_id


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
    assert payload["research_market"] == {
        "enabled": True,
        "configured": True,
        "calibration_version": "equity-research-market-v1",
        "currency": "USD",
        "underlier_types": ["equity", "etf"],
        "sources": ["U.S. Department of the Treasury", "yfinance"],
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
