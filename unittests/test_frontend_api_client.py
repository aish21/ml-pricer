import pytest
import requests

from app.ui.api_client import FrontendApiError, MlPricerApi


class FakeResponse:
    def __init__(self, body, *, ok=True, status_code=200):
        self._body = body
        self.ok = ok
        self.status_code = status_code

    def json(self):
        return self._body


class FakeSession:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls = []

    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        if self.error:
            raise self.error
        return self.response


def test_api_client_routes_seasoned_contract_to_v2_endpoint():
    session = FakeSession(FakeResponse({"status": "success", "result": {"price": 0.9}}))
    client = MlPricerApi("http://pricing", session=session)

    result = client.price(
        market={"spot": 90.0},
        terms={"maturity_years": 1.0},
        contract={"contract_version": "phoenix-single-v2"},
        n_paths=500,
    )

    assert result["price"] == 0.9
    assert session.calls[0][1].endswith(
        "/api/v1/products/phoenix/price/seasoned/term-structure"
    )


def test_api_client_routes_richer_contract_to_v3_endpoints():
    session = FakeSession(FakeResponse({"status": "success", "result": {"price": 1.0}}))
    client = MlPricerApi("http://pricing", session=session)
    contract = {"contract_version": "phoenix-single-v3"}

    client.price(market={}, terms={}, contract=contract, n_paths=500)

    assert session.calls[0][1].endswith(
        "/api/v1/products/phoenix/price/richer/term-structure"
    )


def test_api_client_routes_reverse_convertible_to_focused_endpoint():
    session = FakeSession(FakeResponse({"status": "success", "result": {"price": 1.0}}))
    client = MlPricerApi("http://pricing", session=session)

    client.price_barrier_reverse_convertible(
        market={"spot": 100.0},
        contract={"contract_version": "barrier-reverse-convertible-v1"},
        n_paths=500,
    )

    assert session.calls[0][1].endswith(
        "/api/v1/products/barrier-reverse-convertible/price/term-structure"
    )


def test_api_client_loads_bounded_ml_evidence_snapshot():
    session = FakeSession(
        FakeResponse({"status": "success", "evidence": {"audit": {"available": True}}})
    )
    client = MlPricerApi("http://pricing", session=session)

    evidence = client.ml_evidence(monitoring_limit=999_999, series_limit=0)

    assert evidence["audit"]["available"] is True
    assert session.calls[0][0] == "GET"
    assert session.calls[0][1].endswith(
        "/api/v1/surrogate-shadow/evidence" "?monitoring_limit=100000&series_limit=1"
    )


def test_api_client_surfaces_sanitized_backend_message():
    session = FakeSession(
        FakeResponse(
            {"status": "error", "message": "invalid schedule"},
            ok=False,
            status_code=422,
        )
    )
    client = MlPricerApi("http://pricing", session=session)

    with pytest.raises(FrontendApiError, match="invalid schedule"):
        client.build_research_market(
            symbol="SPY",
            underlier_type="etf",
            maturity_years=1.0,
        )


def test_api_client_identifies_the_first_invalid_request_field():
    session = FakeSession(
        FakeResponse(
            {
                "detail": [
                    {
                        "type": "extra_forbidden",
                        "loc": ["body", "market", "unexpected"],
                        "msg": "Extra inputs are not permitted",
                        "input": "sensitive value",
                    }
                ]
            },
            ok=False,
            status_code=422,
        )
    )
    client = MlPricerApi("http://pricing", session=session)

    with pytest.raises(
        FrontendApiError,
        match="market.unexpected.*Extra inputs are not permitted",
    ) as error:
        client.price(
            market={},
            terms={},
            contract=None,
            n_paths=500,
        )

    assert "sensitive value" not in str(error.value)


def test_api_client_hides_transport_details():
    session = FakeSession(error=requests.ConnectionError("secret upstream detail"))
    client = MlPricerApi("http://pricing", session=session)

    with pytest.raises(FrontendApiError, match="could not be reached") as error:
        client.price(
            market={},
            terms={},
            contract=None,
            n_paths=500,
        )

    assert "secret" not in str(error.value)
