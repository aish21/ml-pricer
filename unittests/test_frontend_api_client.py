import pytest
import requests

from app.ui.api_client import FrontendApiError, NeuralPricerApi


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
    client = NeuralPricerApi("http://pricing", session=session)

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


def test_api_client_surfaces_sanitized_backend_message():
    session = FakeSession(
        FakeResponse(
            {"status": "error", "message": "invalid schedule"},
            ok=False,
            status_code=422,
        )
    )
    client = NeuralPricerApi("http://pricing", session=session)

    with pytest.raises(FrontendApiError, match="invalid schedule"):
        client.build_research_market(
            symbol="SPY",
            underlier_type="etf",
            maturity_years=1.0,
        )


def test_api_client_hides_transport_details():
    session = FakeSession(error=requests.ConnectionError("secret upstream detail"))
    client = NeuralPricerApi("http://pricing", session=session)

    with pytest.raises(FrontendApiError, match="could not be reached") as error:
        client.price(
            market={},
            terms={},
            contract=None,
            n_paths=500,
        )

    assert "secret" not in str(error.value)
