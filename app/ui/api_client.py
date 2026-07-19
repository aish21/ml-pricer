from typing import Any, Mapping

import requests


class FrontendApiError(RuntimeError):
    pass


def _backend_error_message(body: Any, status_code: int) -> str:
    if not isinstance(body, Mapping):
        return f"The pricing request failed (HTTP {status_code})."
    message = body.get("message")
    if isinstance(message, str) and message.strip():
        return message.strip()
    detail = body.get("detail")
    if isinstance(detail, str) and detail.strip():
        return detail.strip()
    if isinstance(detail, list) and detail:
        first = detail[0]
        if isinstance(first, Mapping):
            location = first.get("loc")
            field = ""
            if isinstance(location, (list, tuple)):
                field = ".".join(
                    str(part) for part in location if str(part) not in {"body"}
                )
            validation_message = first.get("msg")
            if isinstance(validation_message, str) and validation_message.strip():
                if field:
                    return f"Invalid request field '{field}': {validation_message}."
                return f"Invalid pricing request: {validation_message}."
    return f"The pricing request failed (HTTP {status_code})."


class MlPricerApi:
    def __init__(
        self,
        base_url: str,
        *,
        timeout_seconds: int = 180,
        session: requests.Session | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = int(timeout_seconds)
        self.session = session or requests.Session()

    def _request(
        self,
        method: str,
        path: str,
        *,
        payload: Mapping[str, Any] | None = None,
        timeout_seconds: int | None = None,
    ) -> dict[str, Any]:
        try:
            response = self.session.request(
                method,
                f"{self.base_url}{path}",
                json=dict(payload) if payload is not None else None,
                timeout=timeout_seconds or self.timeout_seconds,
            )
        except requests.RequestException as exc:
            raise FrontendApiError("The pricing service could not be reached.") from exc
        try:
            body = response.json()
        except ValueError as exc:
            raise FrontendApiError(
                "The pricing service returned an unreadable response."
            ) from exc
        if not response.ok:
            raise FrontendApiError(_backend_error_message(body, response.status_code))
        if not isinstance(body, dict):
            raise FrontendApiError("The pricing service returned an invalid response.")
        return body

    def health(self) -> bool:
        try:
            body = self._request("GET", "/health/ready", timeout_seconds=5)
        except FrontendApiError:
            return False
        return body.get("status") in {"ready", "online", "success"}

    def build_research_market(
        self,
        *,
        symbol: str,
        underlier_type: str,
        maturity_years: float,
    ) -> dict[str, Any]:
        body = self._request(
            "POST",
            "/api/v1/market-data/research-term-structure",
            payload={
                "market": {
                    "symbol": symbol,
                    "underlier_type": underlier_type,
                    "currency": "USD",
                },
                "maturity_years": maturity_years,
            },
        )
        return dict(body["market_term_structure"])

    def price(
        self,
        *,
        market: Mapping[str, Any],
        terms: Mapping[str, Any],
        contract: Mapping[str, Any] | None,
        n_paths: int,
    ) -> dict[str, Any]:
        if contract is None:
            path = "/api/v1/products/phoenix/price/term-structure"
            payload = {
                "market": dict(market),
                "terms": dict(terms),
                "n_paths": n_paths,
            }
        else:
            path = "/api/v1/products/phoenix/price/seasoned/term-structure"
            payload = {
                "market": dict(market),
                "contract": dict(contract),
                "n_paths": n_paths,
            }
        return dict(self._request("POST", path, payload=payload)["result"])

    def diagnostics(
        self,
        *,
        market: Mapping[str, Any],
        terms: Mapping[str, Any],
        contract: Mapping[str, Any] | None,
        n_paths: int,
        seed: int,
        spot_shocks_pct: list[float],
        volatility_shocks_abs: list[float],
    ) -> dict[str, Any]:
        common = {
            "market": dict(market),
            "n_paths": min(int(n_paths), 5_000),
            "seed": int(seed),
            "spot_shocks_pct": spot_shocks_pct,
            "volatility_shocks_abs": volatility_shocks_abs,
        }
        if contract is None:
            path = "/api/v1/products/phoenix/diagnostics/term-structure"
            payload = {**common, "terms": dict(terms)}
        else:
            path = "/api/v1/products/phoenix/diagnostics/seasoned/term-structure"
            payload = {**common, "contract": dict(contract)}
        return dict(self._request("POST", path, payload=payload)["diagnostics"])

    def scenario(
        self,
        *,
        market: Mapping[str, Any],
        terms: Mapping[str, Any],
        shock: Mapping[str, Any],
        n_paths: int,
        seed: int,
    ) -> dict[str, Any]:
        body = self._request(
            "POST",
            "/api/v1/products/phoenix/scenario/term-structure",
            payload={
                "market": dict(market),
                "terms": dict(terms),
                "shock": dict(shock),
                "n_paths": n_paths,
                "seed": seed,
            },
        )
        return dict(body["result"])

    def risk(
        self,
        *,
        market: Mapping[str, Any],
        terms: Mapping[str, Any],
        bumps: Mapping[str, Any],
        n_paths: int,
        seed: int,
    ) -> dict[str, Any]:
        body = self._request(
            "POST",
            "/api/v1/products/phoenix/risk/term-structure",
            payload={
                "market": dict(market),
                "terms": dict(terms),
                "bumps": dict(bumps),
                "n_paths": n_paths,
                "seed": seed,
            },
        )
        return dict(body["result"])
