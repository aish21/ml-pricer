import os
from urllib.parse import urlsplit, urlunsplit

import streamlit as st

from app.ui.api_client import FrontendApiError, MlPricerApi
from app.ui.inputs import render_configuration, render_sidebar
from app.ui.payloads import PricingConfiguration, diagnostic_grids
from app.ui.results import render_pricing_results


API_URL = os.getenv(
    "API_URL",
    "https://aish-ml-pricer-backend.up.railway.app",
)


def browser_facing_api_url(
    service_url: str,
    configured_public_url: str | None = None,
) -> str:
    if configured_public_url and configured_public_url.strip():
        return configured_public_url.strip().rstrip("/")
    parsed = urlsplit(service_url)
    if parsed.hostname == "backend":
        port = f":{parsed.port}" if parsed.port else ""
        return urlunsplit((parsed.scheme or "http", f"localhost{port}", "", "", ""))
    return service_url.rstrip("/")


PUBLIC_API_URL = browser_facing_api_url(
    API_URL,
    os.getenv("API_PUBLIC_URL"),
)


def _initialize_state() -> None:
    defaults = {
        "pricing_result": None,
        "diagnostics_result": None,
        "pricing_configuration": None,
        "frozen_market": None,
        "scenario_result": None,
        "risk_result": None,
    }
    for name, value in defaults.items():
        if name not in st.session_state:
            st.session_state[name] = value


def _render_header(experience_mode: str) -> None:
    if experience_mode == "Guided":
        eyebrow = "Learn structured-product pricing from zero"
        title = "Build a pretend note, one tiny step at a time."
        body = (
            "No finance knowledge needed. Pick something to watch, draw a few "
            "rules, let the computer imagine possible futures, and learn how "
            "those pieces become a price."
        )
    else:
        eyebrow = "Structured-product research workspace"
        title = "Price the contract. See the mechanics."
        body = (
            "A deterministic Monte Carlo reference pricer for Phoenix notes, "
            "with explicit market provenance, contract state, uncertainty, "
            "cashflow decomposition, and interactive risk diagnostics."
        )
    st.markdown(
        f"""
        <div class="mlp-hero">
          <div class="mlp-eyebrow">{eyebrow}</div>
          <h1>{title}</h1>
          <p>{body}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_empty_state() -> None:
    st.markdown("### Start with the economics")
    columns = st.columns(3)
    cards = [
        (
            "1 · Market",
            "Choose a research curve or enter a transparent flat market. "
            "The market drives future paths.",
        ),
        (
            "2 · Contract",
            "Set coupons and barriers. Seasoned trades keep their original "
            "reference and historical knock-in state.",
        ),
        (
            "3 · Evidence",
            "Inspect Monte Carlo uncertainty, path outcomes, cashflows, and "
            "a common-random-number valuation surface.",
        ),
    ]
    for column, (title, body) in zip(columns, cards):
        with column:
            st.markdown(
                f"""
                <div class="mlp-card">
                  <h4>{title}</h4>
                  <p>{body}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
    st.info(
        "This is research software. Prices are model estimates, not executable "
        "quotes or investment advice."
    )


def _run_pricing(
    client: MlPricerApi,
    config: PricingConfiguration,
) -> None:
    if config.market_source == "Research market":
        market = client.build_research_market(
            symbol=config.symbol,
            underlier_type=config.underlier_type,
            maturity_years=config.maturity_years,
        )
    else:
        market = dict(config.manual_market or {})
    result = client.price(
        market=market,
        terms=config.terms,
        contract=config.contract,
        n_paths=config.n_paths,
    )
    grids = diagnostic_grids(market)
    diagnostics = client.diagnostics(
        market=market,
        terms=config.terms,
        contract=config.contract,
        n_paths=config.n_paths,
        seed=config.seed,
        spot_shocks_pct=grids["spot_shocks_pct"],
        volatility_shocks_abs=grids["volatility_shocks_abs"],
    )
    st.session_state["pricing_configuration"] = config
    st.session_state["frozen_market"] = market
    st.session_state["pricing_result"] = result
    st.session_state["diagnostics_result"] = diagnostics
    st.session_state["scenario_result"] = None
    st.session_state["risk_result"] = None


def render_workspace() -> None:
    _initialize_state()
    client = MlPricerApi(API_URL)
    experience_mode, n_paths, seed = render_sidebar()
    st.sidebar.markdown(f"[Open backend API docs ↗]({PUBLIC_API_URL}/docs)")
    st.sidebar.caption(
        "Browser-facing API link; the internal Docker service address stays hidden."
    )
    if st.sidebar.button("Check API connection", width="stretch"):
        if client.health():
            st.sidebar.success("Pricing API is ready.")
        else:
            st.sidebar.error("Pricing API is unavailable.")

    _render_header(experience_mode)
    config, input_error = render_configuration(
        experience_mode=experience_mode,
        n_paths=n_paths,
        seed=seed,
    )
    if input_error:
        st.error(input_error)
    if config is not None:
        try:
            spinner = (
                "Collecting today's numbers, imagining futures, and checking "
                "your rules…"
                if experience_mode == "Guided"
                else "Freezing the market, pricing paths, and building diagnostics…"
            )
            with st.spinner(spinner):
                _run_pricing(client, config)
            st.toast("Pricing workspace updated", icon="✅")
        except FrontendApiError as exc:
            st.error(str(exc))
        except Exception:
            st.error(
                "The workspace could not complete this run. The previous "
                "successful result, if any, is still available."
            )

    result = st.session_state.get("pricing_result")
    diagnostics = st.session_state.get("diagnostics_result")
    stored_config = st.session_state.get("pricing_configuration")
    market = st.session_state.get("frozen_market")
    if (
        result
        and diagnostics
        and stored_config
        and market
        and stored_config.experience_mode == experience_mode
    ):
        render_pricing_results(
            client,
            result=result,
            diagnostics=diagnostics,
            config=stored_config,
            market=market,
        )
    elif experience_mode == "Quant":
        _render_empty_state()
