import os

import streamlit as st

from app.ui.api_client import FrontendApiError, NeuralPricerApi
from app.ui.inputs import render_configuration, render_sidebar
from app.ui.payloads import PricingConfiguration, diagnostic_grids
from app.ui.results import render_pricing_results


API_URL = os.getenv(
    "API_URL",
    "https://aish-ml-pricer-backend.up.railway.app",
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


def _render_header() -> None:
    st.markdown(
        """
        <div class="np-hero">
          <div class="np-eyebrow">Structured-product research workspace</div>
          <h1>Price the contract. See the mechanics.</h1>
          <p>
            A deterministic Monte Carlo reference pricer for Phoenix notes,
            with explicit market provenance, contract state, uncertainty,
            cashflow decomposition, and interactive risk diagnostics.
          </p>
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
                <div class="np-card">
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
    client: NeuralPricerApi,
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
    client = NeuralPricerApi(API_URL)
    experience_mode, n_paths, seed = render_sidebar()
    st.sidebar.caption(f"API · {API_URL}")
    if st.sidebar.button("Check API connection", use_container_width=True):
        if client.health():
            st.sidebar.success("Pricing API is ready.")
        else:
            st.sidebar.error("Pricing API is unavailable.")

    _render_header()
    config, input_error = render_configuration(
        experience_mode=experience_mode,
        n_paths=n_paths,
        seed=seed,
    )
    if input_error:
        st.error(input_error)
    if config is not None:
        try:
            with st.spinner(
                "Freezing the market, pricing paths, and building diagnostics…"
            ):
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
    if result and diagnostics and stored_config and market:
        render_pricing_results(
            client,
            result=result,
            diagnostics=diagnostics,
            config=stored_config,
            market=market,
        )
    else:
        _render_empty_state()
