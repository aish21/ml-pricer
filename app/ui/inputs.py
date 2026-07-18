from typing import Any

import streamlit as st

from app.ui.payloads import (
    FrontendInputError,
    PricingConfiguration,
    build_flat_term_structure,
    build_phoenix_terms,
    build_v2_contract,
    even_observation_schedule,
    parse_observation_schedule,
)


PRESETS: dict[str, dict[str, float]] = {
    "Balanced Phoenix": {
        "autocall": 1.05,
        "coupon_barrier": 1.0,
        "coupon_rate": 0.02,
        "knock_in": 0.7,
    },
    "Defensive barriers": {
        "autocall": 1.0,
        "coupon_barrier": 0.8,
        "coupon_rate": 0.0125,
        "knock_in": 0.6,
    },
    "Higher coupon": {
        "autocall": 1.1,
        "coupon_barrier": 1.0,
        "coupon_rate": 0.03,
        "knock_in": 0.7,
    },
}


def render_sidebar() -> tuple[str, int, int]:
    st.sidebar.markdown("## Neural Pricer")
    st.sidebar.caption("Structured-product research workspace")
    experience_mode = st.sidebar.radio(
        "Experience",
        ["Guided", "Quant"],
        help="Guided mode explains the finance. Quant mode exposes numerical controls.",
    )
    st.sidebar.selectbox("Contract preset", list(PRESETS), key="contract_preset")
    if experience_mode == "Guided":
        n_paths = st.sidebar.select_slider(
            "Monte Carlo quality",
            options=[500, 1_000, 2_000, 5_000],
            value=2_000,
            format_func=lambda value: f"{value:,} paths",
        )
        seed = 42
    else:
        n_paths = int(
            st.sidebar.selectbox(
                "Monte Carlo paths",
                [500, 1_000, 2_000, 5_000],
                index=2,
            )
        )
        seed = int(
            st.sidebar.number_input(
                "Random seed",
                min_value=0,
                max_value=4_294_967_295,
                value=42,
            )
        )
    st.sidebar.markdown("---")
    st.sidebar.caption(
        "Reference prices remain Monte Carlo estimates. Inspect confidence "
        "intervals before interpreting small differences."
    )
    return experience_mode, int(n_paths), seed


def _market_fields(
    *,
    market_source: str,
    experience_mode: str,
    maturity: float,
) -> tuple[str, str, str, dict[str, Any] | None]:
    symbol = st.text_input(
        "Underlier",
        value="SPY",
        help="An equity or ETF symbol such as SPY, QQQ, or AAPL.",
    )
    available_types = ["ETF", "Equity"]
    if market_source == "Manual flat market":
        available_types.append("Index")
    underlier_type = st.selectbox("Underlier type", available_types)
    if market_source == "Research market":
        st.caption(
            "The server builds a USD research curve from Treasury data, "
            "trailing distributions, and near-ATM Yahoo option quotes."
        )
        return symbol, underlier_type.lower(), "USD", None

    currency = st.text_input("Currency", value="USD", max_chars=3)
    spot = st.number_input(
        "Live spot",
        min_value=0.000001,
        value=100.0,
        step=1.0,
        help="Today’s market level. This is separate from a seasoned trade’s reference.",
    )
    rate = st.number_input(
        "Risk-free rate",
        min_value=-0.25,
        max_value=1.0,
        value=0.03,
        step=0.0025,
        format="%.4f",
    )
    dividend = st.number_input(
        "Dividend yield",
        min_value=-0.25,
        max_value=1.0,
        value=0.01,
        step=0.0025,
        format="%.4f",
    )
    volatility = st.number_input(
        "Volatility",
        min_value=0.000001,
        max_value=5.0,
        value=0.20,
        step=0.01,
        format="%.4f",
    )
    if experience_mode == "Guided":
        st.caption(
            "Rates, dividends, and volatility are annualized decimal inputs: "
            "0.20 means 20%."
        )
    market = build_flat_term_structure(
        symbol=symbol,
        underlier_type=underlier_type,
        currency=currency,
        spot=spot,
        risk_free_rate=rate,
        dividend_yield=dividend,
        volatility=volatility,
        maturity_years=maturity,
    )
    return symbol, underlier_type.lower(), currency.upper(), market


def render_configuration(
    *,
    experience_mode: str,
    n_paths: int,
    seed: int,
) -> tuple[PricingConfiguration | None, str | None]:
    trade_stage = st.radio(
        "Trade state",
        ["New issue", "Seasoned trade"],
        horizontal=True,
        help=(
            "A new issue uses today’s spot as its reference. A seasoned trade "
            "keeps the original reference and historical knock-in state."
        ),
    )
    market_source = st.radio(
        "Market source",
        ["Research market", "Manual flat market"],
        horizontal=True,
    )
    preset_name = st.session_state.get("contract_preset", "Balanced Phoenix")
    preset = PRESETS.get(str(preset_name), PRESETS["Balanced Phoenix"])
    preset_key = str(preset_name).lower().replace(" ", "_")

    with st.form("phoenix_configuration", border=False):
        market_column, contract_column = st.columns([1, 1], gap="large")
        with contract_column:
            st.markdown("### 2 · Contract")
            maturity = st.number_input(
                "Remaining maturity (years)",
                min_value=0.01,
                max_value=30.0,
                value=1.0,
                step=0.25,
            )
            display_notional = st.number_input(
                "Display notional",
                min_value=1.0,
                value=100_000.0,
                step=10_000.0,
                help="Scales the displayed value only; pricing remains per unit notional.",
            )
            observation_count = int(
                st.number_input(
                    "Remaining observations",
                    min_value=1,
                    max_value=252,
                    value=6,
                    step=1,
                )
            )
            autocall = st.number_input(
                "Autocall barrier",
                min_value=0.01,
                max_value=3.0,
                value=float(preset["autocall"]),
                step=0.01,
                help="Fraction of the contractual reference level.",
                key=f"autocall_{preset_key}",
            )
            coupon_barrier = st.number_input(
                "Coupon barrier",
                min_value=0.01,
                max_value=3.0,
                value=float(preset["coupon_barrier"]),
                step=0.01,
                help="Coupon is paid when the observation level is at or above this.",
                key=f"coupon_barrier_{preset_key}",
            )
            coupon_rate = st.number_input(
                "Coupon per observation",
                min_value=0.0,
                max_value=1.0,
                value=float(preset["coupon_rate"]),
                step=0.0025,
                format="%.4f",
                key=f"coupon_rate_{preset_key}",
            )
            knock_in = st.number_input(
                "Knock-in barrier",
                min_value=0.01,
                max_value=1.0,
                value=float(preset["knock_in"]),
                step=0.01,
                help="Fraction of reference that activates downside exposure.",
                key=f"knock_in_{preset_key}",
            )
            reference_level = None
            prior_knock_in = False
            raw_schedule = ""
            if trade_stage == "Seasoned trade":
                reference_level = st.number_input(
                    "Original reference level",
                    min_value=0.000001,
                    value=100.0,
                    step=1.0,
                )
                prior_knock_in = st.checkbox(
                    "Knock-in was breached before valuation",
                    value=False,
                    help="This historical event remains relevant at maturity.",
                )
                if experience_mode == "Quant":
                    raw_schedule = st.text_input(
                        "Observation times (years)",
                        value="0.1667, 0.3333, 0.5000, 0.6667, 0.8333, 1.0000",
                        help=(
                            "Comma-separated ACT/365F-like year fractions from "
                            "valuation. The final time must equal maturity."
                        ),
                    )
                else:
                    st.caption(
                        "Guided mode spaces the remaining observations evenly. "
                        "Quant mode accepts exact year fractions."
                    )

        with market_column:
            st.markdown("### 1 · Market")
            symbol, underlier_type, currency, manual_market = _market_fields(
                market_source=market_source,
                experience_mode=experience_mode,
                maturity=maturity,
            )

        submitted = st.form_submit_button(
            "Price and build diagnostics",
            use_container_width=True,
        )

    if not submitted:
        return None, None
    try:
        terms = build_phoenix_terms(
            maturity_years=maturity,
            autocall_barrier_frac=autocall,
            coupon_barrier_frac=coupon_barrier,
            coupon_rate=coupon_rate,
            knock_in_frac=knock_in,
            observation_count=observation_count,
        )
        contract = None
        if trade_stage == "Seasoned trade":
            schedule = (
                parse_observation_schedule(raw_schedule, maturity)
                if experience_mode == "Quant"
                else even_observation_schedule(maturity, observation_count)
            )
            terms = {**terms, "obs_count": len(schedule)}
            contract = build_v2_contract(
                reference_level=float(reference_level),
                terms=terms,
                observation_times_years=schedule,
                prior_knock_in_breached=prior_knock_in,
            )
        return (
            PricingConfiguration(
                experience_mode=experience_mode,
                trade_stage=trade_stage,
                market_source=market_source,
                symbol=symbol.strip().upper(),
                underlier_type=underlier_type,
                currency=currency,
                maturity_years=float(maturity),
                display_notional=float(display_notional),
                n_paths=n_paths,
                seed=seed,
                terms=terms,
                contract=contract,
                manual_market=manual_market,
            ),
            None,
        )
    except FrontendInputError as exc:
        return None, str(exc)
