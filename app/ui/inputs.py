from typing import Any

import streamlit as st

from app.ui.guided import render_guided_configuration
from app.ui.payloads import (
    FrontendInputError,
    PricingConfiguration,
    build_barrier_reverse_convertible_contract,
    build_flat_term_structure,
    build_phoenix_terms,
    build_v2_contract,
    build_v3_contract,
    even_observation_schedule,
    parse_observation_schedule,
    stepped_autocall_schedule,
)
from app.ui.underliers import render_underlier_picker


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
    st.sidebar.markdown("## ML Pricer")
    st.sidebar.caption("Structured-product research workspace")
    experience_mode = st.sidebar.radio(
        "Experience",
        ["Guided", "Quant"],
        help="Guided mode explains the finance. Quant mode exposes numerical controls.",
    )
    if experience_mode == "Guided":
        st.sidebar.markdown(
            """
            <div class="mlp-sidebar-lesson">
              <b>Learning journey</b>
              <span>Pick → Time → Rules → Simulate → Price</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        n_paths = st.sidebar.select_slider(
            "How many pretend futures?",
            options=[500, 1_000, 2_000, 5_000],
            value=2_000,
            format_func=lambda value: f"{value:,} stories",
            help=(
                "More stories usually make the answer steadier, but the "
                "computer takes a little longer."
            ),
        )
        seed = 42
    else:
        st.sidebar.selectbox("Contract preset", list(PRESETS), key="contract_preset")
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
    if experience_mode == "Guided":
        st.sidebar.caption(
            "Guided mode explains each pricing choice as you build the note."
        )
    else:
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
    symbol, underlier_type = render_underlier_picker(
        market_source=market_source,
        key_prefix="quant_underlier",
    )
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


def _render_barrier_reverse_convertible_configuration(
    *,
    n_paths: int,
    seed: int,
) -> tuple[PricingConfiguration | None, str | None]:
    trade_stage = st.radio(
        "Trade state",
        ["New issue", "Seasoned trade"],
        horizontal=True,
        help=(
            "A new issue takes the fetched live spot as its reference. A "
            "seasoned note keeps its original contractual reference."
        ),
        key="brc_trade_state",
    )
    market_source = st.radio(
        "Market source",
        ["Research market", "Manual flat market"],
        horizontal=True,
        key="brc_market_source",
    )
    with st.form("brc_configuration", border=False):
        market_column, contract_column = st.columns([1, 1], gap="large")
        with contract_column:
            st.markdown("### 2 · Contract")
            maturity = st.number_input(
                "Remaining maturity (years)",
                min_value=0.01,
                max_value=30.0,
                value=1.0,
                step=0.25,
                key="brc_maturity",
            )
            display_notional = st.number_input(
                "Display notional",
                min_value=1.0,
                value=100_000.0,
                step=10_000.0,
                key="brc_notional",
            )
            coupon_count = int(
                st.number_input(
                    "Remaining coupon payments",
                    min_value=1,
                    max_value=252,
                    value=4,
                    step=1,
                )
            )
            coupon_rate = st.number_input(
                "Coupon per payment",
                min_value=0.0,
                max_value=1.0,
                value=0.025,
                step=0.0025,
                format="%.4f",
                help=(
                    "This simplified product pays this amount on every coupon "
                    "date, even if the underlier falls."
                ),
            )
            strike = st.number_input(
                "Conversion strike",
                min_value=0.01,
                max_value=3.0,
                value=1.0,
                step=0.01,
                help="Fraction of the contractual reference level.",
            )
            knock_in = st.number_input(
                "Knock-in barrier",
                min_value=0.01,
                max_value=min(1.0, float(strike)),
                value=min(0.7, float(strike)),
                step=0.01,
                help=(
                    "If this line is breached and the final level is below the "
                    "strike, principal becomes linked to the underlier loss."
                ),
            )
            reference_level = 100.0
            prior_knock_in = False
            if trade_stage == "Seasoned trade":
                reference_level = st.number_input(
                    "Original reference level",
                    min_value=0.000001,
                    value=100.0,
                    step=1.0,
                    key="brc_reference",
                )
                prior_knock_in = st.checkbox(
                    "Knock-in was breached before valuation",
                    value=False,
                    key="brc_prior_knock_in",
                )
            else:
                st.caption(
                    "For a new issue, the server freezes the fetched live spot "
                    "as the contractual reference before pricing."
                )
        with market_column:
            st.markdown("### 1 · Market")
            symbol, underlier_type, currency, manual_market = _market_fields(
                market_source=market_source,
                experience_mode="Quant",
                maturity=maturity,
            )
        submitted = st.form_submit_button(
            "Price and build diagnostics",
            width="stretch",
        )
    if not submitted:
        return None, None
    try:
        coupon_times = even_observation_schedule(maturity, coupon_count)
        contract = build_barrier_reverse_convertible_contract(
            reference_level=float(reference_level),
            maturity_years=float(maturity),
            coupon_times_years=coupon_times,
            coupon_rate_per_period=float(coupon_rate),
            strike_frac=float(strike),
            knock_in_frac=float(knock_in),
            prior_knock_in_breached=prior_knock_in,
        )
        return (
            PricingConfiguration(
                experience_mode="Quant",
                trade_stage=trade_stage,
                market_source=market_source,
                symbol=symbol.strip().upper(),
                underlier_type=underlier_type,
                currency=currency,
                maturity_years=float(maturity),
                display_notional=float(display_notional),
                n_paths=n_paths,
                seed=seed,
                terms={
                    "maturity_years": float(maturity),
                    "coupon_rate_per_period": float(coupon_rate),
                    "strike_frac": float(strike),
                    "knock_in_frac": float(knock_in),
                    "obs_count": coupon_count,
                },
                contract=contract,
                manual_market=manual_market,
                product_key="barrier_reverse_convertible",
            ),
            None,
        )
    except FrontendInputError as exc:
        return None, str(exc)


def render_configuration(
    *,
    experience_mode: str,
    n_paths: int,
    seed: int,
) -> tuple[PricingConfiguration | None, str | None]:
    if experience_mode == "Guided":
        return render_guided_configuration(n_paths=n_paths, seed=seed)

    product = st.radio(
        "Product",
        ["Phoenix autocallable", "Barrier reverse convertible"],
        horizontal=True,
        help=(
            "The Phoenix can end early. The reverse convertible is simpler: "
            "fixed coupons plus conditional downside at maturity."
        ),
    )
    if product == "Barrier reverse convertible":
        return _render_barrier_reverse_convertible_configuration(
            n_paths=n_paths,
            seed=seed,
        )

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
            memory_coupon = False
            stepdown_autocall = False
            unpaid_coupon_count = 0
            final_autocall = float(autocall)
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
                st.markdown("#### Richer payoff rules")
                memory_coupon = st.checkbox(
                    "Remember missed coupons",
                    value=False,
                    help=(
                        "If a coupon is missed, it is carried forward. A later "
                        "successful coupon observation pays the current coupon "
                        "plus the carried coupons."
                    ),
                )
                if memory_coupon:
                    unpaid_coupon_count = int(
                        st.number_input(
                            "Missed coupons carried into today",
                            min_value=0,
                            max_value=252,
                            value=0,
                            step=1,
                            help=(
                                "Historical missed coupons that remain unpaid "
                                "at the valuation date."
                            ),
                        )
                    )
                stepdown_autocall = st.checkbox(
                    "Lower the autocall level over time",
                    value=False,
                    help=(
                        "The early-exit hurdle moves linearly from the first "
                        "autocall barrier to the final barrier."
                    ),
                )
                if stepdown_autocall:
                    final_autocall = st.number_input(
                        "Final autocall barrier",
                        min_value=float(coupon_barrier),
                        max_value=float(autocall),
                        value=max(
                            float(coupon_barrier),
                            min(float(autocall), float(autocall) - 0.10),
                        ),
                        step=0.01,
                        help="Fraction of the original reference level.",
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
            width="stretch",
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
            if memory_coupon or stepdown_autocall:
                autocall_schedule = stepped_autocall_schedule(
                    initial_barrier_frac=autocall,
                    final_barrier_frac=final_autocall,
                    observation_count=len(schedule),
                )
                terms = {
                    **terms,
                    "autocall_barrier_fracs": list(autocall_schedule),
                    "memory_coupon": memory_coupon,
                    "unpaid_coupon_count": unpaid_coupon_count,
                }
                contract = build_v3_contract(
                    reference_level=float(reference_level),
                    terms=terms,
                    observation_times_years=schedule,
                    autocall_barrier_fracs=autocall_schedule,
                    prior_knock_in_breached=prior_knock_in,
                    memory_coupon=memory_coupon,
                    unpaid_coupon_count=unpaid_coupon_count,
                )
            else:
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
