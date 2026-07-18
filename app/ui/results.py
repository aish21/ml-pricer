import json
import math
from typing import Any, Mapping

import pandas as pd
import streamlit as st

from app.frontend_support import compact_nonzero_shock
from app.ui.api_client import FrontendApiError, NeuralPricerApi
from app.ui.charts import (
    PLOTLY_CONFIG,
    barrier_ladder_figure,
    cashflow_figure,
    contract_timeline_figure,
    convergence_figure,
    distribution_figure,
    price_uncertainty_figure,
    risk_figure,
    scenario_figure,
    surface_figure,
    term_structure_figure,
)
from app.ui.payloads import (
    PricingConfiguration,
    barrier_levels,
    even_observation_schedule,
)


def _chart(figure, *, key: str) -> None:
    st.plotly_chart(
        figure,
        use_container_width=True,
        config=PLOTLY_CONFIG,
        key=key,
    )


def _contract_context(
    config: PricingConfiguration,
    market: Mapping[str, Any],
) -> tuple[float, list[float]]:
    if config.contract is not None:
        return (
            float(config.contract["reference_level"]),
            [float(value) for value in config.contract["observation_times_years"]],
        )
    return (
        float(market["spot"]),
        list(
            even_observation_schedule(
                config.maturity_years,
                int(config.terms["obs_count"]),
            )
        ),
    )


def _metric_text(value: Any, digits: int = 6) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(numeric):
        return "—"
    return f"{numeric:,.{digits}f}"


def _render_guided_summary(
    diagnostics: Mapping[str, Any],
    *,
    reference_level: float,
    live_spot: float,
) -> None:
    cashflows = diagnostics.get("cashflows") or {}
    spot_to_reference = live_spot / reference_level
    columns = st.columns(3)
    with columns[0]:
        st.markdown(
            f"""
            <div class="np-card">
              <h4>Where is the underlier?</h4>
              <p>Live spot is <b>{spot_to_reference:.1%}</b> of the contractual
              reference. Barriers remain tied to the reference, not today’s spot.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with columns[1]:
        st.markdown(
            f"""
            <div class="np-card">
              <h4>Autocall likelihood</h4>
              <p>About <b>{float(cashflows.get("autocall_probability", 0.0)):.1%}</b>
              of simulated paths redeem early under this market model.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with columns[2]:
        st.markdown(
            f"""
            <div class="np-card">
              <h4>Downside likelihood</h4>
              <p>About <b>{float(cashflows.get("downside_probability", 0.0)):.1%}</b>
              of paths finish with downside redemption after a knock-in.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _render_overview(
    *,
    result: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
    config: PricingConfiguration,
    market: Mapping[str, Any],
) -> None:
    confidence_interval = result.get("confidence_interval") or [None, None]
    metric_columns = st.columns(5)
    metric_columns[0].metric("Reference price", _metric_text(result.get("price")))
    metric_columns[1].metric(
        "Value at notional",
        f"{float(result.get('price', 0.0)) * config.display_notional:,.2f}",
    )
    metric_columns[2].metric(
        "95% interval",
        (
            f"{_metric_text(confidence_interval[0], 5)} – "
            f"{_metric_text(confidence_interval[1], 5)}"
        ),
    )
    metric_columns[3].metric(
        "Monte Carlo SE",
        _metric_text(result.get("standard_error")),
    )
    metric_columns[4].metric("Paths", f"{int(result.get('n_paths', 0)):,}")

    reference_level, _ = _contract_context(config, market)
    if config.experience_mode == "Guided":
        _render_guided_summary(
            diagnostics,
            reference_level=reference_level,
            live_spot=float(market["spot"]),
        )
        st.markdown("")

    left, right = st.columns([1.05, 1], gap="large")
    with left:
        _chart(price_uncertainty_figure(result), key="price_uncertainty")
    with right:
        _chart(
            barrier_ladder_figure(
                barrier_levels(
                    live_spot=float(market["spot"]),
                    reference_level=reference_level,
                    terms=config.terms,
                )
            ),
            key="barrier_ladder",
        )

    if config.experience_mode == "Guided":
        st.info(
            "The price is a present value per unit notional, not a forecast of "
            "the underlier. The confidence interval measures Monte Carlo noise; "
            "it does not include model or market-data uncertainty."
        )


def _render_contract_and_market(
    *,
    result: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
    config: PricingConfiguration,
    market: Mapping[str, Any],
) -> None:
    reference_level, observations = _contract_context(config, market)
    _chart(
        contract_timeline_figure(observations, config.maturity_years),
        key="contract_timeline",
    )
    left, right = st.columns([1.15, 0.85], gap="large")
    with left:
        _chart(term_structure_figure(market), key="term_structure")
    with right:
        st.markdown("#### Contract levels")
        level_frame = pd.DataFrame(
            barrier_levels(
                live_spot=float(market["spot"]),
                reference_level=reference_level,
                terms=config.terms,
            )
        )[["name", "level"]]
        st.dataframe(
            level_frame,
            use_container_width=True,
            hide_index=True,
            column_config={"level": st.column_config.NumberColumn(format="%.4f")},
        )
        st.markdown("#### Market provenance")
        st.caption(
            f"{market.get('symbol')} · {market.get('underlier_type')} · "
            f"{market.get('currency')} · {market.get('source')}"
        )
        st.code(str(market.get("term_structure_id", "request-derived market")))

    if config.experience_mode == "Quant":
        st.markdown("#### Normalized pricing parameters")
        st.dataframe(
            pd.DataFrame(
                [
                    {"parameter": name, "value": value}
                    for name, value in (result.get("params") or {}).items()
                ]
            ),
            hide_index=True,
            use_container_width=True,
        )
        st.caption(
            f"Contract version: {result.get('contract_version')} · "
            f"Market model: {result.get('model_version')} · "
            f"Diagnostics: {diagnostics.get('diagnostic_version')}"
        )


def _render_diagnostics(diagnostics: Mapping[str, Any]) -> None:
    cashflows = diagnostics.get("cashflows") or {}
    metrics = st.columns(3)
    metrics[0].metric(
        "Autocall probability",
        f"{float(cashflows.get('autocall_probability', 0.0)):.1%}",
    )
    metrics[1].metric(
        "Downside probability",
        f"{float(cashflows.get('downside_probability', 0.0)):.1%}",
    )
    metrics[2].metric(
        "Expected paid coupons",
        f"{float(cashflows.get('expected_coupon_count', 0.0)):.2f}",
    )
    left, right = st.columns(2, gap="large")
    with left:
        _chart(convergence_figure(diagnostics), key="convergence")
    with right:
        _chart(cashflow_figure(diagnostics), key="cashflows")
    left, right = st.columns([0.9, 1.1], gap="large")
    with left:
        _chart(distribution_figure(diagnostics), key="distribution")
    with right:
        _chart(surface_figure(diagnostics), key="surface")
    st.caption(
        "The surface reuses the same normal draws across every cell. This "
        "reduces simulation noise when comparing neighboring spot/volatility scenarios."
    )


def _render_scenario_lab(
    client: NeuralPricerApi,
    *,
    config: PricingConfiguration,
    market: Mapping[str, Any],
) -> None:
    with st.form("scenario_lab"):
        st.markdown("#### Scenario builder")
        columns = st.columns(4)
        spot_pct = columns[0].number_input(
            "Spot shock (%)", value=-10.0, min_value=-90.0, max_value=100.0
        )
        volatility_abs = columns[1].number_input(
            "Volatility shift", value=0.02, min_value=-1.0, max_value=1.0
        )
        rate_bps = columns[2].number_input(
            "Rate shift (bp)", value=0.0, min_value=-10_000.0, max_value=10_000.0
        )
        dividend_bps = columns[3].number_input(
            "Dividend shift (bp)",
            value=0.0,
            min_value=-10_000.0,
            max_value=10_000.0,
        )
        submitted = st.form_submit_button(
            "Run paired scenario",
            use_container_width=True,
        )
    if submitted:
        shock = compact_nonzero_shock(
            spot_pct=spot_pct,
            rate_parallel_bps=rate_bps,
            dividend_parallel_bps=dividend_bps,
            volatility_parallel_abs=volatility_abs,
        )
        if not shock:
            st.warning("Choose at least one non-zero scenario shock.")
        else:
            try:
                with st.spinner("Running paired scenario paths…"):
                    st.session_state["scenario_result"] = client.scenario(
                        market=market,
                        terms=config.terms,
                        shock=shock,
                        n_paths=config.n_paths,
                        seed=config.seed,
                    )
            except FrontendApiError as exc:
                st.error(str(exc))
    scenario_result = st.session_state.get("scenario_result")
    if scenario_result:
        pnl = scenario_result["pnl"]
        columns = st.columns(3)
        columns[0].metric(
            "Base value",
            _metric_text(scenario_result["base_valuation"]["price"]),
        )
        columns[1].metric(
            "Shocked value",
            _metric_text(scenario_result["shocked_valuation"]["price"]),
        )
        columns[2].metric(
            "Scenario P&L",
            _metric_text(pnl["value"]),
            delta=f"SE {_metric_text(pnl['standard_error'])}",
        )
        _chart(scenario_figure(scenario_result), key="scenario_bridge")


def _render_risk_lab(
    client: NeuralPricerApi,
    *,
    config: PricingConfiguration,
    market: Mapping[str, Any],
) -> None:
    with st.form("risk_lab"):
        st.markdown("#### Greek bump sizes")
        columns = st.columns(4)
        spot_relative = columns[0].number_input(
            "Spot relative", value=0.01, min_value=0.0001, max_value=0.5
        )
        volatility_absolute = columns[1].number_input(
            "Volatility", value=0.01, min_value=0.0001, max_value=1.0
        )
        rate_bps = columns[2].number_input(
            "Rate (bp)", value=10.0, min_value=0.01, max_value=5_000.0
        )
        dividend_bps = columns[3].number_input(
            "Dividend (bp)", value=10.0, min_value=0.01, max_value=5_000.0
        )
        submitted = st.form_submit_button(
            "Calculate finite-difference Greeks",
            use_container_width=True,
        )
    if submitted:
        try:
            with st.spinner("Running common-random-number bumps…"):
                st.session_state["risk_result"] = client.risk(
                    market=market,
                    terms=config.terms,
                    bumps={
                        "spot_relative": spot_relative,
                        "volatility_absolute": volatility_absolute,
                        "rate_bps": rate_bps,
                        "dividend_bps": dividend_bps,
                    },
                    n_paths=config.n_paths,
                    seed=config.seed,
                )
        except FrontendApiError as exc:
            st.error(str(exc))
    risk_result = st.session_state.get("risk_result")
    if risk_result:
        _chart(risk_figure(risk_result), key="risk_bars")
        rows = [
            {
                "risk": name,
                "value": item["value"],
                "standard_error": item["standard_error"],
                "95% signal": item["statistically_resolved_95pct"],
                "units": item["units"],
            }
            for name, item in risk_result["sensitivities"].items()
        ]
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


def _render_risk_workspace(
    client: NeuralPricerApi,
    *,
    config: PricingConfiguration,
    market: Mapping[str, Any],
) -> None:
    if config.is_seasoned:
        st.info(
            "Seasoned-trade scenario and Greek routes are the next backend risk "
            "phase. The valuation surface in Diagnostics already freezes the v2 "
            "contract while shocking spot and volatility."
        )
        return
    scenario_tab, risk_tab = st.tabs(["Scenario", "Greeks"])
    with scenario_tab:
        _render_scenario_lab(client, config=config, market=market)
    with risk_tab:
        _render_risk_lab(client, config=config, market=market)


def _render_audit(
    *,
    result: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
    config: PricingConfiguration,
    market: Mapping[str, Any],
) -> None:
    columns = st.columns(3)
    columns[0].metric("Contract", str(result.get("contract_version")))
    columns[1].metric("Market model", str(result.get("model_version")))
    columns[2].metric("Seed", str(diagnostics.get("provenance", {}).get("seed")))
    st.markdown("#### Reproducibility identifiers")
    st.code(
        "\n".join(
            [
                f"market: {market.get('term_structure_id', 'request-derived')}",
                f"contract: {(result.get('contract') or {}).get('contract_id', result.get('contract_version'))}",
                f"diagnostic: {diagnostics.get('diagnostic_id')}",
            ]
        )
    )
    with st.expander("Normalized request"):
        st.json(
            {
                "configuration": config.__dict__,
                "market": market,
            }
        )
    with st.expander("Pricing response"):
        st.json(result)
    with st.expander("Diagnostic response"):
        st.json(diagnostics)
    export = json.dumps(
        {"pricing": result, "diagnostics": diagnostics},
        indent=2,
        sort_keys=True,
    ).encode("utf-8")
    st.download_button(
        "Download run JSON",
        data=export,
        file_name="neural-pricer-run.json",
        mime="application/json",
    )


def render_pricing_results(
    client: NeuralPricerApi,
    *,
    result: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
    config: PricingConfiguration,
    market: Mapping[str, Any],
) -> None:
    st.markdown(
        f"### {config.symbol} · {config.trade_stage} · "
        f"{result.get('contract_version')}"
    )
    st.caption(
        f"Market as of {market.get('market_data_time')} · "
        f"source {market.get('source')} · price per unit notional"
    )
    tabs = st.tabs(
        [
            "Overview",
            "Contract & market",
            "Diagnostics",
            "Risk lab",
            "Audit",
        ]
    )
    with tabs[0]:
        _render_overview(
            result=result,
            diagnostics=diagnostics,
            config=config,
            market=market,
        )
    with tabs[1]:
        _render_contract_and_market(
            result=result,
            diagnostics=diagnostics,
            config=config,
            market=market,
        )
    with tabs[2]:
        _render_diagnostics(diagnostics)
    with tabs[3]:
        _render_risk_workspace(client, config=config, market=market)
    with tabs[4]:
        _render_audit(
            result=result,
            diagnostics=diagnostics,
            config=config,
            market=market,
        )
