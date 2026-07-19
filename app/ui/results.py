import json
import math
from html import escape
from typing import Any, Mapping

import streamlit as st

from app.frontend_support import compact_nonzero_shock
from app.ui.api_client import FrontendApiError, MlPricerApi
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


def _chart(
    figure,
    *,
    key: str,
    guide: tuple[str, str, str] | None = None,
) -> None:
    st.plotly_chart(
        figure,
        width="stretch",
        config=PLOTLY_CONFIG,
        key=key,
    )
    if guide is not None:
        question, look_for, meaning = guide
        st.markdown(
            f"""
            <div class="mlp-chart-guide">
              <div><span>Question</span><b>{escape(question)}</b></div>
              <div><span>Look for</span><p>{escape(look_for)}</p></div>
              <div><span>Why it matters</span><p>{escape(meaning)}</p></div>
            </div>
            """,
            unsafe_allow_html=True,
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


def _safe_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _ml_comparison_summary(result: Mapping[str, Any]) -> dict[str, Any]:
    shadow = result.get("surrogate_shadow")
    if not isinstance(shadow, Mapping) or shadow.get("status") != "success":
        return {
            "available": False,
            "status": (
                str(shadow.get("status", "disabled"))
                if isinstance(shadow, Mapping)
                else "disabled"
            ),
            "reason": (
                str(shadow.get("reason", ""))
                if isinstance(shadow, Mapping)
                else ""
            ),
        }

    reference_price = _safe_float(shadow.get("reference_price", result.get("price")))
    ml_price = _safe_float(shadow.get("surrogate_price"))
    if reference_price is None or ml_price is None:
        return {
            "available": False,
            "status": "error",
            "reason": "the comparison response did not contain two valid prices",
        }
    signed_gap = ml_price - reference_price
    absolute_gap = abs(signed_gap)
    relative_gap = absolute_gap / abs(reference_price) if reference_price else None
    reference_latency_ms = _safe_float(result.get("latency_ms"))
    ml_latency_ms = _safe_float(shadow.get("latency_ms"))
    speedup = (
        reference_latency_ms / ml_latency_ms
        if reference_latency_ms is not None
        and ml_latency_ms is not None
        and ml_latency_ms > 0.0
        else None
    )
    interval = result.get("confidence_interval")
    inside_reference_interval = None
    if isinstance(interval, (list, tuple)) and len(interval) == 2:
        low, high = _safe_float(interval[0]), _safe_float(interval[1])
        if low is not None and high is not None:
            inside_reference_interval = low <= ml_price <= high

    return {
        "available": True,
        "status": "success",
        "reference_price": reference_price,
        "ml_price": ml_price,
        "signed_gap": signed_gap,
        "absolute_gap": absolute_gap,
        "relative_gap": relative_gap,
        "error_to_reference_standard_error": _safe_float(
            shadow.get("error_to_reference_standard_error")
        ),
        "reference_latency_ms": reference_latency_ms,
        "ml_latency_ms": ml_latency_ms,
        "speedup": speedup,
        "inside_reference_interval": inside_reference_interval,
        "validation_metrics": (
            dict(shadow["validation_metrics"])
            if isinstance(shadow.get("validation_metrics"), Mapping)
            else {}
        ),
        "model_version": str(shadow.get("model_version", "ML surrogate")),
    }


def _result_explanations(
    *,
    result: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
    config: PricingConfiguration,
    market: Mapping[str, Any],
) -> list[dict[str, str]]:
    price = float(result.get("price", 0.0))
    price_difference = price - 1.0
    if abs(price_difference) < 0.0005:
        price_body = (
            "The estimated value is very close to par (1.00), the unit amount "
            "used to describe the note."
        )
    elif price_difference > 0:
        price_body = (
            f"The estimate is {price_difference:.2%} above par. Under these "
            "assumptions, the discounted conditional payments are worth more "
            "than 1.00 per unit notional."
        )
    else:
        price_body = (
            f"The estimate is {abs(price_difference):.2%} below par. Under these "
            "assumptions, the conditional payments are worth less than 1.00 "
            "per unit notional."
        )

    reference_level, _ = _contract_context(config, market)
    spot = float(market["spot"])
    autocall_level = reference_level * float(config.terms["autocall_barrier_frac"])
    knock_in_level = reference_level * float(config.terms["knock_in_frac"])
    coupon_level = reference_level * float(config.terms["coupon_barrier_frac"])
    if spot >= autocall_level:
        barrier_body = (
            f"Today’s spot ({spot:,.2f}) is above the autocall line "
            f"({autocall_level:,.2f}). That does not end the note today: the "
            "test happens on a contractual observation date."
        )
    elif spot >= coupon_level:
        barrier_body = (
            f"Today’s spot ({spot:,.2f}) is above the coupon line "
            f"({coupon_level:,.2f}) but below the autocall line "
            f"({autocall_level:,.2f}). Only an observation-date level decides "
            "whether either payment rule fires."
        )
    elif spot > knock_in_level:
        barrier_body = (
            f"Today’s spot ({spot:,.2f}) sits between the coupon line "
            f"({coupon_level:,.2f}) and knock-in line ({knock_in_level:,.2f}). "
            "The buffer to the loss-linked rule is now smaller."
        )
    else:
        barrier_body = (
            f"Today’s spot ({spot:,.2f}) is at or below the knock-in line "
            f"({knock_in_level:,.2f}). A knock-in activates the maturity loss "
            "rule; it is not an immediate cash loss by itself."
        )

    cashflows = diagnostics.get("cashflows") or {}
    autocall_probability = float(cashflows.get("autocall_probability", 0.0))
    downside_probability = float(cashflows.get("downside_probability", 0.0))
    outcome_body = (
        f"In this simulation, {autocall_probability:.1%} of paths ended early "
        f"and {downside_probability:.1%} ended with loss-linked redemption. "
        "These are model frequencies under the chosen assumptions, not odds "
        "quoted by the market and not promises about the future."
    )

    explanations = [
        {"title": "What the price says", "body": price_body},
        {"title": "Where the contract stands", "body": barrier_body},
        {"title": "What happened in the pretend futures", "body": outcome_body},
    ]
    comparison = _ml_comparison_summary(result)
    if comparison["available"]:
        explanations.append(
            {
                "title": "What the ML comparison says",
                "body": (
                    f"The ML estimate differs from Monte Carlo by "
                    f"{comparison['absolute_gap']:.6f} per unit "
                    f"({comparison['relative_gap']:.2%}). It is being observed "
                    "in shadow mode, so the displayed reference price still "
                    "comes from Monte Carlo."
                ),
            }
        )
    return explanations


def _interpretation_html(explanations: list[Mapping[str, str]]) -> str:
    cards = "".join(
        '<div class="mlp-interpretation">'
        f"<span>{index:02d}</span>"
        f"<div><strong>{escape(item['title'])}</strong>"
        f"<p>{escape(item['body'])}</p></div>"
        "</div>"
        for index, item in enumerate(explanations, start=1)
    )
    return f'<div class="mlp-interpretation-list">{cards}</div>'


def _render_ml_comparison(
    result: Mapping[str, Any],
    *,
    guided: bool,
) -> None:
    comparison = _ml_comparison_summary(result)
    st.markdown(
        "#### Two estimators, one contract"
        if guided
        else "#### Monte Carlo benchmark vs ML surrogate"
    )
    if not comparison["available"]:
        status = comparison["status"].replace("_", " ").title()
        reason = comparison["reason"] or (
            "ML shadow pricing is not enabled for this run."
        )
        st.markdown(
            f"""
            <div class="mlp-ml-status">
              <span>ML comparison · {escape(status)}</span>
              <strong>The reference price is still available.</strong>
              <p>{escape(reason)}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    relative_gap = comparison["relative_gap"]
    speedup = comparison["speedup"]
    interval_label = comparison["inside_reference_interval"]
    metrics = st.columns(5)
    metrics[0].metric(
        "Slow reference" if guided else "Monte Carlo reference",
        _metric_text(comparison["reference_price"]),
        help="A path-simulation benchmark, not an executable traded quote.",
    )
    metrics[1].metric(
        "Learned shortcut" if guided else "ML surrogate",
        _metric_text(comparison["ml_price"]),
        delta=f"{comparison['signed_gap']:+.6f} vs MC",
        help="The neural surrogate learned the mapping from inputs to prices.",
    )
    metrics[2].metric(
        "Difference" if guided else "Absolute gap",
        _metric_text(comparison["absolute_gap"]),
        help="Absolute difference between the ML estimate and Monte Carlo reference.",
    )
    metrics[3].metric(
        "Difference in %" if guided else "Relative gap",
        f"{relative_gap:.3%}" if relative_gap is not None else "—",
    )
    metrics[4].metric(
        "Shortcut speed" if guided else "Observed speed-up",
        f"{speedup:,.1f}×" if speedup is not None else "Too fast to time",
        help="Reference latency divided by ML inference latency for this request.",
    )
    interval_text = (
        "inside"
        if interval_label is True
        else "outside"
        if interval_label is False
        else "not compared with"
    )
    st.markdown(
        f"""
        <div class="mlp-ml-explainer">
          <div>
            <span>What is happening?</span>
            <p>Monte Carlo prices thousands of invented paths. The ML model has
            learned a much faster approximation from a large collection of
            contracts that were priced in advance.</p>
          </div>
          <div>
            <span>Can ML change the answer?</span>
            <p>No. It runs in <b>shadow mode</b>: we measure it, but the served
            price remains Monte Carlo. This ML estimate is {interval_text} the
            reference simulation’s 95% noise interval.</p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    validation = comparison["validation_metrics"]
    if validation:
        with st.expander("How did this ML model perform on examples it never trained on?"):
            st.write(
                "These frozen audit numbers come from a held-out dataset. "
                "Lower error is better; R² closer to 1 means the model explains "
                "more of the reference-price variation."
            )
            audit_columns = st.columns(3)
            audit_columns[0].metric(
                "Mean absolute error",
                _metric_text(validation.get("mean_absolute_error")),
            )
            audit_columns[1].metric(
                "95th-percentile error",
                _metric_text(validation.get("p95_absolute_error")),
            )
            audit_columns[2].metric(
                "R²",
                _metric_text(validation.get("r_squared"), 4),
            )
            st.caption(
                "Audit passed · model version "
                f"{comparison['model_version']} · values are per unit notional"
            )


def _render_result_interpreter(
    *,
    result: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
    config: PricingConfiguration,
    market: Mapping[str, Any],
) -> None:
    st.markdown(
        "#### Let’s translate the result"
        if config.experience_mode == "Guided"
        else "#### Result interpretation"
    )
    st.markdown(
        _interpretation_html(
            _result_explanations(
                result=result,
                diagnostics=diagnostics,
                config=config,
                market=market,
            )
        ),
        unsafe_allow_html=True,
    )


def _table_cell(value: Any, number_format: str | None = None) -> str:
    if number_format is not None:
        try:
            return format(float(value), number_format)
        except (TypeError, ValueError):
            pass
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, (Mapping, list, tuple)):
        return json.dumps(value, sort_keys=True)
    return str(value)


def _table_html(
    rows: list[Mapping[str, Any]],
    columns: list[tuple[str, str]],
    *,
    number_formats: Mapping[str, str] | None = None,
) -> str:
    formats = number_formats or {}
    headings = "".join(f"<th>{escape(label)}</th>" for _, label in columns)
    body = "".join(
        "<tr>"
        + "".join(
            ("<td>" + escape(_table_cell(row.get(key, ""), formats.get(key))) + "</td>")
            for key, _ in columns
        )
        + "</tr>"
        for row in rows
    )
    return (
        '<div class="mlp-table-wrap"><table class="mlp-table">'
        f"<thead><tr>{headings}</tr></thead><tbody>{body}</tbody>"
        "</table></div>"
    )


def _render_table(
    rows: list[Mapping[str, Any]],
    columns: list[tuple[str, str]],
    *,
    number_formats: Mapping[str, str] | None = None,
) -> None:
    st.markdown(
        _table_html(rows, columns, number_formats=number_formats),
        unsafe_allow_html=True,
    )


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
            <div class="mlp-card">
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
            <div class="mlp-card">
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
            <div class="mlp-card">
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
    guided = config.experience_mode == "Guided"
    if guided:
        price = float(result.get("price", 0.0))
        st.markdown(
            f"""
            <div class="mlp-lesson-answer">
              <span>Your computer's short answer</span>
              <strong>About {price:.4f} today for each 1.0000 promised unit</strong>
              <p>
                For your pretend amount of {config.display_notional:,.0f}, that is
                about {price * config.display_notional:,.2f}. This is a model
                estimate—not a shop price or a promise about the future.
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("#### Read the answer from left to right")
    metric_columns = st.columns(5)
    metric_columns[0].metric(
        "Value for each 1" if guided else "Reference price",
        _metric_text(result.get("price")),
    )
    metric_columns[1].metric(
        "Value for your amount" if guided else "Value at notional",
        f"{float(result.get('price', 0.0)) * config.display_notional:,.2f}",
    )
    metric_columns[2].metric(
        "Most likely computer range" if guided else "95% interval",
        (
            f"{_metric_text(confidence_interval[0], 5)} – "
            f"{_metric_text(confidence_interval[1], 5)}"
        ),
    )
    metric_columns[3].metric(
        "Computer wobble" if guided else "Monte Carlo SE",
        _metric_text(result.get("standard_error")),
    )
    metric_columns[4].metric(
        "Pretend futures" if guided else "Paths",
        f"{int(result.get('n_paths', 0)):,}",
    )

    st.markdown("")
    _render_ml_comparison(result, guided=guided)

    reference_level, _ = _contract_context(config, market)
    if config.experience_mode == "Guided":
        _render_guided_summary(
            diagnostics,
            reference_level=reference_level,
            live_spot=float(market["spot"]),
        )
        st.markdown("")
        with st.expander("What does “computer wobble” mean?"):
            st.write(
                "We only asked the computer to imagine a limited number of "
                "futures. If we run it again with different pretend futures, "
                "the answer moves a little. The range and wobble estimate how "
                "much of that movement comes from simulation luck."
            )

    _render_result_interpreter(
        result=result,
        diagnostics=diagnostics,
        config=config,
        market=market,
    )

    left, right = st.columns([1.05, 1], gap="large")
    with left:
        _chart(
            price_uncertainty_figure(result),
            key="price_uncertainty",
            guide=(
                (
                    "What does the model think the note is worth today?",
                    "Compare the bright dot with par at 1.00. The whisker only "
                    "shows computer wobble.",
                    "A price below 1.00 is a discount to notional; above 1.00 "
                    "is a premium. Neither is an underlier forecast.",
                )
                if guided
                else None
            ),
        )
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
            guide=(
                (
                    "Which contract rule is closest to today's price?",
                    "Find the diamond for live spot, then read the colored rule "
                    "zones above and below it.",
                    "A nearby barrier can make the note sensitive to a small "
                    "market move.",
                )
                if guided
                else None
            ),
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
    autocall_level = reference_level * float(config.terms["autocall_barrier_frac"])
    coupon_level = reference_level * float(config.terms["coupon_barrier_frac"])
    knock_in_level = reference_level * float(config.terms["knock_in_frac"])
    if config.experience_mode == "Guided":
        st.markdown("#### The rulebook, in ordinary language")
        st.markdown(
            f"""
            <div class="mlp-rulebook">
              <div>
                <span>01 · Ask for a reward</span>
                <strong>Coupon barrier · {coupon_level:,.2f}</strong>
                <p>On each observation date, a spot at or above this line pays
                the {float(config.terms["coupon_rate"]):.2%} coupon. This note
                has no memory: a missed coupon is not saved for later.</p>
              </div>
              <div>
                <span>02 · Ask whether to finish</span>
                <strong>Autocall barrier · {autocall_level:,.2f}</strong>
                <p>On an observation date, a spot at or above this line returns
                principal and ends the note early. “Autocall” means the contract
                calls itself back automatically—not that the investor presses
                a button.</p>
              </div>
              <div>
                <span>03 · Remember serious falls</span>
                <strong>Knock-in barrier · {knock_in_level:,.2f}</strong>
                <p>Crossing this line switches on a loss-linked maturity rule.
                It does not create an instant loss. If the note survives to
                maturity, the final underlier level decides the redemption.</p>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        with st.expander("The six contract words worth learning"):
            st.markdown(
                """
                - **Reference level:** the original starting line used to calculate every barrier.
                - **Observation date:** a scheduled day when coupon and autocall rules are tested.
                - **Principal:** the original unit amount—1.00 in the charts—before coupons.
                - **Maturity:** the scheduled final day if the note has not autocalled.
                - **Barrier:** a contract threshold, not a forecast or a physical wall.
                - **Notional:** the money amount used to scale a price quoted per unit.
                """
            )
        st.markdown("#### First, follow the clock")
        st.write(
            "Each orange dot is a day when the note checks the underlier. "
            "The final dotted line is the ending day."
        )
    _chart(
        contract_timeline_figure(observations, config.maturity_years),
        key="contract_timeline",
        guide=(
            (
                "When is the note allowed to check its rules?",
                "Each numbered point is an observation. The final marker is maturity.",
                "The price path matters on these contract dates, not just today "
                "and the ending day.",
            )
            if config.experience_mode == "Guided"
            else None
        ),
    )
    left, right = st.columns([1.15, 0.85], gap="large")
    with left:
        if config.experience_mode == "Guided":
            st.markdown("#### Then see the market ingredients")
            st.caption(
                "Rates describe waiting, distributions describe cash paid by "
                "the asset, and volatility describes how wiggly prices may be."
            )
        _chart(
            term_structure_figure(market),
            key="term_structure",
            guide=(
                (
                    "What assumptions does the model use at each future time?",
                    "Read rates and distributions on the left axis and "
                    "wiggliness on the right.",
                    "Future paths and future-money discounts can change from "
                    "one time segment to another.",
                )
                if config.experience_mode == "Guided"
                else None
            ),
        )
    with right:
        st.markdown(
            "#### Your three lines"
            if config.experience_mode == "Guided"
            else "#### Contract levels"
        )
        _render_table(
            barrier_levels(
                live_spot=float(market["spot"]),
                reference_level=reference_level,
                terms=config.terms,
            ),
            [("name", "Level"), ("level", "Value")],
            number_formats={"level": ",.4f"},
        )
        st.markdown("#### Market provenance")
        st.caption(
            f"{market.get('symbol')} · {market.get('underlier_type')} · "
            f"{market.get('currency')} · {market.get('source')}"
        )
        st.code(str(market.get("term_structure_id", "request-derived market")))

    if config.experience_mode == "Quant":
        st.markdown("#### Normalized pricing parameters")
        _render_table(
            [
                {"parameter": name, "value": value}
                for name, value in (result.get("params") or {}).items()
            ],
            [("parameter", "Parameter"), ("value", "Value")],
        )
        st.caption(
            f"Contract version: {result.get('contract_version')} · "
            f"Market model: {result.get('model_version')} · "
            f"Diagnostics: {diagnostics.get('diagnostic_version')}"
        )


def _render_diagnostics(
    diagnostics: Mapping[str, Any],
    *,
    guided: bool = False,
) -> None:
    cashflows = diagnostics.get("cashflows") or {}
    if guided:
        st.markdown("## How sure should we be about the answer?")
        st.write(
            "Now we inspect the pretend futures instead of blindly trusting "
            "one number. Hover over every chart, then use its reading key."
        )
    metrics = st.columns(3)
    metrics[0].metric(
        "Chance of finishing early" if guided else "Autocall probability",
        f"{float(cashflows.get('autocall_probability', 0.0)):.1%}",
    )
    metrics[1].metric(
        "Chance of a loss-linked ending" if guided else "Downside probability",
        f"{float(cashflows.get('downside_probability', 0.0)):.1%}",
    )
    metrics[2].metric(
        "Average rewards paid" if guided else "Expected paid coupons",
        f"{float(cashflows.get('expected_coupon_count', 0.0)):.2f}",
    )
    left, right = st.columns(2, gap="large")
    with left:
        _chart(
            convergence_figure(diagnostics),
            key="convergence",
            guide=(
                (
                    "Did we simulate enough pretend futures?",
                    "The estimate should settle while the shaded band becomes narrower.",
                    "More paths reduce random computer wobble; they do not repair "
                    "a wrong model or a wrong contract.",
                )
                if guided
                else None
            ),
        )
    with right:
        _chart(
            cashflow_figure(diagnostics),
            key="cashflows",
            guide=(
                (
                    "Which promised payments create the value?",
                    "Compare both the bar heights and the percentage labels.",
                    "The total price may come mostly from returned principal, "
                    "not from the eye-catching coupons.",
                )
                if guided
                else None
            ),
        )
    left, right = st.columns([0.9, 1.1], gap="large")
    with left:
        _chart(
            distribution_figure(diagnostics),
            key="distribution",
            guide=(
                (
                    "How common were the good and bad simulated endings?",
                    "Tall bars are common outcomes. The 5% line marks a bad-tail "
                    "reference and the median splits paths in half.",
                    "One average price can hide a lopsided range of possible payments.",
                )
                if guided
                else None
            ),
        )
    with right:
        _chart(
            surface_figure(diagnostics),
            key="surface",
            guide=(
                (
                    "What happens if price and wiggliness move together?",
                    "Read the printed price, then use red/green color for change "
                    "versus today's market.",
                    "This reveals nonlinear behavior: equal-size market moves "
                    "need not produce equal-size value changes.",
                )
                if guided
                else None
            ),
        )
    st.caption(
        "The surface reuses the same normal draws across every cell. This "
        "reduces simulation noise when comparing neighboring spot/volatility scenarios."
    )


def _render_scenario_lab(
    client: MlPricerApi,
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
            width="stretch",
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
    client: MlPricerApi,
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
            width="stretch",
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
        _render_table(
            rows,
            [
                ("risk", "Risk"),
                ("value", "Value"),
                ("standard_error", "Standard error"),
                ("95% signal", "95% signal"),
                ("units", "Units"),
            ],
            number_formats={"value": ",.6f", "standard_error": ",.6f"},
        )


def _render_risk_workspace(
    client: MlPricerApi,
    *,
    config: PricingConfiguration,
    market: Mapping[str, Any],
) -> None:
    guided = config.experience_mode == "Guided"
    if config.is_seasoned:
        st.info(
            (
                "This older note remembers its history. The simple what-if "
                "games for older notes are still being built; use the colored "
                "grid in “How sure are we?” for now."
                if guided
                else "Seasoned-trade scenario and Greek routes are the next "
                "backend risk phase. The valuation surface in Diagnostics "
                "already freezes the v2 contract while shocking spot and volatility."
            )
        )
        return
    if guided:
        st.markdown("## Play a what-if game")
        st.write(
            "Change one piece of the world and ask the computer to reuse the "
            "same pretend futures. Reusing them makes the comparison less noisy."
        )
    scenario_tab, risk_tab = st.tabs(
        ["Change the world", "Measure tiny nudges"]
        if guided
        else ["Scenario", "Greeks"]
    )
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
    if config.experience_mode == "Guided":
        st.markdown("## Show the computer's recipe")
        st.write(
            "These IDs and raw messages let another person repeat the same "
            "experiment. You can ignore them while learning—or open them when "
            "you are curious about how research becomes reproducible."
        )
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
        file_name="ml-pricer-run.json",
        mime="application/json",
    )


def render_pricing_results(
    client: MlPricerApi,
    *,
    result: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
    config: PricingConfiguration,
    market: Mapping[str, Any],
) -> None:
    guided = config.experience_mode == "Guided"
    if guided:
        st.markdown(f"### Your {config.symbol} learning result")
        st.caption(
            "Walk through the tabs from left to right. Each one answers a "
            "different question about your note."
        )
        tab_labels = [
            "1 · Your answer",
            "2 · How the note works",
            "3 · How sure are we?",
            "4 · Try what-if games",
            "5 · Show the recipe",
        ]
    else:
        st.markdown(
            f"### {config.symbol} · {config.trade_stage} · "
            f"{result.get('contract_version')}"
        )
        st.caption(
            f"Market as of {market.get('market_data_time')} · "
            f"source {market.get('source')} · price per unit notional"
        )
        tab_labels = [
            "Overview",
            "Contract & market",
            "Diagnostics",
            "Risk lab",
            "Audit",
        ]
    tabs = st.tabs(tab_labels)
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
        _render_diagnostics(diagnostics, guided=guided)
    with tabs[3]:
        _render_risk_workspace(client, config=config, market=market)
    with tabs[4]:
        _render_audit(
            result=result,
            diagnostics=diagnostics,
            config=config,
            market=market,
        )
