import json
import math
from html import escape
from typing import Any, Mapping

import streamlit as st

from app.frontend_support import compact_nonzero_shock
from app.ui.api_client import FrontendApiError, MlPricerApi
from app.ui.charts import (
    PLOTLY_CONFIG,
    audit_error_heatmap_figure,
    audit_slice_figure,
    autocall_schedule_figure,
    barrier_ladder_figure,
    calibration_option_figure,
    cashflow_figure,
    contract_timeline_figure,
    convergence_figure,
    distribution_figure,
    latency_comparison_figure,
    price_uncertainty_figure,
    risk_figure,
    scenario_figure,
    shadow_error_history_figure,
    surface_figure,
    term_structure_figure,
)
from app.ui.payloads import (
    PricingConfiguration,
    barrier_levels,
    even_observation_schedule,
)


ML_LESSON_STAGES: dict[str, tuple[str, str, str]] = {
    "1 · Teacher makes answers": (
        "Monte Carlo is the careful teacher",
        "For each practice note, it invents many possible price journeys, "
        "follows the payment rules, and averages the results into one price.",
        "Slow and a little noisy, but transparent enough to be our reference.",
    ),
    "2 · Student practises": (
        "ML studies questions paired with teacher answers",
        "Each question contains market numbers and note rules. The student "
        "changes its internal weights whenever its guessed price misses the "
        "teacher's price.",
        "It learns a pricing pattern. It does not memorise a list of stock tips.",
    ),
    "3 · Student sits an exam": (
        "We test examples the student never saw while practising",
        "A sealed test set tells us how far the student's prices miss the "
        "teacher on fresh contracts. Live shadow mode then repeats the check "
        "on real requests.",
        "The student can be fast and still fail if its answers are not accurate.",
    ),
}


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


def ml_lesson_stage(stage: str) -> tuple[str, str, str]:
    return ML_LESSON_STAGES[stage]


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
                str(shadow.get("reason", "")) if isinstance(shadow, Mapping) else ""
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
    interval_text = (
        "inside"
        if interval_label is True
        else "outside" if interval_label is False else "not compared with"
    )
    if guided:
        gap_text = (
            f"{relative_gap:.2%}" if relative_gap is not None else "not available"
        )
        speed_text = f"{speedup:,.1f}×" if speedup is not None else "still measuring"
        st.markdown(
            f"""
            <div class="mlp-answer-race">
              <span class="mlp-section-label">Same question · two calculators</span>
              <div class="mlp-answer-lanes">
                <div>
                  <small>The careful teacher</small>
                  <strong>{escape(_metric_text(comparison["reference_price"]))}</strong>
                  <p>Monte Carlo builds and checks thousands of pretend futures.</p>
                </div>
                <i>compared with</i>
                <div>
                  <small>The fast student</small>
                  <strong>{escape(_metric_text(comparison["ml_price"]))}</strong>
                  <p>ML uses the shortcut it learned from the teacher's examples.</p>
                </div>
              </div>
              <p class="mlp-answer-verdict">
                The answers are <b>{escape(gap_text)} apart</b>. The student was
                <b>{escape(speed_text)} faster</b> on this request. We still show
                the teacher's answer because the student is practising in
                <b>shadow mode</b>, not making the final decision.
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption(
            "Important: the ML model predicts the note's estimated value. It "
            "does not predict tomorrow's stock direction or promise a profit. "
            f"Its answer is {interval_text} the teacher's simulation-noise range."
        )
    else:
        metrics = st.columns(5)
        metrics[0].metric(
            "Monte Carlo reference",
            _metric_text(comparison["reference_price"]),
            help="A path-simulation benchmark, not an executable traded quote.",
        )
        metrics[1].metric(
            "ML surrogate",
            _metric_text(comparison["ml_price"]),
            delta=f"{comparison['signed_gap']:+.6f} vs MC",
            help="The neural surrogate learned the mapping from inputs to prices.",
        )
        metrics[2].metric(
            "Absolute gap",
            _metric_text(comparison["absolute_gap"]),
            help="Absolute difference between the ML estimate and Monte Carlo reference.",
        )
        metrics[3].metric(
            "Relative gap",
            f"{relative_gap:.3%}" if relative_gap is not None else "—",
        )
        metrics[4].metric(
            "Observed speed-up",
            f"{speedup:,.1f}×" if speedup is not None else "Too fast to time",
            help="Reference latency divided by ML inference latency for this request.",
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
        with st.expander(
            "How did this ML model perform on examples it never trained on?"
        ):
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
    contract = config.contract or {}
    if contract.get("contract_version") == "phoenix-single-v3":
        memory_coupon = bool(contract.get("memory_coupon"))
        unpaid_coupon_count = int(contract.get("unpaid_coupon_count", 0))
        autocall_barriers = list(contract.get("autocall_barrier_fracs") or [])
        first_barrier = float(autocall_barriers[0])
        final_barrier = float(autocall_barriers[-1])
        st.markdown("#### Richer contract mechanics")
        columns = st.columns(3)
        columns[0].metric(
            "Coupon memory",
            "On" if memory_coupon else "Off",
            help=(
                "When on, missed coupons accumulate and can be recovered at a "
                "later successful coupon observation."
            ),
        )
        columns[1].metric(
            "Coupons carried in",
            str(unpaid_coupon_count),
            help="Missed memory coupons already outstanding at valuation.",
        )
        columns[2].metric(
            "Autocall step-down",
            f"{first_barrier:.0%} → {final_barrier:.0%}",
            help=("The early-exit hurdle can become lower at later observations."),
        )
        if memory_coupon:
            st.info(
                "A missed coupon is remembered, not guaranteed. Stored coupons "
                "pay only if a later active observation reaches the coupon line; "
                "the note can still mature without recovering them."
            )
        _chart(
            autocall_schedule_figure(
                observations,
                autocall_barriers,
                reference_level=reference_level,
                coupon_barrier_frac=float(config.terms["coupon_barrier_frac"]),
            ),
            key="autocall_schedule",
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

    calibration = result.get("market_calibration")
    if isinstance(calibration, Mapping) and calibration:
        quality = calibration.get("quality") or {}
        freshness = quality.get("freshness") or {}
        coverage = quality.get("coverage") or {}
        snapshot = result.get("market_snapshot_record") or {}
        st.markdown("#### Research market quality")
        st.caption(
            "These checks describe the inputs that were frozen for this run. "
            "Passing them means usable research data—not an executable dealer quote."
        )
        quality_columns = st.columns(4)
        quality_columns[0].metric(
            "Quality checks",
            f"{int(quality.get('passed_checks', 0))}/{int(quality.get('total_checks', 0))}",
        )
        quality_columns[1].metric(
            "Quote age",
            f"{float(freshness.get('quote_age_seconds', 0.0)) / 60.0:.1f} min",
        )
        quality_columns[2].metric(
            "Widest option spread",
            f"{float(coverage.get('maximum_combined_spread_fraction', 0.0)):.1%}",
        )
        quality_columns[3].metric(
            "Option tenors",
            str(int(coverage.get("option_tenors", 0))),
        )
        _chart(
            calibration_option_figure(calibration),
            key="calibration_options",
        )
        _render_table(
            [
                {
                    "check": str(check.get("name", "")).replace("_", " "),
                    "passed": bool(check.get("passed")),
                    "value": check.get("value"),
                    "limit": (
                        f"≤ {check['maximum']}"
                        if check.get("maximum") is not None
                        else f"≥ {check.get('minimum')}"
                    ),
                    "units": check.get("units"),
                }
                for check in quality.get("checks") or []
            ],
            [
                ("check", "Check"),
                ("passed", "Pass"),
                ("value", "Observed"),
                ("limit", "Research limit"),
                ("units", "Units"),
            ],
        )
        if snapshot:
            st.caption(
                f"Immutable snapshot {snapshot.get('snapshot_id')} · "
                f"stored {snapshot.get('created_at')}"
            )
        with st.expander("Why this is still research data"):
            for warning in calibration.get("warnings") or []:
                st.markdown(f"- {warning}")

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


def _readiness_target(check: Mapping[str, Any]) -> str:
    if check.get("minimum") is not None:
        return f"≥ {_metric_text(check['minimum'], 4)}"
    if check.get("maximum") is not None:
        return f"≤ {_metric_text(check['maximum'], 4)}"
    return "Policy rule"


def _render_guided_ml_school(result: Mapping[str, Any] | None) -> None:
    st.markdown(
        """
        <div class="mlp-model-school mlp-model-school-result">
          <span class="mlp-section-label">First, forget the jargon</span>
          <h3>ML is a fast student copying a careful teacher.</h3>
          <div class="mlp-school-flow">
            <div><b>Note + market</b><small>the question</small></div>
            <i>→</i>
            <div><b>Monte Carlo</b><small>teacher makes an answer</small></div>
            <i>→</i>
            <div><b>ML model</b><small>student learns the shortcut</small></div>
          </div>
          <p>
            In this project, the student's target is one number: <b>the note's
            estimated price today</b>. It is not predicting tomorrow's share
            price and it is not deciding whether anyone should invest.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    chosen_stage = st.radio(
        "Tap through the ML lesson",
        list(ML_LESSON_STAGES),
        key="guided_ml_lesson_stage",
        horizontal=True,
    )
    title, body, takeaway = ml_lesson_stage(str(chosen_stage))
    st.markdown(
        f"""
        <div class="mlp-lesson-stage">
          <span>{escape(str(chosen_stage))}</span>
          <strong>{escape(title)}</strong>
          <p>{escape(body)}</p>
          <aside>{escape(takeaway)}</aside>
        </div>
        """,
        unsafe_allow_html=True,
    )
    answer = st.radio(
        "Quick check: if the underlier rises tomorrow, what has the ML model proved?",
        [
            "Nothing about its pricing accuracy yet",
            "That it predicted the rise",
            "That the note must make money",
        ],
        index=None,
        key="guided_ml_result_check",
        horizontal=True,
    )
    if answer is None:
        st.caption("Choose an answer. This is a learning check, not a test.")
    elif answer == "Nothing about its pricing accuracy yet":
        st.success(
            "Exactly. We judge this ML model by how closely it prices fresh "
            "notes—not by tomorrow's market direction."
        )
    else:
        st.info(
            "Not quite. This model predicts the note's estimated price, not the "
            "next market move or a guaranteed investment result."
        )

    comparison = _ml_comparison_summary(result or {})
    if comparison.get("available"):
        st.markdown(
            f"""
            <div class="mlp-current-lesson">
              <span>On the note you just built</span>
              <p>The teacher answered
              <b>{escape(_metric_text(comparison["reference_price"]))}</b> and
              the student answered
              <b>{escape(_metric_text(comparison["ml_price"]))}</b>. Their gap is
              <b>{escape(_metric_text(comparison["absolute_gap"]))}</b> per unit.
              The teacher's answer is still the one in charge.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _render_guided_ml_evidence_snapshot(evidence: Mapping[str, Any]) -> None:
    audit = evidence.get("audit") or {}
    monitoring = evidence.get("monitoring") or {}
    series = evidence.get("series") or {}
    readiness = evidence.get("readiness") or {}
    if not audit.get("available"):
        st.warning(str(audit.get("reason", "The model report card is unavailable.")))
        return

    artifact = audit.get("artifact") or {}
    sealed = audit.get("sealed_audit") or {}
    price_metrics = sealed.get("price_metrics") or {}
    st.markdown("### The student's report card")
    st.write(
        "We do not trust the student because one answer looked good. We give it "
        "questions it never saw during practice, then watch it quietly on new "
        "requests while the teacher stays in charge."
    )
    st.markdown(
        """
        <div class="mlp-evidence-split">
          <div><span>1 · Fresh exam</span><strong>Unseen questions</strong>
          <p>These notes were hidden while the model practised. That makes the
          score harder to fake by memorising.</p></div>
          <div><span>2 · Quiet practice</span><strong>Shadow mode</strong>
          <p>Both calculators answer live requests. We record the gap, but only
          Monte Carlo supplies the reference price.</p></div>
          <div><span>3 · Safety lock</span><strong>Human review</strong>
          <p>Passing rules can only make the model ready for review. This screen
          cannot put ML in charge.</p></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    exam_columns = st.columns(3)
    exam_columns[0].metric(
        "Fresh exam questions",
        f"{int(price_metrics.get('n_samples') or audit.get('datasets', {}).get('audit_samples') or 0):,}",
        help="Notes hidden from the ML model during training.",
    )
    exam_columns[1].metric(
        "Average miss per 1.00",
        _metric_text(price_metrics.get("mae")),
        help="On average, how far the student's price was from the teacher's price.",
    )
    exam_columns[2].metric(
        "95-in-100 miss limit",
        _metric_text(price_metrics.get("p95_absolute_error")),
        help="About 95% of absolute pricing misses were no larger than this.",
    )
    st.caption(
        "For the two error numbers, smaller is better. They are measured per "
        "1.00 of note principal, so 0.010000 means one cent per dollar."
    )

    live_available = monitoring.get("available") is True
    live_overall = monitoring.get("overall") or {}
    st.markdown("#### How is the student doing on new requests?")
    live_columns = st.columns(4)
    live_columns[0].metric(
        "Requests watched",
        f"{int(live_overall.get('n_observations') or 0):,}",
    )
    live_columns[1].metric(
        "Average live miss",
        _metric_text(live_overall.get("mae")),
    )
    live_columns[2].metric(
        "Large live miss",
        _metric_text(live_overall.get("p95_absolute_error")),
    )
    live_columns[3].metric(
        "Typical speed advantage",
        (
            f"{float(live_overall['median_speedup']):,.1f}×"
            if live_overall.get("median_speedup") is not None
            else "Still collecting"
        ),
    )
    observations = series.get("observations") or []
    if not live_available or not observations:
        st.info(
            "There are not enough fresh shadow-mode requests for a useful trend "
            "yet. The sealed exam still counts; the live report card is simply "
            "waiting for more examples."
        )

    decision = str(readiness.get("decision", "insufficient_evidence"))
    st.markdown("#### Is the student ready to be considered for more responsibility?")
    if decision == "ready_for_review":
        st.success(
            "All frozen safety checks passed. This means ready for an independent "
            "human review—not automatically ready to replace Monte Carlo."
        )
    elif decision == "not_ready":
        st.error(
            "No. There is enough evidence to judge it, and at least one safety "
            "check failed. Monte Carlo remains in charge."
        )
    else:
        st.warning(
            "Not yet. We need more varied, independent live examples. Waiting is "
            "the safe answer; Monte Carlo remains in charge."
        )
    st.caption(
        str(
            readiness.get(
                "next_action",
                "Collect broader independent evidence before changing runtime policy.",
            )
        )
    )

    with st.expander("Show me the report-card pictures"):
        st.write(
            "Each dot or square is a group of unseen questions. Look for small, "
            "even errors rather than one impressive average hiding weak corners."
        )
        left, right = st.columns(2, gap="large")
        with left:
            _chart(audit_slice_figure(audit), key="guided_audit_slice_evidence")
        with right:
            _chart(
                audit_error_heatmap_figure(audit),
                key="guided_audit_joint_evidence",
            )
        if live_available and observations:
            left, right = st.columns(2, gap="large")
            with left:
                _chart(
                    shadow_error_history_figure(series),
                    key="guided_shadow_error_history",
                )
            with right:
                _chart(
                    latency_comparison_figure(series),
                    key="guided_shadow_latency_evidence",
                )

    with st.expander("Open the technical model-audit details"):
        st.caption(
            f"{artifact.get('model_version')} · {artifact.get('contract_version')} · "
            f"{artifact.get('runtime_policy')} · sealed audit "
            f"{'passed' if sealed.get('passed') else 'did not pass'} · "
            f"R² {_metric_text(price_metrics.get('r2'), 4)}"
        )
        evidence_counts = readiness.get("evidence") or {}
        count_columns = st.columns(4)
        count_columns[0].metric(
            "Distinct cases",
            f"{int(evidence_counts.get('n_distinct_cases') or 0):,}",
        )
        count_columns[1].metric(
            "Symbols",
            f"{int(evidence_counts.get('n_unique_symbols') or 0):,}",
        )
        count_columns[2].metric(
            "Market dates",
            f"{int(evidence_counts.get('n_distinct_market_dates') or 0):,}",
        )
        count_columns[3].metric(
            "Evidence span",
            (
                f"{float(evidence_counts['observation_span_days']):.1f} days"
                if evidence_counts.get("observation_span_days") is not None
                else "Collecting"
            ),
        )
        checks = readiness.get("checks") or {}
        check_rows = [
            {
                "gate": name.replace("_", " ").title(),
                "kind": str(check.get("kind", "")).title(),
                "observed": _metric_text(check.get("value"), 4),
                "target": _readiness_target(check),
                "passed": bool(check.get("passed")),
            }
            for name, check in checks.items()
            if isinstance(check, Mapping)
        ]
        if check_rows:
            _render_table(
                check_rows,
                [
                    ("gate", "Gate"),
                    ("kind", "Type"),
                    ("observed", "Observed"),
                    ("target", "Required"),
                    ("passed", "Pass"),
                ],
            )

        expansion = evidence.get("expansion_experiments") or {}
        if expansion.get("available"):
            st.markdown("##### Research models for additional products")
            experiment_rows = []
            for candidate in expansion.get("products") or []:
                candidate_audit = candidate.get("sealed_audit") or {}
                metrics = candidate_audit.get("metrics") or {}
                experiment_rows.append(
                    {
                        "product": str(candidate.get("product_key", "")).replace(
                            "_", " "
                        ),
                        "contract": candidate.get("contract_version"),
                        "audit": (
                            "Passed" if candidate_audit.get("passed") else "Rejected"
                        ),
                        "mae": metrics.get("mae"),
                        "p95": metrics.get("p95_absolute_error"),
                        "runtime": "Shadow candidate",
                    }
                )
            _render_table(
                experiment_rows,
                [
                    ("product", "Product"),
                    ("contract", "Contract"),
                    ("audit", "Sealed audit"),
                    ("mae", "MAE"),
                    ("p95", "P95 error"),
                    ("runtime", "Runtime"),
                ],
                number_formats={"mae": ".6f", "p95": ".6f"},
            )

    _render_expanded_shadow_evidence(evidence, guided=True)
    st.download_button(
        "Download the full evidence snapshot",
        data=json.dumps(evidence, indent=2, sort_keys=True).encode("utf-8"),
        file_name="ml-pricer-evidence.json",
        mime="application/json",
    )


def _render_expanded_shadow_evidence(
    evidence: Mapping[str, Any], *, guided: bool
) -> None:
    expanded = evidence.get("expanded_shadow") or {}
    runtime_products = (expanded.get("runtime") or {}).get("products") or {}
    monitoring_products = (expanded.get("monitoring") or {}).get("products") or {}
    readiness_products = (expanded.get("readiness") or {}).get("products") or {}
    if not runtime_products:
        return
    st.markdown(
        "#### Two new ML students are waiting in the practice room"
        if guided
        else "#### Expanded-product shadow rollout"
    )
    st.write(
        (
            "Phoenix v3 and the reverse convertible now have learned shortcuts. "
            "They can quietly answer the same question as Monte Carlo so we can "
            "compare them, but their answer never becomes the price on this screen."
        )
        if guided
        else (
            "Pinned Phoenix v3 and barrier reverse-convertible artifacts are wired "
            "for fail-closed shadow inference. Both remain independently disabled "
            "until sampling and telemetry are explicitly enabled."
        )
    )
    labels = {
        "phoenix_v3": "Phoenix v3",
        "barrier_reverse_convertible": "Barrier reverse convertible",
    }
    rows = []
    for key, label in labels.items():
        runtime = runtime_products.get(key) or {}
        observed = monitoring_products.get(key) or {}
        readiness = readiness_products.get(key) or {}
        if not runtime.get("artifact_available"):
            state = "Artifact unavailable"
        elif runtime.get("enabled"):
            state = f"Collecting ({float(runtime.get('sample_rate') or 0):.0%})"
        else:
            state = "Packaged · switched off"
        rows.append(
            {
                "product": label,
                "state": state,
                "observations": int(observed.get("n_observations") or 0),
                "mae": observed.get("mae"),
                "p95": observed.get("p95_absolute_error"),
                "review": (
                    "Ready for human review"
                    if readiness.get("ready_for_human_review")
                    else "Needs evidence"
                ),
            }
        )
    _render_table(
        rows,
        [
            ("product", "Product"),
            ("state", "Shadow state"),
            ("observations", "Observed"),
            ("mae", "Live MAE"),
            ("p95", "Live P95 error"),
            ("review", "Decision"),
        ],
        number_formats={"mae": ".6f", "p95": ".6f"},
    )
    st.caption(
        "No automatic promotion exists. Meeting every evidence gate only opens a "
        "separate human review; Monte Carlo remains authoritative."
    )


def _render_ml_evidence(
    client: MlPricerApi,
    *,
    guided: bool,
    result: Mapping[str, Any] | None = None,
) -> None:
    heading = (
        "## Is the learned shortcut earning our trust?"
        if guided
        else "## ML Evidence Lab"
    )
    st.markdown(heading)
    st.write(
        (
            "A good-looking answer once is not enough. We separate the model’s "
            "closed-book exam from what it does on new requests over time."
        )
        if guided
        else (
            "Frozen sealed-audit evidence, bounded live shadow telemetry, drift "
            "diagnostics, and non-promoting readiness gates."
        )
    )
    if guided:
        _render_guided_ml_school(result)
        st.markdown("### Now check the evidence")
        st.write(
            "The lesson above explains the idea. The report below contains the "
            "actual stored scores for this project's ML model."
        )
    refresh = st.button(
        "Refresh evidence",
        key="refresh_ml_evidence",
        help="Reload the sealed artifact report and the latest shadow observations.",
    )
    if refresh or (
        st.session_state.get("ml_evidence_snapshot") is None
        and st.session_state.get("ml_evidence_error") is None
    ):
        try:
            with st.spinner("Loading model evidence…"):
                st.session_state["ml_evidence_snapshot"] = client.ml_evidence()
                st.session_state.pop("ml_evidence_error", None)
        except FrontendApiError as exc:
            st.session_state["ml_evidence_error"] = str(exc)
            st.session_state["ml_evidence_snapshot"] = None

    evidence = st.session_state.get("ml_evidence_snapshot")
    if not isinstance(evidence, Mapping):
        st.error(
            st.session_state.get(
                "ml_evidence_error",
                "The ML evidence snapshot is not available.",
            )
        )
        return
    if guided:
        _render_guided_ml_evidence_snapshot(evidence)
        return

    audit = evidence.get("audit") or {}
    monitoring = evidence.get("monitoring") or {}
    series = evidence.get("series") or {}
    readiness = evidence.get("readiness") or {}
    if not audit.get("available"):
        st.warning(str(audit.get("reason", "The sealed audit is unavailable.")))
        return

    artifact = audit.get("artifact") or {}
    sealed = audit.get("sealed_audit") or {}
    price_metrics = sealed.get("price_metrics") or {}
    st.markdown(
        """
        <div class="mlp-evidence-split">
          <div><span>Closed-book exam</span><strong>Sealed audit</strong>
          <p>A fixed dataset the selected model was not allowed to train on.
          These numbers do not change when you price another note.</p></div>
          <div><span>Work after graduation</span><strong>Live shadow evidence</strong>
          <p>New requests observed while Monte Carlo remains in charge.
          This is where drift, outages and real runtime behaviour appear.</p></div>
          <div><span>Safety decision</span><strong>Promotion gates</strong>
          <p>Frozen rules can permit human review, but this screen can never
          promote the model or replace the reference pricer.</p></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### Sealed audit report card")
    audit_columns = st.columns(4)
    audit_columns[0].metric(
        "Held-out examples",
        f"{int(price_metrics.get('n_samples') or audit.get('datasets', {}).get('audit_samples') or 0):,}",
    )
    audit_columns[1].metric(
        "Mean absolute error", _metric_text(price_metrics.get("mae"))
    )
    audit_columns[2].metric(
        "95th-percentile error",
        _metric_text(price_metrics.get("p95_absolute_error")),
    )
    audit_columns[3].metric("R²", _metric_text(price_metrics.get("r2"), 4))
    st.caption(
        f"{artifact.get('model_version')} · {artifact.get('contract_version')} · "
        f"{artifact.get('runtime_policy')} · audit "
        f"{'passed' if sealed.get('passed') else 'did not pass'}"
    )
    left, right = st.columns(2, gap="large")
    with left:
        _chart(audit_slice_figure(audit), key="audit_slice_evidence")
    with right:
        _chart(audit_error_heatmap_figure(audit), key="audit_joint_evidence")

    expansion = evidence.get("expansion_experiments") or {}
    if expansion.get("available"):
        st.markdown("#### Product-expansion experiments")
        st.caption(
            "These candidates cover newer payoff contracts. Passing packages "
            "a research artifact only; runtime approval remains a separate decision."
        )
        experiment_rows = []
        for candidate in expansion.get("products") or []:
            candidate_audit = candidate.get("sealed_audit") or {}
            metrics = candidate_audit.get("metrics") or {}
            experiment_rows.append(
                {
                    "product": str(candidate.get("product_key", "")).replace("_", " "),
                    "contract": candidate.get("contract_version"),
                    "audit": (
                        "Passed" if candidate_audit.get("passed") else "Rejected"
                    ),
                    "mae": metrics.get("mae"),
                    "p95": metrics.get("p95_absolute_error"),
                    "r2": metrics.get("r2"),
                    "runtime": "Shadow candidate",
                }
            )
        _render_table(
            experiment_rows,
            [
                ("product", "Product"),
                ("contract", "Contract"),
                ("audit", "Sealed audit"),
                ("mae", "MAE"),
                ("p95", "P95 error"),
                ("r2", "R²"),
                ("runtime", "Runtime"),
            ],
            number_formats={"mae": ".6f", "p95": ".6f", "r2": ".4f"},
        )
        st.caption(
            f"{expansion.get('experiment_version')} · generated "
            f"{expansion.get('generated_at')} · runtime policy unchanged"
        )

    _render_expanded_shadow_evidence(evidence, guided=False)

    st.markdown("#### Live shadow report card")
    live_available = monitoring.get("available") is True
    live_overall = monitoring.get("overall") or {}
    live_columns = st.columns(5)
    live_columns[0].metric(
        "Observed requests",
        f"{int(live_overall.get('n_observations') or 0):,}",
    )
    live_columns[1].metric("Live MAE", _metric_text(live_overall.get("mae")))
    live_columns[2].metric(
        "Live P95 error",
        _metric_text(live_overall.get("p95_absolute_error")),
    )
    live_columns[3].metric(
        "Median speed-up",
        (
            f"{float(live_overall['median_speedup']):,.1f}×"
            if live_overall.get("median_speedup") is not None
            else "Collecting"
        ),
    )
    drift = monitoring.get("feature_drift") or {}
    live_columns[4].metric(
        "Inputs beyond 4σ",
        (
            f"{float(drift['above_four_sigma_fraction']):.1%}"
            if drift.get("above_four_sigma_fraction") is not None
            else "Collecting"
        ),
    )
    observations = series.get("observations") or []
    if live_available and observations:
        left, right = st.columns(2, gap="large")
        with left:
            _chart(shadow_error_history_figure(series), key="shadow_error_history")
        with right:
            _chart(latency_comparison_figure(series), key="shadow_latency_evidence")
    else:
        st.info(
            "The sealed audit is valid, but there are not yet enough new-schema "
            "live observations for history and speed charts. Price new-issue "
            "Phoenix notes to collect shadow evidence."
        )

    st.markdown("#### Promotion-readiness gates")
    decision = str(readiness.get("decision", "insufficient_evidence"))
    if decision == "ready_for_review":
        st.success(
            "Every frozen gate passed. The artifact is ready for independent "
            "human review—not automatic production promotion."
        )
    elif decision == "not_ready":
        st.error(
            "There is enough evidence to judge the model, and at least one "
            "quality, drift, integrity or operations gate failed."
        )
    else:
        st.warning(
            "There is not enough independent live evidence yet. The correct "
            "decision is to keep Monte Carlo in charge and collect broader cases."
        )
    evidence_counts = readiness.get("evidence") or {}
    count_columns = st.columns(4)
    count_columns[0].metric(
        "Cases",
        f"{int(evidence_counts.get('n_distinct_cases') or 0):,}",
    )
    count_columns[1].metric(
        "Symbols",
        f"{int(evidence_counts.get('n_unique_symbols') or 0):,}",
    )
    count_columns[2].metric(
        "Market dates",
        f"{int(evidence_counts.get('n_distinct_market_dates') or 0):,}",
    )
    count_columns[3].metric(
        "Evidence span",
        (
            f"{float(evidence_counts['observation_span_days']):.1f} days"
            if evidence_counts.get("observation_span_days") is not None
            else "Collecting"
        ),
    )
    checks = readiness.get("checks") or {}
    check_rows = [
        {
            "gate": name.replace("_", " ").title(),
            "kind": str(check.get("kind", "")).title(),
            "observed": _metric_text(check.get("value"), 4),
            "target": _readiness_target(check),
            "passed": bool(check.get("passed")),
        }
        for name, check in checks.items()
        if isinstance(check, Mapping)
    ]
    if check_rows:
        _render_table(
            check_rows,
            [
                ("gate", "Gate"),
                ("kind", "Type"),
                ("observed", "Observed"),
                ("target", "Required"),
                ("passed", "Pass"),
            ],
        )
    st.caption(
        str(
            readiness.get(
                "next_action",
                "Collect broader independent evidence before changing runtime policy.",
            )
        )
    )
    st.download_button(
        "Download ML evidence snapshot",
        data=json.dumps(evidence, indent=2, sort_keys=True).encode("utf-8"),
        file_name="ml-pricer-evidence.json",
        mime="application/json",
    )


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


def _render_barrier_reverse_convertible_results(
    *,
    result: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
    config: PricingConfiguration,
    market: Mapping[str, Any],
) -> None:
    contract = result.get("contract") or config.contract or {}
    reference = float(contract["reference_level"])
    strike_level = reference * float(contract["strike_frac"])
    knock_in_level = reference * float(contract["knock_in_frac"])
    cashflows = diagnostics.get("cashflows") or {}
    coupon_times = list(contract.get("coupon_times_years") or [])
    st.markdown(
        f"### {config.symbol} · Barrier reverse convertible · " f"{config.trade_stage}"
    )
    st.caption(
        f"Market as of {market.get('market_data_time')} · "
        "Monte Carlo reference price per unit notional"
    )
    overview, mechanics, diagnostics_tab, market_tab, audit_tab = st.tabs(
        ["Overview", "How it works", "Diagnostics", "Market evidence", "Audit"]
    )
    with overview:
        price = float(result["price"])
        interval = result.get("confidence_interval") or [price, price]
        metrics = st.columns(4)
        metrics[0].metric("Reference price", f"{price:.6f}")
        metrics[1].metric(
            "95% simulation interval",
            f"{float(interval[0]):.4f} – {float(interval[1]):.4f}",
        )
        metrics[2].metric(
            "Downside redemption paths",
            f"{float(cashflows.get('downside_probability', 0.0)):.1%}",
        )
        metrics[3].metric(
            "Contractual coupons",
            str(int(cashflows.get("contractual_coupon_count", 0))),
        )
        left, right = st.columns(2, gap="large")
        with left:
            _chart(price_uncertainty_figure(result), key="brc_price")
        with right:
            _chart(
                barrier_ladder_figure(
                    [
                        {
                            "name": "Knock-in barrier",
                            "level": knock_in_level,
                            "kind": "risk",
                        },
                        {
                            "name": "Live spot",
                            "level": float(market["spot"]),
                            "kind": "market",
                        },
                        {
                            "name": "Reference level",
                            "level": reference,
                            "kind": "reference",
                        },
                        {
                            "name": "Conversion strike",
                            "level": strike_level,
                            "kind": "autocall",
                        },
                    ]
                ),
                key="brc_barriers",
            )
        st.info(
            "The displayed price is still Monte Carlo. No ML shortcut is shown "
            "for this product because no reverse-convertible model has passed "
            "the sealed acceptance gates yet."
        )

    with mechanics:
        st.markdown("#### Three pieces, read in order")
        total_coupon = len(coupon_times) * float(contract["coupon_rate_per_period"])
        st.markdown(
            f"""
            <div class="mlp-rulebook">
              <div>
                <span>01 · Coupons</span>
                <strong>{len(coupon_times)} × {float(contract["coupon_rate_per_period"]):.2%}</strong>
                <p>This research contract pays each coupon regardless of the
                underlier level. The undiscounted total is {total_coupon:.2%}
                per unit notional. Issuer default risk is not modelled.</p>
              </div>
              <div>
                <span>02 · Knock-in memory</span>
                <strong>{knock_in_level:,.2f}</strong>
                <p>Touching this line remembers that a serious fall happened.
                It does not create an immediate cash loss; it changes the rule
                checked at maturity.</p>
              </div>
              <div>
                <span>03 · Final redemption</span>
                <strong>Strike · {strike_level:,.2f}</strong>
                <p>If knock-in occurred and the final level is below the strike,
                redemption becomes final level ÷ strike. Otherwise principal
                returns as 1.00.</p>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        _chart(
            contract_timeline_figure(coupon_times, config.maturity_years),
            key="brc_timeline",
        )
        st.warning(
            "A high coupon is compensation for conditional downside exposure, "
            "not free yield. This model also omits issuer credit, funding, fees, "
            "tax, liquidity, and hedging costs."
        )

    with diagnostics_tab:
        diagnostic_metrics = st.columns(3)
        diagnostic_metrics[0].metric(
            "Knock-in frequency",
            f"{float(cashflows.get('knock_in_probability', 0.0)):.1%}",
        )
        diagnostic_metrics[1].metric(
            "Loss-linked ending",
            f"{float(cashflows.get('downside_probability', 0.0)):.1%}",
        )
        diagnostic_metrics[2].metric(
            "Undiscounted coupons",
            f"{float(cashflows.get('total_coupon_per_unit_undiscounted', 0.0)):.2%}",
        )
        first, second = st.columns(2, gap="large")
        with first:
            _chart(convergence_figure(diagnostics), key="brc_convergence")
        with second:
            _chart(cashflow_figure(diagnostics), key="brc_cashflows")
        first, second = st.columns(2, gap="large")
        with first:
            _chart(distribution_figure(diagnostics), key="brc_distribution")
        with second:
            _chart(surface_figure(diagnostics), key="brc_surface")

    with market_tab:
        _chart(term_structure_figure(market), key="brc_term_structure")
        calibration = result.get("market_calibration")
        if isinstance(calibration, Mapping) and calibration:
            quality = calibration.get("quality") or {}
            st.markdown("#### Frozen research calibration")
            quality_columns = st.columns(3)
            quality_columns[0].metric(
                "Checks passed",
                f"{quality.get('passed_checks', 0)}/{quality.get('total_checks', 0)}",
            )
            quality_columns[1].metric(
                "Quote age",
                f"{float((quality.get('freshness') or {}).get('quote_age_seconds', 0)) / 60:.1f} min",
            )
            quality_columns[2].metric(
                "Snapshot",
                (
                    "Immutable"
                    if (result.get("market_snapshot_record") or {}).get("immutable")
                    else "Request-derived"
                ),
            )
            _chart(
                calibration_option_figure(calibration),
                key="brc_calibration_options",
            )
        else:
            st.caption("Manual market inputs were frozen directly from the form.")

    with audit_tab:
        _render_audit(
            result=result,
            diagnostics=diagnostics,
            config=config,
            market=market,
        )


def render_pricing_results(
    client: MlPricerApi,
    *,
    result: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
    config: PricingConfiguration,
    market: Mapping[str, Any],
) -> None:
    if config.product_key == "barrier_reverse_convertible":
        _render_barrier_reverse_convertible_results(
            result=result,
            diagnostics=diagnostics,
            config=config,
            market=market,
        )
        return
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
            "6 · Is ML learning?",
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
            "ML evidence",
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
    with tabs[5]:
        _render_ml_evidence(client, guided=guided, result=result)
