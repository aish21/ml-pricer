from html import escape
from typing import Any

import streamlit as st

from app.ui.charts import barrier_ladder_figure, learning_paths_figure
from app.ui.payloads import (
    FrontendInputError,
    PricingConfiguration,
    barrier_levels,
    build_flat_term_structure,
    build_phoenix_terms,
    build_v2_contract,
    even_observation_schedule,
)
from app.ui.underliers import render_underlier_picker


GUIDED_STEPS = (
    ("1", "Pick a price to watch"),
    ("2", "Choose the clock"),
    ("3", "Build the rules"),
    ("4", "Meet the simulation"),
    ("5", "Review and price"),
)

MATURITIES = {
    "6 months": 0.5,
    "1 year": 1.0,
    "2 years": 2.0,
    "3 years": 3.0,
}

CHECK_FREQUENCIES = {
    "Every month": 12,
    "Every 2 months": 6,
    "Every 3 months": 4,
    "Only at the end": 1,
}

GUIDED_DEFAULTS: dict[str, Any] = {
    "guided_step": 0,
    "guided_market_choice": "Use today's research data",
    "guided_symbol": "SPY",
    "guided_underlier_type": "etf",
    "guided_time_choice": "1 year",
    "guided_check_frequency": "Every 2 months",
    "guided_trade_choice": "We are making it today",
    "guided_reference_level": 100.0,
    "guided_prior_knock_in": False,
    "guided_autocall_pct": 105,
    "guided_coupon_barrier_pct": 100,
    "guided_coupon_pct": 2.0,
    "guided_knock_in_pct": 70,
    "guided_learning_vol_pct": 20,
    "guided_spot": 100.0,
    "guided_rate_pct": 3.0,
    "guided_dividend_pct": 1.0,
    "guided_market_vol_pct": 20.0,
    "guided_display_notional": 100_000.0,
}


FOUNDATION_GLOSSARY: dict[str, tuple[str, str, str]] = {
    "Note": (
        "A note is an IOU with a rulebook attached.",
        "You give money to an issuer. The issuer promises to pay you later, "
        "but the amount and timing follow written rules.",
        "You do not automatically own the watched share or basket. You own "
        "the issuer's promise, so both the rules and the issuer matter.",
    ),
    "Underlier": (
        "The underlier is the thing the rulebook watches.",
        "It can be one company's share, a fund holding many shares, or an "
        "index that measures a market. Its price is an input to the note.",
        "The underlier can change what the note pays even though the note "
        "holder does not necessarily own it.",
    ),
    "Issuer": (
        "The issuer is the bank or company making the promise.",
        "It receives the investor's money and is responsible for payments "
        "described by the note.",
        "A model can estimate the rules perfectly and the promise can still "
        "depend on the issuer being able to pay.",
    ),
    "Notional": (
        "Notional is the amount the note's payments are measured against.",
        "If a price is 0.98 per unit and the notional is 100,000, the model "
        "value is about 98,000.",
        "It lets us describe one reusable contract and then scale the money.",
    ),
    "Price": (
        "Price is today's estimate of all the rulebook's future payments.",
        "The pricer imagines possible futures, applies the rules, and brings "
        "future money back into today's units.",
        "It is a model estimate, not a prediction of the underlier and not a "
        "guaranteed amount you could trade at.",
    ),
    "Coupon": (
        "A coupon is a possible reward payment.",
        "This Phoenix note checks whether the underlier is high enough on "
        "certain dates. A successful check can earn a coupon.",
        "The attractive headline reward is conditional: some paths may pay "
        "fewer coupons or none.",
    ),
    "Autocall": (
        "An autocall is an automatic early ending.",
        "On a scheduled observation date, the note compares the underlier with "
        "its autocall line. At or above that line, it returns principal and "
        "stops; the investor does not choose to trigger it.",
        "An early finish limits how many future coupons can be earned and "
        "changes how long the investor’s money remains in the note.",
    ),
    "Barrier": (
        "A barrier is a line used by a rule.",
        "The note compares the underlier with reward, early-finish, and safety "
        "lines on the dates described by the contract.",
        "A tiny move across a barrier can change a payment, which makes these "
        "products more nonlinear than simply owning a share.",
    ),
    "Autocall barrier": (
        "The autocall barrier is the note’s early-finish line.",
        "If the underlier is at or above it on an observation date, the note "
        "returns principal and ends automatically.",
        "Being above it between observation dates does not by itself end the "
        "note; timing is part of the contract.",
    ),
    "Coupon barrier": (
        "The coupon barrier is the reward test line.",
        "On an observation date, a level at or above this barrier earns that "
        "date’s coupon. In this non-memory note, a missed coupon is not stored.",
        "The coupon percentage alone does not describe value: the chance of "
        "passing this test matters too.",
    ),
    "Knock-in barrier": (
        "The knock-in barrier switches on a loss-linked ending rule.",
        "If the underlier crosses it during monitoring, the note remembers the "
        "event. That crossing is not an immediate payment or loss.",
        "If the note reaches maturity, the remembered knock-in can make "
        "principal fall with a weak final underlier level.",
    ),
    "Observation date": (
        "An observation date is a scheduled rule-check day.",
        "Coupon and autocall conditions are tested on these dates. The knock-in "
        "rule may be monitored more frequently by the pricing model.",
        "The same price can produce different contract outcomes depending on "
        "whether it occurs on a rule-check day.",
    ),
    "Principal": (
        "Principal is the original unit amount the note may return.",
        "The pricer quotes value per 1.00 of principal, then multiplies it by "
        "the displayed notional to show a money amount.",
        "Principal can return early after an autocall or at maturity, and a "
        "knock-in can make the maturity amount loss-linked.",
    ),
    "Maturity": (
        "Maturity is the note's scheduled ending date.",
        "If the note has not already finished early, its final rules are "
        "applied at maturity.",
        "More time creates more possible price stories and more chances for "
        "barriers to matter.",
    ),
    "Volatility": (
        "Volatility is a model's measure of price wiggliness.",
        "Higher volatility makes a wider fan of calm and extreme pretend "
        "futures. It says nothing about which direction wins.",
        "Barrier products care about the path, so changing wiggliness can "
        "change the note's value even when today's price stays fixed.",
    ),
    "Monte Carlo": (
        "Monte Carlo means learning from many pretend futures.",
        "The computer draws random price paths, applies the same note rules to "
        "each path, and averages their discounted payments.",
        "More paths reduce computer wobble, but they do not remove mistakes in "
        "the model, contract, or market data.",
    ),
}


def glossary_entry(term: str) -> tuple[str, str, str]:
    return FOUNDATION_GLOSSARY[term]


def _initialize_guided_state() -> None:
    for name, value in GUIDED_DEFAULTS.items():
        if name not in st.session_state:
            st.session_state[name] = value


def _set_step(step: int) -> None:
    st.session_state["guided_step"] = max(0, min(step, len(GUIDED_STEPS) - 1))


def _step_header(step: int) -> None:
    labels = "".join(
        (
            '<div class="mlp-step mlp-step-active">'
            if index == step
            else '<div class="mlp-step">'
        )
        + f"<span>{number}</span>{title}</div>"
        for index, (number, title) in enumerate(GUIDED_STEPS)
    )
    st.markdown(f'<div class="mlp-stepper">{labels}</div>', unsafe_allow_html=True)
    st.progress((step + 1) / len(GUIDED_STEPS))


def _navigation(step: int, *, can_continue: bool = True) -> None:
    left, middle, right = st.columns([1, 2.4, 1])
    with left:
        if step > 0:
            st.button(
                "← Back",
                key=f"guided_back_{step}",
                on_click=_set_step,
                args=(step - 1,),
                width="stretch",
            )
    with middle:
        st.caption(
            "Nothing here is a test. Change a choice, watch what moves, "
            "and use Back whenever you want."
        )
    with right:
        if step < len(GUIDED_STEPS) - 1:
            st.button(
                "Next →",
                key=f"guided_next_{step}",
                on_click=_set_step,
                args=(step + 1,),
                disabled=not can_continue,
                width="stretch",
            )


def _market_source() -> str:
    return (
        "Research market"
        if st.session_state["guided_market_choice"] == "Use today's research data"
        else "Manual flat market"
    )


def _maturity() -> float:
    return MATURITIES[str(st.session_state["guided_time_choice"])]


def _observation_count() -> int:
    maturity = _maturity()
    per_year = CHECK_FREQUENCIES[str(st.session_state["guided_check_frequency"])]
    return max(1, int(round(maturity * per_year)))


def _terms() -> dict[str, Any]:
    return build_phoenix_terms(
        maturity_years=_maturity(),
        autocall_barrier_frac=float(st.session_state["guided_autocall_pct"]) / 100.0,
        coupon_barrier_frac=float(st.session_state["guided_coupon_barrier_pct"])
        / 100.0,
        coupon_rate=float(st.session_state["guided_coupon_pct"]) / 100.0,
        knock_in_frac=float(st.session_state["guided_knock_in_pct"]) / 100.0,
        observation_count=_observation_count(),
    )


def _render_foundation_lesson() -> None:
    st.markdown(
        """
        <div class="mlp-foundation">
          <span class="mlp-section-label">Before step 1 · the whole idea</span>
          <h3>A note is a promise that watches something else.</h3>
          <div class="mlp-foundation-flow">
            <div><b>1 · Investor</b><span>puts an amount into the note</span></div>
            <i>→</i>
            <div><b>2 · Note</b><span>stores the issuer's payment rules</span></div>
            <i>→</i>
            <div><b>3 · Underlier</b><span>moves and changes the result</span></div>
          </div>
          <p>
            Owning the note is not the same as owning the underlier. Think of
            the underlier as the scoreboard and the note as the prize rules.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    term = st.selectbox(
        "Word shelf — choose any word to unpack",
        list(FOUNDATION_GLOSSARY),
        key="guided_glossary_term",
        help="Come back to this shelf whenever a finance word feels unfamiliar.",
    )
    headline, explanation, why_it_matters = glossary_entry(str(term))
    st.markdown(
        f"""
        <div class="mlp-word-card">
          <span>In one sentence</span>
          <h4>{escape(headline)}</h4>
          <p>{escape(explanation)}</p>
          <aside><b>Why it matters:</b> {escape(why_it_matters)}</aside>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_pick_underlier() -> None:
    _render_foundation_lesson()
    st.markdown("## Step 1: Pick something whose price we can watch")
    st.write(
        "An **underlier** is just the thing our note watches. It might be one "
        "company, or an ETF—a basket containing lots of things."
    )
    market_choice = st.radio(
        "Where should today's numbers come from?",
        ["Use today's research data", "Let me make up simple numbers"],
        index=[
            "Use today's research data",
            "Let me make up simple numbers",
        ].index(str(st.session_state["guided_market_choice"])),
        key="guided_market_choice_widget",
        horizontal=True,
        help=(
            "Research data comes from the connected market-data service. "
            "Simple numbers let you safely experiment."
        ),
    )
    st.session_state["guided_market_choice"] = market_choice
    symbol, underlier_type = render_underlier_picker(
        market_source=_market_source(),
        key_prefix="guided_underlier",
        beginner_language=True,
    )
    st.session_state["guided_symbol"] = symbol
    st.session_state["guided_underlier_type"] = underlier_type
    st.info(
        f"We will watch **{symbol or 'your symbol'}**. The note itself is a "
        "set of promises whose result depends on where that price travels."
    )
    _navigation(0, can_continue=bool(symbol))


def _render_clock() -> None:
    st.markdown("## Step 2: Choose how long the story lasts")
    st.write(
        "A note does not watch forever. We choose an ending day and some "
        "**check-in days**. On each check-in day, the note asks: "
        "“Is the price high enough for a reward or an early finish?”"
    )
    left, right = st.columns(2, gap="large")
    with left:
        time_choice = st.radio(
            "When should the story end?",
            list(MATURITIES),
            index=list(MATURITIES).index(str(st.session_state["guided_time_choice"])),
            key="guided_time_choice_widget",
        )
        st.session_state["guided_time_choice"] = time_choice
    with right:
        check_frequency = st.radio(
            "How often should the note check the price?",
            list(CHECK_FREQUENCIES),
            index=list(CHECK_FREQUENCIES).index(
                str(st.session_state["guided_check_frequency"])
            ),
            key="guided_check_frequency_widget",
        )
        st.session_state["guided_check_frequency"] = check_frequency
    count = _observation_count()
    st.success(
        f"Your note lasts **{st.session_state['guided_time_choice']}** and "
        f"checks the price **{count} {'time' if count == 1 else 'times'}**."
    )
    trade_choice = st.radio(
        "Are we creating this note now, or did it start earlier?",
        ["We are making it today", "It started earlier"],
        index=["We are making it today", "It started earlier"].index(
            str(st.session_state["guided_trade_choice"])
        ),
        key="guided_trade_choice_widget",
        horizontal=True,
    )
    st.session_state["guided_trade_choice"] = trade_choice
    if st.session_state["guided_trade_choice"] == "It started earlier":
        st.write(
            "An older note remembers its original starting line—even if "
            "today's price has moved."
        )
        columns = st.columns(2)
        reference_level = columns[0].number_input(
            "Original starting price",
            min_value=0.000001,
            value=float(st.session_state["guided_reference_level"]),
            step=1.0,
            key="guided_reference_level_widget",
        )
        prior_knock_in = columns[1].checkbox(
            "The safety line was crossed before today",
            value=bool(st.session_state["guided_prior_knock_in"]),
            key="guided_prior_knock_in_widget",
            help="Once crossed, this memory can change the payment at the end.",
        )
        st.session_state["guided_reference_level"] = reference_level
        st.session_state["guided_prior_knock_in"] = prior_knock_in
    _navigation(1)


def _render_rules() -> None:
    st.markdown("## Step 3: Draw the note's three important lines")
    st.write(
        "Pretend the starting price is **100**. Each rule is drawn as a line "
        "above or below 100. Move the controls and watch the ladder."
    )
    controls, picture = st.columns([0.9, 1.1], gap="large")
    with controls:
        autocall_pct = st.slider(
            "🏁 Early-finish line",
            min_value=70,
            max_value=140,
            value=int(st.session_state["guided_autocall_pct"]),
            key="guided_autocall_pct_widget",
            help=(
                "On a check-in day, reaching this line sends the original "
                "money back early."
            ),
        )
        coupon_barrier_pct = st.slider(
            "🍬 Reward line",
            min_value=40,
            max_value=130,
            value=int(st.session_state["guided_coupon_barrier_pct"]),
            key="guided_coupon_barrier_pct_widget",
            help="Reaching this line on a check-in day earns the reward.",
        )
        coupon_pct = st.slider(
            "Reward at each successful check (%)",
            min_value=0.0,
            max_value=10.0,
            value=float(st.session_state["guided_coupon_pct"]),
            step=0.25,
            key="guided_coupon_pct_widget",
        )
        knock_in_pct = st.slider(
            "🛟 Safety line",
            min_value=20,
            max_value=100,
            value=int(st.session_state["guided_knock_in_pct"]),
            key="guided_knock_in_pct_widget",
            help=(
                "Falling through this line can expose the final payment to "
                "the underlier's loss."
            ),
        )
        st.session_state["guided_autocall_pct"] = autocall_pct
        st.session_state["guided_coupon_barrier_pct"] = coupon_barrier_pct
        st.session_state["guided_coupon_pct"] = coupon_pct
        st.session_state["guided_knock_in_pct"] = knock_in_pct
    terms = {
        "autocall_barrier_frac": st.session_state["guided_autocall_pct"] / 100.0,
        "coupon_barrier_frac": (st.session_state["guided_coupon_barrier_pct"] / 100.0),
        "knock_in_frac": st.session_state["guided_knock_in_pct"] / 100.0,
    }
    with picture:
        st.plotly_chart(
            barrier_ladder_figure(
                barrier_levels(
                    live_spot=100.0,
                    reference_level=100.0,
                    terms=terms,
                )
            ),
            width="stretch",
            config={"displayModeBar": False},
            key="guided_barrier_playground",
        )
    valid = (
        st.session_state["guided_knock_in_pct"]
        <= st.session_state["guided_coupon_barrier_pct"]
        <= st.session_state["guided_autocall_pct"]
    )
    if valid:
        st.success(
            "These rules fit together: safety line ≤ reward line ≤ "
            "early-finish line."
        )
    else:
        st.error(
            "The lines are tangled. Keep the safety line lowest, the reward "
            "line in the middle, and the early-finish line highest."
        )
    with st.expander("Tell me the rules as a tiny story"):
        st.write(
            f"- At a check-in, **{st.session_state['guided_autocall_pct']} or "
            "higher** can finish the note early.\n"
            f"- **{st.session_state['guided_coupon_barrier_pct']} or higher** "
            f"can pay a **{st.session_state['guided_coupon_pct']:.2f}% reward**.\n"
            f"- Falling to **{st.session_state['guided_knock_in_pct']} or "
            "lower** can turn off some protection at the end."
        )
    _navigation(2, can_continue=valid)


def _render_simulation() -> None:
    st.markdown("## Step 4: Let the computer imagine many possible tomorrows")
    st.write(
        "Nobody knows the future price. So the computer makes thousands of "
        "different **pretend price stories**. It applies your three rules to "
        "every story, then averages the payments."
    )
    learning_vol_pct = st.slider(
        "Try it: make these example stories calmer or wigglier",
        min_value=5,
        max_value=60,
        value=int(st.session_state["guided_learning_vol_pct"]),
        step=5,
        key="guided_learning_vol_pct_widget",
        help="This teaching slider does not change research-market pricing.",
    )
    st.session_state["guided_learning_vol_pct"] = learning_vol_pct
    st.plotly_chart(
        learning_paths_figure(
            volatility_pct=float(st.session_state["guided_learning_vol_pct"]),
            autocall_level=float(st.session_state["guided_autocall_pct"]),
            coupon_level=float(st.session_state["guided_coupon_barrier_pct"]),
            knock_in_level=float(st.session_state["guided_knock_in_pct"]),
        ),
        width="stretch",
        config={"displayModeBar": False},
        key="guided_learning_paths",
    )
    columns = st.columns(3)
    columns[0].markdown(
        '<div class="mlp-card"><h4>1 · Imagine</h4>'
        "<p>Make many possible price paths.</p></div>",
        unsafe_allow_html=True,
    )
    columns[1].markdown(
        '<div class="mlp-card"><h4>2 · Apply rules</h4>'
        "<p>Check rewards, early finishes, and losses.</p></div>",
        unsafe_allow_html=True,
    )
    columns[2].markdown(
        '<div class="mlp-card"><h4>3 · Average</h4>'
        "<p>Bring future money back to today's value.</p></div>",
        unsafe_allow_html=True,
    )

    if _market_source() == "Manual flat market":
        st.markdown("### Choose the simple numbers used by the real calculation")
        st.caption(
            "These are annual percentages. You can change them and price again "
            "to see what moves."
        )
        columns = st.columns(4)
        spot = columns[0].number_input(
            "Price today",
            min_value=0.000001,
            value=float(st.session_state["guided_spot"]),
            step=1.0,
            key="guided_spot_widget",
        )
        rate_pct = columns[1].number_input(
            "Waiting rate (%)",
            min_value=-25.0,
            max_value=100.0,
            value=float(st.session_state["guided_rate_pct"]),
            step=0.25,
            key="guided_rate_pct_widget",
        )
        dividend_pct = columns[2].number_input(
            "Cash paid by the asset (%)",
            min_value=-25.0,
            max_value=100.0,
            value=float(st.session_state["guided_dividend_pct"]),
            step=0.25,
            key="guided_dividend_pct_widget",
        )
        market_vol_pct = columns[3].number_input(
            "Wiggliness (%)",
            min_value=0.01,
            max_value=500.0,
            value=float(st.session_state["guided_market_vol_pct"]),
            step=1.0,
            key="guided_market_vol_pct_widget",
        )
        st.session_state["guided_spot"] = spot
        st.session_state["guided_rate_pct"] = rate_pct
        st.session_state["guided_dividend_pct"] = dividend_pct
        st.session_state["guided_market_vol_pct"] = market_vol_pct
    else:
        st.info(
            "For the real calculation, the backend will collect today's "
            "research spot, rates, distributions, and option volatility."
        )
    _navigation(3)


def _build_configuration(n_paths: int, seed: int) -> PricingConfiguration:
    symbol = str(st.session_state["guided_symbol"]).strip().upper()
    if not symbol:
        raise FrontendInputError("Go back to Step 1 and choose an underlier.")
    terms = _terms()
    maturity = _maturity()
    trade_stage = (
        "New issue"
        if st.session_state["guided_trade_choice"] == "We are making it today"
        else "Seasoned trade"
    )
    contract = None
    if trade_stage == "Seasoned trade":
        schedule = even_observation_schedule(maturity, int(terms["obs_count"]))
        contract = build_v2_contract(
            reference_level=float(st.session_state["guided_reference_level"]),
            terms=terms,
            observation_times_years=schedule,
            prior_knock_in_breached=bool(st.session_state["guided_prior_knock_in"]),
        )
    manual_market = None
    if _market_source() == "Manual flat market":
        manual_market = build_flat_term_structure(
            symbol=symbol,
            underlier_type=str(st.session_state["guided_underlier_type"]),
            currency="USD",
            spot=float(st.session_state["guided_spot"]),
            risk_free_rate=float(st.session_state["guided_rate_pct"]) / 100.0,
            dividend_yield=float(st.session_state["guided_dividend_pct"]) / 100.0,
            volatility=float(st.session_state["guided_market_vol_pct"]) / 100.0,
            maturity_years=maturity,
        )
    return PricingConfiguration(
        experience_mode="Guided",
        trade_stage=trade_stage,
        market_source=_market_source(),
        symbol=symbol,
        underlier_type=str(st.session_state["guided_underlier_type"]),
        currency="USD",
        maturity_years=maturity,
        display_notional=float(st.session_state["guided_display_notional"]),
        n_paths=n_paths,
        seed=seed,
        terms=terms,
        contract=contract,
        manual_market=manual_market,
    )


def _render_review(
    n_paths: int, seed: int
) -> tuple[PricingConfiguration | None, str | None]:
    st.markdown("## Step 5: Read your recipe, then ask the computer")
    st.write(
        "You have built a Phoenix note. Before pricing, say the recipe out "
        "loud: **watch a price, check it on certain days, and follow the lines.**"
    )
    try:
        terms = _terms()
    except FrontendInputError as exc:
        st.error(str(exc))
        _navigation(4)
        return None, str(exc)
    rows = [
        ("Thing we watch", st.session_state["guided_symbol"]),
        ("Story length", st.session_state["guided_time_choice"]),
        ("Number of check-ins", str(_observation_count())),
        (
            "Reward",
            f"{float(st.session_state['guided_coupon_pct']):.2f}% at a successful check",
        ),
        (
            "Three lines",
            (
                f"safety {terms['knock_in_frac']:.0%} · "
                f"reward {terms['coupon_barrier_frac']:.0%} · "
                f"early finish {terms['autocall_barrier_frac']:.0%}"
            ),
        ),
        ("Pretend futures", f"{n_paths:,}"),
    ]
    st.markdown(
        '<div class="mlp-recipe">'
        + "".join(
            f"<div><span>{escape(str(label))}</span>"
            f"<b>{escape(str(value))}</b></div>"
            for label, value in rows
        )
        + "</div>",
        unsafe_allow_html=True,
    )
    display_notional = st.number_input(
        "How much pretend money should we display?",
        min_value=1.0,
        value=float(st.session_state["guided_display_notional"]),
        step=10_000.0,
        key="guided_display_notional_widget",
        help="The model prices one unit, then the screen scales it to this amount.",
    )
    st.session_state["guided_display_notional"] = display_notional
    st.info(
        "After you click, start on **Your answer**, then visit "
        "**How the note works** and **How sure are we?**"
    )
    left, _, right = st.columns([1, 1.5, 1.5])
    with left:
        st.button(
            "← Back",
            key="guided_back_4",
            on_click=_set_step,
            args=(3,),
            width="stretch",
        )
    with right:
        submitted = st.button(
            "Price my note →",
            key="guided_price",
            type="primary",
            width="stretch",
        )
    if not submitted:
        return None, None
    try:
        return _build_configuration(n_paths, seed), None
    except FrontendInputError as exc:
        return None, str(exc)


def render_guided_configuration(
    *,
    n_paths: int,
    seed: int,
) -> tuple[PricingConfiguration | None, str | None]:
    _initialize_guided_state()
    step = int(st.session_state["guided_step"])
    _step_header(step)
    if step == 0:
        _render_pick_underlier()
    elif step == 1:
        _render_clock()
    elif step == 2:
        _render_rules()
    elif step == 3:
        _render_simulation()
    else:
        return _render_review(n_paths, seed)
    return None, None
