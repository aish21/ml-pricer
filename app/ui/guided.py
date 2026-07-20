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
    "Machine-learning model": (
        "The ML model is a fast student that copies a slower pricing teacher.",
        "It studies many examples where contract and market inputs are paired "
        "with prices from the Monte Carlo pricer. It learns a mathematical "
        "shortcut from those examples.",
        "Here it predicts the note's estimated price—not tomorrow's share price, "
        "not whether you should invest, and not a guaranteed payment.",
    ),
    "Training example": (
        "A training example is one practice question with its answer.",
        "The question contains a note's rules and market numbers. The answer is "
        "the price calculated by the slower Monte Carlo teacher.",
        "Many varied examples help the ML student learn the shape of the pricing "
        "problem instead of memorising one note.",
    ),
    "Shadow mode": (
        "Shadow mode lets the ML student answer without putting it in charge.",
        "Monte Carlo and ML both price the same request. The screen compares "
        "their answers, but the official displayed reference remains Monte Carlo.",
        "This gives us real-world evidence about accuracy and speed before any "
        "human considers trusting the shortcut more.",
    ),
}

GLOSSARY_VISUALS: dict[
    str,
    tuple[str, str, tuple[tuple[str, str], ...], str],
] = {
    "Note": (
        "flow",
        "violet",
        (
            ("$", "Money goes in"),
            ("▤", "Rulebook waits"),
            ("$", "Rules decide money out"),
        ),
        "A note carries money through a written rulebook to possible future payments.",
    ),
    "Underlier": (
        "market",
        "cyan",
        (("◉", "Choose an asset"), ("〽", "Its price moves"), ("▤", "The note reacts")),
        "The underlier is the moving scoreboard watched by the note.",
    ),
    "Issuer": (
        "flow",
        "violet",
        (("$", "Investor pays"), ("▥", "Issuer promises"), ("↩", "Issuer must pay")),
        "The issuer receives money and stands behind the note's promised payments.",
    ),
    "Notional": (
        "scale",
        "gold",
        (
            ("1", "Price one unit"),
            ("×", "Multiply by amount"),
            ("$", "Show money value"),
        ),
        "Notional scales a per-unit model price into the displayed money amount.",
    ),
    "Price": (
        "paths",
        "cyan",
        (
            ("⑂", "Imagine futures"),
            ("▤", "Apply every rule"),
            ("$", "Value them today"),
        ),
        "A note price compresses many possible future payments into one estimate today.",
    ),
    "Coupon": (
        "rule",
        "gold",
        (("●", "Reach check-in"), ("━", "Clear reward line"), ("✦", "Coupon can pay")),
        "A coupon is conditional: the check-in price must pass its reward rule.",
    ),
    "Autocall": (
        "timeline",
        "teal",
        (
            ("●", "Reach check-in"),
            ("◆", "Clear finish line"),
            ("↩", "Principal returns"),
        ),
        "An autocall ends the note automatically when a scheduled test succeeds.",
    ),
    "Barrier": (
        "rule",
        "violet",
        (("〽", "Price approaches"), ("━", "Rule line"), ("?", "Outcome changes")),
        "A barrier is a boundary: crossing it can switch the rulebook's answer.",
    ),
    "Autocall barrier": (
        "rule",
        "teal",
        (("●", "Check-in price"), ("🏁", "Finish line"), ("↩", "Return and stop")),
        "At a scheduled check, this line can send principal back and stop the note.",
    ),
    "Coupon barrier": (
        "rule",
        "gold",
        (("●", "Check-in price"), ("✦", "Reward line"), ("+$", "Earn this coupon")),
        "At or above this line on a check-in day, that date's reward can be earned.",
    ),
    "Knock-in barrier": (
        "rule",
        "rose",
        (
            ("↓", "Price crosses"),
            ("!", "Memory turns on"),
            ("↘", "Weak ending exposed"),
        ),
        "Crossing the safety line stores a memory that can matter at maturity.",
    ),
    "Observation date": (
        "timeline",
        "cyan",
        (("○", "Time passes"), ("●", "Check day arrives"), ("?", "Rules are tested")),
        "An observation date is a marked day when coupon and autocall questions are asked.",
    ),
    "Principal": (
        "flow",
        "teal",
        (("$", "Original unit"), ("▤", "Note runs"), ("↩", "May be returned")),
        "Principal is the original unit amount the note may return early or at the end.",
    ),
    "Maturity": (
        "timeline",
        "violet",
        (("○", "Note starts"), ("···", "Time passes"), ("■", "Final rules run")),
        "Maturity is the planned final stop if the note has not already ended.",
    ),
    "Volatility": (
        "paths",
        "rose",
        (("—", "Calm movement"), ("〽", "Wigglier paths"), ("↕", "Wider outcomes")),
        "Volatility widens the fan of possible journeys without choosing up or down.",
    ),
    "Monte Carlo": (
        "paths",
        "cyan",
        (("⑂", "Many futures"), ("▤", "Rules on each"), ("÷", "Average the values")),
        "Monte Carlo repeats the same rulebook across many pretend futures, then averages.",
    ),
    "Machine-learning model": (
        "learning",
        "gold",
        (("▥", "Market + rules"), ("ML", "Student learns"), ("$", "Fast price guess")),
        "The ML student learns a shortcut from note inputs to the teacher's price.",
    ),
    "Training example": (
        "learning",
        "cyan",
        (("?", "Practice question"), ("$", "Teacher answer"), ("↻", "Student adjusts")),
        "One training example pairs a pricing question with the teacher's answer.",
    ),
    "Shadow mode": (
        "compare",
        "violet",
        (
            ("T", "Teacher answers"),
            ("ML", "Student answers"),
            ("≍", "Compare; teacher leads"),
        ),
        "Shadow mode compares both prices while Monte Carlo remains in charge.",
    ),
}

_GLOSSARY_VISUAL_SVGS = {
    "flow": (
        '<svg viewBox="0 0 280 132" aria-hidden="true">'
        '<path class="mlp-concept-line" d="M24 66 H256"/>'
        '<circle class="mlp-concept-orb" cx="34" cy="66" r="13"/>'
        '<rect class="mlp-concept-shape" x="120" y="42" width="40" height="48" rx="4"/>'
        '<circle class="mlp-concept-shape" cx="246" cy="66" r="18"/>'
        "</svg>"
    ),
    "market": (
        '<svg viewBox="0 0 280 132" aria-hidden="true">'
        '<path class="mlp-concept-grid" d="M20 26 V108 M20 108 H260"/>'
        '<path class="mlp-concept-line" d="M22 91 C55 85 64 39 96 58 S139 104 169 71 '
        'S215 31 258 45"/>'
        '<circle class="mlp-concept-orb" cx="258" cy="45" r="9"/>'
        "</svg>"
    ),
    "scale": (
        '<svg viewBox="0 0 280 132" aria-hidden="true">'
        '<path class="mlp-concept-grid" d="M140 31 V102 M70 102 H210"/>'
        '<path class="mlp-concept-line" d="M61 48 H219"/>'
        '<path class="mlp-concept-shape" d="M61 48 L34 92 H88 Z M219 48 L192 92 H246 Z"/>'
        '<circle class="mlp-concept-orb" cx="140" cy="31" r="10"/>'
        "</svg>"
    ),
    "paths": (
        '<svg viewBox="0 0 280 132" aria-hidden="true">'
        '<path class="mlp-concept-path mlp-path-one" d="M20 66 C62 25 83 101 126 58 '
        'S209 20 260 41"/>'
        '<path class="mlp-concept-path mlp-path-two" d="M20 66 C68 78 84 31 132 76 '
        'S207 111 260 92"/>'
        '<path class="mlp-concept-line" d="M20 66 C67 55 94 70 137 62 S215 59 260 66"/>'
        '<circle class="mlp-concept-orb" cx="20" cy="66" r="10"/>'
        "</svg>"
    ),
    "rule": (
        '<svg viewBox="0 0 280 132" aria-hidden="true">'
        '<path class="mlp-concept-threshold" d="M24 66 H256"/>'
        '<path class="mlp-concept-line" d="M25 102 C80 101 84 89 126 84 '
        'S182 45 255 37"/>'
        '<circle class="mlp-concept-orb" cx="150" cy="71" r="10"/>'
        "</svg>"
    ),
    "timeline": (
        '<svg viewBox="0 0 280 132" aria-hidden="true">'
        '<path class="mlp-concept-line" d="M24 72 H256"/>'
        '<circle class="mlp-concept-shape" cx="44" cy="72" r="8"/>'
        '<circle class="mlp-concept-shape" cx="140" cy="72" r="12"/>'
        '<circle class="mlp-concept-shape" cx="236" cy="72" r="16"/>'
        '<circle class="mlp-concept-orb" cx="44" cy="72" r="9"/>'
        "</svg>"
    ),
    "learning": (
        '<svg viewBox="0 0 280 132" aria-hidden="true">'
        '<path class="mlp-concept-grid" d="M52 37 L137 54 M52 66 L137 66 '
        'M52 95 L137 78 M137 54 L227 66 M137 78 L227 66"/>'
        '<circle class="mlp-concept-shape" cx="52" cy="37" r="8"/>'
        '<circle class="mlp-concept-shape" cx="52" cy="66" r="8"/>'
        '<circle class="mlp-concept-shape" cx="52" cy="95" r="8"/>'
        '<circle class="mlp-concept-shape" cx="137" cy="54" r="11"/>'
        '<circle class="mlp-concept-shape" cx="137" cy="78" r="11"/>'
        '<circle class="mlp-concept-orb" cx="227" cy="66" r="16"/>'
        "</svg>"
    ),
    "compare": (
        '<svg viewBox="0 0 280 132" aria-hidden="true">'
        '<path class="mlp-concept-path mlp-path-one" d="M25 41 C87 41 109 58 167 64 H252"/>'
        '<path class="mlp-concept-path mlp-path-two" d="M25 91 C87 91 109 75 167 68 H252"/>'
        '<circle class="mlp-concept-shape" cx="25" cy="41" r="12"/>'
        '<circle class="mlp-concept-shape" cx="25" cy="91" r="12"/>'
        '<circle class="mlp-concept-orb" cx="252" cy="66" r="13"/>'
        "</svg>"
    ),
}


def glossary_entry(term: str) -> tuple[str, str, str]:
    return FOUNDATION_GLOSSARY[term]


def glossary_visual_html(term: str) -> str:
    kind, tone, steps, caption = GLOSSARY_VISUALS[term]
    step_html = "".join(
        '<div class="mlp-concept-step">'
        f'<span aria-hidden="true">{escape(glyph)}</span>'
        f"<b>{escape(label)}</b></div>"
        for glyph, label in steps
    )
    return (
        f'<div class="mlp-concept-visual mlp-concept-{escape(tone)}" '
        f'role="img" aria-label="{escape(caption, quote=True)}">'
        '<div class="mlp-concept-heading"><span>Watch the idea move</span>'
        f"<strong>{escape(term)}</strong></div>"
        '<div class="mlp-concept-layout"><div class="mlp-concept-canvas">'
        f"{_GLOSSARY_VISUAL_SVGS[kind]}</div>"
        f'<div class="mlp-concept-steps">{step_html}</div></div>'
        f'<p class="mlp-concept-caption">{escape(caption)}</p></div>'
    )


def teaching_check_outcome(
    observed_level: float,
    *,
    autocall_level: float,
    coupon_level: float,
    knock_in_level: float,
) -> tuple[str, str, str]:
    """Explain one toy observation without changing the priced contract."""
    level = float(observed_level)
    if level >= float(autocall_level):
        return (
            "The note finishes early",
            "The price reached the finish line on a check-in day. The rulebook "
            "returns principal, includes the successful reward, and stops.",
            "finish",
        )
    if level >= float(coupon_level):
        return (
            "A reward is earned; the note keeps going",
            "The price cleared the reward line but not the finish line. This "
            "check can pay a coupon, then the story continues.",
            "reward",
        )
    if level <= float(knock_in_level):
        return (
            "The safety memory switches on",
            "The price crossed the safety line. That is not an instant loss. "
            "It means a weak ending may now link principal to the underlier's fall.",
            "risk",
        )
    return (
        "No reward at this check; the note keeps going",
        "The price is below the reward line but above the safety line. Nothing "
        "ends today, and this non-memory note does not save the missed reward.",
        "wait",
    )


def _render_focus_card(number: str, title: str, body: str) -> None:
    st.markdown(
        f"""
        <div class="mlp-focus-card">
          <span>Only one idea for this screen · {escape(number)}</span>
          <strong>{escape(title)}</strong>
          <p>{escape(body)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_quick_check(
    *,
    label: str,
    options: list[str],
    correct: str,
    explanation: str,
    key: str,
) -> None:
    answer = st.radio(
        label,
        options,
        index=None,
        key=key,
        horizontal=True,
    )
    if answer is None:
        st.caption("Pick an answer to get instant feedback. This is not graded.")
    elif answer == correct:
        st.success(f"Yes. {explanation}")
    else:
        st.info(f"Almost. {explanation}")


def _initialize_guided_state() -> None:
    for name, value in GUIDED_DEFAULTS.items():
        if name not in st.session_state:
            st.session_state[name] = value


def _set_step(step: int) -> None:
    st.session_state["guided_step"] = max(0, min(step, len(GUIDED_STEPS) - 1))


def _step_header(step: int) -> None:
    st.caption("Jump to any section—your choices are kept as you move around.")
    with st.container(key="guided_step_navigation"):
        columns = st.columns(len(GUIDED_STEPS), gap="small")
        for index, (number, title) in enumerate(GUIDED_STEPS):
            columns[index].button(
                f"{number} · {title}",
                key=f"guided_jump_{index}",
                type="primary" if index == step else "secondary",
                on_click=_set_step,
                args=(index,),
                width="stretch",
            )
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
          <span class="mlp-section-label">Start here · no finance knowledge needed</span>
          <h3>We are estimating what a promise is worth today.</h3>
          <div class="mlp-foundation-flow">
            <div><b>1 · Money goes in</b><span>an investor buys the note</span></div>
            <i>→</i>
            <div><b>2 · A price moves</b><span>the note watches its scoreboard</span></div>
            <i>→</i>
            <div><b>3 · Rules pay out</b><span>different journeys can pay differently</span></div>
          </div>
          <p>
            The <b>underlier</b> is the scoreboard. The <b>note</b> is the prize
            rulebook. The <b>pricer</b> estimates today's value of all the
            possible prizes. You will build those ideas one at a time.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    term = st.selectbox(
        "Word shelf — choose any word to unpack",
        list(FOUNDATION_GLOSSARY),
        key="guided_glossary_term",
        help=(
            "Choose a word and the animated picture, three-step story, and "
            "plain-language explanation will all change together."
        ),
    )
    headline, explanation, why_it_matters = glossary_entry(str(term))
    st.markdown(
        glossary_visual_html(str(term)),
        unsafe_allow_html=True,
    )
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
    _render_focus_card(
        "1",
        "Choose the scoreboard",
        "For now, forget the note's rules. We only need to choose the changing "
        "number those rules will watch.",
    )
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
    _render_quick_check(
        label="Tiny check: in this lesson, what is the underlier?",
        options=[
            "The price the note watches",
            "The note itself",
            "A guaranteed reward",
        ],
        correct="The price the note watches",
        explanation=(
            "The underlier is the scoreboard. The note is the separate "
            "rulebook that watches it."
        ),
        key="guided_underlier_check",
    )
    _navigation(0, can_continue=bool(symbol))


def _render_clock() -> None:
    st.markdown("## Step 2: Choose how long the story lasts")
    _render_focus_card(
        "2",
        "Give the rulebook a calendar",
        "The note only makes decisions on named days. A price move on another "
        "day may not trigger the same rule.",
    )
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
    _render_quick_check(
        label="Tiny check: when does the note test its reward rule?",
        options=[
            "On its scheduled check-in days",
            "Every second of every day",
            "Only when the investor asks",
        ],
        correct="On its scheduled check-in days",
        explanation=(
            "Coupon and early-finish rules are tested on the contract's "
            "scheduled observation dates."
        ),
        key="guided_calendar_check",
    )
    _navigation(1)


def _render_rules() -> None:
    st.markdown("## Step 3: Draw the note's three important lines")
    _render_focus_card(
        "3",
        "Turn prices into simple yes-or-no rules",
        "At a check, the note asks whether the scoreboard is above or below "
        "each line. Those answers decide what happens next.",
    )
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
    if valid:
        st.markdown("### Try one pretend check-in")
        st.caption(
            "Move the pretend price. This little game explains your rules; it "
            "does not change the contract you will price."
        )
        observed_level = st.slider(
            "The scoreboard says",
            min_value=20,
            max_value=140,
            value=100,
            key="guided_observed_level",
        )
        outcome_title, outcome_body, outcome_tone = teaching_check_outcome(
            observed_level,
            autocall_level=float(st.session_state["guided_autocall_pct"]),
            coupon_level=float(st.session_state["guided_coupon_barrier_pct"]),
            knock_in_level=float(st.session_state["guided_knock_in_pct"]),
        )
        st.markdown(
            f"""
            <div class="mlp-outcome-stage mlp-outcome-{escape(outcome_tone)}">
              <span>Scoreboard {observed_level} · rulebook reacts</span>
              <strong>{escape(outcome_title)}</strong>
              <p>{escape(outcome_body)}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    _navigation(2, can_continue=valid)


def _render_simulation() -> None:
    st.markdown("## Step 4: Let the computer imagine many possible tomorrows")
    _render_focus_card(
        "4",
        "Nobody knows the future, so try many pretend futures",
        "One imagined journey proves nothing. Thousands of different journeys "
        "give the pricer a useful average.",
    )
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
    st.markdown(
        """
        <div class="mlp-model-school">
          <span class="mlp-section-label">Where machine learning enters</span>
          <div class="mlp-school-flow">
            <div><b>Slow teacher</b><small>Monte Carlo solves many pretend futures</small></div>
            <i>teaches with lots of priced examples</i>
            <div><b>Fast student</b><small>ML learns to copy the final note price</small></div>
          </div>
          <p>
            The ML student sees the market numbers and the note's rules. It
            predicts the <b>note price the teacher would produce</b>. It is not
            guessing whether the underlier goes up tomorrow.
          </p>
          <div class="mlp-race-track">
            <span>Monte Carlo teacher</span><div><i class="mlp-runner-slow"></i></div>
            <span>ML student</span><div><i class="mlp-runner-fast"></i></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    _render_quick_check(
        label="Tiny check: what is the ML model trying to predict here?",
        options=[
            "The note's estimated price",
            "Tomorrow's share price",
            "A guaranteed profit",
        ],
        correct="The note's estimated price",
        explanation=(
            "It learns a shortcut from note-and-market inputs to the Monte "
            "Carlo teacher's estimated note price."
        ),
        key="guided_ml_target_check",
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
