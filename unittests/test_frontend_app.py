from pathlib import Path

from streamlit.testing.v1 import AppTest

from app.ui.guided import (
    FOUNDATION_GLOSSARY,
    GLOSSARY_VISUALS,
    glossary_entry,
    glossary_visual_html,
    teaching_check_outcome,
)
from app.ui.results import (
    ML_LESSON_STAGES,
    _interpretation_html,
    _ml_comparison_summary,
    _table_html,
    ml_lesson_stage,
)


def test_streamlit_workspace_starts_without_contacting_backend():
    app_path = Path(__file__).parents[1] / "app" / "frontend.py"

    app = AppTest.from_file(str(app_path)).run(timeout=15)

    assert not app.exception
    assert any(button.label == "Check API connection" for button in app.button)
    assert any(button.key == "guided_next_0" for button in app.button)
    assert any(
        "Build a Phoenix note and see how it is priced." in item.value
        for item in app.markdown
    )
    assert all("Open backend API docs" not in item.value for item in app.markdown)
    assert any(
        selectbox.label == "Pick the thing whose price we will watch"
        for selectbox in app.selectbox
    )
    assert any(
        selectbox.label == "Word shelf — choose any word to unpack"
        for selectbox in app.selectbox
    )


def test_zero_knowledge_glossary_covers_contract_market_and_model_terms():
    assert {
        "Note",
        "Underlier",
        "Issuer",
        "Barrier",
        "Autocall",
        "Autocall barrier",
        "Coupon barrier",
        "Knock-in barrier",
        "Observation date",
        "Principal",
        "Volatility",
        "Monte Carlo",
        "Machine-learning model",
        "Training example",
        "Shadow mode",
    } <= FOUNDATION_GLOSSARY.keys()
    headline, explanation, why_it_matters = glossary_entry("Note")
    assert "IOU" in headline
    assert "issuer" in explanation.lower()
    assert "share" in why_it_matters.lower()


def test_every_glossary_term_has_an_accessible_animated_visual():
    assert GLOSSARY_VISUALS.keys() == FOUNDATION_GLOSSARY.keys()

    for term in FOUNDATION_GLOSSARY:
        rendered = glossary_visual_html(term)
        assert 'class="mlp-concept-visual ' in rendered
        assert 'role="img"' in rendered
        assert 'aria-label="' in rendered
        assert "<svg " in rendered
        assert rendered.count('class="mlp-concept-step"') == 3
        assert "<script" not in rendered


def test_word_shelf_swaps_the_visual_when_the_term_changes():
    app_path = Path(__file__).parents[1] / "app" / "frontend.py"
    app = AppTest.from_file(str(app_path)).run(timeout=15)
    shelf = next(
        selectbox
        for selectbox in app.selectbox
        if selectbox.label.startswith("Word shelf")
    )

    shelf.set_value("Volatility")
    app.run(timeout=15)

    assert not app.exception
    assert any(
        "mlp-concept-rose" in item.value and ">Volatility<" in item.value
        for item in app.markdown
    )


def test_zero_knowledge_rule_game_explains_each_possible_outcome():
    scenarios = {
        110: ("The note finishes early", "finish"),
        102: ("A reward is earned; the note keeps going", "reward"),
        85: ("No reward at this check; the note keeps going", "wait"),
        65: ("The safety memory switches on", "risk"),
    }

    for observed, (expected_title, expected_tone) in scenarios.items():
        title, explanation, tone = teaching_check_outcome(
            observed,
            autocall_level=105,
            coupon_level=100,
            knock_in_level=70,
        )
        assert title == expected_title
        assert explanation
        assert tone == expected_tone


def test_ml_lesson_says_the_model_prices_notes_not_market_direction():
    assert len(ML_LESSON_STAGES) == 3
    title, explanation, takeaway = ml_lesson_stage("2 · Student practises")

    assert "ML" in title
    assert "market numbers and note rules" in explanation
    assert "stock tips" in takeaway


def test_guided_workspace_walks_to_review_without_backend_contact():
    app_path = Path(__file__).parents[1] / "app" / "frontend.py"
    app = AppTest.from_file(str(app_path)).run(timeout=15)

    for key in (
        "guided_next_0",
        "guided_next_1",
        "guided_next_2",
        "guided_next_3",
    ):
        next(button for button in app.button if button.key == key).click()
        app.run(timeout=15)
        assert not app.exception

    assert any(button.key == "guided_price" for button in app.button)
    assert any("Step 5: Read your recipe" in item.value for item in app.markdown)
    notional = next(
        item
        for item in app.number_input
        if item.label == "How much pretend money should we display?"
    )
    assert notional.value == 100_000.0


def test_guided_workspace_can_jump_directly_between_sections():
    app_path = Path(__file__).parents[1] / "app" / "frontend.py"
    app = AppTest.from_file(str(app_path)).run(timeout=15)

    next(button for button in app.button if button.key == "guided_jump_4").click()
    app.run(timeout=15)
    assert not app.exception
    assert any("Step 5: Read your recipe" in item.value for item in app.markdown)

    next(button for button in app.button if button.key == "guided_jump_1").click()
    app.run(timeout=15)
    assert not app.exception
    assert any("Step 2: Choose how long" in item.value for item in app.markdown)


def test_quant_workspace_keeps_direct_pricing_form():
    app_path = Path(__file__).parents[1] / "app" / "frontend.py"
    app = AppTest.from_file(str(app_path)).run(timeout=15)

    next(radio for radio in app.radio if radio.label == "Experience").set_value("Quant")
    app.run(timeout=15)

    assert not app.exception
    assert any(button.label == "Price and build diagnostics" for button in app.button)
    assert any(selectbox.label == "Search underlier" for selectbox in app.selectbox)


def test_native_results_table_formats_and_escapes_values():
    rendered = _table_html(
        [{"label": "<script>", "value": 12.34567, "resolved": True}],
        [("label", "Label"), ("value", "Value"), ("resolved", "Resolved")],
        number_formats={"value": ",.2f"},
    )

    assert "<script>" not in rendered
    assert "&lt;script&gt;" in rendered
    assert "12.35" in rendered
    assert "Yes" in rendered


def test_result_interpretation_is_one_continuous_safe_html_block():
    rendered = _interpretation_html(
        [
            {"title": "First <rule>", "body": "One & only"},
            {"title": "Second rule", "body": "Still inside the list"},
        ]
    )

    assert "\n" not in rendered
    assert rendered.count('class="mlp-interpretation"') == 2
    assert "&lt;rule&gt;" in rendered
    assert "One &amp; only" in rendered
    assert rendered.startswith('<div class="mlp-interpretation-list">')
    assert rendered.endswith("</div>")


def test_ml_comparison_summary_keeps_reference_and_surrogate_distinct():
    summary = _ml_comparison_summary(
        {
            "price": 1.02,
            "latency_ms": 120,
            "confidence_interval": [1.01, 1.03],
            "surrogate_shadow": {
                "status": "success",
                "used_for_price": False,
                "reference_price": 1.02,
                "surrogate_price": 1.018,
                "latency_ms": 0.4,
                "error_to_reference_standard_error": 0.4,
                "model_version": "test-model",
            },
        }
    )

    assert summary["available"] is True
    assert summary["absolute_gap"] == 0.0020000000000000018
    assert summary["relative_gap"] == summary["absolute_gap"] / 1.02
    assert summary["speedup"] == 300.0
    assert summary["inside_reference_interval"] is True
