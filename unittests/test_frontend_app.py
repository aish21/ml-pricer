from pathlib import Path

from streamlit.testing.v1 import AppTest

from app.ui.guided import FOUNDATION_GLOSSARY, glossary_entry
from app.ui.results import _interpretation_html, _ml_comparison_summary, _table_html


def test_streamlit_workspace_starts_without_contacting_backend():
    app_path = Path(__file__).parents[1] / "app" / "frontend.py"

    app = AppTest.from_file(str(app_path)).run(timeout=15)

    assert not app.exception
    assert any(button.label == "Check API connection" for button in app.button)
    assert any(button.key == "guided_next_0" for button in app.button)
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
    } <= FOUNDATION_GLOSSARY.keys()
    headline, explanation, why_it_matters = glossary_entry("Note")
    assert "IOU" in headline
    assert "issuer" in explanation.lower()
    assert "share" in why_it_matters.lower()


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
