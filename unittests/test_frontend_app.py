from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_streamlit_workspace_starts_without_contacting_backend():
    app_path = Path(__file__).parents[1] / "app" / "frontend.py"

    app = AppTest.from_file(str(app_path)).run(timeout=15)

    assert not app.exception
    assert any(button.label == "Check API connection" for button in app.button)
    assert any(button.label == "Price and build diagnostics" for button in app.button)
