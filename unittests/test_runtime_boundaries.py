import subprocess
import sys
import tomllib
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_api_import_does_not_eagerly_load_training_stack():
    script = """
import sys
import app.backend

forbidden = sorted(
    name for name in ("lightgbm", "optuna", "sklearn") if name in sys.modules
)
if forbidden:
    raise SystemExit(f"training dependencies imported by API: {forbidden}")
"""

    subprocess.run([sys.executable, "-c", script], check=True)


def test_api_extra_declares_yfinance_repair_dependencies():
    project = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text("utf-8"))
    api_dependencies = set(project["project"]["optional-dependencies"]["api"])

    assert {"scipy==1.13.0", "yfinance==1.5.1"} <= api_dependencies


def test_frontend_pins_modern_streamlit_arrow_compatibility_boundary():
    project = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text("utf-8"))
    frontend_dependencies = set(project["project"]["optional-dependencies"]["frontend"])

    assert {"streamlit==1.59.0", "pyarrow==21.0.0"} <= frontend_dependencies
