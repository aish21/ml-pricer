import subprocess
import sys


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
