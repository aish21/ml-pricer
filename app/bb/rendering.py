from datetime import datetime
from html import escape
from typing import Iterable

from fastapi.responses import HTMLResponse


def terminal_page(title: str, body: str) -> HTMLResponse:
    html = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=480">
<title>{escape(title)}</title>
<style>
body {{
  background: #f4f4f0;
  color: #111;
  font-family: monospace;
  font-size: 13px;
  line-height: 1.2;
  margin: 6px;
}}
a {{ color: #000; text-decoration: none; }}
.terminal {{ max-width: 460px; }}
.head {{ border-bottom: 1px solid #222; font-weight: bold; margin-bottom: 6px; padding-bottom: 3px; }}
.menu div {{ margin: 3px 0; }}
.status {{ border-top: 1px solid #222; margin-top: 8px; padding-top: 6px; }}
.small {{ font-size: 12px; }}
form {{ margin: 0; }}
label {{ display: inline-block; width: 58px; }}
input, select {{
  font-family: monospace;
  font-size: 14px;
  margin: 2px 0;
  max-width: 150px;
}}
button {{
  font-family: monospace;
  font-size: 14px;
  margin-top: 6px;
}}
pre {{ font-family: monospace; white-space: pre-wrap; margin: 0; }}
</style>
</head>
<body>
<div class="terminal">
{body}
</div>
</body>
</html>"""
    return HTMLResponse(html)


def product_rows(products: Iterable[dict]) -> str:
    rows = []
    for product in products:
        artifacts = product["artifacts"]
        ready = "Y" if artifacts["ready_for_surrogate"] else "N"
        model = "Y" if artifacts["model_available"] else "N"
        scaler = "Y" if artifacts["scaler_available"] else "N"
        key = escape(product["key"])
        rows.append(f"{key:<20} {ready:<5} {model:<5} {scaler:<6}")
    return "\n".join(rows)


def compact_run_id(run_id: str) -> str:
    if not run_id:
        return "N/A"
    if "_" in run_id:
        return run_id.rsplit("_", 1)[-1][-8:]
    return run_id[-8:]


def compact_timestamp(created_at: str) -> str:
    if not created_at:
        return "N/A"
    try:
        return datetime.fromisoformat(created_at).strftime("%H:%M")
    except ValueError:
        return created_at[:16]


def format_number(value, digits: int = 6) -> str:
    if value is None:
        return "N/A"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "N/A"


def format_percent(value) -> str:
    formatted = format_number(value, digits=2)
    return "N/A" if formatted == "N/A" else f"{formatted}%"


def format_shock(value, suffix: str = "") -> str:
    if value is None:
        return "N/A"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "N/A"
    sign = "+" if numeric > 0 else ""
    return f"{sign}{numeric:g}{suffix}"


def run_price(run: dict) -> str:
    result = run.get("result_payload") or {}
    if run.get("run_type") == "scenario":
        return format_number(result.get("shocked_price"))
    return format_number(result.get("price"))


def terminal_error(
    reason: str,
    primary_href: str = "/bb/recent-runs",
    primary_label: str = "RECENT RUNS",
) -> HTMLResponse:
    body = f"""
<div class="head"><pre>ERROR</pre></div>
<pre>Reason:
{escape(reason)}</pre>
<div class="status">
<div><a href="{escape(primary_href)}">[1] {escape(primary_label)}</a></div>
<div><a href="/bb">[2] HOME</a></div>
</div>
"""
    return terminal_page("ML-Pricer Error", body)
