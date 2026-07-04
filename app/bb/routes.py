from html import escape
from typing import Iterable

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

from app.services.product_registry import get_model_info


router = APIRouter(prefix="/bb", tags=["blackberry"])


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
  font-size: 14px;
  line-height: 1.25;
  margin: 6px;
}}
a {{ color: #000; text-decoration: none; }}
.terminal {{ max-width: 460px; }}
.head {{ font-weight: bold; margin-bottom: 8px; }}
.menu div {{ margin: 4px 0; }}
.status {{ border-top: 1px solid #222; margin-top: 8px; padding-top: 6px; }}
.small {{ font-size: 12px; }}
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


@router.get("", response_class=HTMLResponse)
@router.get("/", response_class=HTMLResponse)
def blackberry_home():
    info = get_model_info()
    body = f"""
<div class="head">
<pre>ML-PRICER TERMINAL
BB-9780 CLIENT</pre>
</div>
<div class="menu">
<div>[1] PRICE NOTE</div>
<div>[2] SHOCK SCENARIO</div>
<div>[3] EXPLAIN PAYOFF</div>
<div><a href="/bb/model-status">[4] MODEL STATUS</a></div>
</div>
<div class="status">
<pre>API: {escape(info["api"].upper())}
MODEL: {escape(info["model_family"])}
PRODUCTS: {len(info["available_product_keys"])}/{len(info["supported_product_keys"])} READY</pre>
</div>
<div class="small">READ-ONLY SHELL</div>
"""
    return terminal_page("ML-Pricer BB Terminal", body)


@router.get("/model-status", response_class=HTMLResponse)
def blackberry_model_status():
    info = get_model_info()
    rows = product_rows(info["products"])
    body = f"""
<div class="head">
<pre>MODEL STATUS
ML-PRICER BB</pre>
</div>
<pre>API: {escape(info["api"].upper())}
FAMILY: {escape(info["model_family"])}
MC FALLBACK: YES

PRODUCT              READY MODEL SCALER
{rows}</pre>
<div class="status"><a href="/bb">[0] HOME</a></div>
"""
    return terminal_page("ML-Pricer Model Status", body)
