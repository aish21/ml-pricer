from html import escape
from urllib.parse import parse_qs

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from app.bb.rendering import (
    compact_run_id,
    compact_timestamp,
    format_number,
    format_percent,
    format_shock,
    product_rows,
    run_price,
    terminal_error,
    terminal_page,
)

from app.services.product_registry import get_model_info
from app.services.pricing_service import (
    PricingServiceError,
    get_bb_pricing_products,
    price_product,
)
from app.services.scenario_service import (
    ScenarioServiceError,
    run_scenario,
)
from app.services.run_store import get_run, list_recent_runs, save_run


router = APIRouter(prefix="/bb", tags=["blackberry"])

PHOENIX_FORM_FIELDS = (
    ("S0", "Spot", "100.0"),
    ("sigma", "Vol", "0.2"),
    ("r", "Rate", "0.03"),
    ("T", "Mat", "1.0"),
    ("autocall_barrier_frac", "AutoB", "1.05"),
    ("coupon_barrier_frac", "CpnB", "1.0"),
    ("coupon_rate", "Cpn", "0.02"),
    ("knock_in_frac", "KI", "0.7"),
    ("obs_count", "Obs", "6"),
)

SCENARIO_FORM_FIELDS = (
    ("spot_pct", "Spot %", ""),
    ("vol_abs", "Vol abs", ""),
    ("rate_bps", "Rate bp", ""),
)


async def read_urlencoded_form(request: Request) -> dict[str, str]:
    raw_body = (await request.body()).decode("utf-8")
    parsed = parse_qs(raw_body, keep_blank_values=True)
    return {key: values[-1] if values else "" for key, values in parsed.items()}


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
<div><a href="/bb/price">[1] PRICE NOTE</a></div>
<div><a href="/bb/recent-runs">[2] RECENT RUNS</a></div>
<div><a href="/bb/model-status">[3] MODEL STATUS</a></div>
</div>
<div class="status">
<pre>API: {escape(info["api"].upper())}
MODEL: {escape(info["model_family"])}
PRODUCTS: {len(info["available_product_keys"])}/{len(info["supported_product_keys"])} READY</pre>
</div>
<div class="small">LOCAL TERMINAL</div>
"""
    return terminal_page("ML-Pricer BB Terminal", body)


@router.get("/price", response_class=HTMLResponse)
def blackberry_price_form():
    product_options = "\n".join(
        f'<option value="{escape(product["key"])}">{escape(product["display_name"])}</option>'
        for product in get_bb_pricing_products()
    )
    fields = "\n".join(
        f'<div><label for="{escape(name)}">{escape(label)}:</label>'
        f'<input id="{escape(name)}" name="{escape(name)}" value="{escape(default)}"></div>'
        for name, label, default in PHOENIX_FORM_FIELDS
    )
    body = f"""
<div class="head">
<pre>ML-PRICER BB TERMINAL
PRICE NOTE</pre>
</div>
<form method="post" action="/bb/price">
<div><label for="product_key">Prod:</label>
<select id="product_key" name="product_key">
{product_options}
</select></div>
{fields}
<div><label for="n_paths">Paths:</label><input id="n_paths" name="n_paths" value="500"></div>
<button type="submit">PRICE</button>
</form>
<div class="status"><a href="/bb">[0] HOME</a></div>
"""
    return terminal_page("ML-Pricer Price Note", body)


@router.post("/price")
async def blackberry_price_submit(request: Request):
    form = await read_urlencoded_form(request)
    product_key = form.get("product_key", "")
    n_paths = form.get("n_paths", "500")
    params = {name: form.get(name, "") for name, _, _ in PHOENIX_FORM_FIELDS}

    try:
        result = price_product(product_key=product_key, params=params, n_paths=n_paths)
        request_payload = {
            "product_key": product_key,
            "params": result["params"],
            "n_paths": result["n_paths"],
            "use_log_target": True,
        }
        run_id = save_run(
            product_key=product_key,
            request_payload=request_payload,
            result_payload=result,
        )
    except PricingServiceError as exc:
        return terminal_error(str(exc), "/bb/price", "BACK")
    except Exception:
        return terminal_error("pricing failed", "/bb/price", "BACK")

    return RedirectResponse(f"/bb/result/{run_id}", status_code=303)


@router.get("/result/{run_id}", response_class=HTMLResponse)
def blackberry_result(run_id: str):
    run = get_run(run_id)
    if run is None:
        return terminal_error("run not found")

    if run.get("run_type") == "scenario":
        base_run_id = (
            run.get("parent_run_id")
            or (run.get("request_payload") or {}).get("base_run_id")
            or ""
        )
        return blackberry_scenario_result(run["run_id"], run["result_payload"], base_run_id)

    if run.get("run_type") != "price":
        return terminal_error("unsupported run type")

    result = run["result_payload"]
    product_name = result.get("product_key", run["product_key"]).upper()
    latency = result.get("latency_ms")
    latency_text = "N/A" if latency is None else f"{latency}ms"
    body = f"""
<div class="head"><pre>RUN: {escape(compact_run_id(run["run_id"]))}</pre></div>
<pre>{escape(product_name)}
Price: {format_number(result.get("price"))}
MC: {format_number(result.get("mc_price"))}
Err: {format_number(result.get("abs_error"))}
Latency: {escape(latency_text)}
Model: {escape(str(result.get("model", "N/A")))}
Paths: {escape(str(result.get("n_paths", "N/A")))}</pre>
<div class="status">
<div><a href="/bb/scenario/{escape(run["run_id"])}">[1] SCENARIO SHOCK</a></div>
<div><a href="/bb/price">[2] NEW PRICE</a></div>
<div><a href="/bb/recent-runs">[3] RECENT RUNS</a></div>
<div><a href="/bb/model-status">[4] MODEL STATUS</a></div>
<div><a href="/bb">[5] HOME</a></div>
</div>
"""
    return terminal_page("ML-Pricer Result", body)


@router.get("/scenario/{run_id}", response_class=HTMLResponse)
def blackberry_scenario_form(run_id: str):
    run = get_run(run_id)
    if run is None:
        return terminal_error("base run not found")
    if run.get("run_type") != "price":
        return terminal_error("base run must be a price run")

    request_payload = run.get("request_payload") or {}
    result_payload = run.get("result_payload") or {}
    if not request_payload:
        return terminal_error("base request missing", f"/bb/result/{escape(run_id)}", "BACK")
    if request_payload.get("product_key") != "phoenix":
        return terminal_error("unsupported product", f"/bb/result/{escape(run_id)}", "BACK")

    fields = "\n".join(
        f'<div><label for="{escape(name)}">{escape(label)}:</label>'
        f'<input id="{escape(name)}" name="{escape(name)}" value="{escape(default)}"></div>'
        for name, label, default in SCENARIO_FORM_FIELDS
    )
    body = f"""
<div class="head"><pre>SCENARIO SHOCK</pre></div>
<pre>Base: {escape(compact_run_id(run_id))}
Product: PHOENIX
Price: {format_number(result_payload.get("price"))}</pre>
<form method="post" action="/bb/scenario/{escape(run_id)}">
{fields}
<button type="submit">PRICE SHOCK</button>
</form>
<div class="status">
<div><a href="/bb/result/{escape(run_id)}">[1] BACK</a></div>
<div><a href="/bb">[2] HOME</a></div>
</div>
"""
    return terminal_page("ML-Pricer Scenario Shock", body)


@router.post("/scenario/{run_id}", response_class=HTMLResponse)
async def blackberry_scenario_submit(run_id: str, request: Request):
    base_run = get_run(run_id)
    if base_run is None:
        return terminal_error("base run not found")
    if base_run.get("run_type") != "price":
        return terminal_error("base run must be a price run")

    form = await read_urlencoded_form(request)
    shocks = {name: form.get(name, "") for name, _, _ in SCENARIO_FORM_FIELDS}

    try:
        scenario_result = run_scenario(
            base_request=base_run.get("request_payload") or {},
            base_result=base_run.get("result_payload") or {},
            shocks=shocks,
        )
        scenario_request = {
            "product_key": scenario_result["product_key"],
            "base_run_id": run_id,
            "params": scenario_result["shocked_request"].get("params", {}),
            "n_paths": scenario_result["shocked_request"].get("n_paths"),
            "use_log_target": scenario_result["shocked_request"].get(
                "use_log_target", True
            ),
            "shocks": scenario_result["shocks"],
        }
        scenario_run_id = save_run(
            product_key=scenario_result["product_key"],
            request_payload=scenario_request,
            result_payload=scenario_result,
            run_type="scenario",
            parent_run_id=run_id,
        )
    except (ScenarioServiceError, PricingServiceError) as exc:
        return terminal_error(str(exc), f"/bb/scenario/{escape(run_id)}", "BACK")
    except Exception:
        return terminal_error("scenario failed", f"/bb/scenario/{escape(run_id)}", "BACK")

    return blackberry_scenario_result(scenario_run_id, scenario_result, run_id)


def blackberry_scenario_result(
    scenario_run_id: str, result: dict, base_run_id: str
) -> HTMLResponse:
    shocks = result.get("shocks", {})
    base_link = (
        f'<div><a href="/bb/result/{escape(base_run_id)}">[1] BASE RUN</a></div>'
        if base_run_id
        else ""
    )
    new_shock_link = (
        f'<div><a href="/bb/scenario/{escape(base_run_id)}">[2] NEW SHOCK</a></div>'
        if base_run_id
        else ""
    )
    body = f"""
<div class="head"><pre>SCENARIO RESULT</pre></div>
<pre>Run: {escape(compact_run_id(scenario_run_id))}
Base: {format_number(result.get("base_price"))}
Shock: {format_number(result.get("shocked_price"))}
Move: {format_number(result.get("price_change"))}
Move %: {format_percent(result.get("price_change_pct"))}

Shocks:
Spot: {format_shock(shocks.get("spot_pct"), "%")}
Vol: {format_shock(shocks.get("vol_abs"))}
Rate: {format_shock(shocks.get("rate_bps"), "bp")}

Summary:
{escape(str(result.get("summary", "N/A")))}</pre>
<div class="status">
{base_link}
{new_shock_link}
<div><a href="/bb/recent-runs">[3] RECENT RUNS</a></div>
<div><a href="/bb">[4] HOME</a></div>
</div>
"""
    return terminal_page("ML-Pricer Scenario Result", body)


@router.get("/recent-runs", response_class=HTMLResponse)
def blackberry_recent_runs():
    runs = list_recent_runs(limit=10)
    if not runs:
        body = """
<div class="head"><pre>NO RUNS YET</pre></div>
<div class="menu">
<div><a href="/bb/price">[1] PRICE NOTE</a></div>
<div><a href="/bb">[H] HOME</a></div>
</div>
"""
        return terminal_page("ML-Pricer Recent Runs", body)

    rows = []
    for index, run in enumerate(runs, start=1):
        run_type = (run.get("run_type") or "price").upper()
        product = (run.get("product_key") or "N/A").upper()
        short_id = compact_run_id(run.get("run_id", ""))
        timestamp = compact_timestamp(run.get("created_at", ""))
        price = run_price(run)
        rows.append(
            f'<div><a href="/bb/result/{escape(run["run_id"])}">[{index}] {escape(run_type)} {escape(product)} {escape(short_id)}</a></div>'
        )
        detail = f"    {price}  {timestamp}"
        if run.get("run_type") == "scenario" and run.get("parent_run_id"):
            detail += f"  base:{compact_run_id(run['parent_run_id'])}"
        rows.append(f"<pre>{escape(detail)}</pre>")

    body = f"""
<div class="head"><pre>RECENT RUNS</pre></div>
<div class="menu">
{''.join(rows)}
</div>
<div class="status">
<div><a href="/bb/price">[P] PRICE NOTE</a></div>
<div><a href="/bb">[H] HOME</a></div>
</div>
"""
    return terminal_page("ML-Pricer Recent Runs", body)


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
