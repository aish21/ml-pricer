from fastapi import FastAPI
from typing import Dict, Any
from pathlib import Path
import os
import json
from fastapi.responses import JSONResponse
import csv
from datetime import datetime, timezone

from src.final.market import (
    EQUITY_GBM_FLAT_MODEL_VERSION,
    EQUITY_GBM_PIECEWISE_MODEL_VERSION,
    EQUITY_MARKET_SNAPSHOT_VERSION,
    EQUITY_MARKET_TERM_STRUCTURE_VERSION,
)
from app.api.bb import router as bb_api_router
from app.api.v1 import PricingRequest, execute_pricing_request
from app.api.v1 import router as api_v1_router
from app.bb.routes import router as blackberry_router
from app.services.pricing_service import (
    InvalidPricingInputError,
    PricingServiceError,
    UnsupportedProductError,
)
from app.services.live_market_data import get_live_market_data_status
from app.services.product_registry import (
    REPO_ROOT,
    build_artifact_status,
    get_product_definition,
    get_results_dir,
)

app = FastAPI(title="Neural Pricer API", version="0.4.0")
app.include_router(bb_api_router)
app.include_router(api_v1_router)
app.include_router(blackberry_router)

BASE_RESULTS_DIR = get_results_dir()
# By default write history to a container-writable location. In Docker we mount ./data -> /srv/app/data
HISTORY_FILE = Path(
    os.getenv("MODEL_HISTORY_FILE", str(REPO_ROOT / "data" / "pricing_history.csv"))
)

# Ensure history directory exists
HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)


@app.post("/price/")
def price_instrument(req: PricingRequest):
    """
    Price a single instrument using the saved model + MC baseline.
    Returns {"status": "success", "result": ...} on success.
    """
    try:
        result = execute_pricing_request(req)

        # Also append to server-side history CSV (best-effort, non-blocking)
        try:
            row = {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "payoff_type": req.payoff_type,
                "n_paths": req.n_paths,
                "model_price": "",
                "mc_price": result.get("price"),
                "abs_error": "",
                "rel_error": "",
                "model_time_s": "",
                "mc_time_s": result.get("mc_time_s"),
            }
            append_history(HISTORY_FILE, row)
        except Exception:
            # don't fail pricing if history append fails
            pass

        return {"status": "success", "result": result}
    except UnsupportedProductError as exc:
        return JSONResponse({"status": "error", "message": str(exc)}, status_code=400)
    except InvalidPricingInputError as exc:
        return JSONResponse({"status": "error", "message": str(exc)}, status_code=422)
    except PricingServiceError as exc:
        return JSONResponse({"status": "error", "message": str(exc)}, status_code=503)
    except Exception:
        return JSONResponse(
            {"status": "error", "message": "pricing failed"},
            status_code=500,
        )


def append_history(path: Path, row: Dict[str, Any]):
    """Append a single row (dict) into CSV in a stable way."""
    header = [
        "timestamp_utc",
        "payoff_type",
        "n_paths",
        "model_price",
        "mc_price",
        "abs_error",
        "rel_error",
        "model_time_s",
        "mc_time_s",
    ]
    exists = path.exists()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            if not exists:
                writer.writeheader()
            # ensure only header keys
            row_clean = {k: row.get(k, "") for k in header}
            writer.writerow(row_clean)
    except Exception:
        # swallow errors (best-effort)
        pass


@app.post("/history/append")
def history_append(payload: Dict[str, Any]):
    """Endpoint to append frontend-sent history rows to server-side CSV file."""
    try:
        append_history(HISTORY_FILE, payload)
        return JSONResponse(
            {"status": "success", "message": "Appended"}, status_code=201
        )
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


@app.get("/history")
def get_history():
    """Return server-side history CSV as JSON array (best-effort)."""
    try:
        if not HISTORY_FILE.exists():
            return {"status": "success", "history": []}
        rows = []
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                rows.append(r)
        return {"status": "success", "history": rows}
    except Exception:
        return JSONResponse(
            {"status": "error", "message": "history unavailable"},
            status_code=500,
        )


@app.get("/training/{payoff_type}")
def get_training_info(payoff_type: str):
    """
    Return results.json stored with the trained model for the requested payoff_type.
    """
    try:
        payoff_key = payoff_type.lower()
        product = get_product_definition(payoff_key)
        if product is None:
            return JSONResponse(
                {"status": "error", "message": f"Unknown product: {payoff_type}"},
                status_code=404,
            )
        artifact_status = build_artifact_status(product, BASE_RESULTS_DIR)
        if (
            not product.validated_for_pricing
            or not artifact_status["ready_for_surrogate"]
        ):
            return JSONResponse(
                {
                    "status": "error",
                    "message": "No compatible validated training artifact is available",
                },
                status_code=409,
            )
        results_path = BASE_RESULTS_DIR / payoff_key / "results.json"
        if not results_path.exists():
            return JSONResponse(
                {"status": "error", "message": f"No results.json at {results_path}"},
                status_code=404,
            )
        data = json.loads(results_path.read_text())
        return {"status": "success", "training": data.get("training", data)}
    except Exception:
        return JSONResponse(
            {"status": "error", "message": "training metadata unavailable"},
            status_code=500,
        )


@app.get("/payoff_explanation/{payoff_type}")
def payoff_explanation(payoff_type: str):
    """
    Return a small explanation for the selected payoff type.
    """
    try:
        t = payoff_type.lower()
        if t == "phoenix":
            payload = {
                "title": "Phoenix (Autocallable) payoff",
                "summary": (
                    "Phoenix Single v1 pays a non-memory coupon on each observation date when the underlier is at or above "
                    "the coupon barrier. It redeems principal early on the first observation at or above the autocall barrier. "
                    "If it never autocalls, principal is protected unless the knock-in barrier was touched and the final level "
                    "is below the initial level; in that case redemption is proportional to final performance."
                ),
                "latex": r"""
    	ext{Informal rules (not math):}
    \begin{itemize}
      \item Pay a coupon at each observation above the coupon barrier.
      \item Redeem principal and terminate at the first observation above the autocall barrier.
      \item Maturity loss occurs only if the knock-in was touched and S_T is below S_0.
    \end{itemize}
    """,
                "notes": [
                    "Very plain meanings of common inputs:",
                    "- S0: the starting stock price when you buy the product (think 'starting point').",
                    "- r: interest rate used to discount future money back to today (small number like 0.03 for 3%).",
                    "- sigma: volatility — how jumpy the stock is. Larger sigma means the stock moves around more.",
                    "- T: time until the product ends (years).",
                    "- autocall_barrier_frac: the barrier expressed as a multiple of S0; e.g. 1.05 means 105% of S0 (the stock needs to be 5% up).",
                    "- coupon_rate: coupon per observation as a fraction of notional.",
                    "- knock_in_frac: a lower barrier (as multiple of S0); if the stock goes below this at any time it 'knocks in' and can change the final payout.",
                    "- obs_count: number of evenly spaced observation dates.",
                    "Simple example: S0=100, autocall at 105, coupon 2% — if at any check day the stock ≥105 you get ~2% and you're done early.",
                ],
            }
        elif t == "accumulator":
            payload = {
                "title": "Accumulator payoff",
                "summary": (
                    "Think of an accumulator like a simple rule that says: 'If the stock stays in a certain price range on a check day, "
                    "we pretend we bought the stock then at a small discount and keep doing this over many check days. At the end, "
                    "the payoff is basically the average of the things we accumulated (discounted back to today)."
                    "It's easiest to imagine it as a repeated buy-at-a-discount program that only activates when the price is inside the band."
                ),
                "latex": r"""
    	ext{Informal: average of discounted prices on observation days where the price stayed inside the band.}
    """,
                "notes": [
                    "- upper_barrier_frac / lower_barrier_frac: define the price band (multiples of S0).",
                    "- participation_rate: how much you participate in the stock return (a higher number can mean a larger effective exposure).",
                    "- obs_frequency: how often observations/checks happen (e.g., every 0.25 years).",
                    "Simple example: if S0=100 and the band is [95,105], then on any check day the price between 95 and 105 causes an accumulation event (we count that price into the average).",
                ],
            }
        elif t == "barrier":
            payload = {
                "title": "Down-and-out barrier option (simple)",
                "summary": (
                    "A barrier option is like a normal option (you have the right to buy or sell a stock at a fixed price K at the end), "
                    "but with a twist: if the stock ever touches a special barrier level during the life of the option, the option "
                    "can become worthless (it 'knocks out'). So you only get the usual option payoff at the end if the barrier was never hit."
                ),
                "latex": r"""
    	ext{Informal rules:}
    \begin{itemize}
      \item If the barrier is hit at any time before expiry, payoff = 0 (the option 'dies').
      \item Otherwise, at maturity a call pays max(S_T - K, 0) and a put pays max(K - S_T, 0).
    \end{itemize}
    """,
                "notes": [
                    "- K: the strike — the price at which you can buy (call) or sell (put) at maturity.",
                    "- barrier_frac: barrier level as multiple of S0 (e.g., 0.8 means 80% of the starting price).",
                    "- option type: 'call' = right to buy, 'put' = right to sell.",
                    "- If you are new: think of the barrier as a safety check — touch it and the option disappears.",
                ],
            }
        elif t == "decumulator":
            payload = {
                "title": "Decumulator (opposite of accumulator)",
                "summary": (
                    "A decumulator is the flip side of the accumulator. Instead of acting when the price is inside a band, "
                    "it acts when the price is outside the band. You can think of it as a rule that 'sells' or realizes exposure "
                    "when the price moves outside a comfortable range. The final payoff aggregates those events (again averaged/discounted)."
                ),
                "latex": r"""
    	ext{Informal: average of discounted events where the price was outside the allowed band.}
    """,
                "notes": [
                    "- upper_barrier_frac / lower_barrier_frac: define the band; decumulator triggers when price is outside this range.",
                    "- participation_rate: scales how strongly each triggered event contributes to payoff.",
                    "Simple example: if band is [95,105] and price is 110 on an observation day, the decumulator counts that day into the payoff (you 'sell' or realize value).",
                ],
            }
        else:
            return JSONResponse(
                {"status": "error", "message": f"No explanation for '{payoff_type}'"},
                status_code=404,
            )
        return {"status": "success", "explanation": payload}
    except Exception:
        return JSONResponse(
            {"status": "error", "message": "payoff explanation unavailable"},
            status_code=500,
        )


@app.get("/")
def root():
    return {
        "service": "neural-pricer",
        "status": "online",
        "version": app.version,
        "docs": "/docs",
    }


@app.get("/health/live", include_in_schema=False)
def health_live():
    return {"status": "alive"}


@app.get("/health/ready", include_in_schema=False)
def health_ready():
    product = get_product_definition("phoenix")
    return {
        "status": "ready",
        "pricing_method": "monte_carlo_reference",
        "contract_version": product.contract_version if product else "unavailable",
        "market_snapshot_version": EQUITY_MARKET_SNAPSHOT_VERSION,
        "market_term_structure_version": EQUITY_MARKET_TERM_STRUCTURE_VERSION,
        "market_model_version": EQUITY_GBM_FLAT_MODEL_VERSION,
        "market_model_versions": [
            EQUITY_GBM_FLAT_MODEL_VERSION,
            EQUITY_GBM_PIECEWISE_MODEL_VERSION,
        ],
        "market_data": get_live_market_data_status(),
    }
