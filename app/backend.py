from fastapi import FastAPI
from fastapi import Request
from pydantic import BaseModel
from typing import Dict, Any, Optional
from pathlib import Path
import os
import traceback
import json
from fastapi.responses import JSONResponse
import csv
from datetime import datetime

from src.final.payoffs import (
    PhoenixPayoff,
    AccumulatorPayoff,
    BarrierOptionPayoff,
    DecumulatorPayoff,
)
from src.final.model_trainer import ModelTrainer
from src.final.evaluator import Evaluator

app = FastAPI(title="ML Pricer API", version="1.0")

PAYOFF_MAP = {
    "phoenix": PhoenixPayoff,
    "accumulator": AccumulatorPayoff,
    "barrier": BarrierOptionPayoff,
    "decumulator": DecumulatorPayoff,
}

# Environment-driven locations
BASE_RESULTS_DIR = Path(
    os.getenv(
        "MODEL_RESULTS_DIR",
        r"C:\Users\aisha\OneDrive\Desktop\GitHub\neural-pricer\final\results",
    )
)
# By default write history to a container-writable location. In Docker we mount ./data -> /srv/app/data
HISTORY_FILE = Path(
    os.getenv("MODEL_HISTORY_FILE", "/srv/app/data/pricing_history.csv")
)

# Ensure history directory exists
HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)


class PricingRequest(BaseModel):
    payoff_type: str
    params: Dict[str, Any]
    n_paths: Optional[int] = 2000
    use_log_target: Optional[bool] = True


@app.post("/price/")
def price_instrument(req: PricingRequest):
    """
    Price a single instrument using the saved model + MC baseline.
    Returns {"status": "success", "result": ...} on success.
    """
    try:
        payoff_cls = PAYOFF_MAP.get(req.payoff_type.lower())
        if payoff_cls is None:
            return JSONResponse(
                {
                    "status": "error",
                    "message": f"Unsupported payoff type: {req.payoff_type}",
                },
                status_code=400,
            )

        payoff = payoff_cls()
        model_path = BASE_RESULTS_DIR / req.payoff_type.lower() / "model.joblib"
        scaler_path = BASE_RESULTS_DIR / req.payoff_type.lower() / "scaler.joblib"

        if not model_path.exists() or not scaler_path.exists():
            return JSONResponse(
                {
                    "status": "error",
                    "message": f"Model or scaler not found for payoff '{req.payoff_type}'. Expected at: {model_path} and {scaler_path}",
                },
                status_code=404,
            )

        model, scaler = ModelTrainer.load(model_path, scaler_path)

        evaluator = Evaluator(payoff)
        result = evaluator.evaluate_case(
            params=req.params,
            model=model,
            scaler=scaler,
            n_paths_list=[int(req.n_paths or 2000)],
            use_log_target=bool(req.use_log_target),
        )

        # Also append to server-side history CSV (best-effort, non-blocking)
        try:
            # pick the per-npaths result we just computed
            per_npaths = result.get("per_npaths") or find_first_per_npaths(result)
            # get the key (n_paths as string)
            key = (
                str(int(req.n_paths)) if req.n_paths else next(iter(per_npaths.keys()))
            )
            entry = (
                per_npaths.get(key)
                if per_npaths and key in per_npaths
                else (next(iter(per_npaths.values())) if per_npaths else None)
            )
            model_price = entry.get("Model", {}).get("price") if entry else None
            mc_price = entry.get("MC", {}).get("price") if entry else None
            model_time = entry.get("Model", {}).get("time") if entry else None
            mc_time = entry.get("MC", {}).get("time") if entry else None
            abs_err = entry.get("Model", {}).get("abs_error") if entry else None
            rel_err = entry.get("Model", {}).get("rel_error") if entry else None

            row = {
                "timestamp_utc": datetime.utcnow().isoformat(),
                "payoff_type": req.payoff_type,
                "n_paths": req.n_paths,
                "model_price": model_price,
                "mc_price": mc_price,
                "abs_error": abs_err,
                "rel_error": rel_err,
                "model_time_s": model_time,
                "mc_time_s": mc_time,
            }
            append_history(HISTORY_FILE, row)
        except Exception:
            # don't fail pricing if history append fails
            pass

        return {"status": "success", "result": result}
    except Exception as e:
        return JSONResponse(
            {"status": "error", "message": str(e), "trace": traceback.format_exc()},
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


def find_first_per_npaths(obj: Dict[str, Any]):
    """Fallback to locate per_npaths in complex result objects (used server-side)."""
    if not obj:
        return None
    if isinstance(obj, dict):
        if "per_npaths" in obj and isinstance(obj["per_npaths"], dict):
            return obj["per_npaths"]
        for v in obj.values():
            found = find_first_per_npaths(v) if isinstance(v, dict) else None
            if found:
                return found
    return None


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
    except Exception as e:
        return JSONResponse(
            {"status": "error", "message": str(e), "trace": traceback.format_exc()},
            status_code=500,
        )


@app.get("/training/{payoff_type}")
def get_training_info(payoff_type: str):
    """
    Return results.json stored with the trained model for the requested payoff_type.
    """
    try:
        payoff_key = payoff_type.lower()
        results_path = BASE_RESULTS_DIR / payoff_key / "results.json"
        if not results_path.exists():
            return JSONResponse(
                {"status": "error", "message": f"No results.json at {results_path}"},
                status_code=404,
            )
        data = json.loads(results_path.read_text())
        return {"status": "success", "training": data.get("training", data)}
    except Exception as e:
        return JSONResponse(
            {"status": "error", "message": str(e), "trace": traceback.format_exc()},
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
                    "Imagine a simple money product you buy that promises to pay you a small extra amount (a 'coupon') "
                    "if the stock does okay on certain check days. If the stock is high enough on any check day, the product "
                    "ends early and pays you the coupon right away (this is called an 'autocall'). If it never ends early, "
                    "what you get at the final date depends on whether the stock ever fell below a certain safety level ('knock-in'). "
                    "If it did fall below that safety level at some point and the product didn't end early, your final payment "
                    "might be based on how the stock did relative to where it started (you could get less than your original money). "
                    "If it never fell below the safety level and didn't end early, you usually get back your money plus the coupon at the end."
                ),
                "latex": r"""
    	ext{Informal rules (not math):}
    \begin{itemize}
      \item If the stock is above the autocall barrier on any observation day, the product stops and pays a coupon (early).
      \item Else, if the stock ever breached the knock-in level during the life, the final payoff may be proportional to S_T/S_0 (could lose money).
      \item Else, you receive the coupon (and your capital) at maturity.
    \end{itemize}
    """,
                "notes": [
                    "Very plain meanings of common inputs:",
                    "- S0: the starting stock price when you buy the product (think 'starting point').",
                    "- r: interest rate used to discount future money back to today (small number like 0.03 for 3%).",
                    "- sigma: volatility — how jumpy the stock is. Larger sigma means the stock moves around more.",
                    "- T: time until the product ends (years).",
                    "- autocall_barrier_frac: the barrier expressed as a multiple of S0; e.g. 1.05 means 105% of S0 (the stock needs to be 5% up).",
                    "- coupon_rate: the extra percentage paid if the product autocalls or at maturity (if not knocked-in).",
                    "- knock_in_frac: a lower barrier (as multiple of S0); if the stock goes below this at any time it 'knocks in' and can change the final payout.",
                    "- obs_count: how many check days there are (more checks = more chances to autocall).",
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
    except Exception as e:
        return JSONResponse(
            {"status": "error", "message": str(e), "trace": traceback.format_exc()},
            status_code=500,
        )


@app.get("/")
def root():
    return {"message": "ML Pricer API is live!"}
