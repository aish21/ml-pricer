# app/backend.py
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
                "summary": "Phoenix is an autocallable structured product that either pays a coupon if the underlying is above an autocall barrier on observation dates (early redemption), or pays redemption at maturity depending on knock-in breaches.",
                "latex": r""" \text{Payoff}_i = \begin{cases} (1 + \text{coupon}) e^{-r t_{\text{call}}}, & \exists\ \text{obs s.t. } S_t \ge B_{\text{autocall}}\\ \left(\frac{S_T}{S_0}\right) e^{-r T}, & \text{if knocked-in and not autocall}\\ (1 + \text{coupon}) e^{-r T}, & \text{otherwise} \end{cases} """,
                "notes": [
                    "Key inputs: S0, r, sigma, T, autocall barrier (×S0), coupon rate, knock-in (×S0), obs_count.",
                    "Monte Carlo simulates many paths and applies the same rule per path; model predicts mean price.",
                ],
            }
        elif t == "accumulator":
            payload = {
                "title": "Accumulator payoff",
                "summary": "Accumulator accumulates (buys) when the underlying stays inside a band; payoff is average of accumulated discounted prices.",
                "latex": r"\text{Payoff} \approx \frac{1}{n_{\text{obs}}}\sum_{t\in\text{obs}} \mathbf{1}_{L < S_t < U}\cdot \frac{S_t}{1+\text{participation}}\cdot e^{-rT}",
                "notes": [
                    "Key inputs: S0, r, sigma, T, upper/lower barrier (×S0), participation rate, obs_frequency."
                ],
            }
        elif t == "barrier":
            payload = {
                "title": "Down-and-out Barrier option",
                "summary": "A barrier option becomes worthless if the underlying hits the barrier; otherwise it's a vanilla option payoff (call/put) at maturity.",
                "latex": r"\text{Payoff} = \begin{cases}0 & \text{if hit barrier}\\ \max(S_T - K, 0) & \text{call}\\ \max(K - S_T, 0) & \text{put}\end{cases}",
                "notes": [
                    "Key inputs: S0, K, barrier_frac (×S0), sigma, T, r, option type."
                ],
            }
        elif t == "decumulator":
            payload = {
                "title": "Decumulator payoff",
                "summary": "Decumulator sells when price goes outside the band — mirror of an accumulator.",
                "latex": r"\text{Payoff} \approx \frac{1}{n_{\text{obs}}}\sum_{t\in\text{obs}} \mathbf{1}_{S_t\not\in(L,U)}\cdot S_t(1+\text{participation}) e^{-rT}",
                "notes": [
                    "Key inputs: S0, r, sigma, T, upper/lower barrier (×S0), participation rate, obs_frequency."
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
