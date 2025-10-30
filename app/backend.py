from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, Any, Optional
from pathlib import Path
import os
import traceback
import json
from fastapi.responses import JSONResponse

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


class PricingRequest(BaseModel):
    payoff_type: str
    params: Dict[str, Any]
    n_paths: Optional[int] = 2000
    use_log_target: Optional[bool] = True


@app.post("/price/")
def price_instrument(req: PricingRequest):
    """
    Price a single instrument using the saved model + MC baseline.
    Returns a JSON object with 'status' and either 'result' (on success)
    or 'message'/'trace' (on error).
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

        # Allow overriding results location via env var for containerized deployments
        base_results = os.getenv(
            "MODEL_RESULTS_DIR",
            r"C:\Users\aisha\OneDrive\Desktop\GitHub\neural-pricer\final\results",
        )
        model_path = Path(base_results) / req.payoff_type.lower() / "model.joblib"
        scaler_path = Path(base_results) / req.payoff_type.lower() / "scaler.joblib"

        if not model_path.exists() or not scaler_path.exists():
            return JSONResponse(
                {
                    "status": "error",
                    "message": f"Model or scaler not found for payoff '{req.payoff_type}'. "
                    f"Expected at: {model_path} and {scaler_path}",
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

        return {"status": "success", "result": result}

    except Exception as e:
        return JSONResponse(
            {
                "status": "error",
                "message": str(e),
                "trace": traceback.format_exc(),
            },
            status_code=500,
        )


@app.get("/training/{payoff_type}")
def get_training_info(payoff_type: str):
    """
    Return results.json stored with the trained model for the requested payoff_type.
    Frontend will call this when feature_importance isn't present in /price/ response.
    """
    try:
        payoff_key = payoff_type.lower()
        base_results = os.getenv(
            "MODEL_RESULTS_DIR",
            r"C:\Users\aisha\OneDrive\Desktop\GitHub\neural-pricer\final\results",
        )
        results_path = Path(base_results) / payoff_key / "results.json"
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
    Return a small explanation for the selected payoff type including:
    - title
    - short description
    - simple LaTeX formula or pseudocode
    - (optional) placeholder illustration url
    """
    try:
        t = payoff_type.lower()
        if t == "phoenix":
            payload = {
                "title": "Phoenix (Autocallable) payoff",
                "summary": "Phoenix is an autocallable structured product that either pays a coupon "
                "if the underlying is above an autocall barrier on observation dates (early redemption), "
                "or pays redemption at maturity depending on knock-in breaches.",
                "latex": r"""
\text{Payoff}_i =
\begin{cases}
(1 + \text{coupon}) e^{-r t_{\text{call}}}, & \exists\ \text{obs s.t. } S_t \ge B_{\text{autocall}}\\
\left(\frac{S_T}{S_0}\right) e^{-r T}, & \text{if knocked-in at some time and not autocall}\\
(1 + \text{coupon}) e^{-r T}, & \text{otherwise}
\end{cases}
""",
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
                    "Key inputs: S0, r, sigma, T, upper/lower barrier (×S0), participation rate, obs_frequency.",
                    "Be careful: payoff values can be larger than 1 (depends on participation and contract terms).",
                ],
            }
        elif t == "barrier":
            payload = {
                "title": "Down-and-out Barrier option",
                "summary": "A barrier option becomes worthless if the underlying hits the barrier; otherwise it's a vanilla option payoff (call/put) at maturity.",
                "latex": r"\text{Payoff} = \begin{cases}0 & \text{if hit barrier}\\ \max(S_T - K, 0) & \text{call otherwise}\\ \max(K - S_T, 0) & \text{put otherwise}\end{cases}",
                "notes": [
                    "Key inputs: S0, K, barrier_frac (×S0), sigma, T, r, option type."
                ],
            }
        elif t == "decumulator":
            payload = {
                "title": "Decumulator payoff",
                "summary": "Decumulator sells when price goes outside the band — this is the mirror of an accumulator.",
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
