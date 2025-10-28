from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, Any, Optional
from pathlib import Path
import traceback

from src.final.payoffs import (
    PhoenixPayoff,
    AccumulatorPayoff,
    BarrierOptionPayoff,
    DecumulatorPayoff,
)
from src.final.model_trainer import ModelTrainer
from src.final.evaluator import Evaluator

app = FastAPI(title="AI Derivative Pricer API", version="1.0")

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
    try:
        payoff_cls = PAYOFF_MAP.get(req.payoff_type.lower())
        if payoff_cls is None:
            return {"error": f"Unsupported payoff type: {req.payoff_type}"}

        payoff = payoff_cls()
        model_path = Path(
            f"C:\\Users\\aisha\\OneDrive\\Desktop\\GitHub\\neural-pricer\\final\\results\\{req.payoff_type.lower()}\\model.joblib"
        )
        scaler_path = Path(
            f"C:\\Users\\aisha\\OneDrive\\Desktop\\GitHub\\neural-pricer\\final\\results\\{req.payoff_type.lower()}\\scaler.joblib"
        )

        model, scaler = ModelTrainer.load(model_path, scaler_path)

        evaluator = Evaluator(payoff)
        result = evaluator.evaluate_case(
            params=req.params,
            model=model,
            scaler=scaler,
            n_paths_list=[req.n_paths],
            use_log_target=req.use_log_target,
        )
        return {"status": "success", "result": result}

    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "trace": traceback.format_exc(),
        }


@app.get("/")
def root():
    return {"message": "AI Derivative Pricer API is live!"}
