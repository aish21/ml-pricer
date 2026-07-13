from typing import Any, Dict

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from app.services.pricing_service import (
    InvalidPricingInputError,
    PricingServiceError,
    UnsupportedProductError,
    price_product,
)
from app.services.product_registry import get_model_info, list_products


router = APIRouter(prefix="/api/v1", tags=["api-v1"])


class PricingRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    payoff_type: str
    params: Dict[str, Any]
    n_paths: int = Field(default=2000, ge=1, le=20_000)


def execute_pricing_request(req: PricingRequest) -> dict[str, Any]:
    return price_product(
        product_key=req.payoff_type,
        params=req.params,
        n_paths=req.n_paths,
    )


@router.post("/price")
def price(req: PricingRequest):
    try:
        return {"status": "success", "result": execute_pricing_request(req)}
    except UnsupportedProductError as exc:
        return JSONResponse({"status": "error", "message": str(exc)}, status_code=400)
    except InvalidPricingInputError as exc:
        return JSONResponse({"status": "error", "message": str(exc)}, status_code=422)
    except PricingServiceError as exc:
        return JSONResponse({"status": "error", "message": str(exc)}, status_code=503)
    except Exception:
        return JSONResponse(
            {"status": "error", "message": "pricing failed"}, status_code=500
        )


@router.get("/products")
def products():
    return {
        "status": "success",
        "products": list_products(),
    }


@router.get("/model-info")
def model_info():
    return {
        "status": "success",
        "model_info": get_model_info(),
    }
