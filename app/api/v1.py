from datetime import datetime
from typing import Any, Dict, Literal

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from src.final.market import (
    EQUITY_MARKET_SNAPSHOT_VERSION,
    EquityMarketSnapshot,
    MarketDataValidationError,
)
from app.services.pricing_service import (
    InvalidPricingInputError,
    PricingServiceError,
    UnsupportedProductError,
    price_phoenix_with_market_snapshot,
    price_product,
)
from app.services.product_registry import get_model_info, list_products


router = APIRouter(prefix="/api/v1", tags=["api-v1"])


class PricingRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    payoff_type: str
    params: Dict[str, Any]
    n_paths: int = Field(default=2000, ge=1, le=20_000)


class EquityMarketSnapshotRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["equity-market-snapshot-v1"] = (
        EQUITY_MARKET_SNAPSHOT_VERSION
    )
    symbol: str = Field(min_length=1, max_length=64)
    underlier_type: Literal["equity", "etf", "index"]
    currency: str = Field(min_length=3, max_length=3)
    valuation_time: datetime
    market_data_time: datetime
    spot: float = Field(gt=0.0, le=1_000_000_000.0)
    risk_free_rate: float = Field(ge=-0.25, le=1.0)
    dividend_yield: float = Field(ge=-0.25, le=1.0)
    volatility: float = Field(gt=0.0, le=5.0)
    calendar: str = Field(default="WEEKDAYS", min_length=1, max_length=32)
    day_count: str = Field(default="ACT/365F", min_length=1, max_length=16)
    source: str = Field(default="request", min_length=1, max_length=128)

    def to_domain(self) -> EquityMarketSnapshot:
        return EquityMarketSnapshot(
            symbol=self.symbol,
            underlier_type=self.underlier_type,
            currency=self.currency,
            valuation_time=self.valuation_time,
            market_data_time=self.market_data_time,
            spot=self.spot,
            risk_free_rate=self.risk_free_rate,
            dividend_yield=self.dividend_yield,
            volatility=self.volatility,
            calendar=self.calendar,
            day_count=self.day_count,
            source=self.source,
        )


class PhoenixSingleV1TermsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    maturity_years: float = Field(gt=0.0, le=30.0)
    autocall_barrier_frac: float = Field(gt=0.0, le=3.0)
    coupon_barrier_frac: float = Field(gt=0.0, le=3.0)
    coupon_rate: float = Field(ge=0.0, le=1.0)
    knock_in_frac: float = Field(gt=0.0, le=1.0)
    obs_count: int = Field(ge=1, le=252)


class PhoenixSingleV1PricingRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    market: EquityMarketSnapshotRequest
    terms: PhoenixSingleV1TermsRequest
    n_paths: int = Field(default=2000, ge=1, le=20_000)


def execute_pricing_request(req: PricingRequest) -> dict[str, Any]:
    return price_product(
        product_key=req.payoff_type,
        params=req.params,
        n_paths=req.n_paths,
    )


@router.post("/price", deprecated=True)
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


@router.post("/products/phoenix/price")
def price_phoenix(req: PhoenixSingleV1PricingRequest):
    try:
        result = price_phoenix_with_market_snapshot(
            market_snapshot=req.market.to_domain(),
            terms=req.terms.model_dump(),
            n_paths=req.n_paths,
        )
        return {"status": "success", "result": result}
    except (InvalidPricingInputError, MarketDataValidationError) as exc:
        return JSONResponse({"status": "error", "message": str(exc)}, status_code=422)
    except UnsupportedProductError as exc:
        return JSONResponse({"status": "error", "message": str(exc)}, status_code=400)
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
