from datetime import datetime
from typing import Any, Dict, Literal

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from src.final.barrier_reverse_convertible import (
    BARRIER_REVERSE_CONVERTIBLE_V1,
    BarrierReverseConvertibleV1Contract,
    BarrierReverseConvertibleValidationError,
)
from src.final.market import (
    EQUITY_MARKET_SNAPSHOT_VERSION,
    EQUITY_MARKET_TERM_STRUCTURE_VERSION,
    EQUITY_RESEARCH_MARKET_VERSION,
    EquityMarketSegment,
    EquityMarketSnapshot,
    EquityMarketTermStructure,
    MarketDataValidationError,
)
from src.final.phoenix_contract import (
    PHOENIX_SINGLE_V2_CONTRACT_VERSION,
    PHOENIX_SINGLE_V3_CONTRACT_VERSION,
    PhoenixContractValidationError,
    PhoenixSingleV2Contract,
    PhoenixSingleV3Contract,
)
from app.services.diagnostics_service import (
    InvalidDiagnosticsInputError,
    get_barrier_reverse_convertible_diagnostics,
    get_phoenix_v1_diagnostics,
    get_phoenix_v2_diagnostics,
    get_phoenix_v3_diagnostics,
)
from app.services.live_market_data import (
    LiveMarketDataError,
    MarketDataConfigurationError,
    MarketDataInvalidResponseError,
    MarketDataNotFoundError,
    MarketDataRateLimitError,
    MarketDataRequestError,
    MarketDataStaleError,
    MarketDataTypeMismatchError,
    MarketDataUpstreamError,
    get_live_market_data_service,
    get_live_market_data_status,
)
from app.services.market_snapshot_store import (
    MarketSnapshotStoreError,
    get_research_market_snapshot,
    list_research_market_snapshots,
    save_research_market_snapshot,
)
from app.services.pricing_service import (
    InvalidPricingInputError,
    PricingServiceError,
    UnsupportedProductError,
    price_barrier_reverse_convertible_with_term_structure,
    price_phoenix_v2_with_term_structure,
    price_phoenix_v3_with_term_structure,
    price_phoenix_with_term_structure,
    price_phoenix_with_market_snapshot,
    price_product,
)
from app.services.product_registry import get_model_info, list_products
from app.services.research_market_data import (
    get_research_market_data_service,
    get_research_market_data_status,
)
from app.services.risk_service import (
    InvalidRiskInputError,
    RiskAnalyticsError,
    calculate_phoenix_term_structure_risk,
    run_phoenix_term_structure_scenario,
)
from app.services.run_store import get_run, list_recent_runs, save_run
from app.services.surrogate_service import (
    get_expanded_surrogate_evidence,
    get_surrogate_audit_evidence,
    get_surrogate_status,
)
from app.services.surrogate_monitoring import (
    SurrogateMonitoringError,
    get_surrogate_monitoring_series,
    get_surrogate_monitoring_status,
    get_surrogate_monitoring_summary,
)
from app.services.expanded_shadow_monitoring import (
    ExpandedShadowMonitoringError,
    get_expanded_shadow_readiness,
    get_expanded_shadow_series,
    get_expanded_shadow_summary,
    replay_expanded_shadow_observations,
)
from app.services.expanded_shadow_service import get_expanded_shadow_status
from app.services.surrogate_promotion import get_surrogate_promotion_readiness


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


class EquityMarketSegmentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    end_time_years: float = Field(gt=0.0, le=50.0)
    risk_free_rate: float = Field(ge=-0.25, le=1.0)
    dividend_yield: float = Field(ge=-0.25, le=1.0)
    volatility: float = Field(gt=0.0, le=5.0)

    def to_domain(self) -> EquityMarketSegment:
        return EquityMarketSegment(**self.model_dump())


class EquityMarketTermStructureRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["equity-market-term-structure-v1"] = (
        EQUITY_MARKET_TERM_STRUCTURE_VERSION
    )
    symbol: str = Field(min_length=1, max_length=64)
    underlier_type: Literal["equity", "etf", "index"]
    currency: str = Field(min_length=3, max_length=3)
    valuation_time: datetime
    market_data_time: datetime
    spot: float = Field(gt=0.0, le=1_000_000_000.0)
    segments: list[EquityMarketSegmentRequest] = Field(min_length=1, max_length=252)
    calendar: str = Field(min_length=1, max_length=32)
    day_count: str = Field(min_length=1, max_length=16)
    source: str = Field(min_length=1, max_length=128)
    term_structure_id: str | None = Field(
        default=None,
        min_length=71,
        max_length=71,
        pattern=r"^sha256:[0-9a-f]{64}$",
    )
    age_seconds: float | None = Field(default=None, ge=0.0)
    max_time_years: float | None = Field(default=None, gt=0.0, le=50.0)

    def to_domain(self) -> EquityMarketTermStructure:
        market = EquityMarketTermStructure(
            symbol=self.symbol,
            underlier_type=self.underlier_type,
            currency=self.currency,
            valuation_time=self.valuation_time,
            market_data_time=self.market_data_time,
            spot=self.spot,
            segments=tuple(segment.to_domain() for segment in self.segments),
            calendar=self.calendar,
            day_count=self.day_count,
            source=self.source,
        )
        if (
            self.term_structure_id is not None
            and self.term_structure_id != market.term_structure_id
        ):
            raise MarketDataValidationError(
                "term_structure_id does not match the supplied market inputs"
            )
        if (
            self.age_seconds is not None
            and abs(self.age_seconds - market.age_seconds) > 1e-6
        ):
            raise MarketDataValidationError(
                "age_seconds does not match the supplied market timestamps"
            )
        if (
            self.max_time_years is not None
            and abs(self.max_time_years - market.max_time_years) > 1e-12
        ):
            raise MarketDataValidationError(
                "max_time_years does not match the final market segment"
            )
        return market


class PhoenixSingleV1TermStructurePricingRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    market: EquityMarketTermStructureRequest
    terms: PhoenixSingleV1TermsRequest
    n_paths: int = Field(default=2000, ge=1, le=20_000)


class PhoenixSingleV2ContractRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    contract_version: Literal["phoenix-single-v2"] = PHOENIX_SINGLE_V2_CONTRACT_VERSION
    reference_level: float = Field(gt=0.0, le=1_000_000_000.0)
    maturity_years: float = Field(gt=0.0, le=30.0)
    observation_times_years: list[float] = Field(min_length=1, max_length=252)
    autocall_barrier_frac: float = Field(gt=0.0, le=3.0)
    coupon_barrier_frac: float = Field(gt=0.0, le=3.0)
    coupon_rate: float = Field(ge=0.0, le=1.0)
    knock_in_frac: float = Field(gt=0.0, le=1.0)
    prior_knock_in_breached: bool

    def to_domain(self) -> PhoenixSingleV2Contract:
        payload = self.model_dump()
        payload["observation_times_years"] = tuple(self.observation_times_years)
        return PhoenixSingleV2Contract(**payload)


class PhoenixSingleV2TermStructurePricingRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    market: EquityMarketTermStructureRequest
    contract: PhoenixSingleV2ContractRequest
    n_paths: int = Field(default=2000, ge=1, le=20_000)


class PhoenixSingleV3ContractRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    contract_version: Literal["phoenix-single-v3"] = PHOENIX_SINGLE_V3_CONTRACT_VERSION
    reference_level: float = Field(gt=0.0, le=1_000_000_000.0)
    maturity_years: float = Field(gt=0.0, le=30.0)
    observation_times_years: list[float] = Field(min_length=1, max_length=252)
    autocall_barrier_fracs: list[float] = Field(min_length=1, max_length=252)
    coupon_barrier_frac: float = Field(gt=0.0, le=3.0)
    coupon_rate: float = Field(ge=0.0, le=1.0)
    knock_in_frac: float = Field(gt=0.0, le=1.0)
    prior_knock_in_breached: bool
    memory_coupon: bool
    unpaid_coupon_count: int = Field(default=0, ge=0, le=252)

    def to_domain(self) -> PhoenixSingleV3Contract:
        payload = self.model_dump()
        payload["observation_times_years"] = tuple(self.observation_times_years)
        payload["autocall_barrier_fracs"] = tuple(self.autocall_barrier_fracs)
        return PhoenixSingleV3Contract(**payload)


class PhoenixSingleV3TermStructurePricingRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    market: EquityMarketTermStructureRequest
    contract: PhoenixSingleV3ContractRequest
    n_paths: int = Field(default=2000, ge=1, le=20_000)


class BarrierReverseConvertibleV1ContractRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    contract_version: Literal["barrier-reverse-convertible-v1"] = (
        BARRIER_REVERSE_CONVERTIBLE_V1
    )
    reference_level: float = Field(gt=0.0, le=1_000_000_000.0)
    maturity_years: float = Field(gt=0.0, le=30.0)
    coupon_times_years: list[float] = Field(min_length=1, max_length=252)
    coupon_rate_per_period: float = Field(ge=0.0, le=1.0)
    strike_frac: float = Field(gt=0.0, le=3.0)
    knock_in_frac: float = Field(gt=0.0, le=1.0)
    prior_knock_in_breached: bool = False

    def to_domain(self) -> BarrierReverseConvertibleV1Contract:
        payload = self.model_dump()
        payload["coupon_times_years"] = tuple(self.coupon_times_years)
        return BarrierReverseConvertibleV1Contract(**payload)


class BarrierReverseConvertiblePricingRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    market: EquityMarketTermStructureRequest
    contract: BarrierReverseConvertibleV1ContractRequest
    n_paths: int = Field(default=2000, ge=1, le=20_000)


class PhoenixV1DiagnosticsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    market: EquityMarketTermStructureRequest
    terms: PhoenixSingleV1TermsRequest
    n_paths: int = Field(default=2000, ge=100, le=5_000)
    seed: int = Field(default=42, ge=0, le=4_294_967_295)
    convergence_path_counts: list[int] = Field(default_factory=list, max_length=8)
    spot_shocks_pct: list[float] = Field(default_factory=list, max_length=11)
    volatility_shocks_abs: list[float] = Field(default_factory=list, max_length=11)


class PhoenixV2DiagnosticsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    market: EquityMarketTermStructureRequest
    contract: PhoenixSingleV2ContractRequest
    n_paths: int = Field(default=2000, ge=100, le=5_000)
    seed: int = Field(default=42, ge=0, le=4_294_967_295)
    convergence_path_counts: list[int] = Field(default_factory=list, max_length=8)
    spot_shocks_pct: list[float] = Field(default_factory=list, max_length=11)
    volatility_shocks_abs: list[float] = Field(default_factory=list, max_length=11)


class PhoenixV3DiagnosticsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    market: EquityMarketTermStructureRequest
    contract: PhoenixSingleV3ContractRequest
    n_paths: int = Field(default=2000, ge=100, le=5_000)
    seed: int = Field(default=42, ge=0, le=4_294_967_295)
    convergence_path_counts: list[int] = Field(default_factory=list, max_length=8)
    spot_shocks_pct: list[float] = Field(default_factory=list, max_length=11)
    volatility_shocks_abs: list[float] = Field(default_factory=list, max_length=11)


class BarrierReverseConvertibleDiagnosticsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    market: EquityMarketTermStructureRequest
    contract: BarrierReverseConvertibleV1ContractRequest
    n_paths: int = Field(default=2000, ge=100, le=5_000)
    seed: int = Field(default=42, ge=0, le=4_294_967_295)
    convergence_path_counts: list[int] = Field(default_factory=list, max_length=8)
    spot_shocks_pct: list[float] = Field(default_factory=list, max_length=11)
    volatility_shocks_abs: list[float] = Field(default_factory=list, max_length=11)


class SourcedEquityMarketRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    symbol: str = Field(min_length=1, max_length=64)
    underlier_type: Literal["equity", "etf", "index"]
    risk_free_rate: float = Field(ge=-0.25, le=1.0)
    dividend_yield: float = Field(ge=-0.25, le=1.0)
    volatility: float = Field(gt=0.0, le=5.0)
    day_count: str = Field(default="ACT/365F", min_length=1, max_length=16)


class SourcedPhoenixSingleV1PricingRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    market: SourcedEquityMarketRequest
    terms: PhoenixSingleV1TermsRequest
    n_paths: int = Field(default=2000, ge=1, le=20_000)


class ResearchEquityMarketRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    symbol: str = Field(min_length=1, max_length=64)
    underlier_type: Literal["equity", "etf"]
    currency: Literal["USD"] = "USD"


class ResearchTermStructureBuildRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    market: ResearchEquityMarketRequest
    maturity_years: float = Field(gt=0.0, le=30.0)


class ResearchPhoenixSingleV1PricingRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    market: ResearchEquityMarketRequest
    terms: PhoenixSingleV1TermsRequest
    n_paths: int = Field(default=2000, ge=1, le=20_000)


class EquityMarketSegmentShockRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    segment_index: int = Field(ge=0, le=251)
    rate_bps: float | None = Field(default=None, ge=-10_000.0, le=10_000.0)
    dividend_bps: float | None = Field(default=None, ge=-10_000.0, le=10_000.0)
    volatility_abs: float | None = Field(default=None, ge=-5.0, le=5.0)


class EquityMarketShockRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    spot_pct: float | None = Field(default=None, gt=-100.0, le=1_000.0)
    rate_parallel_bps: float | None = Field(default=None, ge=-10_000.0, le=10_000.0)
    dividend_parallel_bps: float | None = Field(default=None, ge=-10_000.0, le=10_000.0)
    volatility_parallel_abs: float | None = Field(default=None, ge=-5.0, le=5.0)
    segment_shocks: list[EquityMarketSegmentShockRequest] = Field(
        default_factory=list, max_length=252
    )

    def to_service(self) -> dict[str, Any]:
        return self.model_dump(exclude_none=True)


class PhoenixTermStructureScenarioRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    market: EquityMarketTermStructureRequest
    terms: PhoenixSingleV1TermsRequest
    shock: EquityMarketShockRequest
    n_paths: int = Field(default=2000, ge=1, le=20_000)
    seed: int = Field(default=42, ge=0, le=4_294_967_295)


class ResearchPhoenixScenarioRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    market: ResearchEquityMarketRequest
    terms: PhoenixSingleV1TermsRequest
    shock: EquityMarketShockRequest
    n_paths: int = Field(default=2000, ge=1, le=20_000)
    seed: int = Field(default=42, ge=0, le=4_294_967_295)


class RiskBumpRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    spot_relative: float = Field(default=0.01, gt=0.0, le=0.5)
    volatility_absolute: float = Field(default=0.01, gt=0.0, le=1.0)
    rate_bps: float = Field(default=10.0, gt=0.0, le=5_000.0)
    dividend_bps: float = Field(default=10.0, gt=0.0, le=5_000.0)


class PhoenixTermStructureRiskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    market: EquityMarketTermStructureRequest
    terms: PhoenixSingleV1TermsRequest
    bumps: RiskBumpRequest = Field(default_factory=RiskBumpRequest)
    n_paths: int = Field(default=2000, ge=1, le=20_000)
    seed: int = Field(default=42, ge=0, le=4_294_967_295)


class ResearchPhoenixRiskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    market: ResearchEquityMarketRequest
    terms: PhoenixSingleV1TermsRequest
    bumps: RiskBumpRequest = Field(default_factory=RiskBumpRequest)
    n_paths: int = Field(default=2000, ge=1, le=20_000)
    seed: int = Field(default=42, ge=0, le=4_294_967_295)


def _market_data_error_response(exc: LiveMarketDataError) -> JSONResponse:
    if isinstance(exc, MarketDataNotFoundError):
        status_code = 404
    elif isinstance(exc, (MarketDataRequestError, MarketDataTypeMismatchError)):
        status_code = 422
    elif isinstance(
        exc,
        (
            MarketDataConfigurationError,
            MarketDataRateLimitError,
            MarketDataStaleError,
        ),
    ):
        status_code = 503
    elif isinstance(exc, (MarketDataInvalidResponseError, MarketDataUpstreamError)):
        status_code = 502
    else:
        status_code = 503
    return JSONResponse(
        {"status": "error", "message": str(exc)}, status_code=status_code
    )


def _risk_error_response(exc: RiskAnalyticsError) -> JSONResponse:
    status_code = 422 if isinstance(exc, InvalidRiskInputError) else 503
    return JSONResponse(
        {"status": "error", "message": str(exc)}, status_code=status_code
    )


def _persist_research_market(built: Any) -> dict[str, Any]:
    return save_research_market_snapshot(
        market=built.market.to_dict(),
        calibration=built.calibration,
    )


def _save_analysis_run(
    *,
    request_payload: dict[str, Any],
    result: dict[str, Any],
    run_type: Literal["scenario", "risk"],
) -> str:
    return save_run(
        product_key="phoenix",
        request_payload=request_payload,
        result_payload=result,
        run_type=run_type,
    )


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


@router.post("/products/phoenix/price/term-structure")
def price_phoenix_term_structure(
    req: PhoenixSingleV1TermStructurePricingRequest,
):
    try:
        result = price_phoenix_with_term_structure(
            market=req.market.to_domain(),
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
            {"status": "error", "message": "term-structure pricing failed"},
            status_code=500,
        )


@router.post("/products/phoenix/price/seasoned/term-structure")
def price_phoenix_v2_term_structure(
    req: PhoenixSingleV2TermStructurePricingRequest,
):
    try:
        result = price_phoenix_v2_with_term_structure(
            market=req.market.to_domain(),
            contract=req.contract.to_domain(),
            n_paths=req.n_paths,
        )
        return {"status": "success", "result": result}
    except (
        InvalidPricingInputError,
        MarketDataValidationError,
        PhoenixContractValidationError,
    ) as exc:
        return JSONResponse({"status": "error", "message": str(exc)}, status_code=422)
    except UnsupportedProductError as exc:
        return JSONResponse({"status": "error", "message": str(exc)}, status_code=400)
    except PricingServiceError as exc:
        return JSONResponse({"status": "error", "message": str(exc)}, status_code=503)
    except Exception:
        return JSONResponse(
            {"status": "error", "message": "seasoned pricing failed"},
            status_code=500,
        )


@router.post("/products/phoenix/price/richer/term-structure")
def price_phoenix_v3_term_structure(
    req: PhoenixSingleV3TermStructurePricingRequest,
):
    try:
        result = price_phoenix_v3_with_term_structure(
            market=req.market.to_domain(),
            contract=req.contract.to_domain(),
            n_paths=req.n_paths,
        )
        return {"status": "success", "result": result}
    except (
        InvalidPricingInputError,
        MarketDataValidationError,
        PhoenixContractValidationError,
    ) as exc:
        return JSONResponse({"status": "error", "message": str(exc)}, status_code=422)
    except UnsupportedProductError as exc:
        return JSONResponse({"status": "error", "message": str(exc)}, status_code=400)
    except PricingServiceError as exc:
        return JSONResponse({"status": "error", "message": str(exc)}, status_code=503)
    except Exception:
        return JSONResponse(
            {"status": "error", "message": "richer Phoenix pricing failed"},
            status_code=500,
        )


@router.post("/products/phoenix/diagnostics/term-structure")
def phoenix_v1_diagnostics(req: PhoenixV1DiagnosticsRequest):
    try:
        diagnostics = get_phoenix_v1_diagnostics(
            market=req.market.to_domain(),
            terms=req.terms.model_dump(),
            n_paths=req.n_paths,
            seed=req.seed,
            convergence_path_counts=req.convergence_path_counts,
            spot_shocks_pct=req.spot_shocks_pct,
            volatility_shocks_abs=req.volatility_shocks_abs,
        )
        return {"status": "success", "diagnostics": diagnostics}
    except (InvalidDiagnosticsInputError, MarketDataValidationError) as exc:
        return JSONResponse({"status": "error", "message": str(exc)}, status_code=422)
    except Exception:
        return JSONResponse(
            {"status": "error", "message": "diagnostics failed"},
            status_code=500,
        )


@router.post("/products/phoenix/diagnostics/seasoned/term-structure")
def phoenix_v2_diagnostics(req: PhoenixV2DiagnosticsRequest):
    try:
        diagnostics = get_phoenix_v2_diagnostics(
            market=req.market.to_domain(),
            contract=req.contract.to_domain(),
            n_paths=req.n_paths,
            seed=req.seed,
            convergence_path_counts=req.convergence_path_counts,
            spot_shocks_pct=req.spot_shocks_pct,
            volatility_shocks_abs=req.volatility_shocks_abs,
        )
        return {"status": "success", "diagnostics": diagnostics}
    except (
        InvalidDiagnosticsInputError,
        MarketDataValidationError,
        PhoenixContractValidationError,
    ) as exc:
        return JSONResponse({"status": "error", "message": str(exc)}, status_code=422)
    except Exception:
        return JSONResponse(
            {"status": "error", "message": "seasoned diagnostics failed"},
            status_code=500,
        )


@router.post("/products/phoenix/diagnostics/richer/term-structure")
def phoenix_v3_diagnostics(req: PhoenixV3DiagnosticsRequest):
    try:
        diagnostics = get_phoenix_v3_diagnostics(
            market=req.market.to_domain(),
            contract=req.contract.to_domain(),
            n_paths=req.n_paths,
            seed=req.seed,
            convergence_path_counts=req.convergence_path_counts,
            spot_shocks_pct=req.spot_shocks_pct,
            volatility_shocks_abs=req.volatility_shocks_abs,
        )
        return {"status": "success", "diagnostics": diagnostics}
    except (
        InvalidDiagnosticsInputError,
        MarketDataValidationError,
        PhoenixContractValidationError,
    ) as exc:
        return JSONResponse({"status": "error", "message": str(exc)}, status_code=422)
    except Exception:
        return JSONResponse(
            {"status": "error", "message": "richer Phoenix diagnostics failed"},
            status_code=500,
        )


@router.post("/products/barrier-reverse-convertible/price/term-structure")
def price_barrier_reverse_convertible(
    req: BarrierReverseConvertiblePricingRequest,
):
    try:
        result = price_barrier_reverse_convertible_with_term_structure(
            market=req.market.to_domain(),
            contract=req.contract.to_domain(),
            n_paths=req.n_paths,
        )
        return {"status": "success", "result": result}
    except (
        InvalidPricingInputError,
        MarketDataValidationError,
        BarrierReverseConvertibleValidationError,
    ) as exc:
        return JSONResponse({"status": "error", "message": str(exc)}, status_code=422)
    except (UnsupportedProductError, PricingServiceError) as exc:
        return JSONResponse({"status": "error", "message": str(exc)}, status_code=503)
    except Exception:
        return JSONResponse(
            {"status": "error", "message": "reverse convertible pricing failed"},
            status_code=500,
        )


@router.post("/products/barrier-reverse-convertible/diagnostics/term-structure")
def barrier_reverse_convertible_diagnostics(
    req: BarrierReverseConvertibleDiagnosticsRequest,
):
    try:
        diagnostics = get_barrier_reverse_convertible_diagnostics(
            market=req.market.to_domain(),
            contract=req.contract.to_domain(),
            n_paths=req.n_paths,
            seed=req.seed,
            convergence_path_counts=req.convergence_path_counts,
            spot_shocks_pct=req.spot_shocks_pct,
            volatility_shocks_abs=req.volatility_shocks_abs,
        )
        return {"status": "success", "diagnostics": diagnostics}
    except (
        InvalidDiagnosticsInputError,
        MarketDataValidationError,
        BarrierReverseConvertibleValidationError,
    ) as exc:
        return JSONResponse({"status": "error", "message": str(exc)}, status_code=422)
    except Exception:
        return JSONResponse(
            {"status": "error", "message": "reverse convertible diagnostics failed"},
            status_code=500,
        )


@router.get("/market-data/status")
def market_data_status():
    return {
        "status": "success",
        "market_data": get_live_market_data_status(),
        "research_market": get_research_market_data_status(),
    }


@router.get("/market-data/quote")
def market_data_quote(
    symbol: str = Query(min_length=1, max_length=64),
):
    try:
        quote_result, quote_age_seconds = get_live_market_data_service().get_quote(
            symbol=symbol
        )
        return {
            "status": "success",
            "quote": {
                **quote_result.quote.to_dict(),
                "cache_hit": quote_result.cache_hit,
                "quote_age_seconds": quote_age_seconds,
            },
        }
    except LiveMarketDataError as exc:
        return _market_data_error_response(exc)
    except Exception:
        return JSONResponse(
            {"status": "error", "message": "market data request failed"},
            status_code=500,
        )


@router.post("/products/phoenix/price/market")
def price_phoenix_market(req: SourcedPhoenixSingleV1PricingRequest):
    try:
        market = req.market
        sourced_snapshot = get_live_market_data_service().get_snapshot(
            symbol=market.symbol,
            underlier_type=market.underlier_type,
            risk_free_rate=market.risk_free_rate,
            dividend_yield=market.dividend_yield,
            volatility=market.volatility,
            day_count=market.day_count,
        )
        result = price_phoenix_with_market_snapshot(
            market_snapshot=sourced_snapshot.snapshot,
            terms=req.terms.model_dump(),
            n_paths=req.n_paths,
        )
        result["market_data"] = sourced_snapshot.metadata()
        return {"status": "success", "result": result}
    except LiveMarketDataError as exc:
        return _market_data_error_response(exc)
    except (InvalidPricingInputError, MarketDataValidationError) as exc:
        return JSONResponse({"status": "error", "message": str(exc)}, status_code=422)
    except PricingServiceError as exc:
        return JSONResponse({"status": "error", "message": str(exc)}, status_code=503)
    except Exception:
        return JSONResponse(
            {"status": "error", "message": "market-data pricing failed"},
            status_code=500,
        )


@router.post("/market-data/research-term-structure")
def build_research_term_structure(req: ResearchTermStructureBuildRequest):
    try:
        built = get_research_market_data_service().build_term_structure(
            symbol=req.market.symbol,
            underlier_type=req.market.underlier_type,
            maturity_years=req.maturity_years,
        )
        snapshot = _persist_research_market(built)
        return {
            "status": "success",
            "market_term_structure": built.market.to_dict(),
            "market_calibration": built.calibration,
            "market_snapshot": snapshot,
        }
    except LiveMarketDataError as exc:
        return _market_data_error_response(exc)
    except MarketDataValidationError as exc:
        return JSONResponse({"status": "error", "message": str(exc)}, status_code=422)
    except MarketSnapshotStoreError:
        return JSONResponse(
            {"status": "error", "message": "research market could not be frozen"},
            status_code=503,
        )
    except Exception:
        return JSONResponse(
            {"status": "error", "message": "research market build failed"},
            status_code=500,
        )


@router.post("/products/phoenix/price/research-market")
def price_phoenix_research_market(req: ResearchPhoenixSingleV1PricingRequest):
    try:
        built = get_research_market_data_service().build_term_structure(
            symbol=req.market.symbol,
            underlier_type=req.market.underlier_type,
            maturity_years=req.terms.maturity_years,
        )
        snapshot = _persist_research_market(built)
        result = price_phoenix_with_term_structure(
            market=built.market,
            terms=req.terms.model_dump(),
            n_paths=req.n_paths,
        )
        result["market_calibration"] = built.calibration
        result["market_calibration_version"] = EQUITY_RESEARCH_MARKET_VERSION
        result["market_snapshot_record"] = snapshot
        return {"status": "success", "result": result}
    except LiveMarketDataError as exc:
        return _market_data_error_response(exc)
    except (InvalidPricingInputError, MarketDataValidationError) as exc:
        return JSONResponse({"status": "error", "message": str(exc)}, status_code=422)
    except MarketSnapshotStoreError:
        return JSONResponse(
            {"status": "error", "message": "research market could not be frozen"},
            status_code=503,
        )
    except PricingServiceError as exc:
        return JSONResponse({"status": "error", "message": str(exc)}, status_code=503)
    except Exception:
        return JSONResponse(
            {"status": "error", "message": "research market pricing failed"},
            status_code=500,
        )


@router.get("/market-data/research-snapshots")
def recent_research_market_snapshots(
    limit: int = Query(default=20, ge=1, le=100),
):
    try:
        return {
            "status": "success",
            "snapshots": list_research_market_snapshots(limit=limit),
        }
    except MarketSnapshotStoreError:
        return JSONResponse(
            {"status": "error", "message": "market snapshot store is unavailable"},
            status_code=503,
        )


@router.get("/market-data/research-snapshots/{snapshot_id}")
def research_market_snapshot(snapshot_id: str):
    try:
        snapshot = get_research_market_snapshot(snapshot_id)
    except MarketSnapshotStoreError:
        return JSONResponse(
            {"status": "error", "message": "market snapshot store is unavailable"},
            status_code=503,
        )
    if snapshot is None:
        return JSONResponse(
            {"status": "error", "message": "market snapshot was not found"},
            status_code=404,
        )
    return {"status": "success", "snapshot": snapshot}


@router.post("/products/phoenix/scenario/term-structure")
def scenario_phoenix_term_structure(req: PhoenixTermStructureScenarioRequest):
    try:
        result = run_phoenix_term_structure_scenario(
            market=req.market.to_domain(),
            terms=req.terms.model_dump(),
            shock=req.shock.to_service(),
            n_paths=req.n_paths,
            seed=req.seed,
        )
        run_id = _save_analysis_run(
            request_payload=req.model_dump(mode="json", exclude_none=True),
            result=result,
            run_type="scenario",
        )
        return {"status": "success", "run_id": run_id, "result": result}
    except RiskAnalyticsError as exc:
        return _risk_error_response(exc)
    except MarketDataValidationError as exc:
        return JSONResponse({"status": "error", "message": str(exc)}, status_code=422)
    except Exception:
        return JSONResponse(
            {"status": "error", "message": "scenario calculation failed"},
            status_code=500,
        )


@router.post("/products/phoenix/scenario/research-market")
def scenario_phoenix_research_market(req: ResearchPhoenixScenarioRequest):
    try:
        built = get_research_market_data_service().build_term_structure(
            symbol=req.market.symbol,
            underlier_type=req.market.underlier_type,
            maturity_years=req.terms.maturity_years,
        )
        result = run_phoenix_term_structure_scenario(
            market=built.market,
            terms=req.terms.model_dump(),
            shock=req.shock.to_service(),
            n_paths=req.n_paths,
            seed=req.seed,
            market_calibration=built.calibration,
        )
        request_payload = req.model_dump(mode="json", exclude_none=True)
        request_payload["frozen_market"] = built.market.to_dict()
        request_payload["market_calibration_id"] = built.calibration.get(
            "calibration_id"
        )
        run_id = _save_analysis_run(
            request_payload=request_payload,
            result=result,
            run_type="scenario",
        )
        return {"status": "success", "run_id": run_id, "result": result}
    except LiveMarketDataError as exc:
        return _market_data_error_response(exc)
    except RiskAnalyticsError as exc:
        return _risk_error_response(exc)
    except MarketDataValidationError as exc:
        return JSONResponse({"status": "error", "message": str(exc)}, status_code=422)
    except Exception:
        return JSONResponse(
            {"status": "error", "message": "research scenario calculation failed"},
            status_code=500,
        )


@router.post("/products/phoenix/risk/term-structure")
def risk_phoenix_term_structure(req: PhoenixTermStructureRiskRequest):
    try:
        result = calculate_phoenix_term_structure_risk(
            market=req.market.to_domain(),
            terms=req.terms.model_dump(),
            bumps=req.bumps.model_dump(),
            n_paths=req.n_paths,
            seed=req.seed,
        )
        run_id = _save_analysis_run(
            request_payload=req.model_dump(mode="json"),
            result=result,
            run_type="risk",
        )
        return {"status": "success", "run_id": run_id, "result": result}
    except RiskAnalyticsError as exc:
        return _risk_error_response(exc)
    except MarketDataValidationError as exc:
        return JSONResponse({"status": "error", "message": str(exc)}, status_code=422)
    except Exception:
        return JSONResponse(
            {"status": "error", "message": "risk calculation failed"},
            status_code=500,
        )


@router.post("/products/phoenix/risk/research-market")
def risk_phoenix_research_market(req: ResearchPhoenixRiskRequest):
    try:
        built = get_research_market_data_service().build_term_structure(
            symbol=req.market.symbol,
            underlier_type=req.market.underlier_type,
            maturity_years=req.terms.maturity_years,
        )
        result = calculate_phoenix_term_structure_risk(
            market=built.market,
            terms=req.terms.model_dump(),
            bumps=req.bumps.model_dump(),
            n_paths=req.n_paths,
            seed=req.seed,
            market_calibration=built.calibration,
        )
        request_payload = req.model_dump(mode="json")
        request_payload["frozen_market"] = built.market.to_dict()
        request_payload["market_calibration_id"] = built.calibration.get(
            "calibration_id"
        )
        run_id = _save_analysis_run(
            request_payload=request_payload,
            result=result,
            run_type="risk",
        )
        return {"status": "success", "run_id": run_id, "result": result}
    except LiveMarketDataError as exc:
        return _market_data_error_response(exc)
    except RiskAnalyticsError as exc:
        return _risk_error_response(exc)
    except MarketDataValidationError as exc:
        return JSONResponse({"status": "error", "message": str(exc)}, status_code=422)
    except Exception:
        return JSONResponse(
            {"status": "error", "message": "research risk calculation failed"},
            status_code=500,
        )


@router.get("/runs")
def recent_analysis_runs(limit: int = Query(default=20, ge=1, le=100)):
    try:
        return {"status": "success", "runs": list_recent_runs(limit=limit)}
    except Exception:
        return JSONResponse(
            {"status": "error", "message": "run history unavailable"},
            status_code=500,
        )


@router.get("/runs/{run_id}")
def analysis_run(run_id: str):
    try:
        stored = get_run(run_id)
        if stored is None:
            return JSONResponse(
                {"status": "error", "message": "run not found"}, status_code=404
            )
        return {"status": "success", "run": stored}
    except Exception:
        return JSONResponse(
            {"status": "error", "message": "run history unavailable"},
            status_code=500,
        )


@router.get("/products")
def products():
    return {
        "status": "success",
        "products": list_products(),
    }


@router.get("/model-info")
def model_info():
    info = get_model_info()
    info["surrogate_shadow"] = get_surrogate_status()
    info["surrogate_monitoring"] = get_surrogate_monitoring_status()
    info["expanded_surrogate_shadow"] = get_expanded_shadow_status()
    return {
        "status": "success",
        "model_info": info,
    }


@router.get("/surrogate-shadow/metrics")
def surrogate_shadow_metrics(limit: int = Query(default=1000, ge=1, le=100_000)):
    try:
        return {
            "status": "success",
            "monitoring": get_surrogate_monitoring_summary(limit=limit),
        }
    except SurrogateMonitoringError:
        return JSONResponse(
            {"status": "error", "message": "surrogate monitoring unavailable"},
            status_code=503,
        )


@router.get("/surrogate-shadow/promotion-readiness")
def surrogate_shadow_promotion_readiness(
    limit: int = Query(default=100_000, ge=1, le=100_000),
):
    try:
        return {
            "status": "success",
            "readiness": get_surrogate_promotion_readiness(limit=limit),
        }
    except SurrogateMonitoringError:
        return JSONResponse(
            {
                "status": "error",
                "message": "surrogate promotion readiness unavailable",
            },
            status_code=503,
        )


@router.get("/surrogate-shadow/evidence")
def surrogate_shadow_evidence(
    monitoring_limit: int = Query(default=5_000, ge=1, le=100_000),
    series_limit: int = Query(default=250, ge=1, le=5_000),
):
    """Combine frozen audit and live shadow evidence without changing runtime."""
    try:
        return {
            "status": "success",
            "evidence": {
                "audit": get_surrogate_audit_evidence(),
                "expansion_experiments": get_expanded_surrogate_evidence(),
                "monitoring": get_surrogate_monitoring_summary(limit=monitoring_limit),
                "series": get_surrogate_monitoring_series(limit=series_limit),
                "readiness": get_surrogate_promotion_readiness(limit=monitoring_limit),
                "expanded_shadow": {
                    "runtime": get_expanded_shadow_status(),
                    "monitoring": get_expanded_shadow_summary(limit=monitoring_limit),
                    "series": get_expanded_shadow_series(limit=series_limit),
                    "readiness": get_expanded_shadow_readiness(limit=monitoring_limit),
                },
            },
        }
    except (SurrogateMonitoringError, ExpandedShadowMonitoringError):
        return JSONResponse(
            {"status": "error", "message": "surrogate evidence unavailable"},
            status_code=503,
        )


@router.get("/expanded-surrogate-shadow/status")
def expanded_surrogate_shadow_status():
    return {"status": "success", "runtime": get_expanded_shadow_status()}


@router.get("/expanded-surrogate-shadow/metrics")
def expanded_surrogate_shadow_metrics(
    limit: int = Query(default=5_000, ge=1, le=100_000),
):
    try:
        return {
            "status": "success",
            "monitoring": get_expanded_shadow_summary(limit=limit),
        }
    except ExpandedShadowMonitoringError:
        return JSONResponse(
            {"status": "error", "message": "expanded shadow monitoring unavailable"},
            status_code=503,
        )


@router.get("/expanded-surrogate-shadow/promotion-readiness")
def expanded_surrogate_shadow_promotion_readiness(
    limit: int = Query(default=100_000, ge=1, le=100_000),
):
    try:
        return {
            "status": "success",
            "readiness": get_expanded_shadow_readiness(limit=limit),
        }
    except ExpandedShadowMonitoringError:
        return JSONResponse(
            {"status": "error", "message": "expanded shadow readiness unavailable"},
            status_code=503,
        )


@router.post("/expanded-surrogate-shadow/replay/{product_key}")
def replay_expanded_surrogate_shadow(
    product_key: Literal["phoenix_v3", "barrier_reverse_convertible"],
    limit: int = Query(default=100, ge=1, le=1_000),
):
    try:
        return {
            "status": "success",
            "replay": replay_expanded_shadow_observations(product_key, limit=limit),
        }
    except ExpandedShadowMonitoringError:
        return JSONResponse(
            {"status": "error", "message": "expanded shadow replay unavailable"},
            status_code=503,
        )
