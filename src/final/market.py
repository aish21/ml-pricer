import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, ClassVar, Protocol


EQUITY_MARKET_SNAPSHOT_VERSION = "equity-market-snapshot-v1"
EQUITY_GBM_FLAT_MODEL_VERSION = "equity-gbm-flat-v2"
EQUITY_MARKET_TERM_STRUCTURE_VERSION = "equity-market-term-structure-v1"
EQUITY_GBM_PIECEWISE_MODEL_VERSION = "equity-gbm-piecewise-v1"
EQUITY_RESEARCH_MARKET_VERSION = "equity-research-market-v1"
EQUITY_MARKET_SCENARIO_VERSION = "equity-market-scenario-v1"
EQUITY_RISK_ANALYTICS_VERSION = "equity-risk-analytics-v1"
EQUITY_LIKE_UNDERLIER_TYPES = frozenset({"equity", "etf", "index"})


class MarketDataValidationError(ValueError):
    pass


@dataclass(frozen=True)
class EquityMarketSnapshot:
    """Immutable, dated inputs for one equity-like underlier."""

    schema_version: ClassVar[str] = EQUITY_MARKET_SNAPSHOT_VERSION

    symbol: str
    underlier_type: str
    currency: str
    valuation_time: datetime
    market_data_time: datetime
    spot: float
    risk_free_rate: float
    dividend_yield: float
    volatility: float
    calendar: str
    day_count: str
    source: str

    def __post_init__(self) -> None:
        symbol = self.symbol.strip()
        underlier_type = self.underlier_type.strip().lower()
        currency = self.currency.strip().upper()
        calendar = self.calendar.strip().upper()
        day_count = self.day_count.strip().upper()
        source = self.source.strip()

        if not symbol or len(symbol) > 64 or not symbol.isprintable():
            raise MarketDataValidationError(
                "symbol must contain 1 to 64 printable characters"
            )
        if underlier_type not in EQUITY_LIKE_UNDERLIER_TYPES:
            raise MarketDataValidationError(
                "underlier_type must be one of: equity, etf, index"
            )
        if len(currency) != 3 or not currency.isalpha() or not currency.isascii():
            raise MarketDataValidationError(
                "currency must be a three-letter ASCII code"
            )
        if not calendar or len(calendar) > 32:
            raise MarketDataValidationError("calendar must contain 1 to 32 characters")
        if not day_count or len(day_count) > 16:
            raise MarketDataValidationError("day_count must contain 1 to 16 characters")
        if not source or len(source) > 128 or not source.isprintable():
            raise MarketDataValidationError(
                "source must contain 1 to 128 printable characters"
            )

        for field_name, timestamp in (
            ("valuation_time", self.valuation_time),
            ("market_data_time", self.market_data_time),
        ):
            if not isinstance(timestamp, datetime):
                raise MarketDataValidationError(f"{field_name} must be a datetime")
            if timestamp.tzinfo is None or timestamp.utcoffset() is None:
                raise MarketDataValidationError(
                    f"{field_name} must include a UTC offset"
                )
        if self.market_data_time > self.valuation_time:
            raise MarketDataValidationError(
                "market_data_time cannot be after valuation_time"
            )
        object.__setattr__(
            self, "valuation_time", self.valuation_time.astimezone(timezone.utc)
        )
        object.__setattr__(
            self, "market_data_time", self.market_data_time.astimezone(timezone.utc)
        )

        numeric_limits = {
            "spot": (self.spot, 0.0, 1_000_000_000.0, False),
            "risk_free_rate": (self.risk_free_rate, -0.25, 1.0, True),
            "dividend_yield": (self.dividend_yield, -0.25, 1.0, True),
            "volatility": (self.volatility, 0.0, 5.0, False),
        }
        for field_name, (
            raw_value,
            lower,
            upper,
            lower_inclusive,
        ) in numeric_limits.items():
            if isinstance(raw_value, bool):
                raise MarketDataValidationError(f"{field_name} must be numeric")
            try:
                value = float(raw_value)
            except (TypeError, ValueError) as exc:
                raise MarketDataValidationError(
                    f"{field_name} must be numeric"
                ) from exc
            if not math.isfinite(value):
                raise MarketDataValidationError(f"{field_name} must be finite")
            lower_invalid = value < lower if lower_inclusive else value <= lower
            if lower_invalid or value > upper:
                lower_operator = ">=" if lower_inclusive else ">"
                raise MarketDataValidationError(
                    f"{field_name} must be {lower_operator} {lower} and <= {upper}"
                )
            object.__setattr__(self, field_name, value)

        object.__setattr__(self, "symbol", symbol)
        object.__setattr__(self, "underlier_type", underlier_type)
        object.__setattr__(self, "currency", currency)
        object.__setattr__(self, "calendar", calendar)
        object.__setattr__(self, "day_count", day_count)
        object.__setattr__(self, "source", source)

    def _canonical_payload(self) -> dict[str, Any]:
        valuation_time = self.valuation_time.isoformat().replace("+00:00", "Z")
        market_data_time = self.market_data_time.isoformat().replace("+00:00", "Z")
        return {
            "schema_version": self.schema_version,
            "symbol": self.symbol,
            "underlier_type": self.underlier_type,
            "currency": self.currency,
            "valuation_time": valuation_time,
            "market_data_time": market_data_time,
            "spot": self.spot,
            "risk_free_rate": self.risk_free_rate,
            "dividend_yield": self.dividend_yield,
            "volatility": self.volatility,
            "calendar": self.calendar,
            "day_count": self.day_count,
            "source": self.source,
        }

    @property
    def snapshot_id(self) -> str:
        encoded = json.dumps(
            self._canonical_payload(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return f"sha256:{hashlib.sha256(encoded).hexdigest()}"

    @property
    def age_seconds(self) -> float:
        return (self.valuation_time - self.market_data_time).total_seconds()

    def to_dict(self) -> dict[str, Any]:
        return {
            **self._canonical_payload(),
            "snapshot_id": self.snapshot_id,
            "age_seconds": self.age_seconds,
        }


@dataclass(frozen=True)
class EquityMarketSegment:
    """Piecewise-constant model inputs over the preceding tenor interval."""

    end_time_years: float
    risk_free_rate: float
    dividend_yield: float
    volatility: float

    def __post_init__(self) -> None:
        numeric_limits = {
            "end_time_years": (self.end_time_years, 0.0, 50.0, False),
            "risk_free_rate": (self.risk_free_rate, -0.25, 1.0, True),
            "dividend_yield": (self.dividend_yield, -0.25, 1.0, True),
            "volatility": (self.volatility, 0.0, 5.0, False),
        }
        for field_name, (
            raw_value,
            lower,
            upper,
            lower_inclusive,
        ) in numeric_limits.items():
            if isinstance(raw_value, bool):
                raise MarketDataValidationError(f"{field_name} must be numeric")
            try:
                value = float(raw_value)
            except (TypeError, ValueError) as exc:
                raise MarketDataValidationError(
                    f"{field_name} must be numeric"
                ) from exc
            if not math.isfinite(value):
                raise MarketDataValidationError(f"{field_name} must be finite")
            lower_invalid = value < lower if lower_inclusive else value <= lower
            if lower_invalid or value > upper:
                lower_operator = ">=" if lower_inclusive else ">"
                raise MarketDataValidationError(
                    f"{field_name} must be {lower_operator} {lower} and <= {upper}"
                )
            object.__setattr__(self, field_name, value)

    def to_dict(self) -> dict[str, float]:
        return {
            "end_time_years": self.end_time_years,
            "risk_free_rate": self.risk_free_rate,
            "dividend_yield": self.dividend_yield,
            "volatility": self.volatility,
        }


@dataclass(frozen=True)
class EquityMarketTermStructure:
    """Immutable piecewise market inputs for one equity-like underlier."""

    schema_version: ClassVar[str] = EQUITY_MARKET_TERM_STRUCTURE_VERSION

    symbol: str
    underlier_type: str
    currency: str
    valuation_time: datetime
    market_data_time: datetime
    spot: float
    segments: tuple[EquityMarketSegment, ...]
    calendar: str
    day_count: str
    source: str

    def __post_init__(self) -> None:
        symbol = self.symbol.strip()
        underlier_type = self.underlier_type.strip().lower()
        currency = self.currency.strip().upper()
        calendar = self.calendar.strip().upper()
        day_count = self.day_count.strip().upper()
        source = self.source.strip()

        if not symbol or len(symbol) > 64 or not symbol.isprintable():
            raise MarketDataValidationError(
                "symbol must contain 1 to 64 printable characters"
            )
        if underlier_type not in EQUITY_LIKE_UNDERLIER_TYPES:
            raise MarketDataValidationError(
                "underlier_type must be one of: equity, etf, index"
            )
        if len(currency) != 3 or not currency.isalpha() or not currency.isascii():
            raise MarketDataValidationError(
                "currency must be a three-letter ASCII code"
            )
        if not calendar or len(calendar) > 32:
            raise MarketDataValidationError("calendar must contain 1 to 32 characters")
        if not day_count or len(day_count) > 16:
            raise MarketDataValidationError("day_count must contain 1 to 16 characters")
        if not source or len(source) > 128 or not source.isprintable():
            raise MarketDataValidationError(
                "source must contain 1 to 128 printable characters"
            )

        for field_name, timestamp in (
            ("valuation_time", self.valuation_time),
            ("market_data_time", self.market_data_time),
        ):
            if not isinstance(timestamp, datetime):
                raise MarketDataValidationError(f"{field_name} must be a datetime")
            if timestamp.tzinfo is None or timestamp.utcoffset() is None:
                raise MarketDataValidationError(
                    f"{field_name} must include a UTC offset"
                )
        if self.market_data_time > self.valuation_time:
            raise MarketDataValidationError(
                "market_data_time cannot be after valuation_time"
            )

        if isinstance(self.spot, bool):
            raise MarketDataValidationError("spot must be numeric")
        try:
            spot = float(self.spot)
        except (TypeError, ValueError) as exc:
            raise MarketDataValidationError("spot must be numeric") from exc
        if not math.isfinite(spot):
            raise MarketDataValidationError("spot must be finite")
        if spot <= 0 or spot > 1_000_000_000.0:
            raise MarketDataValidationError("spot must be > 0.0 and <= 1000000000.0")

        try:
            segments = tuple(self.segments)
        except TypeError as exc:
            raise MarketDataValidationError("segments must be a sequence") from exc
        if not segments or len(segments) > 252:
            raise MarketDataValidationError(
                "segments must contain between 1 and 252 segments"
            )
        previous_end = 0.0
        for segment in segments:
            if not isinstance(segment, EquityMarketSegment):
                raise MarketDataValidationError(
                    "segments must contain EquityMarketSegment values"
                )
            if segment.end_time_years <= previous_end:
                raise MarketDataValidationError(
                    "segment end_time_years values must be strictly increasing"
                )
            previous_end = segment.end_time_years

        object.__setattr__(self, "symbol", symbol)
        object.__setattr__(self, "underlier_type", underlier_type)
        object.__setattr__(self, "currency", currency)
        object.__setattr__(
            self, "valuation_time", self.valuation_time.astimezone(timezone.utc)
        )
        object.__setattr__(
            self, "market_data_time", self.market_data_time.astimezone(timezone.utc)
        )
        object.__setattr__(self, "spot", spot)
        object.__setattr__(self, "segments", segments)
        object.__setattr__(self, "calendar", calendar)
        object.__setattr__(self, "day_count", day_count)
        object.__setattr__(self, "source", source)

    @property
    def max_time_years(self) -> float:
        return self.segments[-1].end_time_years

    @property
    def age_seconds(self) -> float:
        return (self.valuation_time - self.market_data_time).total_seconds()

    def _canonical_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "symbol": self.symbol,
            "underlier_type": self.underlier_type,
            "currency": self.currency,
            "valuation_time": self.valuation_time.isoformat().replace("+00:00", "Z"),
            "market_data_time": self.market_data_time.isoformat().replace(
                "+00:00", "Z"
            ),
            "spot": self.spot,
            "segments": [segment.to_dict() for segment in self.segments],
            "calendar": self.calendar,
            "day_count": self.day_count,
            "source": self.source,
        }

    @property
    def term_structure_id(self) -> str:
        encoded = json.dumps(
            self._canonical_payload(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return f"sha256:{hashlib.sha256(encoded).hexdigest()}"

    def to_dict(self) -> dict[str, Any]:
        return {
            **self._canonical_payload(),
            "term_structure_id": self.term_structure_id,
            "age_seconds": self.age_seconds,
            "max_time_years": self.max_time_years,
        }

    def _integral(self, field_name: str, start: float, end: float) -> float:
        if isinstance(start, bool) or isinstance(end, bool):
            raise MarketDataValidationError("integration times must be numeric")
        try:
            start_value = float(start)
            end_value = float(end)
        except (TypeError, ValueError) as exc:
            raise MarketDataValidationError(
                "integration times must be numeric"
            ) from exc
        if (
            not math.isfinite(start_value)
            or not math.isfinite(end_value)
            or start_value < 0
            or end_value < start_value
        ):
            raise MarketDataValidationError(
                "integration times must satisfy 0 <= start <= end"
            )
        if end_value > self.max_time_years + 1e-12:
            raise MarketDataValidationError(
                "term structure does not cover the requested time"
            )

        total = 0.0
        segment_start = 0.0
        for segment in self.segments:
            overlap_start = max(start_value, segment_start)
            overlap_end = min(end_value, segment.end_time_years)
            if overlap_end > overlap_start:
                total += float(getattr(segment, field_name)) * (
                    overlap_end - overlap_start
                )
            if end_value <= segment.end_time_years:
                break
            segment_start = segment.end_time_years
        return total

    def integrated_risk_free_rate(self, start: float, end: float) -> float:
        return self._integral("risk_free_rate", start, end)

    def integrated_dividend_yield(self, start: float, end: float) -> float:
        return self._integral("dividend_yield", start, end)

    def integrated_variance(self, start: float, end: float) -> float:
        if isinstance(start, bool) or isinstance(end, bool):
            raise MarketDataValidationError("integration times must be numeric")
        try:
            start_value = float(start)
            end_value = float(end)
        except (TypeError, ValueError) as exc:
            raise MarketDataValidationError(
                "integration times must be numeric"
            ) from exc
        if (
            not math.isfinite(start_value)
            or not math.isfinite(end_value)
            or start_value < 0
            or end_value < start_value
        ):
            raise MarketDataValidationError(
                "integration times must satisfy 0 <= start <= end"
            )
        if end_value > self.max_time_years + 1e-12:
            raise MarketDataValidationError(
                "term structure does not cover the requested time"
            )
        if start_value == end_value:
            return 0.0

        total = 0.0
        segment_start = 0.0
        for segment in self.segments:
            overlap_start = max(start_value, segment_start)
            overlap_end = min(end_value, segment.end_time_years)
            if overlap_end > overlap_start:
                total += segment.volatility**2 * (overlap_end - overlap_start)
            if end_value <= segment.end_time_years:
                break
            segment_start = segment.end_time_years
        return total

    def discount_factor(self, time_years: float) -> float:
        return math.exp(-self.integrated_risk_free_rate(0.0, time_years))

    def equivalent_flat_parameters(self, maturity_years: float) -> dict[str, float]:
        if isinstance(maturity_years, bool):
            raise MarketDataValidationError("maturity_years must be numeric")
        try:
            maturity = float(maturity_years)
        except (TypeError, ValueError) as exc:
            raise MarketDataValidationError("maturity_years must be numeric") from exc
        if not math.isfinite(maturity) or maturity <= 0:
            raise MarketDataValidationError("maturity_years must be finite and > 0")
        return {
            "risk_free_rate": self.integrated_risk_free_rate(0.0, maturity) / maturity,
            "dividend_yield": self.integrated_dividend_yield(0.0, maturity) / maturity,
            "volatility": math.sqrt(self.integrated_variance(0.0, maturity) / maturity),
        }


class EquityMarketDataProvider(Protocol):
    """Boundary implemented by manual, cached, or live market-data adapters."""

    provider_name: str

    def get_snapshot(
        self, symbol: str, valuation_time: datetime
    ) -> EquityMarketSnapshot: ...
