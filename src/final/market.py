import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, ClassVar, Protocol


EQUITY_MARKET_SNAPSHOT_VERSION = "equity-market-snapshot-v1"
EQUITY_GBM_FLAT_MODEL_VERSION = "equity-gbm-flat-v2"
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


class EquityMarketDataProvider(Protocol):
    """Boundary implemented by manual, cached, or live market-data adapters."""

    provider_name: str

    def get_snapshot(
        self, symbol: str, valuation_time: datetime
    ) -> EquityMarketSnapshot: ...
