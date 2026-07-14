import math
import os
import time
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from threading import Lock
from typing import Any, Callable, Mapping, Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import yfinance as yf
from yfinance.exceptions import (
    YFException,
    YFPricesMissingError,
    YFRateLimitError,
    YFTickerMissingError,
    YFTzMissingError,
)

from src.final.market import EquityMarketSnapshot


class LiveMarketDataError(Exception):
    pass


class MarketDataConfigurationError(LiveMarketDataError):
    pass


class MarketDataNotFoundError(LiveMarketDataError):
    pass


class MarketDataRateLimitError(LiveMarketDataError):
    pass


class MarketDataUpstreamError(LiveMarketDataError):
    pass


class MarketDataInvalidResponseError(LiveMarketDataError):
    pass


class MarketDataStaleError(LiveMarketDataError):
    pass


class MarketDataTypeMismatchError(LiveMarketDataError):
    pass


class MarketDataRequestError(LiveMarketDataError):
    pass


@dataclass(frozen=True)
class YFinanceSettings:
    request_timeout_seconds: float = 10.0
    cache_ttl_seconds: float = 60.0
    max_quote_age_seconds: float = 7 * 24 * 60 * 60
    future_tolerance_seconds: float = 5.0
    max_attempts: int = 2
    max_cache_entries: int = 1_000

    def __post_init__(self) -> None:
        numeric_non_negative = {
            "cache_ttl_seconds": self.cache_ttl_seconds,
            "max_quote_age_seconds": self.max_quote_age_seconds,
            "future_tolerance_seconds": self.future_tolerance_seconds,
        }
        for name, value in numeric_non_negative.items():
            if not math.isfinite(value) or value < 0:
                raise MarketDataConfigurationError(f"{name} must be finite and >= 0")
        if (
            not math.isfinite(self.request_timeout_seconds)
            or self.request_timeout_seconds <= 0
        ):
            raise MarketDataConfigurationError(
                "request_timeout_seconds must be finite and > 0"
            )
        if not 1 <= self.max_attempts <= 5:
            raise MarketDataConfigurationError("max_attempts must be between 1 and 5")
        if not 1 <= self.max_cache_entries <= 100_000:
            raise MarketDataConfigurationError(
                "max_cache_entries must be between 1 and 100000"
            )

    @classmethod
    def from_environment(
        cls, environment: Optional[Mapping[str, str]] = None
    ) -> "YFinanceSettings":
        values = environment if environment is not None else os.environ

        def read_float(name: str, default: float) -> float:
            try:
                return float(values.get(name, str(default)))
            except (TypeError, ValueError) as exc:
                raise MarketDataConfigurationError(f"{name} must be numeric") from exc

        def read_int(name: str, default: int) -> int:
            try:
                return int(values.get(name, str(default)))
            except (TypeError, ValueError) as exc:
                raise MarketDataConfigurationError(
                    f"{name} must be an integer"
                ) from exc

        return cls(
            request_timeout_seconds=read_float(
                "MARKET_DATA_REQUEST_TIMEOUT_SECONDS", 10.0
            ),
            cache_ttl_seconds=read_float("MARKET_DATA_CACHE_TTL_SECONDS", 60.0),
            max_quote_age_seconds=read_float(
                "MARKET_DATA_MAX_QUOTE_AGE_SECONDS", 7 * 24 * 60 * 60
            ),
            future_tolerance_seconds=read_float(
                "MARKET_DATA_FUTURE_TOLERANCE_SECONDS", 5.0
            ),
            max_attempts=read_int("MARKET_DATA_MAX_ATTEMPTS", 2),
            max_cache_entries=read_int("MARKET_DATA_MAX_CACHE_ENTRIES", 1_000),
        )


@dataclass(frozen=True)
class LiveEquityQuote:
    provider_name: str
    symbol: str
    spot: float
    currency: str
    market_data_time: datetime
    exchange: Optional[str]
    mic_code: Optional[str]
    instrument_type: Optional[str]
    underlier_type: Optional[str]
    provider_spot: float
    provider_currency: str
    unit_conversion_factor: float
    bar_interval: str
    data_delay_seconds: Optional[int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider_name,
            "symbol": self.symbol,
            "spot": self.spot,
            "currency": self.currency,
            "market_data_time": self.market_data_time.isoformat().replace(
                "+00:00", "Z"
            ),
            "exchange": self.exchange,
            "mic_code": self.mic_code,
            "instrument_type": self.instrument_type,
            "underlier_type": self.underlier_type,
            "provider_spot": self.provider_spot,
            "provider_currency": self.provider_currency,
            "unit_conversion_factor": self.unit_conversion_factor,
            "bar_interval": self.bar_interval,
            "data_delay_seconds": self.data_delay_seconds,
            "research_only": True,
        }


@dataclass(frozen=True)
class QuoteFetchResult:
    quote: LiveEquityQuote
    cache_hit: bool


@dataclass(frozen=True)
class LiveSnapshotResult:
    snapshot: EquityMarketSnapshot
    quote: LiveEquityQuote
    cache_hit: bool
    quote_age_seconds: float

    def metadata(self) -> dict[str, Any]:
        return {
            "provider": self.quote.provider_name,
            "research_only": True,
            "cache_hit": self.cache_hit,
            "quote_age_seconds": self.quote_age_seconds,
            "bar_interval": self.quote.bar_interval,
            "data_delay_seconds": self.quote.data_delay_seconds,
            "exchange": self.quote.exchange,
            "mic_code": self.quote.mic_code,
            "instrument_type": self.quote.instrument_type,
            "input_sources": {
                "spot": self.quote.provider_name,
                "currency": self.quote.provider_name,
                "market_data_time": self.quote.provider_name,
                "underlier_type": (
                    self.quote.provider_name if self.quote.underlier_type else "request"
                ),
                "calendar": (
                    self.quote.provider_name
                    if self.quote.mic_code or self.quote.exchange
                    else "application-default"
                ),
                "risk_free_rate": "request",
                "dividend_yield": "request",
                "volatility": "request",
            },
        }


def _normalize_symbol(
    symbol: str, error_type: type[LiveMarketDataError] = MarketDataInvalidResponseError
) -> str:
    normalized = symbol.strip()
    if not normalized or len(normalized) > 64 or not normalized.isprintable():
        raise error_type("symbol must contain 1 to 64 printable characters")
    return normalized


def _map_underlier_type(instrument_type: Optional[str]) -> Optional[str]:
    if not instrument_type:
        return None
    normalized = instrument_type.strip().lower().replace("_", " ")
    if normalized in {
        "etf",
        "exchange traded fund",
        "exchange traded note",
    }:
        return "etf"
    if normalized in {"index", "indices"}:
        return "index"
    equity_types = {
        "equity",
        "common stock",
        "preferred stock",
        "american depositary receipt",
        "depositary receipt",
        "global depositary receipt",
        "reit",
        "unit",
    }
    if normalized in equity_types or "stock" in normalized:
        return "equity"
    return None


def _normalize_quote_unit(spot: float, currency: str) -> tuple[float, str, float]:
    minor_units = {
        "GBp": ("GBP", 0.01),
        "ZAc": ("ZAR", 0.01),
        "ILA": ("ILS", 0.01),
    }
    major_currency, conversion_factor = minor_units.get(
        currency, (currency.upper(), 1.0)
    )
    return spot * conversion_factor, major_currency, conversion_factor


class YFinanceQuoteProvider:
    provider_name = "yfinance"

    def __init__(
        self,
        settings: Optional[YFinanceSettings] = None,
        ticker_factory: Callable[[str], Any] = yf.Ticker,
        monotonic_clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self.settings = settings or YFinanceSettings()
        self._ticker_factory = ticker_factory
        self._monotonic_clock = monotonic_clock
        self._sleeper = sleeper
        self._cache: OrderedDict[str, tuple[float, LiveEquityQuote]] = OrderedDict()
        self._cache_lock = Lock()

    @staticmethod
    def _cache_key(symbol: str) -> str:
        return symbol.upper()

    def _get_cached(self, key: str, now: float) -> Optional[LiveEquityQuote]:
        with self._cache_lock:
            cached = self._cache.get(key)
            if cached is None:
                return None
            expires_at, quote = cached
            if expires_at <= now:
                del self._cache[key]
                return None
            self._cache.move_to_end(key)
            return quote

    def _store_cached(self, key: str, quote: LiveEquityQuote, now: float) -> None:
        with self._cache_lock:
            self._cache[key] = (
                now + self.settings.cache_ttl_seconds,
                quote,
            )
            self._cache.move_to_end(key)
            while len(self._cache) > self.settings.max_cache_entries:
                self._cache.popitem(last=False)

    def get_quote(
        self,
        symbol: str,
        force_refresh: bool = False,
    ) -> QuoteFetchResult:
        normalized_symbol = _normalize_symbol(symbol, MarketDataRequestError)
        key = self._cache_key(normalized_symbol)
        now = self._monotonic_clock()
        if not force_refresh:
            cached = self._get_cached(key, now)
            if cached is not None:
                return QuoteFetchResult(quote=cached, cache_hit=True)

        quote = self._request_quote(normalized_symbol)
        self._store_cached(key, quote, now)
        return QuoteFetchResult(quote=quote, cache_hit=False)

    def _request_quote(self, symbol: str) -> LiveEquityQuote:
        for attempt in range(self.settings.max_attempts):
            try:
                ticker = self._ticker_factory(symbol)
                history = ticker.history(
                    period="5d",
                    interval="1m",
                    prepost=False,
                    actions=False,
                    auto_adjust=False,
                    repair=True,
                    keepna=False,
                    timeout=self.settings.request_timeout_seconds,
                    raise_errors=True,
                )
                metadata = ticker.get_history_metadata()
                return self._parse_quote(history, metadata, symbol)
            except YFRateLimitError as exc:
                if attempt + 1 == self.settings.max_attempts:
                    raise MarketDataRateLimitError(
                        "Yahoo Finance rate limit reached"
                    ) from exc
            except (
                YFPricesMissingError,
                YFTickerMissingError,
                YFTzMissingError,
            ) as exc:
                raise MarketDataNotFoundError(
                    f"market data not found for {symbol}"
                ) from exc
            except LiveMarketDataError:
                raise
            except YFException as exc:
                if attempt + 1 == self.settings.max_attempts:
                    raise MarketDataUpstreamError(
                        "Yahoo Finance market data is unavailable"
                    ) from exc
            except Exception as exc:
                if attempt + 1 == self.settings.max_attempts:
                    raise MarketDataUpstreamError(
                        "Yahoo Finance market data is unavailable"
                    ) from exc
            self._sleeper(min(0.25 * (2**attempt), 2.0))

        raise MarketDataUpstreamError("Yahoo Finance market data is unavailable")

    @staticmethod
    def _parse_market_time(
        raw_market_time: Any, metadata: Mapping[str, Any]
    ) -> datetime:
        try:
            parsed = (
                raw_market_time.to_pydatetime()
                if hasattr(raw_market_time, "to_pydatetime")
                else raw_market_time
            )
            if not isinstance(parsed, datetime):
                raise TypeError
            if parsed.tzinfo is None or parsed.utcoffset() is None:
                timezone_name = metadata.get("exchangeTimezoneName")
                if not timezone_name:
                    raise ValueError
                parsed = parsed.replace(tzinfo=ZoneInfo(str(timezone_name)))
            return parsed.astimezone(timezone.utc)
        except (TypeError, ValueError, ZoneInfoNotFoundError) as exc:
            raise MarketDataInvalidResponseError(
                "Yahoo Finance returned an invalid bar timestamp"
            ) from exc

    def _parse_quote(
        self,
        history: Any,
        metadata: Any,
        requested_symbol: str,
    ) -> LiveEquityQuote:
        if not isinstance(metadata, Mapping):
            raise MarketDataInvalidResponseError(
                "Yahoo Finance returned invalid quote metadata"
            )
        try:
            closes = history["Close"].dropna()
            if closes.empty:
                raise MarketDataNotFoundError(
                    f"market data not found for {requested_symbol}"
                )
            spot = float(closes.iloc[-1])
            raw_market_time = closes.index[-1]
        except MarketDataNotFoundError:
            raise
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            raise MarketDataInvalidResponseError(
                "Yahoo Finance omitted a valid close price"
            ) from exc
        if not math.isfinite(spot) or spot <= 0:
            raise MarketDataInvalidResponseError(
                "Yahoo Finance returned an invalid close price"
            )

        provider_currency = str(metadata.get("currency", "")).strip()
        if (
            len(provider_currency) != 3
            or not provider_currency.isalpha()
            or not provider_currency.isascii()
        ):
            raise MarketDataInvalidResponseError(
                "Yahoo Finance omitted a valid currency"
            )
        normalized_spot, currency, conversion_factor = _normalize_quote_unit(
            spot, provider_currency
        )
        raw_instrument_type = metadata.get("instrumentType")
        instrument_type = (
            str(raw_instrument_type).strip() if raw_instrument_type else None
        )
        raw_delay = metadata.get("exchangeDataDelayedBy")
        try:
            data_delay_seconds = int(raw_delay) if raw_delay is not None else None
            if data_delay_seconds is not None and data_delay_seconds < 0:
                data_delay_seconds = None
        except (TypeError, ValueError):
            data_delay_seconds = None
        exchange = metadata.get("fullExchangeName") or metadata.get("exchangeName")

        return LiveEquityQuote(
            provider_name=self.provider_name,
            symbol=_normalize_symbol(str(metadata.get("symbol") or requested_symbol)),
            spot=normalized_spot,
            currency=currency,
            market_data_time=self._parse_market_time(raw_market_time, metadata),
            exchange=str(exchange).strip() if exchange else None,
            mic_code=None,
            instrument_type=instrument_type,
            underlier_type=_map_underlier_type(instrument_type),
            provider_spot=spot,
            provider_currency=provider_currency,
            unit_conversion_factor=conversion_factor,
            bar_interval="1m",
            data_delay_seconds=data_delay_seconds,
        )


class LiveMarketDataService:
    def __init__(
        self,
        provider: YFinanceQuoteProvider,
        clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        self.provider = provider
        self._clock = clock

    def get_quote(self, symbol: str) -> tuple[QuoteFetchResult, float]:
        valuation_time = self._valuation_time()
        return self._get_quote_at(symbol, valuation_time)

    def _valuation_time(self) -> datetime:
        valuation_time = self._clock()
        if valuation_time.tzinfo is None or valuation_time.utcoffset() is None:
            raise MarketDataConfigurationError(
                "market data service clock must be timezone-aware"
            )
        return valuation_time.astimezone(timezone.utc)

    def _get_quote_at(
        self, symbol: str, valuation_time: datetime
    ) -> tuple[QuoteFetchResult, float]:
        result = self.provider.get_quote(symbol=symbol)
        age_seconds = (valuation_time - result.quote.market_data_time).total_seconds()
        settings = self.provider.settings
        if age_seconds < -settings.future_tolerance_seconds:
            raise MarketDataInvalidResponseError(
                "market data provider returned a future-dated quote"
            )
        if age_seconds > settings.max_quote_age_seconds:
            raise MarketDataStaleError(
                "market data bar is older than the configured freshness limit"
            )
        return result, max(age_seconds, 0.0)

    def get_snapshot(
        self,
        symbol: str,
        underlier_type: str,
        risk_free_rate: float,
        dividend_yield: float,
        volatility: float,
        day_count: str = "ACT/365F",
    ) -> LiveSnapshotResult:
        valuation_time = self._valuation_time()
        quote_result, age_seconds = self._get_quote_at(symbol, valuation_time)
        quote = quote_result.quote
        expected_type = underlier_type.strip().lower()
        if quote.underlier_type and quote.underlier_type != expected_type:
            raise MarketDataTypeMismatchError(
                "provider instrument type does not match requested underlier_type"
            )
        calendar = quote.mic_code or quote.exchange or "WEEKDAYS"
        snapshot = EquityMarketSnapshot(
            symbol=quote.symbol,
            underlier_type=expected_type,
            currency=quote.currency,
            valuation_time=valuation_time,
            market_data_time=quote.market_data_time,
            spot=quote.spot,
            risk_free_rate=risk_free_rate,
            dividend_yield=dividend_yield,
            volatility=volatility,
            calendar=calendar,
            day_count=day_count,
            source=f"{quote.provider_name}:1m-close+request-model-inputs",
        )
        return LiveSnapshotResult(
            snapshot=snapshot,
            quote=quote,
            cache_hit=quote_result.cache_hit,
            quote_age_seconds=age_seconds,
        )


_SERVICE: Optional[LiveMarketDataService] = None
_SERVICE_LOCK = Lock()


def clear_live_market_data_service() -> None:
    global _SERVICE
    with _SERVICE_LOCK:
        _SERVICE = None


def get_live_market_data_service() -> LiveMarketDataService:
    global _SERVICE
    with _SERVICE_LOCK:
        if _SERVICE is None:
            settings = YFinanceSettings.from_environment()
            _SERVICE = LiveMarketDataService(YFinanceQuoteProvider(settings))
        return _SERVICE


def get_live_market_data_status() -> dict[str, Any]:
    return {
        "enabled": True,
        "configured": True,
        "provider": "yfinance",
        "credentials_required": False,
        "research_only": True,
    }
