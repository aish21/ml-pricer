import hashlib
import json
import math
import os
import statistics
import time
import xml.etree.ElementTree as ET
from collections import OrderedDict
from dataclasses import dataclass, replace
from datetime import date, datetime, time as datetime_time, timedelta, timezone
from threading import Lock
from typing import Any, Callable, Mapping, Optional, Sequence
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import yfinance as yf
from yfinance.exceptions import YFException, YFRateLimitError

from src.final.market import (
    EQUITY_RESEARCH_MARKET_VERSION,
    EquityMarketSegment,
    EquityMarketTermStructure,
    MarketDataValidationError,
)
from app.services.live_market_data import (
    LiveMarketDataError,
    MarketDataConfigurationError,
    MarketDataInvalidResponseError,
    MarketDataRateLimitError,
    MarketDataRequestError,
    MarketDataUpstreamError,
    get_live_market_data_service,
)


TREASURY_CURVE_METHOD = "treasury-cmt-continuous-zero-proxy-v1"
DIVIDEND_METHOD = "yfinance-trailing-cash-distribution-yield-v1"
OPTION_VOLATILITY_METHOD = "yfinance-atm-forward-variance-v1"
RESEARCH_SEGMENT_TENORS = (
    1.0 / 12.0,
    0.25,
    0.5,
    1.0,
    2.0,
    3.0,
    5.0,
    7.0,
    10.0,
    20.0,
    30.0,
)
TREASURY_FEED_URL = (
    "https://home.treasury.gov/resource-center/data-chart-center/"
    "interest-rates/pages/xml"
)
TREASURY_TENOR_FIELDS = (
    (1.0 / 12.0, "BC_1MONTH"),
    (1.5 / 12.0, "BC_1_5MONTH"),
    (2.0 / 12.0, "BC_2MONTH"),
    (0.25, "BC_3MONTH"),
    (4.0 / 12.0, "BC_4MONTH"),
    (0.5, "BC_6MONTH"),
    (1.0, "BC_1YEAR"),
    (2.0, "BC_2YEAR"),
    (3.0, "BC_3YEAR"),
    (5.0, "BC_5YEAR"),
    (7.0, "BC_7YEAR"),
    (10.0, "BC_10YEAR"),
    (20.0, "BC_20YEAR"),
    (30.0, "BC_30YEAR"),
)
SECONDS_PER_YEAR = 365.0 * 24.0 * 60.0 * 60.0


class ResearchMarketDataError(LiveMarketDataError):
    pass


class ResearchMarketUnsupportedError(ResearchMarketDataError, MarketDataRequestError):
    pass


class ResearchMarketInvalidResponseError(
    ResearchMarketDataError, MarketDataInvalidResponseError
):
    pass


@dataclass(frozen=True)
class TreasuryCurveSettings:
    request_timeout_seconds: float = 10.0
    cache_ttl_seconds: float = 6.0 * 60.0 * 60.0
    max_curve_age_days: int = 10
    max_attempts: int = 2
    max_response_bytes: int = 5_000_000

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.request_timeout_seconds)
            or self.request_timeout_seconds <= 0
        ):
            raise MarketDataConfigurationError(
                "treasury request timeout must be finite and > 0"
            )
        if not math.isfinite(self.cache_ttl_seconds) or self.cache_ttl_seconds < 0:
            raise MarketDataConfigurationError(
                "treasury cache TTL must be finite and >= 0"
            )
        if not 0 <= self.max_curve_age_days <= 31:
            raise MarketDataConfigurationError(
                "treasury max curve age must be between 0 and 31 days"
            )
        if not 1 <= self.max_attempts <= 5:
            raise MarketDataConfigurationError(
                "treasury max attempts must be between 1 and 5"
            )
        if not 1_000 <= self.max_response_bytes <= 50_000_000:
            raise MarketDataConfigurationError(
                "treasury max response bytes must be between 1000 and 50000000"
            )

    @classmethod
    def from_environment(
        cls, environment: Optional[Mapping[str, str]] = None
    ) -> "TreasuryCurveSettings":
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
                "TREASURY_CURVE_REQUEST_TIMEOUT_SECONDS", 10.0
            ),
            cache_ttl_seconds=read_float(
                "TREASURY_CURVE_CACHE_TTL_SECONDS", 6.0 * 60.0 * 60.0
            ),
            max_curve_age_days=read_int("TREASURY_CURVE_MAX_AGE_DAYS", 10),
            max_attempts=read_int("TREASURY_CURVE_MAX_ATTEMPTS", 2),
        )


@dataclass(frozen=True)
class TreasuryParYieldPoint:
    tenor_years: float
    par_yield: float

    def to_dict(self) -> dict[str, float]:
        return {
            "tenor_years": self.tenor_years,
            "par_yield": self.par_yield,
        }


@dataclass(frozen=True)
class TreasuryCurveObservation:
    as_of_date: date
    feed_updated_time: Optional[datetime]
    points: tuple[TreasuryParYieldPoint, ...]
    source_url: str
    cache_hit: bool = False

    def continuous_zero_rate(self, tenor_years: float) -> float:
        try:
            tenor = float(tenor_years)
        except (TypeError, ValueError) as exc:
            raise ResearchMarketUnsupportedError("tenor must be numeric") from exc
        if not math.isfinite(tenor) or tenor <= 0:
            raise ResearchMarketUnsupportedError("tenor must be finite and > 0")
        if tenor > self.points[-1].tenor_years + 1e-12:
            raise ResearchMarketUnsupportedError(
                "USD Treasury curve does not cover the requested maturity"
            )

        if tenor <= self.points[0].tenor_years:
            par_yield = self.points[0].par_yield
        else:
            par_yield = self.points[-1].par_yield
            for left, right in zip(self.points, self.points[1:]):
                if tenor <= right.tenor_years:
                    weight = (tenor - left.tenor_years) / (
                        right.tenor_years - left.tenor_years
                    )
                    par_yield = left.par_yield + weight * (
                        right.par_yield - left.par_yield
                    )
                    break

        # Treasury CMTs are semiannual bond-equivalent par yields. Phase 6 uses
        # their continuous-yield equivalent as an explicit zero-rate proxy.
        return 2.0 * math.log1p(par_yield / 2.0)

    def discount_factor(self, tenor_years: float) -> float:
        tenor = float(tenor_years)
        return math.exp(-self.continuous_zero_rate(tenor) * tenor)

    def to_dict(self, valuation_time: datetime) -> dict[str, Any]:
        feed_updated = (
            self.feed_updated_time.isoformat().replace("+00:00", "Z")
            if self.feed_updated_time
            else None
        )
        return {
            "provider": "U.S. Department of the Treasury",
            "method": TREASURY_CURVE_METHOD,
            "as_of_date": self.as_of_date.isoformat(),
            "age_days": (valuation_time.date() - self.as_of_date).days,
            "feed_updated_time": feed_updated,
            "source_url": self.source_url,
            "cache_hit": self.cache_hit,
            "points": [point.to_dict() for point in self.points],
        }


@dataclass(frozen=True)
class _TreasuryYearData:
    feed_updated_time: Optional[datetime]
    observations: tuple[TreasuryCurveObservation, ...]


class TreasuryCurveProvider:
    def __init__(
        self,
        settings: Optional[TreasuryCurveSettings] = None,
        http_get: Callable[..., Any] = requests.get,
        monotonic_clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
        feed_url: str = TREASURY_FEED_URL,
    ) -> None:
        self.settings = settings or TreasuryCurveSettings()
        self._http_get = http_get
        self._monotonic_clock = monotonic_clock
        self._sleeper = sleeper
        self.feed_url = feed_url
        self._cache: dict[int, tuple[float, _TreasuryYearData]] = {}
        self._cache_lock = Lock()

    def get_curve(self, valuation_time: datetime) -> TreasuryCurveObservation:
        normalized_time = _normalize_aware_time(valuation_time, "valuation_time")
        for year in (normalized_time.year, normalized_time.year - 1):
            year_data, cache_hit = self._get_year(year)
            eligible = [
                observation
                for observation in year_data.observations
                if observation.as_of_date <= normalized_time.date()
            ]
            if not eligible:
                continue
            latest = max(eligible, key=lambda observation: observation.as_of_date)
            age_days = (normalized_time.date() - latest.as_of_date).days
            if age_days > self.settings.max_curve_age_days:
                raise ResearchMarketInvalidResponseError(
                    "latest USD Treasury curve is older than the configured limit"
                )
            return replace(latest, cache_hit=cache_hit)
        raise ResearchMarketInvalidResponseError(
            "USD Treasury feed has no curve at or before valuation time"
        )

    def _get_year(self, year: int) -> tuple[_TreasuryYearData, bool]:
        now = self._monotonic_clock()
        with self._cache_lock:
            cached = self._cache.get(year)
            if cached and cached[0] > now:
                return cached[1], True
            if cached:
                del self._cache[year]

        data = self._request_year(year)
        with self._cache_lock:
            self._cache[year] = (now + self.settings.cache_ttl_seconds, data)
        return data, False

    def _request_year(self, year: int) -> _TreasuryYearData:
        params = {
            "data": "daily_treasury_yield_curve",
            "field_tdr_date_value": str(year),
        }
        for attempt in range(self.settings.max_attempts):
            try:
                response = self._http_get(
                    self.feed_url,
                    params=params,
                    timeout=self.settings.request_timeout_seconds,
                    headers={"User-Agent": "ml-pricer-research/0.6"},
                )
                response.raise_for_status()
                content = bytes(response.content)
                if len(content) > self.settings.max_response_bytes:
                    raise ResearchMarketInvalidResponseError(
                        "USD Treasury feed response exceeded the size limit"
                    )
                return self._parse_year(content, year)
            except LiveMarketDataError:
                raise
            except requests.RequestException as exc:
                if attempt + 1 == self.settings.max_attempts:
                    raise MarketDataUpstreamError(
                        "USD Treasury curve is unavailable"
                    ) from exc
            except (ET.ParseError, TypeError, ValueError) as exc:
                raise ResearchMarketInvalidResponseError(
                    "USD Treasury feed returned invalid XML"
                ) from exc
            except Exception as exc:
                if attempt + 1 == self.settings.max_attempts:
                    raise MarketDataUpstreamError(
                        "USD Treasury curve is unavailable"
                    ) from exc
            self._sleeper(min(0.25 * (2**attempt), 2.0))
        raise MarketDataUpstreamError("USD Treasury curve is unavailable")

    def _parse_year(self, content: bytes, year: int) -> _TreasuryYearData:
        upper_content = content.upper()
        if b"<!DOCTYPE" in upper_content or b"<!ENTITY" in upper_content:
            raise ResearchMarketInvalidResponseError(
                "USD Treasury feed contained a prohibited XML declaration"
            )
        root = ET.fromstring(content)
        atom_namespace = "{http://www.w3.org/2005/Atom}"
        metadata_namespace = (
            "{http://schemas.microsoft.com/ado/2007/08/dataservices/metadata}"
        )
        data_namespace = "{http://schemas.microsoft.com/ado/2007/08/dataservices}"

        feed_updated_time = _parse_optional_utc_time(
            root.findtext(f"{atom_namespace}updated")
        )
        observations = []
        for entry in root.findall(f"{atom_namespace}entry"):
            properties = entry.find(
                f"{atom_namespace}content/{metadata_namespace}properties"
            )
            if properties is None:
                continue
            raw_date = properties.findtext(f"{data_namespace}NEW_DATE")
            if not raw_date:
                continue
            as_of_date = datetime.fromisoformat(raw_date.rstrip("Z")).date()
            points = []
            for tenor, field_name in TREASURY_TENOR_FIELDS:
                raw_value = properties.findtext(f"{data_namespace}{field_name}")
                if raw_value is None or not raw_value.strip():
                    continue
                percent_value = float(raw_value)
                par_yield = percent_value / 100.0
                if not math.isfinite(par_yield) or not -0.25 <= par_yield <= 1.0:
                    raise ResearchMarketInvalidResponseError(
                        "USD Treasury feed returned an invalid par yield"
                    )
                points.append(TreasuryParYieldPoint(tenor, par_yield))
            if len(points) < 4:
                continue
            observations.append(
                TreasuryCurveObservation(
                    as_of_date=as_of_date,
                    feed_updated_time=feed_updated_time,
                    points=tuple(points),
                    source_url=(
                        f"{self.feed_url}?data=daily_treasury_yield_curve"
                        f"&field_tdr_date_value={year}"
                    ),
                )
            )
        if not observations:
            raise ResearchMarketInvalidResponseError(
                "USD Treasury feed contained no complete curve observations"
            )
        return _TreasuryYearData(
            feed_updated_time=feed_updated_time,
            observations=tuple(observations),
        )


@dataclass(frozen=True)
class YFinanceDividendSettings:
    request_timeout_seconds: float = 10.0
    cache_ttl_seconds: float = 60.0 * 60.0
    max_attempts: int = 2
    max_cache_entries: int = 1_000

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.request_timeout_seconds)
            or self.request_timeout_seconds <= 0
        ):
            raise MarketDataConfigurationError(
                "dividend request timeout must be finite and > 0"
            )
        if not math.isfinite(self.cache_ttl_seconds) or self.cache_ttl_seconds < 0:
            raise MarketDataConfigurationError(
                "dividend cache TTL must be finite and >= 0"
            )
        if not 1 <= self.max_attempts <= 5:
            raise MarketDataConfigurationError(
                "dividend max attempts must be between 1 and 5"
            )
        if not 1 <= self.max_cache_entries <= 100_000:
            raise MarketDataConfigurationError(
                "dividend max cache entries must be between 1 and 100000"
            )


@dataclass(frozen=True)
class DividendYieldObservation:
    period_start: date
    period_end: date
    cash_distributions: float
    spot: float
    continuous_yield: float
    observations: int
    cache_hit: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": "yfinance",
            "method": DIVIDEND_METHOD,
            "period_start": self.period_start.isoformat(),
            "period_end": self.period_end.isoformat(),
            "cash_distributions": self.cash_distributions,
            "spot": self.spot,
            "continuous_yield": self.continuous_yield,
            "observations": self.observations,
            "cache_hit": self.cache_hit,
        }


@dataclass(frozen=True)
class _DividendCashObservation:
    period_start: date
    period_end: date
    cash_distributions: float
    observations: int


class YFinanceDividendProvider:
    def __init__(
        self,
        settings: Optional[YFinanceDividendSettings] = None,
        ticker_factory: Callable[[str], Any] = yf.Ticker,
        monotonic_clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self.settings = settings or YFinanceDividendSettings()
        self._ticker_factory = ticker_factory
        self._monotonic_clock = monotonic_clock
        self._sleeper = sleeper
        self._cache: OrderedDict[
            tuple[str, date], tuple[float, _DividendCashObservation]
        ] = OrderedDict()
        self._cache_lock = Lock()

    def get_yield(
        self, symbol: str, spot: float, valuation_time: datetime
    ) -> DividendYieldObservation:
        normalized_time = _normalize_aware_time(valuation_time, "valuation_time")
        key = (symbol.upper(), normalized_time.date())
        now = self._monotonic_clock()
        with self._cache_lock:
            cached = self._cache.get(key)
            if cached and cached[0] > now:
                self._cache.move_to_end(key)
                return self._to_yield(cached[1], spot, cache_hit=True)
            if cached:
                del self._cache[key]

        cash_observation = self._request_history(symbol, normalized_time)
        with self._cache_lock:
            self._cache[key] = (
                now + self.settings.cache_ttl_seconds,
                cash_observation,
            )
            self._cache.move_to_end(key)
            while len(self._cache) > self.settings.max_cache_entries:
                self._cache.popitem(last=False)
        return self._to_yield(cash_observation, spot, cache_hit=False)

    def _request_history(
        self, symbol: str, valuation_time: datetime
    ) -> _DividendCashObservation:
        for attempt in range(self.settings.max_attempts):
            try:
                ticker = self._ticker_factory(symbol)
                history = ticker.history(
                    period="1y",
                    interval="1d",
                    prepost=False,
                    actions=True,
                    auto_adjust=False,
                    repair=True,
                    keepna=False,
                    timeout=self.settings.request_timeout_seconds,
                    raise_errors=True,
                )
                return self._parse_history(history, valuation_time)
            except YFRateLimitError as exc:
                if attempt + 1 == self.settings.max_attempts:
                    raise MarketDataRateLimitError(
                        "Yahoo Finance dividend-history rate limit reached"
                    ) from exc
            except LiveMarketDataError:
                raise
            except YFException as exc:
                if attempt + 1 == self.settings.max_attempts:
                    raise MarketDataUpstreamError(
                        "Yahoo Finance dividend history is unavailable"
                    ) from exc
            except Exception as exc:
                if attempt + 1 == self.settings.max_attempts:
                    raise MarketDataUpstreamError(
                        "Yahoo Finance dividend history is unavailable"
                    ) from exc
            self._sleeper(min(0.25 * (2**attempt), 2.0))
        raise MarketDataUpstreamError("Yahoo Finance dividend history is unavailable")

    @staticmethod
    def _parse_history(
        history: Any, valuation_time: datetime
    ) -> _DividendCashObservation:
        try:
            if history.empty or "Dividends" not in history:
                raise ResearchMarketInvalidResponseError(
                    "Yahoo Finance omitted dividend history"
                )
            timestamps = pd.to_datetime(history.index, utc=True)
            dividends = pd.to_numeric(history["Dividends"], errors="coerce")
        except ResearchMarketDataError:
            raise
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            raise ResearchMarketInvalidResponseError(
                "Yahoo Finance returned invalid dividend history"
            ) from exc

        period_end = valuation_time.date()
        period_start = date.fromordinal(period_end.toordinal() - 365)
        date_values = pd.Series(timestamps.date, index=history.index)
        included = (date_values >= period_start) & (date_values <= period_end)
        selected = dividends.loc[included]
        if selected.isna().any() or (selected < 0).any():
            raise ResearchMarketInvalidResponseError(
                "Yahoo Finance returned invalid cash distributions"
            )
        selected_dates = date_values.loc[included]
        if (
            selected.empty
            or selected_dates.min() > period_start + timedelta(days=7)
            or selected_dates.max() < period_end - timedelta(days=10)
        ):
            raise ResearchMarketInvalidResponseError(
                "Yahoo Finance dividend history does not cover the trailing year"
            )
        positive = selected[selected > 0]
        return _DividendCashObservation(
            period_start=period_start,
            period_end=period_end,
            cash_distributions=float(positive.sum()),
            observations=int(positive.count()),
        )

    @staticmethod
    def _to_yield(
        observation: _DividendCashObservation, spot: float, cache_hit: bool
    ) -> DividendYieldObservation:
        if not math.isfinite(spot) or spot <= 0:
            raise ResearchMarketInvalidResponseError(
                "spot must be positive for dividend calibration"
            )
        continuous_yield = math.log1p(observation.cash_distributions / spot)
        return DividendYieldObservation(
            period_start=observation.period_start,
            period_end=observation.period_end,
            cash_distributions=observation.cash_distributions,
            spot=spot,
            continuous_yield=continuous_yield,
            observations=observation.observations,
            cache_hit=cache_hit,
        )


@dataclass(frozen=True)
class YFinanceOptionSettings:
    cache_ttl_seconds: float = 5.0 * 60.0
    min_expiry_days: int = 7
    max_expiry_gap_days: int = 62
    atm_strike_count: int = 5
    max_combined_spread_fraction: float = 0.10
    max_attempts: int = 2

    def __post_init__(self) -> None:
        if not math.isfinite(self.cache_ttl_seconds) or self.cache_ttl_seconds < 0:
            raise MarketDataConfigurationError(
                "option cache TTL must be finite and >= 0"
            )
        if not 1 <= self.min_expiry_days <= 31:
            raise MarketDataConfigurationError(
                "minimum option expiry must be between 1 and 31 days"
            )
        if not 1 <= self.max_expiry_gap_days <= 366:
            raise MarketDataConfigurationError(
                "maximum option expiry gap must be between 1 and 366 days"
            )
        if not 1 <= self.atm_strike_count <= 21:
            raise MarketDataConfigurationError(
                "ATM strike count must be between 1 and 21"
            )
        if (
            not math.isfinite(self.max_combined_spread_fraction)
            or not 0 < self.max_combined_spread_fraction <= 1
        ):
            raise MarketDataConfigurationError(
                "maximum option spread fraction must be > 0 and <= 1"
            )
        if not 1 <= self.max_attempts <= 5:
            raise MarketDataConfigurationError(
                "option max attempts must be between 1 and 5"
            )


@dataclass(frozen=True)
class OptionMarketPoint:
    target_time_years: float
    option_time_years: float
    expiry: str
    representative_strike: float
    volatility: float
    combined_spread_fraction: float
    strikes_used: int
    cache_hit: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_time_years": self.target_time_years,
            "option_time_years": self.option_time_years,
            "expiry": self.expiry,
            "representative_strike": self.representative_strike,
            "volatility": self.volatility,
            "combined_spread_fraction": self.combined_spread_fraction,
            "strikes_used": self.strikes_used,
            "cache_hit": self.cache_hit,
        }


class YFinanceOptionProvider:
    def __init__(
        self,
        settings: Optional[YFinanceOptionSettings] = None,
        ticker_factory: Callable[[str], Any] = yf.Ticker,
        monotonic_clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self.settings = settings or YFinanceOptionSettings()
        self._ticker_factory = ticker_factory
        self._monotonic_clock = monotonic_clock
        self._sleeper = sleeper
        self._chain_cache: OrderedDict[tuple[str, str], tuple[float, Any]] = (
            OrderedDict()
        )
        self._cache_lock = Lock()

    def get_points(
        self,
        symbol: str,
        spot: float,
        valuation_time: datetime,
        target_tenors: Sequence[float],
    ) -> tuple[OptionMarketPoint, ...]:
        for attempt in range(self.settings.max_attempts):
            try:
                return self._get_points_once(
                    symbol, spot, valuation_time, target_tenors
                )
            except YFRateLimitError as exc:
                if attempt + 1 == self.settings.max_attempts:
                    raise MarketDataRateLimitError(
                        "Yahoo Finance option-chain rate limit reached"
                    ) from exc
            except LiveMarketDataError:
                raise
            except YFException as exc:
                if attempt + 1 == self.settings.max_attempts:
                    raise MarketDataUpstreamError(
                        "Yahoo Finance option chains are unavailable"
                    ) from exc
            except Exception as exc:
                if attempt + 1 == self.settings.max_attempts:
                    raise MarketDataUpstreamError(
                        "Yahoo Finance option chains are unavailable"
                    ) from exc
            self._sleeper(min(0.25 * (2**attempt), 2.0))
        raise MarketDataUpstreamError("Yahoo Finance option chains are unavailable")

    def _get_points_once(
        self,
        symbol: str,
        spot: float,
        valuation_time: datetime,
        target_tenors: Sequence[float],
    ) -> tuple[OptionMarketPoint, ...]:
        ticker = self._ticker_factory(symbol)
        expiry_candidates = self._parse_expiries(ticker.options, valuation_time)
        if not expiry_candidates:
            raise ResearchMarketUnsupportedError(
                "no sufficiently dated Yahoo Finance option expiries are available"
            )

        points = []
        for target in target_tenors:
            raw_expiry, option_tenor = min(
                expiry_candidates,
                key=lambda candidate: abs(candidate[1] - target),
            )
            gap_days = abs(option_tenor - target) * 365.0
            if gap_days > self.settings.max_expiry_gap_days:
                raise ResearchMarketUnsupportedError(
                    "option expiries do not cover the requested maturity"
                )
            chain, cache_hit = self._get_chain(ticker, symbol, raw_expiry)
            points.append(
                self._calibrate_point(
                    chain=chain,
                    spot=spot,
                    target_tenor=float(target),
                    option_tenor=option_tenor,
                    expiry=raw_expiry,
                    cache_hit=cache_hit,
                )
            )
        return tuple(points)

    def _parse_expiries(
        self, raw_expiries: Any, valuation_time: datetime
    ) -> list[tuple[str, float]]:
        normalized_time = _normalize_aware_time(valuation_time, "valuation_time")
        exchange_timezone = ZoneInfo("America/New_York")
        parsed = []
        try:
            values = tuple(raw_expiries)
        except TypeError as exc:
            raise ResearchMarketInvalidResponseError(
                "Yahoo Finance returned invalid option expiries"
            ) from exc
        for raw_value in values:
            try:
                expiry_date = date.fromisoformat(str(raw_value))
            except ValueError:
                continue
            expiry_time = datetime.combine(
                expiry_date,
                datetime_time(hour=16),
                tzinfo=exchange_timezone,
            ).astimezone(timezone.utc)
            tenor = (expiry_time - normalized_time).total_seconds() / SECONDS_PER_YEAR
            if tenor >= self.settings.min_expiry_days / 365.0:
                parsed.append((expiry_date.isoformat(), tenor))
        return parsed

    def _get_chain(self, ticker: Any, symbol: str, expiry: str) -> tuple[Any, bool]:
        key = (symbol.upper(), expiry)
        now = self._monotonic_clock()
        with self._cache_lock:
            cached = self._chain_cache.get(key)
            if cached and cached[0] > now:
                self._chain_cache.move_to_end(key)
                return cached[1], True
            if cached:
                del self._chain_cache[key]

        chain = ticker.option_chain(expiry)
        with self._cache_lock:
            self._chain_cache[key] = (
                now + self.settings.cache_ttl_seconds,
                chain,
            )
            self._chain_cache.move_to_end(key)
            while len(self._chain_cache) > 256:
                self._chain_cache.popitem(last=False)
        return chain, False

    def _calibrate_point(
        self,
        chain: Any,
        spot: float,
        target_tenor: float,
        option_tenor: float,
        expiry: str,
        cache_hit: bool,
    ) -> OptionMarketPoint:
        try:
            calls = chain.calls
            puts = chain.puts
            merged = calls.merge(puts, on="strike", suffixes=("_call", "_put"))
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            raise ResearchMarketInvalidResponseError(
                "Yahoo Finance returned an invalid option chain"
            ) from exc

        required = {
            "strike",
            "bid_call",
            "ask_call",
            "impliedVolatility_call",
            "bid_put",
            "ask_put",
            "impliedVolatility_put",
        }
        if not required <= set(merged.columns):
            raise ResearchMarketInvalidResponseError(
                "Yahoo Finance option chain omitted required quote fields"
            )
        for column in required:
            merged[column] = pd.to_numeric(merged[column], errors="coerce")

        valid_quotes = merged[
            merged[list(required)].notna().all(axis=1)
            & (merged["strike"] > 0)
            & (merged["bid_call"] >= 0)
            & (merged["ask_call"] >= merged["bid_call"])
            & (merged["ask_call"] > 0)
            & (merged["bid_put"] >= 0)
            & (merged["ask_put"] >= merged["bid_put"])
            & (merged["ask_put"] > 0)
        ].copy()
        if valid_quotes.empty:
            raise ResearchMarketInvalidResponseError(
                "Yahoo Finance option chain contained no valid two-sided quotes"
            )
        valid_quotes["moneyness_distance"] = (
            (valid_quotes["strike"] / spot).map(math.log).abs()
        )
        nearest = valid_quotes.sort_values("moneyness_distance").head(
            self.settings.atm_strike_count
        )

        volatilities = []
        spreads = []
        strikes = []
        for _, row in nearest.iterrows():
            combined_spread = (
                float(row["ask_call"])
                - float(row["bid_call"])
                + float(row["ask_put"])
                - float(row["bid_put"])
            ) / spot
            if combined_spread > self.settings.max_combined_spread_fraction:
                continue
            strike = float(row["strike"])
            strike_volatilities = [
                float(row["impliedVolatility_call"]),
                float(row["impliedVolatility_put"]),
            ]
            strike_volatilities = [
                volatility
                for volatility in strike_volatilities
                if math.isfinite(volatility) and 0.01 <= volatility <= 5.0
            ]
            if not strike_volatilities:
                continue
            volatilities.append(statistics.median(strike_volatilities))
            spreads.append(combined_spread)
            strikes.append(strike)
        if not strikes:
            raise ResearchMarketInvalidResponseError(
                "Yahoo Finance option chain had no usable near-ATM quotes"
            )

        representative_strike = min(strikes, key=lambda strike: abs(strike - spot))
        return OptionMarketPoint(
            target_time_years=target_tenor,
            option_time_years=option_tenor,
            expiry=expiry,
            representative_strike=representative_strike,
            volatility=statistics.median(volatilities),
            combined_spread_fraction=statistics.median(spreads),
            strikes_used=len(strikes),
            cache_hit=cache_hit,
        )


@dataclass(frozen=True)
class ResearchMarketBuildResult:
    market: EquityMarketTermStructure
    calibration: dict[str, Any]


class ResearchMarketDataService:
    def __init__(
        self,
        quote_service: Any,
        treasury_provider: TreasuryCurveProvider,
        dividend_provider: YFinanceDividendProvider,
        option_provider: YFinanceOptionProvider,
        clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        self.quote_service = quote_service
        self.treasury_provider = treasury_provider
        self.dividend_provider = dividend_provider
        self.option_provider = option_provider
        self._clock = clock

    def build_term_structure(
        self,
        symbol: str,
        underlier_type: str,
        maturity_years: float,
    ) -> ResearchMarketBuildResult:
        maturity = _normalize_maturity(maturity_years)
        valuation_time = _normalize_aware_time(self._clock(), "service clock")
        expected_type = underlier_type.strip().lower()
        if expected_type not in {"equity", "etf"}:
            raise ResearchMarketUnsupportedError(
                "research calibration currently supports equity or etf underliers"
            )

        quote_result, quote_age_seconds = self.quote_service.get_quote(symbol)
        quote = quote_result.quote
        if quote.currency != "USD":
            raise ResearchMarketUnsupportedError(
                "research calibration currently supports USD underliers only"
            )
        if quote.underlier_type and quote.underlier_type != expected_type:
            raise ResearchMarketUnsupportedError(
                "provider instrument type does not match requested underlier_type"
            )

        curve = self.treasury_provider.get_curve(valuation_time)
        dividend = self.dividend_provider.get_yield(
            symbol=quote.symbol,
            spot=quote.spot,
            valuation_time=valuation_time,
        )
        segment_ends = _segment_ends(maturity)
        option_points = self.option_provider.get_points(
            symbol=quote.symbol,
            spot=quote.spot,
            valuation_time=valuation_time,
            target_tenors=segment_ends,
        )

        segments = []
        previous_end = 0.0
        previous_integrated_rate = 0.0
        previous_total_variance = 0.0
        for end, option_point in zip(segment_ends, option_points):
            integrated_rate = curve.continuous_zero_rate(end) * end
            total_variance = option_point.volatility**2 * end
            interval = end - previous_end
            forward_rate = (integrated_rate - previous_integrated_rate) / interval
            forward_variance = (total_variance - previous_total_variance) / interval
            if forward_variance <= 0:
                raise ResearchMarketInvalidResponseError(
                    "option ATM total variance must increase with maturity"
                )
            try:
                segments.append(
                    EquityMarketSegment(
                        end_time_years=end,
                        risk_free_rate=forward_rate,
                        dividend_yield=dividend.continuous_yield,
                        volatility=math.sqrt(forward_variance),
                    )
                )
            except MarketDataValidationError as exc:
                raise ResearchMarketInvalidResponseError(
                    "calibrated segment is outside supported model bounds"
                ) from exc
            previous_end = end
            previous_integrated_rate = integrated_rate
            previous_total_variance = total_variance

        try:
            market = EquityMarketTermStructure(
                symbol=quote.symbol,
                underlier_type=expected_type,
                currency=quote.currency,
                valuation_time=valuation_time,
                market_data_time=quote.market_data_time,
                spot=quote.spot,
                segments=tuple(segments),
                calendar=quote.mic_code or quote.exchange or "WEEKDAYS",
                day_count="ACT/365F",
                source="us-treasury-par-proxy+yfinance-options-v1",
            )
        except MarketDataValidationError as exc:
            raise ResearchMarketInvalidResponseError(
                "calibrated term structure is invalid"
            ) from exc
        calibration = self._build_calibration(
            market=market,
            quote=quote,
            quote_cache_hit=quote_result.cache_hit,
            quote_age_seconds=quote_age_seconds,
            curve=curve,
            dividend=dividend,
            option_points=option_points,
        )
        return ResearchMarketBuildResult(market=market, calibration=calibration)

    @staticmethod
    def _build_calibration(
        market: EquityMarketTermStructure,
        quote: Any,
        quote_cache_hit: bool,
        quote_age_seconds: float,
        curve: TreasuryCurveObservation,
        dividend: DividendYieldObservation,
        option_points: Sequence[OptionMarketPoint],
    ) -> dict[str, Any]:
        quote_payload = {
            **quote.to_dict(),
            "cache_hit": quote_cache_hit,
            "quote_age_seconds": quote_age_seconds,
        }
        canonical_payload = {
            "calibration_version": EQUITY_RESEARCH_MARKET_VERSION,
            "term_structure_id": market.term_structure_id,
            "treasury": {
                "as_of_date": curve.as_of_date.isoformat(),
                "points": [point.to_dict() for point in curve.points],
            },
            "dividend": {
                key: value
                for key, value in dividend.to_dict().items()
                if key != "cache_hit"
            },
            "options": [
                {
                    key: value
                    for key, value in point.to_dict().items()
                    if key != "cache_hit"
                }
                for point in option_points
            ],
        }
        encoded = json.dumps(
            canonical_payload, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        calibration_id = f"sha256:{hashlib.sha256(encoded).hexdigest()}"
        option_expiry_gaps_days = [
            abs(point.option_time_years - point.target_time_years) * 365.0
            for point in option_points
        ]
        option_spreads = [point.combined_spread_fraction for point in option_points]
        minimum_strikes = min(point.strikes_used for point in option_points)
        treasury_age_days = (market.valuation_time.date() - curve.as_of_date).days
        quality_checks = [
            {
                "name": "quote_freshness",
                "passed": quote_age_seconds <= 7 * 24 * 60 * 60,
                "value": quote_age_seconds,
                "maximum": 7 * 24 * 60 * 60,
                "units": "seconds",
            },
            {
                "name": "treasury_freshness",
                "passed": treasury_age_days <= 10,
                "value": treasury_age_days,
                "maximum": 10,
                "units": "calendar_days",
            },
            {
                "name": "option_expiry_alignment",
                "passed": max(option_expiry_gaps_days) <= 62,
                "value": max(option_expiry_gaps_days),
                "maximum": 62,
                "units": "days",
            },
            {
                "name": "option_quote_spread",
                "passed": max(option_spreads) <= 0.10,
                "value": max(option_spreads),
                "maximum": 0.10,
                "units": "fraction_of_mid",
            },
            {
                "name": "option_strike_coverage",
                "passed": minimum_strikes >= 1,
                "value": minimum_strikes,
                "minimum": 1,
                "units": "strikes_per_tenor",
            },
        ]
        passed_checks = sum(bool(check["passed"]) for check in quality_checks)
        quality = {
            "status": (
                "research_ready"
                if passed_checks == len(quality_checks)
                else "review_required"
            ),
            "passed_checks": passed_checks,
            "total_checks": len(quality_checks),
            "checks": quality_checks,
            "freshness": {
                "quote_age_seconds": quote_age_seconds,
                "treasury_age_days": treasury_age_days,
            },
            "coverage": {
                "maturity_years": market.max_time_years,
                "model_segments": len(market.segments),
                "option_tenors": len(option_points),
                "maximum_expiry_gap_days": max(option_expiry_gaps_days),
                "minimum_strikes_per_tenor": minimum_strikes,
                "maximum_combined_spread_fraction": max(option_spreads),
            },
            "cache": {
                "quote": quote_cache_hit,
                "treasury": curve.cache_hit,
                "dividend": dividend.cache_hit,
                "option_tenors": sum(point.cache_hit for point in option_points),
                "option_tenor_count": len(option_points),
            },
            "scope": "research_only",
        }
        return {
            "calibration_version": EQUITY_RESEARCH_MARKET_VERSION,
            "calibration_id": calibration_id,
            "research_only": True,
            "term_structure_id": market.term_structure_id,
            "methods": {
                "spot": "yfinance-regular-session-1m-close",
                "risk_free_rate": TREASURY_CURVE_METHOD,
                "dividend_yield": DIVIDEND_METHOD,
                "volatility": OPTION_VOLATILITY_METHOD,
                "segment_mapping": "nearest-option-expiry-per-model-tenor-v1",
            },
            "quote": quote_payload,
            "treasury_curve": curve.to_dict(market.valuation_time),
            "dividend": dividend.to_dict(),
            "option_points": [point.to_dict() for point in option_points],
            "quality": quality,
            "warnings": [
                "Treasury CMT par yields are continuous zero-rate proxies, not a bootstrapped OIS curve.",
                "Dividend yield uses trailing cash distributions and is not a forward dividend forecast.",
                "Option volatility uses near-ATM Yahoo quotes and does not model volatility skew.",
                "This calibration is for personal research and is not an executable market quote.",
            ],
        }


def _normalize_aware_time(value: Any, label: str) -> datetime:
    if not isinstance(value, datetime):
        raise MarketDataConfigurationError(f"{label} must be a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise MarketDataConfigurationError(f"{label} must include a UTC offset")
    return value.astimezone(timezone.utc)


def _parse_optional_utc_time(raw_value: Optional[str]) -> Optional[datetime]:
    if not raw_value:
        return None
    parsed = datetime.fromisoformat(raw_value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _normalize_maturity(value: Any) -> float:
    if isinstance(value, bool):
        raise ResearchMarketUnsupportedError("maturity_years must be numeric")
    try:
        maturity = float(value)
    except (TypeError, ValueError) as exc:
        raise ResearchMarketUnsupportedError("maturity_years must be numeric") from exc
    if not math.isfinite(maturity) or maturity <= 0 or maturity > 30:
        raise ResearchMarketUnsupportedError(
            "maturity_years must be finite, > 0, and <= 30"
        )
    return maturity


def _segment_ends(maturity: float) -> tuple[float, ...]:
    ends = [tenor for tenor in RESEARCH_SEGMENT_TENORS if tenor < maturity - 1e-12]
    ends.append(maturity)
    return tuple(ends)


_RESEARCH_SERVICE: Optional[ResearchMarketDataService] = None
_RESEARCH_SERVICE_LOCK = Lock()


def clear_research_market_data_service() -> None:
    global _RESEARCH_SERVICE
    with _RESEARCH_SERVICE_LOCK:
        _RESEARCH_SERVICE = None


def get_research_market_data_service() -> ResearchMarketDataService:
    global _RESEARCH_SERVICE
    with _RESEARCH_SERVICE_LOCK:
        if _RESEARCH_SERVICE is None:
            _RESEARCH_SERVICE = ResearchMarketDataService(
                quote_service=get_live_market_data_service(),
                treasury_provider=TreasuryCurveProvider(
                    TreasuryCurveSettings.from_environment()
                ),
                dividend_provider=YFinanceDividendProvider(),
                option_provider=YFinanceOptionProvider(),
            )
        return _RESEARCH_SERVICE


def get_research_market_data_status() -> dict[str, Any]:
    return {
        "enabled": True,
        "configured": True,
        "calibration_version": EQUITY_RESEARCH_MARKET_VERSION,
        "currency": "USD",
        "underlier_types": ["equity", "etf"],
        "sources": ["U.S. Department of the Treasury", "yfinance"],
        "credentials_required": False,
        "research_only": True,
    }
