from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest
from yfinance.exceptions import YFRateLimitError

from app.services.live_market_data import (
    LiveMarketDataService,
    MarketDataConfigurationError,
    MarketDataInvalidResponseError,
    MarketDataNotFoundError,
    MarketDataRateLimitError,
    MarketDataStaleError,
    MarketDataTypeMismatchError,
    YFinanceQuoteProvider,
    YFinanceSettings,
)


QUOTE_TIME = datetime(2026, 7, 13, 12, 0, tzinfo=timezone.utc)


def quote_history(close=620.25, timestamp=QUOTE_TIME):
    return pd.DataFrame(
        {"Close": [close]},
        index=pd.DatetimeIndex([timestamp], name="Datetime"),
    )


def quote_metadata(**overrides):
    metadata = {
        "symbol": "SPY",
        "currency": "USD",
        "exchangeName": "PCX",
        "fullExchangeName": "NYSE Arca",
        "instrumentType": "ETF",
        "exchangeTimezoneName": "America/New_York",
        "exchangeDataDelayedBy": 0,
    }
    metadata.update(overrides)
    return metadata


class FakeTicker:
    def __init__(self, history=None, metadata=None, error=None):
        self.history_result = history if history is not None else quote_history()
        self.metadata = metadata if metadata is not None else quote_metadata()
        self.error = error
        self.history_calls = []

    def history(self, **kwargs):
        self.history_calls.append(kwargs)
        if self.error:
            raise self.error
        return self.history_result

    def get_history_metadata(self):
        return self.metadata


class FakeTickerFactory:
    def __init__(self, *tickers):
        self.tickers = list(tickers)
        self.symbols = []

    def __call__(self, symbol):
        self.symbols.append(symbol)
        return self.tickers.pop(0)


def make_settings(**overrides):
    values = {
        "cache_ttl_seconds": 60.0,
        "max_quote_age_seconds": 7 * 24 * 60 * 60,
        "max_attempts": 2,
    }
    values.update(overrides)
    return YFinanceSettings(**values)


def test_settings_validate_and_parse_optional_environment_controls():
    with pytest.raises(MarketDataConfigurationError, match="timeout"):
        YFinanceSettings(request_timeout_seconds=0)
    with pytest.raises(MarketDataConfigurationError, match="max_attempts"):
        YFinanceSettings(max_attempts=6)
    with pytest.raises(MarketDataConfigurationError, match="must be numeric"):
        YFinanceSettings.from_environment(
            {"MARKET_DATA_CACHE_TTL_SECONDS": "not-a-number"}
        )

    settings = YFinanceSettings.from_environment(
        {
            "MARKET_DATA_REQUEST_TIMEOUT_SECONDS": "3",
            "MARKET_DATA_CACHE_TTL_SECONDS": "30",
            "MARKET_DATA_MAX_QUOTE_AGE_SECONDS": "3600",
            "MARKET_DATA_MAX_ATTEMPTS": "1",
        }
    )

    assert settings.request_timeout_seconds == 3.0
    assert settings.cache_ttl_seconds == 30.0
    assert settings.max_quote_age_seconds == 3600.0
    assert settings.max_attempts == 1


def test_provider_fetches_timestamped_regular_session_bar():
    ticker = FakeTicker()
    factory = FakeTickerFactory(ticker)
    provider = YFinanceQuoteProvider(make_settings(), ticker_factory=factory)

    result = provider.get_quote("SPY")

    assert factory.symbols == ["SPY"]
    request = ticker.history_calls[0]
    assert request["period"] == "5d"
    assert request["interval"] == "1m"
    assert request["prepost"] is False
    assert request["auto_adjust"] is False
    assert request["raise_errors"] is True
    assert result.cache_hit is False
    assert result.quote.spot == 620.25
    assert result.quote.market_data_time == QUOTE_TIME
    assert result.quote.currency == "USD"
    assert result.quote.exchange == "NYSE Arca"
    assert result.quote.instrument_type == "ETF"
    assert result.quote.underlier_type == "etf"
    assert result.quote.bar_interval == "1m"
    assert result.quote.data_delay_seconds == 0
    assert result.quote.to_dict()["research_only"] is True


def test_provider_cache_reuses_then_refreshes_quote():
    now = [0.0]
    first = FakeTicker(history=quote_history(close=100.0))
    second = FakeTicker(history=quote_history(close=101.0))
    factory = FakeTickerFactory(first, second)
    provider = YFinanceQuoteProvider(
        make_settings(cache_ttl_seconds=10.0),
        ticker_factory=factory,
        monotonic_clock=lambda: now[0],
    )

    assert provider.get_quote("SPY").quote.spot == 100.0
    now[0] = 5.0
    cached = provider.get_quote("spy")
    assert cached.cache_hit is True
    assert cached.quote.spot == 100.0
    now[0] = 11.0
    refreshed = provider.get_quote("SPY")
    assert refreshed.cache_hit is False
    assert refreshed.quote.spot == 101.0
    assert factory.symbols == ["SPY", "SPY"]


def test_provider_force_refresh_bypasses_cache():
    factory = FakeTickerFactory(
        FakeTicker(history=quote_history(close=100.0)),
        FakeTicker(history=quote_history(close=102.0)),
    )
    provider = YFinanceQuoteProvider(make_settings(), ticker_factory=factory)

    provider.get_quote("SPY")
    refreshed = provider.get_quote("SPY", force_refresh=True)

    assert refreshed.cache_hit is False
    assert refreshed.quote.spot == 102.0


def test_provider_retries_transient_failure():
    delays = []
    factory = FakeTickerFactory(
        FakeTicker(error=RuntimeError("upstream detail must not escape")),
        FakeTicker(),
    )
    provider = YFinanceQuoteProvider(
        make_settings(max_attempts=2),
        ticker_factory=factory,
        sleeper=delays.append,
    )

    result = provider.get_quote("SPY")

    assert result.quote.spot == 620.25
    assert delays == [0.25]


def test_provider_sanitizes_rate_limit_failure():
    provider = YFinanceQuoteProvider(
        make_settings(max_attempts=1),
        ticker_factory=FakeTickerFactory(FakeTicker(error=YFRateLimitError())),
    )

    with pytest.raises(MarketDataRateLimitError, match="rate limit") as error:
        provider.get_quote("SPY")

    assert "crumb" not in str(error.value)


def test_provider_rejects_empty_or_malformed_data():
    empty = pd.DataFrame({"Close": pd.Series(dtype=float)})
    empty_provider = YFinanceQuoteProvider(
        make_settings(), ticker_factory=FakeTickerFactory(FakeTicker(history=empty))
    )
    with pytest.raises(MarketDataNotFoundError):
        empty_provider.get_quote("MISSING")

    bad_currency_provider = YFinanceQuoteProvider(
        make_settings(),
        ticker_factory=FakeTickerFactory(
            FakeTicker(metadata=quote_metadata(currency="US"))
        ),
    )
    with pytest.raises(MarketDataInvalidResponseError, match="currency"):
        bad_currency_provider.get_quote("SPY")

    bad_price_provider = YFinanceQuoteProvider(
        make_settings(),
        ticker_factory=FakeTickerFactory(FakeTicker(history=quote_history(close=-1.0))),
    )
    with pytest.raises(MarketDataInvalidResponseError, match="close price"):
        bad_price_provider.get_quote("SPY")


def test_provider_localizes_naive_exchange_timestamp():
    local_time = datetime(2026, 7, 13, 9, 30)
    provider = YFinanceQuoteProvider(
        make_settings(),
        ticker_factory=FakeTickerFactory(
            FakeTicker(history=quote_history(timestamp=local_time))
        ),
    )

    quote = provider.get_quote("SPY").quote

    assert quote.market_data_time == datetime(2026, 7, 13, 13, 30, tzinfo=timezone.utc)


@pytest.mark.parametrize(
    ("provider_currency", "provider_spot", "expected_currency", "expected_spot"),
    [
        ("GBp", 1234.0, "GBP", 12.34),
        ("ZAc", 3456.0, "ZAR", 34.56),
        ("ILA", 789.0, "ILS", 7.89),
    ],
)
def test_provider_normalizes_minor_quote_units(
    provider_currency, provider_spot, expected_currency, expected_spot
):
    provider = YFinanceQuoteProvider(
        make_settings(),
        ticker_factory=FakeTickerFactory(
            FakeTicker(
                history=quote_history(close=provider_spot),
                metadata=quote_metadata(currency=provider_currency),
            )
        ),
    )

    quote = provider.get_quote("TEST").quote

    assert quote.currency == expected_currency
    assert quote.spot == pytest.approx(expected_spot)
    assert quote.provider_currency == provider_currency
    assert quote.provider_spot == provider_spot


def test_market_data_service_builds_snapshot_and_records_sources():
    provider = YFinanceQuoteProvider(
        make_settings(), ticker_factory=FakeTickerFactory(FakeTicker())
    )
    service = LiveMarketDataService(
        provider, clock=lambda: QUOTE_TIME + timedelta(seconds=2)
    )

    result = service.get_snapshot(
        symbol="SPY",
        underlier_type="etf",
        risk_free_rate=0.04,
        dividend_yield=0.012,
        volatility=0.2,
    )

    assert result.snapshot.spot == 620.25
    assert result.snapshot.calendar == "NYSE ARCA"
    assert result.snapshot.source == "yfinance:1m-close+request-model-inputs"
    assert result.quote_age_seconds == 2.0
    assert result.metadata()["research_only"] is True
    assert result.metadata()["input_sources"]["spot"] == "yfinance"
    assert result.metadata()["input_sources"]["underlier_type"] == "yfinance"
    assert result.metadata()["input_sources"]["volatility"] == "request"


def test_market_data_service_uses_requested_type_when_metadata_omits_it():
    provider = YFinanceQuoteProvider(
        make_settings(),
        ticker_factory=FakeTickerFactory(
            FakeTicker(metadata=quote_metadata(instrumentType=None))
        ),
    )
    service = LiveMarketDataService(provider, clock=lambda: QUOTE_TIME)

    result = service.get_snapshot(
        symbol="SPY",
        underlier_type="etf",
        risk_free_rate=0.04,
        dividend_yield=0.012,
        volatility=0.2,
    )

    assert result.snapshot.underlier_type == "etf"
    assert result.metadata()["input_sources"]["underlier_type"] == "request"


def test_market_data_service_rejects_stale_future_and_mismatched_bars():
    stale_provider = YFinanceQuoteProvider(
        make_settings(max_quote_age_seconds=1.0),
        ticker_factory=FakeTickerFactory(FakeTicker()),
    )
    stale_service = LiveMarketDataService(
        stale_provider, clock=lambda: QUOTE_TIME + timedelta(seconds=2)
    )
    with pytest.raises(MarketDataStaleError):
        stale_service.get_quote("SPY")

    future_provider = YFinanceQuoteProvider(
        make_settings(future_tolerance_seconds=1.0),
        ticker_factory=FakeTickerFactory(
            FakeTicker(
                history=quote_history(timestamp=QUOTE_TIME + timedelta(seconds=2))
            )
        ),
    )
    future_service = LiveMarketDataService(future_provider, clock=lambda: QUOTE_TIME)
    with pytest.raises(MarketDataInvalidResponseError, match="future-dated"):
        future_service.get_quote("SPY")

    mismatch_provider = YFinanceQuoteProvider(
        make_settings(),
        ticker_factory=FakeTickerFactory(
            FakeTicker(metadata=quote_metadata(instrumentType="EQUITY"))
        ),
    )
    mismatch_service = LiveMarketDataService(
        mismatch_provider, clock=lambda: QUOTE_TIME
    )
    with pytest.raises(MarketDataTypeMismatchError):
        mismatch_service.get_snapshot(
            symbol="SPY",
            underlier_type="etf",
            risk_free_rate=0.04,
            dividend_yield=0.012,
            volatility=0.2,
        )
