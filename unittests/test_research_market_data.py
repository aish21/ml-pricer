import math
from collections import namedtuple
from datetime import date, datetime, timezone

import pandas as pd
import pytest
import requests

from app.services.live_market_data import (
    LiveEquityQuote,
    MarketDataConfigurationError,
    MarketDataUpstreamError,
    QuoteFetchResult,
)
from app.services.research_market_data import (
    DividendYieldObservation,
    OptionMarketPoint,
    ResearchMarketDataService,
    ResearchMarketInvalidResponseError,
    ResearchMarketUnsupportedError,
    TreasuryCurveObservation,
    TreasuryCurveProvider,
    TreasuryCurveSettings,
    TreasuryParYieldPoint,
    YFinanceDividendProvider,
    YFinanceDividendSettings,
    YFinanceOptionProvider,
    YFinanceOptionSettings,
)


VALUATION_TIME = datetime(2026, 7, 15, 14, 0, tzinfo=timezone.utc)
OptionChain = namedtuple("OptionChain", "calls puts")


def treasury_xml(*observations):
    entries = []
    for identifier, as_of, rate_percent in observations:
        fields = "".join(
            f'<d:{name} m:type="Edm.Double">{rate_percent}</d:{name}>'
            for name in (
                "BC_1MONTH",
                "BC_3MONTH",
                "BC_6MONTH",
                "BC_1YEAR",
                "BC_2YEAR",
                "BC_5YEAR",
                "BC_10YEAR",
                "BC_30YEAR",
            )
        )
        entries.append(
            f"""
            <entry>
              <content type="application/xml">
                <m:properties>
                  <d:Id m:type="Edm.Int32">{identifier}</d:Id>
                  <d:NEW_DATE m:type="Edm.DateTime">{as_of}T00:00:00</d:NEW_DATE>
                  {fields}
                </m:properties>
              </content>
            </entry>
            """
        )
    return (
        """<?xml version="1.0" encoding="utf-8"?>
        <feed xmlns:d="http://schemas.microsoft.com/ado/2007/08/dataservices"
              xmlns:m="http://schemas.microsoft.com/ado/2007/08/dataservices/metadata"
              xmlns="http://www.w3.org/2005/Atom">
          <updated>2026-07-15T12:00:00Z</updated>
        """
        + "".join(entries)
        + "</feed>"
    ).encode()


class FakeResponse:
    def __init__(self, content, error=None):
        self.content = content
        self.error = error

    def raise_for_status(self):
        if self.error:
            raise self.error


def make_curve(rate=0.04, cache_hit=False):
    return TreasuryCurveObservation(
        as_of_date=date(2026, 7, 14),
        feed_updated_time=datetime(2026, 7, 14, 22, 0, tzinfo=timezone.utc),
        points=(
            TreasuryParYieldPoint(1.0 / 12.0, rate),
            TreasuryParYieldPoint(0.25, rate),
            TreasuryParYieldPoint(0.5, rate),
            TreasuryParYieldPoint(1.0, rate),
            TreasuryParYieldPoint(2.0, rate),
            TreasuryParYieldPoint(30.0, rate),
        ),
        source_url="https://home.treasury.gov/test-feed",
        cache_hit=cache_hit,
    )


def make_option_chain(spot, tenor, rate, dividend_yield, volatility):
    call_rows = []
    put_rows = []
    discount = math.exp(-rate * tenor)
    prepaid_spot = spot * math.exp(-dividend_yield * tenor)
    for strike in (spot - 1.0, spot, spot + 1.0):
        put_mid = 5.0
        call_mid = put_mid + prepaid_spot - strike * discount
        call_rows.append(
            {
                "strike": strike,
                "bid": call_mid - 0.05,
                "ask": call_mid + 0.05,
                "impliedVolatility": volatility,
            }
        )
        put_rows.append(
            {
                "strike": strike,
                "bid": put_mid - 0.05,
                "ask": put_mid + 0.05,
                "impliedVolatility": volatility,
            }
        )
    return OptionChain(pd.DataFrame(call_rows), pd.DataFrame(put_rows))


class FakeTicker:
    def __init__(self, expiries, chains):
        self.options = tuple(expiries)
        self.chains = chains
        self.chain_calls = []

    def option_chain(self, expiry):
        self.chain_calls.append(expiry)
        return self.chains[expiry]


class FakeDividendTicker:
    def __init__(self, history):
        self.history_result = history
        self.history_calls = []

    def history(self, **kwargs):
        self.history_calls.append(kwargs)
        return self.history_result


class FakeQuoteService:
    def __init__(self, currency="USD", underlier_type="etf"):
        self.quote = LiveEquityQuote(
            provider_name="yfinance",
            symbol="SPY",
            spot=100.0,
            currency=currency,
            market_data_time=datetime(2026, 7, 15, 13, 59, tzinfo=timezone.utc),
            exchange="NYSE Arca",
            mic_code="ARCX",
            instrument_type="ETF",
            underlier_type=underlier_type,
            provider_spot=100.0,
            provider_currency=currency,
            unit_conversion_factor=1.0,
            bar_interval="1m",
            data_delay_seconds=0,
        )

    def get_quote(self, symbol):
        assert symbol == "SPY"
        return QuoteFetchResult(self.quote, cache_hit=False), 60.0


class FakeTreasuryProvider:
    def __init__(self, curve=None):
        self.curve = curve or make_curve()

    def get_curve(self, valuation_time):
        assert valuation_time == VALUATION_TIME
        return self.curve


class FakeOptionProvider:
    def __init__(self, cache_hit=False, volatilities=None):
        self.cache_hit = cache_hit
        self.volatilities = volatilities
        self.targets = None

    def get_points(self, symbol, spot, valuation_time, target_tenors):
        assert symbol == "SPY"
        assert spot == 100.0
        assert valuation_time == VALUATION_TIME
        self.targets = tuple(target_tenors)
        return tuple(
            OptionMarketPoint(
                target_time_years=target,
                option_time_years=target,
                expiry="2027-07-16",
                representative_strike=100.0,
                volatility=(
                    self.volatilities[index] if self.volatilities is not None else 0.20
                ),
                combined_spread_fraction=0.002,
                strikes_used=3,
                cache_hit=self.cache_hit,
            )
            for index, target in enumerate(target_tenors)
        )


class FakeDividendProvider:
    def __init__(self, cache_hit=False, continuous_yield=0.01):
        self.cache_hit = cache_hit
        self.continuous_yield = continuous_yield

    def get_yield(self, symbol, spot, valuation_time):
        assert symbol == "SPY"
        assert spot == 100.0
        assert valuation_time == VALUATION_TIME
        return DividendYieldObservation(
            period_start=date(2025, 7, 15),
            period_end=date(2026, 7, 15),
            cash_distributions=math.expm1(self.continuous_yield) * spot,
            spot=spot,
            continuous_yield=self.continuous_yield,
            observations=4,
            cache_hit=self.cache_hit,
        )


def test_treasury_provider_selects_latest_non_lookahead_curve_and_caches():
    calls = []

    def http_get(url, **kwargs):
        calls.append((url, kwargs))
        return FakeResponse(
            treasury_xml(
                (1, "2026-07-14", 4.0),
                (2, "2026-07-16", 5.0),
            )
        )

    provider = TreasuryCurveProvider(
        TreasuryCurveSettings(cache_ttl_seconds=60, max_curve_age_days=10),
        http_get=http_get,
        monotonic_clock=lambda: 10.0,
    )

    first = provider.get_curve(VALUATION_TIME)
    second = provider.get_curve(VALUATION_TIME)

    assert first.as_of_date == date(2026, 7, 14)
    assert first.cache_hit is False
    assert second.cache_hit is True
    assert len(calls) == 1
    expected_rate = 2.0 * math.log1p(0.04 / 2.0)
    assert first.continuous_zero_rate(1.0) == pytest.approx(expected_rate)
    assert first.discount_factor(1.0) == pytest.approx(math.exp(-expected_rate))


def test_treasury_provider_retries_transport_errors_and_sanitizes_failure():
    attempts = []

    def http_get(*args, **kwargs):
        attempts.append(1)
        raise requests.Timeout("upstream details")

    provider = TreasuryCurveProvider(
        TreasuryCurveSettings(max_attempts=2),
        http_get=http_get,
        sleeper=lambda delay: None,
    )

    with pytest.raises(MarketDataUpstreamError, match="Treasury curve") as error:
        provider.get_curve(VALUATION_TIME)

    assert len(attempts) == 2
    assert "upstream details" not in str(error.value)


def test_treasury_provider_rejects_stale_or_invalid_feed():
    stale = TreasuryCurveProvider(
        TreasuryCurveSettings(max_curve_age_days=1),
        http_get=lambda *args, **kwargs: FakeResponse(
            treasury_xml((1, "2026-07-10", 4.0))
        ),
    )
    with pytest.raises(ResearchMarketInvalidResponseError, match="older"):
        stale.get_curve(VALUATION_TIME)

    invalid = TreasuryCurveProvider(
        http_get=lambda *args, **kwargs: FakeResponse(b"not xml")
    )
    with pytest.raises(ResearchMarketInvalidResponseError, match="invalid XML"):
        invalid.get_curve(VALUATION_TIME)

    prohibited = TreasuryCurveProvider(
        http_get=lambda *args, **kwargs: FakeResponse(
            b'<?xml version="1.0"?><!DOCTYPE feed><feed />'
        )
    )
    with pytest.raises(ResearchMarketInvalidResponseError, match="prohibited"):
        prohibited.get_curve(VALUATION_TIME)


def test_dividend_provider_builds_trailing_cash_yield_and_caches_cash_history():
    history = pd.DataFrame(
        {
            "Close": [99.0, 100.0, 101.0],
            "Dividends": [1.0, 0.0, 1.0],
        },
        index=pd.to_datetime(
            ["2025-07-16T20:00:00Z", "2026-01-02T21:00:00Z", "2026-07-14T20:00:00Z"]
        ),
    )
    ticker = FakeDividendTicker(history)
    provider = YFinanceDividendProvider(
        YFinanceDividendSettings(cache_ttl_seconds=60),
        ticker_factory=lambda symbol: ticker,
        monotonic_clock=lambda: 10.0,
    )

    first = provider.get_yield("SPY", 100.0, VALUATION_TIME)
    cached = provider.get_yield("SPY", 200.0, VALUATION_TIME)

    assert first.cash_distributions == 2.0
    assert first.continuous_yield == pytest.approx(math.log1p(0.02))
    assert first.observations == 2
    assert first.cache_hit is False
    assert cached.continuous_yield == pytest.approx(math.log1p(0.01))
    assert cached.cache_hit is True
    assert len(ticker.history_calls) == 1
    assert ticker.history_calls[0]["period"] == "1y"
    assert ticker.history_calls[0]["actions"] is True


def test_dividend_settings_reject_unbounded_cache():
    with pytest.raises(MarketDataConfigurationError, match="cache entries"):
        YFinanceDividendSettings(max_cache_entries=0)


def test_dividend_provider_rejects_invalid_cash_distributions():
    history = pd.DataFrame(
        {"Dividends": [-1.0]},
        index=pd.to_datetime(["2026-07-01T20:00:00Z"]),
    )
    provider = YFinanceDividendProvider(
        ticker_factory=lambda symbol: FakeDividendTicker(history)
    )

    with pytest.raises(
        ResearchMarketInvalidResponseError,
        match="invalid cash distributions",
    ):
        provider.get_yield("SPY", 100.0, VALUATION_TIME)


def test_dividend_provider_rejects_partial_trailing_history():
    history = pd.DataFrame(
        {"Dividends": [0.0, 0.0]},
        index=pd.to_datetime(["2026-06-01T20:00:00Z", "2026-07-14T20:00:00Z"]),
    )
    provider = YFinanceDividendProvider(
        ticker_factory=lambda symbol: FakeDividendTicker(history)
    )

    with pytest.raises(
        ResearchMarketInvalidResponseError,
        match="does not cover the trailing year",
    ):
        provider.get_yield("SPY", 100.0, VALUATION_TIME)


def test_option_provider_builds_atm_vol_and_put_call_parity_carry():
    expiry = "2026-10-16"
    expiry_time = datetime(2026, 10, 16, 20, 0, tzinfo=timezone.utc)
    tenor = (expiry_time - VALUATION_TIME).total_seconds() / (365.0 * 86400.0)
    curve = make_curve()
    zero_rate = curve.continuous_zero_rate(tenor)
    ticker = FakeTicker(
        [expiry],
        {
            expiry: make_option_chain(
                spot=100.0,
                tenor=tenor,
                rate=zero_rate,
                dividend_yield=0.02,
                volatility=0.25,
            )
        },
    )
    provider = YFinanceOptionProvider(
        YFinanceOptionSettings(cache_ttl_seconds=60),
        ticker_factory=lambda symbol: ticker,
        monotonic_clock=lambda: 10.0,
    )

    first = provider.get_points("SPY", 100.0, VALUATION_TIME, (0.25,))[0]
    second = provider.get_points("SPY", 100.0, VALUATION_TIME, (0.25,))[0]

    assert first.expiry == expiry
    assert first.volatility == pytest.approx(0.25)
    assert first.strikes_used == 3
    assert first.cache_hit is False
    assert second.cache_hit is True
    assert ticker.chain_calls == [expiry]


def test_option_provider_rejects_uncovered_maturity_without_wrapping_error():
    expiry = "2026-08-14"
    ticker = FakeTicker([expiry], {expiry: OptionChain(pd.DataFrame(), pd.DataFrame())})
    provider = YFinanceOptionProvider(ticker_factory=lambda symbol: ticker)

    with pytest.raises(
        ResearchMarketUnsupportedError,
        match="do not cover the requested maturity",
    ):
        provider.get_points("SPY", 100.0, VALUATION_TIME, (1.0,))


def test_research_service_builds_dated_reproducible_term_structure():
    option_provider = FakeOptionProvider()
    service = ResearchMarketDataService(
        quote_service=FakeQuoteService(),
        treasury_provider=FakeTreasuryProvider(),
        dividend_provider=FakeDividendProvider(),
        option_provider=option_provider,
        clock=lambda: VALUATION_TIME,
    )

    result = service.build_term_structure("SPY", "etf", 1.0)

    assert option_provider.targets == (1.0 / 12.0, 0.25, 0.5, 1.0)
    assert result.market.symbol == "SPY"
    assert result.market.currency == "USD"
    assert result.market.max_time_years == 1.0
    assert result.market.discount_factor(1.0) == pytest.approx(
        make_curve().discount_factor(1.0)
    )
    assert result.calibration["calibration_version"] == "equity-research-market-v1"
    assert result.calibration["term_structure_id"] == result.market.term_structure_id
    assert result.calibration["calibration_id"].startswith("sha256:")
    assert result.calibration["research_only"] is True
    assert len(result.calibration["warnings"]) == 4


def test_research_service_converts_term_volatility_to_forward_buckets():
    term_volatilities = (0.20, 0.21, 0.22, 0.23)
    service = ResearchMarketDataService(
        quote_service=FakeQuoteService(),
        treasury_provider=FakeTreasuryProvider(),
        dividend_provider=FakeDividendProvider(continuous_yield=0.015),
        option_provider=FakeOptionProvider(
            volatilities=term_volatilities,
        ),
        clock=lambda: VALUATION_TIME,
    )

    result = service.build_term_structure("SPY", "etf", 1.0)

    for end, volatility in zip(
        (1.0 / 12.0, 0.25, 0.5, 1.0),
        term_volatilities,
    ):
        assert result.market.integrated_dividend_yield(0.0, end) == pytest.approx(
            0.015 * end
        )
        assert result.market.integrated_variance(0.0, end) == pytest.approx(
            volatility**2 * end
        )


def test_research_service_rejects_decreasing_atm_total_variance():
    service = ResearchMarketDataService(
        quote_service=FakeQuoteService(),
        treasury_provider=FakeTreasuryProvider(),
        dividend_provider=FakeDividendProvider(),
        option_provider=FakeOptionProvider(
            volatilities=(0.30, 0.05, 0.20, 0.20),
        ),
        clock=lambda: VALUATION_TIME,
    )

    with pytest.raises(
        ResearchMarketInvalidResponseError,
        match="total variance must increase",
    ):
        service.build_term_structure("SPY", "etf", 1.0)


def test_research_service_classifies_out_of_bounds_calibration_as_upstream_data():
    service = ResearchMarketDataService(
        quote_service=FakeQuoteService(),
        treasury_provider=FakeTreasuryProvider(),
        dividend_provider=FakeDividendProvider(continuous_yield=2.0),
        option_provider=FakeOptionProvider(),
        clock=lambda: VALUATION_TIME,
    )

    with pytest.raises(
        ResearchMarketInvalidResponseError,
        match="outside supported model bounds",
    ):
        service.build_term_structure("SPY", "etf", 1.0)


def test_research_calibration_identity_ignores_operational_cache_hits():
    market_service = ResearchMarketDataService(
        quote_service=FakeQuoteService(),
        treasury_provider=FakeTreasuryProvider(),
        dividend_provider=FakeDividendProvider(cache_hit=False),
        option_provider=FakeOptionProvider(cache_hit=False),
        clock=lambda: VALUATION_TIME,
    )
    cached_service = ResearchMarketDataService(
        quote_service=FakeQuoteService(),
        treasury_provider=FakeTreasuryProvider(make_curve(cache_hit=True)),
        dividend_provider=FakeDividendProvider(cache_hit=True),
        option_provider=FakeOptionProvider(cache_hit=True),
        clock=lambda: VALUATION_TIME,
    )

    first = market_service.build_term_structure("SPY", "etf", 1.0)
    cached = cached_service.build_term_structure("SPY", "etf", 1.0)

    assert first.market.term_structure_id == cached.market.term_structure_id
    assert first.calibration["calibration_id"] == cached.calibration["calibration_id"]


def test_research_service_rejects_non_usd_or_type_mismatch():
    unsupported_index_service = ResearchMarketDataService(
        quote_service=FakeQuoteService(underlier_type="index"),
        treasury_provider=FakeTreasuryProvider(),
        dividend_provider=FakeDividendProvider(),
        option_provider=FakeOptionProvider(),
        clock=lambda: VALUATION_TIME,
    )
    with pytest.raises(ResearchMarketUnsupportedError, match="equity or etf"):
        unsupported_index_service.build_term_structure("SPY", "index", 1.0)

    non_usd_service = ResearchMarketDataService(
        quote_service=FakeQuoteService(currency="EUR"),
        treasury_provider=FakeTreasuryProvider(),
        dividend_provider=FakeDividendProvider(),
        option_provider=FakeOptionProvider(),
        clock=lambda: VALUATION_TIME,
    )
    with pytest.raises(ResearchMarketUnsupportedError, match="USD"):
        non_usd_service.build_term_structure("SPY", "etf", 1.0)

    mismatch_service = ResearchMarketDataService(
        quote_service=FakeQuoteService(underlier_type="equity"),
        treasury_provider=FakeTreasuryProvider(),
        dividend_provider=FakeDividendProvider(),
        option_provider=FakeOptionProvider(),
        clock=lambda: VALUATION_TIME,
    )
    with pytest.raises(ResearchMarketUnsupportedError, match="instrument type"):
        mismatch_service.build_term_structure("SPY", "etf", 1.0)
