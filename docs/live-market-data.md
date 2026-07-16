# Research Market Data

Neural Pricer 0.4 uses yfinance for credential-free research and hobby market
data. The adapter remains behind `LiveMarketDataService`; payoff and Monte Carlo
modules never import yfinance or call Yahoo Finance directly.

No account, API key, `.env` file, or provider selection is required. Installing
the API dependency group installs the pinned yfinance version and SciPy, which
yfinance uses when its enabled price-repair path checks for sporadic currency
unit errors.

Official references:

- [yfinance documentation](https://ranaroussi.github.io/yfinance/)
- [yfinance source and usage disclaimer](https://github.com/ranaroussi/yfinance)

## What the adapter retrieves

For each symbol, the adapter requests five days of one-minute history and uses
the latest non-empty regular-session `Close`. This makes the price and timestamp
come from the same observable bar instead of combining an undated current-price
field with separate history.

The normalized quote contains:

- the latest available one-minute close and bar timestamp;
- currency, including normalization of supported minor currency units;
- exchange and instrument-type metadata when Yahoo supplies them;
- Yahoo's reported data delay when present;
- cache and data-age metadata; and
- an explicit `research_only` flag.

Yahoo symbol syntax identifies the listing. Examples include `AAPL`, `SPY`, and
`^SPX`; there is no separate exchange parameter.

The Phoenix market-pricing route accepts equity, ETF, and equity-index
underliers because those match the current equity-style GBM model. yfinance can
retrieve other symbol classes, but FX, rates, commodities, futures, and crypto
must not be pushed through this model without their own carry, calendar, quote,
and model conventions.

## Honest model scope

yfinance supplies the price, bar time, currency, exchange, and available type
metadata. The `/price/market` request still supplies:

- flat discount rate;
- flat dividend yield; and
- constant volatility.

The response records field-level provenance under `market_data.input_sources`.
This is therefore research pricing from a recent market bar, not a calibrated
live market. The Phase 6
[`equity-research-market-v1`](equity-research-market-v1.md) builder is a separate
composition layer: it combines the normalized quote with an official Treasury
par-yield proxy, trailing equity/ETF cash distributions, and near-ATM yfinance
option volatility estimates. It remains research-only and does not claim to be
an OIS curve, forward dividend forecast, or full volatility surface.

## Endpoints

Provider status:

```text
GET /api/v1/market-data/status
```

Fetch a normalized latest bar:

```text
GET /api/v1/market-data/quote?symbol=SPY
```

Price Phoenix from a server-fetched market bar:

```text
POST /api/v1/products/phoenix/price/market
```

Build or price with the Phase 6 research term structure:

```text
POST /api/v1/market-data/research-term-structure
POST /api/v1/products/phoenix/price/research-market
```

Phase 7 can freeze the calibrated structure for paired scenarios and
finite-difference risk through the Phoenix `scenario/research-market` and
`risk/research-market` endpoints. See
[`equity-risk-analytics-v1`](equity-risk-analytics-v1.md).

Example body:

```json
{
  "market": {
    "symbol": "SPY",
    "underlier_type": "etf",
    "risk_free_rate": 0.04,
    "dividend_yield": 0.012,
    "volatility": 0.2,
    "day_count": "ACT/365F"
  },
  "terms": {
    "maturity_years": 1.0,
    "autocall_barrier_frac": 1.05,
    "coupon_barrier_frac": 1.0,
    "coupon_rate": 0.02,
    "knock_in_frac": 0.7,
    "obs_count": 6
  },
  "n_paths": 2000
}
```

## Operational behavior

- Requests have a bounded timeout and two attempts by default.
- Rate-limit, missing-symbol, malformed-data, and availability failures use
  sanitized application errors.
- Quotes are cached per symbol for 60 seconds in a bounded, thread-safe LRU.
- Future-dated, stale, malformed, non-positive, and type-mismatched bars are
  rejected before simulation.
- The default maximum age is seven days so the previous regular close remains
  usable over weekends and holidays. Every response exposes the actual age.
- Provider minor-unit quotes (`GBp`, `ZAc`, and `ILA`) are converted to `GBP`,
  `ZAR`, and `ILS`; the raw unit, raw price, and conversion factor are retained.
- The manual dated-snapshot endpoint remains available for fully reproducible
  experiments.

Advanced users can override `MARKET_DATA_REQUEST_TIMEOUT_SECONDS`,
`MARKET_DATA_CACHE_TTL_SECONDS`, `MARKET_DATA_MAX_QUOTE_AGE_SECONDS`,
`MARKET_DATA_FUTURE_TOLERANCE_SECONDS`, `MARKET_DATA_MAX_ATTEMPTS`, and
`MARKET_DATA_MAX_CACHE_ENTRIES`. No setting is required for normal use.

The cache is process-local. It is deliberately simple for a hobby application.

## Usage boundary

yfinance is open-source software, but it is not affiliated with or endorsed by
Yahoo. Its own project documentation says it is intended for research and
educational purposes and that Yahoo Finance data is intended for personal use.
Do not treat this adapter as an authoritative valuation feed or expose it as a
commercial redistribution service.
