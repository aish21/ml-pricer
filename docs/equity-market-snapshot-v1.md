# Equity Market Snapshot v1

`equity-market-snapshot-v1` is the immutable input boundary between a market
data provider and a product pricer. It supports one equity-like underlier per
snapshot: an equity, ETF, or equity index. The symbol is provider-neutral
metadata rather than an input to the payoff formula.

This schema does not make client-supplied data live or authoritative. It makes
the data dated, attributable, and reproducible. The yfinance research adapter
constructs the same schema from a server-fetched one-minute bar; see
[`live-market-data.md`](live-market-data.md).

## Fields

| Field | Meaning |
| --- | --- |
| `schema_version` | Always `equity-market-snapshot-v1`. |
| `symbol` | Provider symbol, up to 64 printable characters. |
| `underlier_type` | `equity`, `etf`, or `index`. |
| `currency` | Three-letter price currency. |
| `valuation_time` | Time at which the price is requested, including UTC offset. |
| `market_data_time` | Time of the captured data, including UTC offset. It cannot be later than valuation time. |
| `spot` | Positive underlier level. |
| `risk_free_rate` | Continuously compounded flat discount rate. |
| `dividend_yield` | Continuously compounded flat dividend or carry yield. |
| `volatility` | Constant annualized volatility. |
| `calendar` | Calendar identifier carried as provenance in this version. |
| `day_count` | Year-fraction convention carried as provenance in this version. |
| `source` | Provider, fixture, or capture description. Client-supplied values are not server-attested. |

The service returns a stable `snapshot_id`, calculated as the SHA-256 digest of
the canonical snapshot fields, and `age_seconds`, the difference between the
valuation and market-data timestamps.

## Flat equity GBM v2

Snapshots are consumed by `equity-gbm-flat-v2`. Under the risk-neutral measure,
the simulated dynamics are

```text
dS_t / S_t = (r - q) dt + sigma dW_t
```

where `r` is the snapshot discount rate, `q` is its dividend yield, and `sigma`
is its volatility. Cashflows remain discounted at `r`. Rate, dividend yield,
and volatility are flat over the product life.

When `q = 0`, paths and prices are numerically identical to the legacy
`gbm-flat-v1` implementation for the same seed and simulation settings.

## Product-focused API

Use `POST /api/v1/products/phoenix/price` with market data separated from
Phoenix terms:

```json
{
  "market": {
    "schema_version": "equity-market-snapshot-v1",
    "symbol": "^SPX",
    "underlier_type": "index",
    "currency": "USD",
    "valuation_time": "2026-07-13T12:00:00Z",
    "market_data_time": "2026-07-13T11:59:58Z",
    "spot": 6300.0,
    "risk_free_rate": 0.04,
    "dividend_yield": 0.013,
    "volatility": 0.19,
    "calendar": "XNYS",
    "day_count": "ACT/365F",
    "source": "manual-snapshot"
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

## Provider boundary

The provider layer implements the market-data boundary. The research
integration uses a yfinance adapter plus `LiveMarketDataService`; a future full
market provider can implement `EquityMarketDataProvider` directly. The provider
layer—not the payoff—owns:

- provider symbol mapping;
- exchange calendar and currency metadata;
- timestamp normalization and freshness policy;
- spot, curve, dividend/forward, and volatility retrieval;
- caching, rate limits, retries, usage constraints, and source provenance.

The first adapter supplies spot and quote metadata. Rate, dividend yield, and
volatility remain request assumptions and are identified as such in the market
pricing response. A later calibrated-market version must replace those inputs
with server-side curves and surfaces.

The deterministic term-structure upgrade is specified in
[`equity-market-term-structure-v1.md`](equity-market-term-structure-v1.md).
The credential-free Phase 6 research calibration is documented in
[`equity-research-market-v1.md`](equity-research-market-v1.md). A bootstrapped
collateral curve and strike-dependent arbitrage-controlled volatility surface
remain future work.

## Asset-class boundary

FX, rates, commodities, futures, cryptoassets, baskets, and multi-asset
underliers are intentionally rejected. They require different carry,
discounting, calendar, settlement, and stochastic-model conventions. They can
share the general snapshot/provider pattern, but should not be mislabeled as
equity GBM inputs.
