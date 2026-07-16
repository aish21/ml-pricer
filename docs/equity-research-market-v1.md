# Equity research market v1

`equity-research-market-v1` is the Phase 6 server-side research calibration
contract. It builds an `equity-market-term-structure-v1` instance for a USD
equity or ETF and passes that immutable structure to the existing
`equity-gbm-piecewise-v1` Phoenix reference pricer.

It is credential-free and suitable for hobby research. It is not an
institutional curve build, an executable quote, or a production valuation
market.

## Inputs and methods

### Spot and identity

The spot, quote timestamp, currency, exchange, and available instrument type
come from the normalized yfinance one-minute regular-session quote adapter.
The existing freshness, caching, unit-normalization, and type checks apply.

### USD rate proxy

The curve provider reads the official
[Treasury Daily Interest Rate XML Feed](https://home.treasury.gov/treasury-daily-interest-rate-xml-feed)
and selects the latest Daily Treasury Par Yield Curve observation whose date is
not after the valuation date. Curves older than ten calendar days are rejected
by default.

Treasury states that CMT values are semiannual bond-equivalent par yields and
does not publish a daily zero-coupon curve. Phase 6 therefore labels the result
as a proxy. For a reported par yield `y(T)`, it uses:

```text
z_proxy(T) = 2 log(1 + y(T) / 2)
D_proxy(0,T) = exp(-z_proxy(T) T)
```

Par yields are linearly interpolated between published maturities. The
piecewise forward rate on `(T[i-1], T[i]]` is then chosen so its integrated
rate matches the proxy discount factor at every model knot:

```text
r[i] = (z_proxy(T[i]) T[i] - z_proxy(T[i-1]) T[i-1])
       / (T[i] - T[i-1])
```

This is internally consistent with the Phase 5 discounting contract, but it is
not a bootstrapped SOFR/OIS curve.

### Trailing distribution yield

For equities and ETFs, yfinance daily history supplies cash distributions over
the trailing 365 calendar days. The sum is converted to a continuously
compounded annual yield proxy:

```text
q_trailing = log(1 + trailing_cash_distributions / S0)
```

The same yield applies to every segment. A zero value means that the inspected
history contained no positive cash distributions; it is not a silent missing
data fallback. Equity indexes are rejected because the price index itself does
not distribute cash and this source cannot provide a defensible index dividend
forecast.

### Implied volatility

For each model knot, the option provider selects the nearest Yahoo Finance
expiry subject to these defaults:

- at least seven days to expiry;
- no more than 62 days from the requested model knot;
- up to five strikes nearest spot;
- valid two-sided call and put quotes; and
- combined call/put bid-ask width no greater than 10% of spot.

The maturity volatility proxy is the median valid call/put
`impliedVolatility` across the accepted near-ATM strikes. Maturity total
variance is converted into forward segment values:

```text
sigma[i]^2 = (sigma_term(T[i])^2 T[i]
              - sigma_term(T[i-1])^2 T[i-1])
             / (T[i] - T[i-1])
```

Decreasing ATM total variance is rejected rather than converted into a
negative variance bucket. This remains an ATM term proxy only. It does not fit
skew, perform a full calendar-arbitrage repair, or create a local- or
stochastic-volatility model.

## Segment schedule

The standard model knots are one month, three months, six months, one year,
two years, three years, five years, seven years, ten years, twenty years, and
thirty years. Only knots before the requested product maturity are included,
and the exact maturity is always appended.

The request fails if Treasury or listed-option expiries cannot cover that
maturity. There is no silent zero-dividend, historical-volatility, or
last-value extrapolation fallback.

## Provenance and identity

Every response includes:

- the complete immutable term structure and its `term_structure_id`;
- `calibration_version = equity-research-market-v1`;
- a `calibration_id` derived from the market and source observations;
- the normalized spot quote and cache state;
- the Treasury observation date, feed timestamp, published tenors, and method;
- the trailing cash-distribution period, total, continuous yield, and count;
- each selected option expiry, representative strike, volatility, spread, and
  number of strikes used; and
- explicit research limitations.

Cache hits are operational metadata and do not alter `calibration_id`.

## Build endpoint

Inspect the generated market before pricing:

```text
POST /api/v1/market-data/research-term-structure
```

```json
{
  "market": {
    "symbol": "SPY",
    "underlier_type": "etf",
    "currency": "USD"
  },
  "maturity_years": 1.0
}
```

## Phoenix pricing endpoint

Build and price in one request:

```text
POST /api/v1/products/phoenix/price/research-market
```

```json
{
  "market": {
    "symbol": "SPY",
    "underlier_type": "etf",
    "currency": "USD"
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

## Operational controls

Treasury requests use bounded retries, response-size validation, and a
process-local year cache. Optional environment overrides are:

- `TREASURY_CURVE_REQUEST_TIMEOUT_SECONDS`;
- `TREASURY_CURVE_CACHE_TTL_SECONDS`;
- `TREASURY_CURVE_MAX_AGE_DAYS`; and
- `TREASURY_CURVE_MAX_ATTEMPTS`.

Yahoo trailing distribution histories are cached per symbol/day for one hour;
option chains are cached per symbol and expiry for five minutes. Provider or
data-quality failures are sanitized at the API boundary.

## Upgrade boundary

A later production-oriented market version should replace the Treasury par
proxy with a bootstrapped collateral/OIS curve, use a licensed and timestamped
option feed, fit an arbitrage-controlled volatility surface, incorporate
settlement calendars, and validate the resulting model against independent
prices.

Phase 7 freezes this calibration while applying paired scenarios and
finite-difference sensitivities. See
[equity-risk-analytics-v1.md](equity-risk-analytics-v1.md).
