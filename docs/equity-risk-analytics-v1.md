# Equity Risk Analytics v1

`equity-risk-analytics-v1` is the Phase 7 finite-difference risk contract for
`phoenix-single-v1`. It consumes an immutable
`equity-market-term-structure-v1` and prices every bump with
`equity-gbm-piecewise-v1`.

This is a research risk estimate per unit notional. It is not a hedge or a
capital calculation.

## Frozen-market rule

The base market is never recalibrated while a scenario or Greek is running.
The following values are frozen:

- product terms and observation schedule;
- contractual reference spot and therefore all absolute barrier levels;
- the base term-structure segmentation;
- calibration identifier, model version, path count, step count, and seed.

A spot shock changes the simulated market spot. It does not redefine the
contractual starting level or rescale the barriers. A curve shock changes only
the named segment inputs.

## Common random numbers

Base, up, and down valuations use the same matrix of standard normal shocks.
For a scenario with pathwise present values `PV_base(i)` and `PV_shock(i)`, the
reported P&L estimator is:

```text
dPV = mean(PV_shock(i) - PV_base(i))
SE(dPV) = sample_std(PV_shock(i) - PV_base(i)) / sqrt(N)
```

This paired estimator captures the covariance between the two valuations and
normally has much less Monte Carlo noise than subtracting two independent
prices. Every scenario P&L and Greek includes a standard error and 95%
confidence interval.

## Scenario shocks

`equity-market-scenario-v1` supports additive shocks to:

- `spot_pct`: relative spot movement in percent;
- `rate_parallel_bps`: all forward-rate segments, in basis points;
- `dividend_parallel_bps`: all dividend-yield segments, in basis points;
- `volatility_parallel_abs`: all segment volatilities in decimal volatility;
- `segment_shocks`: additive changes to one or more zero-based segment indexes.

Parallel and segment changes are additive. Duplicate segment indexes, empty
shocks, non-finite values, and shocks that make a market field invalid are
rejected.

Example:

```json
{
  "spot_pct": -10.0,
  "rate_parallel_bps": 25.0,
  "segment_shocks": [
    {
      "segment_index": 1,
      "volatility_abs": 0.02
    }
  ]
}
```

## Risk measures

The default symmetric bumps are 1% spot, 1 volatility point, 10 basis points
of rates, and 10 basis points of dividend yield.

For bump `h`, Phase 7 calculates:

```text
Delta = (V(S+h) - V(S-h)) / (2h)
Gamma = (V(S+h) - 2V(S) + V(S-h)) / h^2
Vega  = (V(vol+h) - V(vol-h)) / (2h), scaled to one vol point
Rho   = (V(r+h) - V(r-h)) / (2h), scaled to 100 basis points
DividendRho = (V(q+h) - V(q-h)) / (2h), scaled to 100 basis points
```

The output states units explicitly:

| Measure | Output unit |
| --- | --- |
| Delta | price per one currency unit of spot |
| Gamma | price per squared currency unit of spot |
| Vega | price change per one volatility point (`0.01`) |
| Rho | price change per 100 basis points of rates |
| Dividend rho | price change per 100 basis points of dividend yield |

Pathwise finite differences also produce a standard error, 95% confidence
interval, signal-to-noise ratio, and `statistically_resolved_95pct` flag.

## API

Use an explicit frozen curve:

```text
POST /api/v1/products/phoenix/scenario/term-structure
POST /api/v1/products/phoenix/risk/term-structure
```

Or build a Phase 6 research curve and analyze it in one request:

```text
POST /api/v1/products/phoenix/scenario/research-market
POST /api/v1/products/phoenix/risk/research-market
```

Research scenario example:

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
    "coupon_barrier_frac": 0.7,
    "coupon_rate": 0.08,
    "knock_in_frac": 0.6,
    "obs_count": 12
  },
  "shock": {
    "spot_pct": -10.0,
    "volatility_parallel_abs": 0.02
  },
  "n_paths": 2000,
  "seed": 42
}
```

Risk requests replace `shock` with optional `bumps`. Omitting `bumps` uses the
documented defaults.

## Persistence and provenance

Successful scenario and risk requests are stored in the configured SQLite run
store. The API response includes a `run_id`. Retrieve results through:

```text
GET /api/v1/runs?limit=20
GET /api/v1/runs/{run_id}
```

Stored payloads include the frozen curve, original calibration identifier,
contract and model versions, seed, path/step counts, normalized shocks or bump
sizes, base and shocked identifiers, values, and uncertainty estimates.

## Frontend

The Streamlit frontend supports:

- direct Phase 6 research-term-structure pricing;
- price plus scenario, including one optional segment shock;
- price plus Delta, Gamma, Vega, Rho, and dividend rho;
- tables and charts showing values, standard errors, and resolution flags.

Analysis always reuses the exact market returned by the pricing response. A
flat snapshot is lifted into an equivalent one-segment curve.

## Limitations

- Bump-and-revalue Greeks are local numerical estimates, not analytic hedges.
- Phoenix barriers and autocall decisions are discontinuous, so Gamma and small
  bumps can remain noisy even with common random numbers.
- There is no theta in v1. Advancing valuation time requires explicit calendar,
  fixing, accrued-coupon, and market-data roll rules.
- There is no cross-Greek, volatility-skew, correlation, credit, funding, or
  issuer-default risk.
- The research-market sources retain all Phase 6 data-quality and licensing
  limitations.
- A seasoned Phoenix needs an explicit historical reference fixing and event
  state; v1 treats the base market spot as the contractual reference level.
