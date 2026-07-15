# Equity market term structure v1

`equity-market-term-structure-v1` is the Phase 5 market contract for the
single-underlier Phoenix reference pricer. It replaces one flat `r`, `q`, and
`sigma` with deterministic piecewise-constant segments while leaving the
`phoenix-single-v1` payoff rules unchanged.

The corresponding model version is `equity-gbm-piecewise-v1`.

## Segment convention

Every segment contains:

- `end_time_years`;
- continuously compounded `risk_free_rate`;
- continuous `dividend_yield`; and
- instantaneous `volatility`.

A segment applies from the previous segment end, exclusive, through its own end
time. The first segment starts at time zero. End times must be strictly
increasing.

For example:

```json
"segments": [
  {
    "end_time_years": 0.5,
    "risk_free_rate": 0.035,
    "dividend_yield": 0.012,
    "volatility": 0.18
  },
  {
    "end_time_years": 1.0,
    "risk_free_rate": 0.04,
    "dividend_yield": 0.013,
    "volatility": 0.21
  }
]
```

means the first values apply over `(0, 0.5]` and the second over `(0.5, 1.0]`.
The final segment must cover the product maturity. The implementation refuses
to extrapolate a final value silently.

## Dynamics

Under the risk-neutral measure, a simulation interval `[t0, t1]` uses:

```text
log(S(t1) / S(t0)) = integral(r-q)dt - 0.5 integral(sigma^2)dt
                      + sqrt(integral(sigma^2)dt) Z
```

The integrals are split exactly at every segment boundary, including when a
Monte Carlo step crosses a tenor knot. Cashflows at time `t` use:

```text
D(0,t) = exp(-integral(0,t) r(u)du)
```

This is still a deterministic-coefficient GBM. It is not local volatility,
stochastic volatility, a smile model, or a full strike-dependent implied
volatility surface.

## Flat-model compatibility

A single segment with constant values produces the same path distribution and
discount factors as `equity-gbm-flat-v2` for the same seed and step count. This
is tested as a compatibility invariant.

The response includes equivalent maturity-average values in the legacy `r` and
`sigma` parameter slots for summary and compatibility purposes. The actual path
generation and discounting use the complete term structure included in
`market_term_structure`.

## Full request

```json
{
  "market": {
    "schema_version": "equity-market-term-structure-v1",
    "symbol": "SPY",
    "underlier_type": "etf",
    "currency": "USD",
    "valuation_time": "2026-07-14T12:00:00Z",
    "market_data_time": "2026-07-14T11:59:58Z",
    "spot": 620.0,
    "segments": [
      {
        "end_time_years": 0.5,
        "risk_free_rate": 0.035,
        "dividend_yield": 0.012,
        "volatility": 0.18
      },
      {
        "end_time_years": 1.0,
        "risk_free_rate": 0.04,
        "dividend_yield": 0.013,
        "volatility": 0.21
      }
    ],
    "calendar": "XNYS",
    "day_count": "ACT/365F",
    "source": "manual-research-curve"
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

Send the request to:

```text
POST /api/v1/products/phoenix/price/term-structure
```

## Provenance and calibration boundary

The schema fingerprints all identity, timestamp, spot, segment, convention, and
source fields into `term_structure_id`. This makes a pricing run attributable
and reproducible.

Phase 5 accepts a caller-supplied research term structure. Phase 6 can build the
same immutable schema through the explicitly approximate, credential-free
[`equity-research-market-v1`](equity-research-market-v1.md) contract. It uses
separately attributed Treasury and option inputs while preserving the manual
route for fully controlled experiments.
