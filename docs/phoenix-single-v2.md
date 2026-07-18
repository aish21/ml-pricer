# Phoenix Single v2 active-trade contract

`phoenix-single-v2` adds the minimum contractual state needed to value an
already-issued, still-active single-underlier Phoenix without confusing live
market data with facts fixed at issuance. It is a separate contract version;
the existing `phoenix-single-v1` new-issue routes and audited surrogate are
unchanged.

## Why this version exists

A live spot is a market observation at valuation time. A Phoenix reference
level is a contractual value fixed when the note was issued. Once a trade is
seasoned, those values generally differ:

```text
live spot = today's observed underlier level
reference level = fixed base used to calculate all contractual barriers
```

The trade may also have crossed its knock-in barrier before today. That event
continues to affect maturity redemption even when every future simulated level
stays above the barrier. Finally, remaining observation dates need not be
evenly spaced from the new valuation date.

## Required state

The v2 contract requires:

- `reference_level`: the original contractual underlier level;
- `maturity_years`: remaining time from valuation to maturity;
- `observation_times_years`: strictly increasing remaining event times,
  measured forward from valuation;
- `prior_knock_in_breached`: whether the knock-in was observed before
  valuation;
- the existing autocall, coupon, and knock-in barrier fractions; and
- the non-memory coupon rate per observation.

The final observation must equal `maturity_years`. The API assumes the note is
still active. An already-autocalled note has no remaining optionality and must
not be sent to this route.

The complete normalized contract is hashed into `contract_id`. Reference level,
schedule, barrier terms, coupon, and historical knock-in state all contribute
to that identity.

## Simulation convention

The piecewise GBM model still uses a 252-step base monitoring grid. Every
contractual observation time is inserted into that grid exactly. This has two
effects:

1. coupons and autocall decisions occur at the requested event time; and
2. knock-in monitoring retains the base grid plus any inserted event times.

Cashflows use the term-structure discount factor at the exact observation
time. Knock-in monitoring remains discrete; continuous barrier corrections are
not part of this version.

## API

Send:

```text
POST /api/v1/products/phoenix/price/seasoned/term-structure
```

with:

```json
{
  "market": {
    "schema_version": "equity-market-term-structure-v1",
    "symbol": "SPY",
    "underlier_type": "etf",
    "currency": "USD",
    "valuation_time": "2026-07-18T12:00:00Z",
    "market_data_time": "2026-07-18T11:59:58Z",
    "spot": 620.0,
    "segments": [
      {
        "end_time_years": 1.0,
        "risk_free_rate": 0.04,
        "dividend_yield": 0.013,
        "volatility": 0.21
      }
    ],
    "calendar": "XNYS",
    "day_count": "ACT/365F",
    "source": "request"
  },
  "contract": {
    "contract_version": "phoenix-single-v2",
    "reference_level": 650.0,
    "maturity_years": 1.0,
    "observation_times_years": [0.18, 0.43, 0.68, 1.0],
    "autocall_barrier_frac": 1.05,
    "coupon_barrier_frac": 1.0,
    "coupon_rate": 0.02,
    "knock_in_frac": 0.7,
    "prior_knock_in_breached": false
  },
  "n_paths": 2000
}
```

The response reports the live spot under `market_term_structure`, the fixed
reference under `contract`, and the reference level again as `params.S0` for
compatibility with the payoff engine.

## Surrogate boundary

The audit-approved surrogate remains governed only for new-issue
`phoenix-single-v1` inputs. A v2 response therefore reports shadow status
`not_applicable`; it never sends v2 observations into the v1 promotion sample.

Before a surrogate can cover v2, its dataset and feature contract must include
live spot/reference moneyness, irregular remaining schedules, and historical
knock-in state, followed by a separate audit and promotion policy.

## Remaining limitations

- Future knock-in monitoring is discrete.
- The route accepts year fractions rather than calendar dates and settlement
  conventions.
- Historical state is supplied by the caller; the service does not reconstruct
  it from fixings.
- Scenario and Greek routes still support only v1 in this phase.
- Credit, funding, issuer default, and volatility-skew effects remain outside
  the reference model.
