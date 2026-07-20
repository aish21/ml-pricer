# Barrier reverse convertible v1

`barrier-reverse-convertible-v1` is the second focused reference product. It is
intentionally simpler than a Phoenix: coupons are fixed, there is no autocall,
and only the maturity principal is conditional.

## Contract state

- `reference_level`: contractual starting level;
- `maturity_years`: remaining time to maturity;
- `coupon_times_years`: exact remaining coupon dates, ending at maturity;
- `coupon_rate_per_period`: fixed coupon per date and unit notional;
- `strike_frac`: conversion strike as a fraction of reference;
- `knock_in_frac`: downside barrier as a fraction of reference; and
- `prior_knock_in_breached`: historical barrier state for a seasoned note.

The normalized state is hashed into an immutable `contract_id`.

## Cashflows

Every coupon is paid and discounted on its exact date. Issuer default is not
modelled.

At maturity:

- redeem 1.00 if knock-in never occurred;
- redeem 1.00 if knock-in occurred but the final level is at or above strike;
  or
- redeem `final_level / strike_level` if knock-in occurred and the final level
  is below strike.

The high coupon is therefore compensation for conditional downside exposure,
not free yield.

## API

```text
POST /api/v1/products/barrier-reverse-convertible/price/term-structure
POST /api/v1/products/barrier-reverse-convertible/diagnostics/term-structure
```

The diagnostics return convergence, discounted cashflow components, payoff
distribution, knock-in/downside frequencies, and a common-random-number
spot/volatility surface. The Quant frontend has a dedicated product builder
and explanatory result story.

## ML and model boundary

This product is Monte Carlo reference-only. Its first expanded-product
LightGBM candidate passed R², label-quality, and latency gates but missed the
unchanged MAE and P95 limits. It was rejected; no model package or runtime
integration was created.

The reference model still omits issuer credit/default, funding, tax, fees,
liquidity, calendars, settlement, continuous-barrier correction, volatility
skew, jumps, and stochastic rates.
