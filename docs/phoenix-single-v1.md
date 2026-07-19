# Phoenix Single v1 contract

`phoenix-single-v1` is the first quantitatively validated product contract in
ML Pricer. It is generic across single equity-like underliers; SPY is the
planned first market-data and calibration universe, not a hardcoded payoff
dependency.

## Scope and units

- Prices and cashflows are quoted per unit notional.
- `S0` is the initial underlier level in any consistent price currency.
- Barriers are fractions of `S0`.
- `coupon_rate` is a coupon per observation, as a fraction of notional.
- `T` is maturity in years and `r` is a continuously compounded flat rate.
- Observation dates are evenly spaced over `(0, T]` in this first contract.
- Knock-in monitoring is discrete on every simulated path step, including the
  initial and final points.
- Coupons are non-memory: a missed coupon is not recovered later.

The product contract is independent of the market model. The original
`gbm-flat-v1` reference uses a constant rate and volatility with zero dividend
yield. The dated [`equity-market-snapshot-v1`](equity-market-snapshot-v1.md)
path uses `equity-gbm-flat-v2`, adding a constant dividend yield to the
risk-neutral drift. The
[`equity-market-term-structure-v1`](equity-market-term-structure-v1.md) path uses
`equity-gbm-piecewise-v1` for deterministic piecewise rates, dividend yields,
and volatility. Strike-dependent volatility surfaces remain a future,
separately versioned upgrade.

## Cashflow rules

At each observation date, while the note is active:

1. If the underlier is at or above the coupon barrier, pay `coupon_rate`.
2. If the underlier is at or above the autocall barrier, redeem one unit of
   principal and terminate the note.

If the note has not autocalled, redeem at maturity:

- one unit of principal if the knock-in barrier was never touched; or
- one unit of principal if the barrier was touched but the final level is at or
  above `S0`; or
- `S_T / S0` if the barrier was touched and the final level is below `S0`.

Every cashflow is discounted from its payment time using either `exp(-r * t)`
for the flat model or the term-structure discount factor `D(0,t)`.

## Valid term relationships

- `S0 > 0`, `sigma > 0`, `T > 0`, and `1 <= obs_count <= 252`.
- `0 < knock_in_frac <= 1` and
  `knock_in_frac <= coupon_barrier_frac <= autocall_barrier_frac`.
- `coupon_rate >= 0`.

The implementation is invariant to a common scaling of `S0` and every path
level. A ticker symbol is therefore metadata used to obtain a market snapshot;
it does not belong in the numerical payoff formula.

## Explicit non-goals for v1

- Multiple underliers or worst-of baskets.
- Memory coupons, step-down barriers, guaranteed coupons, or settlement lags.
- Business-day calendars and explicit irregular observation schedules.
- Continuous barrier monitoring.
- Local-volatility, stochastic-volatility, jump, strike-smile, or
  stochastic-rate dynamics.

Those features require separate contract and pricing-model versions rather
than silent changes to this definition.

The active-trade [`phoenix-single-v2`](phoenix-single-v2.md) contract provides
an explicit historical reference level, exact remaining observation times, and
prior knock-in state without changing this frozen new-issue definition.
