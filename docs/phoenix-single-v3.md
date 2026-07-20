# Phoenix Single v3 contract

`phoenix-single-v3` is the additive memory-coupon and step-down extension of
the active-trade v2 contract. It does not change v1 or v2 cashflows.

## Added state

The contract keeps the v2 reference level, exact remaining observation times,
and prior knock-in flag, then adds:

- `autocall_barrier_fracs`: one barrier for every remaining observation;
- `memory_coupon`: whether missed coupons are carried forward; and
- `unpaid_coupon_count`: missed memory coupons already outstanding at the
  valuation time.

Autocall barriers must be constant or non-increasing and must remain at or
above the coupon barrier. The complete state is included in `contract_id`.

## Event rules

At each observation while the note is active:

1. If spot is at or above the coupon barrier, pay the current coupon. With
   memory on, also pay every carried coupon and reset the balance to zero.
2. Otherwise, with memory on, add the missed coupon to the path's balance.
3. Compare spot with that observation's autocall barrier. If it passes, redeem
   principal and end the note.

Unpaid memory coupons are conditional, not guaranteed. If no later active
observation reaches the coupon line, they are not paid.

Maturity redemption and discrete knock-in monitoring follow v2.

## API

```text
POST /api/v1/products/phoenix/price/richer/term-structure
POST /api/v1/products/phoenix/diagnostics/richer/term-structure
```

The Quant frontend exposes memory state and a linear step-down builder for
seasoned trades, then plots the exact hurdle at every remaining observation.

## ML boundary

The approved v1 model is not valid for v3. v3 remains Monte Carlo reference
priced. The first expanded-product candidate passed R², label-quality, and
latency gates but failed MAE and P95 error gates, so it was rejected and no
runtime artifact was created.

## Limitations

- The frontend builder uses even remaining dates and linear step-downs, while
  the API accepts any valid exact date schedule and non-increasing barriers.
- Knock-in monitoring is discrete on the simulation grid.
- Historical state is caller supplied rather than reconstructed from fixings.
- Credit, funding, fees, calendars, settlement, and volatility skew remain out
  of scope.
