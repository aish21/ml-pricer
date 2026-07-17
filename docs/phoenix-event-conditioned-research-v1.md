# Phoenix event-conditioned research candidate

This phase changes the approximation architecture without changing the
production Phoenix v7 artifact, serving contract, promotion thresholds, or
sealed audit history. The candidate is development-only and cannot be loaded by
the API.

## Finance decomposition

The Phoenix terminal redemption events are mutually exclusive and exhaustive:

1. the note autocalled on an observation date;
2. the note survived and returned protected principal at maturity; or
3. the note survived and paid a downside redemption at maturity.

For event \(E_j\), the unconditional present value of its principal branch is:

```text
PV_j = P(E_j) * E[discounted principal_j | E_j]
```

The full price can therefore be written as:

```text
coupon PV
+ autocall probability * conditional autocall principal PV
+ protected-maturity probability * conditional protected principal PV
+ downside probability * conditional downside principal PV
```

The existing Monte Carlo labels contain every quantity needed for this
identity. Protected-maturity probability is inferred as
`1 - autocall_probability - downside_probability`. The implementation verifies
that the transformed labels reconstruct every original price to floating-point
tolerance.

## ML architecture

The offline candidate contains:

- one network for coupon PV and the three terminal-event probabilities; and
- three conditional-value expert networks, one per terminal branch.

Each conditional expert is trained only on training rows where its event has
positive Monte Carlo probability. This avoids teaching an arbitrary zero value
when a conditional expectation is mathematically undefined. At prediction
time, the three probabilities are projected onto the probability simplex and
all value outputs are projected to be non-negative before the finance identity
is applied.

Only the original training split fits network parameters. The validation split
is scored with the same five-fold, three-repeat, contract-group-held-out policy
as v7. Test results are diagnostic only. The entry point accepts no audit
dataset and writes no serving artifact:

```powershell
python -m src.final.surrogate_pipeline research-events `
  data/surrogates/phoenix-v4/datasets/<development-dataset-id>.npz `
  --hidden-layers 256 128 64 `
  --training-seed 143
```

## Development result

The canonical experiment used development dataset
`sha256:b3dd7e8d7760d4aa2a34b96f69252d6a5e6b01557f201e74d77934b05994981f`
and candidate `event_conditioned__256x128x64__seed143`.

| Metric | Phoenix v7 | Event-conditioned candidate |
| --- | ---: | ---: |
| Validation MAE | 0.007343 | 0.009127 |
| Single-split robust score | 0.014702 | 0.019008 |
| Repeated-fold selection score | 0.019584 | 0.026227 |
| Mean fold score | 0.015206 | 0.019721 |
| Worst fold score | 0.017510 | 0.026021 |

The research candidate's development test MAE was `0.009156`. Its
event-positive conditional-value MAEs on validation were:

- autocall principal: `0.001577`;
- protected-maturity principal: `0.001961`; and
- downside principal: `0.010294`.

The corresponding terminal-probability MAEs were `0.018394` for autocall,
`0.018586` for protected maturity, and `0.019006` for downside. The probability
head, not the conditional-value identity, is the main weakness.

## Decision

The event-conditioned candidate does not replace v7 and does not justify
another sealed audit. Its repeated-fold score is about 34% worse than v7.
Keeping it research-only prevents a plausible finance decomposition from being
mistaken for an empirically better pricing model.

The next experiment should model the actual observation sequence rather than
only terminal aggregate probabilities. The label schema needs first-autocall
probability by observation date, survival probability, and coupon probability
while alive. A discrete hazard/event head can then use probability-aware loss
functions, while deterministic discount factors convert predicted event
probabilities into most principal and coupon PVs. Only downside recovery still
requires a conditional-value regressor.
