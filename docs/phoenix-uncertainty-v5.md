# Phoenix uncertainty-calibrated surrogate v5

Phase 11 changes how the Phoenix surrogate is audited. It does not relax the
price, regime, barrier-region, payoff-component, event-probability, boundary,
or Greek gates introduced by v4.

The v4 model reduced fresh-audit MAE to `0.006984`, but only 31.0% of its
predictions were within two estimated label standard errors. Two randomized
Sobol replications are not enough to estimate a sampling variance reliably:
with one degree of freedom, several labels received an implausibly small or
zero estimated standard error. That made the relative-error gate unstable even
when the absolute pricing error was economically small.

## Frozen audit protocol

The following protocol is fixed before generating or inspecting the v5 audit:

- policy version: `phoenix-audit-uncertainty-v1`;
- audit sampling: independently scrambled Sobol sequences;
- independent randomizations per label: at least 8;
- total paths per label: at least 2,048;
- canonical allocation: 8 randomizations of 256 paths;
- confidence level: 95%;
- confidence multiplier: Student-t with `R - 1` degrees of freedom;
- economic price tolerance: `0.01` per unit notional;
- minimum uncertainty-or-economic coverage: 80%;
- maximum 95th-percentile label confidence half-width: `0.015`;
- canonical fresh-audit seed offset: `5,000,011`; and
- canonical audit composition: 256 contracts, four balanced market regimes per
  contract, for 1,024 cases.

For audit case `i`, let `e_i` be the absolute surrogate error, `se_i` the
standard error across randomized Sobol replication means, and `t*` the
two-sided 95% Student-t critical value. A case is covered when

```text
e_i <= max(t* * se_i, 0.01)
```

At least 80% of the sealed audit cases must be covered. The separate label
precision check prevents a deliberately noisy Monte Carlo audit from passing
by producing wide confidence intervals.

The 1% economic tolerance is a model-risk allowance, not Monte Carlo
uncertainty. On the unit-notional price scale it means one price point. It is
reported separately from the pure confidence-interval coverage so the artifact
cannot imply that model approximation error is statistical noise.

## Why Student-t?

The label is the mean of `R` independently randomized quasi-Monte Carlo
estimates. Their sample standard deviation estimates the randomization
variance. Because that variance is itself estimated from only eight
observations, the normal multiplier `1.96` is too optimistic. Student-t with
seven degrees of freedom uses a multiplier of approximately `2.365`.

This does not claim that path payoffs are normally distributed. It applies a
small-sample confidence construction to the independent replication means,
which is the level at which randomized-QMC uncertainty is estimated.

## Promotion rules

V5 retains the v4 limits:

- audit MAE no greater than `0.02`;
- audit 95th-percentile absolute error no greater than `0.05`;
- audit R-squared at least `0.90`;
- worst regime MAE no greater than `0.03`;
- worst moneyness-region MAE no greater than `0.03`;
- worst regime-by-moneyness-cell MAE no greater than `0.04`;
- maximum payoff-component MAE no greater than `0.08`;
- maximum event-probability MAE no greater than `0.08`;
- mean raw output-boundary violation no greater than `0.005`;
- cashflow reconstruction MAE no greater than `0.08`; and
- Delta, Vega, and Rho sign agreement each at least 50%.

The old `within_two_label_se_fraction` remains a diagnostic, but it is no
longer a promotion gate. It answers whether approximation error is as small as
Monte Carlo estimator noise, which is stricter than the product requirement
and becomes ill-conditioned as labels get more precise.

## Leakage controls

Candidate architecture and random-seed selection still use development
validation data only. The v5 audit is generated with new dataset and label
seeds. Its identifier, uncertainty policy, thresholds, and policy checksum are
written into the artifact manifest.

If any model, feature, threshold, sampling rule, or selection policy is changed
after the audit is inspected, that audit is retired to research evidence. A
new seed family is then required for another promotion decision.

## Commands

Generate and train with the canonical development and audit allocations:

```powershell
python -m src.final.surrogate_pipeline full `
  --output-root data/surrogates/phoenix-v5
```

Generate only a sealed audit:

```powershell
python -m src.final.surrogate_pipeline generate `
  --dataset-role audit `
  --n-contracts 256 `
  --markets-per-contract 4 `
  --paths-per-replication 256 `
  --label-replications 8 `
  --dataset-seed 5000053 `
  --label-seed 5007312 `
  --sampling-profile balanced `
  --output-root data/surrogates/phoenix-v5
```

The audit result must be recorded here without changing the rules above.

## Canonical result

The first sealed v5 audit used:

- development dataset:
  `sha256:b3dd7e8d7760d4aa2a34b96f69252d6a5e6b01557f201e74d77934b05994981f`;
- audit dataset:
  `sha256:578572b8826983985bc4cdc6e180b951828cb896f326a19e930c23f9f1a3704f`;
- uncertainty policy:
  `sha256:b3b797cc84ed20b14ccf79667b5224c184bfd0bf3c449cd88fdcf7c1d785db4c`;
- artifact:
  `sha256:840e812dd565d75c79cb5f55c3ba4285281a37daf5f1f30c964ef40bb252490c`;
  and
- selected candidate: `payoff_aware__128x128__seed42`.

Its principal audit metrics were:

- MAE: `0.006762`;
- 95th-percentile absolute error: `0.018281`;
- R-squared: `0.996834`;
- worst regime MAE, low volatility: `0.008765`;
- worst moneyness-region MAE, coupon barrier: `0.007341`;
- worst joint-cell MAE, low-volatility/coupon: `0.010087`;
- pure 95% label-confidence coverage: 49.32%;
- uncertainty-or-economic coverage: 79.49%;
- 95th-percentile label confidence half-width: `0.011733`;
- Delta sign agreement: 86.67%;
- Vega sign agreement: 93.33%; and
- Rho sign agreement: 81.25%.

All frozen checks passed except the 80% uncertainty-or-economic coverage
requirement. Exactly 814 of 1,024 cases were covered; the minimum integer count
was 820. The threshold was not changed after inspection. The artifact therefore
remains `research_only`.

This is a useful negative result. Increasing the number of randomizations
raised the pure confidence coverage from v4's unstable 31.0% two-SE diagnostic
to 49.3% under a proper small-sample interval, and the label-precision gate
passed. The remaining gap is now an approximation-quality problem concentrated
around low-volatility barrier cases, rather than evidence that the audit
standard error is collapsing to zero.
