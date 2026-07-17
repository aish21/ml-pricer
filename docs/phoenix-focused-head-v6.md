# Phoenix focused-output-head surrogate v6

> Historical specification. V7 confirmed the focused candidate across repeated
> group-held-out validation folds but deliberately did not rerun the unchanged
> predictor on another audit. See
> [`phoenix-robust-selection-v7.md`](phoenix-robust-selection-v7.md).

V6 addresses the only failed v5 promotion check without changing that check.
The inspected v5 audit is retired from promotion use.

## Development-only change

Each payoff-aware MLP still learns its hidden representation on the training
split. V6 adds a candidate that freezes those hidden layers and refits the
final linear output layer using regularized weighted least squares:

- low-volatility training cases: 2x weight;
- coupon-barrier training cases: 2x weight;
- low-volatility coupon-barrier cases: 4x combined weight; and
- ridge penalty on the output coefficients: `0.001`.

Only training rows enter the refit. Validation rows choose between the original
and focused-head candidates. Test and audit rows do not affect fitting or
selection.

The robust validation score is now:

```text
validation MAE
+ 0.35 * worst regime MAE
+ 0.25 * worst moneyness-region MAE
+ 0.25 * worst regime-by-moneyness-cell MAE
```

This is `robust-validation-mae-v2`.

## Frozen development result

The selected candidate before generating a new audit was
`payoff_aware__256x128x64__seed143__focused_head`.

Compared with the v5-selected `payoff_aware__128x128__seed42` candidate:

- validation MAE improved from `0.007549` to `0.007343`;
- worst validation regime MAE improved from `0.008809` to `0.008407`;
- worst validation moneyness-region MAE changed from `0.008301` to `0.008344`;
- worst validation joint-cell MAE improved from `0.009846` to `0.009323`; and
- the v2 robust score was `0.014702`.

These values use development data only.

## Audit protocol

The uncertainty and economic gates remain exactly those in
[`phoenix-uncertainty-v5.md`](phoenix-uncertainty-v5.md), including 8x256
independently scrambled Sobol paths, the 95% Student-t interval, the 1% unit
notional economic tolerance, the 80% coverage requirement, and the label
precision check.

The new balanced audit uses seed offset `7,000,019`, giving dataset seed
`7,000,061` and label seed `7,007,320`. It contains 256 contracts and four
market regimes per contract. If v6 is changed after this audit is inspected,
another seed family is required.

## Canonical audit result

The sealed v6 audit used:

- audit dataset:
  `sha256:8657d58ea2b4dfe2b52081f8444f8fb609f7fc954f7f1df2d7f39324ab19a7d7`;
- artifact:
  `sha256:21d626d1e247489eafad466fcbab12b7211e65402427708b62b4547028db14f1`;
  and
- selected candidate:
  `payoff_aware__256x128x64__seed143__focused_head`.

The principal results were:

- MAE: `0.007247`;
- 95th-percentile absolute error: `0.020177`;
- R-squared: `0.995822`;
- worst regime MAE: `0.010047`;
- worst moneyness-region MAE: `0.007691`;
- worst joint-cell MAE: `0.011148`;
- uncertainty-or-economic coverage: 77.54%;
- 95th-percentile label confidence half-width: `0.011471`;
- Delta sign agreement: 100%;
- Vega sign agreement: 100%; and
- Rho sign agreement: 75%.

All checks passed except the frozen 80% coverage requirement. Exactly 794 of
1,024 cases were covered, versus the minimum count of 820. The artifact
therefore remains `research_only`.

The focused head improved the single development validation split but did not
generalize to the new audit. V6 rules and thresholds must not be tuned against
this audit. A subsequent model phase should use repeated group-disjoint
development folds and explicitly model the payoff's event discontinuities
before reserving another audit.
