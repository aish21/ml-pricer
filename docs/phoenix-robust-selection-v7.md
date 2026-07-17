# Phoenix repeated group-validation selector v7

> Current serving-contract specification. The next development-only
> event-conditioned architecture did not beat this selector and was not
> promoted or audited. See
> [`phoenix-event-conditioned-research-v1.md`](phoenix-event-conditioned-research-v1.md).

V7 tests whether the v6 focused-head candidate won because of one favorable
development validation split. It does not inspect or reuse the v5 or v6 audit
labels.

## Selection policy

The original training split still fits every candidate. The contract-group
disjoint validation split is then partitioned into five folds, repeated three
times with deterministic permutations. Each of the 15 held-out slices reports:

- price MAE;
- worst market-regime MAE;
- worst moneyness-region MAE; and
- worst joint regime-by-moneyness-cell MAE.

The per-fold score uses the v6 weights:

```text
MAE + 0.35 * worst regime
    + 0.25 * worst moneyness region
    + 0.25 * worst joint cell
```

Candidate ranking uses:

```text
mean fold score + 0.25 * worst fold score
```

This is `robust-validation-mae-v3` with
`repeated-group-held-out-validation-v1`. No group enters both model training
and validation.

## Development result

The v6 winner remained first:
`payoff_aware__256x128x64__seed143__focused_head`.

Its development metrics were:

- repeated-fold selection score: `0.019584`;
- mean fold score: `0.015206`;
- worst fold score: `0.017510`;
- single-split robust score: `0.014702`; and
- validation MAE: `0.007343`.

The focused 128x128 candidate ranked second with a repeated-fold score of
`0.019719`. The original v5 128x128 payoff-aware candidate ranked fourth at
`0.020290`.

## Decision

The repeated folds confirm that the focused-head candidate is not merely a
single-validation-fold accident. However, it is numerically the same model
that failed the sealed v6 audit. V7 therefore does not consume another audit:
rerunning an unchanged predictor on new random labels would measure audit
sampling variation rather than model progress.

The next model phase should change the approximation itself by separating
discontinuous event prediction from conditional cashflow regression. Another
audit is justified only after that architecture wins under the repeated-group
policy.
