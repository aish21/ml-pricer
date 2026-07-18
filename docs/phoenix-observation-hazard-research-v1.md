# Phoenix observation-level hazard research model

This phase replaces aggregate terminal-event regression with a model of the
actual Phoenix observation sequence. It is isolated from the Phoenix v7 serving
contract and cannot consume an audit dataset or write a runtime artifact.

## Why aggregate probability was insufficient

The previous event-conditioned model predicted one probability for autocall,
one for protected maturity, and one for downside maturity. That decomposition
was financially valid, but it discarded when an autocall or coupon occurred.
Timing matters because every cashflow has a different discount factor.

For observation date \(t_i\), define:

```text
autocall hazard_i
  = P(first autocall at t_i | note survived before t_i)

coupon hazard_i
  = P(coupon paid at t_i | note survived before t_i)
```

Starting with survival probability one:

```text
first_autocall_probability_i = survival_before_i * autocall_hazard_i
coupon_probability_i         = survival_before_i * coupon_hazard_i
survival_after_i              = survival_before_i
                                - first_autocall_probability_i
```

The terminal downside hazard is conditional on surviving every observation.
Protected-maturity probability is the remaining terminal probability.

## Canonical event ledger

`PhoenixPayoff` now exposes the pathwise observation ledger beside the payoff
rules. For every simulated path it records:

- coupon events by observation;
- first-autocall events by observation;
- survival after each observation;
- protected- and downside-maturity events; and
- downside recovery as a fraction of contractual reference spot.

The stable aggregate cashflow API is unchanged and is now derived from the same
ledger. A research sidecar replays the exact randomizations of an existing
development dataset, so the new event labels do not change its contracts,
markets, splits, or Monte Carlo samples.

The price identity is:

```text
coupon_rate * sum(discount_i * coupon_probability_i)
+ sum(discount_i * first_autocall_probability_i)
+ maturity_discount * protected_probability
+ maturity_discount * downside_probability
                     * conditional_downside_recovery
```

The smoke sidecar reconstructed all 384 original prices with maximum error
`6.7e-16`.

## Probability-aware learner

The research candidate uses four LightGBM regressors:

- observation-level autocall hazard;
- observation-level coupon hazard;
- terminal downside hazard; and
- conditional downside recovery.

The three event models optimize LightGBM's cross-entropy objective directly on
soft Monte Carlo probabilities. This is equivalent to binomial log loss with
fractional event frequency and is more appropriate than ordinary squared error
for a probability target. The coupon hazard is projected to be at least the
autocall hazard because every autocall observation also pays the current
coupon. All hazards and recovery values are constrained to `[0, 1]`.

Only training groups fit the models. Validation groups select and compare the
candidate using the same five-fold, three-repeat policy as v7. Test results are
diagnostic and do not affect selection.

## Reproduction

Generate observation labels from a development dataset:

```powershell
python -m src.final.surrogate_pipeline hazard-generate `
  data/surrogates/phoenix-v4-smoke/datasets/<development-id>.npz
```

Train the research candidate:

```powershell
python -m src.final.surrogate_pipeline research-hazards `
  data/surrogates/phoenix-v4-smoke/datasets/<development-id>.npz `
  data/surrogates/phoenix-hazard-v1/datasets/<hazard-id>.npz
```

Neither command accepts an audit dataset.

## Smoke development result

The smoke experiment used:

- base dataset:
  `sha256:05b96c02ca6730216f78cc9c13ec4bb60e3ba28651348dedf24d9d18e28cb57a`;
- hazard sidecar:
  `sha256:dec6259fc842cbb69f3f5d9d67bf6f5f18cd62b740b6eb7f30efcaef1003b26e`;
- 384 cases from 64 contract groups; and
- 2 independently scrambled Sobol replications of 128 paths per case.

The baseline was retrained and selected on exactly the same training and
validation groups.

| Metric | Payoff-aware baseline | Observation hazard |
| --- | ---: | ---: |
| Validation MAE | 0.021037 | 0.023716 |
| Single-split robust score | 0.045272 | 0.057633 |
| Repeated-fold selection score | 0.072112 | 0.091576 |
| Mean fold score | 0.052002 | 0.066707 |
| Worst fold score | 0.080440 | 0.099478 |
| Development test MAE | 0.037287 | 0.021053 |

The hazard candidate lost the frozen validation comparison by about 27% on the
repeated score. Its better development test MAE is encouraging but cannot
override the selection rule.

Validation hazard diagnostics were:

- autocall hazard MAE: `0.028060`;
- coupon hazard MAE: `0.055755`;
- terminal downside hazard MAE: `0.043821`;
- aggregate autocall probability MAE: `0.040814`;
- downside probability MAE: `0.040262`; and
- conditional downside recovery MAE: `0.028823`.

## Decision

The architecture remains `research_only`. It is not promoted, served, or
audited.

The smoke dataset is deliberately small, while a sequential tree model is
data-hungry. The next step is an operational scale test: freeze this
configuration, replay observation labels for the existing 6,144-case
development dataset, and compare it once against v7 using the repeated-group
policy. No further tuning should use the smoke test split. A new sealed audit
is justified only if the full-development candidate wins.
