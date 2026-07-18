# Phoenix price-first multi-task research model

This phase tests whether observation-event supervision can improve direct
Phoenix pricing when noisy auxiliary objectives are prevented from dominating
the shared representation. It uses development data only, cannot accept an
audit dataset, and cannot write a runtime artifact.

## Frozen architecture

The model uses a shared `256x128x64` ReLU trunk and three separate heads:

- a direct linear price head;
- a 32-unit payoff head for the four unconditional cashflow components and two
  aggregate event probabilities; and
- a 32-unit event head for compact observation summaries.

The focused low-volatility/coupon-region ridge refit is applied only to the
direct price head. Auxiliary predictions never enter price reconstruction.

The event head predicts five non-redundant targets:

1. conditional expected autocall time;
2. conditional autocall-time variance;
3. final survival probability;
4. early coupon mass; and
5. late coupon mass.

Expected coupon count is derived as early plus late mass rather than trained as
a third, redundant coupon target.

## Price-first loss

All targets are standardized on the active training rows. The objective is:

```text
price loss
+ 0.25 * payoff-head loss
+ lambda * event-head loss
```

Autocall-time errors are weighted by the observed autocall probability. A case
with negligible autocall probability therefore cannot dominate training
through an unstable conditional-time label. Survival and coupon-mass targets
retain unit case weights.

Three predeclared event weights are compared: `0`, `0.03`, and `0.10`.
Selection occurs exclusively inside the original training split:

- training contract groups are divided into three deterministic,
  group-disjoint folds;
- each weight is trained on two folds and priced on the third;
- the established regime/moneyness robust score ranks each held-out fold; and
- mean fold score plus 25% of the worst score selects the weight.

No validation or development-test row enters weight selection. After selecting
the weight, one model is fitted on all training groups for 200 deterministic
AdamW epochs with model seed 143. The established repeated validation uses
fold seed 42.

PyTorch is an optional research dependency and is not required by the API or
serving runtime.

```powershell
python -m pip install -e ".[research]"

python -m src.final.surrogate_pipeline research-price-first `
  data/surrogates/phoenix-v4/datasets/<development-id>.npz `
  data/surrogates/phoenix-hazard-v1/datasets/<observation-sidecar-id>.npz `
  --report data/surrogates/phoenix-hazard-v1/full-price-first-report.json
```

Generated reports remain excluded from Git.

## Training-only weight selection

| Event loss weight | Mean internal score | Worst internal score | Selection score |
| ---: | ---: | ---: | ---: |
| 0.00 | 0.018034 | 0.018869 | 0.022752 |
| 0.03 | 0.017497 | 0.018181 | 0.022042 |
| 0.10 | 0.017467 | 0.017887 | 0.021939 |

The training-only policy selected `lambda = 0.10`.

## Development result

| Metric | Phoenix v7 | Price-first multi-task |
| --- | ---: | ---: |
| Validation MAE | 0.007343 | 0.006644 |
| Single-split robust score | 0.014702 | 0.014147 |
| Repeated-fold selection score | 0.019584 | 0.018586 |
| Mean fold score | 0.015206 | 0.014323 |
| Worst fold score | 0.017510 | 0.017052 |
| Development test MAE | 0.008146 | 0.006972 |
| Worst validation regime MAE | 0.008407 | 0.008330 |
| Worst validation moneyness MAE | 0.008344 | 0.008204 |
| Worst validation joint-cell MAE | 0.009323 | 0.010144 |

The primary repeated score improved by about 5.1%, validation MAE by 9.5%, and
development-test MAE by 14.4%. The worst joint cell became about 8.8% worse,
so a fresh audit must still test regional robustness rather than relying on the
aggregate win.

The event head's validation MAEs were:

- expected autocall time, on positive-autocall cases: `0.028666`;
- autocall-time variance, on positive-autocall cases: `0.004592`;
- final survival probability: `0.016038`;
- early coupon mass: `0.081609`; and
- late coupon mass: `0.067317`.

An exact replay selected the same weight and reproduced validation MAE,
repeated score, and test MAE with zero numerical difference.

## Decision

This is the first post-v7 architecture to pass the development promotion gate.
The model, objective, target set, weight candidates, selected weight, training
budget, seeds, and selection policy are now frozen.

It remains `research_only`: development improvement is not sufficient for
serving approval. No prior audit is reusable because those labels informed
earlier model decisions.

The next phase should generate a fresh balanced audit from the unused
`9,000,031` seed-offset family, using 256 contracts, four regimes per contract,
and eight independently scrambled Sobol replications of 256 paths. The frozen
candidate should then be evaluated exactly once against the existing price,
regional, uncertainty, output, and Greek gates. Any post-audit architecture or
threshold change must retire that audit from promotion use.
