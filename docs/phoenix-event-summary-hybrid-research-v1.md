# Phoenix event-summary hybrid research model

> Historical equal-weight experiment. A price-first successor with separate
> heads and masked auxiliary losses subsequently beat v7 on the development
> gate. See
> [`phoenix-price-first-multitask-research-v1.md`](phoenix-price-first-multitask-research-v1.md).

This phase tests whether observation-event labels can improve the Phoenix v7
representation without making price inference depend on a sequential hazard
calculation. It uses development data only, cannot read an audit dataset, and
cannot write a runtime artifact.

## Architecture

One shared MLP has 13 outputs:

- the direct price;
- four unconditional discounted cashflow components;
- aggregate autocall and downside probabilities; and
- six fixed-dimensional observation-event summaries.

The event summaries are:

1. conditional expected first-autocall time as a fraction of the observation
   schedule;
2. conditional first-autocall-time variance;
3. probability of surviving the final observation;
4. expected coupon count;
5. expected coupon mass in the first half of the schedule; and
6. expected coupon mass in the second half.

Expected coupon count is exactly the sum of the early and late masses. The
autocall-time statistics are zero when the simulated autocall probability is
zero.

Every target is standardized using the training split, and the shared network
uses equal squared-error weight for each standardized output. Price is always
read from its direct output. Event predictions never enter a price
reconstruction.

The frozen comparison uses the v7 winner's representation settings:

- hidden layers: `256x128x64`;
- model seed: 143;
- maximum iterations: 1,000; and
- validation-fold seed: 42.

Both the ordinary learned output head and the existing low-volatility,
coupon-region focused ridge head are evaluated. The same training group,
validation group, regime, moneyness, and repeated-fold definitions used by v7
rank the two heads.

```powershell
python -m src.final.surrogate_pipeline research-hybrid `
  data/surrogates/phoenix-v4/datasets/<development-id>.npz `
  data/surrogates/phoenix-hazard-v1/datasets/<observation-sidecar-id>.npz `
  --report data/surrogates/phoenix-hazard-v1/full-hybrid-report.json
```

Generated reports remain excluded from Git.

## Development result

The focused hybrid head ranked ahead of the ordinary hybrid head, but both
lost to v7.

| Metric | Phoenix v7 | Hybrid base | Hybrid focused |
| --- | ---: | ---: | ---: |
| Validation MAE | 0.007343 | 0.011209 | 0.010077 |
| Single-split robust score | 0.014702 | 0.021927 | 0.020142 |
| Repeated-fold selection score | 0.019584 | 0.029456 | 0.027218 |
| Mean fold score | 0.015206 | 0.023104 | 0.021122 |
| Worst fold score | 0.017510 | 0.025408 | 0.024383 |
| Development test MAE | 0.008146 | 0.011599 | 0.010241 |
| Worst validation regime MAE | 0.008407 | 0.012266 | 0.011345 |
| Worst validation moneyness MAE | 0.008344 | 0.012039 | 0.011013 |
| Worst validation joint-cell MAE | 0.009323 | 0.013657 | 0.013364 |

The focused hybrid's validation summary MAEs were:

- expected autocall time: `0.050858`;
- autocall-time variance: `0.004108`;
- final survival probability: `0.017454`;
- expected coupon count: `0.117473`;
- early coupon mass: `0.082119`; and
- late coupon mass: `0.071145`.

The hybrid repeated score is about 39% worse than v7. It is slightly better
than the pure sequential hazard candidate, but that is not the promotion
criterion.

## Interpretation and decision

The model learned useful event summaries, but equal standardized weighting
caused negative transfer into the price task. Adding six summaries changed the
price head from one of seven equally scaled outputs to one of 13. The redundant
coupon-count identity further increased coupon-event influence, while the
conditional autocall-time convention is unstable when autocall probability is
near zero.

The candidate is rejected. It remains `research_only`, the serving contract is
unchanged, and no sealed audit is inspected.

The next architecture experiment should isolate the training objectives:

1. preserve the v7 direct-price control;
2. use separate price, payoff-component, and event-summary heads;
3. give auxiliary losses a deliberately smaller weight;
4. mask autocall-time loss by observed autocall probability;
5. predict early and late coupon mass and derive their total instead of
   training on a redundant third target; and
6. choose any auxiliary-loss weight with group-disjoint folds inside the
   training split before evaluating the existing validation split once.

This would test event supervision without allowing noisy auxiliary targets to
dominate price optimization.

That controlled experiment has now been completed. Training-only group folds
selected a 10% event loss weight, and the frozen price-first model beat v7's
repeated development score without consuming an audit.
