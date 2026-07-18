# Phoenix hazard full-development comparison

This phase executes the frozen scale test specified by the observation-hazard
research plan. It changes neither the hazard model configuration nor the
Phoenix v7 comparison policy, and it does not inspect an audit dataset.

## Full observation sidecar

The canonical development dataset contains 6,144 market-contract cases from
1,024 contract groups. Each case uses two independently scrambled Sobol
replications of 1,024 paths on a 252-step grid.

The observation replay produced:

- base dataset:
  `sha256:b3dd7e8d7760d4aa2a34b96f69252d6a5e6b01557f201e74d77934b05994981f`;
- observation sidecar:
  `sha256:af8f68517df91ec9ec56773ead374470f24769104f07735ea0dad803a6f7d5eb`;
  and
- maximum price-reconstruction error: `6.7e-16`.

Case-level generation can now use multiple worker processes. Seeds remain a
pure function of the base dataset, sample index, and replication index, and the
test suite confirms that one-worker and multi-worker generation produce the
same sidecar fingerprint.

```powershell
python -m src.final.surrogate_pipeline hazard-generate `
  data/surrogates/phoenix-v4/datasets/<development-id>.npz `
  --workers 4
```

Generated sidecars remain excluded from Git.

## Frozen candidate

The full run used the predeclared smoke configuration:

- model: sequential soft-label LightGBM hazard mixture;
- estimators per head: 800;
- learning rate: `0.025`;
- maximum leaves: 15;
- minimum child samples: 10;
- L2 regularization: `0.001`; and
- training seed: 143.

No hyperparameter changed after the smoke result.

## Full-development result

Phoenix v7 and the observation-hazard candidate use the same training,
validation, test, regime, moneyness, and contract-group definitions.

| Metric | Phoenix v7 | Observation hazard |
| --- | ---: | ---: |
| Validation MAE | 0.007343 | 0.009024 |
| Single-split robust score | 0.014702 | 0.020709 |
| Repeated-fold selection score | 0.019584 | 0.027981 |
| Mean fold score | 0.015206 | 0.021760 |
| Worst fold score | 0.017510 | 0.024882 |
| Development test MAE | 0.008146 | 0.008944 |
| Worst validation regime MAE | 0.008407 | 0.013524 |
| Worst validation moneyness MAE | 0.008344 | 0.011399 |
| Worst validation joint-cell MAE | 0.009323 | 0.016408 |

The full hazard candidate's validation diagnostics were:

- autocall hazard MAE: `0.012869`;
- coupon hazard MAE: `0.028057`;
- terminal downside hazard MAE: `0.018642`;
- aggregate autocall probability MAE: `0.017499`;
- downside probability MAE: `0.017778`; and
- conditional downside recovery MAE: `0.008691`.

Scaling from 384 to 6,144 cases materially improved the hazard model, but it
did not reverse the decision. Its repeated score remained about 43% worse than
v7.

## Decision

The pure sequential hazard price is rejected. It remains `research_only`, no
runtime artifact is produced, and no new sealed audit is justified.

The result identifies error propagation as the architectural problem. Small
errors in every conditional hazard compound through survival probabilities and
then affect multiple discounted cashflows.

The next model phase should therefore be hybrid:

1. retain the v7 direct price and unconditional cashflow-component heads;
2. add observation-level event information as auxiliary targets for a shared
   representation;
3. keep price inference independent of sequential probability multiplication;
4. rank the hybrid only on the existing repeated group-validation policy; and
5. reserve another audit only if the direct price materially beats v7.

Fixed-dimensional event summaries should be tested before introducing a
custom masked neural-network loss. Useful summaries include expected first-call
time, first-call-time variance, final survival probability, expected coupon
count, and early-versus-late coupon mass.
