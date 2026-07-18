# Phoenix price-first sealed audit

This phase evaluates the frozen price-first multi-task candidate exactly once
on a new independent audit. The audit does not tune or retrain any
hyperparameter, loss weight, threshold, sampling rule, or model-selection
policy.

## Frozen model

The audited model is the development winner specified by:

- source commit:
  `e1ba32e567a40ca501d1e89b8e7396dead803427`;
- development dataset:
  `sha256:b3dd7e8d7760d4aa2a34b96f69252d6a5e6b01557f201e74d77934b05994981f`;
- observation sidecar:
  `sha256:af8f68517df91ec9ec56773ead374470f24769104f07735ea0dad803a6f7d5eb`;
- event loss weight: `0.10`;
- model seed: 143; and
- validation policy seed: 42.

Before reading audit labels, the evaluator refitted the deterministic model and
required exact reproduction of its frozen validation MAE, repeated-selection
score, and development-test MAE.

## Independent audit

The reserved `9,000,031` seed-offset family produced:

- audit dataset:
  `sha256:117b8c4f4596dffc340be258da0362ac1272fa29fe4ca294af416943a49734e0`;
- dataset seed: `9,000,073`;
- label seed: `9,007,332`;
- 256 contract groups;
- four balanced market regimes per contract;
- 1,024 total cases;
- eight independently scrambled Sobol replications;
- 256 paths per replication; and
- 252 simulation steps.

The uncertainty policy remained
`sha256:b3b797cc84ed20b14ccf79667b5224c184bfd0bf3c449cd88fdcf7c1d785db4c`.
It uses a 95% Student-t interval across the eight randomized replication means
and the existing 1% unit-notional economic tolerance.

```powershell
python -m src.final.surrogate_pipeline audit-price-first `
  data/surrogates/phoenix-v4/datasets/<development-id>.npz `
  data/surrogates/phoenix-hazard-v1/datasets/<observation-sidecar-id>.npz `
  data/surrogates/phoenix-price-first-v1/datasets/<audit-id>.npz `
  --report data/surrogates/phoenix-price-first-v1/sealed-audit-report.json
```

Neither the generated dataset nor the detailed report is committed to Git.

## Audit result

Every frozen promotion check passed.

| Check | Observed | Required |
| --- | ---: | ---: |
| Price MAE | 0.006030 | <= 0.020000 |
| Price p95 absolute error | 0.017726 | <= 0.050000 |
| Price R-squared | 0.997626 | >= 0.900000 |
| Worst regime MAE | 0.008041 | <= 0.030000 |
| Worst moneyness-region MAE | 0.006486 | <= 0.030000 |
| Worst joint-cell MAE | 0.009962 | <= 0.040000 |
| Label confidence half-width p95 | 0.011619 | <= 0.015000 |
| Uncertainty-or-economic coverage | 84.57% | >= 80.00% |
| Worst cashflow-component MAE | 0.012897 | <= 0.080000 |
| Worst event-probability MAE | 0.013753 | <= 0.080000 |
| Mean output-boundary violation | 0.001002 | <= 0.005000 |
| Cashflow reconstruction MAE | 0.005741 | <= 0.080000 |
| Delta sign agreement | 86.67% | >= 50.00% |
| Vega sign agreement | 87.50% | >= 50.00% |
| Rho sign agreement | 93.75% | >= 50.00% |

Low volatility remained the worst regime at MAE `0.008041`. The coupon region
remained the worst moneyness region at `0.006486`, and low-volatility/coupon
was the worst joint cell at `0.009962`. All remained well within their frozen
limits.

Gamma sign agreement was `53.33%` and dividend-Rho sign agreement was 100%.
They remain diagnostics rather than promotion gates.

## Decision

The frozen price-first candidate passes its sealed audit. The audit is now
consumed and cannot be reused to justify a modified architecture or threshold.

The model is audit-approved but remains `research_only` because the branched
PyTorch research object has no versioned, checksum-verified runtime artifact or
production loader. No serving pointer was changed by this phase.

The next phase should export the frozen ReLU trunk and three heads into a
pure-NumPy branched artifact so API deployments do not require PyTorch. The
artifact manifest must bind:

- development, observation, and audit dataset IDs;
- the frozen source commit and full training configuration;
- the selected auxiliary weight;
- the audit uncertainty policy and complete acceptance result;
- feature and output schemas; and
- per-file SHA-256 checksums.

Only that audited artifact should become eligible for monitored shadow loading.
