# Expanded surrogate experiment v2

`expanded-surrogate-experiment-v2` is the follow-up ML study for Phoenix v3 and
the barrier reverse convertible. It improves how candidates are trained and
selected without changing the audit thresholds or the runtime pricing policy.

## What changed

- 60% of sampled contracts are concentrated near a payoff boundary; the other
  40% still cover the wider contract domain.
- Each model receives distances from spot to the barriers that drive its
  cashflows.
- Development, validation, and audit datasets have separate deterministic
  seeds.
- Three LightGBM candidates are compared on validation data. The audit is used
  once, after selection.

The audit still uses 300 contracts per product and 4,096 Monte Carlo paths per
label. Development uses 2,500 contracts, validation uses 500, and both use
1,024 paths per label.

## Unchanged audit gates

| Check | Requirement |
|---|---:|
| MAE | <= 0.015 |
| P95 absolute error | <= 0.040 |
| R-squared | >= 0.90 |
| Mean audit-label standard error | <= 0.010 |
| Median single-row inference latency | <= 5 ms |

## Result

Both selected candidates passed the fresh sealed audit.

| Product | MAE | P95 error | R-squared | Median latency |
|---|---:|---:|---:|---:|
| Phoenix v3 | 0.01409 | 0.03846 | 0.9821 | 0.107 ms |
| Barrier reverse convertible | 0.01328 | 0.03525 | 0.9830 | 0.113 ms |

Both winners were the balanced L1 candidate. They are stored as research
packages so the results can be inspected and replayed. They are not served:
every manifest keeps `runtime_approved = false`, and Monte Carlo remains the
reference price.

The machine-readable report is
`final/research_candidates/experiment_summary.json`.

## Reproduce

```powershell
python -m src.final.expanded_surrogate_experiment `
  --development-samples 2500 `
  --validation-samples 500 `
  --audit-samples 300 `
  --development-paths 1024 `
  --audit-paths 4096 `
  --trees 1000
```
