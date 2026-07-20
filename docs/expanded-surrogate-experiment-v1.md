# Expanded surrogate experiment v1

`expanded-surrogate-experiment-v1` tests whether the richer Phoenix and the
barrier reverse convertible are ready for an ML shadow model. It cannot promote
or serve a model.

This report is kept as the original failed baseline. The follow-up is documented
in [expanded surrogate experiment v2](expanded-surrogate-experiment-v2.md).

## Protocol

- Development and audit datasets use disjoint deterministic seeds.
- Audit labels use 4,096 Monte Carlo paths per contract.
- The audit contains 300 contracts per product.
- Development uses 2,500 contracts and 1,024 paths per label.
- Both domains include live spot/reference moneyness, rates, dividends,
  volatility, maturity, product terms, and historical knock-in state.
- v3 additionally includes coupon memory, unpaid coupons, observation count,
  and first/final linear step-down barriers.
- The learner is a 1,000-round LightGBM L1 regressor.
- A model is written only when every frozen check passes.
- Passing would still create a research candidate with
  `runtime_approved = false`.

## Frozen gates

| Check | Requirement |
|---|---:|
| MAE | <= 0.015 |
| P95 absolute error | <= 0.040 |
| R-squared | >= 0.90 |
| Mean audit-label standard error | <= 0.010 |
| Median single-row inference latency | <= 5 ms |

## Result

Both candidates were rejected.

| Product | MAE | P95 error | R-squared | Median latency |
|---|---:|---:|---:|---:|
| Phoenix v3 | 0.016954 | 0.056088 | 0.9647 | 0.123 ms |
| Barrier reverse convertible | 0.018469 | 0.052879 | 0.9671 | 0.100 ms |

Label uncertainty and latency passed for both models. The remaining limitation
is tail approximation accuracy around discontinuous barrier and redemption
regions. No thresholds were relaxed, no model package was produced, and the
production price remains Monte Carlo.

The machine-readable decision is stored in
`final/research_candidates/experiment_summary.json` and appears in the
frontend ML Evidence Lab.

## Reproduce

```powershell
python -m src.final.expanded_surrogate_experiment `
  --development-samples 2500 `
  --audit-samples 300 `
  --development-paths 1024 `
  --audit-paths 4096 `
  --trees 1000
```
