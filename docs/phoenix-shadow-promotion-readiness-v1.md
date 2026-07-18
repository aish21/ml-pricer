# Phoenix shadow promotion readiness

This phase freezes the evidence requirements for reviewing the audit-approved
Phoenix surrogate after monitored shadow use. It does not promote the model,
change the `shadow-only` runtime policy, or make the surrogate client-visible.

## Frozen policy

- policy version: `phoenix-shadow-promotion-readiness-v1`;
- policy ID:
  `sha256:efbbe2e882555ba8720962090abdbb5d82807eeb258627a95e6058cbccb4dda5`;
- target artifact:
  `sha256:40025cc823b53e3e9a63571732dd54610776292b0c4d7da9d0824cfbc8b0c216`;
- evidence start: `2026-07-18T05:10:00Z`; and
- scope: new-issue, single-underlier Phoenix requests inside the frozen
  training domain.

The policy was fixed before collecting qualifying observations. Changing a
threshold produces a different policy ID and requires a new evidence exercise.

Seasoned trades are excluded. The current contract uses the valuation spot as
the contractual reference level, so it cannot truthfully test a trade whose
barriers were fixed against a different historical initial spot.

## Evidence gates

The report requires:

- 1,000 total and 800 successful observations;
- 500 distinct market-and-terms cases, so repeated identical calls cannot
  inflate the sample;
- five symbols;
- ten distinct market-data dates;
- at least fourteen calendar days between the oldest and newest successful
  observations;
- 25 successful observations in every required volatility regime and
  moneyness region; and
- ten successful observations in every required regime-region joint slice.

Required regimes are low, normal, and high volatility. Required regions are
broad, coupon, and autocall. Crisis observations remain valuable diagnostics,
but they are not mandatory because a short live window cannot guarantee a
crisis. Knock-in-near observations require a seasoned-trade contract and are
outside this policy's declared scope.

## Operations, quality, and drift gates

| Gate | Requirement |
| --- | ---: |
| Successful evaluation fraction | at least 90% |
| Successful rows using the exact approved artifact | 100% |
| Rows using the current observation schema | 100% |
| Out-of-domain fraction | at most 10% |
| Unavailable fraction | at most 1% |
| Runtime-error fraction | at most 1% |
| MAE | at most 0.015 |
| p95 absolute error | at most 0.040 |
| Within two reference standard errors | at least 80% |
| Worst required regime MAE | at most 0.020 |
| Worst required moneyness-region MAE | at most 0.020 |
| Worst required joint-slice MAE | at most 0.025 |
| p95 NumPy inference latency | at most 25 ms |
| Inputs above four training standard deviations | at most 5% |

The error gates allow degradation from the sealed synthetic audit while
remaining tighter than the original audit ceilings. The Monte Carlo uncertainty
coverage gate prevents noisy low-path reference labels from being interpreted
as exact truth.

## Observation integrity

Observation schema `phoenix-shadow-observation-v2` records:

- `artifact_id`: the artifact that actually ran; and
- `target_artifact_id`: the artifact that was intended to run.

This distinction retains unavailable and corrupt-artifact failures in the
operational evidence even though no model completed inference. Existing v1
SQLite stores migrate in place and backfill the target from the historical
artifact ID when one exists.

## Reading the report

With monitoring enabled:

```text
GET /api/v1/surrogate-shadow/promotion-readiness?limit=100000
```

Or from the local database:

```powershell
python -m src.final.surrogate_shadow_readiness `
  --monitoring-db data/surrogate_shadow_observations.sqlite3
```

The decision is one of:

- `insufficient_evidence`: one or more sample-breadth gates are incomplete;
- `not_ready`: evidence is sufficient, but an operations, quality, or drift
  gate failed; or
- `ready_for_review`: every frozen gate passed.

Even `ready_for_review` returns:

```json
{
  "runtime_eligible": false,
  "automatic_promotion_permitted": false
}
```

It triggers a human review, not a deployment. Any limited-canary phase requires
a separate runtime artifact/version, an explicit rollback plan, and another
code change.
