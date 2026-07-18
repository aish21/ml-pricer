# Phoenix live-shadow surrogate v4

> Historical specification. Phase 11 replaces the unstable two-replication
> promotion gate with the pre-registered uncertainty protocol in
> [`phoenix-uncertainty-v5.md`](phoenix-uncertainty-v5.md). The v4 results below
> remain research provenance.

`phoenix-surrogate-payoff-aware-v4` extends the v3 payoff-aware model with
robust candidate selection, targeted low-volatility training coverage, live
input-drift diagnostics, durable shadow observations, and out-of-time replay.
The feature and label definitions are unchanged from v3; the model and artifact
versions are new because their selection and operating controls changed.

The Monte Carlo reference remains the client-visible price. V4 is shadow-only.

## Why this phase exists

The final v3 audit passed its economic price, regime, barrier, auxiliary-output,
boundary, reconciliation, and Greek checks. It failed one calibration gate:
only 23.1% of audit predictions were within two label standard errors, versus a
40% requirement. The weakest cell was the low-volatility regime, where Sobol
label uncertainty is small and barrier transitions still create approximation
error.

V4 does not relax that gate. It changes the research process in two ways:

1. spend more development capacity on the diagnosed low-volatility/barrier
   region and select from more than one arbitrary neural initialization; and
2. measure the model on dated market states after training, so synthetic audit
   accuracy is not confused with live distribution validity.

## Version boundary

- feature schema: `phoenix-surrogate-features-v3`;
- label schema: `phoenix-piecewise-payoff-aware-label-v2`;
- dataset schema: `phoenix-surrogate-dataset-v3`;
- model: `phoenix-surrogate-payoff-aware-v4`;
- artifact schema: `phoenix-surrogate-artifact-v3`;
- historical live observation schema: `phoenix-shadow-observation-v1`.

The audit-approved price-first artifact now records
`phoenix-shadow-observation-v2`, which distinguishes the artifact that ran from
the artifact that was intended to run. The database migrates in place. The
frozen readiness policy is documented in
[`phoenix-shadow-promotion-readiness-v1.md`](phoenix-shadow-promotion-readiness-v1.md).

An old artifact fails closed against the v4 runtime even though its input
features have the same mathematical meaning.

## Focused development data

The default development run uses six markets per Phoenix contract:

- four retain the balanced low/normal/high/crisis schedule; and
- two add low-volatility cases concentrated around the knock-in, coupon, or
  autocall barrier.

The audit remains balanced with four markets per contract. Focused sampling is
a training decision and must not silently make the final exam easier.

Contract groups remain disjoint across train, validation, and development-test
splits. The separately seeded audit contains only `audit` rows.

## Candidate search

V4 always keeps a direct-price MLP as a control. The default payoff-aware search
combines:

- hidden-layer layouts `128x128`, `192x192`, `256x256`, and `256x128x64`; and
- two training seeds for each layout.

No candidate sees the audit during fitting or selection. Candidate selection
uses the external development-validation split and the frozen score

```text
score = validation MAE
      + 0.35 * worst validation market-regime MAE
      + 0.25 * worst validation moneyness-region MAE.
```

This makes the selection criterion explicit: average accuracy matters most,
but a candidate cannot win simply by sacrificing a difficult regime. The
manifest records every candidate, architecture, seed, score component, joint
regime-by-barrier cell, and the twenty worst development-validation cases.

The development-test split remains diagnostic. It does not select the model.

## Fresh audit rule

After architecture and seed selection, the chosen network is evaluated once on
a newly seeded balanced audit. If the audit is inspected and the model or gate
is changed, that audit becomes research data and cannot approve the revised
artifact. Another audit must be generated.

V4 retains the v3 gates and adds a joint regime-by-moneyness-cell MAE ceiling of
`0.04`. This catches a model that appears acceptable after averaging separately
over regimes and barrier regions.

## Live input diagnostics

For an in-domain request, runtime feature drift is measured relative to the
training mean and scale:

```text
z_j = abs((x_j - training_mean_j) / training_scale_j).
```

The shadow result reports the maximum standardized feature distance and up to
five features beyond four training standard deviations. Hard training-domain
violations still return `out_of_domain`; the z-score is an earlier warning for
distribution drift inside the rectangular domain.

These diagnostics are not statistical proof of drift. Features are correlated,
and synthetic training distributions are not Gaussian. They are inexpensive
operational indicators that identify inputs worth investigating.

## Durable shadow observations

When both shadow inference and telemetry are enabled, each term-structure price
stores a bounded SQLite observation containing:

- observation and market timestamps;
- symbol, artifact, model, and status;
- reference price and Monte Carlo standard error;
- surrogate price, absolute error, error-to-reference-SE, and latency;
- maximum standardized feature distance;
- classified volatility regime and nearest barrier region; and
- the versioned market term structure, product terms, and contractual reference
  spot required for replay.

Telemetry is best-effort. A database error sets `telemetry_recorded` to false
but cannot change or fail the Monte Carlo price.

Configuration:

```text
PHOENIX_SURROGATE_SHADOW_ENABLED=true
PHOENIX_SURROGATE_TELEMETRY_ENABLED=true
PHOENIX_SURROGATE_TELEMETRY_DB=data/surrogate_shadow_observations.sqlite3
PHOENIX_SURROGATE_TELEMETRY_MAX_ROWS=100000
```

The summary endpoint is:

```text
GET /api/v1/surrogate-shadow/metrics?limit=1000
```

Promotion-readiness evidence is reported separately:

```text
GET /api/v1/surrogate-shadow/promotion-readiness?limit=100000
```

This endpoint is read-only and cannot change the shadow-only runtime policy.

It reports status counts, MAE, p95 error, fraction within two reference standard
errors, latency, regime and barrier-region slices, symbol count, and feature
drift. The database is local research data and should not be exposed publicly.

## Out-of-time replay

Stored observations can be evaluated through the currently selected artifact
without changing their original Monte Carlo reference:

```powershell
python -m src.final.surrogate_replay `
  --limit 100 `
  --artifact-root data/surrogates/phoenix-v4/artifacts `
  --allow-unapproved
```

Replay answers: "How would the current artifact have behaved on the dated
states that an earlier artifact saw?" It does not reconstruct missing option
quotes or recalibrate the historical market. The versioned term structure is
the frozen replay input.

## Running the offline pipeline

Full focused development, balanced audit, candidate search, and export:

```powershell
python -m src.final.surrogate_pipeline full
```

For a small engineering smoke test:

```powershell
python -m src.final.surrogate_pipeline full `
  --n-contracts 64 `
  --markets-per-contract 6 `
  --audit-contracts 16 `
  --paths-per-replication 128 `
  --hidden-layers 32 32 `
  --no-candidate-search `
  --skip-lightgbm
```

Generated datasets, artifacts, and observation databases remain excluded from
Git.

## Promotion boundary

Offline `shadow_approved` means only that the selected artifact passed every
frozen synthetic audit gate. It does not authorize it to return prices.

A future production-price decision should additionally require:

- enough out-of-time observations across symbols, regimes, and barrier regions;
- stable live MAE and tail error against a higher-fidelity reference;
- no persistent feature drift or domain rejection cluster;
- licensed market data and production curve/volatility calibration;
- model rollback, alerting, and artifact-registry controls; and
- product-specific model-risk approval.

V4 supplies the measurement path; it does not claim those conditions exist.

## Recorded v4 experiment

The canonical experiment used:

- 1,024 contract groups and six markets per group: 6,144 development cases;
- the low-volatility/barrier-focused development profile;
- nine runtime candidates: one direct control and eight payoff-aware
  architecture/seed combinations;
- two scrambled 1,024-path Sobol replications and 252 steps per label; and
- a fresh, balanced 1,024-case audit using seed offset `3,000,003`.

The robust validation score selected `payoff_aware__128x128__seed42`. More
capacity did not win automatically:

| Candidate | Validation MAE | Robust score |
| --- | ---: | ---: |
| Payoff-aware 128x128, seed 42 | **0.007549** | **0.012708** |
| Payoff-aware 256x128x64, seed 143 | 0.007761 | 0.012950 |
| Payoff-aware 192x192, seed 143 | 0.007764 | 0.013119 |
| Direct-price 128x128, seed 42 | 0.011075 | 0.019060 |

The controlled direct comparison shows a 31.8% validation-MAE reduction from
payoff-aware supervision plus focused development coverage. The result also
shows why architecture search must be measured: the widest/deepest candidates
were not the best validators.

The selected model's fresh audit was:

| Audit metric | Value | Gate | Result |
| --- | ---: | ---: | --- |
| Price MAE | 0.006984 | <= 0.020000 | pass |
| Price p95 absolute error | 0.018843 | <= 0.050000 | pass |
| Price R-squared | 0.995880 | >= 0.900000 | pass |
| Worst market-regime MAE | 0.009908 | <= 0.030000 | pass |
| Worst barrier-region MAE | 0.007581 | <= 0.030000 | pass |
| Worst regime-by-barrier cell MAE | 0.013250 | <= 0.040000 | pass |
| Worst cashflow-component MAE | 0.012522 | <= 0.080000 | pass |
| Worst event-probability MAE | 0.016120 | <= 0.080000 | pass |
| Worst mean raw boundary violation | 0.001689 | <= 0.005000 | pass |
| Cashflow-to-price reconciliation MAE | 0.005551 | <= 0.080000 | pass |
| Within two label standard errors | 0.309570 | >= 0.400000 | **fail** |
| Delta / Vega / Rho sign agreement | 1.00 / 1.00 / 0.75 | >= 0.50 each | pass |

V4 therefore remains `research_only`. Relative to the final v3 audit, observed
MAE improved from `0.009989` to `0.006984`, p95 error from `0.027130` to
`0.018843`, and the within-two-SE fraction from 23.1% to 31.0%. Those audits use
different fresh seeds, so the validation comparison above remains the cleaner
controlled result.

The remaining calibration failure is concentrated in low volatility: that
regime has MAE `0.009908` but only 10.2% within two estimated label standard
errors. The worst joint cell is `low_vol:coupon` with MAE `0.013250`. The error
hotspot report also exposes a weakness in the gate itself: with only two Sobol
replications, some estimated label standard errors are extremely close to zero,
making error-to-SE ratios unstable. The frozen v4 gate is not changed after
seeing the audit. A future version should pre-register a better-calibrated
uncertainty experiment—more independent replications and/or a separately
justified economic error floor—and then use another untouched audit.

Identifiers:

- development dataset:
  `sha256:b3dd7e8d7760d4aa2a34b96f69252d6a5e6b01557f201e74d77934b05994981f`;
- audit dataset:
  `sha256:4f3cec8a68bb9576d0823e5237bd2b6c268bd859f9352213412fab57619a2183`;
- artifact:
  `sha256:102ff648926d1dbf0b2e6750f9121f8535b17d8b1e3fca56d6fbafa02f954c15`.
