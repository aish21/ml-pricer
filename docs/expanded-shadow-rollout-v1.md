# Expanded-product shadow rollout v1

Phoenix v3 and the barrier reverse convertible can now run their sealed-audit
winners beside the Monte Carlo reference. This is an evidence-collection
release, not a pricing promotion: the API always returns the Monte Carlo price,
and no code path can automatically promote either model.

## Runtime boundary

The training-side packaging command reads the trusted local `model.joblib`,
exports numerical LightGBM trees to deterministic compressed JSON, proves
prediction parity, and writes a pinned registry:

```powershell
python -m src.final.expanded_shadow_artifact
```

The backend image contains the safe tree export but does not contain the pickle,
joblib, LightGBM, or scikit-learn. At load time it checks the registry pin,
artifact ID, contract version, feature order, tree shape, size limits, runtime
policy, and SHA-256 checksum.

## Rollout controls

Each product has an independent kill switch and sample rate. Defaults are
deliberately disabled:

```powershell
$env:PHOENIX_V3_SHADOW_ENABLED="true"
$env:PHOENIX_V3_SHADOW_SAMPLE_RATE="0.05"
$env:BRC_V1_SHADOW_ENABLED="true"
$env:BRC_V1_SHADOW_SAMPLE_RATE="0.05"
$env:EXPANDED_SURROGATE_TELEMETRY_ENABLED="true"
```

`EXPANDED_SURROGATE_TIMEOUT_MS` defaults to 25 ms. Sampling is deterministic for
the product, contract, and frozen market ID, so retrying the same request does
not randomly move in and out of the sample.

Inference fails closed. A disabled, unsampled, incompatible, out-of-domain,
corrupt, timed-out, or failed model returns a shadow status only. It does not
fail pricing and does not modify `price`, `mc_price`, `pricing_method`, the
Monte Carlo confidence interval, or diagnostics.

## Eligible requests

The candidates cover USD equity-like underliers (`equity`, `etf`, or `index`)
inside the numeric training ranges recorded in each runtime manifest. Phoenix v3
requires evenly spaced observations and a linear autocall step-down. The reverse
convertible requires evenly spaced coupon dates. Other valid contracts remain
fully supported by Monte Carlo and are simply excluded from ML evidence.

## Evidence and replay

Telemetry is stored in a bounded SQLite database and includes product and
contract versions, artifact ID, symbol, market date, error, latency, domain
utilization, market regime, payoff region, and frozen market/contract inputs.

Read-only endpoints:

- `GET /api/v1/expanded-surrogate-shadow/status`
- `GET /api/v1/expanded-surrogate-shadow/metrics`
- `GET /api/v1/expanded-surrogate-shadow/promotion-readiness`
- `GET /api/v1/surrogate-shadow/evidence`

Controlled replay endpoint:

- `POST /api/v1/expanded-surrogate-shadow/replay/{product_key}?limit=100`

Replay re-evaluates the currently pinned artifact against stored frozen inputs;
it does not write telemetry or change runtime policy.

## Daily out-of-time campaign

The evidence campaign avoids depending on sporadic website traffic. Interactive
comparisons remain visible for diagnostics, but they never advance the review
gates. The campaign freezes one two-year research-market calibration for each
underlier, then applies the same versioned contract grid to that snapshot:

- 12 Phoenix v3 cases per underlier, covering broad, first-autocall,
  final-autocall, coupon, and knock-in regions;
- 12 reverse-convertible cases per underlier, covering broad, strike, and
  knock-in regions; and
- 4,096 Monte Carlo paths per reference label, with the exact seed, snapshot,
  artifact, contract, and campaign IDs recorded.

The default universe contains 12 liquid US equities and ETFs. One complete run
therefore records 144 comparisons per product. Fourteen successful market days
produce 2,016 comparisons per product. Calibration failures are reported and
skipped rather than replaced with guessed market inputs.

Preview the immutable plan without using the network:

```powershell
python -m src.final.expanded_shadow_campaign --plan-only
```

Run the complete campaign:

```powershell
python -m src.final.expanded_shadow_campaign
```

For a smaller operational check, explicitly override the universe:

```powershell
python -m src.final.expanded_shadow_campaign `
  --underlier SPY:etf `
  --underlier AAPL:equity
```

The campaign stores immutable market calibrations in
`data/market_snapshots.sqlite3`, comparisons in
`data/expanded_shadow_observations.sqlite3`, and an atomic daily report under
`data/expanded_shadow_campaigns/`. A repeated run restores the frozen snapshots
and skips case IDs already in the evidence database.

The `Expanded shadow evidence` GitHub workflow can also run the campaign
manually. Its weekday schedule remains a no-op until the repository variable
`EXPANDED_SHADOW_CAMPAIGN_ENABLED` is set to `true`. GitHub-hosted runs restore
their prior SQLite state from the Actions cache and publish a portable evidence
artifact. They do not silently write into the deployed backend database. To
show campaign data in a deployed Evidence Lab, run the same command in a job
that shares the backend's persistent `data/` volume.

## Review gates

Each product must independently collect at least 2,000 observations, including
1,800 successful evaluations backed by at least 4,096 Monte Carlo paths, 10
symbols, 10 market dates spanning 14 days, a single matching pinned artifact,
and at least 100
examples in every required market-regime and payoff-region slice. Live MAE must
be at most 0.015, P95 absolute error at most 0.040, P95 inference latency at most
5 ms, mean reference standard error at most 0.010, and the success rate at least
99.5%.

Passing every gate produces only `ready_for_human_review`. It leaves
`runtime_eligible` and `automatic_promotion_permitted` false. A pricing-role
change would require a separate design, independent review, and explicit code
change.
