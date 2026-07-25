# Expanded surrogate training: Phase 1

Phase 1 improves the labels used to train the Phoenix v3 and barrier
reverse-convertible surrogates. It deliberately keeps the existing balanced
LightGBM model and feature sets fixed. That makes this a controlled experiment:
if performance moves, the label protocol caused the move rather than a new
model architecture.

This is a research experiment. It does not package a serving artifact, approve
a model, change shadow traffic, or alter the Monte Carlo price shown by the
application.

## The experiment in one picture

```text
contract + four-part market curve
                 |
       production time grid
   (252 steps + exact event dates)
                 |
  8 independently scrambled Sobol runs
                 |
 price label + standard error + 95% interval
                 |
   fixed LightGBM, with and without weights
                 |
 compare both with the incumbent on fresh data
```

## Finance lecture: what is the model learning?

A structured-product price is the present value of its possible future
cashflows. Under the risk-neutral pricing measure, the reference engine
estimates

```text
V(0) = E^Q [ sum over k of DF(0, t_k) * Cashflow(t_k) ].
```

`DF(0, t_k)` discounts a future payment back to today. The cashflow depends on
the simulated underlier path:

- A Phoenix can pay conditional coupons, autocall early, or suffer
  knock-in-linked loss at maturity.
- A barrier reverse convertible pays fixed coupons and can return less than
  par if its barrier and final-level rules are met.

The path simulator uses risk-neutral geometric Brownian motion with
time-dependent inputs:

```text
dS(t) / S(t) = [r(t) - q(t)] dt + sigma(t) dW(t).
```

Here `r` is the risk-free rate, `q` is the dividend yield, and `sigma` is
volatility. Phase 1 represents each as a four-part piecewise curve. For one
time interval, the simulator uses the exactly integrated form

```text
log(S_next / S_now)
  = integral(r - q - 0.5 * sigma^2) dt
  + sqrt(integral(sigma^2) dt) * Z.
```

That is closer to the production pricer than the old flat-market labels.

### Why the event dates must be on the grid

The Phoenix payoff checks the underlier on contractual observation dates. If a
quarterly observation sits between two simulation points and the code rounds
to a nearby point, the teacher can apply the wrong barrier decision. Phase 1
starts with the production 252-step grid and inserts every observation or
coupon date exactly. There may therefore be slightly more than 252 effective
steps.

## ML lecture: labels are measurements, not perfect truth

The LightGBM surrogate is the student. Monte Carlo is the teacher. For a
contract-market feature vector `x_i`, the desired target is the true
risk-neutral price `f(x_i)`, but a finite simulation only provides

```text
y_i = f(x_i) + simulation noise.
```

This is noisy-label supervised learning. The amount of noise is not constant.
Contracts near a barrier can have much more variable payoffs than quiet,
deep-in-the-money contracts. In ML terms, the label noise is
**heteroscedastic**.

### Randomized quasi-Monte Carlo

Ordinary Monte Carlo uses pseudorandom points. A Sobol sequence spreads points
through the simulation cube more evenly, which often estimates a smooth
expectation more efficiently. A deterministic Sobol run alone does not give a
clean sampling-error estimate, so Phase 1 independently scrambles the Sobol
sequence eight times.

For replication means `m_1, ..., m_R`, the stored label is

```text
label = mean(m_1, ..., m_R)
SE(label) = sample_standard_deviation(m_1, ..., m_R) / sqrt(R).
```

The 95% interval is

```text
label +/- t_(R-1, 97.5%) * SE(label).
```

The Student-t multiplier is used because there are only eight independent
replications. Importantly, uncertainty is calculated between independently
scrambled replication means, not by pretending that dependent Sobol points are
ordinary independent paths.

Development and validation labels use `8 x 128 = 1,024` paths. Evaluation
labels use `8 x 512 = 4,096` paths.

### Uncertainty-aware sample weights

The primary Phase 1 candidate gives a cleaner label more influence:

```text
raw_weight_i = 1 / (SE_i^2 + variance_floor).
```

The raw weights are capped around their median and then normalized to have mean
one. The floor prevents division by zero; the cap prevents a few unusually
quiet labels from controlling the whole fit.

With LightGBM's L1 objective, the fitted model approximately minimizes

```text
sum over i of weight_i * abs(y_i - model(x_i)).
```

This weighting is a hypothesis, not an automatic improvement. High-uncertainty
contracts are often the economically interesting barrier cases. Phase 1
therefore trains an otherwise identical unweighted model as an ablation. The
comparison tells us whether weighting improves generalization or merely makes
the learner avoid difficult regions.

## Programming lecture: how the experiment stays trustworthy

The implementation in
`src/final/expanded_surrogate_phase1.py` uses several reproducibility patterns:

- Frozen dataclasses make configuration, sampled cases, and labels explicit
  value objects rather than mutable bags of state.
- Development, validation, and evaluation use separate random streams.
- `numpy.SeedSequence` derives an independent, reproducible seed from the label
  seed, sample number, and replication number.
- Every dataset stores its feature matrix, labels, uncertainty, replication
  means, effective step counts, and serialized market and contract inputs.
- Dataset and candidate IDs are SHA-256 hashes of their deterministic contents
  and protocol metadata.
- Volatile values such as timestamps, runtime duration, file paths, and latency
  measurements do not affect experiment identity.
- JSON and model files are written through a temporary file and then replaced,
  avoiding half-written outputs.
- The output directory is under ignored research data, never under runtime
  shadow artifacts.

The tests cover configuration failure, exact contractual event times,
deterministic dataset identity, uncertainty-weight behavior, both products,
artifact persistence, and the non-promoting boundary.

## Experimental contract

| Item | Phase 1 choice |
|---|---|
| Products | Phoenix v3 and barrier reverse convertible v1 |
| Development / validation / evaluation | 2,500 / 500 / 300 contracts per product |
| Grid | Production 252 steps plus exact contract event dates |
| Market | Four piecewise rate, dividend, and volatility segments |
| Label paths | 1,024 for development/validation; 4,096 for evaluation |
| Replications | Eight independently scrambled Sobol runs |
| Learner | Existing balanced LightGBM L1 configuration |
| Comparison | Incumbent v2, Phase 1 unweighted, Phase 1 weighted |
| Serving effect | None |

## Deliberate limits

Phase 1 fixes the teacher before expanding the student:

- It keeps the existing numerical domain: volatility remains 10%-50% and
  dividend yield remains 0%-4%. Known high-volatility and high-dividend
  out-of-domain cases belong in Phase 2.
- It keeps the v2 feature vectors. Four-part curves are currently summarized
  into equivalent flat rate, dividend, and volatility inputs for the learner.
  Adding curve-shape features is a later controlled change.
- It keeps geometric Brownian motion. Local volatility, stochastic volatility,
  jumps, and historical regime calibration are later modelling questions.
- It does not use evaluation results to promote a runtime model.

## Run it

A small wiring check:

```powershell
python -m src.final.expanded_surrogate_phase1 `
  --development-samples 12 `
  --validation-samples 6 `
  --evaluation-samples 8 `
  --paths-per-replication 8 `
  --evaluation-paths-per-replication 16 `
  --label-replications 2 `
  --trees 5
```

The canonical Phase 1 study:

```powershell
python -m src.final.expanded_surrogate_phase1
```

Generated datasets, candidates, and reports live under
`data/expanded_surrogate_phase1/` and are not committed. The source protocol
and tests are committed; large reproducible research outputs are not.

## Measured result

The canonical run completed in 632.95 seconds with experiment ID
`sha256:a95b4994ff92a461d26a958a3bc972cdc6c6b31b76d6701a1f9ec574696b898c`.
These are results against the fresh 4,096-path evaluation labels:

| Product and model | MAE | P95 absolute error | RMSE |
|---|---:|---:|---:|
| Phoenix incumbent v2 | 0.01451 | 0.04030 | 0.01994 |
| Phoenix Phase 1 unweighted | 0.01401 | 0.03797 | 0.01929 |
| Phoenix Phase 1 weighted | 0.02161 | 0.06513 | 0.03192 |
| Reverse convertible incumbent v2 | 0.01370 | 0.03994 | 0.02253 |
| Reverse convertible Phase 1 unweighted | 0.01284 | 0.04034 | 0.01988 |
| Reverse convertible Phase 1 weighted | 0.02181 | 0.06362 | 0.04050 |

The unweighted retraining produced a modest improvement. Relative to the
incumbent, Phoenix evaluation MAE fell 3.5% and P95 error fell 5.8%. The reverse
convertible's MAE fell 6.2% and RMSE fell 11.8%, although its P95 error rose
about 1.0%.

The inverse-variance weighting hypothesis was rejected. It made evaluation MAE
48.9% worse than the incumbent for Phoenix and 59.2% worse for the reverse
convertible. The strongest deterioration appeared in the highest-uncertainty
quartile. A likely explanation is that uncertainty is also a signal for
economically difficult barrier cases; aggressively downweighting those labels
teaches the model to neglect exactly the nonlinear region that needs more
capacity. That is an inference from the diagnostics, not a proof of causation.

The evaluation-label mean standard errors were 0.00121 for Phoenix and 0.00084
for the reverse convertible, so this run gives a much clearer view of model
error than the old single-run labels.

The conclusion is intentionally conservative: keep the incumbent serving
policy unchanged, retain the Phase 1 unweighted result as research evidence,
discard inverse-variance weighting in its current form, and address
domain/feature coverage before attempting promotion.
