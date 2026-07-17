# Phoenix payoff-aware surrogate v3

> Historical specification: v3 established payoff-aware supervision and the
> sealed audit. It is superseded operationally by
> [`phoenix-live-shadow-v4.md`](phoenix-live-shadow-v4.md), which adds robust
> candidate selection and live shadow monitoring.

This document is both the operating specification for
`phoenix-surrogate-payoff-aware-v3` and a short university-style lecture on why
the model is designed this way. The system remains a research pricer: the
Monte Carlo reference is always the returned price, and the surrogate runs only
as a measured shadow comparison.

The v3 model is generic across single equity-like underliers. It does not learn
a ticker such as SPY or AAPL. It learns a normalized pricing function for the
versioned `phoenix-single-v1` payoff under the versioned
`equity-gbm-piecewise-v1` market model. Extending to a new payoff or a basket is
a new contract and label problem, not merely another ticker in this model.

## 1. Learning objectives

After reading this document, you should be able to explain:

1. why derivative pricing uses a risk-neutral expectation rather than a
   forecast of the asset's real-world return;
2. how a Phoenix note decomposes into coupons, early redemption, protected
   maturity redemption, and downside redemption;
3. why barriers make the price function harder to learn than a vanilla option;
4. how Monte Carlo labels, standard errors, features, and neural-network outputs
   fit together;
5. why multi-task learning can improve a price model;
6. why model selection and final audit evaluation must use different data; and
7. what must change before this architecture can support another complex
   payoff safely.

## 2. The finance problem

### 2.1 Risk-neutral valuation

For a derivative with discounted pathwise cashflows, the model price is

```text
V(0) = E^Q[sum_k D(0,t_k) C_k(path)].
```

`Q` is the risk-neutral probability measure, `C_k` is a cashflow at time `t_k`,
and `D(0,t_k)` is its discount factor. This is not a claim that the underlier's
actual expected return equals the risk-free rate. It is the no-arbitrage
pricing measure used after the market model has been calibrated.

Within each deterministic term-structure segment, this project simulates an
equity-like underlier as

```text
dS_t / S_t = (r_t - q_t) dt + sigma_t dW_t^Q,
```

where `r_t` is the continuously compounded risk-free rate, `q_t` is the
dividend yield, and `sigma_t` is volatility. The rate, dividend, and volatility
can change between segments. The implementation integrates these quantities
over time, so it consumes curves rather than pretending that the market is
flat.

The model is still deliberately simpler than an institutional equity exotics
model. It has no volatility smile, local or stochastic volatility, jumps,
stochastic rates, issuer credit, funding spread, or gap-risk reserve.

### 2.2 Phoenix cashflows

At each observation date, while the note is alive:

1. it pays a coupon when the underlier is at or above the coupon barrier;
2. it returns principal and terminates when the underlier is at or above the
   autocall barrier.

If the note survives to maturity, it either returns principal or participates
in the underlier loss. The loss applies only when the knock-in barrier was
touched and the final underlier level is below the contractual reference spot.
The complete product rules remain in
[`phoenix-single-v1.md`](phoenix-single-v1.md).

V3 asks the reference pricer to produce four discounted pathwise components:

```text
Phoenix PV
  = coupon PV
  + autocalled principal PV
  + protected maturity principal PV
  + downside maturity redemption PV.
```

It also produces two event indicators per path: whether the note autocalled and
whether it finished in downside redemption. Averaging the indicators estimates
the corresponding risk-neutral event probabilities.

This decomposition matters because two notes can have similar prices for very
different reasons. One might be almost certain to autocall; another might carry
a large coupon but substantial downside exposure. A single price label hides
that structure.

### 2.3 Why barriers are difficult

A vanilla option payoff has a kink. A Phoenix contains several path-dependent
switches:

- crossing the coupon barrier changes a cashflow;
- crossing the autocall barrier terminates all later cashflows;
- touching the knock-in barrier changes the maturity rule; and
- the final level determines the amount of downside redemption.

Small market moves near a barrier can therefore change the distribution of
future cashflows sharply. The true price is smoother than an individual path's
payoff because it is an expectation, but its sensitivities can still change
quickly. Random training points spread uniformly across a wide domain waste
many examples far away from these economically important boundaries. V3 mixes
broad sampling with spot observations concentrated around the knock-in,
coupon, and autocall barriers, and reports audit errors separately for those
regions.

### 2.4 Underlier-generic is not model-generic

Changing from one USD equity or ETF ticker to another changes the market state:
spot, dividend curve, volatility term structure, and observation timestamp. It
does not change the Phoenix formula. This is why normalized inputs allow one
model to cover many single equity-like underliers.

Changing from a single stock to a worst-of basket is different. A basket needs
multiple spots, volatility surfaces, dividends, and a correlation model.
Changing from a Phoenix to an accumulator changes the cashflow rules and the
useful auxiliary targets. Those are new pricing contracts and should receive
new datasets, schemas, models, tests, and audit gates.

## 3. Turning the pricer into an ML problem

### 3.1 Supervised approximation

The Monte Carlo engine defines an expensive function

```text
y = f(product terms, market state).
```

The neural network learns a faster approximation `f_theta(x)`. This is
supervised regression: `x` is a numerical feature vector and `y` is a reference
label. The network does not discover the product contract. The versioned
reference pricer supplies that contract.

This distinction is important for live data. We are not training a time-series
model to predict tomorrow's spot. We are training a cross-sectional pricing
map. A fresh live or delayed market snapshot is transformed into `x`, and the
network estimates the price conditional on that state.

### 3.2 The 24 input features

The v3 feature contract contains:

- current spot divided by contractual reference spot;
- log distance from spot to each of the three barriers;
- maturity, all barrier fractions, coupon per observation, and observation
  count;
- maximum contractual coupon amount and coupon rate per year; and
- average zero rate, average dividend yield, and total variance at 25%, 50%,
  75%, and 100% of maturity.

Ratios make the model invariant to a common rescaling of the underlier. Log
barrier distances tell the network directly where the important discontinuity
surfaces lie. Total variance is `integral sigma(t)^2 dt`; it is more closely
related to diffusion uncertainty over a horizon than a raw point volatility.

The schema is closed and ordered. Adding, deleting, or silently reordering a
feature makes old weights incompatible.

### 3.3 Monte Carlo labels and label noise

For each case, v3 runs two independently scrambled Sobol replications. Sobol is
a quasi-Monte Carlo sequence designed to cover the unit cube more evenly than
independent pseudo-random points. Scrambling supplies independent replications
from which the generator estimates label uncertainty.

If replication estimates are `V_1, ..., V_R`, the stored label is their mean.
The standard error estimates how uncertain that finite-path label is. It does
not measure model error. A neural prediction can differ from the stored label
partly because the label itself is noisy, which is why the audit reports both
absolute errors and error relative to label standard error.

The four component means reconcile to the total price label exactly within
floating-point tolerance. Component and event standard errors are stored as
well. This makes the training data auditable and prevents the ML pipeline from
defining a second, subtly different Phoenix.

### 3.4 Direct learning versus multi-task learning

V3 trains two neural candidates with the same feature input and hidden-layer
shape:

```text
direct candidate:
    x -> shared hidden layers -> price

payoff-aware candidate:
    x -> shared hidden layers -> price
                              -> coupon PV
                              -> autocalled principal PV
                              -> protected maturity PV
                              -> downside maturity PV
                              -> autocall probability
                              -> downside probability
```

Every target is standardized using statistics from the training split. Without
standardization, a large-variance target could dominate the squared-error loss.

The payoff-aware network is an example of multi-task learning. Its hidden
layers must learn a representation useful for several related quantities. The
extra supervision can act as an inductive bias: instead of fitting price by any
available correlation, the representation is encouraged to encode the economic
events that create the price.

This is not guaranteed to help. Unrelated or noisy auxiliary tasks can cause
negative transfer. Therefore v3 trains both candidates and selects the one with
lower development-validation price MAE. The independent audit set is never
used for that selection.

The payoff-aware candidate has a dedicated price head. The quoted shadow price
is not forced to equal the sum of four imperfect component predictions. The
component sum is instead compared with the price head as a reconciliation
diagnostic and promotion gate.

### 3.5 What the MLP computes

After input standardization, each hidden layer performs

```text
h_(l+1) = ReLU(h_l W_l + b_l),
ReLU(z) = max(0,z).
```

The learned last layer is linear because these are continuous regression
targets. At inference, auxiliary cashflow outputs are projected onto the
nonnegative half-line and event outputs onto `[0,1]`. Projection onto the known
feasible interval cannot increase squared or absolute error when the reference
target lies in that interval. The audit still records the raw network's mean
and maximum boundary violations, so projection is a safety rule rather than a
way to conceal unstable outputs.

Training uses Adam to reduce average squared error plus L2 weight
regularization. Scikit-learn is used offline; deployment exports only arrays of
weights, biases, and normalization constants. Runtime inference is pure NumPy,
which keeps the API image smaller and reduces deserialization risk.

## 4. Experimental design

### 4.1 Group-disjoint development splits

Each sampled Phoenix contract is evaluated under four market regimes. All rows
for one contract stay in exactly one of train, validation, or development-test.
Otherwise the network could see the same barriers and coupon terms in training
and validation under slightly different market conditions, overstating
generalization.

- Training rows fit the weights.
- Validation rows select direct versus payoff-aware.
- Development-test rows provide diagnostics during research.

### 4.2 The sealed audit dataset

The final audit dataset is generated separately with different contract and
label seeds. Every row is marked `audit`; it does not participate in fitting,
early stopping, architecture choice, or candidate selection. Only the selected
candidate is evaluated on it.

This protects against adaptive overfitting. Repeatedly looking at a test score
while changing the model gradually turns that test set into training
information. A sealed audit is the ML equivalent of keeping the exam paper
closed until the course is over.

### 4.3 Metrics and gates

The audit records:

- MAE: average absolute price error;
- RMSE: square-root average squared error, which penalizes large misses more;
- 95th-percentile and maximum absolute error;
- R-squared: variance explained relative to predicting the audit mean;
- error by market regime and barrier-focused moneyness region;
- fraction within one and two Monte Carlo label standard errors;
- component, event-probability, output-bound, and reconciliation errors for a
  payoff-aware winner; and
- paired-path Delta, Vega, and Rho sign agreement.

Promotion requires every configured gate to pass. A failure leaves the artifact
`research_only`; thresholds are not loosened merely to make a run pass. Even an
approved artifact is only eligible for shadow execution in v3.

Greek validation deserves special care. A Greek is a derivative of price, so a
model can have small price error but a locally wrong slope. Common random
numbers are used for the bumped Monte Carlo estimates so that much of the path
noise cancels in the difference.

### 4.4 Recorded v3 experiment

The canonical run used 1,024 development contract groups with four markets per
contract (4,096 cases) and a separately seeded audit of 256 groups (1,024
cases). Each label used two scrambled Sobol replications of 1,024 paths and 252
time steps. The direct and payoff-aware candidates had two 128-unit hidden
layers.

Selection used development validation only:

| Candidate | Validation price MAE | Development-test price MAE |
| --- | ---: | ---: |
| Direct-price MLP | 0.013864 | 0.012040 |
| Payoff-aware MLP | **0.010319** | **0.009673** |
| LightGBM research baseline | 0.011840 | 0.011096 |

The payoff-aware candidate reduced validation MAE by about 25.6% relative to
the otherwise comparable direct network, so it was selected before the audit
was opened. LightGBM is diagnostic only and cannot become the runtime artifact.

After adding semantic output projection, a fresh audit with new contract and
Monte Carlo seeds produced:

| Final audit metric | Value | Gate | Result |
| --- | ---: | ---: | --- |
| Price MAE | 0.009989 | <= 0.020000 | pass |
| Price p95 absolute error | 0.027130 | <= 0.050000 | pass |
| Price R-squared | 0.992782 | >= 0.900000 | pass |
| Worst market-regime MAE | 0.014317 | <= 0.030000 | pass |
| Worst barrier-region MAE | 0.010666 | <= 0.030000 | pass |
| Worst cashflow-component MAE | 0.019002 | <= 0.080000 | pass |
| Worst event-probability MAE | 0.018826 | <= 0.080000 | pass |
| Worst mean raw boundary violation | 0.002183 | <= 0.005000 | pass |
| Cashflow-to-price reconciliation MAE | 0.011615 | <= 0.080000 | pass |
| Within two label standard errors | 0.231445 | >= 0.400000 | **fail** |
| Delta / Vega / Rho sign agreement | 0.9375 / 0.9375 / 0.6250 | >= 0.5000 each | pass |

The artifact is therefore `research_only`. Its headline accuracy improved, but
the system does not call twelve passes out of thirteen a release. The remaining
failure is concentrated in the fact that high-quality Sobol labels—especially
in low-volatility regimes—have very small estimated standard errors. The
surrogate is economically close but is not yet within twice that numerical
noise often enough. The next experiment should improve low-volatility/barrier
resolution or model capacity and must use another fresh audit after selection.

## 5. Connection to fresh market data

The product-focused API can build a dated research term structure from a
yfinance spot/option snapshot and Treasury yields. The resulting spot, rates,
dividends, and volatility segments flow through the same 24-feature contract as
synthetic training markets. If shadow mode is enabled, the API evaluates the
surrogate beside the Monte Carlo price and records latency and error; it never
replaces the returned reference price.

This is near-real-time research integration, not a production market-data feed.
yfinance is credential-free and convenient but has no service-level guarantee,
and the calibration uses near-ATM option-chain information rather than a full
arbitrage-controlled volatility surface. Production use would require licensed
data, explicit exchange timestamps, corporate-action handling, robust curve
construction, stale/crossed-market checks, and monitoring for distribution
drift and out-of-domain requests.

## 6. Worked conceptual example

Suppose a one-year note has quarterly observations, a 2% coupon per
observation, a 100% autocall barrier, an 80% coupon barrier, and a 60% knock-in
barrier.

- A path above 100% at the first observation pays the first coupon and returns
  principal immediately. Its value belongs to coupon PV plus autocalled
  principal PV.
- A path below 100% but above 80% can pay coupons and remain alive.
- A path that touches 60%, recovers above the reference spot by maturity, and
  never autocalls still receives protected principal.
- A path that touches 60% and finishes at 50% receives 50% of principal. Its
  redemption belongs to downside maturity PV.

The total price is the average discounted value across all such simulated
paths. The event heads learn how the probabilities of these regions move with
spot, barriers, volatility, carry, and time. The cashflow heads learn their
economic magnitudes. The price head learns their combined value.

## 7. Extending to more complex payoffs

The reusable idea is not "put every product into one neural network." It is a
versioned workflow:

1. freeze the legal/economic payoff definition, units, schedules, and state;
2. implement a deterministic reference pricer and pathwise cashflow
   decomposition beside the payoff code;
3. choose a market model capable of representing the product's material risks;
4. define normalized features and explicit domain bounds;
5. sample heavily near the product's discontinuities and rare loss regions;
6. generate noisy labels with recorded uncertainty;
7. train direct and economically supervised candidates;
8. select on validation, open a fresh audit once, and gate worst regimes; and
9. deploy only in shadow until live error and drift telemetry are credible.

Examples of useful product-specific heads include memory-coupon balance and
call time for a step-down Phoenix, accumulated quantity and knock-out time for
an accumulator, and each name's barrier state plus worst-of identity for a
basket. A worst-of product also requires correlations and multiple-underlier
features, so the current v3 contract cannot represent it safely.

## 8. Versioned contract and operation

The identifiers are:

- feature schema: `phoenix-surrogate-features-v3`;
- label schema: `phoenix-piecewise-payoff-aware-label-v2`;
- model: `phoenix-surrogate-payoff-aware-v3`;
- artifact schema: `phoenix-surrogate-artifact-v2`;
- dataset schema: `phoenix-surrogate-dataset-v3`.

Run the complete development/audit pipeline with:

```powershell
python -m src.final.surrogate_pipeline full
```

Datasets and model binaries are generated under
`data/surrogates/phoenix-v3/` and remain outside Git. The runtime verifies the
manifest versions, ordered feature/output names, file checksum, artifact ID,
training domain, and approval state before loading. Unknown, corrupt,
unapproved, or out-of-domain artifacts fail closed.

Set `PHOENIX_SURROGATE_SHADOW_ENABLED=true` to request shadow evaluation. A
`research_only` artifact also requires the explicit local research override
`PHOENIX_SURROGATE_ALLOW_UNAPPROVED=true`. Neither setting changes the price
returned to the client.

## 9. Current boundary

V3 improves the research model and creates a cleaner bridge to live snapshots
and richer payoffs, but it does not make the system production trading
infrastructure. The strongest next quantitative steps are a larger independent
audit, volatility-surface inputs, seasoned-trade state and real schedules,
licensed data, repeated out-of-time market replay, and live shadow drift/error
monitoring. New payoff families should follow the workflow above rather than
reuse Phoenix weights.
