# Guided and Quant workspace v1

The Streamlit frontend is a pricing workspace for two audiences using the same
backend contracts:

- **Guided** explains the product, progressively discloses numerical detail,
  and translates path statistics into plain language.
- **Quant** exposes the seed, normalized parameters, contract and market
  identifiers, curve segments, bump sizes, and complete API evidence.

The modes never change the payoff or pricing model. They change presentation
and control density only.

## Workspace flow

The screen follows one sequence:

```text
market → contract → reference price → diagnostics → risk → audit
```

The user first chooses a new issue or seasoned trade. A new issue uses the live
spot as its contractual reference under `phoenix-single-v1`. A seasoned trade
uses the explicit reference, remaining schedule, and historical knock-in state
from `phoenix-single-v2`.

The market can be:

- a server-built USD research term structure; or
- a transparent, caller-supplied flat rate/dividend/volatility market lifted
  into `equity-market-term-structure-v1`.

Successful prices and diagnostics remain in Streamlit session state across
widget reruns. Scenario and risk results are cleared only when a new base
valuation succeeds.

## Visual language

The workspace uses:

- blue for live market inputs;
- purple for contractual references;
- amber for autocall events;
- teal for coupons;
- red only for downside risk; and
- tabular numeric presentation for prices and risk measures.

The primary visuals are:

1. reference price with a 95% Monte Carlo error bar;
2. live-spot/reference/barrier ladder;
3. remaining observation timeline;
4. deterministic rate, dividend, and volatility curves;
5. nested Monte Carlo convergence with confidence band;
6. expected PV by cashflow source;
7. pathwise discounted-payoff distribution;
8. common-random-number spot/volatility valuation surface;
9. paired scenario waterfall; and
10. Greek estimates with standard-error bars and signal badges.

Legacy model feature importance is intentionally not a primary serving view.
The production price is still the Monte Carlo reference, and the approved
surrogate remains shadow-only.

## Diagnostic API

New-issue request:

```text
POST /api/v1/products/phoenix/diagnostics/term-structure
```

Seasoned-trade request:

```text
POST /api/v1/products/phoenix/diagnostics/seasoned/term-structure
```

Both accept a frozen market, either v1 terms or a v2 contract, a seed, a bounded
path count, optional nested convergence counts, and bounded spot/volatility
axes.

The response is versioned as:

```text
phoenix-reference-diagnostics-v1
```

It contains:

- nested price, standard error, and confidence-interval estimates;
- expected coupon, autocall-principal, protected-maturity, and downside PVs;
- autocall, downside, and expected-coupon statistics;
- payoff quantiles and a fixed-size histogram;
- spot/volatility surface cells with price, change, and standard error; and
- reproducibility metadata and a content-derived `diagnostic_id`.

## Numerical controls

- Diagnostic paths: 100 to 5,000.
- Surface axes: at most 11 values each.
- Total surface path evaluations: at most 200,000.
- Convergence points: at most 8.
- Raw Monte Carlo paths are never returned.
- Every surface cell uses the same normal draws.

The common draws reduce noise in cell-to-cell comparisons. They do not remove
model risk or turn the grid into a calibrated market volatility surface.

## Code structure

The former monolithic frontend is split into:

```text
app/frontend.py       small Streamlit entry point
app/ui/api_client.py  sanitized HTTP boundary
app/ui/payloads.py    pure request and contract transformations
app/ui/inputs.py      Guided/Quant configuration forms
app/ui/charts.py      pure Plotly figure builders
app/ui/results.py     pricing, diagnostics, risk, and audit views
app/ui/theme.py       visual system
app/ui/workspace.py   state and request orchestration
```

Pure payload, API-client, and chart transformations are unit tested. A
Streamlit application test verifies that the workspace starts without
contacting the backend.

## Run locally

Start the API:

```powershell
python -m uvicorn app.backend:app --reload --host 127.0.0.1 --port 8000
```

Start the workspace:

```powershell
$env:API_URL="http://127.0.0.1:8000"
streamlit run app/frontend.py
```

Then open `http://127.0.0.1:8501`.
