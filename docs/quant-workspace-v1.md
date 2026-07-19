# Guided and Quant workspace v1

The Streamlit frontend is a pricing workspace for two audiences using the same
backend contracts:

- **Guided** is a five-step, child-level learning journey. It defines each idea,
  lets the learner move barriers and path wiggliness, and translates the final
  result into plain language.
- **Quant** exposes the seed, normalized parameters, contract and market
  identifiers, curve segments, bump sizes, and complete API evidence.

The modes never change the payoff or pricing model. They change presentation
and control density only.

## Workspace flow

Quant mode follows one compact sequence:

```text
market → contract → reference price → diagnostics → risk → audit
```

Guided mode teaches the same economics in five small steps:

```text
pick something to watch → choose the clock → draw the rules
→ imagine possible futures → review and price
```

The Guided setup includes a live barrier playground and a deterministic toy
Monte Carlo chart. The toy paths explain simulation and never replace or alter
the real pricing inputs. Guided result tabs continue the lesson with
child-level names such as **Your answer**, **How the note works**, and
**How sure are we?** A zero-knowledge word shelf starts before step 1 and
unpacks note, underlier, issuer, notional, price, coupon, barrier, maturity,
volatility, and Monte Carlo without assuming prior finance knowledge.

Both modes provide a searchable common-underlier picker. The native Streamlit
select box filters as the user types a symbol or company/fund name and shows
suggestions inside that same editable control. Pressing Enter accepts a custom
Yahoo Finance symbol when no suggestion matches. Arbitrary equity/ETF support
is preserved; manual markets can also select indices.

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

The workspace uses a theme-aware cosmic visual language, with separate light,
dark, and sidebar palettes. Native widgets inherit the active Streamlit theme
instead of receiving hardcoded foreground/background colors. The palette uses:

- cyan for live market inputs;
- violet for contractual references;
- warm gold for autocall events;
- teal for coupons;
- rose only for downside risk; and
- tabular numeric presentation for prices and risk measures.

Each Guided chart includes a reading key that states the question, what to look
for, and why the picture matters. The primary visuals are:

1. price versus par with a 95% Monte Carlo error bar;
2. a zoned, vertical live-spot/reference/barrier map;
3. remaining observation timeline;
4. dual-axis deterministic rate, dividend, and volatility curves;
5. nested Monte Carlo convergence with confidence band;
6. expected PV by cashflow source with value shares;
7. payoff distribution shown as path shares with tail and median markers;
8. a common-random-number spot/volatility surface colored by change from base;
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
app/ui/guided.py      five-step beginner learning journey
app/ui/payloads.py    pure request and contract transformations
app/ui/inputs.py      Guided/Quant configuration forms
app/ui/underliers.py  searchable catalog and custom-symbol fallback
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
$env:API_PUBLIC_URL="http://127.0.0.1:8000"
streamlit run app/frontend.py
```

Then open `http://127.0.0.1:8501`.

`API_URL` is the address used by the Streamlit server. `API_PUBLIC_URL` is the
backend/docs address shown to the browser. Docker Compose therefore connects
the frontend to `http://backend:8000` internally while displaying
`http://localhost:8000/docs` to the local user.
