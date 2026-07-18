# Neural Pricer

ML-powered exotic derivatives pricing with a retro BlackBerry quant terminal.

Neural Pricer is an experimental pricing platform for exotic and structured
derivatives. The current product path exposes a versioned, single-underlier
Phoenix contract priced by deterministic Monte Carlo through a FastAPI backend,
a Streamlit frontend, and a local BlackBerry Bold 9780-compatible terminal. A
dated market snapshot separates arbitrary equity-like symbols from product
terms. A repeatedly validated, payoff-aware Phoenix v7 research surrogate can
run in monitored shadow mode against the reference price. A development-only
event-conditioned successor is implemented but was rejected by the repeated
validation gate. An observation-level hazard successor now preserves autocall
and coupon timing, but it also remains research-only after its smoke gate;
legacy artifacts remain ineligible.

This is an educational/demo system. It is not production trading
infrastructure, financial advice, or a risk system suitable for live capital
allocation.

## What It Does

- Prices the validated `phoenix-single-v1` contract per unit notional.
- Reports a deterministic Monte Carlo price, standard error, and 95% confidence
  interval.
- Accepts immutable dated market snapshots for arbitrary equity, ETF, and
  equity-index symbols through the product-focused API.
- Prices Phoenix with versioned piecewise rate, dividend, and volatility term
  structures while preserving flat-model compatibility.
- Builds a credential-free USD equity/ETF research term structure from official
  Treasury par yields, trailing distributions, and near-ATM yfinance option
  chains, with field-level provenance.
- Runs frozen-market spot, rate, dividend, volatility, and individual segment
  scenarios with paired Monte Carlo P&L uncertainty.
- Reports Delta, Gamma, Vega, Rho, and dividend rho with explicit units,
  confidence intervals, and common-random-number diagnostics.
- Generates group-disjoint, barrier-focused Phoenix training data with
  scrambled Sobol price, cashflow, and event labels plus Monte Carlo
  uncertainty.
- Selects direct-price versus payoff-aware multi-task MLPs on validation data,
  searches multiple architectures/seeds with a robust validation score, then
  gates the winner once on an independently generated audit dataset.
- Evaluates an offline event-conditioned mixture of experts without exposing a
  losing research candidate to serving or consuming another audit dataset.
- Replays exact Monte Carlo observation events and evaluates probability-aware
  autocall, coupon, and downside hazards on group-disjoint development data.
- Exports a checksum-verified pure-NumPy runtime; LightGBM remains an offline
  research baseline.
- Records opt-in dated shadow observations, feature-drift diagnostics, sliced
  live error metrics, and out-of-time artifact replay.
- Fetches credential-free research quotes through yfinance with bounded
  retries, caching, and freshness checks.
- Serves pricing through FastAPI.
- Provides a Streamlit UI for desktop experimentation.
- Provides a plain HTML BlackBerry terminal at `/bb`.
- Keeps unvalidated product experiments visible as research code but out of the
  pricing interfaces.
- Supports BlackBerry scenario shocks:
  - spot percentage shock
  - volatility absolute shock
  - rate basis-point shock
- Stores pricing, scenario, and risk runs in SQLite with calibration and model
  provenance.
- Shows recent runs and pricing-method status in a compact terminal UI.

The BlackBerry is a thin client. It does not run the model locally. It sends
simple HTTP requests over local Wi-Fi to the backend, which performs pricing,
scenario analysis, validation, and storage.

## Architecture

```text
BlackBerry Bold 9780
  -> local HTTP over Wi-Fi
  -> FastAPI backend
  -> yfinance research quote adapter
  -> Treasury/yfinance research calibration
  -> versioned payoff + Monte Carlo reference layer
  -> paired scenario and finite-difference risk layer
  -> optional checksum-verified surrogate shadow comparison
  -> SQLite run store
  -> compact terminal result page
```

The same backend also supports the existing Streamlit frontend and legacy JSON
pricing routes.

An optional sideloaded Java ME client lives under `clients/blackberry-legacy/`.
It renders a native terminal UI and calls compact `/api/bb/*` backend endpoints,
but it is still only a thin client. It is not a native pricing engine.

## Repository Structure

```text
app/
  backend.py              FastAPI app and legacy routes
  frontend.py             Streamlit desktop frontend
  api/v1.py               Versioned pricing and product/model status API
  bb/routes.py            BlackBerry terminal routes
  bb/rendering.py         Terminal HTML/formatting helpers
  services/               Product registry, pricing, scenario, run storage
  services/live_market_data.py  Normalized yfinance research-data adapter
  services/research_market_data.py  Treasury/options research calibration
  services/risk_service.py  Frozen-market scenarios and finite-difference risk
  services/surrogate_service.py  Fail-closed shadow artifact runtime
  services/surrogate_monitoring.py  Shadow telemetry, metrics, and replay store

src/final/
  payoffs.py              Core payoff implementations
  inherited_payoffs.py    Extended payoff variants
  data_generator.py       Monte Carlo path/data generation
  model_trainer.py        LightGBM training and model loading
  evaluator.py            Model vs Monte Carlo evaluation
  reference_pricer.py     Deterministic reference price and uncertainty
  surrogate_contract.py   Phoenix v7 feature/output/domain contract
  surrogate_data.py       Group-disjoint Sobol-labelled dataset generator
  surrogate_trainer.py    MLP/baseline evaluation and artifact promotion gates
  surrogate_event_conditioning.py  Offline event/conditional-value architecture
  surrogate_hazard_data.py  Observation-level event-label sidecars
  surrogate_hazard.py     Sequential probability-aware hazard research model
  surrogate_pipeline.py   Versioned generate/train command line entry point
  surrogate_replay.py     Replay dated shadow observations through an artifact
  pipeline.py             Training/evaluation orchestration

final/results/
  */model.joblib          Legacy research artifacts (not served unless compatible)
  */scaler.joblib         Saved feature scalers
  */results.json          Training/evaluation metadata

data/
  training/history data   Demo data and pricing history

docs/
  blackberry-terminal.md  BlackBerry terminal details and testing guide
  phoenix-single-v1.md    Versioned payoff and cashflow specification
  equity-market-snapshot-v1.md  Dated market-data and flat-GBM v2 specification
  equity-market-term-structure-v1.md  Piecewise carry/volatility specification
  equity-research-market-v1.md  Server-built USD research calibration
  equity-risk-analytics-v1.md  Paired scenarios and risk analytics
  phoenix-event-conditioned-research-v1.md  Development-only architecture result
  phoenix-observation-hazard-research-v1.md  Observation-event model result
  phoenix-robust-selection-v7.md  V7 repeated group-validation specification
  phoenix-focused-head-v6.md  Historical v6 focused-head specification
  phoenix-uncertainty-v5.md  Historical v5 uncertainty specification
  phoenix-live-shadow-v4.md  Historical v4 monitoring specification
  phoenix-payoff-aware-v3.md  Historical finance/ML lecture and v3 specification
  phoenix-surrogate-v2.md  Superseded v2 historical specification
  live-market-data.md      Research-provider behavior and usage boundary

clients/
  blackberry-legacy/      Optional Java ME native thin-client spike

unittests/
  pytest coverage for pricing, routes, storage, and terminal helpers
```

## Quickstart

Create and activate an environment:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

Install the complete local development environment:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The dependency groups are defined and pinned in `pyproject.toml`. For a smaller
environment, install only what you need:

```powershell
python -m pip install -e ".[api,test]"       # reference-pricer API
python -m pip install -e ".[frontend]"       # Streamlit UI
python -m pip install -e ".[training,test]"  # surrogate training
```

Run tests:

```powershell
python -m pytest -q
```

Run the same source checks as CI:

```powershell
python -m black --check app src unittests
python -m ruff check app src/final unittests
```

Run the Phoenix surrogate research pipeline:

```powershell
python -m src.final.surrogate_pipeline full
```

Generated datasets and artifacts are excluded from Git. See
[docs/phoenix-robust-selection-v7.md](docs/phoenix-robust-selection-v7.md)
before enabling shadow inference. The rejected event-conditioned experiment and
its reproducible development-only command are documented in
[docs/phoenix-event-conditioned-research-v1.md](docs/phoenix-event-conditioned-research-v1.md).
The observation-level successor is documented in
[docs/phoenix-observation-hazard-research-v1.md](docs/phoenix-observation-hazard-research-v1.md).

Start the FastAPI backend locally:

```powershell
python -m uvicorn app.backend:app --reload --host 127.0.0.1 --port 8000
```

Open the BlackBerry terminal in a desktop browser first:

```text
http://127.0.0.1:8000/bb
```

Start the Streamlit frontend:

```powershell
$env:API_URL="http://127.0.0.1:8000"
streamlit run app/frontend.py
```

Streamlit opens at:

```text
http://127.0.0.1:8501
```

## BlackBerry Terminal Demo Flow

1. Open `/bb`.
2. Select `[1] PRICE NOTE`.
3. Select `PHOENIX`.
4. Submit the product pricing form.
5. View `/bb/result/{run_id}`.
6. Select `[1] SCENARIO SHOCK`.
7. Enter one or more shocks.
8. View the scenario result.
9. Open `/bb/recent-runs` to revisit price and scenario runs.
10. Open `/bb/model-status` to check the active pricing method.

Currently enabled BlackBerry products:

- `phoenix` (`PHOENIX`)

Accumulator, barrier, decumulator, step-down Phoenix, and reverse accumulator
remain research definitions until they have versioned specifications and
quantitative validation.

The BlackBerry UI is intentionally plain:

- server-rendered HTML
- no JavaScript
- no external CSS or fonts
- compact monospace layout
- simple forms and links

## Optional Java ME BlackBerry Client

`clients/blackberry-legacy/` contains a minimal Java ME MIDlet spike named
`AshBerry Terminal`.

The MIDlet is optional. It does not run pricing locally, does not contain model
artifacts, and does not store secrets. Its job is to render a native compact UI
and call plain-text backend endpoints such as:

```text
http://<PC_LOCAL_IP>:8000/api/bb/ping
http://<PC_LOCAL_IP>:8000/api/bb/model-status
```

The existing `/bb` browser terminal remains available as the proven fallback
and manual testing route.

The local machine has Java/Javac, but does not yet have Java ME preverification
or emulator tooling, so the MIDlet source has not yet been compiled or
installed from this checkout. See
[`clients/blackberry-legacy/README.md`](clients/blackberry-legacy/README.md)
for the expected build and sideload workflow.

Expected build entry point after installing a Java ME Wireless Toolkit:

```powershell
$env:WTK_HOME="C:\WTK252"
cd clients\blackberry-legacy\midlet
.\build.ps1
```

## Testing On A BlackBerry Bold 9780

Run the backend on all local network interfaces:

```powershell
python -m uvicorn app.backend:app --reload --host 0.0.0.0 --port 8000
```

Find the PC local IP:

```powershell
ipconfig
```

Confirm from a desktop browser:

```text
http://<PC_LOCAL_IP>:8000/bb
```

Then connect the BlackBerry to the same Wi-Fi network and open:

```text
http://<PC_LOCAL_IP>:8000/bb
```

If the BlackBerry browser fails:

- use plain HTTP, not HTTPS
- use the raw IP address, not a hostname
- confirm both devices are on the same Wi-Fi network
- allow inbound port `8000` on the PC private-network firewall
- test `/bb/model-status` first
- simplify the page further if rendering looks broken

Do not expose the MVP publicly. Treat the BlackBerry as an insecure legacy
client on a trusted local network only.

## API And Routes

BlackBerry terminal:

- `GET /bb`
- `GET /bb/price`
- `POST /bb/price`
- `GET /bb/result/{run_id}`
- `GET /bb/scenario/{run_id}`
- `POST /bb/scenario/{run_id}`
- `GET /bb/recent-runs`
- `GET /bb/model-status`

Versioned API v1:

- `POST /api/v1/products/phoenix/price` (preferred)
- `POST /api/v1/products/phoenix/price/term-structure`
- `POST /api/v1/products/phoenix/price/market`
- `POST /api/v1/products/phoenix/price/research-market`
- `POST /api/v1/products/phoenix/scenario/term-structure`
- `POST /api/v1/products/phoenix/scenario/research-market`
- `POST /api/v1/products/phoenix/risk/term-structure`
- `POST /api/v1/products/phoenix/risk/research-market`
- `POST /api/v1/market-data/research-term-structure`
- `GET /api/v1/market-data/status`
- `GET /api/v1/market-data/quote`
- `POST /api/v1/price` (deprecated generic envelope)
- `GET /api/v1/products`
- `GET /api/v1/model-info`
- `GET /api/v1/surrogate-shadow/metrics`
- `GET /api/v1/runs`
- `GET /api/v1/runs/{run_id}`

Operations:

- `GET /health/live`
- `GET /health/ready`

Java ME plain-text API:

- `GET /api/bb/ping`
- `GET /api/bb/model-status`
- `GET /api/bb/products`

Legacy routes kept for compatibility:

- `POST /price/`
- `GET /history`
- `POST /history/append`
- `GET /training/{payoff_type}`
- `GET /payoff_explanation/{payoff_type}`
- `GET /`

## Current Limitations

- The MVP uses local-only HTTP.
- There is no authentication or PIN enforcement yet.
- The optional Java ME MIDlet is a source-level spike; it has not yet been built
  or sideloaded from this checkout.
- The Phase 6 research builder uses Treasury par yields as explicitly labelled
  zero-rate proxies and near-ATM Yahoo option quotes for carry and implied
  volatility. It is not a bootstrapped OIS curve or a volatility surface.
- The older `/price/market` route still combines a live spot with explicit
  request assumptions; use `/price/research-market` for the server-built
  research term structure.
- yfinance and Yahoo Finance data are intended for personal research use; this
  is not an authoritative or commercial redistribution feed.
- Observation dates are evenly spaced and knock-in monitoring is discrete on
  simulated path steps.
- Legacy model/scaler artifacts remain committed but fail contract/feature
  compatibility checks and are not used for pricing.
- Phoenix surrogate v7 remains shadow-only and disabled by default. Repeated
  development folds confirmed the focused candidate, but it remains the same
  predictor that failed the v6 audit and is therefore `research_only`. A run is
  loadable by default only when every price, region, auxiliary-output,
  uncertainty, and Greek gate passes on its independent audit.
- The event-conditioned research candidate is also `research_only`. It improved
  interpretability, but its repeated development score was worse than v7, so it
  was neither audited nor added to the runtime artifact format.
- The observation-hazard candidate preserves event timing and uses soft-label
  cross-entropy, but it did not beat the payoff-aware model on the smoke
  validation gate. Its full-development scale test has not yet been run.
- Scenario explanations are simple and rule-based.
- Phase 7 Greeks are finite-difference research estimates. Discontinuous
  barriers can produce noisy Gamma and bump sensitivity even with paired paths.
- Old BlackBerry browser rendering may require further simplification after
  more device testing.

## Future Roadmap

- Replace the research par-yield proxy with a bootstrapped collateral curve and
  fit an arbitrage-controlled volatility surface from a licensed feed.
- Expand surrogate labels and shadow telemetry across denser barrier regions,
  market regimes, and materially larger untouched datasets.
- Add observation-level autocall, survival, and coupon-event labels so the next
  surrogate can learn discrete event hazards instead of aggregate terminal
  probabilities. The research label/model contract now exists; the next
  milestone is its frozen full-development comparison.
- Add theta with explicit calendar, fixing, accrual, and market-roll rules.
- Add volatility-skew, credit/funding, and seasoned-trade state models.
- Add optional PIN or gateway-based access control for non-local deployments.
- Move datasets and model artifacts out of normal Git history and add a versioned
  artifact registry.
- Explore an optional WebWorks or native wrapper after the browser MVP is
  stable.

## Docker

Build and run backend/frontend containers:

```powershell
docker compose up --build
```

Research market data works without credentials or configuration. See
[docs/live-market-data.md](docs/live-market-data.md) for data-quality and usage
limits.

The optional surrogate shadow uses `PHOENIX_SURROGATE_SHADOW_ENABLED` and
`PHOENIX_SURROGATE_DIR`. Unapproved artifacts remain blocked unless the
research-only `PHOENIX_SURROGATE_ALLOW_UNAPPROVED` override is set. See
[docs/phoenix-robust-selection-v7.md](docs/phoenix-robust-selection-v7.md).
`PHOENIX_SURROGATE_TELEMETRY_ENABLED` controls the bounded local observation
store used by the metrics endpoint and replay command.

The API and frontend images install separate dependency groups. The API image
contains the NumPy reference runtime but not LightGBM, Optuna, XGBoost, or
CatBoost. Compose waits for `/health/ready` before starting the frontend.

Backend:

```text
http://127.0.0.1:8000
```

Streamlit:

```text
http://127.0.0.1:8501
```

See [docs/DOCKER_WINDOWS.md](docs/DOCKER_WINDOWS.md) for Windows-specific
Docker notes.

## Notes For Contributors

- Treat payoff or market-model changes as new contract/model versions with
  updated quantitative regression tests.
- Keep BlackBerry pages plain and local-network friendly.
- Do not put secrets on the BlackBerry.
- Prefer small service-layer changes over route-level pricing logic.
- Keep generated datasets and model binaries out of new commits. Follow
  [the Phase 2 cleanup plan](docs/phase-2-repository-cleanup.md) before removing
  the currently tracked copies.
