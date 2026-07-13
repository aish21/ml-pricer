# Neural Pricer

ML-powered exotic derivatives pricing with a retro BlackBerry quant terminal.

Neural Pricer is an experimental pricing platform for exotic and structured
derivatives. Phase 1 currently exposes a versioned, single-underlier Phoenix
contract priced by deterministic Monte Carlo through a FastAPI backend, a
Streamlit frontend, and a local BlackBerry Bold 9780-compatible terminal.
Existing LightGBM artifacts are retained as research outputs but are not served
because they predate the validated contract and feature schema.

This is an educational/demo system. It is not production trading
infrastructure, financial advice, or a risk system suitable for live capital
allocation.

## What It Does

- Prices the validated `phoenix-single-v1` contract per unit notional.
- Reports a deterministic Monte Carlo price, standard error, and 95% confidence
  interval.
- Serves pricing through FastAPI.
- Provides a Streamlit UI for desktop experimentation.
- Provides a plain HTML BlackBerry terminal at `/bb`.
- Keeps unvalidated product experiments visible as research code but out of the
  pricing interfaces.
- Supports BlackBerry scenario shocks:
  - spot percentage shock
  - volatility absolute shock
  - rate basis-point shock
- Stores pricing and scenario runs in SQLite.
- Shows recent runs and pricing-method status in a compact terminal UI.

The BlackBerry is a thin client. It does not run the model locally. It sends
simple HTTP requests over local Wi-Fi to the backend, which performs pricing,
scenario analysis, validation, and storage.

## Architecture

```text
BlackBerry Bold 9780
  -> local HTTP over Wi-Fi
  -> FastAPI backend
  -> versioned payoff + Monte Carlo reference layer
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

src/final/
  payoffs.py              Core payoff implementations
  inherited_payoffs.py    Extended payoff variants
  data_generator.py       Monte Carlo path/data generation
  model_trainer.py        LightGBM training and model loading
  evaluator.py            Model vs Monte Carlo evaluation
  reference_pricer.py     Deterministic reference price and uncertainty
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

- `POST /api/v1/price`
- `GET /api/v1/products`
- `GET /api/v1/model-info`

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
- The Phase 1 market model is flat-rate, constant-volatility GBM without a
  dividend/forward curve or volatility surface.
- Observation dates are evenly spaced and knock-in monitoring is discrete on
  simulated path steps.
- Legacy model/scaler artifacts remain committed but fail contract/feature
  compatibility checks and are not used for pricing.
- Scenario explanations are simple and rule-based.
- Old BlackBerry browser rendering may require further simplification after
  more device testing.

## Future Roadmap

- Add JSON `POST /api/v1/scenario` and product-specific request schemas.
- Add dated market snapshots, forward/discount curves, and an implied-volatility
  surface for arbitrary equity-like underlier symbols.
- Train and validate a replacement surrogate against the frozen Phoenix
  contract and an untouched test set.
- Improve payoff explanations and risk summaries.
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
