# Neural Pricer

ML-powered exotic derivatives pricing with a retro BlackBerry quant terminal.

Neural Pricer is an experimental pricing platform for exotic and structured
derivatives. It uses Monte Carlo simulation to produce reference prices and
trained ML surrogate models, currently LightGBM-based, to return fast pricing
estimates through a FastAPI backend, a Streamlit frontend, and a local
BlackBerry Bold 9780-compatible terminal interface.

This is an educational/demo system. It is not production trading
infrastructure, financial advice, or a risk system suitable for live capital
allocation.

## What It Does

- Prices structured/exotic products with reusable payoff classes.
- Compares model-backed prices against Monte Carlo reference results.
- Serves pricing through FastAPI.
- Provides a Streamlit UI for desktop experimentation.
- Provides a plain HTML BlackBerry terminal at `/bb`.
- Supports multiple artifact-backed products in the BlackBerry terminal.
- Supports BlackBerry scenario shocks:
  - spot percentage shock
  - volatility absolute shock
  - rate basis-point shock
- Stores pricing and scenario runs in SQLite.
- Shows recent runs and model status in a compact terminal UI.
- Caches loaded model/scaler bundles in backend process memory.

The BlackBerry is a thin client. It does not run the model locally. It sends
simple HTTP requests over local Wi-Fi to the backend, which performs pricing,
scenario analysis, validation, and storage.

## Architecture

```text
BlackBerry Bold 9780
  -> local HTTP over Wi-Fi
  -> FastAPI backend
  -> pricing/evaluator/model layer
  -> SQLite run store
  -> compact terminal result page
```

The same backend also supports the existing Streamlit frontend and legacy JSON
pricing routes.

An optional sideloaded legacy BlackBerry Java client lives under
`clients/blackberry-legacy/`. It is a thin launcher shell for the `/bb` web
terminal, not a native pricing engine.

## Repository Structure

```text
app/
  backend.py              FastAPI app and legacy routes
  frontend.py             Streamlit desktop frontend
  api/v1.py               Read-only product/model status API
  bb/routes.py            BlackBerry terminal routes
  bb/rendering.py         Terminal HTML/formatting helpers
  services/               Product registry, pricing, scenario, run storage

src/final/
  payoffs.py              Core payoff implementations
  inherited_payoffs.py    Extended payoff variants
  data_generator.py       Monte Carlo path/data generation
  model_trainer.py        LightGBM training and model loading
  evaluator.py            Model vs Monte Carlo evaluation
  pipeline.py             Training/evaluation orchestration

final/results/
  */model.joblib          Saved demo surrogate models
  */scaler.joblib         Saved feature scalers
  */results.json          Training/evaluation metadata

data/
  training/history data   Demo data and pricing history

docs/
  blackberry-terminal.md  BlackBerry terminal details and testing guide

clients/
  blackberry-legacy/      Optional BlackBerry OS 6 launcher spike

unittests/
  pytest coverage for pricing, routes, storage, and terminal helpers
```

## Quickstart

Create and activate an environment:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
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
3. Select a product.
4. Submit the product pricing form.
5. View `/bb/result/{run_id}`.
6. Select `[1] SCENARIO SHOCK`.
7. Enter one or more shocks.
8. View the scenario result.
9. Open `/bb/recent-runs` to revisit price and scenario runs.
10. Open `/bb/model-status` to check available artifacts and cache state.

Currently enabled BlackBerry products:

- `phoenix` (`PHOENIX`)
- `accumulator` (`ACCUM`)
- `barrier` (`BARRIER`)
- `decumulator` (`DECUM`)
- `phoenix_stepdown` (`STEP-PHX`)
- `reverse_accumulator` (`REV-ACC`)

The BlackBerry UI is intentionally plain:

- server-rendered HTML
- no JavaScript
- no external CSS or fonts
- compact monospace layout
- simple forms and links

## Optional Legacy BlackBerry Client

`clients/blackberry-legacy/` contains a minimal BlackBerry OS 6 Java launcher
spike named `AshBerry Terminal`.

The launcher is optional. It does not run pricing locally, does not contain
model artifacts, and does not store secrets. Its job is to open the existing
server-rendered terminal URL:

```text
http://<PC_LOCAL_IP>:8000/bb
```

This keeps the backend as the source of truth while allowing a future
home-screen icon on the BlackBerry.

The local machine used for this repo did not have the legacy BlackBerry Java
toolchain on PATH, so the launcher source has not yet been compiled or
sideloaded from this checkout. See
[`clients/blackberry-legacy/README.md`](clients/blackberry-legacy/README.md)
for the expected build and sideload workflow.

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

Read-only API v1:

- `GET /api/v1/products`
- `GET /api/v1/model-info`

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
- The optional BlackBerry Java launcher is a source-level spike; it has not yet
  been built or sideloaded from this checkout.
- Model/scaler artifacts are committed for demo usage.
- Model caching is in-process only and clears on backend restart.
- Scenario explanations are simple and rule-based.
- Old BlackBerry browser rendering may require further simplification after
  more device testing.

## Future Roadmap

- Add JSON `POST /api/v1/price` and `POST /api/v1/scenario`.
- Improve payoff explanations and risk summaries.
- Add optional PIN or gateway-based access control for non-local deployments.
- Improve model registry/versioning.
- Explore an optional WebWorks or native wrapper after the browser MVP is
  stable.

## Docker

Build and run backend/frontend containers:

```powershell
docker compose up --build
```

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

- Keep payoff formulas, Monte Carlo logic, evaluator semantics, and trained
  artifacts stable unless a change explicitly targets pricing behavior.
- Keep BlackBerry pages plain and local-network friendly.
- Do not put secrets on the BlackBerry.
- Prefer small service-layer changes over route-level pricing logic.
