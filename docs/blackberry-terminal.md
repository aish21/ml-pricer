# BlackBerry Quant Terminal

The BlackBerry Quant Terminal is a legacy-device-compatible extension for
ML-Pricer. The intended client is a BlackBerry Bold 9780 on local Wi-Fi. The
device should only collect inputs and display compact results; pricing,
scenario analysis, model inference, storage, and validation stay on the
FastAPI backend.

## Current Implementation

The BlackBerry terminal demo loop currently supports:

- open `GET /bb/price`
- submit a simple Phoenix pricing form with `POST /bb/price`
- save a completed run in SQLite
- redirect to `GET /bb/result/{run_id}`
- render compact terminal-style pricing output
- open `GET /bb/scenario/{run_id}` from a base pricing result
- submit simple market shocks with `POST /bb/scenario/{run_id}`
- reprice the shocked request and display before/after output
- browse `GET /bb/recent-runs`
- reopen saved price and scenario runs

The BlackBerry pages use server-rendered HTML, minimal CSS, no JavaScript, no
CDN assets, and no template dependency.

The read-only status endpoints from the prior phase remain available:

- `GET /api/v1/products`
- `GET /api/v1/model-info`
- `GET /bb`
- `GET /bb/model-status`

## Supported Pricing Products

The BlackBerry pricing and scenario flows support `phoenix` only.

That product was chosen first because it is supported by the existing `/price/`
path, has committed model/scaler artifacts under `final/results/phoenix`, and
is the clearest fit for the retro autocallable terminal demo.

## Run Locally

From the repository root:

```powershell
python -m uvicorn app.backend:app --reload --host 127.0.0.1 --port 8000
```

Then open:

- `http://127.0.0.1:8000/api/v1/products`
- `http://127.0.0.1:8000/api/v1/model-info`
- `http://127.0.0.1:8000/bb`
- `http://127.0.0.1:8000/bb/price`
- `http://127.0.0.1:8000/bb/recent-runs`
- `http://127.0.0.1:8000/bb/model-status`

## Desktop Browser Test

Use a desktop browser first and resize the window close to 480x360. Confirm
that `/bb`, `/bb/price`, `/bb/result/{run_id}`, `/bb/scenario/{run_id}`,
`/bb/recent-runs`, and `/bb/model-status` render as compact text screens.

Example flow:

1. Open `http://127.0.0.1:8000/bb`.
2. Choose `[1] PRICE NOTE`.
3. Keep the default Phoenix inputs or edit them.
4. Submit `PRICE`.
5. Confirm the browser redirects to `/bb/result/{run_id}` and shows price,
   Monte Carlo estimate, error, latency, model, and path count.
6. Choose `[1] SCENARIO SHOCK`.
7. Enter one or more shocks and submit `PRICE SHOCK`.
8. Confirm the scenario result shows base price, shocked price, move, shocks,
   and a short summary.
9. Open `/bb/recent-runs` and confirm the price and scenario runs can be
   reopened.

## Recent Runs

`GET /bb/recent-runs` shows the latest saved price and scenario runs from
SQLite. The page keeps full run IDs in links but displays compact IDs for the
BlackBerry screen.

Price runs show the saved model price. Scenario runs show the shocked price and
the compact ID of the base run.

## Scenario Shock Conventions

Scenario shock starts from a saved base run. The backend retrieves the original
request payload, shocks a copy of the market inputs, reprices with the existing
pricing service, saves a scenario run, and renders the comparison.

Supported shock fields:

- `spot_pct`: percentage shock to spot. `-10` means `S0 * 0.90`.
- `vol_abs`: absolute volatility shock in decimal terms. `0.05` means
  `sigma + 0.05`.
- `rate_bps`: rate shock in basis points. `50` means `r + 0.005`.

At least one shock is required. Shocked spot and volatility must remain
positive. Shocked rate must remain non-negative because the current trained
Phoenix model was built around non-negative rate ranges.

## Physical BlackBerry Test

1. Start the backend on all interfaces:

```powershell
python -m uvicorn app.backend:app --reload --host 0.0.0.0 --port 8000
```

2. Find the PC local IP:

```powershell
ipconfig
```

3. Confirm a desktop browser can open:

```text
http://<PC_LOCAL_IP>:8000/bb
```

4. Ensure Windows Firewall allows inbound port `8000` on the private network.

5. Connect the BlackBerry Bold 9780 to the same Wi-Fi network.

6. Open the BlackBerry browser:

```text
http://<PC_LOCAL_IP>:8000/bb
```

7. If the BlackBerry browser fails:

- use plain HTTP, not HTTPS
- use the raw IP address instead of a hostname
- confirm both devices are on the same Wi-Fi network
- confirm port `8000` is not blocked by the firewall
- test `http://<PC_LOCAL_IP>:8000/bb/model-status` first
- simplify the page further if rendering looks broken

Do not expose this MVP publicly. Treat the BlackBerry as an insecure legacy
client on a trusted local network only.

## Run Storage

The terminal uses SQLite run storage through `app/services/run_store.py`.
The default database path is `data/pricing_runs.sqlite3`, overrideable with:

```powershell
$env:MODEL_RUN_STORE_FILE="C:\path\to\pricing_runs.sqlite3"
```

Each saved run stores:

- `run_id`
- timestamp
- product key
- original request payload
- compact result payload
- model label
- latency in milliseconds
- `run_type` (`price` or `scenario`)
- `parent_run_id` for scenario runs

Storage was introduced before scenario shock because a scenario needs a stable
base pricing request/result to shock and reprice later.

## Current Limitations

- BlackBerry pricing and scenario shock support only Phoenix in this phase.
- Local-only HTTP; no public exposure.
- No authentication or PIN enforcement yet.
- Models are still loaded per pricing request, matching the existing backend
  behavior.
- Scenario explanations are simple and rule-based.
- No native sideloaded BlackBerry app yet.
- BlackBerry browser rendering may require more simplification after device
  testing.

## Future Phases

- Phase 6: compact payoff explanations and model-status polish.
- Phase 7: optional native or sideloaded BlackBerry wrapper.
