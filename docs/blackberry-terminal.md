# BlackBerry Quant Terminal

The BlackBerry Quant Terminal is a legacy-device-compatible extension for
ML-Pricer. The intended client is a BlackBerry Bold 9780 on local Wi-Fi. The
device should only collect inputs and display compact results; pricing,
scenario analysis, model inference, storage, and validation stay on the
FastAPI backend.

## Current Implementation

The BlackBerry terminal demo loop currently supports:

- open `GET /bb/price`
- choose from all BB-enabled artifact-backed products
- submit a product-specific pricing form with `POST /bb/price`
- save a completed run in SQLite
- redirect to `GET /bb/result/{run_id}`
- render compact terminal-style pricing output
- open `GET /bb/scenario/{run_id}` from a base pricing result
- submit simple market shocks with `POST /bb/scenario/{run_id}`
- reprice the shocked request and display before/after output
- browse `GET /bb/recent-runs`
- reopen saved price and scenario runs
- reuse loaded model/scaler bundles from an in-memory process cache

The BlackBerry pages use server-rendered HTML, minimal CSS, no JavaScript, no
CDN assets, and no template dependency.

An optional sideloaded BlackBerry Java client spike is available under
`clients/blackberry-legacy/`. It is a launcher for the web terminal, not a
native pricing application.

The read-only status endpoints from the prior phase remain available:

- `GET /api/v1/products`
- `GET /api/v1/model-info`
- `GET /bb`
- `GET /bb/model-status`

## Supported Pricing Products

The BlackBerry pricing and scenario flows currently support every product with
an existing payoff class, known terminal fields, and committed model/scaler
artifacts:

- `phoenix` (`PHOENIX`)
- `accumulator` (`ACCUM`)
- `barrier` (`BARRIER`)
- `decumulator` (`DECUM`)
- `phoenix_stepdown` (`STEP-PHX`)
- `reverse_accumulator` (`REV-ACC`)

The legacy JSON `/price/` route still supports only its previous product set.
The BlackBerry flow uses the newer product registry metadata and does not alter
legacy route behavior.

## Product Parameters

Phoenix and Step-Down Phoenix:

- `S0`: spot
- `sigma`: volatility
- `r`: rate
- `T`: maturity
- `autocall_barrier_frac`: autocall barrier fraction
- `coupon_barrier_frac`: coupon barrier fraction
- `coupon_rate`: coupon rate
- `knock_in_frac`: knock-in barrier fraction
- `obs_count`: observation count

Accumulator, Decumulator, and Reverse Accumulator:

- `S0`: spot
- `sigma`: volatility
- `r`: rate
- `T`: maturity
- `upper_barrier_frac`: upper barrier fraction
- `lower_barrier_frac`: lower barrier fraction
- `participation_rate`: participation rate
- `obs_frequency`: observation frequency

Barrier:

- `S0`: spot
- `sigma`: volatility
- `r`: rate
- `T`: maturity
- `K`: strike
- `barrier_frac`: barrier fraction
- `option_type`: call or put selector

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
3. Choose a product such as `PHOENIX`, `ACCUM`, or `BARRIER`.
4. Keep the default inputs or edit them.
5. Submit `PRICE`.
6. Confirm the browser redirects to `/bb/result/{run_id}` and shows price,
   Monte Carlo estimate, error, latency, model, and path count.
7. Choose `[1] SCENARIO SHOCK`.
8. Enter one or more shocks and submit `PRICE SHOCK`.
9. Confirm the scenario result shows base price, shocked price, move, shocks,
   and a short summary.
10. Open `/bb/recent-runs` and confirm the price and scenario runs can be
   reopened.
11. Open `/bb/model-status` and confirm priced products move from `COLD` to
    `CACHED`.

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
positive. Shocked rate must remain non-negative because the current terminal
validation assumes non-negative rate ranges.

The same shock mapping is used for all BB-enabled products in this phase:
`spot_pct` changes `S0`, `vol_abs` changes `sigma`, and `rate_bps` changes `r`.

## Model Cache

`app/services/model_cache.py` keeps loaded model/scaler bundles in process
memory. The first price request for a product loads its artifacts from
`final/results/{product}/model.joblib` and `scaler.joblib`; later requests in
the same backend process reuse that bundle.

`GET /bb/model-status` shows:

- `READY`: artifacts are present and the product is enabled for BlackBerry
  pricing.
- `COLD`: artifacts are present, but the model has not been loaded in this
  backend process yet.
- `CACHED`: the model/scaler bundle is loaded in memory.
- `UNAVAIL`: required artifacts are missing.

The cache is intentionally simple: no Redis, no TTL, no external service. It
clears when the backend process restarts.

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

## Optional Legacy BlackBerry Client

The primary BlackBerry experience remains the `/bb` web terminal. The optional
legacy client in `clients/blackberry-legacy/` is a tiny BlackBerry OS 6 Java
launcher intended to open that terminal from a home-screen app icon.

It does not contain:

- pricing formulas
- model/scaler artifacts
- scenario logic
- product-specific pricing rules
- API keys or secrets

Current launcher behavior:

- shows an `ASHBERRY TERMINAL` screen
- displays a placeholder backend URL
- opens the URL in the native BlackBerry browser when ENTER is pressed
- also exposes an `Open Terminal` menu item

The committed URL is a placeholder:

```text
http://192.168.1.100:8000/bb
```

Before building for a real device, replace it locally with the PC LAN URL:

```text
http://<PC_LOCAL_IP>:8000/bb
```

Do not commit a machine-specific IP address.

### Legacy Tooling Status

The source-level spike was added without a local BlackBerry build toolchain on
PATH. These commands were not found on the current machine:

- `java`
- `javac`
- `rapc`
- `javaloader`
- `bbwp`

Because of that, the `.cod` file was not built and the app was not sideloaded
from this checkout.

Expected tools to confirm:

- legacy BlackBerry Java SDK/JDE or Eclipse plugin
- compatible Java JDK for that SDK
- BlackBerry Desktop Software or USB drivers
- `javaloader.exe` for command-line sideloading

Expected sideload shape after a successful build:

```powershell
javaloader.exe load AshBerryTerminal.cod
```

See `clients/blackberry-legacy/README.md` for the client source layout and
build notes.

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

- Local-only HTTP; no public exposure.
- No authentication or PIN enforcement yet.
- Model cache is in-process only and clears on backend restart.
- Scenario explanations are simple and rule-based.
- The sideloaded BlackBerry app is currently a source-level launcher spike and
  still needs legacy SDK build verification.
- BlackBerry browser rendering may require more simplification after device
  testing.

## Future Phases

- Compact payoff explanations and model-status polish.
- Phase 7: optional native or sideloaded BlackBerry wrapper.
