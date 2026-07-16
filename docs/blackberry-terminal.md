# BlackBerry Quant Terminal

The BlackBerry Quant Terminal is a legacy-device-compatible extension for
ML-Pricer. The intended client is a BlackBerry Bold 9780 on local Wi-Fi. The
device should only collect inputs and display compact results; pricing,
 scenario analysis, reference pricing, storage, and validation stay on the
FastAPI backend.

## Current Implementation

The BlackBerry terminal demo loop currently supports:

- open `GET /bb/price`
- choose the validated `PHOENIX` contract
- submit a product-specific pricing form with `POST /bb/price`
- save a completed run in SQLite
- redirect to `GET /bb/result/{run_id}`
- render compact terminal-style pricing output
- open `GET /bb/scenario/{run_id}` from a base pricing result
- submit simple market shocks with `POST /bb/scenario/{run_id}`
- reprice the shocked request and display before/after output
- browse `GET /bb/recent-runs`
- reopen saved price and scenario runs
- display deterministic reference prices with uncertainty information

The BlackBerry pages use server-rendered HTML, minimal CSS, no JavaScript, no
CDN assets, and no template dependency.

An optional sideloaded Java ME client spike is available under
`clients/blackberry-legacy/`. It renders a native terminal UI and calls compact
plain-text `/api/bb/*` backend endpoints. It is still not a native pricing
application.

The read-only status endpoints from the prior phase remain available:

- `GET /api/v1/products`
- `GET /api/v1/model-info`
- `GET /bb`
- `GET /bb/model-status`

## Supported Pricing Products

The BlackBerry pricing and scenario flows currently expose one validated,
versioned product:

- `phoenix` (`PHOENIX`)

Other payoff classes remain research-only. Their committed artifacts do not
declare the validated contract version and feature order, so they are not
eligible for serving. The legacy JSON `/price/` route now uses the same pricing
service and validation boundary as the BlackBerry flow.

## Product Parameters

Phoenix Single v1:

- `S0`: spot
- `sigma`: volatility
- `r`: rate
- `T`: maturity
- `autocall_barrier_frac`: autocall barrier fraction
- `coupon_barrier_frac`: coupon barrier fraction
- `coupon_rate`: non-memory coupon per observation
- `knock_in_frac`: knock-in barrier fraction
- `obs_count`: observation count

See `docs/phoenix-single-v1.md` for the complete cashflow and validation
specification.

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
3. Choose `PHOENIX`.
4. Keep the default inputs or edit them.
5. Submit `PRICE`.
6. Confirm the browser redirects to `/bb/result/{run_id}` and shows the
   reference price, standard error, confidence interval, latency, and path count.
7. Choose `[1] SCENARIO SHOCK`.
8. Enter one or more shocks and submit `PRICE SHOCK`.
9. Confirm the scenario result shows base price, shocked price, move, shocks,
   and a short summary.
10. Open `/bb/recent-runs` and confirm the price and scenario runs can be
   reopened.
11. Open `/bb/model-status` and confirm Phoenix is marked `REF`.

## Recent Runs

`GET /bb/recent-runs` shows the latest saved price and scenario runs from
SQLite. The page keeps full run IDs in links but displays compact IDs for the
BlackBerry screen.

Price runs show the saved reference price. Scenario runs show the shocked price and
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
positive. Rate bounds are enforced by the versioned product validation.

The shock mapping for Phoenix is:
`spot_pct` changes `S0`, `vol_abs` changes `sigma`, and `rate_bps` changes `r`.

## Pricing status and legacy model cache

`app/services/model_cache.py` remains available for future compatible
surrogates. Phase 1 does not load the committed legacy artifacts: compatibility
requires matching contract-version and feature-order metadata.

`GET /bb/model-status` shows:

- `REF`: validated Monte Carlo reference pricing is active without a compatible
  surrogate.
- `READY`: a compatible surrogate is present and enabled.
- `COLD`: artifacts are present, but the model has not been loaded in this
  backend process yet.
- `CACHED`: the model/scaler bundle is loaded in memory.
- `UNAVAIL`: required artifacts are missing.

The cache is intentionally simple and clears when the backend process restarts.

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

The `/bb` web terminal remains the proven browser-based experience. The optional
legacy client in `clients/blackberry-legacy/` is now a Java ME MIDlet spike with
its own compact UI. It talks to the backend directly over local HTTP instead of
opening or parsing the `/bb` HTML pages.

It does not contain:

- pricing formulas
- model/scaler artifacts
- scenario logic
- product-specific pricing rules
- API keys or secrets

Current MIDlet behavior:

- shows an `ASHBERRY TERMINAL` menu
- stores a backend base URL in RMS settings
- calls `GET /api/bb/ping`
- calls `GET /api/bb/model-status`
- renders compact status text inside the MIDlet
- includes settings and about screens

The committed base URL placeholder is:

```text
http://192.168.1.100:8000
```

On device, update it through the settings screen:

```text
http://<PC_LOCAL_IP>:8000
```

Do not commit a machine-specific IP address.

### Java ME API Endpoints

The native MIDlet uses plain-text endpoints:

- `GET /api/bb/ping`
- `GET /api/bb/model-status`
- `GET /api/bb/products`

Example:

```text
OK
PHOENIX=REF,-
```

Plain text is intentional because Java ME has limited library support and should
not pull in a large JSON dependency for this spike.

### Legacy Tooling Status

The source-level spike was added with Java/Javac available, but without Java ME
build tooling. These commands were not found on the current machine:

- `preverifier`
- `emulator`

Because of that, the `.jar` / `.jad` pair was not built and the app was not
installed from this checkout.

Expected tools to confirm:

- Java ME SDK or compatible Wireless Toolkit with CLDC 1.1 / MIDP 2.0
- preverification tooling
- emulator or BlackBerry installation path
- optional BlackBerry conversion/loading tools if needed later

Recommended install location:

```text
C:\WTK252
```

Set:

```powershell
$env:WTK_HOME="C:\WTK252"
```

Build:

```powershell
cd clients\blackberry-legacy\midlet
.\build.ps1
```

Expected outputs:

```text
dist/AshBerryTerminal.jar
dist/AshBerryTerminal.jad
```

Install options:

- microSD copy: copy the generated `.jar` and `.jad` to the BlackBerry and open
  the `.jad` or `.jar` from File Manager.
- local OTA-style download: serve `dist/` from the PC and open
  `http://<PC_LOCAL_IP>:8010/AshBerryTerminal.jad` from the BlackBerry browser.

For OTA-style install, the server should use:

- `.jad`: `text/vnd.sun.j2me.app-descriptor`
- `.jar`: `application/java-archive`

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
- pricing-method label
- latency in milliseconds
- `run_type` (`price`, `scenario`, or API-only `risk`)
- `parent_run_id` for scenario runs

Storage was introduced before scenario shock because a scenario needs a stable
base pricing request/result to shock and reprice later.

## Current Limitations

- Local-only HTTP; no public exposure.
- No authentication or PIN enforcement yet.
- The current reference model is constant-volatility, flat-rate GBM and does not
  yet consume a live market snapshot.
- Scenario explanations are simple and rule-based.
- The sideloaded Java ME app is currently a source-level MIDlet spike and still
  needs Java ME build/install verification.
- BlackBerry browser rendering may require more simplification after device
  testing.

## Future Phases

- Compact payoff explanations and model-status polish.
- Extend the Phase 7 JSON risk contract to compact BlackBerry risk summaries.
- Optionally verify and sideload the existing native BlackBerry wrapper spike.
