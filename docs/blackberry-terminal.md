# BlackBerry Quant Terminal

The BlackBerry Quant Terminal is a legacy-device-compatible extension for
ML-Pricer. The intended client is a BlackBerry Bold 9780 on local Wi-Fi. The
device should only collect inputs and display compact results; pricing,
scenario analysis, model inference, storage, and validation stay on the
FastAPI backend.

## Current Phase

This phase is a read-only terminal shell. It adds:

- `GET /api/v1/products`
- `GET /api/v1/model-info`
- `GET /bb`
- `GET /bb/model-status`

The BlackBerry pages use server-rendered HTML, minimal CSS, no JavaScript, no
CDN assets, and no template dependency.

## Run Locally

From the repository root:

```powershell
python -m uvicorn app.backend:app --reload --host 127.0.0.1 --port 8000
```

Then open:

- `http://127.0.0.1:8000/api/v1/products`
- `http://127.0.0.1:8000/api/v1/model-info`
- `http://127.0.0.1:8000/bb`
- `http://127.0.0.1:8000/bb/model-status`

## Desktop Browser Test

Use a desktop browser first and resize the window close to 480x360. Confirm
that `/bb` and `/bb/model-status` render as compact text screens and that the
model-status link returns to the terminal home page.

## Local Network Test

For later device testing, run the backend on the LAN interface:

```powershell
python -m uvicorn app.backend:app --host 0.0.0.0 --port 8000
```

Then open `http://<local-ip>:8000/bb` from a device on the same Wi-Fi network.
Do not expose this service publicly in the MVP.

## Future Phases

- Phase 3: pricing form/result plus basic `run_id` creation and storage.
- Phase 4: scenario shock flow using stored base runs.
- Phase 5: recent-runs UI and storage polish.
- Phase 6: compact payoff explanations and model-status polish.
- Phase 7: optional native or sideloaded BlackBerry wrapper.
