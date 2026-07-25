# ML Pricer

Reference pricing and ML research for path-dependent structured products.

[![CI](https://github.com/aish21/ml-pricer/actions/workflows/ci.yml/badge.svg)](https://github.com/aish21/ml-pricer/actions/workflows/ci.yml)
[![Production smoke](https://github.com/aish21/ml-pricer/actions/workflows/production-smoke.yml/badge.svg)](https://github.com/aish21/ml-pricer/actions/workflows/production-smoke.yml)

ML Pricer is a side project for building, pricing, and explaining structured
notes. Monte Carlo remains the reference engine. Machine-learning models run
alongside it in shadow mode so their accuracy and speed can be measured without
silently replacing the reference price.

- [Open the web app](https://aish-ml-pricer-frontend.up.railway.app)
- [Check the API](https://aish-ml-pricer-backend.up.railway.app/health/ready)

This is research software, not a trading system, financial advice, or a source
of executable market quotes.

## Products

| Product | Contract | What is covered |
|---|---|---|
| Phoenix | `phoenix-single-v1` | New-issue autocall, conditional coupons, and knock-in redemption |
| Phoenix | `phoenix-single-v2` | Fixed reference level, remaining schedule, and prior knock-in state |
| Phoenix | `phoenix-single-v3` | Coupon memory, unpaid coupons, and step-down autocall barriers |
| Barrier reverse convertible | `barrier-reverse-convertible-v1` | Fixed coupons and knock-in-linked maturity redemption |

All production-facing prices come from deterministic Monte Carlo with a
reported standard error and 95% confidence interval.

## What the project includes

- A FastAPI backend for pricing, scenarios, risk, diagnostics, run history, and
  model evidence.
- A Streamlit workspace with a direct Quant mode and a beginner-friendly Guided
  mode.
- Credential-free research data built from Treasury yields and yfinance equity,
  ETF, distribution, and option data.
- Immutable market-calibration snapshots with quality checks and provenance.
- Spot, rate, dividend, volatility, and term-structure scenarios.
- Delta, Gamma, Vega, Rho, and dividend Rho with paired Monte Carlo diagnostics.
- Checksum-verified Phoenix v1, Phoenix v3, and reverse-convertible surrogate
  packages with independent shadow controls.
- Sealed model audits, live error and latency telemetry, drift checks, and
  non-promoting review gates.
- A compact server-rendered BlackBerry terminal at `/bb`.

The market-data integration is suitable for hobby and research use. yfinance
is an unofficial interface and the resulting data should not be treated as a
licensed production feed.

## Architecture

```text
Streamlit / BlackBerry terminal / API client
                    |
                 FastAPI
                    |
       product contract + market snapshot
                    |
      deterministic Monte Carlo reference
          |                     |
    diagnostics/risk       ML shadow estimate
          |                     |
        run store         audit + telemetry
```

The ML estimate is observational. It does not change the served reference
price.

## Run locally with Docker

```powershell
docker compose up --build
```

Then open:

- Frontend: <http://localhost:8501>
- Backend readiness: <http://localhost:8000/health/ready>
- Operations metrics: <http://localhost:8000/health/metrics>
- BlackBerry terminal: <http://localhost:8000/bb>

Stop the stack with:

```powershell
docker compose down
```

Runtime state is stored under `data/`. The two expanded shadow packages are
small, safe tree exports pinned under `final/shadow_artifacts/`; unsafe training
pickles remain outside Git and the API image.

## Local development

Python 3.11 is the tested version.

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[api,frontend,training,test]"
```

Run the checks:

```powershell
python -m black --check .
python -m ruff check .
python -m pytest -q
```

Start the services without Docker:

```powershell
# Terminal 1
python -m uvicorn app.backend:app --reload --host 127.0.0.1 --port 8000

# Terminal 2
$env:API_URL="http://127.0.0.1:8000"
streamlit run app/frontend.py
```

## ML research

The deployed Phoenix surrogate was selected on development data, evaluated once
on a sealed audit set, exported as a pure-NumPy artifact, and approved only for
shadow comparison.

The Phoenix v3 and barrier reverse-convertible experiments use separate
development, validation, and sealed audit datasets. Their v2 candidates passed
the unchanged accuracy, uncertainty, and latency gates after training was
focused around payoff boundaries. The winners are now packaged for controlled
shadow collection, but both kill switches and sample rates default to off.
`runtime_approved` remains false, and neither model can alter a live price.

Run the expanded-product experiment with:

```powershell
python -m src.final.expanded_surrogate_experiment
```

The latest machine-readable decisions live under
`final/research_candidates/`.

The next training study fixes the Monte Carlo teacher before changing the
learner: it uses the production time grid, piecewise markets, independently
scrambled Sobol labels, and measured label uncertainty. It retrains the same
LightGBM baseline with and without uncertainty weights, without touching
runtime artifacts:

```powershell
python -m src.final.expanded_surrogate_phase1
```

The design and the ML and pricing ideas behind it are in
[Expanded surrogate training Phase 1](docs/expanded-surrogate-phase1-v1.md).

Package the pinned winners after reproducing the experiment with:

```powershell
python -m src.final.expanded_shadow_artifact
```

Preview or run the daily out-of-time evidence campaign with:

```powershell
python -m src.final.expanded_shadow_campaign --plan-only
python -m src.final.expanded_shadow_campaign
```

It freezes research-market snapshots for a fixed 12-underlier universe and
records 4,096-path reference comparisons separately from interactive traffic.
Rerunning the same campaign does not double-count evidence.

The rollout flags, evidence gates, API endpoints, and replay workflow are in
[Expanded shadow rollout](docs/expanded-shadow-rollout-v1.md).

## Operations and deployment

The API exposes:

- `/health/live` — process liveness
- `/health/ready` — pricing and dependency readiness
- `/health/metrics` — bounded request, error, latency, market-snapshot, and
  shadow-model status

The connected Railway environment deploys from `master`. GitHub Actions runs
tests and clean container builds before merge, then checks the deployed frontend
and backend after a successful deployment and every six hours.

Useful runtime settings are documented in:

- [Research market data](docs/equity-research-market-v1.md)
- [Shadow artifact](docs/phoenix-price-first-shadow-artifact-v1.md)
- [Promotion readiness](docs/phoenix-shadow-promotion-readiness-v1.md)
- [Expanded shadow rollout](docs/expanded-shadow-rollout-v1.md)

## Repository map

```text
app/
  backend.py                  FastAPI application and health endpoints
  api/v1.py                   Versioned product and analytics API
  services/                   Pricing, market data, storage, and monitoring
  ui/                         Streamlit configuration, charts, and results

src/final/
  payoffs.py                  Core payoff rules
  phoenix_contract.py         Versioned active Phoenix contracts
  barrier_reverse_convertible.py
  reference_pricer.py         Monte Carlo reference engines
  expanded_surrogate_experiment.py
  expanded_shadow_artifact.py
  expanded_shadow_campaign.py
  surrogate_*                 Training, audit, replay, and artifact tooling

docs/                         Contract and research specifications
final/research_candidates/    Non-runtime experiment decisions
final/shadow_artifacts/       Pinned safe tree exports for shadow inference
unittests/                    Unit and integration tests
```

Key specifications:

- [Phoenix v1](docs/phoenix-single-v1.md)
- [Phoenix v2](docs/phoenix-single-v2.md)
- [Phoenix v3](docs/phoenix-single-v3.md)
- [Barrier reverse convertible](docs/barrier-reverse-convertible-v1.md)
- [Expanded surrogate experiment](docs/expanded-surrogate-experiment-v2.md)
- [Expanded surrogate training Phase 1](docs/expanded-surrogate-phase1-v1.md)
- [Expanded shadow rollout](docs/expanded-shadow-rollout-v1.md)
- [Quant and Guided workspace](docs/quant-workspace-v1.md)

## Current boundaries

- No authentication or authorization layer is included.
- Market data is research-grade and can be delayed, incomplete, or unavailable.
- Barrier monitoring is discrete on the simulation grid.
- The reference model uses GBM-style dynamics rather than a calibrated local or
  stochastic-volatility model.
- ML candidates never promote themselves; runtime approval remains a separate
  human decision.
- The BlackBerry client is a trusted-local-network demo, not a secure public
  interface.
