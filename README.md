# ml-pricer

## A prototype project that trains LightGBM surrogate models to approximate Monte Carlo (MC) pricing of different derivative payoffs. The repo contains data generation, model training (with Optuna tuning), evaluation against MC at multiple path counts, and a Streamlit frontend + FastAPI backend for interactive pricing and diagnostics.

## Table of contents

1. Overview — what this repo does and what is being priced
2. Architecture — code layout
3. Payoffs explained — what each payoff represents and what the model predicts
4. Data generation — how simulated paths and targets are generated
5. Model training — LightGBM, Optuna, scalers, metrics, feature importance
6. Evaluation — MC benchmarking, evaluation outputs and summary calculations
7. Files to know and where results are saved (`results.json`, models)
8. Frontend & backend — endpoints, expected requests/responses, Streamlit UI notes
9. How to run locally (dev) — environment, commands, and verifying outputs
10. How to increase training samples / n_paths_per_sample
11. Deployment options
12. Troubleshooting & common errors
13. Performance notes
14. Extras

---

## 1. Overview

This project produces ML models (LightGBM - final model) that estimate the present value of structured-payoff instruments _(hopefully)_ faster than Monte Carlo. It uses Monte Carlo to produce target prices for many sampled parameter combinations, trains a model to learn the mapping `(features → price)`, and exposes a small API + frontend to compare model price vs MC price at arbitrary input parameter sets.

Use cases:

- Rapid pricing where MC is too slow for interactive workflows.
- Investigating model error across parameter ranges.
- Visualizing feature importance and timing speedups.

**The repo prices the present value (discounted payoff) of structured product payoffs** computed from simulated underlying asset price paths. For each instrument the code computes the discounted payoff per Monte Carlo path then averages across paths to return a Monte Carlo estimate; the LightGBM model learns to approximate that averaged PV from the instrument parameters.

---

## 2. Architecture — repo layout & responsibilities

Top-level structure (representative):

```
neural-pricer/
├─ app/
│  └─ frontend.py          # Streamlit app (UI)
|  └─ backend.py           # FastAPI backend exposing /price/ and /training/
├─ src/
│  └─ final/
│      ├─ payoffs.py       # BasePayoff and payoff implementations
│      ├─ inherited_payoffs.py
│      ├─ data_generator.py
│      ├─ model_trainer.py
│      ├─ evaluator.py
│      ├─ pipeline.py
│      └─ run.py           # scripts to run pipelines for each payoff
├─ final/
│  └─ results/
│      └─ <payoff_name>/   # model.joblib, scaler.joblib, results.json
├─ requirements.txt
└─ README.md
```

Summarising all the modules:

- `payoffs.py` — payoff classes implementing `compute_payoff(paths, params, r, T)` and `get_feature_order()`; also contains `param_ranges` used for sampling.
- `inherited_payoffs.py` — extended payoff classes
- `data_generator.py` — simulates log-GBM paths (`simulate_gbm_paths`) and generates training data from sampled parameter tuples.
- `model_trainer.py` — trains LightGBM regressors with Optuna hyperparameter search; stores model, scaler, metrics, feature importance.
- `evaluator.py` — calculates MC baseline for given `n_paths` and compares to model predictions; returns structured result dict.
- `pipeline.py` — orchestrates generation, training, evaluation and saving outputs (model, scaler, `results.json`).
- `run.py` — script to run pipelines for different payoffs.
- `app/frontend.py` — Streamlit UI to input parameters and visualize model vs MC.
- `app/backend.py` — FastAPI endpoint `/price/` used by the frontend to run a model vs MC; `/training/{payoff_type}` returns saved training `results.json`.

---

## 3. Payoffs

All payoffs compute a discounted present value of the payoff per Monte Carlo path and then the mean across paths is taken. The ML model is trained to approximate this mean (the price). Here’s what each payoff does:

### PhoenixPayoff (autocall / phoenix structured product)

- Observations at fixed indices (`obs_count`).
- If spot at an observation ≥ `autocall_barrier_frac * S0` then instrument autocalls: investor gets `1 + coupon_rate` paid at the call time (discounted to t=0).
- If not autocalled and the path ever breaches `knock_in_frac * S0`, then at maturity payoff is `S_T / S_0` (lossful redemption).
- If not autocalled and no knock-in, payoff equals `1 + coupon_rate` at maturity.
- Price = discounted expected payoff.

### StepDownPhoenixPayoff (inherits Phoenix)

- Autocall barrier reduces by a `stepdown_rate` at each observation; coupon can scale by the observation index. Useful for step-down coupons.

### AccumulatorPayoff

- At each observation (based on `obs_frequency`), if spot is inside (lower*barrier_frac * S0, upper*barrier_frac * S0) it accumulates (buys) at discounted price `S_t / (1 + participation_rate)`.
- Final payoff equals the discounted average of accumulated contributions scaled by observation fraction.
- Intuitively: an investor accumulates at a discount while price trades inside the corridor.

### ReverseAccumulatorPayoff

- Opposite accumulation logic: accumulates when price is _outside_ the corridor.

### BarrierOptionPayoff (down-and-out)

- If barrier breached at any time, payoff = 0.
- Otherwise payoff at maturity = max(S_T - K, 0) for a call, or max(K - S_T, 0) for a put; discounted to present.

### DecumulatorPayoff

- Sells shares when price is outside barriers and sums discounted proceeds. (Analogous to an inverted accumulator with `participation_rate` multiplied).

**Important**: In all cases, the code returns PVs normalized as the outcome (so some payoffs show values around 1.0 for normalized redemption payoffs, while accumulators / decumulators may produce larger absolute numbers — see `results.json` examples). The model learns whatever is returned by the `compute_payoff` averaging.

---

## 4. Data generation

- `simulate_gbm_paths(s0, r, sigma, T, n_steps, n_paths, seed)` simulates log-GBM paths with `n_steps` (time steps) and `n_paths` independent paths. Returns array of shape `(n_paths, n_steps + 1)`.
- `DataGenerator.sample_parameters(n_samples, payoff, seed)` samples parameter sets uniformly from `payoff.param_ranges`. `obs_count` is sampled as integer; others are floats.
- `DataGenerator.generate(n_samples, n_paths_per_sample)`:

  - For each sampled parameter tuple, simulate `n_paths_per_sample` GBM paths and compute `payoff.compute_payoff(paths, params, r, T)`.
  - The training target is `price = mean(payoffs)` (a float).
  - The features `X` are arranged following `payoff.get_feature_order()`.

Notes:

- For higher quality training labels, increase `n_paths_per_sample`. This is the main lever to reduce label noise at the cost of CPU/time.
- Seed is optional; if provided, per-sample path seeds are `seed + i`.

---

## 5. Model training

- Uses LightGBM (`LGBMRegressor`) with Optuna for hyperparameter tuning.
- Pipeline:

  - Train/validation split using `random_state`.
  - Optionally transform targets with `log1p` (controlled by `use_log_target`).
  - `StandardScaler` applied to features.
  - Optuna searches these params: `n_estimators`, `learning_rate`, `num_leaves`, `min_child_samples`, `subsample`, `colsample_bytree`.
  - After search, fit the final model on combined training+validation (`X_full_s`).
  - Evaluate final model on a held-out test set for metrics: `rmse`, `mae`, `r2`.
  - Feature importance extracted (attempts to use booster gain if available, otherwise `feature_importances_`).

Return object from `ModelTrainer.train()`:

```py
{
  "model": <LGBMRegressor instance>,
  "scaler": <StandardScaler instance>,
  "metrics": {"rmse": float, "mae": float, "r2": float},
  "optuna_study": {"best_value": float, "best_params": {...}},
  "feature_importance": [{"feature": name, "importance": float}, ...],
  "use_log_target": bool
}
```

Saved files (by `PricingPipeline` when `output_dir` specified):

- `model.joblib` — trained LightGBM model
- `scaler.joblib` — StandardScaler
- `results.json` — a comprehensive JSON containing `config`, `training` (metrics, optuna, feature_importance), `evaluation` (test cases & per-npaths results), and `timing`.

---

## 6. Evaluation & Output format

`Evaluator.evaluate_case(params, model, scaler, n_paths_list, use_log_target, seed)` returns a dict:

```py
{
  "params": { ... },                # the tested parameter tuple
  "per_npaths": {
    "<n_paths>": {
      "MC": {"price": float, "std": float, "time": float, "n_paths": int},
      "Model": {"price": float, "time": float, "abs_error": float, "rel_error": float or None, "speedup": float}
    },
    ...
  }
}
```

`Evaluator.evaluate_multiple_cases()` returns a list of such dicts.

`Evaluator.summarize_results(results_list)` aggregates errors, times, and speedups across test cases and returns:

```py
{
  "n_test_cases": int,
  "errors_by_npaths": { n_paths: {"abs_mean": float, "abs_std": float, "rel_mean": float|None, "rel_std": float|None}, ... },
  "times_by_npaths": { n_paths: {"mc_mean": float, "model_mean": float}, ... },
  "speedups_by_npaths": { n_paths: {"mean": float, "std": float}, ... }
}
```

**Key points**

- `rel_error` is `abs_error / abs(mc_price)` if `mc_price != 0` else `None`.
- `speedup = mc_time / model_time` (if `model_time > 0`).
- Results are saved in `results.json` by `PricingPipeline`.

---

## 7. Files / saved outputs

For each payoff you run via pipeline with `output_dir=Path("final/results/<payoff>")`, you will get:

- `final/results/<payoff>/model.joblib` — trained model (joblib)
- `final/results/<payoff>/scaler.joblib` — scaler
- `final/results/<payoff>/results.json` — full training + evaluation summary (human- and machine-readable)

Use `backend` endpoint `/training/{payoff_type}` to fetch the `results.json` training part for the frontend feature importance.

---

## 8. Frontend & Backend — endpoints, payloads, and behavior

### Backend: FastAPI (`src.final.backend`)

**POST** `/price/` — price an instrument using saved model + MC baseline
Request body (Pydantic `PricingRequest`):

```json
{
  "payoff_type": "phoenix",
  "params": { ... },         // parameter dict for selected payoff
  "n_paths": 2000,           // optional
  "use_log_target": true     // optional
}
```

Successful response structure (wrapped by the backend):

```json
{
  "status": "success",
  "result": { ... }   // exactly the dict returned by Evaluator.evaluate_case(...)
}
```

If there is an error, the API returns:

```json
{
  "status": "error",
  "message": "error message",
  "trace": "traceback (for debugging; remove for production)"
}
```

**GET** `/training/{payoff_type}` — returns training info read from `results.json` (used by frontend to show feature importance if not present in `/price/` response).

### Frontend: Streamlit (`app/frontend.py`)

- UI allows selecting payoff type, entering numeric parameters (keyed numeric inputs — not sliders), and choosing MC `n_paths`.
- On "Run Pricing", frontend calls `/price/` and displays:

  - Dashboard tab: model price vs MC price bar, error metrics, timing (bar charts).
  - Feature Analysis tab: feature importance (fetched either from returned `training` or by calling `/training/{payoff}`).
  - Raw JSON tab: entire response for debugging.

Notes:

- The frontend code includes resilient parsing for many backend response shapes.
- The Streamlit UI expects `result` to contain `per_npaths` (or an equivalent numeric-keys dict).
- Plotly configuration deprecation warning: pass configuration through `config` dict (the app uses `st.plotly_chart(fig, config={"responsive": True})`). If Plotly warns about specific keyword arguments, adjust to the new `config` structure as the warning suggests.

---

## 9. How to run locally (development)

### 1) Create and activate virtualenv

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### 2) Install dependencies

Create `requirements.txt` (representative):

```
fastapi
uvicorn
streamlit
plotly
pandas
numpy
scikit-learn
lightgbm
joblib
requests
optuna
```

Install:

```bash
pip install -r requirements.txt
```

### 3) Run backend (FastAPI)

From project root:

```bash
# ensure PYTHONPATH is set so `src` package is importable
# Windows Powershell:
$env:PYTHONPATH="."
python -m uvicorn src.final.backend:app --reload --host 127.0.0.1 --port 8000
```

Or:

```bash
uvicorn src.final.backend:app --reload --host 0.0.0.0 --port 8000
```

Check `http://127.0.0.1:8000` for health. Open docs at `http://127.0.0.1:8000/docs`.

### 4) Run frontend (Streamlit)

In another terminal (from project root):

```bash
streamlit run app/frontend.py
```

The Streamlit UI will open, default port 8501: `http://localhost:8501`.

**Make sure the frontend `API_URL` variable points to your backend** (`http://localhost:8000` by default). In production, configure via environment variable or Streamlit secrets.

---

## 10. Increasing `n_samples` and `n_paths_per_sample`

You control two separate sampling levels:

- `n_samples` = number of parameter tuples the model trains on (X rows). Increase this to cover more of the parameter space.
- `n_paths_per_sample` = number of Monte Carlo paths used to compute the label (target) for each sampled parameter set. Increase this to reduce label noise.

Tradeoffs:

- Increasing `n_paths_per_sample` reduces label variance but costs CPU/time linearly.
- Increasing `n_samples` improves model generalization but increases memory and training cost (O(n_samples)).
- For LightGBM, training scales well on CPU threads; Optuna tuning multiplies runtime by `n_trials`.
- Recommended practice:

  - Start with `n_samples` = 1k–5k and `n_paths_per_sample` = 500–2000 for prototyping.
  - For production-quality models, increase `n_paths_per_sample` to 2k–8k and `n_samples` to 5k–20k depending on model complexity.

- Use the `data_file` caching option in `pipeline.run_full_pipeline()` so you don’t regenerate samples every run.

Hardware estimate (very approximate; workload depends on payoff complexity):

- 1k samples × 2k paths × 252 steps → CPU only, minutes to tens of minutes.
- 5k samples × 4k paths → can be hours on a single CPU machine. Consider parallelization or a beefy multi-core instance.

If you need cleaner labels (low variance), increase `n_paths_per_sample`. If you need better coverage of parameter space, increase `n_samples`.

---

## 11. Deployment

### Docker

- Build two images (backend + frontend) and deploy them with Azure
- Provide persistent storage or embed pre-trained models in the image (not ideal if model artifacts are large — use cloud storage).

**Important**: Keep `final/results/<payoff>/model.joblib` and `scaler.joblib` present on the backend service. Either bake into the container or load from mounted storage.

---

## 12. Troubleshooting & common errors

### 1. `ModuleNotFoundError: No module named 'src.final.payoffs'`

Cause: `PYTHONPATH` not set or working directory wrong.
Fix:

- Run from repo root and ensure Python path includes `.`:

  - Windows Powershell: `$env:PYTHONPATH="."; python -m uvicorn src.final.backend:app ...`
  - Or `export PYTHONPATH='.'` on macOS/Linux.

- Alternatively, install the package (e.g., `pip install -e .`) with a `setup.py`/`pyproject` that includes `src` as package.

### 2. Streamlit secret error: `StreamlitSecretNotFoundError: No secrets found.`

Cause: frontend tries to read `st.secrets[...]` that doesn't exist.
Fix:

- Add `.streamlit/secrets.toml` or remove `st.secrets.get` fallback logic; ensure `API_URL` fallback exists in code.

### 3. KeyError: `'per_npaths'` or blank graphs, zeros

Cause: backend response shape differs (maybe backend returns wrapped `{"status": "success", "result": {...}}` and frontend expects the nested `result` object).
Fix:

- Frontend should extract `result = res.json()` and then find `per_npaths` using `find_per_npaths(result)` — ensure backend returns `{"status":"success","result": <eval_result>}` consistently OR change backend to return the `evaluate_case` dict at top-level. Current frontend expects the nested structure; confirm.
- If your actual backend returns `{"status": "success", "result": {...}}`, ensure frontend sets `result = result.get("result", result)` before scanning for per_npaths. (In our updated frontend we have resilient parsing; but verify.)

### 4. `X does not have valid feature names` warning from sklearn/lightgbm

Cause: The model was trained with feature names and the input to `model.predict()` is a numpy array without column names. This is a warning; predictions still work.
Fix:

- Either convert feature row to a DataFrame with `columns=self.feature_names` before `scaler.transform()` or ignore the warning (harmless).
- Example:

  ```py
  feat = pd.DataFrame([params_list], columns=feature_order)
  feat_s = scaler.transform(feat.values)
  ```

### 5. Feature importance missing in `/price/` response

Cause: `Evaluator.evaluate_case()` returns MC/model comparison but not training info; `results.json` holds feature importance. Frontend attempts to fetch `/training/{payoff}` if missing.
Fix:

- Ensure `final/results/<payoff>/results.json` exists and `/training/{payoff}` returns it.
- Alternatively, add `feature_importance` into the `/price/` response.

### 6. Plotly config deprecation warning

Message: "The keyword arguments have been deprecated — use config instead."
Fix: In `st.plotly_chart(fig, config={...})` pass configuration via `config`. Avoid legacy keyword args; the current code already uses `config={"responsive": True}`.

---

## 13. Performance & scaling notes

- LightGBM trains on CPU. Use `n_jobs=-1` if you want to use all cores; currently the code uses `n_jobs=1` during Optuna trials to avoid oversubscription. After tuning, set `n_jobs` appropriately for final model training.
- Optuna search multiplies training time by `n_trials`. Consider enabling `n_trials` smaller for prototyping (e.g., 10) and increasing later.
- For huge budgets, consider distributed training or generating labels in parallel across multiple worker machines and storing them to a shared `npz` for training.
- Model inference is extremely fast (milliseconds). MC time scales with `n_paths * n_steps`.

---

## Appendix: Example API usage

### Price an instrument (curl)

```bash
curl -X POST "http://localhost:8000/price/" \
  -H "Content-Type: application/json" \
  -d '{
    "payoff_type": "phoenix",
    "params": {
      "S0":100.0,
      "r":0.03,
      "sigma":0.2,
      "T":1.0,
      "autocall_barrier_frac":1.05,
      "coupon_barrier_frac":1.0,
      "coupon_rate":0.02,
      "knock_in_frac":0.7,
      "obs_count":6
    },
    "n_paths":2000,
    "use_log_target":true
  }'
```

### Expected shape in response

```json
{
  "status": "success",
  "result": {
    "params": {
      /* same params */
    },
    "per_npaths": {
      "2000": {
        "MC": { "price": 0.98, "std": 0.08, "time": 0.03, "n_paths": 2000 },
        "Model": {
          "price": 0.98,
          "time": 0.001,
          "abs_error": 0.001,
          "rel_error": 0.001,
          "speedup": 30
        }
      }
    }
  }
}
```
