# ml-pricer

TRY IT OUT HERE: https://aish-ml-pricer-frontend.up.railway.app

A prototype project that trains fast **LightGBM models** to approximate Monte Carlo (MC) pricing of structured derivative payoffs. The repo handles the full ML pipeline: data generation, model training with hyperparameter tuning via Optuna, evaluation against Monte Carlo baselines, and provides both a **Streamlit frontend** and **FastAPI backend** for interactive pricing comparisons and diagnostics.

**The goal:** Get derivative prices that are _nearly as accurate_ as Monte Carlo but **orders of magnitude faster**.

## Table of Contents

1. [Overview](#overview) — What this repo does and why
2. [Architecture](#architecture) — Code layout and responsibilities
3. [Payoffs Explained](#payoffs-explained) — Each instrument type and what the model learns
4. [Data Generation](#data-generation) — Simulating paths and creating training targets
5. [Model Training](#model-training) — LightGBM, Optuna tuning, feature scaling
6. [Evaluation](#evaluation) — Comparing model vs MC, computing errors and speedups
7. [Files & Outputs](#files--outputs) — Where models and results are saved
8. [API & Frontend](#api--frontend) — Endpoints, request/response shapes, UI notes
9. [Running Locally](#running-locally) — Setup and development workflow
10. [Scaling & Performance](#scaling--performance) — Tuning sample sizes and training budget
11. [Deployment](#deployment) — Containerization and cloud options
12. [Troubleshooting](#troubleshooting) — Common errors and fixes
13. [Performance Notes](#performance-notes) — Benchmarks and scaling expectations
14. [Appendix](#appendix) — Example API calls

---

## Overview

This project is all about **speed without sacrificing accuracy**. Monte Carlo simulation is the gold standard for pricing exotic derivatives, but it can be slow—especially when you need rapid, interactive pricing for risk dashboards or pricing engines.

**What we do:** We use Monte Carlo to generate "ground truth" prices for thousands of random parameter combinations (this is our training data). Then we train a lightweight ML model to learn the mapping from instrument parameters → price. The result? A model that predicts prices in **milliseconds** instead of seconds or minutes, typically with very small relative error.

### Use Cases

- **Risk dashboards** where traders need real-time P&L updates and greeks
- **Interactive pricing** where end users can quickly explore price sensitivity
- **What-if analysis** on structured products without waiting for full MC runs
- **Hedging workflows** that require fast re-pricing

### How It Works (High Level)

1. **Data Generation**: For each payoff type (Phoenix, Accumulator, etc.), we simulate many paths using geometric Brownian motion and compute the payoff for randomly sampled parameter sets.
2. **Feature Engineering**: Parameters (spot, volatility, barriers, etc.) become features; the Monte Carlo price is the label.
3. **Model Training**: LightGBM learns the parameter-to-price mapping, with Optuna doing hyperparameter search to maximize accuracy.
4. **Evaluation**: We benchmark the trained model against fresh Monte Carlo runs at different path counts and compute errors and speedups.
5. **Serving**: A FastAPI backend serves pricing requests; a Streamlit UI lets users interactively compare model vs MC.

**Key insight:** The model learns to approximate the _expected value_ of the payoff. It doesn't simulate individual paths (that's expensive); instead it captures the relationship between market conditions and the final price.

---

## Architecture

The codebase is organized around a clean separation of concerns: data generation, model training, evaluation, and serving. Here's the layout:

```
neural-pricer/
├─ app/
│  ├─ frontend.py          # Streamlit web UI
│  └─ backend.py           # FastAPI server with /price/ and /training/ endpoints
├─ src/final/
│  ├─ payoffs.py           # Payoff classes (Phoenix, Accumulator, Barrier, etc.)
│  ├─ inherited_payoffs.py # Extended payoff variants (StepDownPhoenix, etc.)
│  ├─ data_generator.py    # Monte Carlo path simulation and data sampling
│  ├─ model_trainer.py     # LightGBM training with Optuna tuning
│  ├─ evaluator.py         # MC vs model comparison, error calculation
│  ├─ pipeline.py          # Orchestrates the full workflow
│  └─ run.py               # CLI entry point for running pipelines
├─ final/results/          # Saved models, scalers, and results.json for each payoff
│  ├─ phoenix/
│  ├─ accumulator/
│  ├─ barrier/
│  └─ ...
├─ data/                   # Training data and metadata
├─ notebooks/              # Jupyter notebooks for EDA and experimentation
├─ requirements.txt        # Python dependencies
└─ README.md
```

### Module Responsibilities

| Module                | Purpose                                                                                                                                                                                                                                  |
| --------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **payoffs.py**        | Defines instrument types (Phoenix, Accumulator, etc.). Each class implements `compute_payoff(paths, params, r, T)` which returns the discounted payoff for each MC path, and `get_feature_order()` which specifies feature column order. |
| **data_generator.py** | Simulates GBM paths and generates training data. Main functions: `simulate_gbm_paths()` creates price paths; `sample_parameters()` randomly samples from parameter ranges; `generate()` combines these to create (X, y) pairs.           |
| **model_trainer.py**  | Handles the ML pipeline: train/val/test split, feature scaling, Optuna hyperparameter search over LightGBM, and final model evaluation. Returns model, scaler, metrics, and feature importance.                                          |
| **evaluator.py**      | Compares model predictions to MC ground truth at multiple path counts. Computes absolute/relative error, timing, and speedup ratios. Aggregates results across test cases.                                                               |
| **pipeline.py**       | Orchestrates everything: generates data → trains model → evaluates → saves outputs (model.joblib, scaler.joblib, results.json). Single entry point for the full workflow.                                                                |
| **frontend.py**       | Streamlit UI. Lets users select payoff, enter parameters, and view model vs MC comparison with charts.                                                                                                                                   |
| **backend.py**        | FastAPI server. Serves `/price/` (pricing requests) and `/training/` (feature importance) endpoints. Logs pricing history to CSV.                                                                                                        |

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

## 4. Data generation

**The goal:** Create training data by sampling random market conditions and computing Monte Carlo labels.

### How It Works

- **`simulate_gbm_paths(s0, r, sigma, T, n_steps, n_paths, seed)`**  
  Simulates geometric Brownian motion paths for the underlying asset.
  - `n_steps`: Number of time intervals (typically 252 for daily steps over 1 year)
  - `n_paths`: Number of independent simulated paths
  - Returns shape `(n_paths, n_steps + 1)` — one spot price per time step per path

- **`DataGenerator.sample_parameters(n_samples, payoff, seed)`**  
  Uniformly samples random parameter tuples from `payoff.param_ranges`.
  - Parameters like `S0`, `sigma`, `autocall_barrier_frac` are drawn uniformly
  - `obs_count` is sampled as an integer; others as floats
  - Each sample represents a different market scenario

- **`DataGenerator.generate(n_samples, n_paths_per_sample)`**  
  The main data generation loop:
  1. For each sampled parameter tuple:
     - Simulate `n_paths_per_sample` GBM paths
     - Compute `payoff.compute_payoff(paths, params, r, T)` for each path → array of payoffs
     - Average across paths → **training label** (a single float price)
  2. Arrange feature values following `payoff.get_feature_order()` → **training features**
  3. Return `(X, y)` ready for model training

### Key Notes

- **Label Quality:** The training label is noisy if `n_paths_per_sample` is too small. Increasing it reduces label variance but costs CPU time linearly. Sweet spot is usually 500–2000 paths per sample.
- **Feature Scaling:** Features are normalized later in the pipeline by the `StandardScaler`, so raw feature ranges don't matter much.
- **Seeding:** For reproducibility, provide a seed; per-sample path seeds are `seed + i`.
- **Caching:** Generated data can be saved to `.npz` files and reloaded to avoid regenerating when iterating on model training.

---

## 5. Model training

**The goal:** Train a fast, accurate regression model that learns parameter → price mapping.

### Pipeline Overview

- **Train/Val/Test Split:** Data is split using a fixed `random_state` for reproducibility.
- **Target Transform:** If `use_log_target=True`, apply `log1p()` to targets before training. This can help with skewed payoff distributions. Inverse transform is applied during inference.
- **Feature Scaling:** `StandardScaler` is fit on training features and applied to all splits. Models like LightGBM are generally robust to scaling, but it helps Optuna trials converge faster.

### LightGBM + Optuna Hyperparameter Search

- **Model Type:** [`LGBMRegressor`](https://lightgbm.readthedocs.io/) — gradient boosting on decision trees.
- **Why LightGBM?** Fast to train, handles non-linearities well, and produces good feature importance scores.
- **Hyperparameter Tuning:** Optuna performs Bayesian optimization to search this space:
  - `n_estimators`: Number of boosting rounds (trees)
  - `learning_rate`: Step size per update
  - `num_leaves`: Max leaves per tree (controls tree complexity)
  - `min_child_samples`: Minimum samples in leaf (regularization)
  - `subsample`: Fraction of samples per iteration
  - `colsample_bytree`: Fraction of features per tree

- **Search Strategy:** Optuna runs `n_trials` iterations (default 30), each training a full model on train+val combined and evaluating on test. Best trial parameters are returned.

### Output: Training Info Dictionary

```python
{
  "model": <LGBMRegressor>,           # Trained model
  "scaler": <StandardScaler>,         # Fit on training features
  "metrics": {
    "rmse": float,                    # Test set RMSE
    "mae": float,                     # Test set MAE
    "r2": float                       # Test set R² score
  },
  "optuna_study": {
    "best_value": float,              # Best validation metric
    "best_params": {...}              # Winning hyperparameters
  },
  "feature_importance": [
    {"feature": "S0", "importance": 0.35},
    {"feature": "sigma", "importance": 0.20},
    ...
  ],
  "use_log_target": bool              # Whether log transform was used
}
```

### Saved Artifacts

When `PricingPipeline` runs with `output_dir` specified:

- **`model.joblib`** — Serialized LightGBM model (ready for inference)
- **`scaler.joblib`** — Serialized StandardScaler (apply to features before prediction)
- **`results.json`** — Comprehensive results including training config, metrics, Optuna study, feature importance, evaluation results, and timing

### Key Notes

- **Feature Importance:** Extracted using the booster's gain-based importance if available, otherwise falls back to `feature_importances_` attribute. Represents which features the model relied on most.
- **Reproducibility:** Fixed `random_state` and seed ensure repeatable train/val splits and model behavior.
- **Hyperparameter Sensitivity:** For prototyping, use 10–20 Optuna trials; for production, 50–100 trials. More trials = longer search but potentially better hyperparameters.

---

## 6. Evaluation

**The goal:** Benchmark the trained model against Monte Carlo at different path counts and quantify accuracy gains.

### Evaluation Pipeline

`Evaluator.evaluate_case(params, model, scaler, n_paths_list, use_log_target, seed)` does this:

1. **Model Prediction:** Feed parameters into the model (via scaler) to get predicted price in milliseconds.
2. **MC Baseline:** Simulate `n_paths` (e.g., 500, 2000, 8000) and compute MC price and standard error.
3. **Comparison:** Compute absolute error, relative error (if MC price ≠ 0), and speedup (MC time / model time).

### Output Structure

```python
{
  "params": {
    "S0": 100.0,
    "sigma": 0.2,
    ...
  },
  "per_npaths": {
    "500": {
      "MC": {
        "price": 0.98,
        "std": 0.12,                 # Std error of MC estimate
        "time": 0.01,                # Wall-clock time (seconds)
        "n_paths": 500
      },
      "Model": {
        "price": 0.979,
        "time": 0.0005,
        "abs_error": 0.001,          # |model - MC|
        "rel_error": 0.001,          # abs_error / |MC|
        "speedup": 20.0              # MC time / model time
      }
    },
    "2000": { ... },
    "8000": { ... }
  }
}
```

### Summary Statistics

`Evaluator.summarize_results(results_list)` aggregates across test cases:

```python
{
  "n_test_cases": 10,
  "errors_by_npaths": {
    "500": {
      "abs_mean": 0.015,             # Mean absolute error
      "abs_std": 0.008,              # Std of absolute errors
      "rel_mean": 0.02,              # Mean relative error
      "rel_std": 0.015
    },
    ...
  },
  "times_by_npaths": {
    "500": {
      "mc_mean": 0.012,              # Mean MC time
      "model_mean": 0.0006
    },
    ...
  },
  "speedups_by_npaths": {
    "500": {
      "mean": 20.0,                  # Mean speedup ratio
      "std": 5.0                     # Std of speedups
    },
    ...
  }
}
```

### Key Metrics Explained

- **`rel_error`:** Relative error = `abs_error / |MC_price|` if MC price ≠ 0, else `None`. More meaningful than absolute error when prices vary.
- **`speedup`:** Wall-clock speedup = MC time / model time. E.g., a speedup of 50 means the model is 50× faster.
- **`std` (MC):** Standard error of the MC estimate. Larger at lower path counts; helps contextualize model error relative to MC noise.

### Notes

- **Test Cases:** Typically sample 5–20 different parameter tuples for evaluation. Can be default cases or user-specified.
- **Path Counts:** Typical evaluation uses 500, 2000, 8000 paths to show how model error compares to MC variance at different resolutions.
- **Timing:** Model inference includes feature scaling and prediction. MC timing includes path simulation and payoff computation.

Notes:

- For higher quality training labels, increase `n_paths_per_sample`. This is the main lever to reduce label noise at the cost of CPU/time.
- Seed is optional; if provided, per-sample path seeds are `seed + i`.

---

## Files & Outputs

Once you run a `PricingPipeline` with `output_dir=Path("final/results/<payoff>")`, you'll get these files:

```
final/results/phoenix/
├── model.joblib              # Trained LightGBM regressor (pickle format)
├── scaler.joblib             # Fit StandardScaler (pickle format)
├── results.json              # Full results (see below)
└── training_data.npz         # (Optional) Cached training data to avoid regeneration
```

### results.json Structure

A comprehensive JSON file containing everything about the model, training, and evaluation:

```json
{
  "config": {
    "payoff_type": "phoenix",
    "n_samples": 5000,
    "n_paths_per_sample": 1000,
    "n_steps": 252,
    "use_log_target": true,
    "n_trials": 30
  },
  "training": {
    "metrics": {
      "rmse": 0.025,
      "mae": 0.018,
      "r2": 0.98
    },
    "optuna_study": {
      "best_value": 0.022,
      "best_params": {
        "n_estimators": 150,
        "learning_rate": 0.05,
        "num_leaves": 63
      }
    },
    "feature_importance": [
      { "feature": "sigma", "importance": 0.32 },
      { "feature": "S0", "importance": 0.25 }
    ]
  },
  "evaluation": {
    "summary": {
      "errors_by_npaths": { "2000": { "abs_mean": 0.015, "rel_mean": 0.02 } },
      "speedups_by_npaths": { "2000": { "mean": 25.0, "std": 8.0 } }
    }
  },
  "timing": {
    "data_generation_sec": 45.2,
    "model_training_sec": 123.5,
    "total_sec": 236.5
  }
}
```

### Using Saved Models

Backend loads models at startup and serves them via the `/price/` endpoint. For offline use:

```python
import joblib

model = joblib.load("final/results/phoenix/model.joblib")
scaler = joblib.load("final/results/phoenix/scaler.joblib")

# Predict
feature_row = [100.0, 0.2, 0.03, 1.0, ...]  # In correct feature order
feature_scaled = scaler.transform([feature_row])
price = model.predict(feature_scaled)[0]
```

---

## API & Frontend

### Backend: FastAPI

The backend lives in [app/backend.py](app/backend.py) and exposes two main endpoints:

#### **POST** `/price/`

Price an instrument and compare to Monte Carlo.

**Request:**

```json
{
  "payoff_type": "phoenix",
  "params": {
    "S0": 100.0,
    "r": 0.03,
    "sigma": 0.2,
    "T": 1.0,
    "autocall_barrier_frac": 1.05,
    "coupon_rate": 0.02,
    "knock_in_frac": 0.7,
    "obs_count": 6
  },
  "n_paths": 2000
}
```

**Response (Success):**

```json
{
  "status": "success",
  "result": {
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

**Response (Error):**

```json
{
  "status": "error",
  "message": "Model not found for payoff_type 'unknown'",
  "trace": "Full traceback (omitted in production)"
}
```

#### **GET** `/training/{payoff_type}`

Fetch training metadata (feature importance, metrics) from the saved `results.json`.

**Example Response:**

```json
{
  "status": "success",
  "result": {
    "metrics": {"rmse": 0.025, "mae": 0.018, "r2": 0.98},
    "feature_importance": [{"feature": "sigma", "importance": 0.32}, ...]
  }
}
```

**Internals:**

- Loads models from `final/results/<payoff>/` on startup.
- Logs all pricing requests to `data/pricing_history.csv`.
- Environment variables: `MODEL_RESULTS_DIR`, `MODEL_HISTORY_FILE`.

### Frontend: Streamlit

The UI lives in [app/frontend.py](app/frontend.py). Features:

- **Parameter Input:** Numeric fields for each payoff type.
- **Pricing Dashboard:** Bar charts comparing model vs MC prices, errors, timing.
- **Feature Analysis:** Shows model feature importance.
- **Raw JSON Tab:** Full API response for debugging.

**Key Notes:**

- Resilient response parsing for various backend response formats.
- Plotly visualizations with responsive config.
- API URL configured via environment variable or hardcoded fallback (`http://localhost:8000`).

---

## Running Locally

### 1. Environment Setup

```bash
python -m venv venv

# Windows:
venv\Scripts\activate

# macOS/Linux:
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Backend

From project root in one terminal:

```bash
# Windows (PowerShell):
$env:PYTHONPATH="."
python -m uvicorn src.final.backend:app --reload --host 127.0.0.1 --port 8000

# macOS/Linux:
export PYTHONPATH="."
python -m uvicorn src.final.backend:app --reload --host 127.0.0.1 --port 8000
```

Check at `http://127.0.0.1:8000/docs` for interactive API docs.

### 4. Run Frontend

In another terminal:

```bash
streamlit run app/frontend.py
```

Opens at `http://localhost:8501`.

### 5. Verify

- Backend loads models from `final/results/*/` on startup.
- Frontend connects and displays payoff selector.
- Try a pricing request to see model vs MC comparison.

---

## Scaling & Performance

### Tuning Sample Sizes

Two levers control data quality:

| Parameter            | Effect                   | Trade-off                        |
| -------------------- | ------------------------ | -------------------------------- |
| `n_samples`          | Training dataset size    | More coverage vs longer training |
| `n_paths_per_sample` | Paths per training label | Lower noise vs more computation  |

### Recommended Configs

**Prototyping:**

- `n_samples`: 1,000
- `n_paths_per_sample`: 500
- `n_trials` (Optuna): 10–15
- Runtime: ~5–15 minutes

**Production:**

- `n_samples`: 5,000–10,000
- `n_paths_per_sample`: 2,000–4,000
- `n_trials`: 50–100
- Runtime: 1–4 hours

### Performance Estimation

Rough scaling (depends on payoff complexity):

- **1K samples × 500 paths:** ~15 min total
- **5K samples × 2K paths:** ~90 min total
- **10K samples × 4K paths:** ~5 hours total

**Optimization Tips:**

- Cache generated data (`.npz`) to reuse during hyperparameter tuning.
- Use GPU-accelerated LightGBM for 5–10× speedup (if available).
- Parallelize data generation across machines for very large datasets.

---

## Deployment

### Docker

Build and deploy with containerization:

```bash
docker build -f Dockerfile.backend -t neural-pricer-backend .
docker build -f Dockerfile.frontend -t neural-pricer-frontend .
```

**Key Requirements:**

- `final/results/<payoff>/model.joblib` and `scaler.joblib` must be accessible (baked into container or mounted).
- Backend writes pricing history to `data/pricing_history.csv` (use Docker volumes for persistence).

### Environment Variables

- `MODEL_RESULTS_DIR` (Backend): Path to model directory
- `MODEL_HISTORY_FILE` (Backend): Path to pricing CSV history
- `STREAMLIT_API_URL` (Frontend): Backend URL

### Cloud Deployment

- **Azure App Service:** Deploy backend and frontend as separate web apps.
- **AWS ECS:** Use Docker images with model artifacts from S3.
- **Kubernetes:** Deploy services with persistent volumes for models.

---

## Troubleshooting

### 1. `ModuleNotFoundError: No module named 'src.final.payoffs'`

**Fix:**

```bash
# Set PYTHONPATH
export PYTHONPATH="."
# Or install as editable package
pip install -e .
```

### 2. Streamlit secrets not found

**Fix:**

- Ensure `API_URL` has a fallback: `st.secrets.get("api_url", "http://localhost:8000")`

### 3. KeyError on `'per_npaths'`

**Fix:**

- Check backend returns: `{"status": "success", "result": {...}}`
- Frontend should parse nested responses correctly.

### 4. sklearn/LightGBM feature names warning

**Fix:**

```python
import pandas as pd
feat_df = pd.DataFrame([feature_list], columns=feature_order)
feat_scaled = scaler.transform(feat_df)
```

### 5. Feature importance missing in `/price/` response

**Fix:**

- Ensure `final/results/<payoff>/results.json` exists.
- `/training/{payoff}` endpoint should return it.

### 6. Plotly deprecation warning

**Fix:**

```python
st.plotly_chart(fig, config={"responsive": True})
```

---

## Performance Notes

### Training

- LightGBM trains efficiently on CPU; use `n_jobs=-1` after tuning.
- Each Optuna trial = full model training, so `n_trials=50` → 50× training time.
- GPU LightGBM available: 5–10× speedup.

### Inference

- Model: < 1ms per prediction.
- Monte Carlo: 10–50ms (depends on `n_paths * n_steps`).
- Typical speedup: 20–100×.

### Storage

- `model.joblib`: ~10–100 MB
- `scaler.joblib`: < 1 MB
- `results.json`: ~1–10 MB

---

## Examples

### Curl Request

```bash
curl -X POST "http://localhost:8000/price/" \
  -H "Content-Type: application/json" \
  -d '{
    "payoff_type": "phoenix",
    "params": {"S0": 100, "sigma": 0.2, "r": 0.03, "T": 1.0, "autocall_barrier_frac": 1.05, "coupon_rate": 0.02, "knock_in_frac": 0.7, "obs_count": 6},
    "n_paths": 2000
  }'
```

### Python Client

```python
import requests

response = requests.post("http://localhost:8000/price/", json={
    "payoff_type": "phoenix",
    "params": {"S0": 100, "sigma": 0.2, "r": 0.03, "T": 1.0, "autocall_barrier_frac": 1.05, "coupon_rate": 0.02, "knock_in_frac": 0.7, "obs_count": 6},
    "n_paths": 2000
})

result = response.json()
if result["status"] == "success":
    pricing = result["result"]["per_npaths"]["2000"]
    print(f"Model: {pricing['Model']['price']}, MC: {pricing['MC']['price']}, Speedup: {pricing['Model']['speedup']}x")
```

### Training a New Payoff

```python
from pathlib import Path
from src.final.payoffs import YourCustomPayoff
from src.final.pipeline import PricingPipeline

payoff = YourCustomPayoff()
pipeline = PricingPipeline(payoff)

result = pipeline.run_full_pipeline(
    n_samples=5000,
    n_paths_per_sample=2000,
    n_trials=30,
    output_dir=Path("final/results/your_payoff"),
    seed=42
)

print(f"Model R²: {result['training']['metrics']['r2']}")
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
