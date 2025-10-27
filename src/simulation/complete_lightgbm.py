import os
import time
import json
import math
import shutil
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Tree libs
import lightgbm as lgb
from lightgbm import LGBMRegressor

# Optuna
import optuna

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results" / "phoenix_e2e"
for d in (DATA_DIR, MODEL_DIR, RESULTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Generation settings
N_SAMPLES = 50_000  # number of parameter samples to generate (you requested 50k)
N_PATHS_PER_SAMPLE = 2000  # simulation paths per parameter set (you requested 2000)
N_STEPS = 252  # steps per path
OBS_COUNT_DEFAULT = 6

# Training settings
TEST_SIZE = 0.20
RANDOM_STATE = 42
USE_LOG_TARGET = True  # training on log1p(y)
OPTUNA_TRIALS = 30  # optuna trials (reduce to speed up)
OPTUNA_TIMEOUT = None  # or seconds integer

# Benchmark settings (MC path counts to test)
BENCH_N_PATHS = [500, 2000, 8000]

# Quick debug toggle (use small sizes to test pipeline)
QUICK_DEBUG = False
if QUICK_DEBUG:
    N_SAMPLES = 500
    N_PATHS_PER_SAMPLE = 200
    OPTUNA_TRIALS = 6
    BENCH_N_PATHS = [500, 2000]

# Ranges for parameter sampling (realistic, adjustable)
PARAM_RANGES = {
    "S0": (80.0, 120.0),  # initial spot
    "r": (0.0, 0.05),  # interest rate
    "sigma": (0.05, 0.45),  # vol
    "T": (0.5, 2.5),  # tenor in years
    "autocall_barrier_frac": (0.95, 1.15),  # fraction of S0
    "coupon_barrier_frac": (0.7, 1.05),
    "coupon_rate": (0.005, 0.05),  # coupon per observation (0.5% - 5%)
    "knock_in_frac": (0.5, 0.95),
    "obs_count": (4, 12),  # integer number of obs (inclusive)
}

FEATURE_ORDER = [
    "S0",
    "r",
    "sigma",
    "T",
    "autocall_barrier_frac",
    "coupon_barrier_frac",
    "coupon_rate",
    "knock_in_frac",
]

# output files
TRAIN_NPZ = DATA_DIR / "training_data_phoenix_e2e.npz"
TRAIN_META_JSON = DATA_DIR / "training_data_phoenix_e2e_meta.json"
MODEL_PATH = MODEL_DIR / "lightgbm_phoenix_e2e.txt"
SUMMARY_JSON = RESULTS_DIR / "phoenix_e2e_summary.json"
PARITY_PNG = RESULTS_DIR / "parity_plot.png"
ABSERR_PNG = RESULTS_DIR / "abs_error_vs_npaths.png"
TIME_PNG = RESULTS_DIR / "time_vs_npaths.png"


def log(msg: str):
    t = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"[{t}] {msg}", flush=True)


def simulate_gbm_paths(
    s0: float,
    r: float,
    sigma: float,
    T: float,
    n_steps: int,
    n_paths: int,
    seed: int = None,
) -> np.ndarray:
    """
    Simulate log-GBM paths.
    Returns array shape (n_paths, n_steps+1)
    """
    if seed is not None:
        rng = np.random.RandomState(seed)
        Z = rng.randn(n_paths, n_steps)
    else:
        Z = np.random.randn(n_paths, n_steps)
    dt = T / n_steps
    drift = (r - 0.5 * sigma**2) * dt
    diffusion = sigma * math.sqrt(dt)
    increments = drift + diffusion * Z
    log_paths = np.zeros((n_paths, n_steps + 1), dtype=np.float64)
    log_paths[:, 0] = np.log(s0)
    log_paths[:, 1:] = np.log(s0) + np.cumsum(increments, axis=1)
    paths = np.exp(log_paths)
    return paths


def phoenix_payoff_from_paths(
    paths: np.ndarray,
    autocall_b: float,
    coupon_b: float,
    knockin_b: float,
    coupon_rate: float,
    r: float,
    T: float,
    obs_count: int,
) -> np.ndarray:
    """
    paths: (n_paths, n_steps+1)
    obs_count: number of observation times (excluding t0). We'll use evenly spaced obs indices.
    Payment rules:
      - If path >= autocall barrier at an observation -> autocalled, paid (1 + coupon) discounted to call time.
      - Else at maturity:
           - if knocked-in (path touched knock-in barrier at any time) -> principal loss: payoff = S_T / S_0  (returns fraction of principal)
           - else -> full principal + coupon (1 + coupon_rate)
      - Coupon rates are paid only at call / maturity (simplified Phoenix).
    """
    n_paths, n_points = paths.shape
    n_steps = n_points - 1
    # observation indices: evenly spaced excluding 0
    obs_idx = np.linspace(0, n_steps, obs_count + 1, dtype=int)[1:]
    payoffs = np.zeros(n_paths, dtype=np.float64)

    for i in range(n_paths):
        path = paths[i]
        knocked_in = np.any(path < knockin_b)
        call_idx = None
        for idx in obs_idx:
            if path[idx] >= autocall_b:
                call_idx = idx
                break
        if call_idx is not None:
            t_call = (call_idx / n_steps) * T
            payoff = 1.0 + coupon_rate
            payoffs[i] = payoff * math.exp(-r * t_call)
        else:
            t_maturity = T
            if knocked_in:
                payoff = path[-1] / path[0]  # fraction of principal returned
            else:
                payoff = 1.0 + coupon_rate
            payoffs[i] = payoff * math.exp(-r * t_maturity)
    return payoffs


def sample_parameters(
    n_samples: int, param_ranges: Dict[str, Tuple[float, float]], seed: int = None
) -> List[Dict[str, Any]]:
    rng = np.random.RandomState(seed)
    samples = []
    # We'll sample uniformly for each parameter (simple, effective coverage)
    for _ in range(n_samples):
        s = {}
        for k, (lo, hi) in param_ranges.items():
            if k == "obs_count":
                s[k] = int(rng.randint(lo, hi + 1))
            else:
                s[k] = float(rng.uniform(lo, hi))
        # Ensure barrier fractions make sense: autocall >= coupon_barrier? not strictly required
        samples.append(s)
    return samples


def generate_training_data(
    n_samples: int,
    n_paths_per_sample: int,
    n_steps: int,
    feature_order: List[str],
    out_npz: Path,
    meta_json: Path,
    seed: int = None,
) -> None:
    """
    Generate dataset where each sample is a set of params -> MC price (mean of discounted payoffs).
    Save X (n_samples x n_features) and y (n_samples,)
    """
    log_start = time.time()
    log(
        f"Starting data generation: samples={n_samples}, paths_per_sample={n_paths_per_sample}, n_steps={n_steps}"
    )

    params_list = sample_parameters(n_samples, PARAM_RANGES, seed=seed)
    X = np.zeros((n_samples, len(feature_order)), dtype=np.float64)
    y = np.zeros(n_samples, dtype=np.float64)

    for i, p in enumerate(params_list):
        if (i + 1) % 1000 == 0:
            log(f"  generated {i+1}/{n_samples} samples")

        S0 = p["S0"]
        r = p["r"]
        sigma = p["sigma"]
        T = p["T"]
        autocall_b = p["S0"] * p["autocall_barrier_frac"]
        coupon_b = p["S0"] * p["coupon_barrier_frac"]
        knockin_b = p["S0"] * p["knock_in_frac"]
        coupon_rate = p["coupon_rate"]
        obs_count = p.get("obs_count", OBS_COUNT_DEFAULT)

        # simulate paths (we use different seed per sample to decorrelate)
        paths = simulate_gbm_paths(
            S0,
            r,
            sigma,
            T,
            n_steps,
            n_paths_per_sample,
            seed=(None if seed is None else seed + i),
        )
        payoffs = phoenix_payoff_from_paths(
            paths, autocall_b, coupon_b, knockin_b, coupon_rate, r, T, obs_count
        )
        price = float(np.mean(payoffs))

        # record features according to feature_order
        X[i, :] = [p[f] for f in feature_order]
        y[i] = price

    meta = {
        "generated_at": time.time(),
        "param_samples_requested": n_samples,
        "collected_samples": n_samples,
        "n_paths_per_param": n_paths_per_sample,
        "n_steps": n_steps,
        "feature_order": feature_order,
        "use_log_target": USE_LOG_TARGET,
        "notes": "Phoenix generator: exact discounting at call time; coupons paid at call/maturity; principal loss if knocked-in.",
    }

    np.savez_compressed(out_npz, X=X, y=y, meta=meta)
    with open(meta_json, "w") as f:
        json.dump(meta, f, indent=2)
    log(
        f"Saved training data to {out_npz} (meta -> {meta_json}) in {time.time() - log_start:.1f}s"
    )


def train_lightgbm_optuna(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    model_out: Path,
    n_trials: int = 30,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Train a LightGBM regressor on (X,y) using Optuna to tune hyperparameters.
    Returns dictionary with model path, metrics, scaler, etc.
    """
    log("Starting LightGBM training (with Optuna tuning)...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=random_state
    )

    # log-target transformation
    if USE_LOG_TARGET:
        y_train_t = np.log1p(y_train)
        y_val_t = np.log1p(y_val)
    else:
        y_train_t = y_train
        y_val_t = y_val

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    def objective(trial):
        param = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 3000),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 16, 512),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 200),
            "subsample": trial.suggest_float("subsample", 0.4, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "random_state": random_state,
            "n_jobs": 1,
        }
        model = LGBMRegressor(**param)
        # use early stopping on validation
        model.fit(
            X_train_s,
            y_train_t,
            eval_set=[(X_val_s, y_val_t)],
            eval_metric="rmse",
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(period=50),
            ],
        )
        pred_val = model.predict(X_val_s)
        if USE_LOG_TARGET:
            pred_val_inv = np.expm1(pred_val)
            y_val_inv = np.expm1(y_val_t)
        else:
            pred_val_inv = pred_val
            y_val_inv = y_val_t
        # rmse on original target (not log space)
        rmse_val = math.sqrt(np.mean((y_val_inv - pred_val_inv) ** 2))
        trial.set_user_attr("rmse_val", rmse_val)
        return rmse_val

    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=random_state)
    )
    log(f"Optuna: running {n_trials} trials (this can take a while)...")
    study.optimize(objective, n_trials=n_trials, timeout=OPTUNA_TIMEOUT)
    log("Optuna tuning complete.")
    log(f"Best optuna trial: value={study.best_value:.6f}, params={study.best_params}")

    # train final model on train+val using best params
    best_params = study.best_params
    final_model = LGBMRegressor(**best_params)
    X_full_s = scaler.fit_transform(np.vstack([X_train, X_val]))
    if USE_LOG_TARGET:
        y_full_t = np.log1p(np.concatenate([y_train, y_val]))
    else:
        y_full_t = np.concatenate([y_train, y_val])

    start = time.time()
    final_model.fit(X_full_s, y_full_t, eval_metric="rmse")
    train_time = time.time() - start
    log(f"Trained final LightGBM model in {train_time:.2f}s")

    # predictions on hold-out test split from original full X,y (we'll create one)
    X_hold, X_test, y_hold, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=random_state + 1
    )
    X_test_s = scaler.transform(X_test)
    pred_test = final_model.predict(X_test_s)
    if USE_LOG_TARGET:
        pred_test_inv = np.expm1(pred_test)
    else:
        pred_test_inv = pred_test

    rmse_test = math.sqrt(np.mean((y_test - pred_test_inv) ** 2))
    mae_test = mean_absolute_error(y_test, pred_test_inv)
    r2_test = r2_score(y_test, pred_test_inv)
    metrics = {
        "rmse": rmse_test,
        "mae": mae_test,
        "r2": r2_test,
        "train_time": train_time,
    }

    # save trained model (booster)
    try:
        # for LGBMRegressor wrapper, get underlying booster
        booster = final_model.booster_
        booster.save_model(str(model_out))
    except Exception:
        # fallback: save the sklearn-wrapped model via pickle
        import joblib

        joblib.dump(final_model, str(model_out) + ".pkl")
        log("Saved model as sklearn wrapper (joblib)")

    # feature importance (gain if possible)
    try:
        booster = final_model.booster_
        fi_vals = booster.feature_importance(importance_type="gain")
    except Exception:
        fi_vals = final_model.feature_importances_

    fi = sorted(
        [
            {"feature": feature_names[i], "importance": float(fi_vals[i])}
            for i in range(len(feature_names))
        ],
        key=lambda x: -x["importance"],
    )

    out = {
        "model": final_model,
        "scaler": scaler,
        "metrics": metrics,
        "optuna_study": {
            "best_value": study.best_value,
            "best_params": study.best_params,
        },
        "feature_importance": fi,
        "model_path": str(model_out),
    }
    log(f"Saved model to {model_out}")
    return out


def evaluate_case_vs_mc(
    params: Dict[str, Any],
    model: LGBMRegressor,
    scaler: StandardScaler,
    n_paths_list: List[int],
    seed: int = None,
) -> Dict[str, Any]:
    """
    For a single param set, run MC with different n_paths and compare model prediction.
    Returns a dict with per-n_paths results.
    """
    results = {"params": params, "per_npaths": {}}
    # prepare feature vector (1 x d)
    feat = np.array([[params[k] for k in FEATURE_ORDER]])
    feat_s = scaler.transform(feat)

    # model pred time (warm-up)
    t0 = time.time()
    pred_raw = model.predict(feat_s)[0]
    t_model = time.time() - t0
    if USE_LOG_TARGET:
        model_price = float(np.expm1(pred_raw))
    else:
        model_price = float(pred_raw)

    for n in n_paths_list:
        t_mc0 = time.time()
        paths = simulate_gbm_paths(
            params["S0"],
            params["r"],
            params["sigma"],
            params["T"],
            N_STEPS,
            n,
            seed=seed,
        )
        payoffs = phoenix_payoff_from_paths(
            paths,
            params["S0"] * params["autocall_barrier_frac"],
            params["S0"] * params["coupon_barrier_frac"],
            params["S0"] * params["knock_in_frac"],
            params["coupon_rate"],
            params["r"],
            params["T"],
            params["obs_count"],
        )
        mc_time = time.time() - t_mc0
        mc_price = float(np.mean(payoffs))
        mc_var = float(np.var(payoffs))

        # record
        results["per_npaths"][str(n)] = {
            "MC": {"price": mc_price, "var": mc_var, "time": mc_time, "n_paths": n},
            "Model": {
                "algo": "LightGBM",
                "model_price": model_price,
                "model_time_estimate": t_model,
                "abs_error": abs(model_price - mc_price),
                "rel_error": (
                    abs(model_price - mc_price) / abs(mc_price)
                    if mc_price != 0
                    else None
                ),
            },
        }
        log(
            f"   n_paths={n}: MC_price={mc_price:.6f} (t={mc_time:.3f}s), Model_price={model_price:.6f} (t={t_model:.4f}s), abs_err={abs(model_price-mc_price):.4f}"
        )

    return results


def plot_parity(y_true: np.ndarray, y_pred: np.ndarray, out_png: Path):
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.25, s=8)
    mn = min(y_true.min(), y_pred.min())
    mx = max(y_true.max(), y_pred.max())
    plt.plot([mn, mx], [mn, mx], "k--", lw=1)
    plt.xlabel("MC Price (test)")
    plt.ylabel("Model Predicted Price")
    plt.title("Parity: Model vs Monte Carlo")
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()
    log(f"Saved parity plot to {out_png}")


def plot_abs_error_vs_npaths(results_all: List[Dict[str, Any]], out_png: Path):
    # results_all: list per-case dicts
    rows = []
    for case in results_all:
        for n_str, vals in case["per_npaths"].items():
            rows.append(
                {
                    "test_case": json.dumps(case["params"]),
                    "n_paths": int(n_str),
                    "abs_error": float(vals["Model"]["abs_error"]),
                }
            )
    df = pd.DataFrame(rows)
    plt.figure(figsize=(8, 5))
    for k, g in df.groupby("test_case"):
        plt.plot(
            sorted(g["n_paths"].unique()),
            [
                g[g["n_paths"] == n]["abs_error"].values[0]
                for n in sorted(g["n_paths"].unique())
            ],
            marker="o",
            label=k[:80],
        )
    plt.xscale("log")
    plt.xlabel("MC Paths")
    plt.ylabel("Absolute Error (Model vs MC)")
    plt.title("Absolute Error vs MC Paths")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()
    log(f"Saved abs-error plot to {out_png}")


def plot_time_vs_npaths(results_all: List[Dict[str, Any]], out_png: Path):
    rows = []
    for case in results_all:
        for n_str, vals in case["per_npaths"].items():
            rows.append(
                {
                    "test_case": json.dumps(case["params"]),
                    "n_paths": int(n_str),
                    "mc_time": float(vals["MC"]["time"]),
                    "model_time": float(vals["Model"]["model_time_estimate"]),
                }
            )
    df = pd.DataFrame(rows)
    plt.figure(figsize=(8, 5))
    # MC time
    for k, g in df.groupby("test_case"):
        plt.plot(
            sorted(g["n_paths"].unique()),
            [
                g[g["n_paths"] == n]["mc_time"].values[0]
                for n in sorted(g["n_paths"].unique())
            ],
            marker="o",
            label=f"MC {k[:60]}",
        )
    # single model_time as horizontal lines:
    for k, g in df.groupby("test_case"):
        mt = g["model_time"].iloc[0]
        xs = sorted(g["n_paths"].unique())
        plt.hlines(mt, xmin=xs[0], xmax=xs[-1], linestyles="--")
    plt.xscale("log")
    plt.xlabel("MC Paths")
    plt.ylabel("Time (s)")
    plt.title("Compute time vs MC Paths (Model horizontal)")
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()
    log(f"Saved time plot to {out_png}")


def main():
    start_all = time.time()
    # --- 1) Generate training data (unless already exists) ---
    if TRAIN_NPZ.exists():
        log(f"Training data exists at {TRAIN_NPZ}, loading...")
        arr = np.load(TRAIN_NPZ, allow_pickle=True)
        X = arr["X"]
        y = arr["y"]
        meta = arr.get("meta", None)
    else:
        log("No training file found, generating fresh dataset (this will take time)...")
        generate_training_data(
            N_SAMPLES,
            N_PATHS_PER_SAMPLE,
            N_STEPS,
            FEATURE_ORDER,
            TRAIN_NPZ,
            TRAIN_META_JSON,
            seed=RANDOM_STATE,
        )
        arr = np.load(TRAIN_NPZ, allow_pickle=True)
        X = arr["X"]
        y = arr["y"]
        meta = arr.get("meta", None)

    log(f"Loaded X shape: {X.shape} y shape: {y.shape}")

    # --- 2) Train LightGBM with Optuna ---
    train_info = train_lightgbm_optuna(
        X,
        y,
        FEATURE_ORDER,
        MODEL_PATH,
        n_trials=OPTUNA_TRIALS,
        random_state=RANDOM_STATE,
    )

    # --- 3) Evaluate vs MC for a set of test cases ---
    TEST_CASES = [
        {
            "S0": 100.0,
            "r": 0.03,
            "sigma": 0.2,
            "T": 1.0,
            "autocall_barrier_frac": 1.05,
            "coupon_barrier_frac": 1.0,
            "coupon_rate": 0.02,
            "knock_in_frac": 0.7,
            "obs_count": 6,
        },
        {
            "S0": 100.0,
            "r": 0.01,
            "sigma": 0.25,
            "T": 2.0,
            "autocall_barrier_frac": 1.0,
            "coupon_barrier_frac": 0.95,
            "coupon_rate": 0.015,
            "knock_in_frac": 0.6,
            "obs_count": 8,
        },
        {
            "S0": 110.0,
            "r": 0.015,
            "sigma": 0.3,
            "T": 2.0,
            "autocall_barrier_frac": 1.0,
            "coupon_barrier_frac": 0.9,
            "coupon_rate": 0.025,
            "knock_in_frac": 0.6,
            "obs_count": 8,
        },
    ]

    results_all = []
    model = train_info["model"]
    scaler = train_info["scaler"]
    for case in TEST_CASES:
        log(f"Evaluating test case: {case}")
        res = evaluate_case_vs_mc(case, model, scaler, BENCH_N_PATHS, seed=RANDOM_STATE)
        results_all.append(res)

    # --- Save results JSON + model summary ---
    summary = {
        "generated_at": time.time(),
        "config": {
            "n_samples": N_SAMPLES,
            "n_paths_per_sample": N_PATHS_PER_SAMPLE,
            "n_steps": N_STEPS,
            "optuna_trials": OPTUNA_TRIALS,
            "bench_n_paths": BENCH_N_PATHS,
        },
        "train_info": {
            "metrics": train_info["metrics"],
            "optuna_best": train_info["optuna_study"],
            "feature_importance": train_info["feature_importance"],
            "model_path": train_info["model_path"],
        },
        "results": results_all,
    }
    with open(SUMMARY_JSON, "w") as f:
        json.dump(
            summary,
            f,
            indent=2,
            default=lambda x: (
                float(x) if isinstance(x, (np.floating, np.integer)) else x
            ),
        )
    log(f"Saved full summary to {SUMMARY_JSON}")

    # --- parity plot on test holdout: use last test split predictions from train_lightgbm step ---
    # For a parity we need test set predictions: use a small holdout from original X
    X_train_all, X_test_all, y_train_all, y_test_all = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE + 2
    )
    X_test_s = scaler.transform(X_test_all)
    preds_test_raw = model.predict(X_test_s)
    preds_test = np.expm1(preds_test_raw) if USE_LOG_TARGET else preds_test_raw
    plot_parity(y_test_all, preds_test, PARITY_PNG)

    # plots for benchmark
    plot_abs_error_vs_npaths(results_all, ABSERR_PNG)
    plot_time_vs_npaths(results_all, TIME_PNG)

    log(f"All done. Total elapsed: {time.time() - start_all:.1f}s")


if __name__ == "__main__":
    main()
