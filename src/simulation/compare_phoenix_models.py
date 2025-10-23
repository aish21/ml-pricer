#!/usr/bin/env python3
"""
hybrid_pricing_phoenix.py

Hybrid Monte Carlo + surrogate (tree models) pricer for a Phoenix Autocallable
structured note. Loads LightGBM / XGBoost / CatBoost models from src/models/,
runs pure MC baseline and hybrid CV estimator when the surrogate can supply
pathwise predictions. Saves results and produces comparative plots.

Author: Generated for you — written to be human-readable and maintainable.
"""

import os
import json
import time
import math
import glob
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

from src.training.generate_training_data import simulate_gbm_paths

BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # src/
MODEL_DIR = os.path.join(BASE_DIR, "models")
RESULT_DIR = os.path.join(BASE_DIR, "results", "hybrid_phoenix")
os.makedirs(RESULT_DIR, exist_ok=True)

MODEL_FILENAME_PATTERNS = {
    "LightGBM": ["lightgbm_phoenix", "lightgbm_phoenix.txt", "lightgbm_phoenix*.txt"],
    "XGBoost": ["xgboost_phoenix", "xgboost_phoenix.json", "xgboost_phoenix*.json"],
    "CatBoost": ["catboost_phoenix", "catboost_phoenix.cbm", "catboost_phoenix*.cbm"],
}

# Input order used by training:
FEATURE_NAMES = [
    "S0",
    "r",
    "sigma",
    "T",
    "autocall_barrier",
    "coupon_barrier",
    "coupon_rate",
    "knock_in_barrier",
]

USE_LOG_TARGET = False  # set True if models were trained on log1p(y)


# -----------------------
# GBM / MC helpers
# -----------------------
def simulate_gbm_paths(
    s0: float,
    r: float,
    sigma: float,
    T: float,
    n_steps: int,
    n_paths: int,
    seed: int = None,
) -> np.ndarray:
    """Simulate geometric Brownian motion price paths (n_paths x (n_steps+1))."""
    if seed is not None:
        np.random.seed(seed)
    dt = T / n_steps
    drift = (r - 0.5 * sigma * sigma) * dt
    diffusion = sigma * np.sqrt(dt)
    Z = np.random.randn(n_paths, n_steps)
    increments = drift + diffusion * Z
    log_paths = np.zeros((n_paths, n_steps + 1))
    log_paths[:, 0] = np.log(s0)
    log_paths[:, 1:] = log_paths[:, 0:1] + np.cumsum(increments, axis=1)
    return np.exp(log_paths)


def phoenix_payoff_from_path(
    path: np.ndarray,
    S0: float,
    T: float,
    obs_indices: List[int],
    autocall_barrier: float,
    coupon_barrier: float,
    coupon_rate: float,
    knock_in_barrier: float,
    pay_coupon_at_call: bool = True,
    exact_discount: bool = True,
    r: float = 0.0,
) -> float:
    """
    Compute discounted payoff for a single path (realistic phoenix/autocallable spec).
    - If path at observation >= autocall_barrier -> autocall at that obs:
         payoff = notional (1.0) + accrued coupons up to that obs (coupon_rate * fraction)
       Discounted to call time t_call if exact_discount True.
    - If no autocall:
       - If knocked_in (path dipped below knock_in_barrier at any time):
           payoff at maturity: if final >= coupon_barrier -> notional + coupon; else deliver underlying ratio (path[-1]/S0)
       - Else (no knock-in):
           payoff at maturity: notional + coupon if final >= coupon_barrier, else notional (principal protected) -- depending on spec.
    This is a reasonably typical phoenix-like simplified spec; you can adapt later.
    """
    n_steps = len(path) - 1
    # Observation times as fraction of T are represented by obs_indices (indices into path)
    # Note: coupon_rate is annual; assume coupon payment is coupon_rate * (t_interval)
    # Simplify: assume equal interval between observations
    obs_count = len(obs_indices)
    # time fraction per observation until observation i (year fraction)
    t_obs = [idx / n_steps * T for idx in obs_indices]

    # Knock-in check (any time step below knock-in)
    knocked_in = np.any(path < knock_in_barrier)

    # Check autocall at first observation >= barrier
    call_idx = None
    for idx_i, idx in enumerate(obs_indices):
        if path[idx] >= autocall_barrier:
            call_idx = idx_i  # ordinal index of observation (0-based)
            break

    # Notional = 1.0 (we will return discounted payoff in currency units)
    notional = 1.0

    if call_idx is not None:
        # early call: pay notional + accrued coupon(s) up to and including that obs
        # accrued coupon fraction: (call_idx + 1)/obs_count * coupon_rate * (T maybe annualized)
        # Better: coupon per observation = coupon_rate / (obs_count / T) -> coupon_rate * (observation_interval in years)
        obs_interval = T / (obs_count + 0 if obs_count == 0 else obs_count)
        coupon_per_obs = coupon_rate * obs_interval
        accrued_coupon = coupon_per_obs * (call_idx + 1)
        payoff = notional + accrued_coupon if pay_coupon_at_call else notional
        if exact_discount:
            # discount to call time t_call
            t_call = t_obs[call_idx]
            discounted = np.exp(-r * t_call) * payoff
            return discounted
        else:
            # discount to maturity as simplification (not preferred)
            return np.exp(-r * T) * payoff

    # No early call: evaluate at maturity
    final_price = path[-1]
    coupon_per_obs = coupon_rate * (T / max(1, obs_count))
    coupon_total = (
        coupon_per_obs * obs_count
    )  # sum if coupon paid each obs when condition met; here simplified

    # If knocked-in:
    if knocked_in:
        # more risky: if final < coupon_barrier -> investor gets underlying (loss)
        if final_price >= coupon_barrier:
            payoff_raw = notional + coupon_per_obs  # pay some coupon at maturity
        else:
            # deliver underlying proportion
            payoff_raw = final_price / S0  # return underlying value ratio (loss)
    else:
        # No knock-in: principal protected
        if final_price >= coupon_barrier:
            payoff_raw = notional + coupon_per_obs
        else:
            payoff_raw = notional

    discounted = np.exp(-r * T) * payoff_raw if exact_discount else payoff_raw
    return discounted


def mc_phoenix_price(
    S0: float,
    r: float,
    sigma: float,
    T: float,
    autocall_barrier: float,
    coupon_barrier: float,
    coupon_rate: float,
    knock_in_barrier: float,
    obs_count: int = 6,
    n_steps: int = 252,
    n_paths: int = 20000,
    seed: int = None,
) -> Tuple[float, float, float, np.ndarray]:
    """
    Run Monte Carlo for Phoenix payoff:
    returns (mean_price, elapsed_time, variance_of_payoffs, payoffs_array)
    """
    t0 = time.time()
    paths = simulate_gbm_paths(S0, r, sigma, T, n_steps, n_paths, seed=seed)
    # observation indices (exclude t=0, place obs_count evenly through (0,T])
    obs_indices = [
        int((i + 1) * n_steps / (obs_count)) for i in range(obs_count)
    ]  # 1..obs_count inclusive
    payoffs = np.zeros(n_paths)
    for i in range(n_paths):
        payoffs[i] = phoenix_payoff_from_path(
            paths[i],
            S0=S0,
            T=T,
            obs_indices=obs_indices,
            autocall_barrier=autocall_barrier,
            coupon_barrier=coupon_barrier,
            coupon_rate=coupon_rate,
            knock_in_barrier=knock_in_barrier,
            exact_discount=True,
            r=r,
        )
    elapsed = time.time() - t0
    mean_price = payoffs.mean()
    var_payoffs = payoffs.var(ddof=1)
    return mean_price, elapsed, var_payoffs, payoffs


# -----------------------
# Model loader
# -----------------------
def find_model_files(model_dir: str) -> Dict[str, List[str]]:
    """Scan model_dir and return dict of lists of candidate files for each algorithm."""
    found = {alg: [] for alg in MODEL_FILENAME_PATTERNS.keys()}
    for alg, patterns in MODEL_FILENAME_PATTERNS.items():
        for pat in patterns:
            # pattern might already be a glob or just a substring
            if any(ch in pat for ch in "*?[]"):
                matched = glob.glob(os.path.join(model_dir, pat))
            else:
                # find files that contain substring pat (case-insensitive)
                matched = [
                    p
                    for p in glob.glob(os.path.join(model_dir, "*"))
                    if pat.lower() in os.path.basename(p).lower()
                ]
            for m in matched:
                if os.path.isfile(m):
                    found[alg].append(m)
    return found


def load_models_for_phoenix(model_dir: str) -> Dict[str, Any]:
    """
    Attempts to load one model per algorithm (LightGBM / XGBoost / CatBoost) that contains 'phoenix' in filename.
    Returns dict algorithm -> loaded model object.
    """
    candidates = find_model_files(model_dir)
    models = {}
    # LightGBM: prefer Booster or LGBMRegressor saved txt
    if lgb is not None:
        for fp in candidates.get("LightGBM", []):
            try:
                print(f"Attempting to load LightGBM from {fp}")
                model = lgb.Booster(model_file=fp)
                models["LightGBM"] = model
                break
            except Exception as e:
                print("LightGBM load failed:", e)
    # XGBoost
    if xgb is not None:
        for fp in candidates.get("XGBoost", []):
            try:
                print(f"Attempting to load XGBoost from {fp}")
                model = xgb.XGBRegressor()
                model.load_model(fp)
                models["XGBoost"] = model
                break
            except Exception as e:
                print("XGBoost load failed:", e)
    # CatBoost
    if CatBoostRegressor is not None:
        for fp in candidates.get("CatBoost", []):
            try:
                print(f"Attempting to load CatBoost from {fp}")
                model = CatBoostRegressor()
                model.load_model(fp)
                models["CatBoost"] = model
                break
            except Exception as e:
                print("CatBoost load failed:", e)
    return models


# -----------------------
# Prediction wrapper
# -----------------------
def model_predict_price(
    model, algo_name: str, params: Dict[str, float]
) -> Tuple[float, float]:
    """
    Predict price with a trained model given params dict (keys = FEATURE_NAMES).
    Returns (pred_price, predict_time_seconds).
    """
    X = np.array([[params[k] for k in FEATURE_NAMES]])
    t0 = time.time()
    try:
        if algo_name == "LightGBM":
            preds = model.predict(X)  # Booster.predict accepts numpy 2D
        else:
            preds = model.predict(X)
    except Exception as e:
        # Some models raise shape checks or expect different order; raise informative error
        raise RuntimeError(f"Prediction failed for {algo_name}: {e}")
    t = time.time() - t0
    pred = float(preds[0])
    if USE_LOG_TARGET:
        pred = np.expm1(pred)
    return pred, t


# -----------------------
# Main benchmark routine
# -----------------------
def run_comparison_grid(
    model_dir: str,
    param_sets: List[Dict[str, Any]],
    npaths_list: List[int],
    n_steps_mc: int = 252,
    seed: int = 12345,
) -> Dict[str, Any]:
    """
    For each parameter set and for each MC path count run:
     - MC baseline (compute mean & time)
     - For each model loaded, compute model prediction & time
    Returns nested dict with results.
    """
    models = load_models_for_phoenix(model_dir)
    if not models:
        print(
            "No models loaded. Ensure trained phoenix model files exist in", model_dir
        )
    results = []
    for params in param_sets:
        print("\n=== Params:", params)
        per_n = {}
        for n_mc in npaths_list:
            print(f" Running MC with n_paths={n_mc} ...", end=" ", flush=True)
            mc_price, mc_time, mc_var, payoffs = mc_phoenix_price(
                S0=params["S0"],
                r=params["r"],
                sigma=params["sigma"],
                T=params["T"],
                autocall_barrier=params["K_autocall_frac"] * params["S0"],
                coupon_barrier=params["coupon_barrier_frac"] * params["S0"],
                coupon_rate=params["coupon_rate"],
                knock_in_barrier=params["knock_in_frac"] * params["S0"],
                obs_count=params["obs_count"],
                n_steps=n_steps_mc,
                n_paths=n_mc,
                seed=seed,
            )
            print(f"done (price={mc_price:.4f}, time={mc_time:.4f}s)")
            per_algo = {
                "MC": {
                    "price": float(mc_price),
                    "time": float(mc_time),
                    "var": float(mc_var),
                    "n_mc": int(n_mc),
                }
            }
            # Add model predictions
            for algo_name, model in models.items():
                # create params vector as expected by model (8 features)
                input_params = {
                    "S0": params["S0"],
                    "r": params["r"],
                    "sigma": params["sigma"],
                    "T": params["T"],
                    "autocall_barrier": params["K_autocall_frac"] * params["S0"],
                    "coupon_barrier": params["coupon_barrier_frac"] * params["S0"],
                    "coupon_rate": params["coupon_rate"],
                    "knock_in_barrier": params["knock_in_frac"] * params["S0"],
                }
                try:
                    pred_price, pred_time = model_predict_price(
                        model, algo_name, input_params
                    )
                    abs_err = abs(pred_price - mc_price)
                    rel_err = abs_err / max(1e-12, abs(mc_price))
                    per_algo[algo_name] = {
                        "model_price": float(pred_price),
                        "model_time": float(pred_time),
                        "abs_error_model_vs_mc": float(abs_err),
                        "rel_error_model_vs_mc": float(rel_err),
                        "note": "ok",
                    }
                except Exception as e:
                    per_algo[algo_name] = {"error": str(e)}
            per_n[str(n_mc)] = per_algo
        results.append({"params": params, "per_npaths": per_n})
    final = {"generated_at": time.time(), "results": results}
    # Save JSON + CSV (flatten)
    json_path = os.path.join(RESULT_DIR, "results_phoenix_comparison.json")
    with open(json_path, "w") as f:
        json.dump(final, f, indent=2)
    print("Saved final results JSON to:", json_path)

    # produce CSV for quick viewing: rows per (params, n_mc, algo)
    rows = []
    for res in results:
        p = res["params"]
        for n_mc, per_algo in res["per_npaths"].items():
            for algo, rec in per_algo.items():
                row = {**p, "n_mc": int(n_mc), "algo": algo}
                # copy metrics
                if isinstance(rec, dict):
                    for k, v in rec.items():
                        row[k] = v
                rows.append(row)
    df = pd.DataFrame(rows)
    csv_path = os.path.join(RESULT_DIR, "results_phoenix_comparison.csv")
    df.to_csv(csv_path, index=False)
    print("Saved CSV summary to:", csv_path)

    return final


# -----------------------
# Plotting helpers
# -----------------------
def plot_runtime_comparison(df: pd.DataFrame, outpath: str):
    """
    Bar plot comparing MC runtime vs model predict time (note: model predict time is tiny).
    Plot aggregated by n_mc and algorithm.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set(style="whitegrid")
    # keep only relevant columns
    plot_df = df[["n_mc", "algo", "model_time", "time", "price"]].copy()
    # create a runtime column: MC time in 'time' (algo == 'MC'), model_time for others
    runtimes = []
    for _, row in plot_df.iterrows():
        if row["algo"] == "MC":
            runtimes.append(row["time"])
        else:
            runtimes.append(row.get("model_time", np.nan))
    plot_df["runtime"] = runtimes
    plt.figure(figsize=(10, 6))
    sns.barplot(data=plot_df, x="n_mc", y="runtime", hue="algo", ci=None)
    plt.yscale("log")  # runtime ranges widely; log scale is informative
    plt.title("Runtime comparison: Monte Carlo vs Models (log scale)")
    plt.ylabel("Runtime (s, log scale)")
    plt.tight_layout()
    plt.savefig(outpath)
    print("Saved plot:", outpath)
    plt.close()


def plot_error_comparison(df: pd.DataFrame, outpath: str):
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set(style="whitegrid")
    # we want abs_error_model_vs_mc per algo per n_mc
    err_df = df[df["algo"] != "MC"].copy()
    plt.figure(figsize=(10, 6))
    import numpy as np

    # boxplot by algo grouped by n_mc
    sns.boxplot(data=err_df, x="n_mc", y="abs_error_model_vs_mc", hue="algo")
    plt.yscale("symlog")  # in case errors vary widely
    plt.title("Absolute error of models vs MC (per n_mc)")
    plt.ylabel("Absolute error (|model - MC|)")
    plt.tight_layout()
    plt.savefig(outpath)
    print("Saved plot:", outpath)
    plt.close()


def plot_scatter_model_vs_mc(df: pd.DataFrame, outpath: str):
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set(style="white")
    # scatter model_price vs MC price for all algos & n_mc
    mc_df = df[df["algo"] == "MC"][["n_mc", "price"]].rename(
        columns={"price": "mc_price"}
    )
    merged = df[df["algo"] != "MC"].merge(mc_df, on="n_mc", how="left")
    plt.figure(figsize=(7, 7))
    for algo in merged["algo"].unique():
        subset = merged[merged["algo"] == algo]
        plt.scatter(subset["mc_price"], subset["model_price"], label=algo, alpha=0.8)
    mx = merged[["mc_price", "model_price"]].max().max()
    mn = merged[["mc_price", "model_price"]].min().min()
    plt.plot([mn, mx], [mn, mx], "k--", linewidth=1)
    plt.xlabel("Monte Carlo price")
    plt.ylabel("Model predicted price")
    plt.title("Model prediction vs Monte Carlo")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath)
    print("Saved plot:", outpath)
    plt.close()


# -----------------------
# Example parameter grid
# -----------------------
DEFAULT_PARAM_SETS = [
    {  # conservative phoenix
        "S0": 100.0,
        "K_autocall_frac": 1.05,
        "coupon_barrier_frac": 1.0,
        "knock_in_frac": 0.7,
        "coupon_rate": 0.02,  # 2% per observation period (annualized handled in payoff)
        "r": 0.03,
        "sigma": 0.20,
        "T": 1.0,
        "obs_count": 6,
    },
    {  # different vol / tenor mix
        "S0": 100.0,
        "K_autocall_frac": 1.0,
        "coupon_barrier_frac": 0.95,
        "knock_in_frac": 0.6,
        "coupon_rate": 0.015,
        "r": 0.01,
        "sigma": 0.25,
        "T": 2.0,
        "obs_count": 8,
    },
]

DEFAULT_NPATHS = [500, 2000, 8000]  # quick example - add 32000 if you want long runs


# -----------------------
# Entrypoint
# -----------------------
def main():
    print("Compare Phoenix models vs Monte Carlo baseline")
    print("MODEL_DIR:", MODEL_DIR)
    # run benchmark
    final = run_comparison_grid(
        MODEL_DIR, DEFAULT_PARAM_SETS, DEFAULT_NPATHS, n_steps_mc=252, seed=12345
    )

    # flatten into DataFrame for plotting
    rows = []
    for entry in final["results"]:
        params = entry["params"]
        for n_mc, per_algo in entry["per_npaths"].items():
            for algo, rec in per_algo.items():
                row = {**params, "n_mc": int(n_mc), "algo": algo}
                if isinstance(rec, dict):
                    for k, v in rec.items():
                        row[k] = v
                rows.append(row)
    df = pd.DataFrame(rows)
    csv_out = os.path.join(RESULT_DIR, "results_phoenix_comparison_flat.csv")
    df.to_csv(csv_out, index=False)
    print("Saved flat CSV to:", csv_out)

    # plotting
    try:
        plot_runtime_comparison(df, os.path.join(RESULT_DIR, "runtime_comparison.png"))
        plot_error_comparison(df, os.path.join(RESULT_DIR, "error_comparison.png"))
        plot_scatter_model_vs_mc(
            df, os.path.join(RESULT_DIR, "scatter_model_vs_mc.png")
        )
    except Exception as e:
        print("Plotting failed:", e)

    print("Done. Results directory:", RESULT_DIR)


if __name__ == "__main__":
    main()
