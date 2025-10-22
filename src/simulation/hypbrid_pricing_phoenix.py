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
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# model libs
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

# local generator (uses your generate_training_data.simulate_gbm_paths)
# change import path if needed
from src.training.generate_training_data import simulate_gbm_paths

# -------------------------
# Config
# -------------------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # src/
MODEL_DIR = os.path.join(BASE_DIR, "models")
OUT_DIR = os.path.join(BASE_DIR, "results", "hybrid_phoenix")
os.makedirs(OUT_DIR, exist_ok=True)

# Monte Carlo default
DEFAULT_N_STEPS = 252  # daily-ish discretisation for 1-year (typical)
S0 = 100.0

# Choose n_paths for benchmarking (small -> large)
NPATHS_LIST = [500, 2000, 8000, 32000]

# If your trained models used log1p(y) transform, set this True (match training)
USE_LOG_TARGET = True

# Notional (per note)
NOTIONAL = 100.0

# -------------------------
# Phoenix payoff specification
# (common, realistic-ish defaults)
# -------------------------
# - Autocall (knock-out) occurs at observation if S_obs >= K_autocall_frac*S0
# - Coupon paid at each observation if S_obs >= coupon_barrier_frac*S0 (if coupon barrier set)
# - If autocall: return NOTIONAL + coupon for that obs (we also include previously paid coupons)
# - If not called by maturity:
#     - If knock-in never occurred -> return NOTIONAL + sum(coupons_paid)
#     - If knock-in occurred -> principal at-risk -> return NOTIONAL * min(1, S_T / S0) + sum(coupons_paid)
#
# Everything is discounted to time 0 using exp(-r * t_obs).
# Coupon_rate is percentage per observation (e.g. 2.0 for 2% per obs).
# -------------------------


def phoenix_payoff_from_path(
    path: np.ndarray,
    S0: float,
    K_autocall_frac: float,
    coupon_barrier_frac: float,
    knock_in_frac: float,
    coupon_rate: float,
    r: float,
    T: float,
    obs_count: int,
) -> float:
    """
    Compute discounted payoff for a single simulated path.
    - path: array of prices length n_steps+1 (including S0 at index 0)
    - obs_count: number of equally-spaced observation dates (including final maturity observation)
    """
    n_steps_total = len(path) - 1
    # observation indices: evenly spaced points among time steps (exclude t=0)
    obs_indices = np.linspace(1, n_steps_total, obs_count, dtype=int)

    coupons_paid_discounted = 0.0
    knocked_in = False
    called = False
    call_time = None
    call_index = None

    for j, idx in enumerate(obs_indices, start=1):
        S_obs = path[idx]
        t_obs = idx / n_steps_total * T

        # knock-in check (anywhere during path)
        if np.min(path[: idx + 1]) < knock_in_frac * S0:
            knocked_in = True

        # coupon accrual check for this observation
        if S_obs >= coupon_barrier_frac * S0:
            coupon_amt = (coupon_rate / 100.0) * NOTIONAL
            coupons_paid_discounted += coupon_amt * math.exp(-r * t_obs)

        # autocall check
        if (not called) and (S_obs >= K_autocall_frac * S0):
            # autocall happens at this observation
            called = True
            call_time = t_obs
            call_index = idx
            break

    if called:
        # Upon call, principal returned + coupons paid so far (coupons already discounted)
        payoff = NOTIONAL * math.exp(-r * call_time) + coupons_paid_discounted
        return payoff

    # Not called -> outcome at maturity
    S_T = path[-1]
    t_T = T

    if not knocked_in:
        # principal protected
        payoff = NOTIONAL * math.exp(-r * t_T) + coupons_paid_discounted
    else:
        # principal at risk: typically linear participation on downside
        # if S_T >= S0 -> full principal returned
        # if S_T < S0 -> proportionate return
        principal_return = NOTIONAL * min(1.0, S_T / S0)
        payoff = principal_return * math.exp(-r * t_T) + coupons_paid_discounted

    return payoff


# -------------------------
# Model loading utilities
# -------------------------
def load_models_for_product(product_name: str) -> Dict[str, Any]:
    """
    Loads LightGBM, XGBoost, CatBoost models if available for the product_name
    (e.g. 'phoenix' or option types naming).
    Expected filenames:
      - lightgbm_{product_name}.txt
      - xgboost_{product_name}.json
      - catboost_{product_name}.cbm
    """
    models = {}

    # LightGBM
    lgb_path = os.path.join(MODEL_DIR, f"lightgbm_{product_name}.txt")
    if os.path.exists(lgb_path):
        try:
            print(f"Loading LightGBM model from {lgb_path}")
            models["LightGBM"] = lgb.Booster(model_file=lgb_path)
        except Exception as e:
            print(f"Failed to load LightGBM model: {e}")

    # XGBoost
    xgb_path = os.path.join(MODEL_DIR, f"xgboost_{product_name}.json")
    if os.path.exists(xgb_path):
        try:
            print(f"Loading XGBoost model from {xgb_path}")
            model = xgb.XGBRegressor()
            model.load_model(xgb_path)
            models["XGBoost"] = model
        except Exception as e:
            print(f"Failed to load XGBoost model: {e}")

    # CatBoost
    cat_path = os.path.join(MODEL_DIR, f"catboost_{product_name}.cbm")
    if os.path.exists(cat_path):
        try:
            print(f"Loading CatBoost model from {cat_path}")
            model = CatBoostRegressor()
            model.load_model(cat_path)
            models["CatBoost"] = model
        except Exception as e:
            print(f"Failed to load CatBoost model: {e}")

    return models


def model_input_dim(model_obj: Any) -> Optional[int]:
    """
    Attempt to infer model expected input dimensionality.
    Returns None on failure.
    """
    try:
        # LightGBM Booster has attribute num_feature()
        if isinstance(model_obj, lgb.Booster):
            return int(model_obj.num_feature())
        # XGBoost XGBRegressor: use n_features_in_ if present (sklearn-like)
        if isinstance(model_obj, xgb.XGBRegressor):
            return int(getattr(model_obj, "n_features_in_", None) or 0)
        # CatBoost
        if isinstance(model_obj, CatBoostRegressor):
            # CatBoost stores feature count in feature_names_ or model_info
            return int(
                len(
                    getattr(model_obj, "feature_names_", [])
                    or getattr(model_obj, "feature_importances_", [])
                )
            )
    except Exception:
        pass
    return None


# -------------------------
# Surrogate prediction helpers
# -------------------------
def predict_surrogate_for_params(
    model_obj: Any,
    params: Dict[str, Any],
    feature_order: List[str],
    use_log_target: bool = USE_LOG_TARGET,
) -> float:
    """
    Predict surrogate price from product-level parameters (parameter-only surrogate).
    feature_order: list of feature names in the correct order expected by model (length should match input dim)
    params: dictionary containing feature values
    """
    x = np.array([[params[k] for k in feature_order]])
    # LightGBM Booster expects raw numpy
    try:
        if isinstance(model_obj, lgb.Booster):
            pred = model_obj.predict(x)
        elif isinstance(model_obj, xgb.XGBRegressor):
            pred = model_obj.predict(x)
        elif isinstance(model_obj, CatBoostRegressor):
            pred = model_obj.predict(x)
        else:
            pred = model_obj.predict(x)
    except Exception as e:
        raise RuntimeError(f"Surrogate prediction failed: {e}")

    val = float(pred[0])
    if use_log_target:
        val = float(np.expm1(val))
    return val


def predict_surrogate_pathwise(
    model_obj: Any,
    path_features: np.ndarray,
    use_log_target: bool = USE_LOG_TARGET,
) -> np.ndarray:
    """
    Predict surrogate for an array of per-path features (n_paths x n_features).
    Returns a numpy array of predictions (one per row).
    """
    try:
        if isinstance(model_obj, lgb.Booster):
            preds = model_obj.predict(path_features)
        elif isinstance(model_obj, xgb.XGBRegressor):
            preds = model_obj.predict(path_features)
        elif isinstance(model_obj, CatBoostRegressor):
            preds = model_obj.predict(path_features)
        else:
            preds = model_obj.predict(path_features)
    except Exception as e:
        raise RuntimeError(f"Surrogate pathwise prediction failed: {e}")

    preds = np.array(preds, dtype=float).flatten()
    if USE_LOG_TARGET:
        preds = np.expm1(preds)
    return preds


# -------------------------
# Monte Carlo Phoenix (returns array of discounted payoffs)
# -------------------------
def mc_phoenix_payoffs(
    S0: float,
    K_autocall_frac: float,
    coupon_barrier_frac: float,
    knock_in_frac: float,
    coupon_rate: float,
    r: float,
    sigma: float,
    T: float,
    obs_count: int,
    n_steps: int,
    n_paths: int,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Simulate n_paths GBM paths and compute discounted payoffs for each
    using phoenix_payoff_from_path.
    """
    if seed is not None:
        np.random.seed(seed)
    paths = simulate_gbm_paths(S0, r, sigma, T, n_steps, n_paths)
    payoffs = np.zeros(n_paths, dtype=float)
    for i in range(n_paths):
        payoffs[i] = phoenix_payoff_from_path(
            path=paths[i],
            S0=S0,
            K_autocall_frac=K_autocall_frac,
            coupon_barrier_frac=coupon_barrier_frac,
            knock_in_frac=knock_in_frac,
            coupon_rate=coupon_rate,
            r=r,
            T=T,
            obs_count=obs_count,
        )
    return payoffs


# -------------------------
# Hybrid estimator (control-variate)
# -------------------------
def hybrid_control_variate_estimator_pathwise(
    payoffs: np.ndarray,
    g_preds: np.ndarray,
) -> Tuple[float, float]:
    """
    Given per-path payoffs and per-path surrogate predictions g_i,
    compute the control-variate corrected mean and variance of the hybrid estimator:
      hybrid_payoffs_i = payoffs_i - (g_i - mean(g))
    Return (hybrid_mean, hybrid_variance).
    """
    g_mean = float(np.mean(g_preds))
    hybrid_payoffs = payoffs - (g_preds - g_mean)
    return float(np.mean(hybrid_payoffs)), float(np.var(hybrid_payoffs, ddof=0))


# -------------------------
# Running one comparison (MC vs model vs hybrid)
# -------------------------
def compare_single_configuration(
    params: Dict[str, Any],
    models: Dict[str, Any],
    feature_order_params: List[str],
    n_steps: int,
    n_paths: int,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Compare MC baseline to stored models and hybrid estimator (if surrogate is pathwise-capable).
    Returns a dictionary of results for this (params, n_paths) combo.
    """
    result = {"params": params.copy(), "n_paths": n_paths}
    # 1) Monte Carlo baseline (we time it)
    t0 = time.time()
    payoffs = mc_phoenix_payoffs(
        S0=params["S0"],
        K_autocall_frac=params["K_autocall_frac"],
        coupon_barrier_frac=params["coupon_barrier_frac"],
        knock_in_frac=params["knock_in_frac"],
        coupon_rate=params["coupon_rate"],
        r=params["r"],
        sigma=params["sigma"],
        T=params["T"],
        obs_count=params["obs_count"],
        n_steps=n_steps,
        n_paths=n_paths,
        seed=seed,
    )
    t_mc = time.time() - t0
    mc_mean = float(np.mean(payoffs))
    mc_var = float(np.var(payoffs, ddof=0))
    result["MC"] = {"price": mc_mean, "time": t_mc, "var": mc_var}

    # 2) For each saved model, do:
    result_models = {}
    for algo_name, model_obj in models.items():
        info = {"algo": algo_name}
        # infer input dimension and decide if parameter-only or pathwise
        input_dim = model_input_dim(model_obj)
        info["input_dim"] = input_dim

        # Attempt parameter-only prediction (easy)
        try:
            model_price = predict_surrogate_for_params(
                model_obj, params, feature_order_params
            )
            info["model_price"] = model_price
            info["model_time_estimate"] = None
        except Exception as e:
            info["model_price"] = None
            info["model_error"] = str(e)

        # Pathwise hybrid: only possible if model expects pathwise features (we need per-path features)
        # We'll try to detect: If model input dim > len(feature_order_params) => maybe pathwise. Otherwise fallback.
        pathwise_applicable = False
        if input_dim is not None:
            # if input_dim equals number of params -> it's parameter-only
            if input_dim > len(feature_order_params):
                pathwise_applicable = True

        # Build per-path features if pathwise_applicable; here we pick a simple set:
        # For each path we could provide prefix & stats. But we must match the model's training shape.
        # We attempt a commonly-used pathwise feature vector:
        #   prefix (first m prices) + prefix_min, prefix_max, prefix_mean, prefix_std, realized_vol + params
        # If model expects exactly that layout (and dimension matches), do pathwise preds.
        if pathwise_applicable:
            # Try building a pathwise feature matrix that matches common training used in your earlier notebooks:
            # We'll attempt PREFIX_LEN from model input shape - len(params) - 5 (for the 5 stats)
            n_params = len(feature_order_params)
            prefix_and_stats_len = input_dim - n_params
            # ensure at least 5 stats and some prefix elements
            if prefix_and_stats_len >= 5:
                prefix_len = prefix_and_stats_len - 5
                # simulate new fresh paths to compute these features per path
                # (We already have 'paths' inside mc_phoenix_payoffs but not returned; recreate small efficient path simulation)
                paths = simulate_gbm_paths(
                    params["S0"],
                    params["r"],
                    params["sigma"],
                    params["T"],
                    n_steps,
                    n_paths,
                    seed=seed,
                )
                features = []
                for i in range(n_paths):
                    path = paths[i]
                    prefix = path[:prefix_len] if prefix_len > 0 else np.array([])
                    prefix_min = np.min(prefix) if prefix.size else np.min(path)
                    prefix_max = np.max(prefix) if prefix.size else np.max(path)
                    prefix_mean = np.mean(prefix) if prefix.size else np.mean(path)
                    prefix_std = np.std(prefix) if prefix.size else np.std(path)
                    realized_vol = (
                        np.std(np.diff(np.log(prefix + 1e-8)))
                        if prefix.size
                        else np.std(np.diff(np.log(path + 1e-8)))
                    )
                    feats = []
                    if prefix_len > 0:
                        feats.extend(prefix.tolist())
                    feats.extend(
                        [prefix_min, prefix_max, prefix_mean, prefix_std, realized_vol]
                    )
                    feats.extend([params[k] for k in feature_order_params])
                    feats = np.array(feats, dtype=float)
                    assert (
                        feats.shape[0] == input_dim
                    ), f"constructed features dim {feats.shape[0]} != model input_dim {input_dim}"
                    features.append(feats)
                features = np.vstack(features)
                # Predict per-path
                try:
                    t_pred0 = time.time()
                    preds_pathwise = predict_surrogate_pathwise(model_obj, features)
                    t_pred = time.time() - t_pred0
                    # get hybrid corrected mean/var
                    hybrid_mean, hybrid_var = hybrid_control_variate_estimator_pathwise(
                        payoffs, preds_pathwise
                    )
                    var_reduction = (
                        (1.0 - hybrid_var / mc_var) * 100.0 if mc_var > 0 else None
                    )
                    info.update(
                        {
                            "pathwise_ok": True,
                            "pred_time": t_pred,
                            "hybrid_price": hybrid_mean,
                            "hybrid_var": hybrid_var,
                            "var_reduction_pct": var_reduction,
                        }
                    )
                except Exception as e:
                    info.update({"pathwise_ok": False, "pathwise_error": str(e)})
            else:
                info.update(
                    {
                        "pathwise_ok": False,
                        "pathwise_error": "input_dim suggests pathwise but cannot deduce prefix_len/stats layout.",
                    }
                )
        else:
            info.update(
                {
                    "pathwise_ok": False,
                    "note": "model appears parameter-only; pathwise hybrid not applicable.",
                }
            )

        result_models[algo_name] = info

    result["models"] = result_models
    return result


# -------------------------
# Runner for a grid of parameter sets
# -------------------------
def run_benchmark(
    product_name: str = "phoenix",
    grid_params: Optional[List[Dict[str, Any]]] = None,
    n_steps: int = DEFAULT_N_STEPS,
    npaths_list: Optional[List[int]] = None,
    seed: Optional[int] = 12345,
    feature_order_params: Optional[List[str]] = None,
):
    if grid_params is None:
        # sensible defaults: two scenarios
        grid_params = [
            dict(
                S0=S0,
                K_autocall_frac=1.05,
                coupon_barrier_frac=1.0,
                knock_in_frac=0.7,
                coupon_rate=2.0,
                r=0.03,
                sigma=0.2,
                T=1.0,
                obs_count=6,
            ),
            dict(
                S0=S0,
                K_autocall_frac=1.0,
                coupon_barrier_frac=0.95,
                knock_in_frac=0.6,
                coupon_rate=1.5,
                r=0.01,
                sigma=0.25,
                T=2.0,
                obs_count=8,
            ),
        ]

    if npaths_list is None:
        npaths_list = NPATHS_LIST

    if feature_order_params is None:
        feature_order_params = [
            "K_autocall_frac",
            "coupon_barrier_frac",
            "knock_in_frac",
            "coupon_rate",
            "r",
            "sigma",
            "T",
            "obs_count",
        ]

    # load models
    models = load_models_for_product(product_name)
    if not models:
        print("No models found; exiting.")
        return {}

    results = []
    for params in grid_params:
        print(f"\n--- Running param set: {params} ---")
        per_npaths = {}
        for n_paths in npaths_list:
            print(f"Running n_paths={n_paths} ...")
            res = compare_single_configuration(
                params, models, feature_order_params, n_steps, n_paths, seed=seed
            )
            per_npaths[str(n_paths)] = res["models"]
            # Add MC baseline info as top-level for this n_paths
            per_npaths[str(n_paths)]["MC"] = res["MC"]
        results.append({"params": params, "per_npaths": per_npaths})
        # Save interim results after each param set
        out_file = os.path.join(
            OUT_DIR, f"hybrid_phoenix_results_{int(time.time())}.json"
        )
        with open(out_file, "w") as fh:
            json.dump(results, fh, indent=2, default=float)
        print(f"Saved interim results to {out_file}")
    # Save final
    final_path = os.path.join(OUT_DIR, "hybrid_phoenix_results.json")
    with open(final_path, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nSaved final results to {final_path}")
    return results


# -------------------------
# Visualization (simple)
# -------------------------
def plot_results(results: List[Dict[str, Any]], out_dir: str = OUT_DIR):
    """
    Plots:
      - absolute model error vs n_paths (per model)
      - variance reduction vs n_paths (per model)
      - runtime MC vs model (per n_paths)
    """
    # flatten results for easy plotting
    rows = []
    for entry in results:
        params = entry["params"]
        for n_paths_str, block in entry["per_npaths"].items():
            n_paths = int(n_paths_str)
            mc = block.get("MC", {})
            for algo, info in block.items():
                if algo == "MC":
                    continue
                row = dict(params=params, n_paths=n_paths, algo=algo)
                row.update(info)
                # copy MC baseline price/var/time for comparison
                row["mc_price"] = mc.get("price")
                row["mc_var"] = mc.get("var")
                row["mc_time"] = mc.get("time")
                rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        print("No model results to plot.")
        return

    # Convert columns to numeric where possible
    numeric_cols = [
        "n_paths",
        "mc_price",
        "mc_var",
        "mc_time",
        "model_price",
        "hybrid_price",
        "hybrid_var",
        "pred_time",
        "model_time_estimate",
        "var_reduction_pct",
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Plot abs error (model vs MC) vs n_paths per algo
    plt.figure(figsize=(8, 5))
    for algo, g in df.groupby("algo"):
        g = g.sort_values("n_paths")
        # model absolute error vs MC
        if "model_price" in g:
            g["abs_err_model"] = (g["model_price"] - g["mc_price"]).abs()
            plt.plot(
                g["n_paths"], g["abs_err_model"], marker="o", label=f"{algo} (model)"
            )
        # hybrid abs error
        if "hybrid_price" in g and g["hybrid_price"].notna().any():
            g["abs_err_hybrid"] = (g["hybrid_price"] - g["mc_price"]).abs()
            plt.plot(
                g["n_paths"],
                g["abs_err_hybrid"],
                marker="x",
                linestyle="--",
                label=f"{algo} (hybrid)",
            )
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Number of MC paths (log scale)")
    plt.ylabel("Absolute error vs MC (log scale)")
    plt.title("Model / Hybrid Absolute Error vs Number of Paths")
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.tight_layout()
    p1 = os.path.join(out_dir, "abs_error_vs_npaths.png")
    plt.savefig(p1)
    print(f"Saved plot: {p1}")
    plt.close()

    # Plot variance reduction vs n_paths
    plt.figure(figsize=(8, 5))
    for algo, g in df.groupby("algo"):
        g = g.sort_values("n_paths")
        if "var_reduction_pct" in g:
            plt.plot(g["n_paths"], g["var_reduction_pct"], marker="o", label=algo)
    plt.xscale("log")
    plt.xlabel("Number of MC paths (log scale)")
    plt.ylabel("Variance reduction (%)")
    plt.title("Variance reduction achieved by hybrid estimator")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    p2 = os.path.join(out_dir, "var_reduction_vs_npaths.png")
    plt.savefig(p2)
    print(f"Saved plot: {p2}")
    plt.close()

    # Runtime comparison: MC vs surrogate prediction time
    plt.figure(figsize=(8, 5))
    for algo, g in df.groupby("algo"):
        g = g.sort_values("n_paths")
        if "mc_time" in g and ("pred_time" in g or "model_time_estimate" in g):
            pred_time = g["pred_time"].fillna(g["model_time_estimate"])
            plt.plot(
                g["n_paths"],
                g["mc_time"],
                marker="o",
                label="MC (baseline)",
                color="black",
            )
            plt.plot(
                g["n_paths"],
                pred_time,
                marker="x",
                label=f"{algo} (surrogate)",
                linestyle="--",
            )
            break
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Number of MC paths")
    plt.ylabel("Time (s) (log scale)")
    plt.title("Runtime: MC vs Surrogate prediction")
    plt.legend()
    plt.grid(True)
    p3 = os.path.join(out_dir, "runtime_comparison.png")
    plt.savefig(p3)
    print(f"Saved plot: {p3}")
    plt.close()

    print("All plots saved to:", out_dir)


# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    # You can change GRID scenarios here or pass via CLI in future
    GRID = [
        dict(
            S0=S0,
            K_autocall_frac=1.05,
            coupon_barrier_frac=1.0,
            knock_in_frac=0.7,
            coupon_rate=2.0,
            r=0.03,
            sigma=0.2,
            T=1.0,
            obs_count=6,
        ),
        dict(
            S0=S0,
            K_autocall_frac=1.0,
            coupon_barrier_frac=0.95,
            knock_in_frac=0.6,
            coupon_rate=1.5,
            r=0.01,
            sigma=0.25,
            T=2.0,
            obs_count=8,
        ),
    ]

    print("Loading models for 'phoenix' from:", MODEL_DIR)
    results = run_benchmark(
        product_name="phoenix",
        grid_params=GRID,
        n_steps=DEFAULT_N_STEPS,
        npaths_list=NPATHS_LIST,
        seed=12345,
    )
    # Save a copy with timestamp
    ts = int(time.time())
    outjson = os.path.join(OUT_DIR, f"hybrid_phoenix_results_final_{ts}.json")
    with open(outjson, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print("Saved final results JSON to:", outjson)

    # Plot results
    try:
        plot_results(results, out_dir=OUT_DIR)
    except Exception as e:
        print("Plotting failed:", e)

    print("Done.")
