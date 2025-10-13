"""
hybrid_pricing_trees.py

Hybrid Monte Carlo pricing using tree-based surrogate models (LightGBM, XGBoost, CatBoost).
Implements:
    - Plain Monte Carlo pricing
    - Surrogate model pricing
    - Hybrid (Control Variate) Monte Carlo pricing
Also generates benchmark comparisons, saves results, and visualizes performance metrics.
"""

import os
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

from src.training.generate_training_data import simulate_gbm_paths

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # /src
MODEL_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

N_STEPS = 100
N_PATHS = 100_000
S0 = 100.0
USE_LOG_TARGET = True  # must match training


# ---------------------------------------------------------------------
# Model Loading
# ---------------------------------------------------------------------
def load_models(option_type: str):
    """Load all trained tree-based models for a given option type."""
    models = {}
    lgb_path = os.path.join(MODEL_DIR, f"lightgbm_{option_type}.txt")
    xgb_path = os.path.join(MODEL_DIR, f"xgboost_{option_type}.json")
    cat_path = os.path.join(MODEL_DIR, f"catboost_{option_type}.cbm")

    if os.path.exists(lgb_path):
        print(f"Loading LightGBM model from {lgb_path}")
        models["LightGBM"] = lgb.Booster(model_file=lgb_path)

    if os.path.exists(xgb_path):
        print(f"Loading XGBoost model from {xgb_path}")
        xgb_model = xgb.XGBRegressor()
        xgb_model.load_model(xgb_path)
        models["XGBoost"] = xgb_model

    if os.path.exists(cat_path):
        print(f"Loading CatBoost model from {cat_path}")
        cat_model = CatBoostRegressor(verbose=0)
        cat_model.load_model(cat_path)
        models["CatBoost"] = cat_model

    return models


# ---------------------------------------------------------------------
# Monte Carlo Pricing
# ---------------------------------------------------------------------
def mc_pricer(s0, K, T, r, sigma, option_type, n_steps=N_STEPS, n_paths=N_PATHS):
    """Standard Monte Carlo pricing."""
    paths = simulate_gbm_paths(s0, r, sigma, T, n_steps, n_paths)

    if option_type == "call":
        payoffs = np.maximum(paths[:, -1] - K, 0.0)
    elif option_type == "put":
        payoffs = np.maximum(K - paths[:, -1], 0.0)
    elif option_type == "digital":
        payoffs = (paths[:, -1] > K).astype(float)
    elif option_type == "asian":
        payoffs = np.maximum(np.mean(paths, axis=1) - K, 0.0)
    else:
        raise ValueError(f"Unknown option type: {option_type}")

    return np.exp(-r * T) * np.mean(payoffs)


# ---------------------------------------------------------------------
# Surrogate Pricer
# ---------------------------------------------------------------------
def surrogate_pricer(model, algo, K, T, r, sigma, option_type):
    """Predict option price using surrogate model."""
    x = np.array([[K, T, r, sigma]])
    pred = model.predict(x)[0]

    if USE_LOG_TARGET:
        pred = np.expm1(pred)

    return float(pred)


# ---------------------------------------------------------------------
# Hybrid (Control Variate) Monte Carlo
# ---------------------------------------------------------------------
def hybrid_pricer_control_variate(
    model, algo, s0, K, T, r, sigma, option_type, n_steps=N_STEPS, n_paths=5000
):
    """
    Hybrid Monte Carlo pricer using surrogate model as control variate.
    - Runs a reduced-path MC.
    - Uses surrogate predictions to reduce MC variance.
    """
    paths = simulate_gbm_paths(s0, r, sigma, T, n_steps, n_paths)

    if option_type == "call":
        payoffs = np.maximum(paths[:, -1] - K, 0.0)
    elif option_type == "put":
        payoffs = np.maximum(K - paths[:, -1], 0.0)
    elif option_type == "digital":
        payoffs = (paths[:, -1] > K).astype(float)
    elif option_type == "asian":
        payoffs = np.maximum(np.mean(paths, axis=1) - K, 0.0)
    else:
        raise ValueError(f"Unknown option type: {option_type}")

    mc_partial = np.exp(-r * T) * payoffs
    features = np.array([[K, T, r, sigma]] * n_paths)
    surrogate_preds = model.predict(features)

    if USE_LOG_TARGET:
        surrogate_preds = np.expm1(surrogate_preds)

    # Control variate adjustment
    residuals = mc_partial - surrogate_preds
    hybrid_price = np.mean(surrogate_preds) + np.mean(residuals)

    return float(hybrid_price)


# ---------------------------------------------------------------------
# Comparison Function
# ---------------------------------------------------------------------
def compare_pricing(K=100, T=1.0, r=0.03, sigma=0.2, option_type="call"):
    print(
        f"\n=== Comparing {option_type.upper()} (K={K}, T={T}, r={r}, sigma={sigma}) ==="
    )

    # Full Monte Carlo
    t0 = time.time()
    mc_price = mc_pricer(S0, K, T, r, sigma, option_type)
    mc_time = time.time() - t0

    results = {"MC": {"price": mc_price, "time": mc_time}}

    # Surrogate and hybrid
    models = load_models(option_type)
    for algo, model in models.items():
        t1 = time.time()
        surrogate_price = surrogate_pricer(model, algo, K, T, r, sigma, option_type)
        surrogate_time = time.time() - t1

        t2 = time.time()
        hybrid_price = hybrid_pricer_control_variate(
            model, algo, S0, K, T, r, sigma, option_type
        )
        hybrid_time = time.time() - t2

        results[algo] = {
            "surrogate_price": surrogate_price,
            "hybrid_price": hybrid_price,
            "surrogate_error": abs(surrogate_price - mc_price),
            "hybrid_error": abs(hybrid_price - mc_price),
            "surrogate_time": surrogate_time,
            "hybrid_time": hybrid_time,
        }

        print(
            f"{algo}: "
            f"Surrogate={surrogate_price:.4f} (err={results[algo]['surrogate_error']:.4f}, t={surrogate_time:.5f}s), "
            f"Hybrid={hybrid_price:.4f} (err={results[algo]['hybrid_error']:.4f}, t={hybrid_time:.5f}s)"
        )

    print(f"MC Price = {mc_price:.4f} (t={mc_time:.4f}s)")
    return results


# ---------------------------------------------------------------------
# Benchmark Runner
# ---------------------------------------------------------------------
def run_benchmark(test_grid=None):
    if test_grid is None:
        test_grid = [
            (100, 1.0, 0.03, 0.2),
            (120, 0.5, 0.01, 0.25),
            (80, 2.0, 0.05, 0.15),
        ]

    results_all = []
    for opt in ["put", "call", "digital", "asian"]:
        for K, T, r, sigma in test_grid:
            res = compare_pricing(K, T, r, sigma, opt)
            for algo, vals in res.items():
                vals.update(
                    {
                        "option": opt,
                        "K": K,
                        "T": T,
                        "r": r,
                        "sigma": sigma,
                        "algo": algo,
                    }
                )
                results_all.append(vals)

    df = pd.DataFrame(results_all)
    results_path = os.path.join(RESULTS_DIR, "hybrid_results.csv")
    df.to_csv(results_path, index=False)
    print(f"\nResults saved to {results_path}")
    return df


# ---------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------
def plot_results(df):
    plt.style.use("seaborn-v0_8-darkgrid")

    # Filter out MC rows
    df_plot = df[df["algo"] != "MC"]

    # Error comparison
    plt.figure(figsize=(10, 6))
    for algo in df_plot["algo"].unique():
        subset = df_plot[df_plot["algo"] == algo]
        plt.scatter(
            subset["option"],
            subset["surrogate_error"],
            label=f"{algo} - Surrogate",
            marker="o",
        )
        plt.scatter(
            subset["option"],
            subset["hybrid_error"],
            label=f"{algo} - Hybrid",
            marker="x",
        )
    plt.title("Absolute Error: Surrogate vs Hybrid")
    plt.ylabel("Absolute Error")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "error_comparison.png"), dpi=300)

    # Timing comparison
    plt.figure(figsize=(10, 6))
    for algo in df_plot["algo"].unique():
        subset = df_plot[df_plot["algo"] == algo]
        plt.scatter(
            subset["option"],
            subset["surrogate_time"],
            label=f"{algo} - Surrogate",
            marker="o",
        )
        plt.scatter(
            subset["option"],
            subset["hybrid_time"],
            label=f"{algo} - Hybrid",
            marker="x",
        )
    plt.title("Computation Time: Surrogate vs Hybrid")
    plt.ylabel("Seconds")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "time_comparison.png"), dpi=300)
    print(f"Plots saved in {RESULTS_DIR}")


# ---------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------
if __name__ == "__main__":
    df = run_benchmark()
    plot_results(df)
    print("\nHybrid Monte Carlo pricing benchmark completed.")
