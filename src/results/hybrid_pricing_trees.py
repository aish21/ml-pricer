# =====================================================
# Hybrid Pricing Pipeline: MC vs Tree-Based Surrogates
# =====================================================

import os
import time
import numpy as np

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

# ----------------------------
# Config
# ----------------------------
MODEL_DIR = "../models"
N_STEPS = 100
N_PATHS = 100_000
S0 = 100.0

option_map = {"put": 0.0, "call": 1.0, "digital": 2.0, "asian": 3.0}


def simulate_gbm_paths(s0, r, sigma, T, n_steps, n_paths, seed=None):
    """
    Simulate GBM paths using Euler discretization.
    """
    if seed is not None:
        np.random.seed(seed)

    dt = T / n_steps
    Z = np.random.randn(n_paths, n_steps)
    increments = (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z

    log_paths = np.zeros((n_paths, n_steps + 1))
    log_paths[:, 0] = np.log(s0)

    for t in range(1, n_steps + 1):
        log_paths[:, t] = log_paths[:, t - 1] + increments[:, t - 1]

    return np.exp(log_paths)  # Convert back to price paths


# ----------------------------
# Load models
# ----------------------------
def load_models(option_type):
    models = {}

    # LightGBM
    lgb_path = os.path.join(MODEL_DIR, f"lightgbm_{option_type}.txt")
    print(f"Loading LightGBM model from {lgb_path}")
    if os.path.exists(lgb_path):
        models["LightGBM"] = lgb.Booster(model_file=lgb_path)

    # XGBoost
    xgb_path = os.path.join(MODEL_DIR, f"xgboost_{option_type}.json")
    if os.path.exists(xgb_path):
        models["XGBoost"] = xgb.XGBRegressor()
        models["XGBoost"].load_model(xgb_path)

    # CatBoost
    cat_path = os.path.join(MODEL_DIR, f"catboost_{option_type}.cbm")
    if os.path.exists(cat_path):
        models["CatBoost"] = CatBoostRegressor()
        models["CatBoost"].load_model(cat_path)

    return models


# ----------------------------
# Monte Carlo pricer
# ----------------------------
def mc_pricer(s0, K, T, r, sigma, option_type, n_steps=N_STEPS, n_paths=N_PATHS):
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
        raise ValueError("Unknown option type")
    return np.exp(-r * T) * np.mean(payoffs)


# ----------------------------
# Surrogate pricer
# ----------------------------
def surrogate_pricer(model, algo, K, T, r, sigma, option_type):
    x = np.array([[K, T, r, sigma, option_map[option_type]]])
    if algo == "LightGBM":
        return model.predict(x)[0]
    elif algo == "XGBoost":
        return model.predict(x)[0]
    elif algo == "CatBoost":
        return model.predict(x)[0]
    else:
        raise ValueError(f"Unsupported algo: {algo}")


# ----------------------------
# Comparison function
# ----------------------------
def compare_pricing(K=100, T=1.0, r=0.03, sigma=0.2, option_type="call"):
    print(f"\n=== Comparing {option_type.upper()} ===")

    # MC pricing
    t0 = time.time()
    mc_price = mc_pricer(S0, K, T, r, sigma, option_type)
    mc_time = time.time() - t0

    results = {"MC": {"price": mc_price, "time": mc_time}}

    # Surrogate models
    models = load_models(option_type)
    for algo, model in models.items():
        t1 = time.time()
        pred_price = surrogate_pricer(model, algo, K, T, r, sigma, option_type)
        model_time = time.time() - t1

        results[algo] = {
            "price": pred_price,
            "time": model_time,
            "abs_error": abs(pred_price - mc_price),
        }

    # Print results
    print(f"MC Price = {mc_price:.4f} (time {mc_time:.4f}s)")
    for algo, res in results.items():
        if algo != "MC":
            print(
                f"{algo}: {res['price']:.4f} "
                f"(time {res['time']:.6f}s, abs_error={res['abs_error']:.4f})"
            )
    return results


# ----------------------------
# Batch testing
# ----------------------------
def run_benchmark():
    results_all = {}
    test_grid = [
        (100, 1.0, 0.03, 0.2),
        (120, 0.5, 0.01, 0.25),
        (80, 2.0, 0.05, 0.15),
    ]
    for opt in ["put", "call", "digital", "asian"]:
        results_all[opt] = []
        models = load_models(opt)
        if not models:
            print(f"No trained models found for {opt}, skipping...")
            continue
        for K, T, r, sigma in test_grid:
            res = compare_pricing(K, T, r, sigma, opt)
            results_all[opt].append(res)
    return results_all


if __name__ == "__main__":
    summary = run_benchmark()

    print("\n=== Final Summary ===")
    for opt, results in summary.items():
        print(f"\nOption: {opt}")
        for res in results:
            print(res)
