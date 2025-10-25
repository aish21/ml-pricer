import os
import json
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
import matplotlib.pyplot as plt

MODEL_DIR = "C:\\Users\\aisha\\OneDrive\\Desktop\\GitHub\\neural-pricer\\src\\models"
RESULTS_DIR = "C:\\Users\\aisha\\OneDrive\\Desktop\\GitHub\\neural-pricer\\src\\results\\phoenix_compare"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Monte Carlo control parameters
N_PATHS_LIST = [500, 2000, 8000]
N_STEPS = 252
SEED = 42

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
    {
        "S0": 90.0,
        "r": 0.025,
        "sigma": 0.15,
        "T": 1.5,
        "autocall_barrier_frac": 1.1,
        "coupon_barrier_frac": 0.95,
        "coupon_rate": 0.015,
        "knock_in_frac": 0.75,
        "obs_count": 6,
    },
]


def simulate_gbm_paths(S0, r, sigma, T, n_steps, n_paths, seed=None):
    if seed is not None:
        np.random.seed(seed)
    dt = T / n_steps
    drift = (r - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt)
    increments = drift + diffusion * np.random.randn(n_paths, n_steps)
    log_paths = np.zeros((n_paths, n_steps + 1))
    log_paths[:, 0] = np.log(S0)
    log_paths[:, 1:] = np.log(S0) + np.cumsum(increments, axis=1)
    return np.exp(log_paths)


def phoenix_payoff(
    paths, autocall_b, coupon_b, knockin_b, coupon_rate, r, T, obs_count
):
    n_steps = paths.shape[1] - 1
    obs_idx = np.linspace(0, n_steps, obs_count + 1, dtype=int)[1:]
    payoffs = np.zeros(paths.shape[0])

    for i, path in enumerate(paths):
        knocked_in = np.any(path < knockin_b)
        call_time = None
        for idx in obs_idx:
            if path[idx] >= autocall_b:
                call_time = idx
                break

        if call_time is not None:
            t_call = call_time / n_steps * T
            payoff = 1 + coupon_rate
            payoffs[i] = payoff * np.exp(-r * t_call)
        else:
            t_maturity = T
            if knocked_in:
                payoff = path[-1] / path[0]
            else:
                payoff = 1 + coupon_rate
            payoffs[i] = payoff * np.exp(-r * t_maturity)

    return payoffs


def mc_price(params, n_paths):
    start = time.time()
    paths = simulate_gbm_paths(
        params["S0"],
        params["r"],
        params["sigma"],
        params["T"],
        N_STEPS,
        n_paths,
        seed=SEED,
    )
    payoffs = phoenix_payoff(
        paths,
        params["S0"] * params["autocall_barrier_frac"],
        params["S0"] * params["coupon_barrier_frac"],
        params["S0"] * params["knock_in_frac"],
        params["coupon_rate"],
        params["r"],
        params["T"],
        params["obs_count"],
    )
    elapsed = time.time() - start
    return np.mean(payoffs), np.var(payoffs), elapsed


def load_models():
    models = {}
    try:
        models["LightGBM"] = lgb.Booster(
            model_file=os.path.join(MODEL_DIR, "lightgbm_phoenix.txt")
        )
    except:
        print("LightGBM model not found.")
    try:
        models["XGBoost"] = xgb.XGBRegressor()
        models["XGBoost"].load_model(os.path.join(MODEL_DIR, "xgboost_phoenix.json"))
    except:
        print("XGBoost model not found.")
    try:
        models["CatBoost"] = cb.CatBoostRegressor()
        models["CatBoost"].load_model(os.path.join(MODEL_DIR, "catboost_phoenix.cbm"))
    except:
        print("CatBoost model not found.")
    return models


def model_predict(model, features):
    """Unified prediction function for all models."""
    if isinstance(model, lgb.Booster):
        return model.predict(features)
    else:
        return model.predict(features)


def run_benchmark_suite(test_cases, n_paths_list, models, use_log_target=True, seed=42):
    results_all = []

    for idx, params in enumerate(test_cases):
        print(
            f"\n🔹 Evaluating Test Case {idx+1}/{len(test_cases)}:\n{json.dumps(params, indent=2)}"
        )

        for n_paths in n_paths_list:
            mc_p, mc_var, mc_t = mc_price(params, n_paths)

            for model_name, model in models.items():
                feats = np.array(
                    [
                        params[k]
                        for k in [
                            "S0",
                            "r",
                            "sigma",
                            "T",
                            "autocall_barrier_frac",
                            "coupon_barrier_frac",
                            "coupon_rate",
                            "knock_in_frac",
                        ]
                    ]
                ).reshape(1, -1)

                start = time.time()
                raw_pred = model_predict(model, feats)[0]
                model_t = time.time() - start

                # ✅ Correct inverse transform for log1p target
                model_price = np.expm1(raw_pred) if use_log_target else raw_pred

                abs_err = abs(model_price - mc_p)
                rel_err = abs_err / abs(mc_p) if mc_p != 0 else np.nan

                results_all.append(
                    {
                        "test_case": idx + 1,
                        "n_paths": n_paths,
                        "model": model_name,
                        "MC_price": float(mc_p),
                        "MC_time": float(mc_t),
                        "MC_var": float(mc_var),
                        "Model_price": float(model_price),
                        "Model_time": float(model_t),
                        "Abs_Error": float(abs_err),
                        "Rel_Error": float(rel_err),
                    }
                )

    df = pd.DataFrame(results_all)
    summary = (
        df.groupby("model")[["Abs_Error", "Rel_Error", "Model_time"]]
        .mean()
        .reset_index()
        .sort_values("Abs_Error")
    )

    print("\n=== Aggregate Summary ===")
    print(summary.round(6))

    out_path = os.path.join(RESULTS_DIR, "phoenix_benchmark_suite.json")
    with open(out_path, "w") as f:
        json.dump(
            {
                "generated_at": time.time(),
                "test_cases": test_cases,
                "results": df.to_dict(orient="records"),
            },
            f,
            indent=2,
            default=lambda o: float(o),
        )

    print(f"\nBenchmark suite results saved to: {out_path}")
    return df


if __name__ == "__main__":
    models = load_models()
    df_bench = run_benchmark_suite(
        TEST_CASES, N_PATHS_LIST, models, use_log_target=True
    )
