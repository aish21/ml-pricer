import os
import numpy as np
from tensorflow.keras.models import load_model
from src.training.generate_training_data import simulate_gbm_paths


def hybrid_pricing_nn_mc(
    model_path,
    s0,
    K,
    T,
    r,
    sigma,
    option_type="call",
    n_steps_partial=15,
    n_paths=10000,
    discount=True,
):
    """
    Hybrid Pricing: Use NN to predict payoff from partial MC paths.
    """
    # Load trained NN model
    model = load_model(model_path, compile=False)

    # Simulate partial GBM paths up to t0
    paths = simulate_gbm_paths(s0, r, sigma, T, n_steps_partial, n_paths)
    prefix = paths[:, :n_steps_partial]

    # Add statistical features to prefix (match training)
    prefix_min = np.min(prefix, axis=1, keepdims=True)
    prefix_max = np.max(prefix, axis=1, keepdims=True)
    prefix_mean = np.mean(prefix, axis=1, keepdims=True)
    prefix_std = np.std(prefix, axis=1, keepdims=True)
    realized_vol = np.std(np.diff(np.log(prefix + 1e-8), axis=1), axis=1, keepdims=True)

    # Prepare contract params for each path
    opt_flag = {
        "call": 1.0,
        "put": 0.0,
        "digital": 2.0,
        "asian": 3.0,
    }.get(option_type, 1.0)
    params = np.array([K, T, r, sigma, opt_flag])[None, :].repeat(n_paths, axis=0)

    # Feature vector: prefix + stats + params (match training)
    X = np.hstack(
        [prefix, prefix_min, prefix_max, prefix_mean, prefix_std, realized_vol, params]
    )

    # Predict payoff using NN
    y_pred = model.predict(X).flatten()

    # Average and discount
    avg_payoff = np.mean(y_pred)
    if discount:
        price = np.exp(-r * T) * avg_payoff
    else:
        price = avg_payoff

    print(f"Hybrid NN-MC price: {price:.4f} (avg NN payoff: {avg_payoff:.4f})")
    return price


def standard_mc_price(s0, K, T, r, sigma, option_type, n_steps, n_paths, discount=True):
    """
    Standard Monte Carlo pricing for basic options.
    """
    paths = simulate_gbm_paths(s0, r, sigma, T, n_steps, n_paths)
    if option_type == "call":
        payoff = np.maximum(paths[:, -1] - K, 0)
    elif option_type == "put":
        payoff = np.maximum(K - paths[:, -1], 0)
    elif option_type == "digital":
        payoff = (paths[:, -1] > K).astype(float)
    elif option_type == "asian":
        payoff = np.maximum(np.mean(paths, axis=1) - K, 0)
    else:
        payoff = np.maximum(paths[:, -1] - K, 0)
    avg_payoff = np.mean(payoff)
    if discount:
        price = np.exp(-r * T) * avg_payoff
    else:
        price = avg_payoff
    return price


if __name__ == "__main__":
    # Test multiple payoffs and strikes
    model_path = os.path.join("src", "models", "mlp_large_tuned.h5")
    s0 = 100
    T = 1.0
    r = 0.03
    sigma = 0.2
    n_steps_partial = 15
    n_paths = 10000
    strikes = [80, 90, 100, 110, 120]
    option_types = ["call", "put", "digital", "asian"]
    results = []
    n_steps_mc = 100  # Use full path for MC pricing

    # Diagnostics: Print sample MC paths, payoffs, and NN predictions for one setup
    diag_strike = 100
    diag_type = "call"
    print("\n--- Diagnostics for strike=100, type=call ---")
    mc_paths = simulate_gbm_paths(s0, r, sigma, T, n_steps_mc, 5)
    print("Sample MC paths (5):")
    print(mc_paths)
    mc_payoffs = np.maximum(mc_paths[:, -1] - diag_strike, 0)
    print("Sample MC payoffs:", mc_payoffs)
    nn_paths = simulate_gbm_paths(s0, r, sigma, T, n_steps_partial, 5)
    nn_prefix = nn_paths[:, :n_steps_partial]
    nn_prefix_min = np.min(nn_prefix, axis=1, keepdims=True)
    nn_prefix_max = np.max(nn_prefix, axis=1, keepdims=True)
    nn_prefix_mean = np.mean(nn_prefix, axis=1, keepdims=True)
    nn_prefix_std = np.std(nn_prefix, axis=1, keepdims=True)
    nn_realized_vol = np.std(
        np.diff(np.log(nn_prefix + 1e-8), axis=1), axis=1, keepdims=True
    )
    nn_opt_flag = 1.0
    nn_params = np.array([diag_strike, T, r, sigma, nn_opt_flag])[None, :].repeat(
        5, axis=0
    )
    nn_X = np.hstack(
        [
            nn_prefix,
            nn_prefix_min,
            nn_prefix_max,
            nn_prefix_mean,
            nn_prefix_std,
            nn_realized_vol,
            nn_params,
        ]
    )
    model = load_model(model_path, compile=False)
    nn_preds = model.predict(nn_X).flatten()
    print("Sample NN predictions:", nn_preds)
    print(
        f"NN pred stats: mean={nn_preds.mean():.4f}, min={nn_preds.min():.4f}, max={nn_preds.max():.4f}"
    )
    # Test multiple payoffs and strikes
    model_path = os.path.join("src", "models", "mlp_large_tuned.h5")
    s0 = 100
    T = 1.0
    r = 0.03
    sigma = 0.2
    n_steps_partial = 15
    n_paths = 10000
    strikes = [80, 90, 100, 110, 120]
    option_types = ["call", "put", "digital", "asian"]
    results = []
    n_steps_mc = 100  # Use full path for MC pricing
    # Diagnostics: Print sample MC paths, payoffs, and NN predictions for one setup
    diag_strike = 100
    diag_type = "call"
    print("\n--- Diagnostics for strike=100, type=call ---")
    mc_paths = simulate_gbm_paths(s0, r, sigma, T, n_steps_mc, 5)
    print("Sample MC paths (5):")
    print(mc_paths)
    mc_payoffs = np.maximum(mc_paths[:, -1] - diag_strike, 0)
    print("Sample MC payoffs:", mc_payoffs)
    nn_paths = simulate_gbm_paths(s0, r, sigma, T, n_steps_partial, 5)
    nn_prefix = nn_paths[:, :n_steps_partial]
    nn_prefix_min = np.min(nn_prefix, axis=1, keepdims=True)
    nn_prefix_max = np.max(nn_prefix, axis=1, keepdims=True)
    nn_prefix_mean = np.mean(nn_prefix, axis=1, keepdims=True)
    nn_prefix_std = np.std(nn_prefix, axis=1, keepdims=True)
    nn_realized_vol = np.std(
        np.diff(np.log(nn_prefix + 1e-8), axis=1), axis=1, keepdims=True
    )
    nn_opt_flag = 1.0
    nn_params = np.array([diag_strike, T, r, sigma, nn_opt_flag])[None, :].repeat(
        5, axis=0
    )
    nn_X = np.hstack(
        [
            nn_prefix,
            nn_prefix_min,
            nn_prefix_max,
            nn_prefix_mean,
            nn_prefix_std,
            nn_realized_vol,
            nn_params,
        ]
    )
    model = load_model(model_path, compile=False)
    nn_preds = model.predict(nn_X).flatten()
    print("Sample NN predictions:", nn_preds)
    print(
        f"NN pred stats: mean={nn_preds.mean():.4f}, min={nn_preds.min():.4f}, max={nn_preds.max():.4f}"
    )

    print("\nHybrid NN-MC vs Standard MC Pricing Results:")
    print("Type    Strike    NN-MC Price    MC Price    Abs Error    % Error")
    for option_type in option_types:
        for K in strikes:
            nn_mc_price = hybrid_pricing_nn_mc(
                model_path,
                s0,
                K,
                T,
                r,
                sigma,
                option_type,
                n_steps_partial,
                n_paths,
                discount=True,
            )
            mc_price = standard_mc_price(
                s0,
                K,
                T,
                r,
                sigma,
                option_type,
                n_steps_mc,
                n_paths,
                discount=True,
            )
            abs_error = abs(nn_mc_price - mc_price)
            pct_error = 100 * abs_error / (mc_price if mc_price != 0 else 1)
            results.append(
                {
                    "type": option_type,
                    "strike": K,
                    "nn_mc": nn_mc_price,
                    "mc": mc_price,
                    "abs_error": abs_error,
                    "pct_error": pct_error,
                }
            )
            print(
                f"{option_type:7} {K:7} {nn_mc_price:12.4f} {mc_price:10.4f} {abs_error:10.4f} {pct_error:9.2f}%"
            )
    print("\nSummary Table:")
    print("Type    Strike    NN-MC Price    MC Price    Abs Error    % Error")
    for res in results:
        print(
            f"{res['type']:7} {res['strike']:7} {res['nn_mc']:12.4f} {res['mc']:10.4f} {res['abs_error']:10.4f} {res['pct_error']:9.2f}%"
        )
