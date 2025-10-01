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


if __name__ == "__main__":
    # Example usage
    model_path = os.path.join("src", "models", "mlp_large_tuned.h5")
    s0 = 100
    K = 100
    T = 1.0
    r = 0.03
    sigma = 0.2
    option_type = "call"
    n_steps_partial = 15
    n_paths = 10000
    price = hybrid_pricing_nn_mc(
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
