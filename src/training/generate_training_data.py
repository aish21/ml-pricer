import os
import numpy as np


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

    return np.exp(log_paths)  # shape: (n_paths, n_steps+1)


def generate_training_data(
    s0=100,
    r_list=[0.0, 0.03],
    sigma_list=[0.1, 0.2, 0.3],
    T_list=[0.25, 0.5, 1.0],
    K_list=[80, 90, 100, 110, 120],
    n_steps=50,
    n_paths=2500,
    option_types=("call", "put"),
    prefix_len=10,
    out_file="data/raw/training_data.npz",
    seed=42,
):
    """
    Generate (prefix + contract params -> payoff) dataset for NN training.
    """
    if seed is not None:
        np.random.seed(seed)

    X_all, y_all = [], []

    for r in r_list:
        for sigma in sigma_list:
            for T in T_list:
                for K in K_list:
                    for option_type in option_types:
                        paths = simulate_gbm_paths(s0, r, sigma, T, n_steps, n_paths)

                        if option_type == "call":
                            payoffs = np.maximum(paths[:, -1] - K, 0.0)
                            opt_flag = 1.0
                        else:
                            payoffs = np.maximum(K - paths[:, -1], 0.0)
                            opt_flag = 0.0

                        discounted = np.exp(-r * T) * payoffs

                        # Features: prefix of path + params
                        prefix = paths[:, :prefix_len]
                        params = np.array([K, T, r, sigma, opt_flag])[None, :].repeat(
                            n_paths, axis=0
                        )
                        features = np.hstack([prefix, params])

                        X_all.append(features)
                        y_all.append(discounted)

    X_all = np.vstack(X_all)
    y_all = np.concatenate(y_all)

    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    np.savez_compressed(out_file, X=X_all, y=y_all)

    print(f"Saved dataset to {out_file}")
    print(f"X shape: {X_all.shape}, y shape: {y_all.shape}")
    print(f"Feature vector = prefix({prefix_len}) + [K, T, r, sigma, option_flag]")
    return X_all, y_all


if __name__ == "__main__":
    generate_training_data()
