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
    r_list=[0.0, 0.01, 0.02, 0.03, 0.05],
    sigma_list=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4],
    T_list=[0.1, 0.25, 0.5, 0.75, 1.0, 2.0],
    K_list=[60, 70, 80, 90, 100, 110, 120, 130, 140],
    n_steps=100,
    n_paths=3000,
    option_types=("call", "put", "digital", "asian"),
    prefix_len=15,
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

                        # Option payoff logic
                        if option_type == "call":
                            payoffs = np.maximum(paths[:, -1] - K, 0.0)
                            opt_flag = 1.0
                        elif option_type == "put":
                            payoffs = np.maximum(K - paths[:, -1], 0.0)
                            opt_flag = 0.0
                        elif option_type == "digital":
                            payoffs = (paths[:, -1] > K).astype(float)
                            opt_flag = 2.0
                        elif option_type == "asian":
                            payoffs = np.maximum(np.mean(paths, axis=1) - K, 0.0)
                            opt_flag = 3.0
                        else:
                            continue

                        discounted = np.exp(-r * T) * payoffs

                        # Features: prefix of path + params
                        prefix = paths[:, :prefix_len]
                        # Add Gaussian noise to prefix for robustness
                        prefix += np.random.normal(
                            0, 0.01 * np.abs(prefix), prefix.shape
                        )

                        # Additional features: min, max, mean, std of prefix, realized volatility
                        prefix_min = np.min(prefix, axis=1, keepdims=True)
                        prefix_max = np.max(prefix, axis=1, keepdims=True)
                        prefix_mean = np.mean(prefix, axis=1, keepdims=True)
                        prefix_std = np.std(prefix, axis=1, keepdims=True)
                        realized_vol = np.std(
                            np.diff(np.log(prefix + 1e-8), axis=1),
                            axis=1,
                            keepdims=True,
                        )

                        params = np.array([K, T, r, sigma, opt_flag])[None, :].repeat(
                            n_paths, axis=0
                        )
                        features = np.hstack(
                            [
                                prefix,
                                prefix_min,
                                prefix_max,
                                prefix_mean,
                                prefix_std,
                                realized_vol,
                                params,
                            ]
                        )

                        X_all.append(features)
                        y_all.append(discounted)

    X_all = np.vstack(X_all)
    y_all = np.concatenate(y_all)

    # Shuffle data for randomness
    idx = np.random.permutation(X_all.shape[0])
    X_all = X_all[idx]
    y_all = y_all[idx]

    # Normalize features (except categorical opt_flag)
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_all_scaled = scaler.fit_transform(X_all[:, :-1])
    X_all_final = np.hstack([X_all_scaled, X_all[:, -1].reshape(-1, 1)])

    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    np.savez_compressed(out_file, X=X_all_final, y=y_all)

    print(f"Saved improved dataset to {out_file}")
    print(f"X shape: {X_all_final.shape}, y shape: {y_all.shape}")
    print(
        f"Feature vector = prefix({prefix_len}) + [min, max, mean, std, realized_vol] + [K, T, r, sigma, option_flag]"
    )
    return X_all_final, y_all


if __name__ == "__main__":
    generate_training_data()
