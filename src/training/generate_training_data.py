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

    return np.exp(log_paths)  # Convert back to price paths


def generate_training_data(
    s0=100,
    r_range=(0.0, 0.05),
    sigma_range=(0.05, 0.5),
    T_range=(0.05, 2.0),
    K_range=(60, 140),
    option_types=("call", "put", "digital", "asian"),
    n_steps=50,
    n_paths=10000,
    max_total_samples=500000,
    prefix_len=10,
    out_file="data/raw/training_data_trees.npz",
    seed=42,
):
    """
    Generate dataset: (prefix + stats + params -> discounted payoff).
    """
    if seed is not None:
        np.random.seed(seed)

    X_all, y_all = [], []
    sample_count = 0

    while sample_count < max_total_samples:
        # Sample random parameters instead of fixed grid
        r = np.random.uniform(*r_range)
        sigma = np.random.uniform(*sigma_range)
        T = np.random.uniform(*T_range)
        K = np.random.uniform(*K_range)
        option_type = np.random.choice(option_types)

        paths = simulate_gbm_paths(s0, r, sigma, T, n_steps, n_paths)

        # Option payoff
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

        # Prefix features
        prefix = paths[:, :prefix_len]

        # Stats on prefix
        prefix_min = np.min(prefix, axis=1, keepdims=True)
        prefix_max = np.max(prefix, axis=1, keepdims=True)
        prefix_mean = np.mean(prefix, axis=1, keepdims=True)
        prefix_std = np.std(prefix, axis=1, keepdims=True)
        realized_vol = np.std(
            np.diff(np.log(prefix + 1e-8), axis=1),
            axis=1,
            keepdims=True,
        )

        params = np.array([K, T, r, sigma, opt_flag])[None, :].repeat(n_paths, axis=0)

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

        # Check sample cap
        remaining = max_total_samples - sample_count
        if features.shape[0] > remaining:
            features = features[:remaining]
            discounted = discounted[:remaining]

        X_all.append(features)
        y_all.append(discounted)
        sample_count += features.shape[0]

    X_all = np.vstack(X_all)
    y_all = np.concatenate(y_all)

    # Shuffle
    idx = np.random.permutation(X_all.shape[0])
    X_all = X_all[idx]
    y_all = y_all[idx]

    # Save raw data
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    np.savez(out_file, X=X_all, y=y_all)

    print(f"Saved raw dataset to {out_file}")
    print(f"X shape: {X_all.shape}, y shape: {y_all.shape}")
    print(
        f"Feature vector = prefix({prefix_len}) + [min, max, mean, std, realized_vol] + [K, T, r, sigma, option_flag]"
    )
    print(
        f"y stats: mean={y_all.mean():.4f}, std={y_all.std():.4f}, min={y_all.min():.4f}, max={y_all.max():.4f}"
    )
    return X_all, y_all


def generate_expected_training_data(
    s0=100,
    r_list=[0.0, 0.01, 0.02, 0.03, 0.05],
    sigma_list=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4],
    T_list=[0.1, 0.25, 0.5, 0.75, 1.0, 2.0],
    K_list=[60, 70, 80, 90, 100, 110, 120, 130, 140],
    option_types=("call", "put", "digital", "asian"),
    n_steps=50,
    n_paths=5000,
    out_file="data/raw/training_data_expected.npz",
    seed=123,
):
    """
    New version: compute *expected discounted payoff per parameter combo*.
    Much smoother labels, better for tree models.
    """
    if seed is not None:
        np.random.seed(seed)

    rows = []

    for r in r_list:
        for sigma in sigma_list:
            for T in T_list:
                for K in K_list:
                    for option_type in option_types:
                        paths = simulate_gbm_paths(s0, r, sigma, T, n_steps, n_paths)

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
                        expected_payoff = np.mean(discounted)

                        # Feature vector = [K, T, r, sigma, option_flag]
                        features = [K, T, r, sigma, opt_flag]
                        rows.append((features, expected_payoff))

    X_all = np.array([row[0] for row in rows])
    y_all = np.array([row[1] for row in rows])

    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    np.savez(out_file, X=X_all, y=y_all)

    print(f"Saved expected dataset to {out_file}")
    print(f"X shape: {X_all.shape}, y shape: {y_all.shape}")
    print(
        f"y stats: mean={y_all.mean():.4f}, std={y_all.std():.4f}, min={y_all.min():.4f}, max={y_all.max():.4f}"
    )
    return X_all, y_all


if __name__ == "__main__":
    # generate_training_data()
    generate_expected_training_data()
