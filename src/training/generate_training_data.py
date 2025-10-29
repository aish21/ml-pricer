import os
import numpy as np


def simulate_gbm_paths(S0, r, sigma, T, num_steps, num_paths, seed=None):
    """
    Whips up a bunch of stock price paths using Geometric Brownian Motion.

    It's the standard $S_{t+dt} = S_t \exp(...)$ formula.
    Works in log-space for stability, then converts back at the end.
    """
    if seed is not None:
        np.random.seed(seed)

    dt = T / num_steps

    # Random shocks
    z = np.random.randn(num_paths, num_steps)

    # Calculate all the log increments at once
    log_increments = (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z

    # Start in log-space
    log_paths = np.zeros((num_paths, num_steps + 1))
    log_paths[:, 0] = np.log(S0)

    # Cumulatively sum the increments
    for t in range(1, num_steps + 1):
        log_paths[:, t] = log_paths[:, t - 1] + log_increments[:, t - 1]

    # Convert back to normal prices
    return np.exp(log_paths)


def generate_training_data(
    S0=100,
    r_bounds=(0.0, 0.05),
    sigma_bounds=(0.05, 0.5),
    T_bounds=(0.05, 2.0),
    K_bounds=(60, 140),
    option_types=("call", "put", "digital", "asian"),
    num_steps=50,
    num_paths=10000,
    max_samples=500000,
    prefix_length=10,
    out_file="data/raw/training_data_trees.npz",
    seed=42,
):
    """
    Creates a big dataset for training a model. Each row is one simulated path.

    - Features (X): The first few price points ('prefix'), some stats
      about that prefix, and all the option parameters (K, T, etc.).
    - Label (y): The final discounted payoff for that *single* path.

    This version is NOISY because the label comes from just one path,
    not an average.
    """
    if seed is not None:
        np.random.seed(seed)

    all_X, all_y = [], []
    total_samples = 0

    print(f"Starting data generation (target: {max_samples} samples)...")

    while total_samples < max_samples:
        # Sample random parameters from the given ranges
        r = np.random.uniform(*r_bounds)
        sigma = np.random.uniform(*sigma_bounds)
        T = np.random.uniform(*T_bounds)
        K = np.random.uniform(*K_bounds)
        option_type = np.random.choice(option_types)

        # Simulate a batch of paths with these params
        paths = simulate_gbm_paths(S0, r, sigma, T, num_steps, num_paths)

        # Figure out the payoff for each path
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
            # Average price call
            payoffs = np.maximum(np.mean(paths, axis=1) - K, 0.0)
            opt_flag = 3.0
        else:
            continue

        disc_payoffs = np.exp(-r * T) * payoffs

        # --- Feature Engineering ---

        # 1. The path prefix
        prefix = paths[:, :prefix_length]

        # 2. Stats on the prefix
        prefix_min = np.min(prefix, axis=1, keepdims=True)
        prefix_max = np.max(prefix, axis=1, keepdims=True)
        prefix_mean = np.mean(prefix, axis=1, keepdims=True)
        prefix_std = np.std(prefix, axis=1, keepdims=True)

        # Realized vol over the prefix
        realized_vol = np.std(
            np.diff(np.log(prefix + 1e-8), axis=1),
            axis=1,
            keepdims=True,
        )

        # 3. Option parameters
        # Tile them to match the number of paths
        params = np.array([K, T, r, sigma, opt_flag])[None, :].repeat(num_paths, axis=0)

        # Stack everything together
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

        # Make sure we don't go over the max sample count
        remaining = max_samples - total_samples
        if features.shape[0] > remaining:
            features = features[:remaining]
            disc_payoffs = disc_payoffs[:remaining]

        all_X.append(features)
        all_y.append(disc_payoffs)
        total_samples += features.shape[0]

    # Combine all batches
    all_X = np.vstack(all_X)
    all_y = np.concatenate(all_y)

    # Shuffle everything up
    shuffle_idx = np.random.permutation(all_X.shape[0])
    all_X = all_X[shuffle_idx]
    all_y = all_y[shuffle_idx]

    # Save to disk
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    np.savez(out_file, X=all_X, y=all_y)

    print(f"Done. Saved raw dataset to {out_file}")
    print(f"X shape: {all_X.shape}, y shape: {all_y.shape}")
    print(
        f"Features = prefix({prefix_length}) + [min, max, mean, std, vol] + [K, T, r, sigma, opt_flag]"
    )
    print(
        f"y stats: mean={all_y.mean():.4f}, std={all_y.std():.4f}, min={all_y.min():.4f}, max={all_y.max():.4f}"
    )
    return all_X, all_y


def generate_expected_training_data(
    S0=100,
    r_grid=[0.0, 0.01, 0.02, 0.03, 0.05],
    sigma_grid=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4],
    T_grid=[0.1, 0.25, 0.5, 0.75, 1.0, 2.0],
    K_grid=[60, 70, 80, 90, 100, 110, 120, 130, 140],
    option_types=("call", "put", "digital", "asian"),
    num_steps=50,
    num_paths=5000,  # MC paths per parameter set
    out_file="data/raw/training_data_expected.npz",
    seed=123,
):
    """
    This one is different. It's one row per *combination of parameters*.

    - Features (X): Just the parameters [K, T, r, sigma, option_flag].
    - Label (y): The *average* discounted payoff from a full MC simulation.

    This creates a much cleaner, less noisy dataset. Good for pricing models.
    """
    if seed is not None:
        np.random.seed(seed)

    rows = []
    total_combos = (
        len(r_grid) * len(sigma_grid) * len(T_grid) * len(K_grid) * len(option_types)
    )
    print(f"Generating expected values for {total_combos} param combos...")

    count = 0

    # Loop over the grid of all parameters
    for r in r_grid:
        for sigma in sigma_grid:
            for T in T_grid:
                for K in K_grid:
                    for option_type in option_types:

                        count += 1
                        if count % 1000 == 0:
                            print(f"Processing combo {count} / {total_combos}...")

                        # Run a full MC sim *just for this parameter set*
                        paths = simulate_gbm_paths(
                            S0, r, sigma, T, num_steps, num_paths
                        )

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

                        # Get the discounted payoffs...
                        disc_payoffs = np.exp(-r * T) * payoffs

                        # ...and average them. This is our "true" price.
                        avg_payoff = np.mean(disc_payoffs)

                        # Feature vector is just the params
                        features = [K, T, r, sigma, opt_flag]
                        rows.append((features, avg_payoff))

    # Convert list of tuples to X and y arrays
    all_X = np.array([row[0] for row in rows])
    all_y = np.array([row[1] for row in rows])

    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    np.savez(out_file, X=all_X, y=all_y)

    print(f"\nDone. Saved expected dataset to {out_file}")
    print(f"X shape: {all_X.shape}, y shape: {all_y.shape}")
    print(
        f"y stats: mean={all_y.mean():.4f}, std={all_y.std():.4f}, min={all_y.min():.4f}, max={all_y.max():.4f}"
    )
    return all_X, all_y


if __name__ == "__main__":
    # generate_training_data()
    generate_expected_training_data()
