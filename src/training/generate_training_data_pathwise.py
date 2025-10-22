import numpy as np
import pandas as pd
import os

OUTPUT_DIR = os.path.join("src", "data")
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_PARAM_SETS = 200  # number of random market conditions
N_PATHS_PER_SET = 2000  # per-parameter Monte Carlo paths
N_STEPS = 252  # daily discretization
S0 = 100.0


def simulate_gbm_paths(S0, r, sigma, T, n_steps, n_paths):
    dt = T / n_steps
    Z = np.random.normal(size=(n_paths, n_steps))
    increments = (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
    log_paths = np.cumsum(increments, axis=1)
    paths = S0 * np.exp(log_paths)
    paths = np.hstack((np.full((n_paths, 1), S0), paths))
    return paths


def phoenix_payoff_from_path(
    path,
    S0,
    K_autocall_frac,
    coupon_barrier_frac,
    knock_in_frac,
    coupon_rate,
    r,
    T,
    obs_count,
):
    n_steps = len(path) - 1
    obs_indices = np.linspace(0, n_steps, obs_count + 1, dtype=int)[1:]
    obs_prices = path[obs_indices]
    obs_times = obs_indices / n_steps * T

    discount = np.exp(-r * obs_times)

    coupon_barrier = S0 * coupon_barrier_frac
    autocall_barrier = S0 * K_autocall_frac
    knock_in_barrier = S0 * knock_in_frac

    coupons = 0.0
    for i, price in enumerate(obs_prices):
        if price >= coupon_barrier:
            coupons += coupon_rate / obs_count * discount[i]

        if price >= autocall_barrier:
            # Autocall triggers → redeem at par + coupon, discounted to call time
            total = (1.0 + coupon_rate / obs_count) * discount[i]
            return S0 * total  # early call payoff

    # No autocall → check maturity
    final_price = path[-1]
    if final_price < knock_in_barrier:
        redemption = final_price / S0  # proportional redemption
    else:
        redemption = 1.0  # full redemption

    total_payoff = (redemption + coupons) * np.exp(-r * T)
    return S0 * total_payoff


def generate_phoenix_pathwise_dataset(
    n_param_sets=N_PARAM_SETS, n_paths=N_PATHS_PER_SET
):
    records = []

    for _ in range(n_param_sets):
        print(f"Generating set {_+1}/{n_param_sets}...")
        # Sample random realistic parameters
        r = np.random.uniform(0.01, 0.05)
        sigma = np.random.uniform(0.1, 0.3)
        T = np.random.choice([1.0, 1.5, 2.0])
        coupon_rate = np.random.uniform(0.01, 0.03)
        K_autocall_frac = np.random.uniform(1.0, 1.05)
        coupon_barrier_frac = np.random.uniform(0.9, 1.0)
        knock_in_frac = np.random.uniform(0.6, 0.8)
        obs_count = np.random.randint(4, 12)

        paths = simulate_gbm_paths(S0, r, sigma, T, N_STEPS, n_paths)

        for p in paths:
            payoff = phoenix_payoff_from_path(
                p,
                S0,
                K_autocall_frac,
                coupon_barrier_frac,
                knock_in_frac,
                coupon_rate,
                r,
                T,
                obs_count,
            )

            stats = {
                "path_min": np.min(p),
                "path_max": np.max(p),
                "path_mean": np.mean(p),
                "path_std": np.std(p),
                "path_final": p[-1],
                "r": r,
                "sigma": sigma,
                "T": T,
                "coupon_rate": coupon_rate,
                "K_autocall_frac": K_autocall_frac,
                "coupon_barrier_frac": coupon_barrier_frac,
                "knock_in_frac": knock_in_frac,
                "obs_count": obs_count,
                "payoff": payoff,
            }
            records.append(stats)

    df = pd.DataFrame(records)
    return df


if __name__ == "__main__":
    df = generate_phoenix_pathwise_dataset()
    print(f"Generated dataset: {df.shape}")
    print(df.head())

    out_path = os.path.join(OUTPUT_DIR, "phoenix_pathwise_training.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved to {out_path}")
