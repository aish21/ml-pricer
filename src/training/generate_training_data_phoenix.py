import os
import numpy as np
from typing import List


def simulate_gbm_paths(s0, r, sigma, T, n_steps, n_paths, seed=None):
    if seed is not None:
        np.random.seed(seed)
    dt = T / n_steps
    drift = (r - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt)
    increments = drift + diffusion * np.random.randn(n_paths, n_steps)
    log_paths = np.zeros((n_paths, n_steps + 1))
    log_paths[:, 0] = np.log(s0)
    log_paths[:, 1:] = np.log(s0) + np.cumsum(increments, axis=1)
    return np.exp(log_paths)


def phoenix_payoff_with_memory(
    path: np.ndarray,
    S0: float,
    obs_indices: List[int],
    r: float,
    T: float,
    autocall_barrier: float,
    coupon_barrier: float,
    coupon_rate_annual: float,
    knock_in_barrier: float,
    principal: float = 1.0,
):
    """
    Phoenix payoff with:
      - Monthly observation dates
      - Coupon memory (missed coupons paid later if barrier recovered)
      - Early autocall with accrued coupons
      - Continuous knock-in barrier
      - Exact discounting by call/maturity time
    """
    n_steps = len(path) - 1
    dt = T / n_steps
    knocked_in = np.any(path < (knock_in_barrier * S0))

    coupon_per_obs = coupon_rate_annual / 12.0
    accrued_coupons = 0
    total_pv = 0.0

    for obs_idx in obs_indices:
        S_obs = path[obs_idx]
        t_obs = obs_idx * dt
        df = np.exp(-r * t_obs)

        if S_obs >= coupon_barrier * S0:
            # pay all accrued + current coupon
            total_pv += principal * (accrued_coupons + coupon_per_obs) * df
            accrued_coupons = 0
        else:
            # missed coupon gets remembered
            accrued_coupons += coupon_per_obs

        # autocall event
        if S_obs >= autocall_barrier * S0:
            total_pv += principal * df
            return total_pv

    # final redemption (no autocall)
    S_T = path[-1]
    df_T = np.exp(-r * T)

    # if no knock-in or S_T >= KI barrier -> full redemption + any due coupons
    if not knocked_in or S_T >= knock_in_barrier * S0:
        total_pv += principal * df_T
        # if at maturity above coupon barrier, pay any remaining coupons
        if S_T >= coupon_barrier * S0:
            total_pv += principal * accrued_coupons * df_T
    else:
        # knocked in and below KI -> capital loss proportional to S_T/S0
        redemption = principal * (S_T / S0)
        total_pv += redemption * df_T

    return total_pv


def expected_phoenix_price(
    S0,
    r,
    sigma,
    T,
    autocall_barrier,
    coupon_barrier,
    coupon_rate,
    knock_in_barrier,
    n_paths=20000,
    n_steps=252 * 3,
    principal=1.0,
    seed=None,
):
    paths = simulate_gbm_paths(S0, r, sigma, T, n_steps, n_paths, seed)
    obs_per_year = 12
    obs_times = np.linspace(T / obs_per_year, T, int(T * obs_per_year))
    obs_indices = (obs_times / (T / n_steps)).astype(int)

    payoffs = np.zeros(n_paths)
    for i in range(n_paths):
        payoffs[i] = phoenix_payoff_with_memory(
            path=paths[i],
            S0=S0,
            obs_indices=obs_indices,
            r=r,
            T=T,
            autocall_barrier=autocall_barrier,
            coupon_barrier=coupon_barrier,
            coupon_rate_annual=coupon_rate,
            knock_in_barrier=knock_in_barrier,
            principal=principal,
        )
    return float(np.mean(payoffs))


def generate_phoenix_dataset(
    out_file="data/raw/training_data_phoenix_memory.npz",
    n_samples=1000,
    n_paths=10000,
    n_steps=252 * 3,
    S0=100.0,
    seed=123,
):
    rng = np.random.default_rng(seed)

    r_range = (0.0, 0.05)
    sigma_range = (0.1, 0.4)
    T_range = (0.5, 3.0)
    autocall_barrier_range = (1.00, 1.05)
    coupon_barrier_range = (0.65, 0.8)
    knock_in_barrier_range = (0.6, 0.7)
    coupon_rate_range = (0.06, 0.12)

    X, y = [], []

    for i in range(n_samples):
        r = rng.uniform(*r_range)
        sigma = rng.uniform(*sigma_range)
        T = rng.uniform(*T_range)
        autocall_barrier = rng.uniform(*autocall_barrier_range)
        coupon_barrier = rng.uniform(*coupon_barrier_range)
        knock_in_barrier = rng.uniform(*knock_in_barrier_range)
        coupon_rate = rng.uniform(*coupon_rate_range)

        price = expected_phoenix_price(
            S0,
            r,
            sigma,
            T,
            autocall_barrier,
            coupon_barrier,
            coupon_rate,
            knock_in_barrier,
            n_paths=n_paths,
            n_steps=int(252 * T),
            principal=1.0,
            seed=seed + i,
        )

        X.append(
            [
                S0,
                r,
                sigma,
                T,
                autocall_barrier,
                coupon_barrier,
                coupon_rate,
                knock_in_barrier,
            ]
        )
        y.append(price)

        if (i + 1) % 10 == 0:
            print(
                f"[{i+1}/{n_samples}] r={r:.3f}, σ={sigma:.2f}, T={T:.2f} → Price={price:.5f}"
            )

    X, y = np.array(X), np.array(y)
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    np.savez(out_file, X=X, y=y)
    print(f"\nSaved dataset to {out_file}")
    print(f"Shape: X={X.shape}, y={y.shape}")
    return X, y


if __name__ == "__main__":
    generate_phoenix_dataset()
