import numpy as np
from scipy.stats import norm


def simulate_gbm_paths(s0, r, sigma, T, num_steps, num_paths, seed=None):
    """
    Simulates a bunch of stock paths using gbm

    This is the standard $S(t+dt) = S(t) * exp(...)$ formula, all vectorized.
    We pre-generate all the random shocks at once for speed.
    """
    if seed is not None:
        np.random.seed(seed)

    dt = T / num_steps

    # Set up the array to hold the paths
    paths = np.zeros((num_paths, num_steps + 1))
    paths[:, 0] = s0

    # Generate all the random 'shocks' at once
    z = np.random.normal(size=(num_paths, num_steps))

    # Loop through and apply the GBM formula
    for t in range(1, num_steps + 1):
        paths[:, t] = paths[:, t - 1] * np.exp(
            (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z[:, t - 1]
        )

    return paths


def price_european_option_mc(
    s0, k, r, sigma, T, num_steps, num_paths, option_type="call", seed=None
):
    """
    Prices a European option using a basic Monte Carlo.

    It just does three things:
    1. Simulates a ton of paths using `simulate_gbm_paths`.
    2. Figures out the payoff for each path (e.g., max(S_T - K, 0)).
    3. Averages all those payoffs and discounts them back to today.
    """
    paths = simulate_gbm_paths(s0, r, sigma, T, num_steps, num_paths, seed)

    # Get the final price from all paths
    final_prices = paths[:, -1]

    # Calculate payoffs
    if option_type.lower() == "call":
        payoffs = np.maximum(final_prices - k, 0)
    elif option_type.lower() == "put":
        payoffs = np.maximum(k - final_prices, 0)
    else:
        raise ValueError("Option type must be 'call' or 'put'")

    # Discount the average payoff
    price = np.exp(-r * T) * np.mean(payoffs)
    return price


def price_european_option_bs(s0, k, r, sigma, T, option_type="call"):
    """
    The classic Black-Scholes formula.

    Calculates d1 and d2 and plugs them into the closed-form
    solution for a European call or put.
    """

    # Handle the edge case of 0 time left
    if T <= 1e-8:  # avoid division by zero
        if option_type.lower() == "call":
            return np.maximum(s0 - k, 0.0)
        else:
            return np.maximum(k - s0, 0.0)

    #
    d1 = (np.log(s0 / k) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type.lower() == "call":
        price = s0 * norm.cdf(d1) - k * np.exp(-r * T) * norm.cdf(d2)
    elif option_type.lower() == "put":
        price = k * np.exp(-r * T) * norm.cdf(-d2) - s0 * norm.cdf(-d1)
    else:
        raise ValueError("Option type must be 'call' or 'put'")

    return price
