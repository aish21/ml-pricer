import numpy as np
import matplotlib.pyplot as plt
import time
from src.simulation.black_scholes import price_european_option_bs


def mc_with_confidence_interval(
    S0, K, r, sigma, T, num_steps, num_paths, option_type="call", seed=None
):
    """
    Prices an option using Monte Carlo and spits out a 95% confidence interval.

    Basically:
    1. Simulates a bunch of price paths using gbm
    2. Calculates the payoff for each path (call or put).
    3. Discounts those payoffs back to today.
    4. Averages them to get the price.
    5. Calculates the standard error to build the confidence interval.
    """
    if seed:
        np.random.seed(seed)

    dt = T / num_steps

    # Generate all random shocks at once
    z = np.random.randn(num_paths, num_steps)
    paths = np.zeros((num_paths, num_steps + 1))
    paths[:, 0] = S0

    # Vectorized path generation
    for t in range(1, num_steps + 1):
        paths[:, t] = paths[:, t - 1] * np.exp(
            (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z[:, t - 1]
        )

    # Get final prices
    S_T = paths[:, -1]

    if option_type == "call":
        payoffs = np.maximum(S_T - K, 0.0)
    else:
        payoffs = np.maximum(K - S_T, 0.0)

    # Discount and find stats
    disc_payoffs = np.exp(-r * T) * payoffs

    price = np.mean(disc_payoffs)
    se = np.std(disc_payoffs, ddof=1) / np.sqrt(num_paths)  # standard error

    # 95% CI
    ci_lower = price - 1.96 * se
    ci_upper = price + 1.96 * se

    return price, ci_lower, ci_upper


if __name__ == "__main__":
    S0 = 100
    K = 100
    r = 0.05
    sigma = 0.2
    T = 1.0
    num_steps = 252  # daily steps
    option_type = "call"

    # Try a bunch of different path counts
    path_nums = [500, 1000, 5000, 10000, 20000, 50000, 100000, 200000]

    # Get the "true" price from Black-Scholes
    bs_price = price_european_option_bs(S0, K, r, sigma, T, option_type)

    prices = []
    lows = []
    highs = []
    runtimes = []
    errors = []

    print(f"Running MC simulations for {option_type} option...")
    print(f"Black-Scholes Price: {bs_price:.4f}")
    print("-" * 60)

    for n in path_nums:
        t0 = time.time()
        price, ci_l, ci_h = mc_with_confidence_interval(
            S0, K, r, sigma, T, num_steps, n, option_type, seed=42
        )
        t1 = time.time()

        prices.append(price)
        lows.append(ci_l)
        highs.append(ci_h)
        runtimes.append(t1 - t0)
        errors.append(abs(price - bs_price))

        print(
            f"{n:>8d} paths: Price={price:.4f}, 95% CI=({ci_l:.4f}, {ci_h:.4f}), "
            f"Time={t1-t0:.4f}s, Err={abs(price-bs_price):.4f}"
        )

    # 1. Convergence Plot
    plt.figure(figsize=(9, 6))
    plt.axhline(bs_price, color="red", linestyle="--", label="Black-Scholes Price")

    # Calculate error bars
    y_err = [np.array(prices) - np.array(lows), np.array(highs) - np.array(prices)]

    plt.errorbar(
        path_nums,
        prices,
        yerr=y_err,
        fmt="o-",  # added lines
        capsize=5,
        label="MC Estimate ± 95% CI",
        markersize=4,
    )

    plt.xscale("log")
    plt.xlabel("Number of Paths (log scale)")
    plt.ylabel("Option Price")
    plt.title(f"MC Convergence Plot ({option_type.capitalize()})")
    plt.legend()
    plt.grid(True, which="major", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()

    # 2. Runtime vs Error Plot
    plt.figure(figsize=(9, 6))
    plt.loglog(runtimes, errors, "bx-")  # different marker/line
    plt.xlabel("Runtime (s) - log scale")
    plt.ylabel("Absolute Error - log scale")
    plt.title(f"MC Accuracy vs. Runtime ({option_type.capitalize()})")
    plt.grid(True, which="both", linestyle=":", alpha=0.7)
    plt.tight_layout()
    plt.show()
