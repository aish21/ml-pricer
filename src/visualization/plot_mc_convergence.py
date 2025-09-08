import numpy as np
import matplotlib.pyplot as plt
import time
from src.simulation.black_scholes import price_european_option_bs


def mc_with_confidence_interval(
    s0, K, r, sigma, T, n_steps, n_paths, option_type="call", seed=None
):
    if seed is not None:
        np.random.seed(seed)

    dt = T / n_steps
    Z = np.random.randn(n_paths, n_steps)
    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = s0
    for t in range(1, n_steps + 1):
        paths[:, t] = paths[:, t - 1] * np.exp(
            (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[:, t - 1]
        )

    if option_type == "call":
        payoffs = np.maximum(paths[:, -1] - K, 0.0)
    else:
        payoffs = np.maximum(K - paths[:, -1], 0.0)

    discounted = np.exp(-r * T) * payoffs
    mean = np.mean(discounted)
    std_err = np.std(discounted, ddof=1) / np.sqrt(n_paths)
    ci_low = mean - 1.96 * std_err
    ci_high = mean + 1.96 * std_err
    return mean, ci_low, ci_high


if __name__ == "__main__":
    s0 = 100
    K = 100
    r = 0.05
    sigma = 0.2
    T = 1.0
    n_steps = 252
    option_type = "call"

    path_counts = [500, 1_000, 5_000, 10_000, 20_000, 50_000, 100_000, 200_000]

    bs_price = price_european_option_bs(s0, K, r, sigma, T, option_type)
    mc_means, ci_lows, ci_highs, runtimes, errors = [], [], [], [], []

    for n in path_counts:
        start = time.time()
        mean, low, high = mc_with_confidence_interval(
            s0, K, r, sigma, T, n_steps, n, option_type, seed=42
        )
        end = time.time()
        mc_means.append(mean)
        ci_lows.append(low)
        ci_highs.append(high)
        runtimes.append(end - start)
        errors.append(abs(mean - bs_price))

        print(
            f"{n:>7d} paths: MC={mean:.4f}, 95% CI=({low:.4f}, {high:.4f}), "
            f"time={end-start:.4f}s, error={abs(mean-bs_price):.4f}"
        )

    # Plot MC convergence
    plt.figure(figsize=(8, 5))
    plt.axhline(bs_price, color="red", linestyle="--", label="Black-Scholes Price")
    plt.errorbar(
        path_counts,
        mc_means,
        yerr=[
            np.array(mc_means) - np.array(ci_lows),
            np.array(ci_highs) - np.array(mc_means),
        ],
        fmt="o",
        capsize=5,
        label="MC Estimate ± 95% CI",
    )
    plt.xscale("log")
    plt.xlabel("Number of Paths (log scale)")
    plt.ylabel("Option Price")
    plt.title(f"MC Convergence ({option_type.capitalize()})")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()

    # Plot runtime vs error
    plt.figure(figsize=(8, 5))
    plt.loglog(runtimes, errors, marker="o")
    plt.xlabel("Runtime (s)")
    plt.ylabel("Absolute Error vs BS Price")
    plt.title(f"MC Accuracy vs Runtime ({option_type.capitalize()})")
    plt.grid(True, which="both", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()
