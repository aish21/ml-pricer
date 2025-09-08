from src.simulation.black_scholes import (
    price_european_option_mc,
    price_european_option_bs,
)

if __name__ == "__main__":
    # Parameters
    s0 = 100
    K = 100
    r = 0.05
    sigma = 0.2
    T = 1.0
    n_steps = 252
    n_paths = 100_000
    option_type = "call"

    mc_price = price_european_option_mc(
        s0, K, r, sigma, T, n_steps, n_paths, option_type, seed=123
    )
    bs_price = price_european_option_bs(s0, K, r, sigma, T, option_type)

    print(f"Monte Carlo price ({n_paths} paths): {mc_price:.4f}")
    print(f"Black-Scholes closed-form:           {bs_price:.4f}")
    print(f"Absolute error:                      {abs(mc_price - bs_price):.4f}")
