import numpy as np
from scipy.stats import norm


def simulate_gbm_paths(
    S0: float,
    r: float,
    sigma: float,
    T: float,
    n_steps: int,
    n_paths: int,
    seed: int = None,  # type: ignore
) -> np.ndarray:
    """
    Simulate GBM paths under the Black-Scholes model.

    Parameters:
        S0 : float
            Initial stock price.
        r : float
            Risk-free interest rate.
        sigma : float
            Volatility.
        T : float
            Time to maturity (in years).
        n_steps : int
            Number of discrete timesteps.
        n_paths : int
            Number of Monte Carlo paths.
        seed : int, optional
            Random seed for reproducibility.

    Returns:
        numpy.ndarray of shape (n_paths, n_steps+1)
            Simulated stock price paths
    """
    if seed is not None:
        np.random.seed(seed)

    dt = T / n_steps
    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = S0

    Z = np.random.normal(size=(n_paths, n_steps))

    for t in range(1, n_steps + 1):
        paths[:, t] = paths[:, t - 1] * np.exp(
            (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[:, t - 1]
        )

    return paths


def price_european_option_mc(
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    n_steps: int,
    n_paths: int,
    option_type: str = "call",
    seed: int = None,  # type: ignore
) -> float:
    """
    Price an European option using Monte Carlo simulation under the Black-Scholes model.

    Parameters:
        S0 : float
            Initial stock price.
        K : float
            Strike price.
        r : float
            Risk-free interest rate.
        sigma : float
            Volatility.
        T : float
            Time to maturity (in years).
        n_steps : int
            Number of discrete timesteps.
        n_paths : int
            Number of Monte Carlo paths.
        option_type : str, optional
            Type of the option ('call' or 'put'). Default is 'call'.
        seed : int, optional
            Random seed for reproducibility.

    Returns:
        float
            Estimated option price.
    """
    paths = simulate_gbm_paths(S0, r, sigma, T, n_steps, n_paths, seed)
    S_T = paths[:, -1]  # Stock prices at maturity

    if option_type.lower() == "call":
        payoffs = np.maximum(S_T - K, 0)
    elif option_type.lower() == "put":
        payoffs = np.maximum(K - S_T, 0)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    option_price = np.exp(-r * T) * np.mean(payoffs)
    return option_price


def price_european_option_bs(
    s0: float, K: float, r: float, sigma: float, T: float, option_type: str = "call"
) -> float:
    """
    Closed-form Black-Scholes formula for European call/put.

    Parameters
        s0 : float
            Initial stock price.
        K : float
            Strike price.
        r : float
            Risk-free rate.
        sigma : float
            Volatility.
        T : float
            Time to maturity (in years).
        option_type : str
            'call' or 'put'.

    Returns
        float
            Black-Scholes price.
    """
    if T <= 0:
        return max(0.0, (s0 - K) if option_type == "call" else (K - s0))

    d1 = (np.log(s0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type.lower() == "call":
        return s0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type.lower() == "put":
        return K * np.exp(-r * T) * norm.cdf(-d2) - s0 * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")
