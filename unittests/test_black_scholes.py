import pytest
import numpy as np
from src.simulation.black_scholes import (
    price_european_option_mc,
    price_european_option_bs,
)


def test_black_scholes_call_put_parity():
    """
    Test put-call parity: C - P ≈ S0 - K e^{-rT}
    """
    s0, K, r, sigma, T = 100, 100, 0.05, 0.2, 1.0
    call = price_european_option_bs(s0, K, r, sigma, T, "call")
    put = price_european_option_bs(s0, K, r, sigma, T, "put")
    lhs = call - put
    rhs = s0 - K * np.exp(-r * T)
    assert np.isclose(lhs, rhs, atol=1e-8)


def test_mc_vs_bs_call():
    """
    Monte Carlo call price should converge to Black-Scholes.
    """
    s0, K, r, sigma, T = 100, 100, 0.05, 0.2, 1.0
    mc = price_european_option_mc(s0, K, r, sigma, T, 252, 50_000, "call", seed=42)
    bs = price_european_option_bs(s0, K, r, sigma, T, "call")
    assert np.isclose(mc, bs, atol=0.15)  # within a few cents


def test_mc_vs_bs_put():
    """
    Monte Carlo put price should converge to Black-Scholes.
    """
    s0, K, r, sigma, T = 100, 100, 0.05, 0.2, 1.0
    mc = price_european_option_mc(s0, K, r, sigma, T, 252, 50_000, "put", seed=42)
    bs = price_european_option_bs(s0, K, r, sigma, T, "put")
    assert np.isclose(mc, bs, atol=0.15)
