import pytest
import numpy as np
from src.simulation.black_scholes import (
    price_european_option_mc as price_mc,
    price_european_option_bs as price_bs,
)


def test_black_scholes_call_put_parity():
    s0, K, r, sigma, T = 100, 100, 0.05, 0.2, 1.0
    call_price = price_bs(s0, K, r, sigma, T, "call")
    put_price = price_bs(s0, K, r, sigma, T, "put")
    lhs = call_price - put_price
    rhs = s0 - K * np.exp(-r * T)
    assert abs(lhs - rhs) < 1e-8


def test_mc_vs_bs_call():
    s0, K, r, sigma, T = 100, 100, 0.05, 0.2, 1.0
    mc_price = price_mc(s0, K, r, sigma, T, 252, 50000, "call", seed=42)
    bs_price = price_bs(s0, K, r, sigma, T, "call")
    assert abs(mc_price - bs_price) < 0.15  # should be pretty close


def test_mc_vs_bs_put():
    s0, K, r, sigma, T = 100, 100, 0.05, 0.2, 1.0
    mc_val = price_mc(s0, K, r, sigma, T, 252, 50000, "put", seed=42)
    bs_val = price_bs(s0, K, r, sigma, T, "put")

    assert abs(mc_val - bs_val) < 0.15
