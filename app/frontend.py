import streamlit as st
import requests
import json

API_URL = "http://localhost:8000/price/"

st.set_page_config(page_title="AI Derivative Pricer", page_icon="💹", layout="wide")

st.title("💹 AI-Powered Derivative Pricer")

payoff_type = st.selectbox(
    "Select a payoff type",
    ["phoenix", "accumulator", "barrier", "decumulator"],
)

st.subheader("Enter parameters")


def numeric_input(label, value):
    return st.number_input(label, value=value, format="%.4f")


params = {}
if payoff_type == "phoenix":
    params = {
        "S0": numeric_input("S0", 100.0),
        "r": numeric_input("Risk-free rate (r)", 0.03),
        "sigma": numeric_input("Volatility (σ)", 0.2),
        "T": numeric_input("Maturity (T)", 1.0),
        "autocall_barrier_frac": numeric_input("Autocall barrier", 1.05),
        "coupon_barrier_frac": numeric_input("Coupon barrier", 1.0),
        "coupon_rate": numeric_input("Coupon rate", 0.02),
        "knock_in_frac": numeric_input("Knock-in barrier", 0.7),
        "obs_count": numeric_input("Observation count", 6),
    }
elif payoff_type == "accumulator":
    params = {
        "S0": numeric_input("S0", 100.0),
        "r": numeric_input("r", 0.03),
        "sigma": numeric_input("σ", 0.2),
        "T": numeric_input("T", 1.0),
        "upper_barrier_frac": numeric_input("Upper barrier", 1.05),
        "lower_barrier_frac": numeric_input("Lower barrier", 0.95),
        "participation_rate": numeric_input("Participation rate", 2.0),
        "obs_frequency": numeric_input("Observation frequency", 0.25),
    }
elif payoff_type == "barrier":
    params = {
        "S0": numeric_input("S0", 100.0),
        "r": numeric_input("r", 0.03),
        "sigma": numeric_input("σ", 0.2),
        "T": numeric_input("T", 1.0),
        "K": numeric_input("Strike (K)", 100.0),
        "barrier_frac": numeric_input("Barrier fraction", 0.8),
        "option_type": numeric_input("Option type (1=call, 0=put)", 1.0),
    }
elif payoff_type == "decumulator":
    params = {
        "S0": numeric_input("S0", 100.0),
        "r": numeric_input("r", 0.03),
        "sigma": numeric_input("σ", 0.2),
        "T": numeric_input("T", 1.0),
        "upper_barrier_frac": numeric_input("Upper barrier", 1.05),
        "lower_barrier_frac": numeric_input("Lower barrier", 0.95),
        "participation_rate": numeric_input("Participation rate", 2.0),
        "obs_frequency": numeric_input("Observation frequency", 0.25),
    }

n_paths = st.slider("Monte Carlo benchmark paths", 100, 10000, 2000, step=100)

if st.button("Run Pricing"):
    with st.spinner("Running model inference and Monte Carlo comparison..."):
        try:
            response = requests.post(
                API_URL,
                json={"payoff_type": payoff_type, "params": params, "n_paths": n_paths},
                timeout=120,
            )
            data = response.json()

            if data.get("status") == "success":
                st.success("✅ Pricing completed successfully")
                st.json(data["result"])
            else:
                st.error(f"❌ Error: {data.get('message')}")
                st.code(data.get("trace"))
        except Exception as e:
            st.error(f"Request failed: {e}")
