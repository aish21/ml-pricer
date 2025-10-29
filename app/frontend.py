# app/frontend.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import requests
import json
import math
from typing import Any, Dict, Optional

# ----------------------------
# CONFIG
# ----------------------------
API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="AI Derivative Pricer",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("AI Derivative Pricer")
st.markdown(
    "Compare surrogate model predictions vs Monte Carlo. Use the controls on the left to configure a test case."
)


# ----------------------------
# Helper utilities
# ----------------------------
def find_per_npaths(obj: Any) -> Optional[Dict]:
    """Recursively find the first dict that contains 'per_npaths' or looks like a npaths dict."""
    if obj is None:
        return None

    if isinstance(obj, dict):
        if (
            "per_npaths" in obj
            and isinstance(obj["per_npaths"], dict)
            and obj["per_npaths"]
        ):
            return obj["per_npaths"]
        if all(isinstance(k, str) and k.isdigit() for k in obj.keys()):
            sample_key = next(iter(obj.keys()))
            if isinstance(obj[sample_key], dict):
                return obj
        for v in obj.values():
            found = find_per_npaths(v)
            if found:
                return found

    elif isinstance(obj, list):
        for item in obj:
            found = find_per_npaths(item)
            if found:
                return found
    return None


def safe_get(d: dict, *keys, default=None):
    for k in keys:
        if isinstance(d, dict) and k in d and d[k] is not None:
            return d[k]
    return default


def as_float(x, default=float("nan")):
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


# ----------------------------
# PAYOFF TYPE SELECTION + INPUTS (number inputs, not sliders)
# ----------------------------
payoff_type = st.sidebar.selectbox(
    "Payoff Type", ["Phoenix", "Accumulator", "Barrier", "Decumulator"]
)
st.sidebar.markdown("### Advanced settings")
n_paths = st.sidebar.selectbox("Monte Carlo Paths", [500, 2000, 8000], index=1)
model_name = st.sidebar.selectbox("Model", ["LightGBM (default)"], index=0)

st.markdown(f"### Selected payoff: **{payoff_type}**")

col1, col2 = st.columns(2)

# Use numeric inputs (number_input) so users can key values precisely
if payoff_type == "Phoenix":
    with col1:
        S0 = st.number_input(
            "Initial Spot (S₀)", value=100.0, min_value=0.0, step=0.1, format="%.4f"
        )
        r = st.number_input(
            "Interest Rate (r)", value=0.03, min_value=0.0, step=0.0001, format="%.6f"
        )
        sigma = st.number_input(
            "Volatility (σ)", value=0.2, min_value=0.0, step=0.001, format="%.6f"
        )
        T = st.number_input(
            "Tenor (T)", value=1.0, min_value=0.0, step=0.1, format="%.4f"
        )
    with col2:
        autocall = st.number_input(
            "Autocall Barrier (×S₀)",
            value=1.05,
            min_value=0.0,
            step=0.01,
            format="%.4f",
        )
        coupon_b = st.number_input(
            "Coupon Barrier (×S₀)", value=1.0, min_value=0.0, step=0.01, format="%.4f"
        )
        coupon_rate = st.number_input(
            "Coupon Rate", value=0.02, min_value=0.0, step=0.001, format="%.6f"
        )
        knockin = st.number_input(
            "Knock-In Barrier (×S₀)", value=0.7, min_value=0.0, step=0.01, format="%.4f"
        )
        obs = st.number_input(
            "Observation Count", value=6, min_value=1, step=1, format="%d"
        )
    params = {
        "S0": S0,
        "r": r,
        "sigma": sigma,
        "T": T,
        "autocall_barrier_frac": autocall,
        "coupon_barrier_frac": coupon_b,
        "coupon_rate": coupon_rate,
        "knock_in_frac": knockin,
        "obs_count": int(obs),
    }

elif payoff_type in ("Accumulator", "Decumulator"):
    with col1:
        S0 = st.number_input(
            "Initial Spot (S₀)", value=100.0, min_value=0.0, step=0.1, format="%.4f"
        )
        r = st.number_input(
            "Interest Rate (r)", value=0.03, min_value=0.0, step=0.0001, format="%.6f"
        )
        sigma = st.number_input(
            "Volatility (σ)", value=0.2, min_value=0.0, step=0.001, format="%.6f"
        )
        T = st.number_input(
            "Tenor (T)", value=1.0, min_value=0.0, step=0.1, format="%.4f"
        )
    with col2:
        upper_b = st.number_input(
            "Upper Barrier (×S₀)", value=1.05, min_value=0.0, step=0.01, format="%.4f"
        )
        lower_b = st.number_input(
            "Lower Barrier (×S₀)", value=0.95, min_value=0.0, step=0.01, format="%.4f"
        )
        participation = st.number_input(
            "Participation Rate", value=2.0, min_value=0.0, step=0.1, format="%.4f"
        )
        obs_freq = st.number_input(
            "Observation Frequency (Years)",
            value=0.25,
            min_value=0.0,
            step=0.01,
            format="%.4f",
        )
    params = {
        "S0": S0,
        "r": r,
        "sigma": sigma,
        "T": T,
        "upper_barrier_frac": upper_b,
        "lower_barrier_frac": lower_b,
        "participation_rate": participation,
        "obs_frequency": obs_freq,
    }

else:  # Barrier
    with col1:
        S0 = st.number_input(
            "Initial Spot (S₀)", value=100.0, min_value=0.0, step=0.1, format="%.4f"
        )
        r = st.number_input(
            "Interest Rate (r)", value=0.03, min_value=0.0, step=0.0001, format="%.6f"
        )
        sigma = st.number_input(
            "Volatility (σ)", value=0.2, min_value=0.0, step=0.001, format="%.6f"
        )
        T = st.number_input(
            "Tenor (T)", value=1.0, min_value=0.0, step=0.1, format="%.4f"
        )
    with col2:
        K = st.number_input(
            "Strike (K)", value=100.0, min_value=0.0, step=0.1, format="%.4f"
        )
        barrier_frac = st.number_input(
            "Barrier (×S₀)", value=0.8, min_value=0.0, step=0.01, format="%.4f"
        )
        option_type = st.selectbox("Option Type", ["Call", "Put"])
        option_type_val = 1.0 if option_type == "Call" else 0.0
    params = {
        "S0": S0,
        "r": r,
        "sigma": sigma,
        "T": T,
        "K": K,
        "barrier_frac": barrier_frac,
        "option_type": option_type_val,
    }

st.markdown("---")

# ----------------------------
# Run button
# ----------------------------
run_clicked = st.button("Run Pricing")

if run_clicked:
    payload = {
        "payoff_type": payoff_type,
        "params": params,
        "n_paths": n_paths,
        "use_log_target": True,
    }
    with st.spinner("Running model and Monte Carlo benchmark..."):
        try:
            res = requests.post(f"{API_URL}/price/", json=payload, timeout=120)
        except Exception as e:
            st.error(f"Failed to contact backend at {API_URL}/price/: {e}")
            st.stop()

        try:
            result = res.json()
        except Exception:
            st.error("Backend did not return JSON. See raw response below.")
            st.text(res.text)
            st.stop()

        with st.expander("Debug: full backend response (collapsed)"):
            st.json(result)

        # robustly find per_npaths
        per_npaths = find_per_npaths(result)

        if per_npaths is None:
            st.error(
                "No pricing comparison data found in response. Expand Debug to inspect the raw response."
            )
            st.stop()

        # pick the npaths of interest
        npaths_key = (
            str(n_paths)
            if str(n_paths) in per_npaths
            else next(iter(per_npaths.keys()))
        )
        entry = per_npaths.get(npaths_key, {})
        if not entry:
            st.error(
                f"No entry found for n_paths={npaths_key}. Available keys: {list(per_npaths.keys())}"
            )
            st.stop()

        mc_entry = safe_get(entry, "MC", "Monte Carlo", "mc")
        model_entry = safe_get(entry, "Model", "model", "Model")

        if mc_entry is None:
            st.warning(
                "MC entry missing in per_npaths — showing raw entry for debugging."
            )
            st.json(entry)
            st.stop()

        if model_entry is None:
            alt_model_keys = [
                k for k in entry.keys() if k not in ("MC", "Monte Carlo", "mc")
            ]
            model_entry = entry[alt_model_keys[0]] if alt_model_keys else None
            if model_entry is None:
                st.warning(
                    "Model entry missing — showing raw per_npaths entry for debugging."
                )
                st.json(entry)
                st.stop()

        # normalize values
        mc_price = as_float(
            safe_get(mc_entry, "price", "mean", "value"), default=math.nan
        )
        mc_time = as_float(
            safe_get(mc_entry, "time", "elapsed", "timing"), default=math.nan
        )
        mc_std = as_float(safe_get(mc_entry, "std", "stddev", "var"), default=math.nan)

        model_price = as_float(
            safe_get(model_entry, "price", "model_price", "value"), default=math.nan
        )
        model_time = as_float(
            safe_get(model_entry, "time", "model_time", "elapsed", "timing"),
            default=math.nan,
        )
        abs_error = as_float(
            safe_get(model_entry, "abs_error", "abs_err", "abs"), default=math.nan
        )
        rel_error = as_float(
            safe_get(model_entry, "rel_error", "rel_err", "rel"), default=math.nan
        )
        speedup = as_float(
            safe_get(model_entry, "speedup", "speed_up"),
            default=(
                mc_time / model_time if (model_time and model_time > 0) else math.nan
            ),
        )

        summary = {
            "n_paths": int(npaths_key) if npaths_key.isdigit() else npaths_key,
            "mc": {"price": mc_price, "time": mc_time, "std": mc_std},
            "model": {
                "price": model_price,
                "time": model_time,
                "abs_error": abs_error,
                "rel_error": rel_error,
                "speedup": speedup,
            },
        }

        # ---- Tabs ----
        tab_dashboard, tab_feature, tab_json = st.tabs(
            ["Dashboard", "Feature Analysis", "Raw JSON"]
        )

        # ---------- Dashboard ----------
        with tab_dashboard:
            a, b, c = st.columns([1, 1, 1])
            a.metric("Model price", f"{model_price:.6f}")
            b.metric("Monte Carlo", f"{mc_price:.6f}")
            c.metric("Speedup (x)", f"{speedup:.2f}")

            # Price comparison bar (use config param to avoid deprecation)
            price_fig = go.Figure()
            price_fig.add_trace(go.Bar(name="Model", x=["Model"], y=[model_price]))
            price_fig.add_trace(
                go.Bar(name="Monte Carlo", x=["Monte Carlo"], y=[mc_price])
            )
            price_fig.update_layout(
                title=f"Model vs Monte Carlo (n_paths={summary['n_paths']})",
                barmode="group",
                template="plotly_white",
            )
            st.plotly_chart(price_fig, config={"responsive": True})

            # Error bar
            err_df = pd.DataFrame(
                {
                    "metric": ["Abs Error", "Rel Error (%)"],
                    "value": [
                        abs_error if not math.isnan(abs_error) else 0.0,
                        (rel_error * 100) if not math.isnan(rel_error) else 0.0,
                    ],
                }
            )
            err_fig = px.bar(
                err_df,
                x="metric",
                y="value",
                text="value",
                title="Error metrics",
                template="plotly_white",
            )
            err_fig.update_traces(texttemplate="%{text:.4f}", textposition="outside")
            st.plotly_chart(err_fig, config={"responsive": True})

            # Timing comparison
            time_df = pd.DataFrame(
                {
                    "component": ["Model Time (s)", "MC Time (s)"],
                    "time": [model_time, mc_time],
                }
            )
            timing_fig = px.bar(
                time_df,
                x="component",
                y="time",
                text="time",
                title="Timing (seconds)",
                template="plotly_white",
            )
            timing_fig.update_traces(texttemplate="%{text:.6f}", textposition="outside")
            st.plotly_chart(timing_fig, config={"responsive": True})

        # ---------- Feature analysis ----------
        with tab_feature:
            # first try to get feature importance from result
            fi = None
            training_part = (
                result.get("training")
                or result.get("train_info")
                or result.get("train")
            )
            if training_part:
                fi = training_part.get("feature_importance") or training_part.get(
                    "feature_importances"
                )
            # if not found, query backend training endpoint for this payoff_type
            if fi is None:
                try:
                    tt = payoff_type.lower()
                    r2 = requests.get(f"{API_URL}/training/{tt}", timeout=10)
                    if r2.status_code == 200:
                        train_json = r2.json()
                        fi = train_json.get("training", {}).get(
                            "feature_importance"
                        ) or train_json.get("training", {}).get("feature_importances")
                except Exception:
                    fi = None

            if fi and isinstance(fi, (list, dict)):
                if isinstance(fi, dict):
                    fi_df = pd.DataFrame(
                        list(fi.items()), columns=["feature", "importance"]
                    )
                else:
                    fi_df = pd.DataFrame(fi)
                if "importance" in fi_df.columns and "feature" in fi_df.columns:
                    fi_df = fi_df.sort_values("importance", ascending=True)
                    fig_fi = px.bar(
                        fi_df,
                        x="importance",
                        y="feature",
                        orientation="h",
                        color="importance",
                        title="Feature importance",
                        template="plotly_white",
                    )
                    st.plotly_chart(fig_fi, config={"responsive": True})
                else:
                    st.info(
                        "Feature importance returned in unexpected format. Check Raw JSON."
                    )
                    st.write(fi_df.head())
            else:
                st.info(
                    "No feature importance found in response. If you trained the model previously, make sure results.json exists in the model folder. You can also open Raw JSON to debug."
                )

        # ---------- Raw json ----------
        with tab_json:
            st.json(result)
