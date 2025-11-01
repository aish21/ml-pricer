# app/frontend.py
import streamlit as st
import pandas as pd
import plotly.express as px
import requests
import json
import math
import os
from typing import Any, Dict, Optional
from io import StringIO
from datetime import datetime

# API endpoint (container-friendly default)
API_URL = os.getenv("API_URL", "http://backend:8000")

# Where frontend will store a local copy of history if available (mounted volume recommended)
LOCAL_HISTORY_PATH = os.getenv(
    "FRONTEND_HISTORY_PATH", "/srv/app/data/pricing_history.csv"
)

# Plotly chart configuration (streamlit's `config=` param)
plotly_config = {
    "displayModeBar": True,
    "scrollZoom": False,
    "editable": False,
}

# Page config and theme hint
st.set_page_config(
    page_title="ML Pricer",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("ML Pricer")
st.markdown(
    "Compare ML model predictions vs Monte Carlo baseline. Configure a test case on the left and press **Run Pricing**."
)


# ---- Helper utilities ----
def find_per_npaths(obj: Any) -> Optional[Dict]:
    """Find a dict that represents per_npaths mapping."""
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


def ensure_history_dir(path: str):
    p = os.path.abspath(path)
    d = os.path.dirname(p)
    os.makedirs(d, exist_ok=True)


def load_history(path: str) -> pd.DataFrame:
    if os.path.exists(path):
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def append_history(path: str, row: dict):
    ensure_history_dir(path)
    df = load_history(path)
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(path, index=False)


# ---- Sidebar controls ----
payoff_type = st.sidebar.selectbox(
    "Payoff Type", ["Phoenix", "Accumulator", "Barrier", "Decumulator"]
)
st.sidebar.markdown("### Advanced settings")
n_paths = st.sidebar.selectbox("Monte Carlo Paths", [500, 2000, 8000], index=1)
model_name = st.sidebar.selectbox("Model", ["LightGBM (default)"], index=0)
learn_mode = st.sidebar.checkbox("Show payoff explanation", value=True)
save_history = st.sidebar.checkbox("Save run to server history", value=True)

st.markdown(f"### Selected payoff: **{payoff_type}**")

# ---- Parameter inputs (kept same) ----
col1, col2 = st.columns(2)

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

# Run button (shows spinner)
run_clicked = st.button("Run Pricing")

# Prepare containers for charts/tables (helps avoid stacking)
container_top = st.container()
container_charts = st.container()
container_feature = st.container()
container_json = st.container()

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
            res.raise_for_status()
        except Exception as e:
            st.error(f"Failed to contact backend at {API_URL}/price/: {e}")
            st.stop()

        try:
            result_raw = res.json()
        except Exception:
            st.error("Backend did not return JSON. See raw response below.")
            st.text(res.text)
            st.stop()

    # Normalize response structure
    if (
        isinstance(result_raw, dict)
        and result_raw.get("status") == "success"
        and "result" in result_raw
    ):
        result = result_raw["result"]
        meta_status = "success"
    else:
        result = result_raw
        meta_status = (
            result_raw.get("status") if isinstance(result_raw, dict) else "unknown"
        )

    # Display debug expander
    with st.expander("Debug: full backend response (collapsed)"):
        st.json(result_raw)

    per_npaths = find_per_npaths(result)
    if per_npaths is None:
        st.error(
            "No pricing comparison data found in response. Expand Debug to inspect the raw response."
        )
        st.stop()

    npaths_key = (
        str(n_paths) if str(n_paths) in per_npaths else next(iter(per_npaths.keys()))
    )
    entry = per_npaths.get(npaths_key, {})
    if not entry:
        st.error(
            f"No entry found for n_paths={npaths_key}. Available keys: {list(per_npaths.keys())}"
        )
        st.stop()

    # Try to robustly extract MC and Model entries
    mc_entry = safe_get(entry, "MC", "Monte Carlo", "mc")
    model_entry = safe_get(entry, "Model", "model", "Model")
    if mc_entry is None:
        for k, v in entry.items():
            if k.lower().startswith("m") and isinstance(v, dict) and "price" in v:
                mc_entry = v
                break
    if model_entry is None:
        for k, v in entry.items():
            if (
                k not in ("MC", "Monte Carlo", "mc")
                and isinstance(v, dict)
                and "price" in v
            ):
                model_entry = v
                break

    if mc_entry is None or model_entry is None:
        st.warning(
            "MC or Model entry missing — showing raw per_npaths entry for debugging."
        )
        st.json(entry)
        st.stop()

    mc_price = as_float(safe_get(mc_entry, "price", "mean", "value"), default=math.nan)
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
        default=(mc_time / model_time if (model_time and model_time > 0) else math.nan),
    )

    # Summarize a row for history
    now = datetime.utcnow().isoformat()
    history_row = {
        "timestamp_utc": now,
        "payoff_type": payoff_type,
        "n_paths": int(npaths_key) if npaths_key.isdigit() else npaths_key,
        "model_price": model_price,
        "mc_price": mc_price,
        "abs_error": abs_error,
        "rel_error_pct": (rel_error * 100.0) if not math.isnan(rel_error) else math.nan,
        "model_time_s": model_time,
        "mc_time_s": mc_time,
    }

    # Save history server-side (backend) if available and requested
    if save_history:
        try:
            hres = requests.post(
                f"{API_URL}/history/append", json=history_row, timeout=10
            )
            if not (hres.status_code == 200 or hres.status_code == 201):
                # fallback to local save if server rejected
                append_history(LOCAL_HISTORY_PATH, history_row)
        except Exception:
            append_history(LOCAL_HISTORY_PATH, history_row)
    else:
        # still keep a local record for session usage (optional)
        append_history(LOCAL_HISTORY_PATH, history_row)

    # Show success / toast if available
    try:
        # some streamlit versions have st.toast
        if hasattr(st, "toast"):
            st.toast("Pricing run complete", icon="✅")
        else:
            st.success("Pricing run complete")
    except Exception:
        st.success("Pricing run complete")

    # ----- Tabs: Dashboard, Diagnostics/Feature, Raw JSON, History -----
    tab_dashboard, tab_feature, tab_json, tab_history = st.tabs(
        ["Dashboard", "Feature Analysis / Explanation", "Raw JSON", "History"]
    )

    with tab_dashboard:
        a, b, c = st.columns([1, 1, 1])
        a.metric("Model price", f"{model_price:.6f}")
        b.metric("Monte Carlo", f"{mc_price:.6f}")
        c.metric("Speedup (x)", f"{speedup:.2f}")

        price_df = pd.DataFrame(
            {"source": ["Model", "Monte Carlo"], "price": [model_price, mc_price]}
        )

        price_fig = px.bar(
            price_df,
            x="source",
            y="price",
            text="price",
            title=f"Model vs Monte Carlo (n_paths={history_row['n_paths']})",
        )
        price_fig.update_traces(texttemplate="%{text:.6f}", textposition="outside")
        price_fig.update_layout(margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(
            price_fig, use_container_width=True, config=plotly_config, height=340
        )

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
            err_df, x="metric", y="value", text="value", title="Error metrics"
        )
        err_fig.update_traces(texttemplate="%{text:.4f}", textposition="outside")
        err_fig.update_layout(margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(
            err_fig, use_container_width=True, config=plotly_config, height=320
        )

        time_df = pd.DataFrame(
            {
                "component": ["Model Time (s)", "MC Time (s)"],
                "time": [model_time, mc_time],
            }
        )
        timing_fig = px.bar(
            time_df, x="component", y="time", text="time", title="Timing (seconds)"
        )
        timing_fig.update_traces(texttemplate="%{text:.6f}", textposition="outside")
        timing_fig.update_layout(margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(
            timing_fig, use_container_width=True, config=plotly_config, height=320
        )

    with tab_feature:
        # Feature importance: check response training blob or call backend training endpoint
        fi = None
        training_part = (
            result.get("training")
            or result.get("train_info")
            or result.get("train")
            or {}
        )
        if training_part:
            fi = training_part.get("feature_importance") or training_part.get(
                "feature_importances"
            )

        if fi is None:
            try:
                r2 = requests.get(
                    f"{API_URL}/training/{payoff_type.lower()}", timeout=5
                )
                if r2.status_code == 200:
                    training_blob = r2.json().get("training", {})
                    fi = training_blob.get("feature_importance") or training_blob.get(
                        "feature_importances"
                    )
            except Exception:
                fi = None

        if fi and isinstance(fi, (list, dict)):
            if isinstance(fi, dict):
                fi_df = pd.DataFrame(
                    list(fi.items()), columns=["feature", "importance"]
                )
            else:
                fi_df = pd.DataFrame(fi)
            if "feature" in fi_df.columns and "importance" in fi_df.columns:
                fi_df = fi_df.sort_values("importance", ascending=True)
                fig_fi = px.bar(
                    fi_df,
                    x="importance",
                    y="feature",
                    orientation="h",
                    color="importance",
                    title="Feature importance",
                )
                fig_fi.update_layout(margin=dict(l=40, r=20, t=30, b=20))
                st.plotly_chart(
                    fig_fi, use_container_width=True, config=plotly_config, height=420
                )
                # separate the plot from the importance table to avoid layout merging
                st.markdown("---")
                st.dataframe(
                    fi_df.sort_values("importance", ascending=False).reset_index(
                        drop=True
                    )
                )
            else:
                st.info(
                    "Feature importance returned in unexpected format. Check Raw JSON."
                )
                st.write(fi_df.head())
        else:
            st.info(
                "No feature importance found in response. The frontend automatically queries /training/<payoff_type> as a fallback."
            )
            st.write(
                "Expected file (on backend machine): `final/results/<payoff_type>/results.json`"
            )

        # Payoff explanation (short)
        if learn_mode:
            try:
                r = requests.get(
                    f"{API_URL}/payoff_explanation/{payoff_type.lower()}", timeout=5
                )
                if r.status_code == 200:
                    expl = r.json().get("explanation", {})
                    st.subheader(expl.get("title", ""))
                    st.write(expl.get("summary", ""))
                    if expl.get("latex"):
                        try:
                            st.latex(expl["latex"])
                        except Exception:
                            st.text(expl["latex"])
                    if expl.get("notes"):
                        st.markdown("**Notes:**")
                        for n in expl["notes"]:
                            st.write("-", n)
                else:
                    st.write("No explanation available from backend (status != 200).")
            except Exception:
                st.write("Explanation endpoint unavailable.")

    with tab_json:
        st.json(result)

    with tab_history:
        # Try server history first, fallback to local CSV
        hist_df = pd.DataFrame()
        try:
            r = requests.get(f"{API_URL}/history", timeout=5)
            if r.status_code == 200:
                hist_df = pd.DataFrame(r.json().get("history", []))
        except Exception:
            hist_df = load_history(LOCAL_HISTORY_PATH)

        if hist_df.empty:
            st.info(
                "No history found. Runs are appended either to the backend history file (if enabled) or to a local CSV."
            )
        else:
            # Normalize columns
            st.dataframe(
                hist_df.sort_values("timestamp_utc", ascending=False).reset_index(
                    drop=True
                )
            )

            csv_buf = StringIO()
            hist_df.to_csv(csv_buf, index=False)
            csv_bytes = csv_buf.getvalue().encode("utf-8")
            st.download_button(
                "Download history CSV",
                data=csv_bytes,
                file_name="pricing_history.csv",
                mime="text/csv",
            )

# End run_clicked handling

# If not run yet: provide quick note & allow viewing history anyway
if not run_clicked:
    st.info(
        "Configure parameters and click 'Run Pricing' to compare ML model vs Monte Carlo. You can also view historical runs if any exist in History tab after running once."
    )
