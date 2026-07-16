import streamlit as st
import pandas as pd
import plotly.express as px
import requests
import math
import os
from typing import Any, Dict, Optional
from io import StringIO
from datetime import datetime, timezone

from app.frontend_support import (
    compact_nonzero_shock,
    frozen_term_structure_from_pricing_result,
)

# API endpoint (container-friendly default)
API_URL = os.getenv("API_URL", "https://aish-ml-pricer-backend.up.railway.app")

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
    "Price Phoenix Single v1 for an equity, ETF, or equity index from a dated market snapshot."
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


def load_history(path: str) -> pd.DataFrame:
    if os.path.exists(path):
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


# ---- Sidebar controls ----
payoff_type = st.sidebar.selectbox("Payoff Type", ["Phoenix"])
st.sidebar.markdown("### Advanced settings")
n_paths = st.sidebar.selectbox("Monte Carlo Paths", [500, 2000, 8000], index=1)
model_name = st.sidebar.selectbox("Method", ["Monte Carlo reference"], index=0)
learn_mode = st.sidebar.checkbox("Show payoff explanation", value=True)
analysis_mode = st.sidebar.selectbox(
    "After pricing",
    ["Price only", "Price + scenario", "Price + risk analytics"],
)
analysis_seed = int(
    st.sidebar.number_input(
        "Analysis random seed", min_value=0, max_value=4_294_967_295, value=42
    )
)

st.markdown(f"### Selected payoff: **{payoff_type}**")

# ---- Parameter inputs (kept same) ----
col1, col2 = st.columns(2)

if payoff_type == "Phoenix":
    market_mode = st.radio(
        "Market source",
        [
            "Research term structure",
            "Manual snapshot",
            "Yahoo latest bar + manual assumptions",
        ],
        horizontal=True,
    )
    use_research_market = market_mode == "Research term structure"
    use_market_quote = market_mode == "Yahoo latest bar + manual assumptions"
    with col1:
        symbol = st.text_input("Underlier symbol", value="SPY", max_chars=64)
        underlier_choices = ["ETF", "Equity"]
        if not use_research_market:
            underlier_choices.append("Index")
        underlier_type = st.selectbox("Underlier type", underlier_choices)
        if use_research_market:
            st.caption(
                "Server-built USD research curve from Treasury data, trailing "
                "distributions, and near-ATM Yahoo option volatility."
            )
        if use_market_quote:
            st.caption(
                "Spot and metadata use yfinance's latest regular-session one-minute bar. "
                "Research/personal use only; data may be delayed."
            )
        elif not use_research_market:
            currency = st.text_input("Currency", value="USD", max_chars=3)
            S0 = st.number_input(
                "Spot", value=100.0, min_value=0.000001, step=0.1, format="%.4f"
            )
        if not use_research_market:
            r = st.number_input(
                "Flat discount rate",
                value=0.03,
                min_value=-0.25,
                max_value=1.0,
                step=0.0001,
                format="%.6f",
            )
            dividend_yield = st.number_input(
                "Flat dividend yield",
                value=0.0,
                min_value=-0.25,
                max_value=1.0,
                step=0.0001,
                format="%.6f",
            )
            sigma = st.number_input(
                "Flat volatility",
                value=0.2,
                min_value=0.000001,
                max_value=5.0,
                step=0.001,
                format="%.6f",
            )
        T = st.number_input(
            "Maturity (years)",
            value=1.0,
            min_value=0.000001,
            step=0.1,
            format="%.4f",
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
        if not use_market_quote and not use_research_market:
            calendar = st.text_input("Calendar", value="XNYS", max_chars=32)
        day_count = st.text_input("Day count", value="ACT/365F", max_chars=16)
    phoenix_terms = {
        "maturity_years": T,
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

scenario_shock = {}
risk_bumps = {}
if analysis_mode == "Price + scenario":
    with st.expander("Scenario shocks", expanded=True):
        shock_col1, shock_col2 = st.columns(2)
        with shock_col1:
            scenario_spot_pct = st.number_input(
                "Spot shock (%)", value=-10.0, min_value=-99.99, max_value=1000.0
            )
            scenario_rate_bps = st.number_input(
                "Parallel rate shock (bp)",
                value=0.0,
                min_value=-10000.0,
                max_value=10000.0,
            )
        with shock_col2:
            scenario_vol_abs = st.number_input(
                "Parallel volatility shock", value=0.0, min_value=-5.0, max_value=5.0
            )
            scenario_dividend_bps = st.number_input(
                "Parallel dividend shock (bp)",
                value=0.0,
                min_value=-10000.0,
                max_value=10000.0,
            )
        use_segment_shock = st.checkbox("Add one segment-specific shock", value=False)
        segment_shock = None
        if use_segment_shock:
            bucket_col1, bucket_col2, bucket_col3, bucket_col4 = st.columns(4)
            with bucket_col1:
                segment_index = int(
                    st.number_input(
                        "Segment index", min_value=0, max_value=251, value=0
                    )
                )
            with bucket_col2:
                segment_rate_bps = st.number_input(
                    "Segment rate (bp)",
                    value=0.0,
                    min_value=-10000.0,
                    max_value=10000.0,
                )
            with bucket_col3:
                segment_dividend_bps = st.number_input(
                    "Segment dividend (bp)",
                    value=0.0,
                    min_value=-10000.0,
                    max_value=10000.0,
                )
            with bucket_col4:
                segment_vol_abs = st.number_input(
                    "Segment volatility", value=0.01, min_value=-5.0, max_value=5.0
                )
            segment_shock = {
                "segment_index": segment_index,
                "rate_bps": segment_rate_bps,
                "dividend_bps": segment_dividend_bps,
                "volatility_abs": segment_vol_abs,
            }
        scenario_shock = compact_nonzero_shock(
            spot_pct=scenario_spot_pct,
            rate_parallel_bps=scenario_rate_bps,
            dividend_parallel_bps=scenario_dividend_bps,
            volatility_parallel_abs=scenario_vol_abs,
            segment_shock=segment_shock,
        )
elif analysis_mode == "Price + risk analytics":
    with st.expander("Finite-difference bump sizes", expanded=False):
        bump_col1, bump_col2 = st.columns(2)
        with bump_col1:
            risk_spot_relative = st.number_input(
                "Spot relative bump", value=0.01, min_value=0.000001, max_value=0.5
            )
            risk_rate_bps = st.number_input(
                "Rate bump (bp)", value=10.0, min_value=0.000001, max_value=5000.0
            )
        with bump_col2:
            risk_volatility_absolute = st.number_input(
                "Volatility bump", value=0.01, min_value=0.000001, max_value=1.0
            )
            risk_dividend_bps = st.number_input(
                "Dividend bump (bp)", value=10.0, min_value=0.000001, max_value=5000.0
            )
        risk_bumps = {
            "spot_relative": risk_spot_relative,
            "volatility_absolute": risk_volatility_absolute,
            "rate_bps": risk_rate_bps,
            "dividend_bps": risk_dividend_bps,
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
    if payoff_type == "Phoenix":
        if use_research_market:
            payload = {
                "market": {
                    "symbol": symbol,
                    "underlier_type": underlier_type.lower(),
                    "currency": "USD",
                },
                "terms": phoenix_terms,
                "n_paths": n_paths,
            }
            pricing_url = f"{API_URL}/api/v1/products/phoenix/price/research-market"
        elif use_market_quote:
            payload = {
                "market": {
                    "symbol": symbol,
                    "underlier_type": underlier_type.lower(),
                    "risk_free_rate": r,
                    "dividend_yield": dividend_yield,
                    "volatility": sigma,
                    "day_count": day_count,
                },
                "terms": phoenix_terms,
                "n_paths": n_paths,
            }
            pricing_url = f"{API_URL}/api/v1/products/phoenix/price/market"
        else:
            snapshot_time = datetime.now(timezone.utc).isoformat()
            payload = {
                "market": {
                    "schema_version": "equity-market-snapshot-v1",
                    "symbol": symbol,
                    "underlier_type": underlier_type.lower(),
                    "currency": currency,
                    "valuation_time": snapshot_time,
                    "market_data_time": snapshot_time,
                    "spot": S0,
                    "risk_free_rate": r,
                    "dividend_yield": dividend_yield,
                    "volatility": sigma,
                    "calendar": calendar,
                    "day_count": day_count,
                    "source": "streamlit-manual",
                },
                "terms": phoenix_terms,
                "n_paths": n_paths,
            }
            pricing_url = f"{API_URL}/api/v1/products/phoenix/price"
    else:
        payload = {
            "payoff_type": payoff_type,
            "params": params,
            "n_paths": n_paths,
        }
        pricing_url = f"{API_URL}/price/"
    with st.spinner("Running deterministic Monte Carlo reference pricing..."):
        try:
            res = requests.post(pricing_url, json=payload, timeout=120)
        except requests.RequestException as e:
            st.error(f"Failed to contact backend at {pricing_url}: {e}")
            st.stop()

        try:
            result_raw = res.json()
        except Exception:
            st.error("Backend did not return JSON. See raw response below.")
            st.text(res.text)
            st.stop()

        if not res.ok:
            message = (
                result_raw.get("message", f"Backend returned HTTP {res.status_code}")
                if isinstance(result_raw, dict)
                else f"Backend returned HTTP {res.status_code}"
            )
            st.error(message)
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

    analysis_result_raw = None
    analysis_result = None
    analysis_error = None
    if analysis_mode != "Price only":
        frozen_market = frozen_term_structure_from_pricing_result(result, T)
        if frozen_market is None:
            analysis_error = "Pricing response did not contain reusable market data."
        else:
            analysis_payload = {
                "market": frozen_market,
                "terms": phoenix_terms,
                "n_paths": n_paths,
                "seed": analysis_seed,
            }
            if analysis_mode == "Price + scenario":
                analysis_payload["shock"] = scenario_shock
                analysis_url = (
                    f"{API_URL}/api/v1/products/phoenix/scenario/term-structure"
                )
                analysis_label = "paired scenario"
            else:
                analysis_payload["bumps"] = risk_bumps
                analysis_url = f"{API_URL}/api/v1/products/phoenix/risk/term-structure"
                analysis_label = "risk analytics"
            with st.spinner(f"Running {analysis_label} on the frozen market..."):
                try:
                    analysis_response = requests.post(
                        analysis_url, json=analysis_payload, timeout=180
                    )
                    analysis_result_raw = analysis_response.json()
                    if analysis_response.ok:
                        analysis_result = analysis_result_raw.get("result")
                    else:
                        analysis_error = analysis_result_raw.get(
                            "message",
                            f"Analysis returned HTTP {analysis_response.status_code}",
                        )
                except (requests.RequestException, ValueError) as exc:
                    analysis_error = f"Analysis request failed: {exc}"

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

    reference_entry = safe_get(entry, "Reference", "reference")
    reference_only = reference_entry is not None
    mc_entry = reference_entry or safe_get(entry, "MC", "Monte Carlo", "mc")
    model_entry = safe_get(entry, "Model", "model", "Model")
    if mc_entry is None:
        for k, v in entry.items():
            if k.lower().startswith("m") and isinstance(v, dict) and "price" in v:
                mc_entry = v
                break
    if model_entry is None and not reference_only:
        for k, v in entry.items():
            if (
                k not in ("MC", "Monte Carlo", "mc")
                and isinstance(v, dict)
                and "price" in v
            ):
                model_entry = v
                break

    if mc_entry is None or (model_entry is None and not reference_only):
        st.warning(
            "MC or Model entry missing — showing raw per_npaths entry for debugging."
        )
        st.json(entry)
        st.stop()

    mc_price = as_float(safe_get(mc_entry, "price", "mean", "value"), default=math.nan)
    mc_time = as_float(
        safe_get(mc_entry, "time_s", "time", "elapsed", "timing"), default=math.nan
    )
    mc_std = as_float(
        safe_get(mc_entry, "payoff_std", "std", "stddev", "var"), default=math.nan
    )

    standard_error = as_float(safe_get(mc_entry, "standard_error"), default=math.nan)
    confidence_interval = safe_get(mc_entry, "confidence_interval", default=[])

    if reference_only:
        model_price = math.nan
        model_time = math.nan
        abs_error = math.nan
        rel_error = math.nan
        speedup = math.nan
    else:
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

    # Summarize a row for history
    now = datetime.now(timezone.utc).isoformat()
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

    # Show success / toast if available
    try:
        # some streamlit versions have st.toast
        if hasattr(st, "toast"):
            st.toast("Pricing run complete", icon="✅")
        else:
            st.success("Pricing run complete")
    except Exception:
        st.success("Pricing run complete")

    # ----- Tabs: Dashboard, Diagnostics/Feature, Raw JSON, History, Analysis -----
    tab_names = ["Dashboard", "Feature Analysis / Explanation", "Raw JSON", "History"]
    if analysis_mode != "Price only":
        tab_names.append("Scenario" if analysis_mode == "Price + scenario" else "Risk")
    tabs = st.tabs(tab_names)
    tab_dashboard, tab_feature, tab_json, tab_history = tabs[:4]
    tab_analysis = tabs[4] if len(tabs) > 4 else None

    with tab_dashboard:
        market_snapshot = result.get("market_snapshot", {})
        market_display = market_snapshot or result.get("market_term_structure", {})
        if market_display:
            st.caption(
                f"{market_display.get('symbol')} | "
                f"{market_display.get('underlier_type')} | "
                f"as of {market_display.get('market_data_time')} | "
                f"source: {market_display.get('source')}"
            )
        if reference_only:
            a, b, c = st.columns([1, 1, 1])
            a.metric("Reference price", f"{mc_price:.6f}")
            b.metric("Standard error", f"{standard_error:.6f}")
            ci_text = (
                f"{float(confidence_interval[0]):.6f} - "
                f"{float(confidence_interval[1]):.6f}"
                if isinstance(confidence_interval, list)
                and len(confidence_interval) == 2
                else "N/A"
            )
            c.metric("95% confidence interval", ci_text)
            time_df = pd.DataFrame(
                {"component": ["Reference MC Time (s)"], "time": [mc_time]}
            )
        else:
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
                price_fig,
                use_container_width=True,
                config=plotly_config,
                height=340,
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

    if tab_analysis is not None:
        with tab_analysis:
            if analysis_error:
                st.error(analysis_error)
                if analysis_result_raw is not None:
                    st.json(analysis_result_raw)
            elif analysis_result is None:
                st.info("No analysis result was returned.")
            elif analysis_mode == "Price + scenario":
                base_value = analysis_result["base_valuation"]["price"]
                shocked_value = analysis_result["shocked_valuation"]["price"]
                pnl = analysis_result["pnl"]
                metric1, metric2, metric3 = st.columns(3)
                metric1.metric("Base value", f"{base_value:.6f}")
                metric2.metric("Shocked value", f"{shocked_value:.6f}")
                metric3.metric(
                    "Scenario P&L",
                    f"{pnl['value']:.6f}",
                    delta=f"SE {pnl['standard_error']:.6f}",
                )
                st.caption(
                    f"Run {analysis_result_raw.get('run_id')} | paired paths: "
                    f"{analysis_result['provenance']['common_random_numbers']}"
                )
                scenario_frame = pd.DataFrame(
                    [
                        {"valuation": "Base", "price": base_value},
                        {"valuation": "Shocked", "price": shocked_value},
                    ]
                )
                st.plotly_chart(
                    px.bar(
                        scenario_frame,
                        x="valuation",
                        y="price",
                        text="price",
                        title="Base and shocked Phoenix value",
                    ),
                    use_container_width=True,
                    config=plotly_config,
                )
                st.json(analysis_result)
            else:
                sensitivities = analysis_result["sensitivities"]
                risk_rows = [
                    {
                        "risk": name,
                        "value": item["value"],
                        "standard_error": item["standard_error"],
                        "resolved_95pct": item["statistically_resolved_95pct"],
                        "units": item["units"],
                    }
                    for name, item in sensitivities.items()
                ]
                risk_frame = pd.DataFrame(risk_rows)
                st.caption(
                    f"Run {analysis_result_raw.get('run_id')} | common random numbers: "
                    f"{analysis_result['provenance']['common_random_numbers']}"
                )
                st.dataframe(risk_frame, use_container_width=True)
                st.plotly_chart(
                    px.bar(
                        risk_frame,
                        x="risk",
                        y="value",
                        color="resolved_95pct",
                        title="Finite-difference sensitivities",
                    ),
                    use_container_width=True,
                    config=plotly_config,
                )
                st.json(analysis_result)

# End run_clicked handling

# If not run yet: provide quick note & allow viewing history anyway
if not run_clicked:
    st.info(
        "Configure parameters and click 'Run Pricing' to compare ML model vs Monte Carlo. You can also view historical runs if any exist in History tab after running once."
    )
