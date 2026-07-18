import streamlit as st


def apply_workspace_theme() -> None:
    st.markdown(
        """
        <style>
        :root {
          --ink: #10233f;
          --muted: #637089;
          --paper: #ffffff;
          --canvas: #f4f6fa;
          --line: #dde3ec;
          --market: #1e88e5;
          --amber: #f59e0b;
        }
        .stApp {
          background:
            radial-gradient(circle at 85% 0%, rgba(30,136,229,.08), transparent 24rem),
            var(--canvas);
          color: var(--ink);
        }
        [data-testid="stSidebar"] {
          background: #10233f;
          border-right: 1px solid rgba(255,255,255,.08);
        }
        [data-testid="stSidebar"] * {
          color: #f7f9fc;
        }
        [data-testid="stSidebar"] [data-baseweb="select"] *,
        [data-testid="stSidebar"] input {
          color: #10233f;
        }
        .np-hero {
          background: linear-gradient(120deg, #10233f 0%, #183d66 70%, #1e88e5 145%);
          border-radius: 18px;
          color: white;
          padding: 1.45rem 1.65rem;
          margin: .2rem 0 1.2rem 0;
          box-shadow: 0 16px 42px rgba(16,35,63,.14);
        }
        .np-eyebrow {
          color: #8ec8ff;
          font-size: .75rem;
          font-weight: 700;
          letter-spacing: .13em;
          text-transform: uppercase;
        }
        .np-hero h1 {
          color: white;
          font-size: 2.05rem;
          margin: .25rem 0 .35rem 0;
        }
        .np-hero p {
          color: rgba(255,255,255,.78);
          margin: 0;
          max-width: 54rem;
        }
        .np-card {
          background: var(--paper);
          border: 1px solid var(--line);
          border-radius: 14px;
          padding: 1rem 1.1rem;
          min-height: 7rem;
          box-shadow: 0 8px 24px rgba(16,35,63,.045);
        }
        .np-card h4 {
          color: var(--ink);
          margin: 0 0 .4rem 0;
        }
        .np-card p {
          color: var(--muted);
          font-size: .9rem;
          margin: 0;
        }
        [data-testid="stMetric"] {
          background: var(--paper);
          border: 1px solid var(--line);
          border-radius: 13px;
          padding: .8rem 1rem;
          box-shadow: 0 7px 20px rgba(16,35,63,.04);
        }
        [data-testid="stMetricValue"] {
          color: var(--ink);
          font-variant-numeric: tabular-nums;
        }
        .stButton > button, .stFormSubmitButton > button {
          border-radius: 10px;
          border: 1px solid #1e88e5;
          background: #1e88e5;
          color: white;
          font-weight: 700;
          min-height: 2.8rem;
        }
        .stButton > button:hover, .stFormSubmitButton > button:hover {
          border-color: #10233f;
          background: #10233f;
          color: white;
        }
        [data-baseweb="tab-list"] {
          gap: .3rem;
          background: rgba(255,255,255,.66);
          border: 1px solid var(--line);
          border-radius: 12px;
          padding: .3rem;
        }
        [data-baseweb="tab"] {
          border-radius: 9px;
          padding-left: 1rem;
          padding-right: 1rem;
        }
        div[data-testid="stPlotlyChart"] {
          background: white;
          border: 1px solid var(--line);
          border-radius: 14px;
          padding: .25rem;
          box-shadow: 0 8px 24px rgba(16,35,63,.04);
        }
        code, pre, .stCode {
          font-variant-numeric: tabular-nums;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
