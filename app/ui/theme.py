import streamlit as st


def apply_workspace_theme() -> None:
    st.markdown(
        """
        <style>
        :root {
          --mlp-ink: var(--text-color);
          --mlp-muted: color-mix(in srgb, var(--text-color) 64%, transparent);
          --mlp-panel: color-mix(
            in srgb, var(--secondary-background-color) 94%, transparent
          );
          --mlp-panel-strong: var(--secondary-background-color);
          --mlp-canvas: var(--background-color);
          --mlp-line: color-mix(in srgb, var(--text-color) 16%, transparent);
          --mlp-line-strong: color-mix(in srgb, var(--text-color) 28%, transparent);
          --mlp-accent: var(--primary-color);
          --mlp-cyan: #7895a7;
          --mlp-violet: #a35b52;
          --mlp-teal: #4f938b;
          --mlp-gold: #c3a260;
          --mlp-rose: #bc5b5b;
          --mlp-space: #0b0c0f;
          --mlp-space-soft: #191b20;
          --mlp-shadow: 0 18px 54px rgba(12, 9, 24, .10);
        }

        .stApp {
          background:
            radial-gradient(
              circle at 86% -8%,
              color-mix(in srgb, var(--mlp-accent) 9%, transparent),
              transparent 27rem
            ),
            radial-gradient(
              circle at 4% 28%,
              color-mix(in srgb, var(--mlp-cyan) 7%, transparent),
              transparent 24rem
            ),
            var(--mlp-canvas);
        }

        [data-testid="stSidebar"] {
          border-right: 1px solid var(--mlp-line);
        }

        [data-testid="stSidebar"]::before {
          content: "";
          display: block;
          height: 3px;
          background: linear-gradient(
            90deg,
            var(--mlp-accent),
            var(--mlp-gold),
            var(--mlp-cyan)
          );
        }

        .mlp-hero {
          position: relative;
          overflow: hidden;
          color: #f8f6ff;
          background:
            radial-gradient(circle at 82% 22%, rgba(201, 101, 82, .28), transparent 13rem),
            radial-gradient(circle at 63% 115%, rgba(195, 162, 96, .18), transparent 22rem),
            linear-gradient(118deg, #090a0c 0%, #15171b 58%, #242125 118%);
          border: 1px solid rgba(232, 227, 218, .20);
          border-radius: 3px 18px 3px 18px;
          padding: 1.7rem 1.85rem 1.75rem;
          margin: .25rem 0 1.35rem;
          box-shadow: 0 22px 65px rgba(9, 10, 12, .24);
          isolation: isolate;
        }

        .mlp-hero::before,
        .mlp-hero::after {
          content: "";
          position: absolute;
          pointer-events: none;
          z-index: -1;
        }

        .mlp-hero::before {
          width: 18rem;
          height: 11rem;
          right: 2.5rem;
          top: -2rem;
          opacity: .45;
          background:
            repeating-linear-gradient(
              90deg,
              transparent 0 17px,
              rgba(232, 227, 218, .13) 18px,
              transparent 19px
            );
          transform: rotate(-8deg);
        }

        .mlp-hero::after {
          inset: 0;
          opacity: .30;
          background-image:
            linear-gradient(90deg, transparent 0 74%, rgba(201, 101, 82, .7) 74% 74.2%, transparent 74.2%),
            repeating-linear-gradient(
              0deg,
              transparent 0 25px,
              rgba(232, 227, 218, .07) 26px,
              transparent 27px
            );
        }

        .mlp-eyebrow,
        .mlp-section-label {
          display: inline-block;
          color: #a9bfcc;
          font-family: ui-monospace, SFMono-Regular, Consolas, monospace;
          font-size: .72rem;
          font-weight: 700;
          letter-spacing: .14em;
          text-transform: uppercase;
        }

        .mlp-hero h1 {
          max-width: 45rem;
          color: #ffffff;
          font-size: clamp(1.95rem, 4vw, 2.65rem);
          line-height: 1.08;
          margin: .42rem 0 .55rem;
        }

        .mlp-hero p {
          max-width: 54rem;
          color: rgba(248, 246, 255, .78);
          margin: 0;
        }

        .mlp-card {
          position: relative;
          min-height: 7rem;
          overflow: hidden;
          color: var(--mlp-ink);
          background: var(--mlp-panel);
          border: 1px solid var(--mlp-line);
          border-top: 2px solid var(--mlp-violet);
          border-radius: 3px 13px 3px 13px;
          padding: 1rem 1.1rem;
          box-shadow: 0 10px 28px rgba(12, 9, 24, .055);
        }

        .mlp-card::after {
          content: "";
          position: absolute;
          width: 2.5rem;
          height: 2.5rem;
          right: -1.35rem;
          bottom: -1.35rem;
          border: 1px solid color-mix(in srgb, var(--mlp-accent) 40%, transparent);
          transform: rotate(45deg);
        }

        .mlp-card h4 {
          color: var(--mlp-ink);
          margin: 0 0 .4rem;
        }

        .mlp-card p {
          color: var(--mlp-muted);
          font-size: .9rem;
          margin: 0;
        }

        .mlp-sidebar-lesson {
          display: flex;
          flex-direction: column;
          gap: .25rem;
          margin: .2rem 0 .8rem;
          padding: .78rem .85rem;
          border: 1px solid var(--mlp-line-strong);
          border-left: 3px solid var(--mlp-cyan);
          border-radius: 2px 10px 2px 10px;
          background: color-mix(
            in srgb, var(--secondary-background-color) 84%, transparent
          );
        }

        .mlp-sidebar-lesson span {
          color: color-mix(in srgb, var(--text-color) 72%, transparent);
          font-size: .78rem;
        }

        .mlp-foundation {
          position: relative;
          overflow: hidden;
          padding: 1.2rem 1.25rem 1.1rem;
          margin: .3rem 0 .9rem;
          background:
            linear-gradient(
              105deg,
              color-mix(in srgb, var(--mlp-violet) 10%, var(--mlp-panel)),
              var(--mlp-panel) 58%
            );
          border: 1px solid var(--mlp-line);
          border-left: 4px solid var(--mlp-violet);
          border-radius: 2px 16px 2px 16px;
        }

        .mlp-foundation h3 {
          margin: .35rem 0 .85rem;
        }

        .mlp-foundation > p {
          color: var(--mlp-muted);
          margin: .85rem 0 0;
          font-size: .9rem;
        }

        .mlp-foundation-flow {
          display: grid;
          grid-template-columns: 1fr auto 1fr auto 1fr;
          align-items: stretch;
          gap: .55rem;
        }

        .mlp-foundation-flow div {
          display: flex;
          flex-direction: column;
          gap: .22rem;
          padding: .75rem .8rem;
          background: color-mix(in srgb, var(--mlp-panel-strong) 84%, transparent);
          border: 1px solid var(--mlp-line);
          border-radius: 2px 10px 2px 10px;
        }

        .mlp-foundation-flow span {
          color: var(--mlp-muted);
          font-size: .8rem;
        }

        .mlp-foundation-flow i {
          align-self: center;
          color: var(--mlp-gold);
          font-size: 1.15rem;
          font-style: normal;
        }

        .mlp-word-card {
          margin: -.25rem 0 1.2rem;
          padding: .95rem 1rem;
          background: var(--mlp-panel);
          border: 1px solid var(--mlp-line);
          border-bottom: 2px solid var(--mlp-cyan);
          border-radius: 3px 11px 3px 11px;
        }

        .mlp-word-card > span,
        .mlp-chart-guide span,
        .mlp-recipe span {
          color: var(--mlp-muted);
          font-family: ui-monospace, SFMono-Regular, Consolas, monospace;
          font-size: .68rem;
          font-weight: 700;
          letter-spacing: .09em;
          text-transform: uppercase;
        }

        .mlp-word-card h4 {
          margin: .25rem 0 .35rem;
        }

        .mlp-word-card p {
          color: var(--mlp-muted);
          margin: 0 0 .7rem;
        }

        .mlp-word-card aside {
          color: var(--mlp-ink);
          padding: .65rem .75rem;
          background: color-mix(in srgb, var(--mlp-gold) 10%, transparent);
          border-left: 2px solid var(--mlp-gold);
          font-size: .86rem;
        }

        .mlp-stepper {
          display: grid;
          grid-template-columns: repeat(5, minmax(0, 1fr));
          gap: .45rem;
          margin: .55rem 0 .65rem;
        }

        .mlp-step {
          display: flex;
          align-items: center;
          gap: .45rem;
          min-height: 3.1rem;
          padding: .55rem .65rem;
          color: var(--mlp-muted);
          background: color-mix(in srgb, var(--mlp-panel) 82%, transparent);
          border: 1px solid var(--mlp-line);
          border-radius: 2px 9px 2px 9px;
          font-size: .76rem;
          line-height: 1.15;
        }

        .mlp-step span {
          display: inline-grid;
          place-items: center;
          flex: 0 0 1.55rem;
          height: 1.55rem;
          color: var(--mlp-muted);
          background: color-mix(in srgb, var(--mlp-ink) 9%, transparent);
          border-radius: 50%;
          font-family: ui-monospace, SFMono-Regular, Consolas, monospace;
          font-weight: 800;
        }

        .mlp-step-active {
          color: var(--mlp-ink);
          background: color-mix(in srgb, var(--mlp-accent) 11%, var(--mlp-panel));
          border-color: color-mix(in srgb, var(--mlp-accent) 65%, var(--mlp-line));
          border-top-width: 2px;
          font-weight: 700;
        }

        .mlp-step-active span {
          color: #0b0914;
          background: var(--mlp-cyan);
        }

        .mlp-recipe {
          display: grid;
          grid-template-columns: repeat(2, minmax(0, 1fr));
          gap: .65rem;
          margin: .8rem 0 1rem;
        }

        .mlp-recipe div {
          display: flex;
          flex-direction: column;
          gap: .2rem;
          padding: .8rem .9rem;
          color: var(--mlp-ink);
          background: var(--mlp-panel);
          border: 1px solid var(--mlp-line);
          border-radius: 2px 10px 2px 10px;
        }

        .mlp-lesson-answer {
          display: flex;
          position: relative;
          overflow: hidden;
          flex-direction: column;
          gap: .35rem;
          margin: .4rem 0 1rem;
          padding: 1.2rem 1.3rem;
          color: #f8f6ff;
          background:
            radial-gradient(circle at 88% 20%, rgba(201, 101, 82, .28), transparent 10rem),
            linear-gradient(120deg, #111216, #2a2222);
          border: 1px solid rgba(232, 227, 218, .20);
          border-radius: 3px 16px 3px 16px;
          box-shadow: 0 15px 36px rgba(9, 10, 12, .20);
        }

        .mlp-lesson-answer span {
          color: #a9bfcc;
          font-size: .75rem;
          font-weight: 700;
          letter-spacing: .08em;
          text-transform: uppercase;
        }

        .mlp-lesson-answer strong {
          font-size: 1.25rem;
        }

        .mlp-lesson-answer p {
          color: rgba(248, 246, 255, .80);
          margin: 0;
        }

        .mlp-mini-lesson {
          display: flex;
          flex-direction: column;
          gap: .3rem;
          min-height: 6.6rem;
          margin-bottom: .8rem;
          padding: .8rem .9rem;
          background: color-mix(in srgb, var(--mlp-gold) 9%, var(--mlp-panel));
          border: 1px solid color-mix(in srgb, var(--mlp-gold) 48%, var(--mlp-line));
          border-radius: 2px 10px 2px 10px;
        }

        .mlp-mini-lesson span {
          color: var(--mlp-muted);
          font-size: .82rem;
        }

        .mlp-ml-status {
          display: flex;
          flex-direction: column;
          gap: .28rem;
          margin: .25rem 0 1.1rem;
          padding: .95rem 1rem;
          color: var(--mlp-ink);
          background: var(--mlp-panel);
          border: 1px solid var(--mlp-line);
          border-left: 3px solid var(--mlp-gold);
          border-radius: 2px 10px 2px 10px;
        }

        .mlp-ml-status span,
        .mlp-ml-explainer span,
        .mlp-rulebook span {
          color: var(--mlp-muted);
          font-family: "IBM Plex Mono", ui-monospace, SFMono-Regular, Consolas, monospace;
          font-size: .68rem;
          font-weight: 700;
          letter-spacing: .08em;
          text-transform: uppercase;
        }

        .mlp-ml-status p,
        .mlp-ml-explainer p,
        .mlp-rulebook p {
          color: var(--mlp-muted);
          margin: 0;
        }

        .mlp-ml-explainer {
          display: grid;
          grid-template-columns: repeat(2, minmax(0, 1fr));
          margin: .2rem 0 1rem;
          background: var(--mlp-panel);
          border: 1px solid var(--mlp-line);
          border-top: 2px solid var(--mlp-accent);
          border-radius: 2px 10px 2px 10px;
        }

        .mlp-ml-explainer div {
          padding: .82rem .9rem;
          border-right: 1px solid var(--mlp-line);
        }

        .mlp-ml-explainer div:last-child {
          border-right: 0;
        }

        .mlp-ml-explainer p {
          margin-top: .3rem;
          font-size: .86rem;
          line-height: 1.5;
        }

        .mlp-interpretation-list {
          display: grid;
          gap: .55rem;
          margin: .4rem 0 1.2rem;
        }

        .mlp-interpretation {
          display: grid;
          grid-template-columns: 2.2rem 1fr;
          gap: .75rem;
          align-items: start;
          padding: .78rem .9rem;
          color: var(--mlp-ink);
          background: var(--mlp-panel);
          border: 1px solid var(--mlp-line);
          border-radius: 2px 9px 2px 9px;
        }

        .mlp-interpretation > span {
          display: grid;
          place-items: center;
          width: 2rem;
          height: 2rem;
          color: var(--mlp-accent);
          border: 1px solid color-mix(in srgb, var(--mlp-accent) 52%, var(--mlp-line));
          border-radius: 50%;
          font-family: "IBM Plex Mono", ui-monospace, monospace;
          font-size: .72rem;
          font-weight: 700;
        }

        .mlp-interpretation strong {
          display: block;
          margin-bottom: .18rem;
        }

        .mlp-interpretation p {
          color: var(--mlp-muted);
          font-size: .86rem;
          line-height: 1.45;
          margin: 0;
        }

        .mlp-rulebook {
          display: grid;
          grid-template-columns: repeat(3, minmax(0, 1fr));
          gap: .65rem;
          margin: .4rem 0 1rem;
        }

        .mlp-rulebook > div {
          min-height: 10.8rem;
          padding: .9rem .95rem;
          color: var(--mlp-ink);
          background: var(--mlp-panel);
          border: 1px solid var(--mlp-line);
          border-top: 2px solid var(--mlp-accent);
          border-radius: 2px 10px 2px 10px;
        }

        .mlp-rulebook > div:first-child {
          border-top-color: var(--mlp-teal);
        }

        .mlp-rulebook > div:last-child {
          border-top-color: var(--mlp-rose);
        }

        .mlp-rulebook strong {
          display: block;
          margin: .4rem 0 .3rem;
        }

        .mlp-rulebook p {
          font-size: .84rem;
          line-height: 1.5;
        }

        .mlp-chart-guide {
          display: grid;
          grid-template-columns: .85fr 1.15fr 1.15fr;
          gap: 0;
          margin: -.25rem 0 1rem;
          background: var(--mlp-panel);
          border: 1px solid var(--mlp-line);
          border-top: 2px solid var(--mlp-violet);
          border-radius: 2px 10px 2px 10px;
        }

        .mlp-chart-guide div {
          padding: .72rem .8rem;
          border-right: 1px solid var(--mlp-line);
        }

        .mlp-chart-guide div:last-child {
          border-right: 0;
        }

        .mlp-chart-guide b,
        .mlp-chart-guide p {
          display: block;
          color: var(--mlp-ink);
          font-size: .79rem;
          line-height: 1.4;
          margin: .24rem 0 0;
        }

        .mlp-chart-guide p {
          color: var(--mlp-muted);
        }

        [data-testid="stMetric"] {
          background: var(--mlp-panel);
          border: 1px solid var(--mlp-line);
          border-top: 2px solid color-mix(in srgb, var(--mlp-accent) 64%, transparent);
          border-radius: 2px 11px 2px 11px;
          padding: .8rem 1rem;
          box-shadow: 0 8px 22px rgba(12, 9, 24, .045);
        }

        [data-testid="stMetricValue"] {
          color: var(--mlp-ink);
          font-variant-numeric: tabular-nums;
        }

        .stButton > button[kind="primary"],
        .stFormSubmitButton > button[kind="primary"] {
          box-shadow: 0 7px 20px color-mix(in srgb, var(--mlp-accent) 23%, transparent);
          font-weight: 700;
        }

        [data-baseweb="tab-list"] {
          gap: .3rem;
          background: color-mix(in srgb, var(--mlp-panel) 78%, transparent);
          border: 1px solid var(--mlp-line);
          border-radius: 2px 11px 2px 11px;
          padding: .3rem;
        }

        [data-baseweb="tab"] {
          border-radius: 2px 8px 2px 8px;
          padding-left: 1rem;
          padding-right: 1rem;
        }

        div[data-testid="stPlotlyChart"] {
          background: var(--mlp-panel);
          border: 1px solid var(--mlp-line);
          border-radius: 3px 14px 3px 14px;
          padding: .2rem;
          box-shadow: var(--mlp-shadow);
        }

        .mlp-table-wrap {
          overflow-x: auto;
          background: var(--mlp-panel);
          border: 1px solid var(--mlp-line);
          border-radius: 2px 10px 2px 10px;
        }

        .mlp-table {
          width: 100%;
          border-collapse: collapse;
          color: var(--mlp-ink);
          font-size: .88rem;
          font-variant-numeric: tabular-nums;
        }

        .mlp-table th {
          color: var(--mlp-muted);
          background: color-mix(in srgb, var(--mlp-ink) 5%, transparent);
          font-size: .72rem;
          letter-spacing: .06em;
          text-align: left;
          text-transform: uppercase;
        }

        .mlp-table th,
        .mlp-table td {
          border-bottom: 1px solid var(--mlp-line);
          padding: .68rem .8rem;
          white-space: nowrap;
        }

        .mlp-table tbody tr:last-child td {
          border-bottom: 0;
        }

        code,
        pre,
        .stCode {
          font-variant-numeric: tabular-nums;
        }

        @media (max-width: 900px) {
          .mlp-stepper,
          .mlp-chart-guide,
          .mlp-foundation-flow,
          .mlp-ml-explainer,
          .mlp-rulebook {
            grid-template-columns: 1fr;
          }

          .mlp-step {
            min-height: auto;
          }

          .mlp-recipe {
            grid-template-columns: 1fr;
          }

          .mlp-foundation-flow i {
            transform: rotate(90deg);
          }

          .mlp-chart-guide div {
            border-right: 0;
            border-bottom: 1px solid var(--mlp-line);
          }

          .mlp-chart-guide div:last-child {
            border-bottom: 0;
          }

          .mlp-ml-explainer div {
            border-right: 0;
            border-bottom: 1px solid var(--mlp-line);
          }

          .mlp-ml-explainer div:last-child {
            border-bottom: 0;
          }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
