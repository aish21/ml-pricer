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

        .mlp-concept-visual {
          --mlp-concept: var(--mlp-cyan);
          position: relative;
          overflow: hidden;
          margin: .25rem 0 .75rem;
          padding: 1rem 1.05rem .9rem;
          color: var(--mlp-ink);
          background:
            radial-gradient(
              circle at 12% 0,
              color-mix(in srgb, var(--mlp-concept) 12%, transparent),
              transparent 13rem
            ),
            var(--mlp-panel);
          border: 1px solid var(--mlp-line);
          border-top: 3px solid var(--mlp-concept);
          border-radius: 3px 15px 3px 15px;
          box-shadow: 0 12px 30px rgba(12, 9, 24, .055);
          animation: mlp-concept-arrive .35s ease-out both;
        }

        .mlp-concept-violet {
          --mlp-concept: var(--mlp-violet);
        }

        .mlp-concept-cyan {
          --mlp-concept: var(--mlp-cyan);
        }

        .mlp-concept-gold {
          --mlp-concept: var(--mlp-gold);
        }

        .mlp-concept-teal {
          --mlp-concept: var(--mlp-teal);
        }

        .mlp-concept-rose {
          --mlp-concept: var(--mlp-rose);
        }

        .mlp-concept-heading {
          display: flex;
          align-items: baseline;
          justify-content: space-between;
          gap: 1rem;
          margin-bottom: .55rem;
        }

        .mlp-concept-heading > span {
          color: var(--mlp-muted);
          font-family: "IBM Plex Mono", ui-monospace, SFMono-Regular, Consolas, monospace;
          font-size: .68rem;
          font-weight: 700;
          letter-spacing: .09em;
          text-transform: uppercase;
        }

        .mlp-concept-heading > strong {
          color: var(--mlp-concept);
          font-size: 1.05rem;
        }

        .mlp-concept-layout {
          display: grid;
          grid-template-columns: minmax(15rem, .9fr) minmax(18rem, 1.1fr);
          gap: 1rem;
          align-items: stretch;
        }

        .mlp-concept-canvas {
          display: grid;
          place-items: center;
          min-height: 9.2rem;
          overflow: hidden;
          color: var(--mlp-concept);
          background:
            linear-gradient(
              color-mix(in srgb, var(--mlp-ink) 4%, transparent) 1px,
              transparent 1px
            ),
            linear-gradient(
              90deg,
              color-mix(in srgb, var(--mlp-ink) 4%, transparent) 1px,
              transparent 1px
            ),
            color-mix(in srgb, var(--mlp-panel-strong) 70%, transparent);
          background-size: 22px 22px;
          border: 1px solid var(--mlp-line);
          border-radius: 2px 11px 2px 11px;
        }

        .mlp-concept-canvas svg {
          width: min(100%, 21rem);
          height: auto;
          overflow: visible;
        }

        .mlp-concept-line,
        .mlp-concept-path,
        .mlp-concept-grid,
        .mlp-concept-threshold,
        .mlp-concept-shape {
          vector-effect: non-scaling-stroke;
        }

        .mlp-concept-line {
          fill: none;
          stroke: var(--mlp-concept);
          stroke-width: 3;
          stroke-linecap: round;
          stroke-dasharray: 10 9;
          animation: mlp-concept-flow 1.8s linear infinite;
        }

        .mlp-concept-grid {
          fill: none;
          stroke: color-mix(in srgb, var(--mlp-ink) 25%, transparent);
          stroke-width: 1.2;
        }

        .mlp-concept-path {
          fill: none;
          stroke: var(--mlp-concept);
          stroke-width: 2;
          stroke-linecap: round;
          opacity: .34;
          stroke-dasharray: 6 8;
          animation: mlp-concept-flow 2.8s linear infinite;
        }

        .mlp-path-two {
          animation-direction: reverse;
          animation-duration: 2.2s;
        }

        .mlp-concept-threshold {
          fill: none;
          stroke: color-mix(in srgb, var(--mlp-gold) 80%, var(--mlp-concept));
          stroke-width: 2;
          stroke-dasharray: 5 5;
          animation: mlp-concept-threshold 2.2s ease-in-out infinite;
        }

        .mlp-concept-shape {
          fill: color-mix(in srgb, var(--mlp-concept) 10%, var(--mlp-panel));
          stroke: var(--mlp-concept);
          stroke-width: 2;
          transform-box: fill-box;
          transform-origin: center;
          animation: mlp-concept-breathe 2.6s ease-in-out infinite;
        }

        .mlp-concept-orb {
          fill: var(--mlp-concept);
          stroke: color-mix(in srgb, var(--mlp-panel) 76%, transparent);
          stroke-width: 4;
          transform-box: fill-box;
          transform-origin: center;
          animation: mlp-concept-pulse 1.55s ease-in-out infinite;
          filter: drop-shadow(
            0 0 7px color-mix(in srgb, var(--mlp-concept) 62%, transparent)
          );
        }

        .mlp-concept-steps {
          display: grid;
          gap: .45rem;
        }

        .mlp-concept-step {
          display: grid;
          grid-template-columns: 2.35rem 1fr;
          gap: .65rem;
          align-items: center;
          min-height: 2.75rem;
          padding: .48rem .62rem;
          background: color-mix(in srgb, var(--mlp-panel-strong) 84%, transparent);
          border: 1px solid var(--mlp-line);
          border-radius: 2px 9px 2px 9px;
          opacity: 0;
          animation: mlp-concept-step-in .42s ease-out forwards;
        }

        .mlp-concept-step:nth-child(2) {
          animation-delay: .16s;
        }

        .mlp-concept-step:nth-child(3) {
          animation-delay: .32s;
        }

        .mlp-concept-step > span {
          display: grid;
          place-items: center;
          width: 2.2rem;
          height: 2.2rem;
          color: color-mix(in srgb, var(--mlp-concept) 78%, var(--mlp-ink));
          background: color-mix(in srgb, var(--mlp-concept) 11%, transparent);
          border: 1px solid color-mix(in srgb, var(--mlp-concept) 45%, var(--mlp-line));
          border-radius: 50%;
          font-family: "IBM Plex Mono", ui-monospace, monospace;
          font-size: .78rem;
          font-weight: 800;
        }

        .mlp-concept-step > b {
          font-size: .86rem;
          line-height: 1.25;
        }

        .mlp-concept-caption {
          color: var(--mlp-muted);
          font-size: .83rem;
          line-height: 1.45;
          margin: .7rem 0 0;
          padding-left: .7rem;
          border-left: 2px solid var(--mlp-concept);
        }

        @keyframes mlp-concept-arrive {
          from { opacity: 0; transform: translateY(.4rem) scale(.995); }
          to { opacity: 1; transform: translateY(0) scale(1); }
        }

        @keyframes mlp-concept-step-in {
          from { opacity: 0; transform: translateX(.65rem); }
          to { opacity: 1; transform: translateX(0); }
        }

        @keyframes mlp-concept-flow {
          to { stroke-dashoffset: -38; }
        }

        @keyframes mlp-concept-pulse {
          0%, 100% { transform: scale(.8); opacity: .62; }
          50% { transform: scale(1.14); opacity: 1; }
        }

        @keyframes mlp-concept-breathe {
          0%, 100% { transform: scale(.96); opacity: .68; }
          50% { transform: scale(1.04); opacity: 1; }
        }

        @keyframes mlp-concept-threshold {
          0%, 100% { opacity: .45; }
          50% { opacity: 1; }
        }

        .mlp-stepper {
          display: grid;
          grid-template-columns: repeat(5, minmax(0, 1fr));
          gap: .45rem;
          margin: .55rem 0 .65rem;
        }

        .st-key-guided_step_navigation [data-testid="stHorizontalBlock"] {
          gap: .45rem;
        }

        .st-key-guided_step_navigation [data-testid="stColumn"] {
          min-width: 7.5rem;
        }

        .st-key-guided_step_navigation .stButton > button {
          min-height: 3.2rem;
          padding: .55rem .65rem;
          white-space: normal;
          line-height: 1.15;
        }

        .st-key-guided_step_navigation .stButton > button[kind="primary"] {
          border-top-width: 2px;
          box-shadow: 0 7px 20px color-mix(
            in srgb,
            var(--mlp-accent) 20%,
            transparent
          );
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

        .mlp-evidence-split {
          display: grid;
          grid-template-columns: repeat(3, minmax(0, 1fr));
          margin: .45rem 0 1.1rem;
          color: var(--mlp-ink);
          background: var(--mlp-panel);
          border: 1px solid var(--mlp-line);
          border-top: 2px solid var(--mlp-accent);
          border-radius: 2px 11px 2px 11px;
        }

        .mlp-evidence-split > div {
          min-height: 8.4rem;
          padding: .85rem .9rem;
          border-right: 1px solid var(--mlp-line);
        }

        .mlp-evidence-split > div:last-child {
          border-right: 0;
        }

        .mlp-evidence-split span {
          display: block;
          color: var(--mlp-muted);
          font-family: "IBM Plex Mono", ui-monospace, monospace;
          font-size: .68rem;
          font-weight: 700;
          letter-spacing: .08em;
          text-transform: uppercase;
        }

        .mlp-evidence-split strong {
          display: block;
          margin: .32rem 0 .25rem;
        }

        .mlp-evidence-split p {
          color: var(--mlp-muted);
          font-size: .83rem;
          line-height: 1.45;
          margin: 0;
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

        .mlp-focus-card {
          display: grid;
          gap: .25rem;
          margin: .35rem 0 .9rem;
          padding: .8rem .95rem;
          color: var(--mlp-ink);
          background:
            linear-gradient(
              100deg,
              color-mix(in srgb, var(--mlp-cyan) 10%, var(--mlp-panel)),
              var(--mlp-panel) 72%
            );
          border: 1px solid var(--mlp-line);
          border-left: 3px solid var(--mlp-cyan);
          border-radius: 2px 11px 2px 11px;
        }

        .mlp-focus-card span,
        .mlp-outcome-stage span,
        .mlp-lesson-stage > span,
        .mlp-current-lesson > span,
        .mlp-answer-lanes small {
          color: var(--mlp-muted);
          font-family: "IBM Plex Mono", ui-monospace, SFMono-Regular, Consolas, monospace;
          font-size: .68rem;
          font-weight: 700;
          letter-spacing: .08em;
          text-transform: uppercase;
        }

        .mlp-focus-card strong {
          font-size: 1rem;
        }

        .mlp-focus-card p,
        .mlp-outcome-stage p,
        .mlp-model-school p,
        .mlp-lesson-stage p,
        .mlp-current-lesson p,
        .mlp-answer-race p {
          color: var(--mlp-muted);
          line-height: 1.5;
          margin: 0;
        }

        .mlp-outcome-stage {
          position: relative;
          overflow: hidden;
          display: grid;
          gap: .3rem;
          margin: .35rem 0 .85rem;
          padding: 1rem 1.05rem;
          background: var(--mlp-panel);
          border: 1px solid var(--mlp-line);
          border-left: 4px solid var(--mlp-cyan);
          border-radius: 2px 12px 2px 12px;
          animation: mlp-reveal .28s ease-out both;
        }

        .mlp-outcome-stage::after {
          content: "";
          position: absolute;
          width: 7rem;
          height: 7rem;
          right: -5rem;
          top: -5rem;
          border: 1.2rem solid color-mix(in srgb, var(--mlp-cyan) 12%, transparent);
          border-radius: 50%;
          animation: mlp-breathe 2.8s ease-in-out infinite;
        }

        .mlp-outcome-finish {
          border-left-color: var(--mlp-teal);
        }

        .mlp-outcome-reward {
          border-left-color: var(--mlp-gold);
        }

        .mlp-outcome-risk {
          border-left-color: var(--mlp-rose);
        }

        .mlp-model-school {
          position: relative;
          overflow: hidden;
          margin: 1rem 0;
          padding: 1.05rem 1.1rem;
          color: var(--mlp-ink);
          background:
            radial-gradient(
              circle at 92% 12%,
              color-mix(in srgb, var(--mlp-gold) 13%, transparent),
              transparent 9rem
            ),
            var(--mlp-panel);
          border: 1px solid var(--mlp-line);
          border-top: 3px solid var(--mlp-gold);
          border-radius: 3px 15px 3px 15px;
        }

        .mlp-model-school h3 {
          margin: .35rem 0 .8rem;
        }

        .mlp-school-flow {
          display: flex;
          align-items: stretch;
          gap: .55rem;
          margin: .6rem 0 .8rem;
        }

        .mlp-school-flow div {
          display: grid;
          flex: 1;
          gap: .18rem;
          align-content: center;
          min-height: 4.7rem;
          padding: .65rem .75rem;
          background: color-mix(in srgb, var(--mlp-panel-strong) 88%, transparent);
          border: 1px solid var(--mlp-line);
          border-radius: 2px 10px 2px 10px;
          animation: mlp-reveal .45s ease-out both;
        }

        .mlp-school-flow div:nth-of-type(2) {
          animation-delay: .12s;
        }

        .mlp-school-flow div:nth-of-type(3) {
          animation-delay: .24s;
        }

        .mlp-school-flow small {
          color: var(--mlp-muted);
          line-height: 1.35;
        }

        .mlp-school-flow > i {
          align-self: center;
          max-width: 8.5rem;
          color: var(--mlp-gold);
          font-size: .72rem;
          font-style: normal;
          font-weight: 700;
          line-height: 1.35;
          text-align: center;
          animation: mlp-nudge 1.7s ease-in-out infinite;
        }

        .mlp-race-track {
          display: grid;
          grid-template-columns: 9rem 1fr;
          gap: .42rem .65rem;
          align-items: center;
          margin-top: .9rem;
          padding-top: .8rem;
          border-top: 1px solid var(--mlp-line);
          color: var(--mlp-muted);
          font-size: .76rem;
        }

        .mlp-race-track > div {
          position: relative;
          height: .42rem;
          overflow: hidden;
          background: color-mix(in srgb, var(--mlp-ink) 9%, transparent);
          border-radius: 99rem;
        }

        .mlp-race-track i {
          position: absolute;
          top: 0;
          left: 0;
          width: 1.2rem;
          height: 100%;
          background: var(--mlp-cyan);
          border-radius: 99rem;
        }

        .mlp-runner-slow {
          animation: mlp-race 4.8s ease-in-out infinite;
        }

        .mlp-runner-fast {
          background: var(--mlp-gold) !important;
          animation: mlp-race 1.8s ease-in-out infinite;
        }

        .mlp-lesson-stage {
          display: grid;
          gap: .32rem;
          min-height: 9.5rem;
          margin: .2rem 0 .8rem;
          padding: 1rem 1.05rem;
          align-content: start;
          color: var(--mlp-ink);
          background: var(--mlp-panel);
          border: 1px solid var(--mlp-line);
          border-left: 3px solid var(--mlp-violet);
          border-radius: 2px 12px 2px 12px;
          animation: mlp-reveal .28s ease-out both;
        }

        .mlp-lesson-stage aside {
          margin-top: .35rem;
          padding: .58rem .65rem;
          color: var(--mlp-ink);
          background: color-mix(in srgb, var(--mlp-gold) 10%, transparent);
          border-left: 2px solid var(--mlp-gold);
          font-size: .84rem;
        }

        .mlp-current-lesson {
          display: grid;
          gap: .28rem;
          margin: .45rem 0 1rem;
          padding: .85rem .95rem;
          background: color-mix(in srgb, var(--mlp-teal) 8%, var(--mlp-panel));
          border: 1px solid color-mix(in srgb, var(--mlp-teal) 42%, var(--mlp-line));
          border-radius: 2px 10px 2px 10px;
        }

        .mlp-answer-race {
          margin: .25rem 0 .8rem;
          padding: 1rem 1.05rem;
          color: var(--mlp-ink);
          background: var(--mlp-panel);
          border: 1px solid var(--mlp-line);
          border-top: 3px solid var(--mlp-cyan);
          border-radius: 2px 13px 2px 13px;
        }

        .mlp-answer-lanes {
          display: grid;
          grid-template-columns: 1fr auto 1fr;
          gap: .7rem;
          align-items: stretch;
          margin: .65rem 0;
        }

        .mlp-answer-lanes > div {
          display: grid;
          gap: .25rem;
          padding: .8rem .85rem;
          background: color-mix(in srgb, var(--mlp-panel-strong) 86%, transparent);
          border: 1px solid var(--mlp-line);
          border-radius: 2px 10px 2px 10px;
        }

        .mlp-answer-lanes > i {
          align-self: center;
          color: var(--mlp-gold);
          font-size: .7rem;
          font-style: normal;
          font-weight: 700;
          text-transform: uppercase;
        }

        .mlp-answer-lanes strong {
          font-size: 1.3rem;
          font-variant-numeric: tabular-nums;
        }

        .mlp-answer-lanes p {
          font-size: .82rem;
        }

        .mlp-answer-verdict {
          padding: .7rem .8rem;
          background: color-mix(in srgb, var(--mlp-cyan) 8%, transparent);
          border-left: 2px solid var(--mlp-cyan);
        }

        .mlp-foundation-flow i {
          animation: mlp-nudge 1.7s ease-in-out infinite;
        }

        @keyframes mlp-reveal {
          from { opacity: 0; transform: translateY(.45rem); }
          to { opacity: 1; transform: translateY(0); }
        }

        @keyframes mlp-nudge {
          0%, 100% { transform: translateX(0); opacity: .58; }
          50% { transform: translateX(.22rem); opacity: 1; }
        }

        @keyframes mlp-race {
          0% { left: 0; }
          45%, 72% { left: calc(100% - 1.2rem); }
          100% { left: 0; }
        }

        @keyframes mlp-breathe {
          0%, 100% { transform: scale(.85); opacity: .35; }
          50% { transform: scale(1.06); opacity: .7; }
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
          .mlp-answer-lanes,
          .mlp-ml-explainer,
          .mlp-evidence-split,
          .mlp-rulebook {
            grid-template-columns: 1fr;
          }

          .mlp-concept-layout {
            grid-template-columns: 1fr;
          }

          .mlp-concept-canvas {
            min-height: 7.8rem;
          }

          .st-key-guided_step_navigation [data-testid="stHorizontalBlock"] {
            flex-wrap: wrap;
          }

          .st-key-guided_step_navigation [data-testid="stColumn"] {
            flex: 1 1 9rem !important;
            width: auto !important;
          }

          .mlp-step {
            min-height: auto;
          }

          .mlp-recipe {
            grid-template-columns: 1fr;
          }

          .mlp-foundation-flow i {
            animation: none;
            transform: rotate(90deg);
          }

          .mlp-school-flow {
            flex-direction: column;
          }

          .mlp-school-flow > i {
            max-width: none;
          }

          .mlp-answer-lanes > i {
            text-align: center;
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

          .mlp-evidence-split > div {
            min-height: auto;
            border-right: 0;
            border-bottom: 1px solid var(--mlp-line);
          }

          .mlp-evidence-split > div:last-child {
            border-bottom: 0;
          }
        }

        @media (prefers-reduced-motion: reduce) {
          .mlp-concept-visual,
          .mlp-concept-step,
          .mlp-concept-line,
          .mlp-concept-path,
          .mlp-concept-threshold,
          .mlp-concept-shape,
          .mlp-concept-orb,
          .mlp-foundation-flow i,
          .mlp-school-flow div,
          .mlp-school-flow > i,
          .mlp-outcome-stage,
          .mlp-outcome-stage::after,
          .mlp-lesson-stage,
          .mlp-race-track i {
            animation: none !important;
            transform: none !important;
          }

          .mlp-concept-step {
            opacity: 1;
          }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
