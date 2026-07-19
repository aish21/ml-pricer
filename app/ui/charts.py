from typing import Any, Mapping, Sequence

import numpy as np
import plotly.graph_objects as go


INK = "#111216"
MUTED = "#767981"
GRID = "rgba(126, 127, 132, .22)"
MARKET = "#7197AA"
REFERENCE = "#A35B52"
COUPON = "#4F938B"
AUTOCALL = "#C3A260"
RISK = "#BC5B5B"
POSITIVE = "#5C9A7D"
PAPER = "rgba(0,0,0,0)"

PLOTLY_CONFIG = {
    "displayModeBar": True,
    "displaylogo": False,
    "scrollZoom": False,
    "editable": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
}


def _layout(
    title: str,
    *,
    subtitle: str | None = None,
    height: int = 360,
    x_title: str | None = None,
    y_title: str | None = None,
) -> dict[str, Any]:
    title_text = f"<b>{title}</b>"
    if subtitle:
        title_text += f"<br><span style='font-size:12px'>{subtitle}</span>"
    return {
        "title": {"text": title_text, "x": 0.0, "font": {"size": 17}},
        "height": height,
        "paper_bgcolor": PAPER,
        "plot_bgcolor": PAPER,
        "font": {"family": "IBM Plex Sans, Arial, sans-serif"},
        "margin": {"l": 54, "r": 30, "t": 82 if subtitle else 62, "b": 78},
        "hoverlabel": {"bgcolor": INK, "font_color": "#F8F6FF"},
        "xaxis": {
            "title": x_title,
            "gridcolor": GRID,
            "zerolinecolor": GRID,
            "showline": False,
        },
        "yaxis": {
            "title": y_title,
            "gridcolor": GRID,
            "zerolinecolor": GRID,
            "showline": False,
        },
        "legend": {
            "orientation": "h",
            "yanchor": "top",
            "y": -0.2,
            "xanchor": "left",
            "x": 0.0,
        },
    }


def price_uncertainty_figure(result: Mapping[str, Any]) -> go.Figure:
    price = float(result["price"])
    interval = result.get("confidence_interval") or [price, price]
    lower, upper = float(interval[0]), float(interval[1])
    delta = price - 1.0
    relationship = (
        f"{abs(delta):.2%} {'above' if delta > 0 else 'below'} par"
        if abs(delta) > 1e-9
        else "at par"
    )
    figure = go.Figure(
        go.Scatter(
            x=[price],
            y=["Model value"],
            mode="markers+text",
            text=[f"{price:.4f} · {relationship}"],
            textposition="top center",
            marker={
                "size": 17,
                "color": MARKET,
                "line": {"color": "#F8F6FF", "width": 2},
            },
            error_x={
                "type": "data",
                "symmetric": False,
                "array": [max(upper - price, 0.0)],
                "arrayminus": [max(price - lower, 0.0)],
                "color": MARKET,
                "thickness": 2,
                "width": 8,
            },
            hovertemplate=(
                "Price %{x:.6f}<br>95% interval "
                f"[{lower:.6f}, {upper:.6f}]<extra></extra>"
            ),
            showlegend=False,
        )
    )
    figure.add_vline(
        x=1.0,
        line_dash="dot",
        line_color=MUTED,
        annotation_text="Par",
        annotation_position="top",
    )
    figure.update_layout(
        **_layout(
            "What is the note worth today?",
            subtitle=(
                "The dot is the estimate; the whisker is simulation noise; "
                "par = 1.00."
            ),
            height=325,
            x_title="Present value per unit notional",
        )
    )
    padding = max(0.025, upper - lower, abs(delta) * 0.25)
    figure.update_xaxes(
        range=[
            min(lower, 1.0) - padding,
            max(upper, 1.0) + padding,
        ]
    )
    figure.update_yaxes(showgrid=False)
    return figure


def barrier_ladder_figure(levels: Sequence[Mapping[str, Any]]) -> go.Figure:
    colors = {
        "market": MARKET,
        "reference": REFERENCE,
        "coupon": COUPON,
        "autocall": AUTOCALL,
        "risk": RISK,
    }
    ordered = sorted(levels, key=lambda item: float(item["level"]))
    values_by_kind = {str(item.get("kind")): float(item["level"]) for item in ordered}
    live_spot = values_by_kind.get("market")
    figure = go.Figure()
    all_values = [float(item["level"]) for item in ordered]
    lower_bound = min(all_values) * 0.9
    upper_bound = max(all_values) * 1.08
    knock_in = values_by_kind.get("risk")
    coupon = values_by_kind.get("coupon")
    autocall = values_by_kind.get("autocall")
    if knock_in is not None:
        figure.add_hrect(
            y0=lower_bound,
            y1=knock_in,
            fillcolor="rgba(240,93,122,.10)",
            line_width=0,
            annotation_text="loss-linked zone",
            annotation_position="top left",
        )
    if knock_in is not None and coupon is not None and coupon > knock_in:
        figure.add_hrect(
            y0=knock_in,
            y1=coupon,
            fillcolor="rgba(139,92,246,.06)",
            line_width=0,
            annotation_text="below reward line",
            annotation_position="top left",
        )
    if coupon is not None:
        reward_top = autocall if autocall is not None else upper_bound
        if reward_top > coupon:
            figure.add_hrect(
                y0=coupon,
                y1=reward_top,
                fillcolor="rgba(45,212,191,.08)",
                line_width=0,
                annotation_text="reward can pay",
                annotation_position="top left",
            )
    if autocall is not None:
        figure.add_hrect(
            y0=autocall,
            y1=upper_bound,
            fillcolor="rgba(246,200,95,.10)",
            line_width=0,
            annotation_text="early finish can trigger",
            annotation_position="top left",
        )
    for item in ordered:
        level = float(item["level"])
        kind = str(item.get("kind"))
        distance = (
            (level / live_spot - 1.0) if live_spot is not None and live_spot else None
        )
        hover_distance = (
            f"<br>{distance:+.1%} versus live spot"
            if distance is not None and kind != "market"
            else ""
        )
        figure.add_hline(
            y=level,
            line_color=colors.get(kind, MUTED),
            line_width=2 if kind != "market" else 3,
            line_dash="solid" if kind == "market" else "dot",
            opacity=0.75,
        )
        figure.add_trace(
            go.Scatter(
                x=[0.54],
                y=[level],
                mode="markers+text",
                text=[f"{item['name']} · {level:,.2f}"],
                textposition="middle right",
                marker={
                    "size": 16 if kind == "market" else 13,
                    "color": colors.get(kind, MUTED),
                    "symbol": "diamond" if kind == "market" else "circle",
                    "line": {"color": "#F8F6FF", "width": 2},
                },
                name=str(item["name"]),
                showlegend=False,
                hovertemplate=(
                    f"{item['name']}: {level:,.4f}{hover_distance}<extra></extra>"
                ),
            )
        )
    figure.update_layout(
        **_layout(
            "Where is spot relative to every rule?",
            subtitle="Read bottom to top. The diamond is today's live spot.",
            height=430,
            y_title="Underlier level",
        )
    )
    figure.update_xaxes(visible=False, range=[0.0, 1.0], fixedrange=True)
    figure.update_yaxes(range=[lower_bound, upper_bound], showgrid=False)
    return figure


def learning_paths_figure(
    *,
    volatility_pct: float,
    autocall_level: float,
    coupon_level: float,
    knock_in_level: float,
) -> go.Figure:
    """Small deterministic path toy used only to explain Monte Carlo."""
    volatility = max(float(volatility_pct), 0.0) / 100.0
    months = np.arange(13)
    dt = 1.0 / 12.0
    rng = np.random.default_rng(17)
    shocks = rng.standard_normal((12, 12))
    log_returns = -0.5 * volatility**2 * dt + volatility * np.sqrt(dt) * shocks
    paths = 100.0 * np.exp(
        np.c_[np.zeros(log_returns.shape[0]), np.cumsum(log_returns, axis=1)]
    )
    figure = go.Figure()
    for index, path in enumerate(paths):
        crossed_safety = float(np.min(path)) <= float(knock_in_level)
        reached_early_finish = float(np.max(path[1:])) >= float(autocall_level)
        outcome = (
            "crossed the safety line"
            if crossed_safety
            else (
                "reached the early-finish line"
                if reached_early_finish
                else "stayed between the outer lines"
            )
        )
        highlight = index < 3
        color = (
            RISK
            if crossed_safety and highlight
            else (
                AUTOCALL
                if reached_early_finish and highlight
                else (MARKET if highlight else "rgba(125,220,255,.16)")
            )
        )
        figure.add_trace(
            go.Scatter(
                x=months,
                y=path,
                mode="lines",
                line={
                    "color": color,
                    "width": 2.5 if highlight else 1,
                },
                name=f"Pretend story {index + 1}",
                showlegend=False,
                hovertemplate=(
                    f"Story {index + 1}<br>Month %{{x}}"
                    f"<br>Price %{{y:.1f}}<br>{outcome}<extra></extra>"
                ),
            )
        )
    for level, label, color, dash in (
        (autocall_level, "Early finish", AUTOCALL, "dash"),
        (coupon_level, "Reward", COUPON, "dot"),
        (knock_in_level, "Safety", RISK, "dashdot"),
    ):
        figure.add_hline(
            y=float(level),
            line_color=color,
            line_dash=dash,
            annotation_text=label,
            annotation_position="right",
        )
    figure.update_layout(
        **_layout(
            "How can the same starting price end differently?",
            subtitle=(
                "Hover over a path. Brighter paths show different rule outcomes."
            ),
            height=460,
            x_title="Months from today",
            y_title="Price, starting at 100",
        )
    )
    figure.update_xaxes(dtick=1)
    return figure


def contract_timeline_figure(
    observation_times: Sequence[float],
    maturity_years: float,
) -> go.Figure:
    times = [float(value) for value in observation_times]
    figure = go.Figure(
        go.Scatter(
            x=times,
            y=[0.0] * len(times),
            mode="lines+markers+text",
            line={"color": GRID, "width": 4},
            marker={"size": 13, "color": AUTOCALL},
            text=[str(index) for index in range(1, len(times) + 1)],
            textposition="top center",
            hovertemplate=(
                "Observation %{text}<br>%{x:.4f} years from valuation<extra></extra>"
            ),
            showlegend=False,
        )
    )
    figure.add_vline(
        x=float(maturity_years),
        line_dash="dash",
        line_color=REFERENCE,
        annotation_text="Maturity",
        annotation_position="bottom right",
    )
    figure.update_layout(
        **_layout(
            "When can the note make a decision?",
            subtitle="Each numbered point is a rule-check; maturity is the final day.",
            height=270,
            x_title="Years from today",
        )
    )
    figure.update_yaxes(visible=False, range=[-0.2, 0.2])
    figure.update_xaxes(range=[0.0, float(maturity_years) * 1.05])
    return figure


def term_structure_figure(market: Mapping[str, Any]) -> go.Figure:
    segments = list(market.get("segments") or [])
    if not segments:
        return go.Figure()
    ends = [0.0] + [float(segment["end_time_years"]) for segment in segments]
    series = (
        ("Risk-free rate", "risk_free_rate", MARKET, "y"),
        ("Dividend yield", "dividend_yield", COUPON, "y"),
        ("Volatility", "volatility", AUTOCALL, "y2"),
    )
    figure = go.Figure()
    for label, field, color, axis in series:
        values = [float(segments[0][field])] + [
            float(segment[field]) for segment in segments
        ]
        figure.add_trace(
            go.Scatter(
                x=ends,
                y=[value * 100.0 for value in values],
                mode="lines",
                line={"shape": "hv", "width": 3, "color": color},
                name=label,
                yaxis=axis,
                hovertemplate=f"{label}: %{{y:.3f}}%<br>t=%{{x:.3f}}y<extra></extra>",
            )
        )
    figure.update_layout(
        **_layout(
            "What market assumptions change through time?",
            subtitle=(
                "Rates and distributions use the left scale; wiggliness uses the right."
            ),
            height=420,
            x_title="Years",
            y_title="Rate / distribution yield (%)",
        )
    )
    figure.update_layout(
        yaxis2={
            "title": "Volatility (%)",
            "overlaying": "y",
            "side": "right",
            "showgrid": False,
            "zeroline": False,
        }
    )
    return figure


def convergence_figure(diagnostics: Mapping[str, Any]) -> go.Figure:
    points = list(diagnostics.get("convergence") or [])
    paths = [int(point["n_paths"]) for point in points]
    prices = [float(point["price"]) for point in points]
    lower = [float(point["confidence_interval_low"]) for point in points]
    upper = [float(point["confidence_interval_high"]) for point in points]
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=paths + list(reversed(paths)),
            y=upper + list(reversed(lower)),
            fill="toself",
            fillcolor="rgba(30, 136, 229, 0.13)",
            line={"color": "rgba(255,255,255,0)"},
            hoverinfo="skip",
            name="95% confidence band",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=paths,
            y=prices,
            mode="lines+markers",
            line={"color": MARKET, "width": 3},
            marker={"size": 8},
            name="Nested estimate",
            hovertemplate="%{x:,} paths<br>Price %{y:.6f}<extra></extra>",
        )
    )
    figure.update_layout(
        **_layout(
            "Did the computer use enough pretend futures?",
            subtitle=(
                "A settling line and narrowing band mean less simulation wobble."
            ),
            height=410,
            x_title="Number of pretend futures",
            y_title="Present value",
        )
    )
    return figure


def cashflow_figure(diagnostics: Mapping[str, Any]) -> go.Figure:
    rows = list((diagnostics.get("cashflows") or {}).get("components") or [])
    labels = {
        "coupon_pv": "Coupons",
        "autocall_principal_pv": "Autocall principal",
        "maturity_protected_pv": "Protected maturity",
        "maturity_downside_pv": "Downside redemption",
    }
    colors = {
        "coupon_pv": COUPON,
        "autocall_principal_pv": AUTOCALL,
        "maturity_protected_pv": MARKET,
        "maturity_downside_pv": RISK,
    }
    values = [float(row["expected_pv"]) for row in rows]
    total = sum(values)
    shares = [(value / total if total else 0.0) for value in values]
    figure = go.Figure(
        go.Bar(
            x=[
                labels.get(str(row["component"]), str(row["component"])) for row in rows
            ],
            y=values,
            error_y={
                "type": "data",
                "array": [float(row["standard_error"]) for row in rows],
                "color": MUTED,
            },
            marker={
                "color": [colors.get(str(row["component"]), MUTED) for row in rows]
            },
            text=[
                f"{value:.4f}<br>{share:.0%}" for value, share in zip(values, shares)
            ],
            textposition="outside",
            customdata=shares,
            hovertemplate=(
                "%{x}<br>Expected PV %{y:.6f}"
                "<br>%{customdata:.1%} of total value<extra></extra>"
            ),
            showlegend=False,
        )
    )
    figure.update_layout(
        **_layout(
            "Which payments create today's value?",
            subtitle="Bars show expected present value; labels also show each share.",
            height=390,
            y_title="PV per unit notional",
        )
    )
    return figure


def distribution_figure(diagnostics: Mapping[str, Any]) -> go.Figure:
    distribution = diagnostics.get("distribution") or {}
    histogram = distribution.get("histogram") or {}
    edges = [float(value) for value in histogram.get("bin_edges") or []]
    counts = [int(value) for value in histogram.get("counts") or []]
    centers = [(left + right) / 2.0 for left, right in zip(edges[:-1], edges[1:])]
    widths = [right - left for left, right in zip(edges[:-1], edges[1:])]
    total = sum(counts)
    percentages = [count / total * 100.0 if total else 0.0 for count in counts]
    figure = go.Figure(
        go.Bar(
            x=centers,
            y=percentages,
            width=widths,
            marker={
                "color": percentages,
                "colorscale": [
                    [0.0, "rgba(139,92,246,.28)"],
                    [1.0, REFERENCE],
                ],
                "line": {"color": "rgba(248,246,255,.42)", "width": 1},
            },
            customdata=counts,
            hovertemplate=(
                "Payoff around %{x:.4f}<br>%{y:.2f}% of paths"
                "<br>%{customdata:,} pretend futures<extra></extra>"
            ),
        )
    )
    quantiles = {
        float(row["probability"]): float(row["value"])
        for row in distribution.get("quantiles") or []
    }
    for probability, label, color, position in (
        (0.05, "5% tail", RISK, "top left"),
        (0.5, "median", AUTOCALL, "top right"),
    ):
        if probability in quantiles:
            figure.add_vline(
                x=quantiles[probability],
                line_color=color,
                line_dash="dot",
                annotation_text=label,
                annotation_position=position,
            )
    figure.update_layout(
        **_layout(
            "What outcomes did the simulation produce?",
            subtitle="Height is the share of paths; dotted lines mark useful percentiles.",
            height=370,
            x_title="Discounted payoff per unit notional",
            y_title="Share of pretend futures (%)",
        )
    )
    return figure


def surface_figure(diagnostics: Mapping[str, Any]) -> go.Figure:
    surface = diagnostics.get("surface") or {}
    x_values = [float(value) for value in surface.get("spot_shocks_pct") or []]
    y_values = [float(value) for value in surface.get("volatility_shocks_abs") or []]
    cell_rows = [
        cell for cell in surface.get("cells") or [] if cell.get("price") is not None
    ]
    base_row = next(
        (
            cell
            for cell in cell_rows
            if float(cell["spot_shock_pct"]) == 0.0
            and float(cell["volatility_shock_abs"]) == 0.0
        ),
        None,
    )
    base_price = float(base_row["price"]) if base_row is not None else 0.0
    prices = {
        (
            float(cell["volatility_shock_abs"]),
            float(cell["spot_shock_pct"]),
        ): float(cell["price"])
        for cell in cell_rows
    }
    changes = {
        (
            float(cell["volatility_shock_abs"]),
            float(cell["spot_shock_pct"]),
        ): float(cell.get("price_change", float(cell["price"]) - base_price))
        for cell in cell_rows
    }
    z_values = [
        [changes.get((volatility, spot)) for spot in x_values]
        for volatility in y_values
    ]
    price_values = [
        [prices.get((volatility, spot)) for spot in x_values] for volatility in y_values
    ]
    figure = go.Figure(
        go.Heatmap(
            x=x_values,
            y=[value * 100.0 for value in y_values],
            z=z_values,
            colorscale=[
                [0.0, RISK],
                [0.5, "#24213C"],
                [1.0, COUPON],
            ],
            zmid=0.0,
            customdata=price_values,
            text=[
                [f"{price:.4f}" if price is not None else "" for price in row]
                for row in price_values
            ],
            texttemplate="%{text}",
            colorbar={"title": "Change"},
            hovertemplate=(
                "Spot shock %{x:.1f}%<br>Vol shift %{y:.1f} pts"
                "<br>Price %{customdata:.6f}"
                "<br>Change %{z:+.6f}<extra></extra>"
            ),
        )
    )
    figure.update_layout(
        **_layout(
            "What changes the note's value most?",
            subtitle=(
                "Cell text is price. Color is change versus today's base market."
            ),
            height=430,
            x_title="Spot shock (%)",
            y_title="Volatility shift (points)",
        )
    )
    return figure


def scenario_figure(result: Mapping[str, Any]) -> go.Figure:
    base = float(result["base_valuation"]["price"])
    pnl = float(result["pnl"]["value"])
    shocked = float(result["shocked_valuation"]["price"])
    figure = go.Figure(
        go.Waterfall(
            x=["Base value", "Scenario P&L", "Shocked value"],
            y=[base, pnl, shocked],
            measure=["absolute", "relative", "total"],
            connector={"line": {"color": GRID}},
            increasing={"marker": {"color": POSITIVE}},
            decreasing={"marker": {"color": RISK}},
            totals={"marker": {"color": MARKET}},
            text=[f"{base:.5f}", f"{pnl:+.5f}", f"{shocked:.5f}"],
            textposition="outside",
        )
    )
    figure.update_layout(
        **_layout(
            "How did this scenario change the note?",
            subtitle="The middle bar isolates scenario P&L using paired random paths.",
            height=360,
            y_title="PV per unit notional",
        )
    )
    return figure


def risk_figure(result: Mapping[str, Any]) -> go.Figure:
    sensitivities = result.get("sensitivities") or {}
    names = list(sensitivities)
    rows = [sensitivities[name] for name in names]
    resolved = [bool(row.get("statistically_resolved_95pct", False)) for row in rows]
    figure = go.Figure(
        go.Bar(
            x=[name.replace("_", " ").title() for name in names],
            y=[float(row["value"]) for row in rows],
            error_y={
                "type": "data",
                "array": [float(row["standard_error"]) for row in rows],
                "color": INK,
            },
            marker={"color": [POSITIVE if value else MUTED for value in resolved]},
            customdata=[
                [row.get("units", ""), "resolved" if value else "noisy"]
                for row, value in zip(rows, resolved)
            ],
            hovertemplate=(
                "%{x}<br>Value %{y:.6f}<br>%{customdata[0]}"
                "<br>95% signal: %{customdata[1]}<extra></extra>"
            ),
        )
    )
    figure.update_layout(
        **_layout(
            "Which small market nudge matters most?",
            subtitle="Brighter bars clear the 95% simulation-noise test.",
            height=390,
            y_title="Reported sensitivity",
        )
    )
    return figure
