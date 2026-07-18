from typing import Any, Mapping, Sequence

import plotly.graph_objects as go


INK = "#10233F"
MUTED = "#637089"
GRID = "#DDE3EC"
MARKET = "#1E88E5"
REFERENCE = "#6D5BD0"
COUPON = "#00A896"
AUTOCALL = "#F59E0B"
RISK = "#D1495B"
POSITIVE = "#11875D"
PAPER = "#FFFFFF"

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
    height: int = 360,
    x_title: str | None = None,
    y_title: str | None = None,
) -> dict[str, Any]:
    return {
        "title": {"text": title, "x": 0.0, "font": {"size": 17, "color": INK}},
        "height": height,
        "paper_bgcolor": PAPER,
        "plot_bgcolor": PAPER,
        "font": {"family": "Inter, Arial, sans-serif", "color": INK},
        "margin": {"l": 48, "r": 24, "t": 58, "b": 48},
        "hoverlabel": {"bgcolor": INK, "font_color": "white"},
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
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1.0,
        },
    }


def price_uncertainty_figure(result: Mapping[str, Any]) -> go.Figure:
    price = float(result["price"])
    interval = result.get("confidence_interval") or [price, price]
    lower, upper = float(interval[0]), float(interval[1])
    figure = go.Figure(
        go.Scatter(
            x=[price],
            y=["Reference value"],
            mode="markers+text",
            text=[f"{price:.6f}"],
            textposition="top center",
            marker={"size": 15, "color": MARKET},
            error_x={
                "type": "data",
                "symmetric": False,
                "array": [max(upper - price, 0.0)],
                "arrayminus": [max(price - lower, 0.0)],
                "color": INK,
                "thickness": 2,
                "width": 8,
            },
            hovertemplate=(
                "Price %{x:.6f}<br>95% interval "
                f"[{lower:.6f}, {upper:.6f}]<extra></extra>"
            ),
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
            "Reference price and Monte Carlo uncertainty",
            height=300,
            x_title="Present value per unit notional",
        )
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
    figure = go.Figure()
    for item in ordered:
        figure.add_trace(
            go.Scatter(
                x=[float(item["level"])],
                y=[str(item["name"])],
                mode="markers+text",
                text=[f"{float(item['level']):,.2f}"],
                textposition="middle right",
                marker={
                    "size": 15,
                    "color": colors.get(str(item.get("kind")), MUTED),
                    "line": {"color": "white", "width": 2},
                },
                name=str(item["name"]),
                showlegend=False,
                hovertemplate="%{y}: %{x:,.4f}<extra></extra>",
            )
        )
    figure.update_layout(
        **_layout(
            "Contract barrier ladder",
            height=350,
            x_title="Underlier level",
        )
    )
    figure.update_yaxes(showgrid=False)
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
            "Remaining observation timeline",
            height=270,
            x_title="Years from valuation",
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
        ("Risk-free rate", "risk_free_rate", MARKET),
        ("Dividend yield", "dividend_yield", COUPON),
        ("Volatility", "volatility", AUTOCALL),
    )
    figure = go.Figure()
    for label, field, color in series:
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
                hovertemplate=f"{label}: %{{y:.3f}}%<br>t=%{{x:.3f}}y<extra></extra>",
            )
        )
    figure.update_layout(
        **_layout(
            "Deterministic market term structure",
            height=380,
            x_title="Years",
            y_title="Annualized value (%)",
        )
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
            "Monte Carlo convergence",
            height=370,
            x_title="Nested path count",
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
    figure = go.Figure(
        go.Bar(
            x=[
                labels.get(str(row["component"]), str(row["component"])) for row in rows
            ],
            y=[float(row["expected_pv"]) for row in rows],
            error_y={
                "type": "data",
                "array": [float(row["standard_error"]) for row in rows],
                "color": INK,
            },
            marker={
                "color": [colors.get(str(row["component"]), MUTED) for row in rows]
            },
            text=[f"{float(row['expected_pv']):.4f}" for row in rows],
            textposition="outside",
            hovertemplate="%{x}<br>Expected PV %{y:.6f}<extra></extra>",
        )
    )
    figure.update_layout(
        **_layout(
            "Expected present value by cashflow source",
            height=390,
            y_title="PV per unit notional",
        )
    )
    return figure


def distribution_figure(diagnostics: Mapping[str, Any]) -> go.Figure:
    histogram = (diagnostics.get("distribution") or {}).get("histogram") or {}
    edges = [float(value) for value in histogram.get("bin_edges") or []]
    counts = [int(value) for value in histogram.get("counts") or []]
    centers = [(left + right) / 2.0 for left, right in zip(edges[:-1], edges[1:])]
    widths = [right - left for left, right in zip(edges[:-1], edges[1:])]
    figure = go.Figure(
        go.Bar(
            x=centers,
            y=counts,
            width=widths,
            marker={"color": REFERENCE, "line": {"color": PAPER, "width": 1}},
            hovertemplate="Payoff around %{x:.4f}<br>Paths %{y:,}<extra></extra>",
        )
    )
    figure.update_layout(
        **_layout(
            "Discounted payoff distribution",
            height=370,
            x_title="Pathwise discounted payoff",
            y_title="Path count",
        )
    )
    return figure


def surface_figure(diagnostics: Mapping[str, Any]) -> go.Figure:
    surface = diagnostics.get("surface") or {}
    x_values = [float(value) for value in surface.get("spot_shocks_pct") or []]
    y_values = [float(value) for value in surface.get("volatility_shocks_abs") or []]
    cells = {
        (
            float(cell["volatility_shock_abs"]),
            float(cell["spot_shock_pct"]),
        ): float(cell["price"])
        for cell in surface.get("cells") or []
        if cell.get("price") is not None
    }
    z_values = [
        [cells.get((volatility, spot)) for spot in x_values] for volatility in y_values
    ]
    figure = go.Figure(
        go.Heatmap(
            x=x_values,
            y=[value * 100.0 for value in y_values],
            z=z_values,
            colorscale=[
                [0.0, "#DCEBFA"],
                [0.5, "#6D9EEB"],
                [1.0, "#10233F"],
            ],
            colorbar={"title": "PV"},
            hovertemplate=(
                "Spot shock %{x:.1f}%<br>Vol shift %{y:.1f} pts"
                "<br>Price %{z:.6f}<extra></extra>"
            ),
        )
    )
    figure.update_layout(
        **_layout(
            "Spot / volatility valuation surface",
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
            "Paired-path scenario bridge",
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
            "Finite-difference sensitivities",
            height=390,
            y_title="Reported sensitivity",
        )
    )
    return figure
