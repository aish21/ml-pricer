from app.ui.charts import (
    barrier_ladder_figure,
    cashflow_figure,
    convergence_figure,
    distribution_figure,
    learning_paths_figure,
    surface_figure,
    term_structure_figure,
)


DIAGNOSTICS = {
    "convergence": [
        {
            "n_paths": 100,
            "price": 0.98,
            "confidence_interval_low": 0.95,
            "confidence_interval_high": 1.01,
        },
        {
            "n_paths": 500,
            "price": 0.99,
            "confidence_interval_low": 0.98,
            "confidence_interval_high": 1.0,
        },
    ],
    "cashflows": {
        "components": [
            {
                "component": "coupon_pv",
                "expected_pv": 0.04,
                "standard_error": 0.001,
            },
            {
                "component": "autocall_principal_pv",
                "expected_pv": 0.7,
                "standard_error": 0.01,
            },
        ]
    },
    "distribution": {
        "quantiles": [
            {"probability": 0.05, "value": 0.7},
            {"probability": 0.5, "value": 0.95},
        ],
        "histogram": {
            "bin_edges": [0.5, 0.75, 1.0],
            "counts": [10, 90],
        },
    },
    "surface": {
        "spot_shocks_pct": [-10.0, 0.0, 10.0],
        "volatility_shocks_abs": [0.0, 0.05],
        "cells": [
            {
                "spot_shock_pct": spot,
                "volatility_shock_abs": volatility,
                "price": 1.0 + spot / 1000.0 - volatility,
                "price_change": spot / 1000.0 - volatility,
            }
            for volatility in (0.0, 0.05)
            for spot in (-10.0, 0.0, 10.0)
        ],
    },
}


def test_quant_charts_build_interactive_traces_from_api_summaries():
    assert len(convergence_figure(DIAGNOSTICS).data) == 2
    assert len(cashflow_figure(DIAGNOSTICS).data) == 1
    distribution = distribution_figure(DIAGNOSTICS)
    assert len(distribution.data) == 1
    assert sum(distribution.data[0].y) == 100.0
    assert len(distribution.layout.shapes) == 2
    surface = surface_figure(DIAGNOSTICS)
    assert len(surface.data) == 1
    assert len(surface.data[0].z) == 2
    assert surface.data[0].z[0][1] == 0.0
    assert surface.data[0].customdata[0][1] == 1.0


def test_contract_and_market_figures_keep_financial_series_separate():
    ladder = barrier_ladder_figure(
        [
            {"name": "Live spot", "level": 90.0, "kind": "market"},
            {"name": "Reference", "level": 100.0, "kind": "reference"},
        ]
    )
    curve = term_structure_figure(
        {
            "segments": [
                {
                    "end_time_years": 0.5,
                    "risk_free_rate": 0.03,
                    "dividend_yield": 0.01,
                    "volatility": 0.2,
                },
                {
                    "end_time_years": 1.0,
                    "risk_free_rate": 0.04,
                    "dividend_yield": 0.012,
                    "volatility": 0.22,
                },
            ]
        }
    )

    assert len(ladder.data) == 2
    assert ladder.layout.yaxis.title.text == "Underlier level"
    assert ladder.layout.xaxis.visible is False
    assert {trace.name for trace in curve.data} == {
        "Risk-free rate",
        "Dividend yield",
        "Volatility",
    }


def test_learning_paths_are_deterministic_and_include_three_rule_lines():
    first = learning_paths_figure(
        volatility_pct=20.0,
        autocall_level=105.0,
        coupon_level=100.0,
        knock_in_level=70.0,
    )
    second = learning_paths_figure(
        volatility_pct=20.0,
        autocall_level=105.0,
        coupon_level=100.0,
        knock_in_level=70.0,
    )

    assert len(first.data) == 12
    assert list(first.data[0].y) == list(second.data[0].y)
    assert len(first.layout.shapes) == 3
