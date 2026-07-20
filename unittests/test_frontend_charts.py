from app.ui.charts import (
    audit_error_heatmap_figure,
    audit_slice_figure,
    autocall_schedule_figure,
    barrier_ladder_figure,
    cashflow_figure,
    convergence_figure,
    distribution_figure,
    learning_paths_figure,
    latency_comparison_figure,
    shadow_error_history_figure,
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

    schedule = autocall_schedule_figure(
        [0.25, 0.5, 1.0],
        [1.10, 1.05, 1.0],
        reference_level=100.0,
        coupon_barrier_frac=0.8,
    )
    assert [round(value, 8) for value in schedule.data[0].y] == [
        110.0,
        105.0,
        100.0,
    ]
    assert len(schedule.layout.shapes) == 1


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


def test_evidence_figures_compare_audit_slices_and_live_shadow_history():
    audit = {
        "sealed_audit": {
            "by_market_regime": {
                "low_vol": {
                    "n_samples": 20,
                    "mae": 0.008,
                    "p95_absolute_error": 0.02,
                },
                "normal": {
                    "n_samples": 20,
                    "mae": 0.005,
                    "p95_absolute_error": 0.01,
                },
            },
            "by_regime_and_moneyness": {
                "low_vol:knock_in": {"mae": 0.009},
                "normal:coupon": {"mae": 0.004},
            },
        }
    }
    series = {
        "observations": [
            {
                "created_at": "2026-07-19T10:00:00+00:00",
                "symbol": "SPY",
                "status": "success",
                "absolute_error": 0.006,
                "error_to_reference_standard_error": 1.2,
                "market_regime": "normal",
                "moneyness_region": "coupon",
                "reference_latency_ms": 100.0,
                "latency_ms": 4.0,
                "speedup": 25.0,
            }
        ]
    }

    assert len(audit_slice_figure(audit).data) == 1
    heatmap = audit_error_heatmap_figure(audit)
    assert heatmap.data[0].z[0][0] == 0.009
    assert len(shadow_error_history_figure(series).data) == 1
    assert len(latency_comparison_figure(series).data) == 2
