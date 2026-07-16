from typing import Any, Mapping


def frozen_term_structure_from_pricing_result(
    result: Mapping[str, Any], maturity_years: float
) -> dict[str, Any] | None:
    """Return the exact priced curve, or lift a flat snapshot into one segment."""
    if not isinstance(result, Mapping):
        return None
    term_structure = result.get("market_term_structure")
    if isinstance(term_structure, Mapping):
        required_curve_fields = {
            "schema_version",
            "symbol",
            "underlier_type",
            "currency",
            "valuation_time",
            "market_data_time",
            "spot",
            "segments",
            "calendar",
            "day_count",
            "source",
        }
        if not required_curve_fields.issubset(term_structure):
            return None
        return {name: term_structure[name] for name in required_curve_fields}

    snapshot = result.get("market_snapshot")
    if not isinstance(snapshot, Mapping):
        return None
    required = {
        "symbol",
        "underlier_type",
        "currency",
        "valuation_time",
        "market_data_time",
        "spot",
        "risk_free_rate",
        "dividend_yield",
        "volatility",
        "calendar",
        "day_count",
        "source",
    }
    if not required.issubset(snapshot):
        return None
    try:
        maturity = float(maturity_years)
    except (TypeError, ValueError):
        return None
    if maturity <= 0.0:
        return None
    return {
        "schema_version": "equity-market-term-structure-v1",
        "symbol": snapshot["symbol"],
        "underlier_type": snapshot["underlier_type"],
        "currency": snapshot["currency"],
        "valuation_time": snapshot["valuation_time"],
        "market_data_time": snapshot["market_data_time"],
        "spot": snapshot["spot"],
        "segments": [
            {
                "end_time_years": maturity,
                "risk_free_rate": snapshot["risk_free_rate"],
                "dividend_yield": snapshot["dividend_yield"],
                "volatility": snapshot["volatility"],
            }
        ],
        "calendar": snapshot["calendar"],
        "day_count": snapshot["day_count"],
        "source": snapshot["source"],
    }


def compact_nonzero_shock(
    *,
    spot_pct: float,
    rate_parallel_bps: float,
    dividend_parallel_bps: float,
    volatility_parallel_abs: float,
    segment_shock: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a concise API shock payload from Streamlit controls."""
    shock: dict[str, Any] = {}
    for name, value in {
        "spot_pct": spot_pct,
        "rate_parallel_bps": rate_parallel_bps,
        "dividend_parallel_bps": dividend_parallel_bps,
        "volatility_parallel_abs": volatility_parallel_abs,
    }.items():
        numeric = float(value)
        if numeric != 0.0:
            shock[name] = numeric
    if segment_shock:
        item = {"segment_index": int(segment_shock["segment_index"])}
        for name in ("rate_bps", "dividend_bps", "volatility_abs"):
            numeric = float(segment_shock.get(name, 0.0))
            if numeric != 0.0:
                item[name] = numeric
        if len(item) > 1:
            shock["segment_shocks"] = [item]
    return shock
