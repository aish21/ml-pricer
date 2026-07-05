from app.bb.rendering import (
    compact_run_id,
    compact_timestamp,
    format_number,
    format_percent,
    format_shock,
    product_rows,
    run_price,
    terminal_error,
)


def test_compact_run_id_uses_suffix_without_changing_storage_id():
    assert compact_run_id("run_20260704-072247_c23ce8") == "c23ce8"
    assert compact_run_id("short") == "short"


def test_compact_timestamp_formats_iso_time():
    assert compact_timestamp("2026-07-04T07:22:47+00:00") == "07:22"
    assert compact_timestamp("") == "N/A"


def test_numeric_format_helpers_are_terminal_safe():
    assert format_number(0.9849451) == "0.984945"
    assert format_number(None) == "N/A"
    assert format_percent(-5.4321) == "-5.43%"
    assert format_shock(50, "bp") == "+50bp"


def test_run_price_selects_price_or_shocked_price():
    price_run = {"run_type": "price", "result_payload": {"price": 0.98}}
    scenario_run = {"run_type": "scenario", "result_payload": {"shocked_price": 0.91}}

    assert run_price(price_run) == "0.980000"
    assert run_price(scenario_run) == "0.910000"


def test_product_rows_show_terminal_cache_states():
    rows = product_rows(
        [
            {
                "key": "phoenix",
                "terminal_label": "PHOENIX",
                "enabled_for_bb": True,
                "artifacts": {"ready_for_surrogate": True},
            },
            {
                "key": "barrier",
                "terminal_label": "BARRIER",
                "enabled_for_bb": True,
                "artifacts": {"ready_for_surrogate": False},
            },
        ],
        {"phoenix": True},
    )

    assert "PHOENIX" in rows
    assert "READY" in rows
    assert "CACHED" in rows
    assert "BARRIER" in rows
    assert "UNAVAIL" in rows


def test_terminal_error_escapes_reason_and_omits_tracebacks():
    response = terminal_error("<bad>", "/bb/price", "BACK")
    body = response.body.decode("utf-8")

    assert "&lt;bad&gt;" in body
    assert "[1] BACK" in body
    assert "Traceback" not in body
