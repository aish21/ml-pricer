from app.ui.underliers import (
    UNDERLIER_CATALOG,
    available_underliers,
    parse_underlier_selection,
    underlier_by_symbol,
)


def test_underlier_catalog_has_unique_searchable_symbols():
    symbols = [item.symbol for item in UNDERLIER_CATALOG]

    assert len(symbols) == len(set(symbols))
    assert underlier_by_symbol(" spy ").name == "S&P 500 ETF"
    assert underlier_by_symbol("missing") is None


def test_research_picker_excludes_indices_but_manual_picker_keeps_them():
    research_types = {
        item.underlier_type for item in available_underliers("Research market")
    }
    manual_types = {
        item.underlier_type for item in available_underliers("Manual flat market")
    }

    assert "index" not in research_types
    assert "index" in manual_types


def test_picker_resolves_catalog_labels_and_new_symbols_from_the_same_box():
    symbol, option = parse_underlier_selection(
        "AAPL — Apple",
        market_source="Research market",
    )
    assert symbol == "AAPL"
    assert option is not None
    assert option.underlier_type == "equity"

    symbol, option = parse_underlier_selection(
        " nflx ",
        market_source="Research market",
    )
    assert symbol == "NFLX"
    assert option is None
