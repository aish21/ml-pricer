from dataclasses import dataclass

import streamlit as st


@dataclass(frozen=True)
class UnderlierOption:
    symbol: str
    name: str
    underlier_type: str
    plain_english: str

    @property
    def label(self) -> str:
        return f"{self.symbol} — {self.name}"


UNDERLIER_CATALOG: tuple[UnderlierOption, ...] = (
    UnderlierOption(
        "SPY",
        "S&P 500 ETF",
        "etf",
        "A basket holding shares in many large U.S. companies.",
    ),
    UnderlierOption(
        "QQQ",
        "Nasdaq-100 ETF",
        "etf",
        "A basket tilted toward large technology companies.",
    ),
    UnderlierOption(
        "IWM",
        "Russell 2000 ETF",
        "etf",
        "A basket of smaller U.S. companies.",
    ),
    UnderlierOption(
        "DIA",
        "Dow Jones ETF",
        "etf",
        "A basket following 30 established U.S. companies.",
    ),
    UnderlierOption(
        "EFA",
        "Developed Markets ETF",
        "etf",
        "A basket of companies outside the U.S. and Canada.",
    ),
    UnderlierOption(
        "EEM",
        "Emerging Markets ETF",
        "etf",
        "A basket of companies in developing economies.",
    ),
    UnderlierOption(
        "GLD",
        "Gold ETF",
        "etf",
        "A fund designed to follow the price of gold.",
    ),
    UnderlierOption(
        "TLT",
        "Long Treasury Bond ETF",
        "etf",
        "A basket of long-dated U.S. government bonds.",
    ),
    UnderlierOption(
        "XLK",
        "Technology Sector ETF",
        "etf",
        "A basket of U.S. technology companies.",
    ),
    UnderlierOption(
        "XLF",
        "Financial Sector ETF",
        "etf",
        "A basket of U.S. banks and financial companies.",
    ),
    UnderlierOption(
        "AAPL",
        "Apple",
        "equity",
        "One company: Apple.",
    ),
    UnderlierOption(
        "MSFT",
        "Microsoft",
        "equity",
        "One company: Microsoft.",
    ),
    UnderlierOption(
        "NVDA",
        "NVIDIA",
        "equity",
        "One company: NVIDIA.",
    ),
    UnderlierOption(
        "AMZN",
        "Amazon",
        "equity",
        "One company: Amazon.",
    ),
    UnderlierOption(
        "GOOGL",
        "Alphabet",
        "equity",
        "One company: Alphabet, Google's parent.",
    ),
    UnderlierOption(
        "META",
        "Meta Platforms",
        "equity",
        "One company: Meta Platforms.",
    ),
    UnderlierOption(
        "TSLA",
        "Tesla",
        "equity",
        "One company: Tesla.",
    ),
    UnderlierOption(
        "JPM",
        "JPMorgan Chase",
        "equity",
        "One company: JPMorgan Chase.",
    ),
    UnderlierOption(
        "JNJ",
        "Johnson & Johnson",
        "equity",
        "One company: Johnson & Johnson.",
    ),
    UnderlierOption(
        "XOM",
        "Exxon Mobil",
        "equity",
        "One company: Exxon Mobil.",
    ),
    UnderlierOption(
        "^SPX",
        "S&P 500 Index",
        "index",
        "A number measuring a basket of large U.S. companies.",
    ),
    UnderlierOption(
        "^NDX",
        "Nasdaq-100 Index",
        "index",
        "A number measuring 100 large Nasdaq-listed companies.",
    ),
)


def available_underliers(market_source: str) -> tuple[UnderlierOption, ...]:
    if market_source == "Research market":
        return tuple(
            item for item in UNDERLIER_CATALOG if item.underlier_type != "index"
        )
    return UNDERLIER_CATALOG


def underlier_by_symbol(symbol: str) -> UnderlierOption | None:
    normalized = symbol.strip().upper()
    return next(
        (item for item in UNDERLIER_CATALOG if item.symbol == normalized),
        None,
    )


def parse_underlier_selection(
    selection: str,
    *,
    market_source: str,
) -> tuple[str, UnderlierOption | None]:
    """Resolve either a catalog label or a newly typed Yahoo-style symbol."""
    typed = selection.strip()
    choices = available_underliers(market_source)
    by_label = {item.label.casefold(): item for item in choices}
    matched_label = by_label.get(typed.casefold())
    if matched_label is not None:
        return matched_label.symbol, matched_label

    symbol = typed.split("—", 1)[0].strip().upper()
    matched_symbol = next(
        (item for item in choices if item.symbol == symbol),
        None,
    )
    return symbol, matched_symbol


def render_underlier_picker(
    *,
    market_source: str,
    key_prefix: str,
    beginner_language: bool = False,
) -> tuple[str, str]:
    choices = available_underliers(market_source)
    labels = [item.label for item in choices]
    label = (
        "Pick the thing whose price we will watch"
        if beginner_language
        else "Search underlier"
    )
    selected = st.selectbox(
        label,
        labels,
        index=0,
        key=f"{key_prefix}_symbol_search",
        accept_new_options=True,
        filter_mode="contains",
        placeholder="Type a symbol or name, then choose or press Enter",
        help=(
            "Type directly in this box. Matching symbols and names appear below. "
            "If nothing matches, enter a Yahoo Finance symbol and press Enter."
        ),
    )
    symbol, option = parse_underlier_selection(
        str(selected or ""),
        market_source=market_source,
    )
    if option is not None:
        st.caption(option.plain_english)
        return option.symbol, option.underlier_type

    if not symbol:
        return "", "equity"
    st.caption(
        f"**{symbol}** is not in the teaching list, so tell us what kind of "
        "thing it represents."
    )
    type_options = ["Equity", "ETF"]
    if market_source != "Research market":
        type_options.append("Index")
    underlier_type = st.selectbox(
        "What kind of underlier is this?",
        type_options,
        key=f"{key_prefix}_custom_type",
    )
    return symbol, underlier_type.lower()
