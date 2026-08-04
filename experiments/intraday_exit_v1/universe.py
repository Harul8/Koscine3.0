"""Locked liquid, high-movement underlying universe for the exit study."""
from __future__ import annotations


# Selected by rank_universe.py from 80 evenly sampled F&O sessions spanning
# 2025-01-01 through 2026-07-22. Score = 65% option liquidity and 35% balanced
# buy-side ATM CE/PE movement. Indices are explicitly included below.
TOP30_STOCKS = (
    "RELIANCE",
    "BSE",
    "BHARTIARTL",
    "ICICIBANK",
    "DIXON",
    "SBIN",
    "BAJFINANCE",
    "TMPV",
    "MARUTI",
    "INFY",
    "AXISBANK",
    "M&M",
    "TRENT",
    "ETERNAL",
    "COALINDIA",
    "VEDL",
    "TCS",
    "ASIANPAINT",
    "RECLTD",
    "HINDUNILVR",
    "HDFCBANK",
    "HAL",
    "ADANIPORTS",
    "PFC",
    "SUNPHARMA",
    "HEROMOTOCO",
    "INDUSINDBK",
    "PERSISTENT",
    "ITC",
    "BAJAJFINSV",
)
INDEX_UNDERLYINGS = ("NIFTY", "BANKNIFTY", "SENSEX", "FINNIFTY")
INTRADAY_UNIVERSE = frozenset((*TOP30_STOCKS, *INDEX_UNDERLYINGS))


def normalize_underlying(value: object) -> str:
    symbol = str(value).strip().upper().replace(" ", "")
    aliases = {
        "NIFTY50": "NIFTY",
        "NIFTYBANK": "BANKNIFTY",
        "NIFTYFINANCIALSERVICES": "FINNIFTY",
    }
    return aliases.get(symbol, symbol)


def require_in_universe(values: object) -> None:
    symbols = {normalize_underlying(value) for value in values}
    outside = sorted(symbols - INTRADAY_UNIVERSE)
    if outside:
        raise ValueError(
            "manifest contains underlyings outside the approved top-30 + four-index scope: "
            f"{outside}"
        )
