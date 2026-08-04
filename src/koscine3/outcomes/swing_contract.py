from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


SIDES = ("long", "short")
CONTRACT_VERSION = "swing_v1_next_open_5d_peak"


@dataclass(frozen=True)
class SwingContract:
    window_days: int = 5
    near_fraction: float = 0.80


def _base_windows_for_symbol(symbol_df: pd.DataFrame, contract: SwingContract) -> pd.DataFrame:
    g = symbol_df.sort_values("date").reset_index(drop=True).copy()
    required = ["date", "symbol", "open", "high", "low", "close"]
    missing = [c for c in required if c not in g.columns]
    if missing:
        raise ValueError(f"Missing required OHLC columns: {missing}")

    out = g[["date", "symbol"]].copy()
    out["entry_date"] = g["date"].shift(-1)
    out["entry_open"] = g["open"].shift(-1)
    out["window_end_date"] = g["date"].shift(-contract.window_days)
    out["window_close"] = g["close"].shift(-contract.window_days)

    high_shifts = [g["high"].shift(-i) for i in range(1, contract.window_days + 1)]
    low_shifts = [g["low"].shift(-i) for i in range(1, contract.window_days + 1)]
    close_shifts = [g["close"].shift(-i) for i in range(1, contract.window_days + 1)]
    window_highs = pd.concat(high_shifts, axis=1)
    window_lows = pd.concat(low_shifts, axis=1)
    window_closes = pd.concat(close_shifts, axis=1)

    out["window_high"] = window_highs.max(axis=1, skipna=False)
    out["window_low"] = window_lows.min(axis=1, skipna=False)
    out["window_observations"] = window_closes.notna().sum(axis=1)
    return out


def compute_swing_outcomes(
    market_df: pd.DataFrame,
    universe: pd.DataFrame | None = None,
    contract: SwingContract | None = None,
) -> pd.DataFrame:
    contract = contract or SwingContract()
    required = {"date", "symbol", "open", "high", "low", "close"}
    missing = sorted(required - set(market_df.columns))
    if missing:
        raise ValueError(f"Missing required columns for swing outcomes: {missing}")

    prices = market_df[list(required)].copy()
    prices["date"] = pd.to_datetime(prices["date"])
    if universe is not None:
        symbols = set(universe["symbol"].astype(str))
        prices = prices[prices["symbol"].astype(str).isin(symbols)].copy()

    base = pd.concat(
        [_base_windows_for_symbol(g, contract) for _, g in prices.groupby("symbol", sort=False)],
        ignore_index=True,
    )

    if universe is None:
        universe_map = pd.DataFrame(
            {
                "symbol": sorted(base["symbol"].astype(str).unique()),
                "band": "default",
                "threshold": 0.04,
            }
        )
    else:
        universe_map = universe[["symbol", "band", "threshold"]].copy()
        universe_map["symbol"] = universe_map["symbol"].astype(str)

    base["symbol"] = base["symbol"].astype(str)
    base = base.merge(universe_map, on="symbol", how="left")
    if base["threshold"].isna().any():
        missing_symbols = sorted(base.loc[base["threshold"].isna(), "symbol"].unique())
        raise ValueError(f"Missing threshold for symbols: {missing_symbols[:10]}")

    long_df = _side_outcome(base, side="long", contract=contract)
    short_df = _side_outcome(base, side="short", contract=contract)
    return pd.concat([long_df, short_df], ignore_index=True).sort_values(
        ["date", "symbol", "side"]
    )


def _side_outcome(base: pd.DataFrame, side: str, contract: SwingContract) -> pd.DataFrame:
    out = base.copy()
    out["side"] = side
    pending_entry = out["entry_date"].isna() | out["entry_open"].isna()
    pending_window = (
        ~pending_entry
        & (
            out["window_observations"].lt(contract.window_days)
            | out["window_high"].isna()
            | out["window_low"].isna()
            | out["window_close"].isna()
        )
    )
    evaluated = ~(pending_entry | pending_window)

    if side == "long":
        out["favorable_move"] = out["window_high"] / out["entry_open"] - 1.0
        out["signed_close_return"] = out["window_close"] / out["entry_open"] - 1.0
    elif side == "short":
        out["favorable_move"] = out["entry_open"] / out["window_low"] - 1.0
        out["signed_close_return"] = out["entry_open"] / out["window_close"] - 1.0
    else:
        raise ValueError(f"Unsupported side: {side}")

    out["hit"] = evaluated & out["favorable_move"].ge(out["threshold"])
    out["near"] = (
        evaluated
        & ~out["hit"]
        & out["favorable_move"].ge(out["threshold"] * contract.near_fraction)
    )
    out["opposite_close"] = evaluated & ~out["hit"] & out["signed_close_return"].lt(0)
    out["opposite"] = out["opposite_close"] & ~out["near"]
    out["small"] = evaluated & ~out["hit"] & ~out["near"] & ~out["opposite"]
    out["status"] = np.select(
        [pending_entry, pending_window],
        ["pending_entry", "pending_window"],
        default="evaluated",
    )
    out["verdict"] = np.select(
        [
            out["status"].eq("pending_entry"),
            out["status"].eq("pending_window"),
            out["hit"],
            out["near"],
            out["opposite"],
            out["small"],
        ],
        ["pending_entry", "pending_window", "hit", "near", "opposite", "small"],
        default="unknown",
    )
    return out
