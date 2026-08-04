"""Clean-move outcome contract (Koscine 3.0 v2 objective).

Replaces the fixed-threshold hit/near contract for the new modeling track. The v19
swing_contract pipeline is locked and untouched; this is a separate, parallel module.

Trade definition
----------------
- Signal EOD t, enter at t+1 open, evaluate the 5-trading-day window.
- A single FIXED stop is placed at ``entry_open * (1 - atr_mult * ATR%)`` for a long
  (mirrored above entry for a short). It is never trailed. The trade is "clean" iff the
  window's worst adverse excursion never breaches that stop -- which is exactly testable
  from daily lows/highs, sidestepping intraday high-vs-low ordering.

Targets / outputs (per symbol, date, side)
------------------------------------------
- ``clean``    : binary, the stop was never hit (model classification target)
- ``ceiling``  : continuous favourable peak vs entry_open (model regression target)
- ``floor_depth``: worst adverse excursion vs entry_open (stop pressure)
- ``atr_pct``  : ATR(14)/close at signal day t (drives the stop width)
- diagnostics for offline exit management (NOT targets):
  ``days_to_peak`` (window day 1-5 of the favourable peak) and
  ``reaches_big_by_day`` (first window day the favourable move >= big_move_threshold).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


SIDES = ("long", "short")
CONTRACT_VERSION = "clean_move_v1_next_open_5d_atr_stop"


@dataclass(frozen=True)
class CleanMoveContract:
    window_days: int = 5
    atr_window: int = 14
    atr_mult: float = 0.60
    big_move_threshold: float = 0.05


def _true_range(g: pd.DataFrame) -> pd.Series:
    prev_close = g["close"].shift(1)
    return pd.concat(
        [
            (g["high"] - g["low"]).abs(),
            (g["high"] - prev_close).abs(),
            (g["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)


def _base_windows_for_symbol(symbol_df: pd.DataFrame, contract: CleanMoveContract) -> pd.DataFrame:
    g = symbol_df.sort_values("date").reset_index(drop=True).copy()
    required = ["date", "symbol", "open", "high", "low", "close"]
    missing = [c for c in required if c not in g.columns]
    if missing:
        raise ValueError(f"Missing required OHLC columns: {missing}")

    w = contract.window_days
    out = g[["date", "symbol"]].copy()
    out["entry_date"] = g["date"].shift(-1)
    out["entry_open"] = g["open"].shift(-1)
    out["window_end_date"] = g["date"].shift(-w)

    high_cols = {f"d{i}": g["high"].shift(-i) for i in range(1, w + 1)}
    low_cols = {f"d{i}": g["low"].shift(-i) for i in range(1, w + 1)}
    close_shifts = [g["close"].shift(-i) for i in range(1, w + 1)]
    highs = pd.DataFrame(high_cols)
    lows = pd.DataFrame(low_cols)

    out["window_high"] = highs.max(axis=1, skipna=False)
    out["window_low"] = lows.min(axis=1, skipna=False)
    out["window_observations"] = pd.concat(close_shifts, axis=1).notna().sum(axis=1)

    # ATR(14) as % of close, known at signal day t.
    atr = _true_range(g).rolling(contract.atr_window, min_periods=contract.atr_window).mean()
    out["atr_pct"] = atr / g["close"]

    # Attach the raw per-day forward highs/lows for diagnostics (days_to_peak / reaches_big).
    for i in range(1, w + 1):
        out[f"_h{i}"] = high_cols[f"d{i}"]
        out[f"_l{i}"] = low_cols[f"d{i}"]
    return out


def compute_clean_move_outcomes(
    market_df: pd.DataFrame,
    universe: pd.DataFrame | None = None,
    contract: CleanMoveContract | None = None,
) -> pd.DataFrame:
    contract = contract or CleanMoveContract()
    required = {"date", "symbol", "open", "high", "low", "close"}
    missing = sorted(required - set(market_df.columns))
    if missing:
        raise ValueError(f"Missing required columns for clean-move outcomes: {missing}")

    prices = market_df[list(required)].copy()
    prices["date"] = pd.to_datetime(prices["date"])
    if universe is not None:
        symbols = set(universe["symbol"].astype(str))
        prices = prices[prices["symbol"].astype(str).isin(symbols)].copy()

    base = pd.concat(
        [_base_windows_for_symbol(g, contract) for _, g in prices.groupby("symbol", sort=False)],
        ignore_index=True,
    )
    base["symbol"] = base["symbol"].astype(str)

    if universe is not None and "band" in universe.columns:
        band_map = universe[["symbol", "band"]].copy()
        band_map["symbol"] = band_map["symbol"].astype(str)
        base = base.merge(band_map, on="symbol", how="left")
        base["band"] = base["band"].fillna("default")
    else:
        base["band"] = "default"

    long_df = _side_outcome(base, side="long", contract=contract)
    short_df = _side_outcome(base, side="short", contract=contract)
    result = pd.concat([long_df, short_df], ignore_index=True)
    return result.sort_values(["date", "symbol", "side"]).reset_index(drop=True)


def _first_true_day(bool_frame: pd.DataFrame) -> pd.Series:
    """First column index (1-based) that is True per row; NaN if no True."""
    has_any = bool_frame.any(axis=1)
    vals = bool_frame.to_numpy()
    first = np.where(has_any.to_numpy(), vals.argmax(axis=1) + 1, np.nan)
    return pd.Series(first, index=bool_frame.index)


def _extreme_day(frame: pd.DataFrame, how: str) -> pd.Series:
    """1-based window day of the row max/min, ignoring NaN; NaN for all-NaN rows."""
    vals = frame.to_numpy(dtype=float)
    has = ~np.isnan(vals).all(axis=1)
    out = np.full(len(frame), np.nan)
    if has.any():
        picker = np.nanargmax if how == "max" else np.nanargmin
        out[has] = picker(vals[has], axis=1) + 1
    return pd.Series(out, index=frame.index)


def _side_outcome(base: pd.DataFrame, side: str, contract: CleanMoveContract) -> pd.DataFrame:
    w = contract.window_days
    out = base.drop(columns=[c for c in base.columns if c.startswith(("_h", "_l"))]).copy()
    out["side"] = side
    out["contract_version"] = CONTRACT_VERSION

    day_labels = [f"d{i}" for i in range(1, w + 1)]
    highs = base[[f"_h{i}" for i in range(1, w + 1)]].copy()
    highs.columns = day_labels
    lows = base[[f"_l{i}" for i in range(1, w + 1)]].copy()
    lows.columns = day_labels

    entry = base["entry_open"]
    pending_entry = base["entry_date"].isna() | entry.isna()
    pending_window = (
        ~pending_entry
        & (
            base["window_observations"].lt(w)
            | base["window_high"].isna()
            | base["window_low"].isna()
            | base["atr_pct"].isna()
        )
    )
    evaluated = ~(pending_entry | pending_window)

    if side == "long":
        out["floor_depth"] = (entry - base["window_low"]) / entry
        out["ceiling"] = (base["window_high"] - entry) / entry
        gains = highs.sub(entry, axis=0).div(entry, axis=0)          # favourable = up
        peak_day = _extreme_day(highs, "max")                         # day of max high
    else:
        out["floor_depth"] = (base["window_high"] - entry) / entry
        out["ceiling"] = (entry - base["window_low"]) / entry
        gains = lows.rsub(entry, axis=0).div(entry, axis=0)           # favourable = down
        peak_day = _extreme_day(lows, "min")                          # day of min low

    out["atr_pct"] = base["atr_pct"]
    stop_tol = contract.atr_mult * base["atr_pct"]
    out["stop_tol"] = stop_tol
    out["clean"] = (evaluated & out["floor_depth"].le(stop_tol)).where(evaluated, other=False)
    out["favorable_move"] = out["ceiling"]  # alias for downstream familiarity

    reaches = gains.ge(contract.big_move_threshold)
    out["days_to_peak"] = peak_day.where(evaluated, other=np.nan)
    out["reaches_big_by_day"] = _first_true_day(reaches).where(evaluated, other=np.nan)
    out["reaches_big"] = (evaluated & reaches.any(axis=1)).where(evaluated, other=False)

    out["status"] = np.select(
        [pending_entry, pending_window], ["pending_entry", "pending_window"], default="evaluated"
    )
    out["verdict"] = np.select(
        [
            out["status"].eq("pending_entry"),
            out["status"].eq("pending_window"),
            out["clean"],
        ],
        ["pending_entry", "pending_window", "clean"],
        default="stopped",
    )
    # Null out targets where not evaluated to avoid accidental training leakage.
    for col in ["floor_depth", "ceiling", "favorable_move"]:
        out.loc[~evaluated, col] = np.nan
    return out
