"""Leakage-safe price-only 5-min feature builder for the NIFTY intraday breakout model.

Reads data/intraday/nifty_5m.parquet (native 5-min NIFTY 50 index candles -- no real
volume/OI, the raw index has neither).

Rolling return/vol/ATR windows are CONTINUOUS across the session boundary (2026-08 change,
explicit user decision -- previously grouped/reset per trading date). Resetting them at every
day boundary meant the first ~10-20 bars of EVERY session had NaN atr_20bar_pct/realized_vol_*
(rolling min_periods not yet satisfied within that day alone), which dropna()-based training
(dataset.py/finalize_magnitude.py) silently excluded from the training set entirely -- the
model had never seen an early-session feature vector, so live scoring couldn't start until
~25 same-day bars accumulated (~11:20 IST) even though the underlying true-range/volatility
math is well-defined using the prior session's tail bars; nothing about it requires resetting
at 09:15. `feature_gap_from_prev_close` and the opening-range features below are still
deliberately day-scoped -- those represent the discrete overnight jump and each day's own
specific opening window, not a rolling estimate, so they're supposed to reset.

Column convention (matches experiments/intraday_exit_v1's leakage discipline): every
backward-looking column here is prefixed `feature_`. This module produces ONLY features --
forward-looking targets live in labels.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SESSION_START_MIN = 9 * 60 + 15   # 09:15 IST
SESSION_LEN_MIN = 375              # 09:15 -> 15:30


def build_price_features(bars: pd.DataFrame) -> pd.DataFrame:
    """bars: nifty_5m.parquet columns (timestamp, open, high, low, close, volume, oi),
    timestamp tz-aware Asia/Kolkata. Returns bars + a `date` column + feature_* columns,
    one row per bar."""
    df = bars.sort_values("timestamp").reset_index(drop=True).copy()
    df["date"] = df["timestamp"].dt.date

    minutes = df["timestamp"].dt.hour * 60 + df["timestamp"].dt.minute
    df["feature_session_fraction"] = (minutes - SESSION_START_MIN) / SESSION_LEN_MIN
    df["feature_day_of_week"] = df["timestamp"].dt.dayofweek

    # group_keys=False groupby kept ONLY for the day-scoped sections below (opening-range,
    # gap, cumcount) -- the rolling return/vol/ATR features intentionally do NOT use it, see
    # the module docstring.
    day = df.groupby("date", sort=False, group_keys=False)
    ret1 = df["close"].pct_change()
    df["feature_return_1bar"] = ret1
    df["feature_return_3bar"] = df["close"].pct_change(3)
    df["feature_return_6bar"] = df["close"].pct_change(6)
    df["feature_return_12bar"] = df["close"].pct_change(12)
    df["feature_realized_vol_6bar"] = ret1.rolling(6, min_periods=3).std()
    df["feature_realized_vol_12bar"] = ret1.rolling(12, min_periods=6).std()
    df["feature_realized_vol_24bar"] = ret1.rolling(24, min_periods=12).std()

    # True Range %, continuous across sessions -- same formula as
    # src/koscine3/outcomes/clean_move_contract.py::_true_range / pipeline/labels.py::_compute_atr,
    # applied to 5-min bars instead of daily. For the first bar of a new day, prev_close is now
    # the prior session's actual last close (not NaN'd out by a day-groupby shift), so true range
    # correctly reflects the overnight gap itself for that bar instead of silently understating
    # it to just that bar's own high-low.
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    df["feature_true_range_pct"] = tr / df["close"]
    df["feature_atr_20bar_pct"] = df["feature_true_range_pct"].rolling(20, min_periods=10).mean()
    df["feature_range_compression"] = (
        df["feature_true_range_pct"] / df["feature_atr_20bar_pct"].replace(0, np.nan))

    # overnight gap: today's open vs the PRIOR session's close, held constant through the day
    day_summary = df.groupby("date", sort=False).agg(day_open=("open", "first"), day_close=("close", "last"))
    day_summary["prev_day_close"] = day_summary["day_close"].shift(1)
    day_summary["gap_pct"] = (day_summary["day_open"] - day_summary["prev_day_close"]) / day_summary["prev_day_close"]
    df = df.merge(day_summary[["gap_pct"]].rename(columns={"gap_pct": "feature_gap_from_prev_close"}),
                   left_on="date", right_index=True, how="left")

    # Opening-range width: the first 30 min's (6 bars) high-low range, as % of price -- 2026-08
    # addition, individually AUC 0.616 for predicting the forward-30min busy/quiet regime (vs
    # 0.65-0.66 for the rolling realized-vol features, 0.50-0.53 for everything else already
    # above -- see the NIFTY 5-min leading-indicator research this session). A wide opening range
    # tends to precede a wider move for the rest of the day (vol persistence), and it's NOT
    # redundant with the rolling-vol features since it's fixed at the open rather than
    # continuously updating. Leakage-safe: an EXPANDING high/low within the 30-min window (so a
    # 9:20 bar only sees 9:15-9:20, not the full eventual 9:15-9:45 range), frozen at its final
    # (fully-formed) value once the window closes -- never uses information from bars ahead of
    # the row it's attached to.
    min_since_open = day.cumcount()
    in_opening_window = min_since_open < 6
    running_high = df["high"].where(in_opening_window).groupby(df["date"]).cummax()
    running_low = df["low"].where(in_opening_window).groupby(df["date"]).cummin()
    or_high = running_high.groupby(df["date"]).ffill()
    or_low = running_low.groupby(df["date"]).ffill()
    # Normalize by the day's OPEN price (frozen, from day_summary above), not the current bar's
    # close -- dividing by a continuously-moving close made this "frozen" range drift all through
    # the day even after the opening window closed (caught in testing: values kept changing right
    # up to 15:25 instead of freezing at ~09:45).
    df = df.merge(day_summary[["day_open"]], left_on="date", right_index=True, how="left")
    df["feature_opening_range_width_pct"] = (or_high - or_low) / df["day_open"]
    df = df.drop(columns=["day_open"])

    return df
