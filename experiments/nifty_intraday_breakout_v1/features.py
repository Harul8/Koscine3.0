"""Leakage-safe price-only 5-min feature builder for the NIFTY intraday breakout model.

Reads data/intraday/nifty_5m.parquet (native 5-min NIFTY 50 index candles -- no real
volume/OI, the raw index has neither). All rolling windows reset at the session boundary
(grouped by trading date) so an overnight gap never contaminates an intraday realized-vol
or momentum estimate -- the gap itself is captured separately as its own feature.

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

    day = df.groupby("date", sort=False, group_keys=False)
    ret1 = day["close"].transform(lambda s: s.pct_change())
    df["feature_return_1bar"] = ret1
    df["feature_return_3bar"] = day["close"].transform(lambda s: s.pct_change(3))
    df["feature_return_6bar"] = day["close"].transform(lambda s: s.pct_change(6))
    df["feature_return_12bar"] = day["close"].transform(lambda s: s.pct_change(12))
    df["feature_realized_vol_6bar"] = ret1.groupby(df["date"]).transform(lambda s: s.rolling(6, min_periods=3).std())
    df["feature_realized_vol_12bar"] = ret1.groupby(df["date"]).transform(lambda s: s.rolling(12, min_periods=6).std())
    df["feature_realized_vol_24bar"] = ret1.groupby(df["date"]).transform(lambda s: s.rolling(24, min_periods=12).std())

    # True Range %, session-scoped -- same formula as
    # src/koscine3/outcomes/clean_move_contract.py::_true_range / pipeline/labels.py::_compute_atr,
    # applied to 5-min bars instead of daily.
    prev_close = day["close"].transform(lambda s: s.shift(1))
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    df["feature_true_range_pct"] = tr / df["close"]
    df["feature_atr_20bar_pct"] = df["feature_true_range_pct"].groupby(df["date"]).transform(
        lambda s: s.rolling(20, min_periods=10).mean())
    df["feature_range_compression"] = (
        df["feature_true_range_pct"] / df["feature_atr_20bar_pct"].replace(0, np.nan))

    # overnight gap: today's open vs the PRIOR session's close, held constant through the day
    day_summary = df.groupby("date", sort=False).agg(day_open=("open", "first"), day_close=("close", "last"))
    day_summary["prev_day_close"] = day_summary["day_close"].shift(1)
    day_summary["gap_pct"] = (day_summary["day_open"] - day_summary["prev_day_close"]) / day_summary["prev_day_close"]
    df = df.merge(day_summary[["gap_pct"]].rename(columns={"gap_pct": "feature_gap_from_prev_close"}),
                   left_on="date", right_index=True, how="left")

    return df
