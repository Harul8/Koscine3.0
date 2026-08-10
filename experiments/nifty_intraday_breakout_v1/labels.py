"""Direction-agnostic breakout label for the NIFTY intraday breakout model.

For each bar, over horizon H (bars), the forward move magnitude is

    move_mag_H = max(fwd_high_H - entry, entry - fwd_low_H) / entry

mirrors src/koscine3/largemove/mover_v2.py::load_book()'s up_move/down_move/move_mag,
on 5-min OHLC instead of daily. Computed only using bars from the SAME trading day -- a
window that would spill past the session close is left NaN (this model times same-day
option entries, not overnight risk).

The binary breakout flag is NOT baked in here. calibrate_k derives a data-driven
threshold K from a given (training-only) slice, targeting a fixed positive base rate --
mirrors pipeline/labels.py::_calibrate_k. apply_threshold then turns move_mag into a 0/1
label for any slice using that K. These two steps are deliberately split: K must be
calibrated INSIDE each walk-forward fold, from that fold's training rows only, or the
label leaks later volatility regimes into earlier folds.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

HORIZONS = (6, 12, 24)          # 30min / 1hr / 2hr, at 5-min bars
TARGET_RATE = 0.13              # between this repo's LABEL_TARGET_RATE_XL=0.10 and
                                 # LABEL_TARGET_RATE_BASE=0.20 for daily large-move labels
DEFAULT_VOL_COL = "feature_atr_20bar_pct"


def add_forward_extremes(df: pd.DataFrame, horizons: tuple[int, ...] = HORIZONS) -> pd.DataFrame:
    """Adds fwd_move_mag_{H} (raw, continuous, direction-agnostic magnitude) for each H.
    df must already have a `date` column (session date, from features.py) and be sorted
    by timestamp."""
    out = df.copy()
    day = out.groupby("date", sort=False, group_keys=False)
    entry = out["close"]
    for h in horizons:
        highs = pd.concat([day["high"].transform(lambda s, k=k: s.shift(-k)) for k in range(1, h + 1)], axis=1)
        lows = pd.concat([day["low"].transform(lambda s, k=k: s.shift(-k)) for k in range(1, h + 1)], axis=1)
        complete = highs.notna().sum(axis=1).eq(h)   # drop windows that spill past the session close
        fwd_high = highs.max(axis=1)
        fwd_low = lows.min(axis=1)
        up = (fwd_high - entry) / entry
        down = (entry - fwd_low) / entry
        out[f"fwd_move_mag_{h}"] = pd.concat([up, down], axis=1).max(axis=1).where(complete)
    return out


def calibrate_k(train: pd.DataFrame, horizon: int, vol_col: str = DEFAULT_VOL_COL,
                 target_rate: float = TARGET_RATE) -> float:
    """K = (1 - target_rate) percentile of (fwd_move_mag_H / vol_norm), computed on
    TRAINING rows only. thresh = K * vol_norm will then be exceeded by ~target_rate of
    training rows (before any downstream masking)."""
    ratio = train[f"fwd_move_mag_{horizon}"] / train[vol_col]
    ratio = ratio.replace([np.inf, -np.inf], np.nan).dropna()
    if ratio.empty:
        raise ValueError(f"no valid rows to calibrate K for horizon={horizon}")
    return float(np.percentile(ratio, 100.0 * (1.0 - target_rate)))


def apply_threshold(df: pd.DataFrame, horizon: int, k: float, vol_col: str = DEFAULT_VOL_COL) -> pd.Series:
    """Binary breakout label using a K already calibrated elsewhere (see calibrate_k)."""
    thresh = k * df[vol_col]
    mag = df[f"fwd_move_mag_{horizon}"]
    label = (mag >= thresh).astype("float64")
    label[mag.isna() | df[vol_col].isna() | (df[vol_col] <= 0)] = np.nan
    return label.rename(f"target_breakout_{horizon}bar")
