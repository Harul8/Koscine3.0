"""Stage-1 expansion features: compression/breakout/trend structure that
precedes a clean directional move. Added on top of existing features.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=max(2, span // 2)).mean()


def add_compression_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values(["symbol", "date"]).copy()
    grouped = out.groupby("symbol", group_keys=False)
    high = out["high"]
    low = out["low"]
    close = out["close"]

    day_range = (high - low) / close.replace(0, np.nan)
    out["range_pct_today"] = day_range
    # NR7: today's range = smallest in last 7 days
    rolling_min_range = grouped["range_pct_today"].transform(lambda s: s.rolling(7, min_periods=5).min())
    out["nr7_flag"] = (day_range <= rolling_min_range + 1e-9).fillna(False).astype(float)
    # Inside bar: today's range inside yesterday's range
    prev_high = grouped["high"].shift(1)
    prev_low = grouped["low"].shift(1)
    out["inside_bar_flag"] = ((high <= prev_high) & (low >= prev_low)).fillna(False).astype(float)
    out["inside_bar_count_5d"] = grouped["inside_bar_flag"].transform(
        lambda s: s.rolling(5, min_periods=2).sum()
    )
    # ATR/BB percentile at shorter window (60d)
    if "atr_pct_14" in out.columns:
        out["atr_pct_14_rank_60d"] = grouped["atr_pct_14"].transform(
            lambda s: s.rolling(60, min_periods=20).rank(pct=True)
        )
    if "bb_width_20" in out.columns:
        out["bb_width_20_rank_60d"] = grouped["bb_width_20"].transform(
            lambda s: s.rolling(60, min_periods=20).rank(pct=True)
        )
    # Range contraction: today's 5d avg range vs 20d avg range (low = contracting)
    avg_range_5 = grouped["range_pct_today"].transform(lambda s: s.rolling(5, min_periods=3).mean())
    avg_range_20 = grouped["range_pct_today"].transform(lambda s: s.rolling(20, min_periods=10).mean())
    out["range_contraction_5v20"] = (avg_range_5 / avg_range_20.replace(0, np.nan)).clip(0.0, 3.0)
    # Compression composite: blend three ranks (lower = tighter)
    parts = []
    if "atr_pct_14_rank_60d" in out.columns:
        parts.append(1.0 - out["atr_pct_14_rank_60d"])
    if "bb_width_20_rank_60d" in out.columns:
        parts.append(1.0 - out["bb_width_20_rank_60d"])
    if parts:
        out["compression_composite"] = pd.concat(parts, axis=1).mean(axis=1)
    else:
        out["compression_composite"] = np.nan
    return out


def add_trend_structure_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values(["symbol", "date"]).copy()
    grouped = out.groupby("symbol", group_keys=False)
    close = out["close"]
    high = out["high"]
    low = out["low"]

    for span in (5, 10, 20, 50):
        col = f"ema_{span}"
        out[col] = grouped["close"].transform(lambda s, sp=span: _ema(s, sp))
        out[f"{col}_dist"] = close / out[col] - 1.0
    # EMA slope (5d change in ema, normalized)
    for span in (20, 50):
        ema_col = f"ema_{span}"
        out[f"{ema_col}_slope_5d"] = grouped[ema_col].transform(
            lambda s: s.pct_change(5)
        )

    # Higher highs / lower lows count over 10 days
    rolling_max = grouped["high"].transform(lambda s: s.rolling(10, min_periods=5).max())
    rolling_min = grouped["low"].transform(lambda s: s.rolling(10, min_periods=5).min())
    out["new_high_10d"] = (high >= rolling_max).astype(float)
    out["new_low_10d"] = (low <= rolling_min).astype(float)
    out["new_high_count_20d"] = grouped["new_high_10d"].transform(
        lambda s: s.rolling(20, min_periods=10).sum()
    )
    out["new_low_count_20d"] = grouped["new_low_10d"].transform(
        lambda s: s.rolling(20, min_periods=10).sum()
    )

    # ADX-like trend strength (simplified DMI)
    prev_close = grouped["close"].shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    up_move = high - grouped["high"].shift(1)
    down_move = grouped["low"].shift(1) - low
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = pd.Series(plus_dm, index=out.index)
    minus_dm = pd.Series(minus_dm, index=out.index)
    atr14 = grouped.apply(lambda g: tr.loc[g.index].rolling(14, min_periods=7).mean()).reset_index(level=0, drop=True)
    plus_di = 100.0 * grouped.apply(lambda g: plus_dm.loc[g.index].rolling(14, min_periods=7).mean()).reset_index(level=0, drop=True) / atr14.replace(0, np.nan)
    minus_di = 100.0 * grouped.apply(lambda g: minus_dm.loc[g.index].rolling(14, min_periods=7).mean()).reset_index(level=0, drop=True) / atr14.replace(0, np.nan)
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan) * 100.0
    out["adx_14"] = grouped.apply(lambda g: dx.loc[g.index].rolling(14, min_periods=7).mean()).reset_index(level=0, drop=True)
    out["di_diff"] = plus_di - minus_di
    return out


def add_oi_dynamics_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values(["symbol", "date"]).copy()
    if "fut_oi" not in out.columns:
        return out
    grouped = out.groupby("symbol", group_keys=False)
    # OI acceleration: change in (change in OI)
    oi_chg_1d = grouped["fut_oi"].diff(1)
    out["oi_acceleration"] = oi_chg_1d - oi_chg_1d.groupby(out["symbol"]).shift(1)
    # OI z-score 60d (we already have fut_oi_z_60d from tiered but not in base dataset)
    oi_mean = grouped["fut_oi"].transform(lambda s: s.rolling(60, min_periods=20).mean())
    oi_std = grouped["fut_oi"].transform(lambda s: s.rolling(60, min_periods=20).std())
    out["fut_oi_z_60d"] = (out["fut_oi"] - oi_mean) / oi_std.replace(0, np.nan)
    # Price-OI divergence flags
    ret_1d = grouped["close"].pct_change()
    out["price_oi_divergence"] = (
        ((ret_1d > 0.005) & (oi_chg_1d < 0)).astype(float)  # price up, OI dropping = weak rally
        - ((ret_1d < -0.005) & (oi_chg_1d > 0)).astype(float)  # price down, OI rising = aggressive shorts
    )
    return out


def add_volume_dryup_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values(["symbol", "date"]).copy()
    if "volume" not in out.columns:
        return out
    grouped = out.groupby("symbol", group_keys=False)
    vol = out["volume"].replace(0, np.nan)
    vol_5 = grouped["volume"].transform(lambda s: s.rolling(5, min_periods=3).mean())
    vol_20 = grouped["volume"].transform(lambda s: s.rolling(20, min_periods=10).mean())
    out["vol_5v20_ratio"] = (vol_5 / vol_20.replace(0, np.nan)).clip(0.0, 5.0)
    # Volume dryup: today's volume vs 20d average, with low standard deviation
    vol_std_20 = grouped["volume"].transform(lambda s: s.rolling(20, min_periods=10).std())
    out["volume_dryup_score"] = (vol / vol_20.replace(0, np.nan)).clip(0.0, 5.0)
    # Lower = more dried up; we want 1 - normalized
    return out


def add_market_breadth_features(df: pd.DataFrame) -> pd.DataFrame:
    """Computes breadth as % of universe above various SMAs each day."""
    out = df.copy()
    if "close_sma20_dist" not in out.columns or "close_sma50_dist" not in out.columns:
        return out
    by_date = out.groupby("date")
    out["mkt_pct_above_sma20"] = by_date["close_sma20_dist"].transform(lambda s: (s > 0).mean())
    out["mkt_pct_above_sma50"] = by_date["close_sma50_dist"].transform(lambda s: (s > 0).mean())
    # Advance/decline (positive ret_1d count vs negative)
    if "ret_1d" in out.columns:
        out["mkt_advance_ratio"] = by_date["ret_1d"].transform(lambda s: (s > 0).mean())
    return out


def add_calendar_event_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    dates = pd.to_datetime(out["date"])
    # Last Thursday of the month (NSE F&O monthly expiry)
    out["is_expiry_week"] = ((dates + pd.tseries.offsets.MonthEnd(0)).dt.day - dates.dt.day <= 7).astype(float)
    # Distance to next month end
    out["days_to_month_end"] = (
        (dates + pd.tseries.offsets.MonthEnd(0)) - dates
    ).dt.days.clip(lower=0, upper=31).astype(float)
    return out


def enrich_with_expansion_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = add_compression_features(out)
    out = add_trend_structure_features(out)
    out = add_oi_dynamics_features(out)
    out = add_volume_dryup_features(out)
    out = add_market_breadth_features(out)
    out = add_calendar_event_features(out)
    return out.sort_values(["date", "symbol"]).reset_index(drop=True)
