"""
Feature builder — ~90 features in 9 groups + compound interactions.

See ARCHITECTURE.md §4 for the full design. Each `_group_*` function returns a
DataFrame keyed on (date, symbol) that is merged into the final feature panel.

Output: gold/features.parquet  — columns:
    date, symbol, <feature_1>, <feature_2>, ..., <feature_N>

Note: labels are NOT included. Join with gold/labels.parquet in train/evaluate.

Run:
    python -m pipeline.features              # full build
    python -m pipeline.features --no-zones   # skip zone features (much faster)
"""
from __future__ import annotations
import sys
import warnings
from typing import Optional

import numpy as np
import pandas as pd

import pickle

from .config import (
    SILVER_TABLES, GOLD_FEATURES, GOLD_DIR, GOLD_ZONES,
    TICKER_SECTOR, SECTOR_INDICES,
    TICKER_NSDL_SECTOR, FII_SECTOR_PUBLICATION_LAG_DAYS,
    MODEL_DIR, ZONE_CLUSTER_PCT,
    LIQUID_TIER_SIZE,
)
from . import zones as _zones

warnings.simplefilter("ignore", category=RuntimeWarning)
GOLD_DIR.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────
def _rolling_atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=5).mean()


def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(n, min_periods=n // 2).mean()
    loss = (-delta.clip(upper=0)).rolling(n, min_periods=n // 2).mean()
    rs   = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _macd_hist(close: pd.Series) -> pd.Series:
    ema_fast = close.ewm(span=12, adjust=False).mean()
    ema_slow = close.ewm(span=26, adjust=False).mean()
    macd     = ema_fast - ema_slow
    signal   = macd.ewm(span=9, adjust=False).mean()
    return macd - signal


def _pct_rank(series: pd.Series, window: int) -> pd.Series:
    """Percentile rank of last value vs the rolling window."""
    return series.rolling(window, min_periods=max(20, window // 5)).apply(
        lambda x: (x.iloc[-1] > x.iloc[:-1]).mean() if len(x) > 1 else np.nan,
        raw=False,
    )


# ──────────────────────────────────────────────────────────────────────
# Group 1 — Returns & Momentum
# ──────────────────────────────────────────────────────────────────────
def _group_returns_momentum(stock: pd.DataFrame) -> pd.DataFrame:
    g = stock.groupby("symbol", sort=False, group_keys=False)
    out = pd.DataFrame(index=stock.index)
    out["ret_1d"]  = g["close"].pct_change(1)
    out["ret_3d"]  = g["close"].pct_change(3)
    out["ret_5d"]  = g["close"].pct_change(5)
    out["ret_10d"] = g["close"].pct_change(10)
    out["ret_20d"] = g["close"].pct_change(20)

    sma200 = g["close"].transform(lambda s: s.rolling(200, min_periods=100).mean())
    out["above_200ma"] = stock["close"] / sma200 - 1

    hi52 = g["close"].transform(lambda s: s.rolling(252, min_periods=120).max())
    lo52 = g["close"].transform(lambda s: s.rolling(252, min_periods=120).min())
    out["dist_52w_high"] = (hi52 - stock["close"]) / stock["close"]
    out["dist_52w_low"]  = (stock["close"] - lo52) / stock["close"]

    out["rsi_14"]    = g["close"].transform(_rsi)
    out["macd_hist"] = g["close"].transform(_macd_hist)

    # Overbought duration — not just that RSI is high, but HOW LONG it has been ≥ 70.
    # A stock overbought for 5+ consecutive days is far more ripe for reversal than
    # one that just crossed 70 today.  Key dn_5 / dn_5_xl signal.
    rsi_ob = (out["rsi_14"] >= 70).fillna(False).astype(int)
    out["rsi_overbought_days"] = g.apply(
        lambda s: _consecutive_true(rsi_ob.loc[s.index]),
        include_groups=False,
    ).reset_index(level=0, drop=True)

    if {"high", "low"}.issubset(stock.columns):
        rng = (stock["high"] - stock["low"]).replace(0, np.nan)
        out["close_position"] = (stock["close"] - stock["low"]) / rng
    else:
        out["close_position"] = np.nan

    # ── Return quality (trend consistency, not just magnitude) ──────────
    ret_raw = g["close"].transform(lambda s: s.pct_change(1))

    # % of days with a positive return over trailing 20d
    out["win_rate_20d"] = ret_raw.groupby(stock["symbol"], sort=False).transform(
        lambda s: s.rolling(20, min_periods=10).apply(
            lambda x: (x > 0).mean(), raw=True
        )
    )

    # Annualised Sharpe of daily returns over trailing 20d
    out["sharpe_20d"] = ret_raw.groupby(stock["symbol"], sort=False).transform(
        lambda s: s.rolling(20, min_periods=10).apply(
            lambda x: x.mean() / (x.std() + 1e-8) * np.sqrt(252), raw=True
        )
    )

    # Max drawdown of close prices over trailing 20d (≤ 0)
    out["max_dd_20d"] = g["close"].transform(
        lambda s: s.rolling(20, min_periods=10).apply(
            lambda x: (
                (x - np.maximum.accumulate(x)) / (np.maximum.accumulate(x) + 1e-8)
            ).min(),
            raw=True,
        )
    )

    return out


# ──────────────────────────────────────────────────────────────────────
# Group 2 — Volatility & Volume
# ──────────────────────────────────────────────────────────────────────
def _group_vol_volume(stock: pd.DataFrame) -> pd.DataFrame:
    g = stock.groupby("symbol", sort=False, group_keys=False)
    out = pd.DataFrame(index=stock.index)

    if {"high", "low"}.issubset(stock.columns):
        atr = g.apply(
            lambda s: _rolling_atr(s["high"], s["low"], s["close"]),
            include_groups=False,
        ).reset_index(level=0, drop=True)
        out["atr_14"] = atr
    else:
        out["atr_14"] = g["close"].transform(
            lambda s: s.pct_change().abs().rolling(14, min_periods=5).mean()
        )

    out["hv_20"] = g["close"].transform(
        lambda s: s.pct_change().rolling(20, min_periods=10).std() * np.sqrt(252)
    )
    atr_60 = g.apply(
        lambda s: out.loc[s.index, "atr_14"].rolling(60, min_periods=20).mean(),
        include_groups=False,
    ).reset_index(level=0, drop=True) if "atr_14" in out else None
    out["vol_expansion"] = out["atr_14"] / atr_60 if atr_60 is not None else np.nan

    vol_sma5  = g["volume"].transform(lambda s: s.rolling(5,  min_periods=3).mean())
    vol_sma20 = g["volume"].transform(lambda s: s.rolling(20, min_periods=10).mean())
    out["vol_ratio_5d"]  = stock["volume"] / vol_sma5.replace(0, np.nan)
    out["vol_ratio_20d"] = stock["volume"] / vol_sma20.replace(0, np.nan)

    vol_sma60 = g["volume"].transform(lambda s: s.rolling(60, min_periods=30).mean())
    out["vol_ratio_60d"] = stock["volume"] / vol_sma60.replace(0, np.nan)

    # Days since last 2× volume surge — captures "cooling off" post-catalyst
    vol_surge_day = (stock["volume"] > vol_sma20 * 2.0).astype(int)
    out["days_since_vol_surge"] = g.apply(
        lambda s: _days_since_event(vol_surge_day.loc[s.index]),
        include_groups=False,
    ).reset_index(level=0, drop=True)

    # Amihud illiquidity: |return| / volume — high = stock moves a lot per unit traded
    ret_abs = g["close"].transform(lambda s: s.pct_change().abs())
    out["amihud_illiquidity"] = g.apply(
        lambda s: (ret_abs.loc[s.index] / s["volume"].replace(0, np.nan))
                  .rolling(20, min_periods=10).mean(),
        include_groups=False,
    ).reset_index(level=0, drop=True)

    if "deliv_pct" in stock.columns:
        deliv_sma20 = g["deliv_pct"].transform(lambda s: s.rolling(20, min_periods=10).mean())
        out["delivery_pct"]       = stock["deliv_pct"]
        out["delivery_ratio_20d"] = stock["deliv_pct"] / deliv_sma20.replace(0, np.nan)
        # Delivery streak: consecutive days of above-average delivery (institutional accumulation)
        above_avg_deliv = (stock["deliv_pct"] > deliv_sma20).astype(int)
        out["delivery_streak"] = g.apply(
            lambda s: _consecutive_true(above_avg_deliv.loc[s.index]),
            include_groups=False,
        ).reset_index(level=0, drop=True)
    else:
        out["delivery_pct"]       = np.nan
        out["delivery_ratio_20d"] = np.nan
        out["delivery_streak"]    = np.nan

    # ── Institutional participation proxy — avg trade value (INR per trade) ──
    # Large avg trade size = institutional involvement (block trades, basket
    # orders).  Small avg trade size = retail-dominated tape.  Empirically:
    # rising avg trade size *with* rising volume → institutional accumulation
    # (UP signal); rising avg trade size *with* falling price → distribution
    # (DN signal — smart money exiting before retail catches on).
    if {"turnover", "n_trades"}.issubset(stock.columns):
        n_trd = stock["n_trades"].replace(0, np.nan)
        atv = stock["turnover"] / n_trd                # INR per trade
        out["avg_trade_value"] = atv

        # 20-day SMA-normalized (today's size vs typical recent size)
        atv_sma20 = g.apply(
            lambda s: atv.loc[s.index].rolling(20, min_periods=10).mean(),
            include_groups=False,
        ).reset_index(level=0, drop=True)
        out["avg_trade_value_ratio_20d"] = atv / atv_sma20.replace(0, np.nan)

        # 60-day z-score — anomaly detection (today is unusual vs own 3-month base)
        atv_m60 = g.apply(
            lambda s: atv.loc[s.index].rolling(60, min_periods=20).mean(),
            include_groups=False,
        ).reset_index(level=0, drop=True)
        atv_s60 = g.apply(
            lambda s: atv.loc[s.index].rolling(60, min_periods=20).std(),
            include_groups=False,
        ).reset_index(level=0, drop=True)
        out["avg_trade_value_zscore_60d"] = (atv - atv_m60) / atv_s60.replace(0, np.nan)

        # 5-day momentum: is the avg trade size trending up or down?
        atv_5d   = g.apply(
            lambda s: atv.loc[s.index].rolling(5, min_periods=3).mean(),
            include_groups=False,
        ).reset_index(level=0, drop=True)
        atv_prev = g.apply(
            lambda s: atv.loc[s.index].rolling(5, min_periods=3).mean().shift(5),
            include_groups=False,
        ).reset_index(level=0, drop=True)
        out["avg_trade_value_5d_mom"] = atv_5d / atv_prev.replace(0, np.nan)
    else:
        out["avg_trade_value"]           = np.nan
        out["avg_trade_value_ratio_20d"] = np.nan
        out["avg_trade_value_zscore_60d"]= np.nan
        out["avg_trade_value_5d_mom"]    = np.nan
    return out


# ──────────────────────────────────────────────────────────────────────
# Group 3 — Consolidation & Breakout
# ──────────────────────────────────────────────────────────────────────
def _group_consolidation_breakout(stock: pd.DataFrame, base: pd.DataFrame) -> pd.DataFrame:
    """
    Inputs:
        stock: silver eod_stock columns (date, symbol, OHLCV, deliv_pct)
        base:  already-computed group-2 frame (for atr_14, vol_ratio_20d, delivery_ratio_20d)
    """
    g = stock.groupby("symbol", sort=False, group_keys=False)
    out = pd.DataFrame(index=stock.index)

    # Bollinger Bands (20d, 2σ)
    bb_mid = g["close"].transform(lambda s: s.rolling(20, min_periods=10).mean())
    bb_std = g["close"].transform(lambda s: s.rolling(20, min_periods=10).std())
    bb_up  = bb_mid + 2 * bb_std
    bb_lo  = bb_mid - 2 * bb_std
    bb_width = (bb_up - bb_lo) / bb_mid.replace(0, np.nan)
    out["bb_width"] = bb_width
    # Distance from upper Bollinger Band, normalised by band width.
    # Positive = price is above the upper band (overextended, dn reversion setup).
    # Negative = price is below the upper band (room to run upward).
    out["dist_from_bb_upper"] = (stock["close"] - bb_up) / bb_width.replace(0, np.nan)
    out["bb_squeeze"] = bb_width / g.apply(
        lambda s: bb_width.loc[s.index].rolling(252, min_periods=60).mean(),
        include_groups=False,
    ).reset_index(level=0, drop=True)
    out["squeeze_rank"] = g.apply(
        lambda s: _pct_rank(bb_width.loc[s.index], 252),
        include_groups=False,
    ).reset_index(level=0, drop=True)

    # ATR compression
    out["atr_compression"] = base["atr_14"] / g.apply(
        lambda s: base.loc[s.index, "atr_14"].rolling(100, min_periods=30).mean(),
        include_groups=False,
    ).reset_index(level=0, drop=True)

    # Range compression
    rng_20 = (
        g["high"].transform(lambda s: s.rolling(20, min_periods=10).max())
        - g["low"].transform(lambda s: s.rolling(20, min_periods=10).min())
    ) / stock["close"].replace(0, np.nan)
    out["range_compression"] = rng_20 / g.apply(
        lambda s: rng_20.loc[s.index].rolling(100, min_periods=30).mean(),
        include_groups=False,
    ).reset_index(level=0, drop=True)

    # Days in squeeze (consecutive bb_squeeze < 0.5)
    in_squeeze = (out["bb_squeeze"] < 0.5).astype(int)
    out["days_in_squeeze"] = g.apply(
        lambda s: _consecutive_true(in_squeeze.loc[s.index]),
        include_groups=False,
    ).reset_index(level=0, drop=True)

    # Breakout signal: yesterday squeeze + today breaks 20d range
    high_20  = g["close"].transform(lambda s: s.rolling(20, min_periods=10).max())
    low_20   = g["close"].transform(lambda s: s.rolling(20, min_periods=10).min())
    squeezed_yesterday = (out["bb_squeeze"].groupby(stock["symbol"], group_keys=False).shift(1) < 0.5)
    broke_up = (stock["close"] >= high_20) & squeezed_yesterday
    broke_dn = (stock["close"] <= low_20)  & squeezed_yesterday
    breakout = broke_up | broke_dn
    vol_conf = broke_up.fillna(False).copy() | broke_dn.fillna(False).copy()
    out["squeeze_breakout"] = breakout.astype(np.int8)

    # Volume + delivery confirmations
    vc = (breakout & (base["vol_ratio_20d"] > 1.5)).astype(np.int8)
    dc = (breakout & (base["delivery_ratio_20d"] > 1.3)).astype(np.int8)
    out["breakout_vol_confirm"]      = vc
    out["breakout_delivery_confirm"] = dc
    out["breakout_quality"]          = (
        out["squeeze_breakout"].fillna(0) + vc.fillna(0) + dc.fillna(0)
    ).astype(np.int8)

    # Wyckoff ADL (Accumulation/Distribution Line)
    rng = (stock["high"] - stock["low"]).replace(0, np.nan)
    money_flow_mult = ((stock["close"] - stock["low"]) - (stock["high"] - stock["close"])) / rng
    money_flow_vol  = money_flow_mult * stock["volume"]
    adl = g.apply(
        lambda s: money_flow_vol.loc[s.index].cumsum(),
        include_groups=False,
    ).reset_index(level=0, drop=True)
    adl_slope_5d   = g.apply(
        lambda s: adl.loc[s.index].diff(5),
        include_groups=False,
    ).reset_index(level=0, drop=True)
    price_slope_5d = g["close"].transform(lambda s: s.diff(5))
    # normalised divergence: rolling 252d z-score to avoid look-ahead from full-series stats
    def _rolling_zscore(x: pd.Series, w: int = 252) -> pd.Series:
        m = x.rolling(w, min_periods=60).mean()
        s = x.rolling(w, min_periods=60).std()
        return (x - m) / (s + 1e-8)

    adl_z   = adl_slope_5d.groupby(stock["symbol"], group_keys=False).apply(_rolling_zscore)
    price_z = price_slope_5d.groupby(stock["symbol"], group_keys=False).apply(_rolling_zscore)
    out["adl_divergence"] = adl_z.values - price_z.values

    # OBV slope
    obv = g.apply(
        lambda s: (np.sign(s["close"].diff()).fillna(0) * s["volume"]).cumsum(),
        include_groups=False,
    ).reset_index(level=0, drop=True)
    out["obv_slope_10d"] = g.apply(
        lambda s: obv.loc[s.index].diff(10),
        include_groups=False,
    ).reset_index(level=0, drop=True)

    # 52-week high breakout: close at or above 252-day rolling max (strong continuation signal)
    hi52 = g["close"].transform(lambda s: s.rolling(252, min_periods=120).max())
    out["hi52_breakout"] = (stock["close"] >= hi52).astype(np.int8)

    # 10-day tight range: (max-min)/close — compressed price = coiled spring
    tight_hi = g["close"].transform(lambda s: s.rolling(10, min_periods=5).max())
    tight_lo = g["close"].transform(lambda s: s.rolling(10, min_periods=5).min())
    out["tight_range_10d"] = (tight_hi - tight_lo) / stock["close"].replace(0, np.nan)

    # Price acceleration: momentum of momentum (ret_5d change over last 5 days)
    ret5 = g["close"].transform(lambda s: s.pct_change(5))
    out["price_acceleration"] = g.apply(
        lambda s: ret5.loc[s.index].diff(5),
        include_groups=False,
    ).reset_index(level=0, drop=True)

    return out


def _consecutive_true(s: pd.Series) -> pd.Series:
    """Run-length of consecutive 1s ending at each position."""
    v = s.fillna(0).astype(int).values
    out = np.zeros_like(v)
    run = 0
    for i, x in enumerate(v):
        run = run + 1 if x else 0
        out[i] = run
    return pd.Series(out, index=s.index)


def _days_since_event(s: pd.Series) -> pd.Series:
    """Days elapsed since the last 1 in a binary Series. NaN before first event."""
    v = s.fillna(0).astype(int).values
    out = np.full(len(v), np.nan)
    last = -1
    for i, x in enumerate(v):
        if x:
            last = i
        if last >= 0:
            out[i] = i - last
    return pd.Series(out, index=s.index)


# ──────────────────────────────────────────────────────────────────────
# Group 3b — Downside-specific signals
# ──────────────────────────────────────────────────────────────────────
# Six features designed to capture bearish setups that the existing
# upside-biased features miss.  Added after the DN model was found to be
# ignoring most of its 30 features and getting only val AUC 0.53.
#
# These should give the DN model new signals to lean on:
#   gap_down_count_20d     — recurring overnight weakness
#   consecutive_red_days   — momentum of selling pressure
#   days_since_20d_high    — how stale the recent high is (trend-rolling-over)
#   close_in_range_pct     — where in the day's range did it close
#                             (low value = closed near low = bearish distribution)
#   dist_above_50ma        — overextension above 50-day SMA (mean-reversion short)
#   red_day_vol_ratio      — vol on down days vs up days (last 20)
# ──────────────────────────────────────────────────────────────────────
def _group_downside_signals(stock: pd.DataFrame) -> pd.DataFrame:
    g   = stock.groupby("symbol", sort=False, group_keys=False)
    out = pd.DataFrame(index=stock.index)

    close = stock["close"]
    prev_close = g["close"].shift(1)
    ret_1d = close / prev_close - 1

    # 1. Gap-down count over 20 days: open < prev_close × 0.99 (>1% overnight gap down)
    if "open" in stock.columns:
        gap_down = (stock["open"] < prev_close * 0.99).astype(int)
        out["gap_down_count_20d"] = g.apply(
            lambda s: gap_down.loc[s.index].rolling(20, min_periods=5).sum(),
            include_groups=False,
        ).reset_index(level=0, drop=True)
        gap_up = (stock["open"] > prev_close * 1.01).astype(int)
        out["gap_up_count_20d"] = g.apply(
            lambda s: gap_up.loc[s.index].rolling(20, min_periods=5).sum(),
            include_groups=False,
        ).reset_index(level=0, drop=True)
    else:
        out["gap_down_count_20d"] = np.nan
        out["gap_up_count_20d"]   = np.nan

    # 2. Current run-length of consecutive red days (ret_1d < 0)
    red_day = (ret_1d < 0).astype(int)
    out["consecutive_red_days"] = g.apply(
        lambda s: _consecutive_true(red_day.loc[s.index]),
        include_groups=False,
    ).reset_index(level=0, drop=True)

    # 2b. Run-length of consecutive GREEN days — extended uptrend = mean reversion dn setup.
    # This is the single most direct dn signal we were missing: a stock making its
    # 7th or 8th consecutive green day is the primary short candidate in momentum reversal.
    green_day = (ret_1d > 0).astype(int)
    out["consecutive_green_days"] = g.apply(
        lambda s: _consecutive_true(green_day.loc[s.index]),
        include_groups=False,
    ).reset_index(level=0, drop=True)

    # 2c. Distribution days in last 20d: high-volume (>1.25× 20d avg) bearish close.
    # Smart money distributes shares to retail on high-volume down days before a
    # larger move down.  3+ distribution days in 20 sessions = distribution pattern.
    if "volume" in stock.columns:
        vol_sma20_dn = g["volume"].transform(lambda s: s.rolling(20, min_periods=10).mean())
        dist_day = ((stock["volume"] > vol_sma20_dn * 1.25) & (ret_1d < 0)).astype(int)
        out["distribution_days_20d"] = g.apply(
            lambda s: dist_day.loc[s.index].rolling(20, min_periods=5).sum(),
            include_groups=False,
        ).reset_index(level=0, drop=True)
    else:
        out["distribution_days_20d"] = np.nan

    # 3. Days since the most recent 20-day high (older = trend rolling over)
    # argmax-from-right on a rolling window gives days back to the peak.
    rolling_high = g["close"].transform(
        lambda s: s.rolling(20, min_periods=5).max()
    )
    # Days since high: count back from today until we hit the high level.
    # Cheap approximation: count days where close < rolling 20d high.
    below_high = (close < rolling_high).astype(int)
    out["days_since_20d_high"] = g.apply(
        lambda s: _consecutive_true(below_high.loc[s.index]),
        include_groups=False,
    ).reset_index(level=0, drop=True)

    # 4. Close-in-range pct: where in the day's range did the stock close?
    # Low values (closer to 0) = closed near the day's low = bearish distribution.
    # High values = closed near high = bullish accumulation.
    if {"high", "low"}.issubset(stock.columns):
        rng = (stock["high"] - stock["low"]).replace(0, np.nan)
        close_pos = (close - stock["low"]) / rng
        # 5-day average to smooth out noise
        out["close_in_range_5d"] = g.apply(
            lambda s: close_pos.loc[s.index].rolling(5, min_periods=3).mean(),
            include_groups=False,
        ).reset_index(level=0, drop=True)
    else:
        out["close_in_range_5d"] = np.nan

    # 5. Distance above 50-day SMA — overextension flag for mean-reversion shorts
    sma50 = g["close"].transform(lambda s: s.rolling(50, min_periods=20).mean())
    out["dist_above_50ma"] = close / sma50 - 1

    # 6. Red-day volume ratio: volume on down days vs up days (last 20 days)
    if "volume" in stock.columns:
        vol = stock["volume"]
        # Volume on red days
        vol_red   = (vol * (ret_1d < 0).astype(float))
        vol_green = (vol * (ret_1d > 0).astype(float))
        sum_red   = g.apply(
            lambda s: vol_red.loc[s.index].rolling(20, min_periods=5).sum(),
            include_groups=False,
        ).reset_index(level=0, drop=True)
        sum_green = g.apply(
            lambda s: vol_green.loc[s.index].rolling(20, min_periods=5).sum(),
            include_groups=False,
        ).reset_index(level=0, drop=True)
        out["red_day_vol_ratio"] = sum_red / sum_green.replace(0, np.nan)
    else:
        out["red_day_vol_ratio"] = np.nan

    return out


# ──────────────────────────────────────────────────────────────────────
# Group 3c — Volatility-scaled signal strength
# ──────────────────────────────────────────────────────────────────────
# Normalises returns and distance metrics by each stock's own historical
# volatility, so a 2% move in HDFCBANK and a 5% move in ADANIENT are
# represented on the same "1-sigma move" scale.  This is what the model
# needs to compare signals across stocks of vastly different vol regimes.
def _group_vol_scaled(base: pd.DataFrame) -> pd.DataFrame:
    """Vol-normalised versions of return / distance features.

    Reads from `base` (which already contains the raw features) and emits
    parallel columns scaled by either atr_14 (intraday) or hv_20 (multi-day).
    Both denominators are guarded against div-by-zero.
    """
    out = pd.DataFrame(index=base.index)
    atr = base["atr_14"].replace(0, np.nan) if "atr_14" in base.columns else np.nan
    hv  = base["hv_20"].replace(0, np.nan)  if "hv_20"  in base.columns else np.nan

    # Day-scale moves in ATR-sigma units (gap-like behaviour)
    if "ret_1d" in base.columns:
        out["ret_1d_vol_scaled"]  = base["ret_1d"]  / atr
    # Multi-day moves in HV-sigma units (trend strength)
    if "ret_5d" in base.columns:
        out["ret_5d_vol_scaled"]  = base["ret_5d"]  / hv
    if "ret_20d" in base.columns:
        out["ret_20d_vol_scaled"] = base["ret_20d"] / hv

    # Distance-to-key-level features, normalised
    if "dist_52w_high" in base.columns:
        out["dist_52w_high_z"]    = base["dist_52w_high"] / hv
    if "dist_52w_low" in base.columns:
        out["dist_52w_low_z"]     = base["dist_52w_low"] / hv
    if "dist_above_50ma" in base.columns:
        out["dist_above_50ma_z"]  = base["dist_above_50ma"] / hv
    if "nearest_resistance_dist" in base.columns:
        out["nearest_resistance_z"] = base["nearest_resistance_dist"] / atr
    if "nearest_support_dist" in base.columns:
        out["nearest_support_z"]    = base["nearest_support_dist"]    / atr
    # Long-term zone distances — ATR-normalised (typically large, ≥1)
    if "lt_resistance_dist" in base.columns:
        out["lt_resistance_z"] = base["lt_resistance_dist"] / atr
    if "lt_support_dist" in base.columns:
        out["lt_support_z"]    = base["lt_support_dist"]    / atr

    return out


# ──────────────────────────────────────────────────────────────────────
# Group 4 — Support / Resistance Zones (delegated to zones.py)
# ──────────────────────────────────────────────────────────────────────
def _group_zones(stock: pd.DataFrame, refresh_cadence_days: int = 5) -> pd.DataFrame:
    """Compute zone features by calling zones.compute_zone_features."""
    cols = ["date", "symbol", "open", "high", "low", "close"]
    zdf = _zones.compute_zone_features(
        stock[cols],
        asof_dates=None,
        sample_every_n_days=refresh_cadence_days,
    )
    return zdf


# ──────────────────────────────────────────────────────────────────────
# Group 5 — Weekly Candlestick Patterns
# ──────────────────────────────────────────────────────────────────────
def _group_weekly_candles(stock: pd.DataFrame) -> pd.DataFrame:
    """
    Resample to weekly (W-FRI), compute candle geometry + cross-week relationships,
    then join back to daily rows using each daily row's most recent *completed* week.
    """
    if not {"open", "high", "low", "close"}.issubset(stock.columns):
        return pd.DataFrame(index=stock.index)

    out_chunks = []
    for sym, sdf in stock.groupby("symbol", sort=False):
        s = sdf.set_index("date").sort_index()
        w = s.resample("W-FRI").agg({
            "open": "first", "high": "max", "low": "min", "close": "last",
        }).dropna(subset=["close"])
        if len(w) < 3:
            continue

        rng    = (w["high"] - w["low"]).replace(0, np.nan)
        body   = w["close"] - w["open"]
        upper  = w["high"] - w[["open", "close"]].max(axis=1)
        lower  = w[["open", "close"]].min(axis=1) - w["low"]

        wfeat = pd.DataFrame(index=w.index)
        wfeat["w_body_pct"]   = body / w["open"]
        wfeat["w_body_ratio"] = body.abs() / rng
        wfeat["w_upper_wick"] = upper / rng
        wfeat["w_lower_wick"] = lower / rng
        wfeat["w_close_pos"]  = (w["close"] - w["low"]) / rng
        wfeat["w_range_pct"]  = rng / w["close"]

        wfeat["w_gap"]          = w["open"] / w["close"].shift(1) - 1
        wfeat["w_body_expand"]  = body.abs() / body.abs().shift(1).replace(0, np.nan)
        wfeat["w_range_expand"] = wfeat["w_range_pct"] / wfeat["w_range_pct"].shift(1)
        wfeat["w_inside_bar"]   = ((w["high"] < w["high"].shift(1)) &
                                   (w["low"]  > w["low"].shift(1))).astype(np.int8)
        wfeat["w_outside_bar"]  = ((w["high"] > w["high"].shift(1)) &
                                   (w["low"]  < w["low"].shift(1))).astype(np.int8)
        wfeat["w_bull_engulf"]  = ((body > 0) &
                                   (w["close"] > w["high"].shift(1)) &
                                   (w["open"]  < w["close"].shift(1))).astype(np.int8)
        wfeat["w_bear_engulf"]  = ((body < 0) &
                                   (w["close"] < w["low"].shift(1)) &
                                   (w["open"]  > w["close"].shift(1))).astype(np.int8)
        wfeat["w_is_hammer"]        = ((wfeat["w_lower_wick"] > 0.6) &
                                       (wfeat["w_body_ratio"] < 0.2)).astype(np.int8)
        wfeat["w_is_shooting_star"] = ((wfeat["w_upper_wick"] > 0.6) &
                                       (wfeat["w_body_ratio"] < 0.2)).astype(np.int8)
        wfeat["w_is_doji"]          = (wfeat["w_body_ratio"] < 0.1).astype(np.int8)
        wfeat["w_is_marubozu"]      = ((wfeat["w_upper_wick"] < 0.05) &
                                       (wfeat["w_lower_wick"] < 0.05)).astype(np.int8)

        # Shift weekly features by 1 week — only use LAST COMPLETED week (no current-week leakage)
        wfeat = wfeat.shift(1).dropna(how="all")

        # Map daily rows → most recent completed week's features (asof-merge)
        daily_idx = sdf[["date"]].sort_values("date").reset_index(drop=True)
        wfeat_reset = wfeat.reset_index().rename(columns={"date": "week_end"})
        joined = pd.merge_asof(
            daily_idx, wfeat_reset,
            left_on="date", right_on="week_end",
            direction="backward",
        )
        joined.insert(1, "symbol", sym)
        joined = joined.drop(columns=["week_end"])
        out_chunks.append(joined)

    if not out_chunks:
        return pd.DataFrame(columns=["date", "symbol"])
    return pd.concat(out_chunks, ignore_index=True)


# ──────────────────────────────────────────────────────────────────────
# Group 6 — Derivatives Positioning
# ──────────────────────────────────────────────────────────────────────
def _group_derivatives(deriv: pd.DataFrame, stock: pd.DataFrame) -> pd.DataFrame:
    """
    Inputs:
        deriv: eod_deriv_daily with (date, symbol, fut_close, fut_oi, fut_chg_oi,
               fut_vol, pcr_oi, pcr_vol, max_pain, opt_call_oi, opt_put_oi,
               fut_near_expiry)
        stock: eod_stock (for spot close + ret_1d to derive long/short buildup)
    """
    keep = ["date", "symbol"]
    optional = ["fut_close", "fut_oi", "fut_chg_oi", "fut_vol",
                "opt_call_oi", "opt_put_oi", "opt_call_vol", "opt_put_vol",
                "call_wall_1", "put_wall_1",
                "pcr_oi", "pcr_vol", "max_pain", "fut_near_expiry",
                "atm_iv", "atm_ce_iv", "atm_pe_iv", "put_call_iv_skew"]
    cols = keep + [c for c in optional if c in deriv.columns]
    d = deriv[cols].copy()
    d["date"] = pd.to_datetime(d["date"])

    s = stock[["date", "symbol", "close"]].copy()
    s["date"] = pd.to_datetime(s["date"])

    df = d.merge(s, on=["date", "symbol"], how="left").sort_values(["symbol", "date"])
    g = df.groupby("symbol", sort=False, group_keys=False)

    if "fut_oi" in df.columns:
        df["fut_oi_chg_1d"] = g["fut_oi"].pct_change(1)
        df["fut_oi_chg_5d"] = g["fut_oi"].pct_change(5)
    else:
        df["fut_oi_chg_1d"] = np.nan
        df["fut_oi_chg_5d"] = np.nan

    ret_1d = g["close"].pct_change(1)
    df["long_buildup"]   = ((ret_1d > 0)  & (df["fut_oi_chg_1d"] > 0)).astype(np.int8)
    df["short_buildup"]  = ((ret_1d < 0)  & (df["fut_oi_chg_1d"] > 0)).astype(np.int8)
    df["short_covering"] = ((ret_1d > 0)  & (df["fut_oi_chg_1d"] < 0)).astype(np.int8)
    df["long_unwinding"] = ((ret_1d < 0)  & (df["fut_oi_chg_1d"] < 0)).astype(np.int8)

    # Consecutive-day streaks of buildup patterns
    df["long_buildup_streak"] = g.apply(
        lambda s: _consecutive_true(df.loc[s.index, "long_buildup"]),
        include_groups=False,
    ).reset_index(level=0, drop=True)
    df["short_buildup_streak"] = g.apply(
        lambda s: _consecutive_true(df.loc[s.index, "short_buildup"]),
        include_groups=False,
    ).reset_index(level=0, drop=True)

    # Long unwinding streak: consecutive days where price is DOWN and OI is DOWN
    # (longs exiting their positions = smart money reducing bullish exposure).
    # Each additional day of unwinding increases the probability of continued weakness.
    df["long_unwinding_streak"] = g.apply(
        lambda s: _consecutive_true(df.loc[s.index, "long_unwinding"]),
        include_groups=False,
    ).reset_index(level=0, drop=True)

    if "fut_close" in df.columns:
        df["basis_pct"] = (df["fut_close"] - df["close"]) / df["close"]
    else:
        df["basis_pct"] = np.nan

    if "pcr_oi" in df.columns:
        df["pcr_oi_rank_60d"] = g.apply(
            lambda x: _pct_rank(x["pcr_oi"], 60), include_groups=False,
        ).reset_index(level=0, drop=True)
        # PCR rate of change: direction of options flow shift (drop = calls being bought)
        df["pcr_chg_5d"] = g["pcr_oi"].pct_change(5)
    else:
        df["pcr_oi_rank_60d"] = np.nan
        df["pcr_chg_5d"]      = np.nan

    if "max_pain" in df.columns:
        df["max_pain_dist"] = (df["close"] - df["max_pain"]) / df["close"]
    else:
        df["max_pain_dist"] = np.nan

    # Options wall distances — OI-based support/resistance (complement zone features)
    # call_wall_1: strike with highest call OI (dealer resistance above price)
    # put_wall_1:  strike with highest put OI  (dealer support below price)
    if "call_wall_1" in df.columns:
        df["dist_call_wall"] = (df["call_wall_1"] - df["close"]) / df["close"]
    else:
        df["dist_call_wall"] = np.nan
    if "put_wall_1" in df.columns:
        df["dist_put_wall"] = (df["close"] - df["put_wall_1"]) / df["close"]
    else:
        df["dist_put_wall"] = np.nan

    if "fut_near_expiry" in df.columns:
        df["days_to_expiry"] = (pd.to_datetime(df["fut_near_expiry"]) - df["date"]).dt.days
    else:
        df["days_to_expiry"] = np.nan

    # Futures speculation intensity: volume / OI (high = active trading, precedes large moves)
    if "fut_vol" in df.columns and "fut_oi" in df.columns:
        df["fut_vol_oi_ratio"] = df["fut_vol"] / df["fut_oi"].replace(0, np.nan)
    else:
        df["fut_vol_oi_ratio"] = np.nan

    # Basis trend: cost of carry direction (rising = longs building, falling = unwinding)
    if "basis_pct" in df.columns:
        df["basis_chg_5d"] = g["basis_pct"].diff(5)
    else:
        df["basis_chg_5d"] = np.nan

    # Total options OI (call + put) normalised by 20d average — high = hedging activity spike
    if {"opt_call_oi", "opt_put_oi"}.issubset(df.columns):
        df["opt_oi_total"] = df["opt_call_oi"] + df["opt_put_oi"]
        opt_oi_sma20 = g["opt_oi_total"].transform(lambda s: s.rolling(20, min_periods=10).mean())
        df["opt_oi_ratio_20d"] = df["opt_oi_total"] / opt_oi_sma20.replace(0, np.nan)
    else:
        df["opt_oi_ratio_20d"] = np.nan

    # ── Options structure features (Tier 1 & 2) ─────────────────────────────────

    # Put OI concentration — fraction of total options OI held in puts.
    # Distinct from pcr_oi (ratio): put_oi_pct=0.6 means 60% of all open contracts
    # are puts, regardless of call OI magnitude. High = structural bearish hedging.
    if {"opt_put_oi", "opt_call_oi"}.issubset(df.columns):
        _total_oi_denom = (df["opt_call_oi"] + df["opt_put_oi"]).replace(0, np.nan)
        df["put_oi_pct"] = df["opt_put_oi"] / _total_oi_denom
    else:
        df["put_oi_pct"] = np.nan

    # Options OI 5-day change rank — captures fresh options accumulation episodes.
    # High rank = OI growing fastest vs 60d history = informed hedging building fast.
    # opt_oi_total is not stored in silver; derive it from call + put OI.
    if "opt_oi_total" not in df.columns and {"opt_call_oi", "opt_put_oi"}.issubset(df.columns):
        df["opt_oi_total"] = df["opt_call_oi"] + df["opt_put_oi"]
    if "opt_oi_total" in df.columns:
        df["opt_total_oi_chg_5d"] = g["opt_oi_total"].diff(5)
        df["opt_total_oi_chg_5d_rank"] = g.apply(
            lambda x: _pct_rank(df.loc[x.index, "opt_total_oi_chg_5d"], 60),
            include_groups=False,
        ).reset_index(level=0, drop=True)
    else:
        df["opt_total_oi_chg_5d_rank"] = np.nan

    # Annualised basis — time-normalised cost of carry for cross-stock comparison.
    # Raw basis_pct is DTE-dependent: 0.5% with 7 days left ≈ 26% annualised vs
    # 0.5% with 28 days ≈ 6.5%.  Clip to [-1, 1] to handle near-expiry distortion.
    if "basis_pct" in df.columns and "days_to_expiry" in df.columns:
        _dte_yr = (df["days_to_expiry"] / 365.0).replace(0, np.nan)
        df["annualized_basis"] = (df["basis_pct"] / _dte_yr).clip(-1.0, 1.0)
    else:
        df["annualized_basis"] = np.nan

    # Wall compression — how asymmetric is the options market's range?
    # Low value = limited upside to call wall, large gap to put wall = resistance close,
    # support far = structurally vulnerable to rejection at current level.
    if "dist_call_wall" in df.columns and "dist_put_wall" in df.columns:
        _headroom = df["dist_call_wall"].clip(lower=-0.5)   # (call_wall_1 - close)/close
        _support  = df["dist_put_wall"].clip(lower=0.001)   # (close - put_wall_1)/close
        df["wall_compression"] = (_headroom / _support).clip(-5.0, 5.0)
    else:
        df["wall_compression"] = np.nan

    # ── dn_5_xl-specific vulnerability features ──────────────────────────────────
    # Put-call VOLUME flow ratio (active bearish hedging signal).
    # > 1 = more puts traded than calls → market participants hedging against this stock.
    # Differs from pcr_oi (open interest) — volume reflects TODAY's flow conviction.
    if {"opt_put_vol", "opt_call_vol"}.issubset(df.columns):
        df["put_call_vol_ratio"] = df["opt_put_vol"] / df["opt_call_vol"].replace(0, np.nan)
        df["put_call_vol_rank_60d"] = g.apply(
            lambda x: _pct_rank(df.loc[x.index, "put_call_vol_ratio"], 60), include_groups=False,
        ).reset_index(level=0, drop=True)
    else:
        df["put_call_vol_ratio"]    = np.nan
        df["put_call_vol_rank_60d"] = np.nan

    # Short-buildup and long-unwinding rolling 5d counts (non-consecutive).
    # Captures "3 out of 5 days had smart-money shorts building" which is more
    # robust than a consecutive streak (streak resets on a single green day).
    df["short_buildup_5d_count"] = g.apply(
        lambda s: df.loc[s.index, "short_buildup"].rolling(5, min_periods=3).sum(),
        include_groups=False,
    ).reset_index(level=0, drop=True)
    df["long_unwind_5d_count"] = g.apply(
        lambda s: df.loc[s.index, "long_unwinding"].rolling(5, min_periods=3).sum(),
        include_groups=False,
    ).reset_index(level=0, drop=True)

    # Futures OI z-score vs 60d history.
    # High z-score = unusually crowded long positioning → larger flush when market turns.
    if "fut_oi" in df.columns:
        oi_mean = g["fut_oi"].transform(lambda s: s.rolling(60, min_periods=20).mean())
        oi_std  = g["fut_oi"].transform(lambda s: s.rolling(60, min_periods=20).std())
        df["fut_oi_z_60d"] = (df["fut_oi"] - oi_mean) / oi_std.replace(0, np.nan)
    else:
        df["fut_oi_z_60d"] = np.nan

    # Basis percentile rank in 60d (low rank = basis near multi-month low = bearish).
    # Futures discount (negative basis) means smart money isn't willing to hold longs
    # at spot price — a leading signal for downside pressure.
    if "basis_pct" in df.columns:
        df["basis_rank_60d"] = g.apply(
            lambda x: _pct_rank(df.loc[x.index, "basis_pct"], 60), include_groups=False,
        ).reset_index(level=0, drop=True)
    else:
        df["basis_rank_60d"] = np.nan

    # ── ATM implied volatility features ─────────────────────────────────────
    # atm_iv / atm_ce_iv / atm_pe_iv / put_call_iv_skew come from silver
    # (computed via Black-Scholes from per-strike bhavcopy prices).
    # We compute rolling rank + IV-vs-HV ratio here; the raw columns pass through.
    if "atm_iv" in df.columns:
        df["atm_iv_rank_252d"] = g.apply(
            lambda x: _pct_rank(df.loc[x.index, "atm_iv"], 252), include_groups=False,
        ).reset_index(level=0, drop=True)
    else:
        df["atm_iv"] = np.nan
        df["atm_ce_iv"] = np.nan
        df["atm_pe_iv"] = np.nan
        df["put_call_iv_skew"] = np.nan
        df["atm_iv_rank_252d"] = np.nan

    if "put_call_iv_skew" in df.columns:
        df["put_call_iv_skew_rank_60d"] = g.apply(
            lambda x: _pct_rank(df.loc[x.index, "put_call_iv_skew"], 60), include_groups=False,
        ).reset_index(level=0, drop=True)
    else:
        df["put_call_iv_skew_rank_60d"] = np.nan

    out_cols = ["date", "symbol",
                "fut_oi_chg_1d", "fut_oi_chg_5d",
                "long_buildup", "short_buildup", "short_covering", "long_unwinding",
                "long_buildup_streak", "short_buildup_streak", "long_unwinding_streak",
                "basis_pct", "basis_chg_5d",
                "pcr_oi", "pcr_vol", "pcr_oi_rank_60d", "pcr_chg_5d",
                "fut_vol_oi_ratio", "opt_oi_ratio_20d",
                "max_pain_dist", "dist_call_wall", "dist_put_wall",
                "days_to_expiry",
                # Options structure (Tier 1 & 2)
                "put_oi_pct", "opt_total_oi_chg_5d_rank",
                "annualized_basis", "wall_compression",
                # ATM IV
                "atm_iv", "atm_ce_iv", "atm_pe_iv",
                "put_call_iv_skew", "atm_iv_rank_252d", "put_call_iv_skew_rank_60d",
                # dn_5_xl-specific
                "put_call_vol_ratio", "put_call_vol_rank_60d",
                "short_buildup_5d_count", "long_unwind_5d_count",
                "fut_oi_z_60d", "basis_rank_60d"]
    return df[[c for c in out_cols if c in df.columns]]


# ──────────────────────────────────────────────────────────────────────
# Group 7 — Participant Flows (market-wide, shifted by 1 day)
# ──────────────────────────────────────────────────────────────────────
def _group_participant_flows(fii_dii: pd.DataFrame, part_oi: pd.DataFrame) -> pd.DataFrame:
    """
    Returns one row per date (cross-broadcast to all symbols at merge time).
    """
    fd = fii_dii.copy().sort_values("date")
    fd["date"] = pd.to_datetime(fd["date"])

    # Lag 1 day — participant data is published with a 1-day lag
    for col in ["fii_net", "dii_net"]:
        if col in fd.columns:
            fd[col] = fd[col].shift(1)

    out = pd.DataFrame({"date": fd["date"]})
    if "fii_net" in fd.columns:
        # ── Existing 3 FII features ───────────────────────────────────
        out["fii_cash_net_5d"]    = fd["fii_net"].rolling(5,  min_periods=2).sum()
        rm20 = fd["fii_net"].rolling(20, min_periods=5).mean()
        rs20 = fd["fii_net"].rolling(20, min_periods=5).std()
        out["fii_cash_zscore"]    = (fd["fii_net"] - rm20) / rs20
        out["fii_cash_streak"]    = _signed_streak(fd["fii_net"])

        # ── New: longer-horizon flow (30d sum) ────────────────────────
        out["fii_cash_net_30d"]   = fd["fii_net"].rolling(30, min_periods=10).sum()

        # ── New: acceleration — change in 5-day flow (inflection signal) ─
        out["fii_cash_acceleration"] = out["fii_cash_net_5d"].diff()

        # ── New: short-vs-long trend reversal flag ────────────────────
        sgn_5d  = np.sign(out["fii_cash_net_5d"].fillna(0))
        sgn_30d = np.sign(out["fii_cash_net_30d"].fillna(0))
        out["fii_cash_reversal_flag"] = (sgn_5d != sgn_30d).astype(np.int8)

        # ── New: extreme-flow binary flags (252-day adaptive percentiles) ─
        # Bottom 5% of last 252 trading days = "extreme outflow today"
        # Top 5% = "extreme inflow today"
        roll252 = fd["fii_net"].rolling(252, min_periods=60)
        p5  = roll252.quantile(0.05)
        p95 = roll252.quantile(0.95)
        out["fii_extreme_outflow"] = (fd["fii_net"] <= p5).astype(np.int8)
        out["fii_extreme_inflow"]  = (fd["fii_net"] >= p95).astype(np.int8)

    # ── New: conviction ratio — buy/sell on the same day ──────────────
    # Captures churn vs conviction: net=+100 with buy=5000/sell=4900 is high-churn
    # low-conviction; net=+100 with buy=500/sell=400 is much higher conviction.
    if {"fii_buy", "fii_sell"}.issubset(fd.columns):
        out["fii_buy_sell_ratio"] = fd["fii_buy"] / fd["fii_sell"].replace(0, np.nan)

    if "dii_net" in fd.columns:
        out["dii_cash_net_5d"]    = fd["dii_net"].rolling(5, min_periods=2).sum()
    if {"fii_net", "dii_net"}.issubset(fd.columns):
        client_net = -(fd["fii_net"].fillna(0) + fd["dii_net"].fillna(0))
        out["smart_vs_retail"]    = (fd["fii_net"].fillna(0) + fd["dii_net"].fillna(0)) - client_net

    # Participant OI features (from participant_oi silver table).
    # All levels use .shift(1) — NSE publishes this data at end-of-day,
    # available the next morning, so T features can only use T-1 values.
    if part_oi is not None and not part_oi.empty:
        po = part_oi.copy()
        po["date"] = pd.to_datetime(po["date"])

        fii_rows = po[po["participant"] == "FII"].copy().sort_values("date")
        cli_rows = po[po["participant"] == "Client"].copy().sort_values("date")
        pro_rows = po[po["participant"] == "Pro"].copy().sort_values("date")

        # ── Existing: FII index futures net change ────────────────────────
        if "fut_idx_long" in fii_rows.columns and "fut_idx_short" in fii_rows.columns:
            fii_rows["fii_idx_fut_net"] = fii_rows["fut_idx_long"] - fii_rows["fut_idx_short"]
            fii_rows["fii_idx_fut_net_chg"] = fii_rows["fii_idx_fut_net"].diff().shift(1)
            out = out.merge(fii_rows[["date", "fii_idx_fut_net_chg"]], on="date", how="left")

        # ── Tier 1: FII stock futures net (level + 5d change) ─────────────
        # FII reducing longs or adding shorts in stock futures → directional
        # smart-money signal distinct from index futures positioning.
        if "fut_stk_long" in fii_rows.columns and "fut_stk_short" in fii_rows.columns:
            fii_rows["fii_stk_fut_net"] = (
                (fii_rows["fut_stk_long"] - fii_rows["fut_stk_short"]).shift(1)
            )
            fii_rows["fii_stk_fut_net_chg_5d"] = fii_rows["fii_stk_fut_net"].diff(5)
            out = out.merge(
                fii_rows[["date", "fii_stk_fut_net", "fii_stk_fut_net_chg_5d"]],
                on="date", how="left",
            )

        # ── Tier 1: Client (retail) stock futures net ─────────────────────
        # Contrarian signal: retail is persistently net long stock futures.
        # Extreme readings signal overcrowding → flush risk when reversals hit.
        if "fut_stk_long" in cli_rows.columns and "fut_stk_short" in cli_rows.columns:
            cli_rows["client_stk_fut_net"] = (
                (cli_rows["fut_stk_long"] - cli_rows["fut_stk_short"]).shift(1)
            )
            out = out.merge(cli_rows[["date", "client_stk_fut_net"]], on="date", how="left")

        # ── Tier 1: FII vs Client divergence ─────────────────────────────
        # Negative ratio = FII net short while retail net long = classic
        # smart/dumb money divergence. Most actionable when strongly negative.
        if "fii_stk_fut_net" in out.columns and "client_stk_fut_net" in out.columns:
            _cli_safe = out["client_stk_fut_net"].replace(0, np.nan)
            out["fii_vs_client_stk"] = (out["fii_stk_fut_net"] / _cli_safe).clip(-5.0, 5.0)

        # ── Tier 2: FII put buying in stock options (accumulation speed) ──
        # FII holding large long put positions in stock options = institutional
        # downside protection. 5d change captures acceleration of that hedging.
        if "opt_stk_put_long" in fii_rows.columns:
            fii_rows["fii_put_long_stk"] = fii_rows["opt_stk_put_long"].shift(1)
            fii_rows["fii_put_long_stk_chg_5d"] = fii_rows["fii_put_long_stk"].diff(5)
            out = out.merge(
                fii_rows[["date", "fii_put_long_stk", "fii_put_long_stk_chg_5d"]],
                on="date", how="left",
            )

        # ── Tier 2: Client put selling in stock options ───────────────────
        # Retail selling stock puts = complacency / short premium. When FII is
        # buying puts AND Client is selling puts: maximum setup divergence.
        if "opt_stk_put_short" in cli_rows.columns:
            cli_rows["client_put_short_stk"] = cli_rows["opt_stk_put_short"].shift(1)
            out = out.merge(cli_rows[["date", "client_put_short_stk"]], on="date", how="left")

        # ── Tier 2: Proprietary desk stock futures net ────────────────────
        # Pro desks are the most informed intra-day participants. Net short in
        # stock futures = market-wide bearish conviction from informed desks.
        if "fut_stk_long" in pro_rows.columns and "fut_stk_short" in pro_rows.columns:
            pro_rows["pro_stk_fut_net"] = (
                (pro_rows["fut_stk_long"] - pro_rows["fut_stk_short"]).shift(1)
            )
            out = out.merge(pro_rows[["date", "pro_stk_fut_net"]], on="date", how="left")

        # ── FII directional options bias (synthetic positioning) ──────────────
        # fii_stk_put_call_oi_ratio: long_puts / long_calls for FII in stock opts.
        #   > 1 = FII holds more put longs than call longs → bearish options bias.
        # fii_stk_net_opt_dir: (put_long + call_short) − (call_long + put_short)
        #   = FII's total synthetic short minus synthetic long in stock options.
        #   Positive = FII is net synthetically short via options (bearish).
        if ("opt_stk_put_long" in fii_rows.columns and
                "opt_stk_call_long" in fii_rows.columns):
            fii_rows["fii_stk_put_call_oi_ratio"] = (
                (fii_rows["opt_stk_put_long"] /
                 fii_rows["opt_stk_call_long"].replace(0, np.nan))
                .shift(1).clip(0, 8)
            )
            bearish = fii_rows["opt_stk_put_long"].copy()
            bullish = fii_rows["opt_stk_call_long"].copy()
            if "opt_stk_call_short" in fii_rows.columns:
                bearish = bearish + fii_rows["opt_stk_call_short"]
            if "opt_stk_put_short" in fii_rows.columns:
                bullish = bullish + fii_rows["opt_stk_put_short"]
            fii_rows["fii_stk_net_opt_dir"] = (bearish - bullish).shift(1)
            fii_rows["fii_stk_net_opt_dir_chg_5d"] = fii_rows["fii_stk_net_opt_dir"].diff(5)
            out = out.merge(
                fii_rows[["date", "fii_stk_put_call_oi_ratio",
                           "fii_stk_net_opt_dir", "fii_stk_net_opt_dir_chg_5d"]],
                on="date", how="left",
            )

        # ── Client net call preference in stock options ───────────────────────
        # Retail holds far more long calls than long puts (≈2.6× on average).
        # When this gap widens = maximum retail complacency = contrarian bearish.
        if ("opt_stk_call_long" in cli_rows.columns and
                "opt_stk_put_long" in cli_rows.columns):
            cli_rows["client_stk_call_put_net"] = (
                (cli_rows["opt_stk_call_long"] - cli_rows["opt_stk_put_long"]).shift(1)
            )
            out = out.merge(cli_rows[["date", "client_stk_call_put_net"]],
                            on="date", how="left")

    return out


def _signed_streak(s: pd.Series) -> pd.Series:
    sign = np.sign(s.fillna(0)).astype(int)
    out = np.zeros(len(sign), dtype=int)
    for i in range(len(sign)):
        if sign.iloc[i] == 0:
            out[i] = 0
        elif i == 0 or sign.iloc[i] != sign.iloc[i - 1]:
            out[i] = sign.iloc[i]
        else:
            out[i] = out[i - 1] + sign.iloc[i]
    return pd.Series(out, index=s.index)


# ──────────────────────────────────────────────────────────────────────
# Group 7b — NSDL fortnightly sectoral FII flows
# ──────────────────────────────────────────────────────────────────────
# Sectoral FII flow data is published every ~15 days with a ~7-10 day lag.
# We compute features at FORTNIGHT granularity per sector first, then
# broadcast to daily stock rows via merge_asof (using publication lag to
# avoid look-ahead leakage).
#
# 10 features in 3 groups:
#   Time-series (per-sector):
#     fii_sector_flow_14d           — most recent fortnight net flow (Rs Cr)
#     fii_sector_flow_30d           — 2-fortnight sum (~1 month)
#     fii_sector_flow_90d           — 6-fortnight sum (~1 quarter)
#     fii_sector_flow_zscore        — z-score vs 24-fortnight (~1yr) history
#     fii_sector_flow_streak        — signed consecutive same-direction fortnights
#     fii_sector_flow_acceleration  — change vs previous fortnight (inflection)
#   AUM-normalized:
#     fii_sector_flow_pct_aum       — flow / AUM (size-independent magnitude)
#     fii_sector_aum_pct_change_90d — quarter-on-quarter AUM growth
#   Cross-sectional (within-fortnight market state):
#     fii_sector_rotation_rank      — pct-rank of this sector's flow vs others
#     fii_sector_breadth_pos        — # of sectors with positive flow this fortnight
# ──────────────────────────────────────────────────────────────────────
def _group_fii_sector_flow(stock_dates: pd.DataFrame,
                           fii_sector: pd.DataFrame | None) -> pd.DataFrame:
    """Build per-stock-day features from the fortnightly FII sectoral silver.

    stock_dates: DataFrame[date, symbol]
    fii_sector:  long-format DataFrame[fortnight_end, sector, fii_equity_net_cr,
                                       fii_equity_aum_cr] (the silver table)
    Returns DataFrame[date, symbol, <10 feature columns>].
    """
    feat_names = [
        "fii_sector_flow_14d", "fii_sector_flow_30d", "fii_sector_flow_90d",
        "fii_sector_flow_zscore", "fii_sector_flow_pct_aum",
        "fii_sector_flow_streak", "fii_sector_flow_acceleration",
        "fii_sector_aum_pct_change_90d",
        "fii_sector_rotation_rank", "fii_sector_breadth_pos",
    ]
    empty = pd.DataFrame(columns=["date", "symbol"] + feat_names)

    if fii_sector is None or fii_sector.empty:
        return empty

    # ── Compute per-sector longitudinal features at fortnight granularity ──
    sd = fii_sector.copy()
    sd["fortnight_end"] = pd.to_datetime(sd["fortnight_end"])
    sd = sd.sort_values(["sector", "fortnight_end"]).reset_index(drop=True)

    g = sd.groupby("sector", sort=False, group_keys=False)

    sd["fii_sector_flow_14d"] = sd["fii_equity_net_cr"]
    sd["fii_sector_flow_30d"] = g["fii_equity_net_cr"].transform(
        lambda s: s.rolling(2, min_periods=1).sum())
    sd["fii_sector_flow_90d"] = g["fii_equity_net_cr"].transform(
        lambda s: s.rolling(6, min_periods=2).sum())

    rm = g["fii_equity_net_cr"].transform(
        lambda s: s.rolling(24, min_periods=4).mean())
    rs = g["fii_equity_net_cr"].transform(
        lambda s: s.rolling(24, min_periods=4).std())
    sd["fii_sector_flow_zscore"] = (sd["fii_equity_net_cr"] - rm) / rs.replace(0, np.nan)

    sd["fii_sector_flow_pct_aum"] = (
        sd["fii_equity_net_cr"] / sd["fii_equity_aum_cr"].replace(0, np.nan))

    sd["fii_sector_flow_streak"] = g.apply(
        lambda s: _signed_streak(s["fii_equity_net_cr"]),
        include_groups=False,
    ).reset_index(level=0, drop=True)

    sd["fii_sector_flow_acceleration"] = g["fii_equity_net_cr"].transform(
        lambda s: s.diff())

    sd["fii_sector_aum_pct_change_90d"] = g["fii_equity_aum_cr"].transform(
        lambda s: s.pct_change(6))

    # ── Cross-sectional features (rank/breadth across sectors within a fortnight) ──
    by_fort = sd.groupby("fortnight_end", sort=False)
    sd["fii_sector_rotation_rank"] = by_fort["fii_equity_net_cr"].rank(pct=True)
    sd["fii_sector_breadth_pos"] = by_fort["fii_equity_net_cr"].transform(
        lambda s: (s > 0).sum())

    # ── Broadcast to daily stock rows ──
    sm = stock_dates[["date", "symbol"]].copy()
    sm["date"] = pd.to_datetime(sm["date"])
    # Direct ticker -> NSDL canonical sector (single-step lookup, no bridge)
    sm["nsdl_sector"] = sm["symbol"].map(TICKER_NSDL_SECTOR)
    sm = sm.dropna(subset=["nsdl_sector"]).copy()
    if sm.empty:
        return empty

    # Publication-lag-aware lookup date: feature at prediction date d sees the
    # most recent fortnight_end e such that (e + LAG_DAYS) <= d.
    # Force both keys to identical datetime resolution (ns) — pandas merge_asof
    # refuses to join across us/ms/ns even when values are otherwise compatible.
    lag = pd.Timedelta(days=FII_SECTOR_PUBLICATION_LAG_DAYS)
    sm["lookup_date"] = (sm["date"] - lag).astype("datetime64[ns]")
    sm = sm.sort_values("lookup_date").reset_index(drop=True)

    sd_keep = sd[["fortnight_end", "sector"] + feat_names].rename(
        columns={"fortnight_end": "lookup_date"}).copy()
    sd_keep["lookup_date"] = sd_keep["lookup_date"].astype("datetime64[ns]")
    sd_keep = sd_keep.sort_values("lookup_date")

    # Per-sector merge_asof — each stock's nsdl_sector dictates which sector
    # rows to draw from.  Direction='backward' forward-fills the latest
    # available fortnight to all subsequent daily rows until the next one lands.
    chunks = []
    for sec, sub in sm.groupby("nsdl_sector", sort=False, group_keys=False):
        sec_data = (sd_keep[sd_keep["sector"] == sec]
                    .drop(columns=["sector"]).reset_index(drop=True))
        if sec_data.empty:
            continue
        merged = pd.merge_asof(
            sub.sort_values("lookup_date").reset_index(drop=True),
            sec_data, on="lookup_date", direction="backward",
        )
        chunks.append(merged[["date", "symbol"] + feat_names])

    if not chunks:
        return empty
    return pd.concat(chunks, ignore_index=True)


# ──────────────────────────────────────────────────────────────────────
# Group 8 — Earnings & Fundamentals
# ──────────────────────────────────────────────────────────────────────
def _group_earnings(stock_keys: pd.DataFrame,
                    earnings: Optional[pd.DataFrame],
                    eps:      Optional[pd.DataFrame],
                    fund:     Optional[pd.DataFrame],
                    corp_actions: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    stock_keys: (date, symbol) rows we need features for.
    """
    out = stock_keys.copy().sort_values(["symbol", "date"]).reset_index(drop=True)
    out["date"] = pd.to_datetime(out["date"])

    # days_to_next_earnings / days_since_last_earnings
    if earnings is not None and not earnings.empty:
        ea = earnings[["date", "symbol"]].copy()
        ea["date"] = pd.to_datetime(ea["date"])
        ea = ea.dropna().drop_duplicates().sort_values(["symbol", "date"])
        out = out.sort_values(["symbol", "date"])
        out["days_to_next_earnings"] = np.nan
        out["days_since_last_earnings"] = np.nan
        for sym, grp in out.groupby("symbol", sort=False):
            sym_ea = ea[ea["symbol"] == sym]["date"].values
            if len(sym_ea) == 0:
                continue
            dates = grp["date"].values
            next_idx = np.searchsorted(sym_ea, dates, side="left")
            prev_idx = np.where(next_idx > 0, next_idx - 1, 0)
            has_next = next_idx < len(sym_ea)
            has_prev = (next_idx > 0)
            d_next = np.where(
                has_next,
                (sym_ea[np.clip(next_idx, 0, len(sym_ea) - 1)] - dates) / np.timedelta64(1, "D"),
                np.nan,
            )
            d_prev = np.where(
                has_prev,
                (dates - sym_ea[prev_idx]) / np.timedelta64(1, "D"),
                np.nan,
            )
            out.loc[grp.index, "days_to_next_earnings"]    = d_next
            out.loc[grp.index, "days_since_last_earnings"] = d_prev

        out["is_pre_earnings"] = (out["days_to_next_earnings"] <= 5).astype(np.int8)
    else:
        out["days_to_next_earnings"]    = np.nan
        out["days_since_last_earnings"] = np.nan
        out["is_pre_earnings"]          = 0

    # EPS surprise: merge most recent confirmed surprise per (symbol, date)
    if eps is not None and not eps.empty:
        e = eps.copy()
        e["earnings_date"] = pd.to_datetime(e["earnings_date"])
        e = e.dropna(subset=["earnings_date"]).sort_values(["symbol", "earnings_date"])

        # Winsorise surprise_pct — raw data has outliers like -56321% / +5358%
        # (EPS near zero causes division noise). Cap at ±150%.
        e["surprise_pct"] = e["surprise_pct"].clip(-150, 150)

        # ── per-quarter derived columns ─────────────────────────────────────
        grp = e.groupby("symbol")

        # beat / miss streaks (positive = beat, negative = miss, consecutive)
        def _streak(s: "pd.Series") -> "pd.Series":
            sign  = np.sign(s.fillna(0))
            streak = np.zeros(len(s), dtype=np.float32)
            for i in range(len(s)):
                if i == 0:
                    streak[i] = sign.iloc[i]
                elif sign.iloc[i] == 0:
                    streak[i] = 0
                elif sign.iloc[i] == sign.iloc[i - 1]:
                    streak[i] = streak[i - 1] + sign.iloc[i]
                else:
                    streak[i] = sign.iloc[i]
            return pd.Series(streak, index=s.index)

        e["streak"]           = grp["surprise_pct"].transform(_streak)
        e["eps_beat_streak"]  = e["streak"].clip(lower=0)
        e["eps_miss_streak"]  = (-e["streak"]).clip(lower=0)
        e["big_eps_miss"]     = (e["surprise_pct"] < -20).astype(np.int8)
        e["eps_surprise_3q_avg"] = grp["surprise_pct"].transform(
            lambda s: s.rolling(3, min_periods=1).mean()
        )

        # ── asof-merge all columns to daily panel ──────────────────────────
        eps_daily = e[["symbol", "earnings_date",
                        "surprise_pct", "eps_surprise_3q_avg",
                        "eps_beat_streak", "eps_miss_streak", "big_eps_miss",
                        "eps_growth_yoy"]].rename(
            columns={"earnings_date": "date",
                     "surprise_pct":  "last_eps_surprise_pct"}
        ).sort_values("date")

        out = pd.merge_asof(
            out.sort_values("date"),
            eps_daily,
            on="date", by="symbol", direction="backward",
        ).sort_values(["symbol", "date"])
    else:
        for col in ("last_eps_surprise_pct", "eps_surprise_3q_avg",
                    "eps_beat_streak", "eps_miss_streak",
                    "big_eps_miss", "eps_growth_yoy"):
            out[col] = np.nan

    # Revenue growth / profit margin from fundamentals
    if fund is not None and not fund.empty:
        f = fund.copy()
        f["avail_from"] = pd.to_datetime(f["avail_from"])
        f = f.dropna(subset=["avail_from", "symbol"]).sort_values(["symbol", "avail_from"])
        f_sub = f[["symbol", "avail_from"]].copy().rename(columns={"avail_from": "date"})
        for c in ("rev_growth_yoy", "pat_margin"):
            if c in f.columns:
                f_sub[c] = f[c].values
        if "rev_growth_yoy" not in f_sub.columns:
            f_sub["rev_growth_yoy"] = np.nan
        if "pat_margin" not in f_sub.columns:
            f_sub["pat_margin"] = np.nan
        out = pd.merge_asof(
            out.sort_values("date"),
            f_sub.sort_values("date"),
            on="date", by="symbol", direction="backward",
        ).sort_values(["symbol", "date"])
    else:
        out["rev_growth_yoy"] = np.nan
        out["pat_margin"]     = np.nan

    # Ex-dividend proximity — known downside catalyst; models must not confuse with real sell-off
    if corp_actions is not None and not corp_actions.empty:
        divs = corp_actions[corp_actions["action_type"].str.lower().str.contains(
            "dividend|div", na=False
        )][["symbol", "ex_date"]].dropna().drop_duplicates()
        divs["ex_date"] = pd.to_datetime(divs["ex_date"])
        divs = divs.sort_values(["symbol", "ex_date"])
        out["days_to_ex_div"]  = np.nan
        out["days_since_ex_div"] = np.nan
        for sym, grp in out.groupby("symbol", sort=False):
            sym_divs = divs[divs["symbol"] == sym]["ex_date"].values
            if len(sym_divs) == 0:
                continue
            dates = grp["date"].values
            next_idx = np.searchsorted(sym_divs, dates, side="left")
            prev_idx = np.where(next_idx > 0, next_idx - 1, 0)
            has_next = next_idx < len(sym_divs)
            has_prev = next_idx > 0
            d_next = np.where(
                has_next,
                (sym_divs[np.clip(next_idx, 0, len(sym_divs) - 1)] - dates) / np.timedelta64(1, "D"),
                np.nan,
            )
            d_prev = np.where(
                has_prev,
                (dates - sym_divs[prev_idx]) / np.timedelta64(1, "D"),
                np.nan,
            )
            out.loc[grp.index, "days_to_ex_div"]    = d_next
            out.loc[grp.index, "days_since_ex_div"] = d_prev
        out["is_ex_div_week"] = (out["days_to_ex_div"].between(0, 5)).astype(np.int8)
    else:
        out["days_to_ex_div"]    = np.nan
        out["days_since_ex_div"] = np.nan
        out["is_ex_div_week"]    = 0

    return out


# ──────────────────────────────────────────────────────────────────────
# Phase persistence — state machine
# ──────────────────────────────────────────────────────────────────────
def _apply_phase_persistence(phase_series: pd.Series,
                              window: int = 10,
                              min_hits: int = 8) -> pd.Series:
    """
    Rolling-window phase confirmation — no clock resets.

    At each day T, look at the last `window` trading days.  Whichever
    phase appears >= `min_hits` times in that window becomes the
    confirmed phase.  If no phase clears the threshold (e.g. an even
    split), the previously confirmed phase is kept.

    Example  window=10, min_hits=8:
        raw:       bull bull bull bear bear X bear bear bear bear bear ...
        window at day 10:  [bull bull bull bear X bear bear bear bear bear]
                            bull=3, bear=7  → no switch yet (7 < 8)
        window at day 11:  [bull bull bear X bear bear bear bear bear bear]
                            bull=2, bear=8  → switch to bear  ✓

    One interruption costs 1 extra calendar day; two cost 2 — but the
    window just slides forward, nothing resets.

    Parameters
    ----------
    window   : rolling lookback in trading days  (stock=10, market=14)
    min_hits : minimum occurrences of a phase within the window to confirm it
    """
    vals = phase_series.to_numpy(dtype=object)
    n    = len(vals)
    out  = np.empty(n, dtype=object)

    if n == 0:
        return phase_series.copy()

    current = vals[0]

    for i in range(n):
        start  = max(0, i - window + 1)
        w_vals = vals[start : i + 1]
        w_len  = len(w_vals)

        # Count occurrences of each phase in the current window
        counts: dict = {}
        for v in w_vals:
            counts[v] = counts.get(v, 0) + 1

        # Scale threshold proportionally when window isn't full yet
        # (early rows) so early data isn't stuck at the seed phase
        threshold = max(1, round(min_hits * w_len / window))

        # Phase with the most hits — switch if it clears the threshold
        best_phase, best_count = max(counts.items(), key=lambda kv: kv[1])
        if best_count >= threshold:
            current = best_phase
        # else: no dominant phase this window → keep current

        out[i] = current

    return pd.Series(out, index=phase_series.index, dtype=object)


# ──────────────────────────────────────────────────────────────────────
# Group 8a2 — Investing.com analyst EPS estimates vs actuals
# ──────────────────────────────────────────────────────────────────────

def _group_investing_eps(stock_keys: pd.DataFrame,
                         inv_eps: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Point-in-time analyst EPS surprise features from Investing.com data.

    Uses asof-merge so each row gets the most recent quarter released
    strictly before that trading date (no look-ahead).

    Features produced:
        inv_surprise_pct      — analyst EPS surprise % (last quarter)
        inv_beat              — 1 beat / 0 miss / NaN no estimate
        inv_beat_rate_4q      — fraction of last 4 quarters where beat estimate
        inv_avg_surprise_4q   — mean surprise % over last 4 quarters
        inv_rev_surprise_pct  — revenue surprise % (last quarter, often NaN)
    """
    out = stock_keys[["date", "symbol"]].copy()
    out["date"] = pd.to_datetime(out["date"])

    _cols = ("inv_surprise_pct", "inv_beat", "inv_beat_rate_4q",
             "inv_avg_surprise_4q", "inv_rev_surprise_pct")

    if inv_eps is None or inv_eps.empty:
        for c in _cols:
            out[c] = np.nan
        return out

    e = inv_eps.copy()
    e["date"] = pd.to_datetime(e["date"])
    e = e.dropna(subset=["date", "symbol"]).sort_values(["symbol", "date"])

    # Winsorise surprise — Investing.com can have data-quality outliers
    e["surprise_pct"] = e["surprise_pct"].clip(-150, 150)

    # Revenue surprise: (reported - estimate) / |estimate| * 100
    rev_mask = e["rev_estimate"].notna() & (e["rev_estimate"] != 0)
    e["rev_surprise_pct"] = np.where(
        rev_mask,
        (e["rev_reported"] - e["rev_estimate"]) / e["rev_estimate"].abs() * 100,
        np.nan,
    )
    e["rev_surprise_pct"] = e["rev_surprise_pct"].clip(-150, 150)

    # Per-symbol rolling stats (computed on the quarterly series, not daily)
    grp = e.groupby("symbol", sort=False)
    e["inv_beat_rate_4q"]    = grp["surprise_pct"].transform(
        lambda s: (s > 0).rolling(4, min_periods=1).mean()
    )
    e["inv_avg_surprise_4q"] = grp["surprise_pct"].transform(
        lambda s: s.rolling(4, min_periods=1).mean()
    )

    # Build the daily lookup table (asof-merge will propagate latest quarter)
    daily = e[["symbol", "date", "surprise_pct", "rev_surprise_pct",
               "inv_beat_rate_4q", "inv_avg_surprise_4q"]].rename(
        columns={"surprise_pct": "inv_surprise_pct",
                 "rev_surprise_pct": "inv_rev_surprise_pct"}
    ).sort_values("date")

    out = pd.merge_asof(
        out.sort_values("date"),
        daily,
        on="date", by="symbol", direction="backward",
    ).sort_values(["symbol", "date"])

    out["inv_beat"] = np.where(
        out["inv_surprise_pct"].notna(),
        (out["inv_surprise_pct"] > 0).astype(np.float32),
        np.nan,
    )

    return out[["date", "symbol"] + list(_cols)].reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────────
# Group 8b — Block deals (NSE large-block transactions)
# ──────────────────────────────────────────────────────────────────────

def _group_block_deals(stock_keys: pd.DataFrame) -> pd.DataFrame:
    """Rolling 5-day block deal aggregates per (date, symbol).

    Block deal data only exists from the date pipeline.fetch starts running
    daily; earlier rows will be NaN and LightGBM handles that natively.

    Features produced:
        block_sell_qty_5d   — total sell quantity (shares) over trailing 5 td
        block_buy_qty_5d    — total buy  quantity (shares) over trailing 5 td
        block_net_qty_5d    — buy - sell (positive = net institutional buying)
        block_sell_val_5d   — sell value (crores) over trailing 5 td
        block_buy_val_5d    — buy  value (crores) over trailing 5 td
        block_deal_flag_5d  — 1 if any deal (buy or sell) in trailing 5 td
    """
    bd_path = SILVER_TABLES.get("block_deals")
    if bd_path is None or not bd_path.exists():
        out = stock_keys[["date", "symbol"]].copy()
        for c in ("block_sell_qty_5d","block_buy_qty_5d","block_net_qty_5d",
                  "block_sell_val_5d","block_buy_val_5d","block_deal_flag_5d"):
            out[c] = np.nan
        return out

    bd = pd.read_parquet(bd_path)
    bd["date"] = pd.to_datetime(bd["date"])

    # Aggregate to (date, symbol) × deal_type
    sells = (bd[bd["deal_type"] == "SELL"]
             .groupby(["date", "symbol"])
             .agg(sell_qty=("quantity", "sum"), sell_val=("value_cr", "sum"))
             .reset_index())
    buys  = (bd[bd["deal_type"] == "BUY"]
             .groupby(["date", "symbol"])
             .agg(buy_qty=("quantity", "sum"), buy_val=("value_cr", "sum"))
             .reset_index())

    # Merge onto the full stock calendar so rolling windows work
    out = stock_keys[["date", "symbol"]].copy()
    out = out.merge(sells, on=["date", "symbol"], how="left")
    out = out.merge(buys,  on=["date", "symbol"], how="left")
    out[["sell_qty", "sell_val", "buy_qty", "buy_val"]] = \
        out[["sell_qty", "sell_val", "buy_qty", "buy_val"]].fillna(0)

    # Rolling 5-trading-day sums (per symbol, data is already sorted by symbol+date)
    out = out.sort_values(["symbol", "date"]).reset_index(drop=True)
    g = out.groupby("symbol", sort=False, group_keys=False)

    def _roll5(s):
        return s.rolling(5, min_periods=1).sum()

    out["block_sell_qty_5d"]  = g["sell_qty"].transform(_roll5)
    out["block_buy_qty_5d"]   = g["buy_qty"].transform(_roll5)
    out["block_net_qty_5d"]   = out["block_buy_qty_5d"] - out["block_sell_qty_5d"]
    out["block_sell_val_5d"]  = g["sell_val"].transform(_roll5)
    out["block_buy_val_5d"]   = g["buy_val"].transform(_roll5)
    # block_net_val_5d: net institutional value flow in crores (buy − sell).
    # Positive = net accumulation, negative = net distribution.
    # Prefer this over block_net_qty_5d (raw shares) for neural-net models.
    out["block_net_val_5d"]   = out["block_buy_val_5d"] - out["block_sell_val_5d"]
    out["block_deal_flag_5d"] = ((out["block_sell_qty_5d"] + out["block_buy_qty_5d"]) > 0
                                  ).astype(np.int8)

    # Zero-fill rows with no block deals ever (rolling sums on zeros = 0, not NaN)
    # but flag them as NaN so the model knows we have no data yet (before first fetch)
    # Heuristic: if the silver table has dates but this symbol never appears, rows are 0.
    # That is fine — genuine 0 is indistinguishable from "no data" until data accumulates.

    keep = ["date", "symbol",
            "block_sell_qty_5d","block_buy_qty_5d","block_net_qty_5d",
            "block_sell_val_5d","block_buy_val_5d","block_net_val_5d",
            "block_deal_flag_5d"]
    return out[keep].reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────────
# Group 8c — Bulk deals (NSE regular-session large-volume trades)
# ──────────────────────────────────────────────────────────────────────

def _group_bulk_deals(stock_keys: pd.DataFrame) -> pd.DataFrame:
    """Rolling 5-day bulk deal aggregates per (date, symbol).

    Bulk deal data is capped at 70 rows/day (largest deals by value) so
    coverage is sparse — only ~5–6% of F&O universe appears on a given day.
    Features are intentionally value-based rather than qty-based because the
    70-row selection criterion is deal value.

    Features produced:
        bulk_sell_val_5d   — sell value (crores) over trailing 5 td
        bulk_buy_val_5d    — buy  value (crores) over trailing 5 td
        bulk_net_val_5d    — buy - sell value (positive = net buying)
        bulk_sell_flag_5d  — 1 if any sell deal in trailing 5 td
        bulk_buy_flag_5d   — 1 if any buy  deal in trailing 5 td
    """
    bd_path = SILVER_TABLES.get("bulk_deals")
    if bd_path is None or not bd_path.exists():
        out = stock_keys[["date", "symbol"]].copy()
        for c in ("bulk_sell_val_5d", "bulk_buy_val_5d", "bulk_net_val_5d",
                  "bulk_sell_flag_5d", "bulk_buy_flag_5d"):
            out[c] = np.nan
        return out

    bd = pd.read_parquet(bd_path)
    bd["date"] = pd.to_datetime(bd["date"])

    sells = (bd[bd["deal_type"] == "SELL"]
             .groupby(["date", "symbol"])
             .agg(sell_val=("value_cr", "sum"), sell_cnt=("value_cr", "count"))
             .reset_index())
    buys  = (bd[bd["deal_type"] == "BUY"]
             .groupby(["date", "symbol"])
             .agg(buy_val=("value_cr", "sum"), buy_cnt=("value_cr", "count"))
             .reset_index())

    out = stock_keys[["date", "symbol"]].copy()
    out = out.merge(sells, on=["date", "symbol"], how="left")
    out = out.merge(buys,  on=["date", "symbol"], how="left")
    out[["sell_val", "sell_cnt", "buy_val", "buy_cnt"]] = \
        out[["sell_val", "sell_cnt", "buy_val", "buy_cnt"]].fillna(0)

    out = out.sort_values(["symbol", "date"]).reset_index(drop=True)
    g = out.groupby("symbol", sort=False, group_keys=False)

    def _roll5(s):
        return s.rolling(5, min_periods=1).sum()

    out["bulk_sell_val_5d"]  = g["sell_val"].transform(_roll5)
    out["bulk_buy_val_5d"]   = g["buy_val"].transform(_roll5)
    out["bulk_net_val_5d"]   = out["bulk_buy_val_5d"] - out["bulk_sell_val_5d"]
    out["bulk_sell_flag_5d"] = (g["sell_cnt"].transform(_roll5) > 0).astype(np.int8)
    out["bulk_buy_flag_5d"]  = (g["buy_cnt"].transform(_roll5) > 0).astype(np.int8)

    keep = ["date", "symbol",
            "bulk_sell_val_5d", "bulk_buy_val_5d", "bulk_net_val_5d",
            "bulk_sell_flag_5d", "bulk_buy_flag_5d"]
    return out[keep].reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────────
# Group 9a — Stock phase (per symbol)
# ──────────────────────────────────────────────────────────────────────
def _group_stock_phase(stock: pd.DataFrame) -> pd.DataFrame:
    """
    Classify each (date, symbol) row into one of five phases based on
    trend structure (HH/HL or LH/LL) and price level relative to moving
    averages and 52-week range.

    Phases (applied in priority order — bull overrides distribution etc.):
        bull         — HH + HL + above 200MA
        bear         — LH + LL + below 200MA
        recovery     — HL (not yet HH) + above 200MA + recovered ≥35% of 52w range
        distribution — LH (not yet LL) + above 200MA  (topping, bull weakening)
        consolidation — everything else
    """
    g = stock.groupby("symbol", sort=False, group_keys=False)

    # 200-day MA
    ma200       = g["close"].transform(lambda s: s.rolling(200, min_periods=100).mean())
    above_200ma = stock["close"] > ma200

    # 52-week range
    hi_52w       = g["close"].transform(lambda s: s.rolling(252, min_periods=120).max())
    lo_52w       = g["close"].transform(lambda s: s.rolling(252, min_periods=120).min())
    rng_52w      = (hi_52w - lo_52w).replace(0, np.nan)
    dd_52w       = (stock["close"] - hi_52w) / hi_52w.replace(0, np.nan)
    recovery_52w = (stock["close"] - lo_52w) / rng_52w

    # 20-day swing high/low — compare current window to the one 20 days ago
    rh20 = g["close"].transform(lambda s: s.rolling(20, min_periods=10).max())
    rl20 = g["close"].transform(lambda s: s.rolling(20, min_periods=10).min())
    ph20 = g["close"].transform(lambda s: s.rolling(20, min_periods=10).max().shift(20))
    pl20 = g["close"].transform(lambda s: s.rolling(20, min_periods=10).min().shift(20))

    higher_high = rh20 > ph20   # recent swing high > prior → upward structure
    higher_low  = rl20 > pl20
    lower_high  = rh20 < ph20   # recent swing high < prior → downward structure
    lower_low   = rl20 < pl20

    # Raw phase signal (can flip day-to-day)
    phase_raw = pd.Series("consolidation", index=stock.index, dtype=object)
    phase_raw[lower_high & ~lower_low & above_200ma]                            = "distribution"
    phase_raw[higher_low & ~higher_high & above_200ma & (recovery_52w > 0.35)] = "recovery"
    phase_raw[lower_high & lower_low   & ~above_200ma]                          = "bear"
    phase_raw[higher_high & higher_low & above_200ma]                           = "bull"

    # Persistence filter — phase confirmed after 8 qualifying days (tolerance: 2 breaks)
    # Applied per symbol so the state machine doesn't bleed across symbols
    phase = (
        pd.concat([stock["symbol"], phase_raw.rename("phase")], axis=1)
        .groupby("symbol", sort=False)["phase"]
        .transform(lambda s: _apply_phase_persistence(s, window=10, min_hits=8))
    )

    out = stock[["date", "symbol"]].copy()
    out["stock_phase"]        = phase.values
    out["stock_dd_52w"]       = dd_52w.values
    out["stock_recovery_52w"] = recovery_52w.values
    out["higher_high"]        = higher_high.astype(np.int8).values
    out["higher_low"]         = higher_low.astype(np.int8).values
    out["lower_high"]         = lower_high.astype(np.int8).values
    out["lower_low"]          = lower_low.astype(np.int8).values
    return out


# ──────────────────────────────────────────────────────────────────────
# Group 9b — Market Regime (Nifty-based, one row per date)
# ──────────────────────────────────────────────────────────────────────
def _group_regime(indices: pd.DataFrame) -> pd.DataFrame:
    """One row per date with market-wide regime features."""
    idx = indices.copy()
    idx["date"] = pd.to_datetime(idx["date"])

    nifty = (idx[idx["index_name"] == "Nifty 50"]
             .sort_values("date")[["date", "close"]]
             .rename(columns={"close": "nifty_close"}))
    nifty["nifty_ret_5d"]  = nifty["nifty_close"].pct_change(5)
    nifty["nifty_ret_20d"] = nifty["nifty_close"].pct_change(20)

    # Trend strength vs long-term anchors
    nifty["nifty_above_200ma"]   = (nifty["nifty_close"]
                                    / nifty["nifty_close"].rolling(200, min_periods=100).mean() - 1)
    nifty["nifty_dist_52w_high"] = (nifty["nifty_close"].rolling(252, min_periods=120).max()
                                    - nifty["nifty_close"]) / nifty["nifty_close"]
    nifty["nifty_dist_52w_low"]  = (nifty["nifty_close"]
                                    - nifty["nifty_close"].rolling(252, min_periods=120).min()
                                    ) / nifty["nifty_close"]

    # Momentum acceleration: short vs long momentum (positive = trend strengthening)
    nifty["nifty_momentum_accel"] = nifty["nifty_ret_5d"] - nifty["nifty_ret_20d"]

    # Nifty realised volatility
    nifty["nifty_hv_20"] = (nifty["nifty_close"].pct_change()
                            .rolling(20, min_periods=10).std() * np.sqrt(252))

    vix = (idx[idx["index_name"] == "India VIX"]
           .sort_values("date")[["date", "close"]]
           .rename(columns={"close": "vix_level"}))
    vix["vix_chg_5d"]    = vix["vix_level"].diff(5)
    vix["vix_chg_10d"]   = vix["vix_level"].diff(10)
    vix["vix_rank_252d"] = _pct_rank(vix["vix_level"], 252)
    # VIX expanding: fear building (rising over 10 days)
    vix["vix_expanding"] = (vix["vix_chg_10d"] > 0).astype(np.int8)

    # ── Market phase via HH/HL structure on Nifty (30-day rolling) ──────
    # Computed on nifty BEFORE merging with vix so the index stays aligned.
    # 30-day windows (vs 20-day per-stock) — index moves are smoother.
    nc = nifty["nifty_close"]
    rh30 = nc.rolling(30, min_periods=15).max()
    rl30 = nc.rolling(30, min_periods=15).min()
    ph30 = rh30.shift(30)
    pl30 = rl30.shift(30)

    mkt_hi_52w      = nc.rolling(252, min_periods=120).max()
    mkt_lo_52w      = nc.rolling(252, min_periods=120).min()
    mkt_rng         = (mkt_hi_52w - mkt_lo_52w).replace(0, np.nan)
    mkt_rec_52w     = (nc - mkt_lo_52w) / mkt_rng
    mkt_above_200ma = nc > nc.rolling(200, min_periods=100).mean()

    mkt_hh = rh30 > ph30
    mkt_hl = rl30 > pl30
    mkt_lh = rh30 < ph30
    mkt_ll = rl30 < pl30

    mkt_phase_raw = pd.Series("consolidation", index=nifty.index, dtype=object)
    mkt_phase_raw[mkt_lh & ~mkt_ll & mkt_above_200ma]                         = "distribution"
    mkt_phase_raw[mkt_hl & ~mkt_hh & mkt_above_200ma & (mkt_rec_52w > 0.40)] = "recovery"
    mkt_phase_raw[mkt_lh & mkt_ll  & ~mkt_above_200ma]                        = "bear"
    mkt_phase_raw[mkt_hh & mkt_hl  & mkt_above_200ma]                         = "bull"

    # Persistence filter — market phase stickier than stock phase (12 qualifying days,
    # tolerance 2 breaks ≈ up to 3 weeks of real calendar days)
    nifty["market_phase"] = _apply_phase_persistence(
        mkt_phase_raw, window=14, min_hits=12
    ).values

    # Single merge — market_phase carried over from nifty
    regime = nifty.merge(vix, on="date", how="outer").sort_values("date").reset_index(drop=True)
    regime["market_phase"] = regime["market_phase"].fillna("consolidation")

    # ── Regime field — derived from the (better) market_phase classifier ─
    # The old (vix>22 AND nifty<-3%) AND/AND rule almost never fired bear,
    # so dn_bear_lgbm.pkl was never trained and bear-phase predictions used
    # regime-blind base models.  Now mirror market_phase (which uses HH/LL
    # market structure + persistence filter and correctly identifies bears),
    # mapping "consolidation" → "range" to preserve existing file naming.
    _MARKET_TO_REGIME = {"bull": "bull", "bear": "bear",
                         "consolidation": "range",
                         "recovery": "bull", "distribution": "bear"}
    regime["regime"] = regime["market_phase"].map(_MARKET_TO_REGIME).fillna("range")
    regime["regime_duration_days"] = _regime_duration(regime["regime"])
    regime["regime_changed_5d"] = (
        regime["regime"] != regime["regime"].shift(5)
    ).astype(np.int8)

    # Expiry-week flag
    regime["is_expiry_week"] = regime["date"].apply(_is_expiry_week).astype(np.int8)

    return regime[["date",
                   "nifty_ret_5d", "nifty_ret_20d",
                   "nifty_above_200ma", "nifty_dist_52w_high", "nifty_dist_52w_low",
                   "nifty_momentum_accel", "nifty_hv_20",
                   "vix_level", "vix_rank_252d", "vix_chg_5d", "vix_expanding",
                   "regime", "regime_duration_days", "regime_changed_5d",
                   "market_phase", "is_expiry_week"]]


def _regime_duration(regime_series: pd.Series) -> pd.Series:
    """Consecutive days the current regime has been active (resets on change)."""
    out = np.zeros(len(regime_series), dtype=int)
    count = 0
    prev = None
    for i, r in enumerate(regime_series):
        if r == prev:
            count += 1
        else:
            count = 1
            prev = r
        out[i] = count
    return pd.Series(out, index=regime_series.index)


def _is_expiry_week(d: pd.Timestamp) -> bool:
    """True if d is within 4 calendar days of the last Thursday of the month."""
    last_day = (d + pd.offsets.MonthEnd(0)).date()
    last_dt  = pd.Timestamp(last_day)
    last_thu = last_dt - pd.Timedelta(days=(last_dt.weekday() - 3) % 7)
    return 0 <= (last_thu - d).days <= 4


# ──────────────────────────────────────────────────────────────────────
# Group 10 — Sector relative strength
# ──────────────────────────────────────────────────────────────────────
def _group_sector(stock: pd.DataFrame, indices: pd.DataFrame) -> pd.DataFrame:
    """Sector index 5d return + stock alpha vs sector."""
    idx = indices.copy()
    idx["date"] = pd.to_datetime(idx["date"])
    sector_dfs = {}
    for sec, idx_name in SECTOR_INDICES.items():
        sub = idx[idx["index_name"] == idx_name].sort_values("date")
        if sub.empty:
            continue
        sub = sub[["date", "close"]].copy()
        sub[f"sector_ret_5d_{sec}"] = sub["close"].pct_change(5)
        sector_dfs[sec] = sub[["date", f"sector_ret_5d_{sec}"]]

    s = stock[["date", "symbol", "close"]].copy()
    s["date"] = pd.to_datetime(s["date"])
    s = s.sort_values(["symbol", "date"])
    s["stock_ret_5d"] = s.groupby("symbol", sort=False)["close"].pct_change(5)
    s["sector"] = s["symbol"].map(TICKER_SECTOR).fillna("other")

    # Build a single sector_ret_5d column per row
    s["sector_ret_5d"] = np.nan
    for sec, sdf in sector_dfs.items():
        mask = s["sector"] == sec
        if not mask.any():
            continue
        m = s.loc[mask, ["date"]].merge(sdf, on="date", how="left")
        s.loc[mask, "sector_ret_5d"] = m[f"sector_ret_5d_{sec}"].values

    s["rel_strength_sector"] = s["stock_ret_5d"] - s["sector_ret_5d"]
    return s[["date", "symbol", "sector_ret_5d", "rel_strength_sector"]]


# ──────────────────────────────────────────────────────────────────────
# Group 11 — Path Quality (from past realized label windows)
# ──────────────────────────────────────────────────────────────────────
def _group_path_quality(stock: pd.DataFrame) -> pd.DataFrame:
    """
    Lagged path quality features derived from realized 5-day OHLC swing windows.

    Computed directly from silver OHLC data (no labels dependency).
        upside_t5   = max(high[t+1..t+5]) / close[t] − 1
        downside_t5 = min(low[t+1..t+5])  / close[t] − 1

    Features returned (all lagged ≥6 rows — no look-ahead):
        net_swing_lag1   upside_t5 + downside_t5 from 6 rows back
        path_asym_lag1   upside / |downside| from 6 rows back
        net_swing_lag2   net_swing from 11 rows back
        net_swing_roll20 20-day rolling mean of net_swing (from shift(6))
    """
    df = stock[["date", "symbol", "close", "high", "low"]].copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)

    g = df.groupby("symbol", sort=False, group_keys=False)

    # s.rolling(5).max().shift(-5) at position i = max(s[i+1 .. i+5])
    hi5 = g["high"].transform(lambda s: s.rolling(5, min_periods=5).max().shift(-5))
    lo5 = g["low"].transform(lambda s: s.rolling(5, min_periods=5).min().shift(-5))

    df["upside_t5"]   = hi5 / df["close"] - 1
    df["downside_t5"] = lo5 / df["close"] - 1
    df["_net_swing"]  = df["upside_t5"] + df["downside_t5"]
    df["_path_asym"]  = df["upside_t5"] / (df["downside_t5"].abs() + 0.01)

    out = df[["date", "symbol"]].copy()
    out["net_swing_lag1"]   = g["_net_swing"].transform(lambda s: s.shift(6))
    out["path_asym_lag1"]   = g["_path_asym"].transform(lambda s: s.shift(6))
    out["net_swing_lag2"]   = g["_net_swing"].transform(lambda s: s.shift(11))
    out["net_swing_roll20"] = g["_net_swing"].transform(
        lambda s: s.shift(6).rolling(20, min_periods=5).mean()
    )
    return out


def _path_quality_for_date(stock: pd.DataFrame, asof: pd.Timestamp) -> pd.DataFrame:
    """
    Path quality features for a single prediction date (daily inference path).

    Uses OHLC data whose 5-day forward window has fully closed before `asof`
    — conservative cutoff: D <= asof - 8 calendar days.

    Returns one row per symbol with the same columns as _group_path_quality.
    """
    df = stock[["date", "symbol", "close", "high", "low"]].copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)

    g = df.groupby("symbol", sort=False, group_keys=False)
    hi5 = g["high"].transform(lambda s: s.rolling(5, min_periods=5).max().shift(-5))
    lo5 = g["low"].transform(lambda s: s.rolling(5, min_periods=5).min().shift(-5))
    df["upside_t5"]   = hi5 / df["close"] - 1
    df["downside_t5"] = lo5 / df["close"] - 1
    df["_net_swing"] = df["upside_t5"] + df["downside_t5"]
    df["_path_asym"] = df["upside_t5"] / (df["downside_t5"].abs() + 0.01)

    cutoff = asof - pd.Timedelta(days=8)   # ~5-6 trading days back
    safe = df[df["date"] <= cutoff]

    rows = []
    for sym, grp in safe.groupby("symbol", sort=False):
        grp = grp.sort_values("date")
        n = len(grp)
        rows.append({
            "date":             asof,
            "symbol":           sym,
            "net_swing_lag1":   grp.iloc[-1]["_net_swing"]  if n >= 1 else np.nan,
            "path_asym_lag1":   grp.iloc[-1]["_path_asym"]  if n >= 1 else np.nan,
            "net_swing_lag2":   grp.iloc[-2]["_net_swing"]  if n >= 2 else np.nan,
            "net_swing_roll20": grp["_net_swing"].tail(20).mean() if n >= 5 else np.nan,
        })
    result = pd.DataFrame(rows)
    if not result.empty:
        result["date"] = pd.to_datetime(result["date"])
    return result


# ──────────────────────────────────────────────────────────────────────
# Compound interactions
# ──────────────────────────────────────────────────────────────────────
def _group_fii_proxy_score(base: pd.DataFrame) -> pd.DataFrame:
    """Score each (date, symbol) with the FII accumulation proxy model.

    Returns a DataFrame with columns [date, symbol, fii_accum_prob].
    Rows where the model artifact is absent or features are missing get NaN —
    LightGBM in the main production model handles NaN natively.
    """
    proxy_path = MODEL_DIR / "fii_proxy_lgbm.pkl"
    if not proxy_path.exists():
        print("[features] fii_proxy_lgbm.pkl not found — fii_accum_prob will be NaN")
        return base[["date", "symbol"]].assign(fii_accum_prob=np.nan)

    with open(proxy_path, "rb") as f:
        art = pickle.load(f)

    features      = art["features"]
    regime_models = art.get("regime_models", {})

    X_all = base.reindex(columns=features)
    prob  = np.full(len(base), np.nan)

    regimes = base["regime"].unique() if "regime" in base.columns else []
    for regime in regimes:
        if regime is None or (isinstance(regime, float) and np.isnan(regime)):
            continue
        mask = (base["regime"] == regime).values
        if mask.sum() == 0:
            continue
        rm = regime_models.get(regime)
        if rm is not None:
            m, cal = rm["model"], rm["calibrator"]
        else:
            m, cal = art["model"], art["calibrator"]
        raw        = m.predict_proba(X_all[mask])[:, 1]
        prob[mask] = cal.predict(raw)

    out = base[["date", "symbol"]].copy()
    out["fii_accum_prob"] = prob
    return out


def _compound_interactions(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)

    def _has(*cols): return all(c in df.columns for c in cols)

    if _has("breakout_quality", "vol_ratio_20d", "delivery_ratio_20d"):
        out["mega_breakout"] = (
            df["breakout_quality"].fillna(0)
            * df["vol_ratio_20d"].fillna(0)
            * df["delivery_ratio_20d"].fillna(0)
        )
    if _has("adl_divergence", "w_close_pos"):
        bullish_weekly = (df["w_close_pos"] > 0.7).astype(int)
        positive_adl   = (df["adl_divergence"] > 0).astype(int)
        out["bull_confluence"] = bullish_weekly * positive_adl

    if _has("is_pre_earnings"):
        out["earnings_vol_setup"] = df["is_pre_earnings"].fillna(0)

    if _has("w_inside_bar", "squeeze_rank"):
        out["double_squeeze"] = (
            df["w_inside_bar"].fillna(0).astype(int)
            * (df["squeeze_rank"].fillna(1) < 0.2).astype(int)
        )

    # FII flow × market direction divergence
    # +2 = FII buying into a falling market   (bullish accumulation)
    # -2 = FII selling into a rising market   (bearish — retail-driven rally)
    #  0 = FII and market aligned (normal)
    if _has("fii_cash_net_5d", "nifty_ret_5d"):
        sgn_fii    = np.sign(df["fii_cash_net_5d"].fillna(0))
        sgn_nifty  = np.sign(df["nifty_ret_5d"].fillna(0))
        out["fii_nifty_divergence"] = (sgn_fii - sgn_nifty).astype(np.int8)

    if _has("resistance_zone_strength", "consolidating_at_resistance",
            "support_zone_strength",    "consolidating_at_support"):
        out["zone_energy"] = (
            df["resistance_zone_strength"].fillna(0) * df["consolidating_at_resistance"].fillna(0)
          + df["support_zone_strength"].fillna(0)    * df["consolidating_at_support"].fillna(0)
        )

    # FII buy on quiet volume above 50DMA — proxy-model top pattern
    if _has("vol_ratio_5d", "dist_above_50ma"):
        out["quiet_accum"] = (
            (df["vol_ratio_5d"].fillna(1) < 0.9) &
            (df["dist_above_50ma"].fillna(0) > 0)
        ).astype(np.int8)

    # ── dn_5_xl-specific compound interactions ────────────────────────────────────
    # stretch_beta: overbought stock WITH high market beta = maximum vulnerability.
    # When Nifty drops, high-beta stocks fall more; when already stretched above 50MA,
    # the mean reversion is even more violent. This product captures both conditions.
    if _has("dist_above_50ma_z", "beta_nifty_20d"):
        stretch = df["dist_above_50ma_z"].fillna(0).clip(lower=0)
        beta    = df["beta_nifty_20d"].fillna(1.0).clip(lower=0)
        out["stretch_beta"] = stretch * beta

    # dn_smart_short: overbought stock with short positions actively building.
    # Smart money shorts stretched stocks while retail continues to hold → setup
    # for a flush when the macro catalyst arrives.
    if _has("dist_above_50ma_z", "short_buildup_streak"):
        stretch = df["dist_above_50ma_z"].fillna(0).clip(lower=0)
        out["dn_smart_short"] = stretch * df["short_buildup_streak"].fillna(0)

    # dn_basis_stretch: overbought stock with basis at multi-month low.
    # basis_rank_60d near 0 = futures basis is at the bottom of its 60d range →
    # smart-money futures traders reducing longs relative to recent history.
    # Uses rank (always populated) rather than raw negative basis (only 8% of rows).
    if _has("dist_above_50ma_z", "basis_rank_60d"):
        stretch       = df["dist_above_50ma_z"].fillna(0).clip(lower=0)
        low_basis_rank = (1.0 - df["basis_rank_60d"].fillna(0.5))  # high when basis at 60d low
        out["dn_basis_stretch"] = stretch * low_basis_rank

    # put_activity_stretch: puts being actively bought on a stretched stock.
    # Options market front-running a downmove on an already overbought stock.
    if _has("dist_above_50ma_z", "put_call_vol_rank_60d"):
        stretch      = df["dist_above_50ma_z"].fillna(0).clip(lower=0)
        put_activity = df["put_call_vol_rank_60d"].fillna(0.5)
        out["put_activity_stretch"] = stretch * put_activity

    # dn_oi_crowded: high OI z-score on a stretched stock → crowded longs.
    # When the unwind starts, dealer hedging forces further selling (gamma cascade).
    if _has("dist_above_50ma_z", "fut_oi_z_60d"):
        stretch = df["dist_above_50ma_z"].fillna(0).clip(lower=0)
        oi_z    = df["fut_oi_z_60d"].fillna(0).clip(lower=0)
        out["dn_oi_crowded"] = stretch * oi_z

    # dn_exhaustion: overbought stock with extended green-day streak.
    # A stock making its 6th+ consecutive green day while stretched above 50MA is
    # the classic momentum exhaustion pattern — most likely to reverse sharply.
    if _has("dist_above_50ma_z", "consecutive_green_days"):
        stretch    = df["dist_above_50ma_z"].fillna(0).clip(lower=0, upper=5)
        green_days = df["consecutive_green_days"].fillna(0).clip(upper=15)  # cap at 15; beyond=data issue
        out["dn_exhaustion"] = stretch * green_days

    # dn_macro_stock_timing: macro risk (VIX rising) × individual stock vulnerability.
    # Combines the market timer signal (VIX acceleration) with stock-specific exposure
    # (stretch × beta) — the "right stock at the right macro moment" composite.
    if _has("vix_chg_5d", "dist_above_50ma_z", "beta_nifty_20d"):
        vix_up  = df["vix_chg_5d"].fillna(0).clip(lower=0)      # only when VIX rising
        stretch = df["dist_above_50ma_z"].fillna(0).clip(lower=0)
        beta    = df["beta_nifty_20d"].fillna(1.0).clip(lower=0)
        out["dn_macro_stock_timing"] = vix_up * stretch * beta

    # fii_put_rush_on_stretch: FII accelerating put accumulation × stock extended above 50MA.
    # Identifies which stocks face the highest downside risk when FII is actively hedging
    # via put buying market-wide. On high-FII-put-rush days, the most stretched stocks
    # bear the most risk — this compound captures that cross-sectional variation.
    # fii_put_long_stk_chg_5d is market-wide (same for all stocks on a date); the
    # per-stock dist_above_50ma_z determines which stocks are most at risk that day.
    if _has("fii_put_long_stk_chg_5d", "dist_above_50ma_z"):
        # Rank the 5d FII put change within the val history (rolling 252d window per date)
        # — use raw clip+scale since this column has no per-symbol groupby context
        # fii_put_long_stk_chg_5d is date-level (same for all stocks on a date).
        # Compute expanding p90 on UNIQUE dates only — avoids inflating the window
        # N-fold (once per stock) when the panel is sorted by (symbol, date).
        _daily_pr = (
            df.groupby("date")["fii_put_long_stk_chg_5d"]
            .first().sort_index().fillna(0).clip(lower=0)
        )
        _p90_daily = _daily_pr.expanding(min_periods=20).quantile(0.90).replace(0, np.nan)
        put_rush = df["fii_put_long_stk_chg_5d"].fillna(0).clip(lower=0)
        p90      = df["date"].map(_p90_daily)
        put_rush_norm = (put_rush / p90).fillna(0).clip(upper=3.0)
        # clip stretch at 5 σ — penny-stock spikes can reach 100 σ and explode the compound
        stretch = df["dist_above_50ma_z"].fillna(0).clip(lower=0, upper=5.0)
        out["fii_put_rush_on_stretch"] = (put_rush_norm * stretch).clip(upper=5.0)

    # iv_vs_hv: ATM implied vol / realized historical vol.
    # > 1 = options market pricing a bigger move than recent history suggests —
    # informed fear premium.  A high ratio on an overbought stock (stretch_beta > 0)
    # is a strong dn_5_xl signal: smart money is actively buying protection.
    if _has("atm_iv", "hv_20"):
        iv  = df["atm_iv"].fillna(np.nan)
        hv  = df["hv_20"].replace(0, np.nan)
        out["iv_vs_hv"] = (iv / hv).clip(upper=5.0)   # cap at 5× to suppress outliers

    # ── Phase 2 "calm before storm" features ──────────────────────────
    # range_vs_vol_implied: how tight is the 10-day range relative to what
    # realized volatility would predict?  < 1 = unusually compressed (coiled spring).
    if _has("tight_range_10d", "hv_20"):
        vol_implied_10d = (df["hv_20"].fillna(0.20) / np.sqrt(252)) * np.sqrt(10)
        vol_implied_10d = vol_implied_10d.replace(0, np.nan)
        out["range_vs_vol_implied"] = df["tight_range_10d"].fillna(0) / vol_implied_10d

    # vol_recovery_score: stock in deep drawdown + volume expanding.
    # High score = "money flowing in while price is still depressed" = recovery setup.
    if _has("dist_52w_high", "vol_ratio_20d"):
        dip_depth = (-df["dist_52w_high"].fillna(0)).clip(lower=0, upper=0.5)
        vol_surge = (df["vol_ratio_20d"].fillna(1).clip(lower=1.0, upper=3.0) - 1.0) / 2.0
        out["vol_recovery_score"] = dip_depth * vol_surge

    # dip_vol_spike_5d: recent down-move on elevated volume = capitulation / washout.
    # Combines magnitude of 5-day drop with relative volume surge.
    if _has("ret_5d", "vol_ratio_20d"):
        dip      = (-df["ret_5d"].fillna(0)).clip(lower=0)
        vol_mult = df["vol_ratio_20d"].fillna(1).clip(lower=1.0, upper=4.0)
        out["dip_vol_spike_5d"] = dip * vol_mult

    # fii_macro_stock_divergence: FII is a net buyer at the market level but the
    # stock is still down — smart-money accumulation while retail is selling.
    # Positive = bullish divergence; negative = FII selling into a rising stock.
    if _has("fii_cash_net_5d", "ret_5d"):
        fii_norm  = df["fii_cash_net_5d"].fillna(0).clip(-8000, 8000) / 2000
        stock_ret = df["ret_5d"].fillna(0)
        out["fii_macro_stock_divergence"] = fii_norm * (-stock_ret)

    # ── SHAP-derived interaction crosses (clean_up_5_liq top-10 pairs) ──────────
    # Top pairwise SHAP interactions from analysis/shap_interactions.py.
    # These capture joint/conditional signal invisible to univariate AUC ranking.
    # regime_duration_days features: scales 1-252d, capped at 252 to suppress outliers.
    # days_since/to_earnings: capped at 90 (> 90d is "far from earnings", flat signal).
    # fii_cash_net_30d: normalized by /1000 so the product stays in a tree-friendly range.

    # 1. FII 30d accumulation × earnings recency
    #    Interpretation: sustained FII buying is most predictive just after earnings
    #    (re-rating period) and just before (pre-positioning).
    if _has("fii_cash_net_30d", "days_since_last_earnings"):
        fii30 = df["fii_cash_net_30d"].fillna(0).clip(-50_000, 50_000) / 1000
        d_since = df["days_since_last_earnings"].fillna(45).clip(1, 90)
        out["fii_cash_net_30d_x_days_since_last_earn"] = fii30 * d_since

    # 2. Regime maturity × options OI ratio
    #    Long-running regimes (trending strongly) amplify options positioning signal.
    if _has("regime_duration_days", "opt_oi_ratio_20d"):
        rdur = df["regime_duration_days"].fillna(1).clip(1, 252) / 100
        oir  = df["opt_oi_ratio_20d"].fillna(1).clip(0, 5)
        out["regime_duration_days_x_opt_oi_ratio_20d"] = rdur * oir

    # 3. Options OI ratio × earnings recency
    #    Options positioning (call vs put OI imbalance) is most informative near earnings.
    if _has("opt_oi_ratio_20d", "days_since_last_earnings"):
        oir     = df["opt_oi_ratio_20d"].fillna(1).clip(0, 5)
        d_since = df["days_since_last_earnings"].fillna(45).clip(1, 90)
        out["opt_oi_ratio_20d_x_days_since_last_earn"] = oir * d_since

    # 4. Regime maturity × vol regime rank
    #    A long-running bull regime in a low-VIX environment ≠ a young regime in panic vol.
    if _has("regime_duration_days", "vix_rank_252d"):
        rdur    = df["regime_duration_days"].fillna(1).clip(1, 252) / 100
        vix_rk  = df["vix_rank_252d"].fillna(0.5)
        out["regime_duration_days_x_vix_rank_252d"] = rdur * vix_rk

    # 5. Market 20d return × vol rank
    #    Rising market in low vol = sustainable; rising market in high vol = fragile.
    if _has("nifty_ret_20d", "vix_rank_252d"):
        nret   = df["nifty_ret_20d"].fillna(0).clip(-0.3, 0.3)
        vix_rk = df["vix_rank_252d"].fillna(0.5)
        out["nifty_ret_20d_x_vix_rank_252d"] = nret * vix_rk

    # 6. Market above 200MA (0/1) × distance from 52-week high
    #    When market is in structural uptrend (above 200MA) AND near 52w high,
    #    individual stocks have the strongest tailwind for continuation moves.
    if _has("nifty_above_200ma", "nifty_dist_52w_high"):
        above = df["nifty_above_200ma"].fillna(0)
        dist  = df["nifty_dist_52w_high"].fillna(-0.15).clip(-0.5, 0)
        out["nifty_above_200ma_x_nifty_dist_52w_high"] = above * (1.0 + dist)

    # 7. Regime maturity × days to expiry
    #    Options expiry effect is strongest in regimes that have persisted
    #    (directional follow-through vs pin risk).
    if _has("regime_duration_days", "days_to_expiry"):
        rdur = df["regime_duration_days"].fillna(1).clip(1, 252) / 100
        dte  = df["days_to_expiry"].fillna(15).clip(1, 35) / 35
        out["regime_duration_days_x_days_to_expiry"] = rdur * dte

    # 8. Regime maturity × earnings recency
    #    Regimes are most powerful when they have built momentum AND an earnings
    #    catalyst just cleared the field (post-earnings continuation).
    if _has("regime_duration_days", "days_since_last_earnings"):
        rdur    = df["regime_duration_days"].fillna(1).clip(1, 252) / 100
        d_since = df["days_since_last_earnings"].fillna(45).clip(1, 90)
        out["regime_duration_days_x_days_since_last_earn"] = rdur * d_since

    # 9. Market distance from 52w high × vol rank
    #    A market near its 52w high in low vol is a totally different setup than
    #    near its high in high vol (exhaustion vs breakout).
    if _has("nifty_dist_52w_high", "vix_rank_252d"):
        dist   = df["nifty_dist_52w_high"].fillna(-0.15).clip(-0.5, 0)
        vix_rk = df["vix_rank_252d"].fillna(0.5)
        out["nifty_dist_52w_high_x_vix_rank_252d"] = dist * vix_rk

    # 10. Regime maturity × days to next earnings
    #     Upcoming earnings resets regime direction; a long-running regime with
    #     imminent earnings is less reliable than one with clear runway ahead.
    if _has("regime_duration_days", "days_to_next_earnings"):
        rdur   = df["regime_duration_days"].fillna(1).clip(1, 252) / 100
        d_next = df["days_to_next_earnings"].fillna(45).clip(1, 90)
        out["regime_duration_days_x_days_to_next_earning"] = rdur * d_next

    return out


# ──────────────────────────────────────────────────────────────────────
# Group 13 — Rolling Nifty Beta
# ──────────────────────────────────────────────────────────────────────
def _add_nifty_beta(base: pd.DataFrame, indices: pd.DataFrame) -> pd.DataFrame:
    """
    Add rolling 20d and 60d beta of each stock to Nifty 50.

    Beta = cov(stock_ret, nifty_ret) / var(nifty_ret).
    Separates market-driven moves from stock-specific signal — a 3% gain
    on a +2% Nifty day (beta=1.5) is very different from a 3% gain on a
    flat market (stock-specific alpha).

    Works on a slim 3-column frame to avoid groupby overhead on the full
    160-column base DataFrame.
    """
    idx = indices.copy()
    idx["date"] = pd.to_datetime(idx["date"])
    nifty_ret = (
        idx[idx["index_name"] == "Nifty 50"]
        .sort_values("date")
        .assign(nifty_ret_1d=lambda d: d["close"].pct_change(1))
        [["date", "nifty_ret_1d"]]
    )

    # Slim frame: only what's needed for the cov/var computation
    tmp = base[["date", "symbol", "ret_1d"]].copy()
    tmp = tmp.merge(nifty_ret, on="date", how="left")

    g = tmp.groupby("symbol", sort=False, group_keys=False)

    new_cols: dict[str, pd.Series] = {}
    for w, label in [(20, "20d"), (60, "60d")]:
        def _beta(grp, w=w):
            cov = grp["ret_1d"].rolling(w, min_periods=w // 2).cov(grp["nifty_ret_1d"])
            var = grp["nifty_ret_1d"].rolling(w, min_periods=w // 2).var()
            return cov / var.replace(0, np.nan)
        new_cols[f"beta_nifty_{label}"] = (
            g.apply(_beta, include_groups=False).reset_index(level=0, drop=True)
        )

    return pd.concat([base, pd.DataFrame(new_cols, index=base.index)], axis=1)


# ──────────────────────────────────────────────────────────────────────
# Build orchestrator
# ──────────────────────────────────────────────────────────────────────
FEATURE_COLS = [
    # Group 1
    "ret_1d","ret_3d","ret_5d","ret_10d","ret_20d",
    "above_200ma","dist_52w_high","dist_52w_low",
    "rsi_14","macd_hist","close_position",
    "rsi_overbought_days",           # consecutive days RSI ≥ 70 (exhaustion duration)
    # Return quality
    "win_rate_20d","sharpe_20d","max_dd_20d",
    # Group 2
    "atr_14","hv_20","vol_expansion",
    "vol_ratio_5d","vol_ratio_20d",
    "amihud_illiquidity","days_since_vol_surge",
    "delivery_pct","delivery_ratio_20d","delivery_streak",
    # Institutional participation proxy (avg INR per trade) — ratio + z-score only
    "avg_trade_value_ratio_20d","avg_trade_value_zscore_60d",
    # Group 3
    "bb_width","bb_squeeze","squeeze_rank","dist_from_bb_upper",
    "atr_compression","days_in_squeeze",
    "squeeze_breakout","breakout_vol_confirm","breakout_quality",
    "adl_divergence","obv_slope_10d",
    "hi52_breakout","tight_range_10d","price_acceleration",
    # Group 3b — downside-specific signals
    "gap_down_count_20d","gap_up_count_20d",
    "consecutive_red_days","consecutive_green_days",   # dn: extended uptrend = reversal setup
    "distribution_days_20d",                           # high-vol down closes (distribution)
    "days_since_20d_high",
    "close_in_range_5d","dist_above_50ma","red_day_vol_ratio",
    # Group 3c — vol-scaled signal strength
    "ret_1d_vol_scaled","ret_5d_vol_scaled","ret_20d_vol_scaled",
    "dist_52w_high_z","dist_52w_low_z","dist_above_50ma_z",
    "nearest_resistance_z","nearest_support_z",
    "lt_resistance_z","lt_support_z",              # ATR-normalised distance to >2yr zones
    # Group 4 (zones) — raw % dist dropped; ATR-normalised _z versions in group 3c are sufficient
    "resistance_valid_touches","support_valid_touches",
    "resistance_zone_strength","support_zone_strength",
    "consolidating_at_resistance","consolidating_at_support",
    "resistance_zone_age_weeks","support_zone_age_weeks",  # age of nearest zone (older = stronger memory)
    "zone_box_width_pct","weeks_in_box",
    "zone_breakout","zone_breakdown",           # daily-accurate: close vs level today
    "zone_breakout_ffill","zone_breakdown_ffill", # momentum: broke out in last 5d window
    # Group 5 (weekly)
    "w_body_pct","w_body_ratio","w_upper_wick","w_lower_wick","w_close_pos","w_range_pct",
    "w_gap","w_body_expand","w_range_expand","w_inside_bar","w_outside_bar",
    "w_bull_engulf","w_bear_engulf",
    "w_is_hammer","w_is_shooting_star","w_is_doji","w_is_marubozu",
    # Group 6
    "fut_oi_chg_1d","fut_oi_chg_5d",
    "long_buildup","short_buildup","short_covering","long_unwinding",  # daily flags
    "long_buildup_streak","short_buildup_streak","long_unwinding_streak",
    "basis_pct","basis_chg_5d",
    "pcr_oi","pcr_vol","pcr_oi_rank_60d","pcr_chg_5d",
    "fut_vol_oi_ratio","opt_oi_ratio_20d",
    "max_pain_dist","dist_call_wall","dist_put_wall","days_to_expiry",
    # Options structure — Tier 1 & 2
    "put_oi_pct","opt_total_oi_chg_5d_rank","annualized_basis","wall_compression",
    # ATM implied volatility (Black-Scholes from per-strike bhavcopy)
    "atm_iv","atm_ce_iv","atm_pe_iv",
    "put_call_iv_skew","atm_iv_rank_252d","put_call_iv_skew_rank_60d",
    # Group 6 — dn_5_xl-specific F&O vulnerability signals
    "put_call_vol_ratio","put_call_vol_rank_60d",  # put flow > call flow = active bearish hedge
    "short_buildup_5d_count","long_unwind_5d_count",  # 5-day rolling count (vs consecutive only)
    "fut_oi_z_60d","basis_rank_60d",              # crowded longs + basis at multi-month low
    # Group 7
    "fii_cash_net_5d","fii_cash_zscore","fii_cash_streak",
    "fii_idx_fut_net_chg","dii_cash_net_5d","smart_vs_retail",
    # FII expansion: longer-horizon, dynamics, extreme flags, conviction
    "fii_cash_net_30d","fii_cash_acceleration","fii_cash_reversal_flag",
    "fii_extreme_outflow","fii_extreme_inflow","fii_buy_sell_ratio",
    # Participant OI — Tier 1: FII/Client stock futures positioning
    "fii_stk_fut_net","fii_stk_fut_net_chg_5d",
    "client_stk_fut_net","fii_vs_client_stk",
    # Participant OI — Tier 2: put accumulation + pro desk
    "fii_put_long_stk","fii_put_long_stk_chg_5d",
    "client_put_short_stk","pro_stk_fut_net",
    # Participant OI — FII synthetic positioning + client complacency
    "fii_stk_put_call_oi_ratio","fii_stk_net_opt_dir","fii_stk_net_opt_dir_chg_5d",
    "client_stk_call_put_net",
    # Group 7b — NSDL fortnightly sectoral FII flows (per-sector + cross-sectional)
    # fii_sector_flow_14d dropped — fortnightly pub lag makes it noisier than 30d
    "fii_sector_flow_30d","fii_sector_flow_90d",
    "fii_sector_flow_zscore","fii_sector_flow_pct_aum",
    "fii_sector_flow_streak","fii_sector_flow_acceleration",
    "fii_sector_aum_pct_change_90d",
    "fii_sector_rotation_rank","fii_sector_breadth_pos",
    # Group 8
    "days_to_next_earnings","days_since_last_earnings",
    "last_eps_surprise_pct","eps_surprise_3q_avg",
    "eps_beat_streak","eps_miss_streak",
    "big_eps_miss","eps_growth_yoy",
    "days_to_ex_div","days_since_ex_div",
    # Group 8a2 — Investing.com analyst consensus
    "inv_surprise_pct","inv_beat","inv_beat_rate_4q",
    "inv_avg_surprise_4q","inv_rev_surprise_pct",
    # Group 8b — block deals
    "block_sell_qty_5d","block_buy_qty_5d","block_net_qty_5d",
    "block_sell_val_5d","block_buy_val_5d","block_net_val_5d","block_deal_flag_5d",
    # Group 8c — bulk deals
    "bulk_sell_val_5d","bulk_buy_val_5d","bulk_net_val_5d",
    "bulk_sell_flag_5d","bulk_buy_flag_5d",
    # Group 9a — stock phase
    "stock_phase",           # categorical: bull/bear/recovery/distribution/consolidation
    "stock_dd_52w",          # drawdown from 52-week high (continuous)
    "stock_recovery_52w",    # recovery ratio from 52-week low (0=at low, 1=at high)
    "higher_high","higher_low","lower_high","lower_low",   # binary structure flags
    # Group 9b — market regime
    "market_phase",          # categorical: bull/bear/recovery/distribution/consolidation
    "nifty_ret_5d","nifty_ret_20d",
    "nifty_above_200ma","nifty_dist_52w_high","nifty_dist_52w_low",
    "nifty_momentum_accel","nifty_hv_20",
    "vix_level","vix_rank_252d","vix_chg_5d",
    "regime","regime_duration_days","regime_changed_5d","is_expiry_week",
    # combined_phase is intentionally excluded from FEATURE_COLS —
    # it is used only for regime-model routing in predict.py, not as a model input
    # Group 10
    "sector_ret_5d","rel_strength_sector",
    # Compound
    "mega_breakout","bull_confluence","earnings_vol_setup","double_squeeze","zone_energy",
    "fii_nifty_divergence","quiet_accum",
    # dn_5_xl compound interactions
    "fii_put_rush_on_stretch",
    "stretch_beta","dn_smart_short","dn_basis_stretch","put_activity_stretch","dn_oi_crowded",
    "dn_exhaustion","dn_macro_stock_timing","iv_vs_hv",
    # FII proxy
    "fii_accum_prob",
    # Group 11 — path quality (lagged realized label windows, no look-ahead)
    # Phase 2 — coiled-spring features
    "net_swing_lag1",     # net signed swing 6 trading days ago (+ = clean up, - = clean dn)
    "path_asym_lag1",     # upside / |downside| ratio 6 days ago (clean move asymmetry)
    "net_swing_lag2",     # same 11 days ago (second completed window)
    "net_swing_roll20",   # 20-day rolling mean of net swing (sustained momentum quality)
    # Group 12 — cross-sectional ranks (within-day, 0..1 where 1 = strongest in universe)
    "ret_5d_rank","ret_20d_rank","ret_20d_vol_scaled_rank",
    "vol_expansion_rank","breakout_quality_rank","vol_ratio_20d_rank",
    "delivery_ratio_20d_rank","pcr_oi_rank","atr_compression_rank",
    "dist_52w_high_rank","dist_above_50ma_rank","fut_oi_chg_5d_rank",
    "avg_trade_value_ratio_20d_rank",
    # Group 12b — within-sector cross-sectional ranks
    "ret_5d_rank_sector","ret_20d_rank_sector",
    "breakout_quality_rank_sector","vol_expansion_rank_sector",
    # Phase 2 — coiled-spring features
    "range_vs_vol_implied",
    "vol_recovery_score",
    "dip_vol_spike_5d",
    "fii_macro_stock_divergence",
    # Group 13 — rolling Nifty beta + cross-sectional beta rank
    "beta_nifty_20d","beta_nifty_60d",
    "beta_nifty_20d_rank","sharpe_20d_rank",
    # dn reversal cross-sectional ranks (1 = most extended uptrend / most above BB in universe)
    "consecutive_green_days_rank","dist_from_bb_upper_rank",
    # dn_5_xl cross-sectional vulnerability ranks
    "short_buildup_streak_rank",  # most consecutive smart-money shorts in universe
    "put_call_vol_ratio_rank",    # most put activity vs call activity today
    "fut_oi_z_60d_rank",          # most crowded long positioning vs own history
    "stretch_beta_rank",          # most overbought × high-beta = maximum vulnerability rank
]


# ──────────────────────────────────────────────────────────────────────
# Group 12 — Path Asymmetry (Layer-2 features for clean directional models)
# ──────────────────────────────────────────────────────────────────────
# Quantifies how directionally "clean" recent price action was.
# All features are purely backward-looking (end of day T, using bars T-N..T).
#
# 19 stock-level features + 4 NIFTY regime features = 23 total.
# Computed for the full universe (not just liquid-30) so they land in
# features.parquet and train.py can filter to liquid rows as needed.
# ──────────────────────────────────────────────────────────────────────
def _group_path_asym(stock: pd.DataFrame) -> pd.DataFrame:
    """
    Compute path-asymmetry features from OHLCV silver data.

    Parameters
    ----------
    stock : eod_stock silver DataFrame (date, symbol, open, high, low, close, volume)

    Returns
    -------
    DataFrame with columns (date, symbol, <23 path-asym features>)
    """
    s = stock[["date", "symbol", "open", "high", "low", "close", "volume"]].copy()
    s["date"] = pd.to_datetime(s["date"])
    s = s.sort_values(["symbol", "date"]).reset_index(drop=True)

    g = s.groupby("symbol", sort=False, group_keys=False)
    prev_close = g["close"].shift(1)

    body        = s["close"] - s["open"]
    abs_body    = body.abs()
    up_body     = body.clip(lower=0)
    dn_body     = (-body).clip(lower=0)
    upper_wick  = s["high"] - np.maximum(s["open"], s["close"])
    lower_wick  = np.minimum(s["open"], s["close"]) - s["low"]
    rng         = s["high"] - s["low"]
    abs_ret_1d  = ((s["close"] - prev_close) / prev_close.replace(0, np.nan)).abs()
    tr = pd.concat([rng,
                    (s["high"] - prev_close).abs(),
                    (s["low"]  - prev_close).abs()], axis=1).max(axis=1)

    s["_up_body"]    = up_body
    s["_dn_body"]    = dn_body
    s["_abs_body"]   = abs_body
    s["_upper_wick"] = upper_wick
    s["_lower_wick"] = lower_wick
    s["_range"]      = rng
    s["_abs_ret_1d"] = abs_ret_1d
    s["_tr"]         = tr
    # Green day: close > open AND close > prior close
    s["_is_green"] = ((s["close"] > s["open"]) & (s["close"] > prev_close)).astype(int)
    # Red day: close < open AND close < prior close
    s["_is_red"]   = ((s["close"] < s["open"]) & (s["close"] < prev_close)).astype(int)

    g2 = s.groupby("symbol", sort=False, group_keys=False)

    def roll_sum(col: str, n: int, mp: int) -> pd.Series:
        return g2[col].transform(lambda x: x.rolling(n, min_periods=mp).sum())

    def roll_mean(col: str, n: int, mp: int) -> pd.Series:
        return g2[col].transform(lambda x: x.rolling(n, min_periods=mp).mean())

    eps = 1e-9

    # ── 5d sums ───────────────────────────────────────────────────────────────
    up_body_5d   = roll_sum("_up_body",    5, 3)
    dn_body_5d   = roll_sum("_dn_body",    5, 3)
    abs_body_5d  = roll_sum("_abs_body",   5, 3)
    upper_w_5d   = roll_sum("_upper_wick", 5, 3)
    lower_w_5d   = roll_sum("_lower_wick", 5, 3)
    range_5d     = roll_sum("_range",      5, 3)
    abs_ret_5d   = roll_sum("_abs_ret_1d", 5, 3)
    atr20        = roll_mean("_tr",       20, 10)
    close_lag5   = g2["close"].shift(5)
    net_ret_5d   = (s["close"] - close_lag5) / close_lag5.replace(0, np.nan)

    # Body asymmetry — fraction of recent body that was bullish vs bearish
    s["up_pressure_5d"] = up_body_5d / (abs_body_5d + eps)
    s["dn_pressure_5d"] = dn_body_5d / (abs_body_5d + eps)
    # Wick asymmetry — sellers on top vs buyers on bottom
    s["upper_wick_pressure_5d"] = upper_w_5d / (range_5d + eps)
    s["lower_wick_pressure_5d"] = lower_w_5d / (range_5d + eps)
    # Directional purity — net move / total path (1=monotonic, 0=pure whipsaw)
    s["directional_purity_5d"]        = net_ret_5d.abs() / (abs_ret_5d + eps)
    s["directional_purity_5d_signed"] = net_ret_5d       / (abs_ret_5d + eps)
    # Range purity — net move per unit of intraday range
    s["range_purity_5d_signed"] = (s["close"] - close_lag5) / (range_5d + eps)
    # Whipsaw count — days where range > 1.5×ATR20 (noisy, wide-ranging candles)
    is_whip = (s["_range"] > 1.5 * atr20).astype(int)
    s["whipsaw_count_5d"] = g2["symbol"].transform(  # group already defined on s
        lambda _: None  # placeholder — computed below
    )
    # recompute with correct groupby on is_whip
    s["whipsaw_count_5d"] = is_whip.groupby(s["symbol"], sort=False).transform(
        lambda x: x.rolling(5, min_periods=3).sum()
    )
    # Clean run counts — consecutive green/red days (3+ green = momentum up)
    s["clean_runup_5d"] = roll_sum("_is_green", 5, 3)
    s["clean_rundn_5d"] = roll_sum("_is_red",   5, 3)

    # ── 10d versions for longer regime context ────────────────────────────────
    abs_body_10d = roll_sum("_abs_body",   10, 5)
    up_body_10d  = roll_sum("_up_body",    10, 5)
    abs_ret_10d  = roll_sum("_abs_ret_1d", 10, 5)
    range_10d    = roll_sum("_range",      10, 5)
    close_lag10  = g2["close"].shift(10)
    net_ret_10d  = (s["close"] - close_lag10) / close_lag10.replace(0, np.nan)

    s["up_pressure_10d"]               = up_body_10d / (abs_body_10d + eps)
    s["directional_purity_10d_signed"] = net_ret_10d / (abs_ret_10d  + eps)
    s["range_purity_10d_signed"]       = (s["close"] - close_lag10) / (range_10d + eps)

    # ── Volume asymmetry — fraction of recent volume on green vs red days ─────
    s["_vol_green"] = s["volume"] * s["_is_green"]
    s["_vol_red"]   = s["volume"] * s["_is_red"]
    vol_green_5d = roll_sum("_vol_green", 5, 3)
    vol_red_5d   = roll_sum("_vol_red",   5, 3)
    vol_total_5d = roll_sum("volume",     5, 3)
    s["vol_up_ratio_5d"] = vol_green_5d / (vol_total_5d + eps)
    s["vol_dn_ratio_5d"] = vol_red_5d   / (vol_total_5d + eps)

    # ── Gap persistence — gap in direction that held through end of day ───────
    gap_up = ((s["open"] - prev_close) / prev_close.replace(0, np.nan) > 0.005
              ).fillna(False).astype(int)
    gap_dn = ((prev_close - s["open"]) / prev_close.replace(0, np.nan) > 0.005
              ).fillna(False).astype(int)
    gap_up_held = (gap_up & (s["close"] > s["open"])).astype(int)
    gap_dn_held = (gap_dn & (s["close"] < s["open"])).astype(int)
    s["_gap_up_held"] = gap_up_held
    s["_gap_dn_held"] = gap_dn_held
    s["gap_up_persistence_5d"] = roll_sum("_gap_up_held", 5, 3)
    s["gap_dn_persistence_5d"] = roll_sum("_gap_dn_held", 5, 3)

    # ── Clear-air structure — close above prior 5d max-high (no overhead) ────
    prior_5d_high = g2["high"].transform(
        lambda x: x.shift(1).rolling(5, min_periods=3).max()
    )
    prior_5d_low = g2["low"].transform(
        lambda x: x.shift(1).rolling(5, min_periods=3).min()
    )
    s["_clear_air_up"] = (s["close"] > prior_5d_high).astype(int)
    s["_clear_air_dn"] = (s["close"] < prior_5d_low ).astype(int)
    s["clear_air_up_streak_5d"] = roll_sum("_clear_air_up", 5, 3)
    s["clear_air_dn_streak_5d"] = roll_sum("_clear_air_dn", 5, 3)

    # ── NIFTY regime path features (broadcast to all symbols by date) ─────────
    try:
        nifty = pd.read_parquet(
            SILVER_TABLES["indices"],
            columns=["date", "index_name", "open", "high", "low", "close"],
        )
        nifty = nifty[nifty["index_name"] == "Nifty 50"].copy()
        nifty["date"] = pd.to_datetime(nifty["date"])
        nifty = nifty.sort_values("date").reset_index(drop=True)

        n_prev  = nifty["close"].shift(1)
        n_body  = nifty["close"] - nifty["open"]
        n_up    = n_body.clip(lower=0)
        n_abs   = n_body.abs()
        n_range = nifty["high"] - nifty["low"]
        n_ret   = ((nifty["close"] - n_prev) / n_prev.replace(0, np.nan)).abs()

        n_up_5d  = n_up.rolling(5, min_periods=3).sum()
        n_abs_5d = n_abs.rolling(5, min_periods=3).sum()
        n_ret_5d = n_ret.rolling(5, min_periods=3).sum()
        n_rng_5d = n_range.rolling(5, min_periods=3).sum()
        n_lag5   = nifty["close"].shift(5)
        n_net_5d = (nifty["close"] - n_lag5) / n_lag5.replace(0, np.nan)

        nifty["nifty_up_pressure_5d"]               = n_up_5d  / (n_abs_5d + eps)
        nifty["nifty_directional_purity_5d_signed"] = n_net_5d / (n_ret_5d  + eps)
        nifty["nifty_range_purity_5d_signed"]       = (nifty["close"] - n_lag5
                                                        ) / (n_rng_5d + eps)
        nifty["nifty_directional_purity_5d"]        = n_net_5d.abs() / (n_ret_5d + eps)

        nifty_feats = ["nifty_up_pressure_5d", "nifty_directional_purity_5d_signed",
                       "nifty_range_purity_5d_signed", "nifty_directional_purity_5d"]
        nifty_df = nifty[["date"] + nifty_feats]
    except Exception as _e:
        print(f"[features] path_asym: NIFTY read failed ({_e}) — NIFTY path features will be NaN")
        nifty_df = pd.DataFrame(columns=["date", "nifty_up_pressure_5d",
                                         "nifty_directional_purity_5d_signed",
                                         "nifty_range_purity_5d_signed",
                                         "nifty_directional_purity_5d"])
        nifty_feats = []

    stock_feat_cols = [
        "up_pressure_5d", "dn_pressure_5d",
        "upper_wick_pressure_5d", "lower_wick_pressure_5d",
        "directional_purity_5d", "directional_purity_5d_signed",
        "range_purity_5d_signed", "whipsaw_count_5d",
        "clean_runup_5d", "clean_rundn_5d",
        "up_pressure_10d", "directional_purity_10d_signed", "range_purity_10d_signed",
        "vol_up_ratio_5d", "vol_dn_ratio_5d",
        "gap_up_persistence_5d", "gap_dn_persistence_5d",
        "clear_air_up_streak_5d", "clear_air_dn_streak_5d",
    ]
    out = s[["date", "symbol"] + stock_feat_cols]
    if nifty_feats:
        out = out.merge(nifty_df, on="date", how="left")

    return out


def _load_silver():
    """Load all silver tables needed for feature computation."""
    stock   = pd.read_parquet(SILVER_TABLES["eod_stock"])
    deriv   = pd.read_parquet(SILVER_TABLES["eod_deriv_daily"])
    indices = pd.read_parquet(SILVER_TABLES["indices"])
    stock["date"] = pd.to_datetime(stock["date"])
    stock = stock.sort_values(["symbol", "date"]).reset_index(drop=True)
    fo_syms = set(deriv["symbol"].unique())
    stock = stock[stock["symbol"].isin(fo_syms)].reset_index(drop=True)
    extras = {
        "fii_dii":       _maybe_read(SILVER_TABLES["fii_dii_cash"]),
        "part_oi":       _maybe_read(SILVER_TABLES["participant_oi"]),
        "fii_sector":    _maybe_read(SILVER_TABLES["fii_sector"]),
        "earnings":      _maybe_read(SILVER_TABLES["earnings"]),
        "eps":           _maybe_read(SILVER_TABLES["earnings_eps"]),
        "fund":          _maybe_read(SILVER_TABLES["fundamentals"]),
        "corp_actions":  _maybe_read(SILVER_TABLES["corp_actions"]),
        "block_deals":   _maybe_read(SILVER_TABLES.get("block_deals")),
        "bulk_deals":    _maybe_read(SILVER_TABLES.get("bulk_deals")),
        "investing_eps": _maybe_read(SILVER_TABLES.get("investing_eps")),
    }
    return stock, deriv, indices, extras


def _assemble(stock: pd.DataFrame, deriv: pd.DataFrame, indices: pd.DataFrame,
              extras: dict, force_zones: bool = False) -> pd.DataFrame:
    """Compute all feature groups and return the assembled base DataFrame."""
    print("[features] group 1: returns & momentum …")
    g1 = _group_returns_momentum(stock)

    print("[features] group 2: volatility & volume …")
    g2 = _group_vol_volume(stock)

    print("[features] group 3: consolidation & breakout …")
    g3 = _group_consolidation_breakout(stock, g2)

    print("[features] group 3b: downside-specific signals …")
    g3b = _group_downside_signals(stock)

    base = pd.concat([stock[["date", "symbol", "close"]], g1, g2, g3, g3b], axis=1)

    # ── Zone levels — loaded from pre-built cache (run pipeline.zones weekly) ──
    # Zone levels change at most weekly (built from weekly bars + 2-yr lookback).
    # Recomputing them here on every features rebuild is wasteful.  Instead:
    #   • Run `python -m pipeline.zones` once a week (Sunday night).
    #   • features.py loads the cache; if missing or > 7 calendar days stale,
    #     it triggers an automatic rebuild before proceeding — no silent staleness.
    #   • Pass force_zones=True to force a rebuild regardless of cache age.
    _ZONE_STALE_DAYS = 7

    def _need_zone_rebuild(reason: str) -> None:
        print(f"[features] group 4: zones — {reason}. Rebuilding now …")

    z = pd.DataFrame()
    _do_rebuild = force_zones

    if not _do_rebuild:
        if not GOLD_ZONES.exists():
            _need_zone_rebuild("gold/zones.parquet not found")
            _do_rebuild = True
        else:
            try:
                _z_dates = pd.read_parquet(GOLD_ZONES, columns=["date"])
                _z_dates["date"] = pd.to_datetime(_z_dates["date"])
                _stock_last = pd.to_datetime(stock["date"]).max()
                _cache_last = _z_dates["date"].max()
                lag_days = (_stock_last - _cache_last).days
                if lag_days > _ZONE_STALE_DAYS:
                    _need_zone_rebuild(
                        f"cache is {lag_days}d stale "
                        f"(last={_cache_last.date()}, stock ends {_stock_last.date()})"
                    )
                    _do_rebuild = True
                else:
                    print(f"[features] group 4: zones — loaded cache "
                          f"(through {_cache_last.date()}, {lag_days}d lag)")
                    z = pd.read_parquet(GOLD_ZONES)
            except Exception as e:
                _need_zone_rebuild(f"cache unreadable ({e})")
                _do_rebuild = True

    if _do_rebuild:
        from .zones import build_zones as _build_zones
        _build_zones(cadence_days=5)
        if GOLD_ZONES.exists():
            z = pd.read_parquet(GOLD_ZONES)

    if not z.empty:
        base = base.merge(z, on=["date", "symbol"], how="left")

        # ── Clean forward-fill artifacts where R < S ──────────────────────
        # resistance and support are forward-filled independently; when they
        # come from different snapshot dates they can be inverted. Null both
        # out so downstream features don't see impossible zone geometry.
        if "resistance_level" in base.columns and "support_level" in base.columns:
            _bad = (
                base["resistance_level"].notna() &
                base["support_level"].notna() &
                (base["resistance_level"] <= base["support_level"])
            )
            base.loc[_bad, ["resistance_level", "support_level",
                             "nearest_resistance_dist", "nearest_support_dist",
                             "resistance_valid_touches", "support_valid_touches",
                             "resistance_zone_strength", "support_zone_strength",
                             "consolidating_at_resistance", "consolidating_at_support",
                             "resistance_zone_age_weeks", "support_zone_age_weeks",
                             "zone_box_width_pct", "weeks_in_box",
                             "zone_breakout", "zone_breakdown"]] = np.nan

        # ── Daily-accurate breakout / breakdown flags ─────────────────────
        # zone_breakout / zone_breakdown in zones.parquet are computed at
        # weekly sample points (last-5-days check) and forward-filled — so
        # a Tuesday breakout only appears in features on the following Monday.
        # Recompute daily using the forward-filled zone levels (which change
        # slowly, so this doesn't introduce future data) so the model sees
        # the breakout on the day it happens.
        # Original sampled flags are kept as zone_breakout_ffill / zone_breakdown_ffill
        # in case the model finds the "momentum of recent breakout" useful.
        if "zone_breakout" in base.columns:
            base = base.rename(columns={"zone_breakout": "zone_breakout_ffill",
                                        "zone_breakdown": "zone_breakdown_ffill"})
        if "resistance_level" in base.columns:
            res_band_hi = (base["resistance_level"]
                           * (1 + ZONE_CLUSTER_PCT / 2))
            base["zone_breakout"] = (
                (base["close"] > res_band_hi) & base["resistance_level"].notna()
            ).astype(np.int8)
        if "support_level" in base.columns:
            sup_band_lo = (base["support_level"]
                           * (1 - ZONE_CLUSTER_PCT / 2))
            base["zone_breakdown"] = (
                (base["close"] < sup_band_lo) & base["support_level"].notna()
            ).astype(np.int8)

    print("[features] group 5: weekly candles …")
    w = _group_weekly_candles(stock)
    if not w.empty:
        base = base.merge(w, on=["date", "symbol"], how="left")

    print("[features] group 6: derivatives positioning …")
    d = _group_derivatives(deriv, stock)
    base = base.merge(d, on=["date", "symbol"], how="left")

    print("[features] group 7: participant flows …")
    p = _group_participant_flows(extras["fii_dii"], extras["part_oi"]) if extras["fii_dii"] is not None else None
    if p is not None and not p.empty:
        base = base.merge(p, on="date", how="left")

    print("[features] group 7b: fii sectoral flows (NSDL fortnightly) …")
    fs = _group_fii_sector_flow(base[["date", "symbol"]], extras.get("fii_sector"))
    if fs is not None and not fs.empty:
        base = base.merge(fs, on=["date", "symbol"], how="left")

    print("[features] group 8: earnings & fundamentals …")
    e = _group_earnings(stock[["date", "symbol"]], extras["earnings"], extras["eps"],
                        extras["fund"], extras.get("corp_actions"))
    base = base.merge(e, on=["date", "symbol"], how="left")

    print("[features] group 8a2: analyst EPS estimates (Investing.com) …")
    ie = _group_investing_eps(stock[["date", "symbol"]], extras.get("investing_eps"))
    if not ie.empty:
        base = base.merge(ie, on=["date", "symbol"], how="left")

    print("[features] group 8b: block deals …")
    bd = _group_block_deals(stock[["date", "symbol"]])
    if not bd.empty:
        base = base.merge(bd, on=["date", "symbol"], how="left")

    print("[features] group 8c: bulk deals …")
    bk = _group_bulk_deals(stock[["date", "symbol"]])
    if not bk.empty:
        base = base.merge(bk, on=["date", "symbol"], how="left")

    print("[features] group 9a: stock phase (HH/HL structure) …")
    sp = _group_stock_phase(stock)
    base = base.merge(sp, on=["date", "symbol"], how="left")

    print("[features] group 9b: market regime & market phase …")
    r = _group_regime(indices)
    base = base.merge(r, on="date", how="left")

    # Combined phase = stock's own trend phase × overall market phase
    # e.g. "bull_bull", "bear_bear", "recovery_bull" — used for regime-specific model routing.
    # Defragment + add via .assign() to avoid PerformanceWarning from incremental inserts.
    if "stock_phase" in base.columns and "market_phase" in base.columns:
        base = base.copy()   # defragment after many prior merges
        base = base.assign(combined_phase=(
            base["stock_phase"].fillna("consolidation") + "_" +
            base["market_phase"].fillna("consolidation")
        ))

    print("[features] group 10: sector relative strength …")
    sec = _group_sector(stock, indices)
    base = base.merge(sec, on=["date", "symbol"], how="left")

    print("[features] group 13: rolling Nifty beta (20d, 60d) …")
    base = _add_nifty_beta(base, indices)

    print("[features] group 3c: vol-scaled signals …")
    vs = _group_vol_scaled(base)
    base = pd.concat([base, vs], axis=1)

    print("[features] compound interactions …")
    comp = _compound_interactions(base)
    base = pd.concat([base, comp], axis=1)

    print("[features] cross-sectional ranks (within-day, within-sector) …")
    base = _add_cross_sectional_ranks(base)

    print("[features] fii proxy score …")
    fii_proxy = _group_fii_proxy_score(base)
    base = base.merge(fii_proxy[["date", "symbol", "fii_accum_prob"]],
                      on=["date", "symbol"], how="left")

    # ── Encode categorical phase / regime columns as integers ──────────────
    # stock_phase / market_phase are stored as strings; _numeric_feat_cols
    # (used by all model wrappers) silently drops non-numeric columns — so
    # these must be label-encoded here to reach the models.
    _PHASE_MAP = {"bull": 4, "recovery": 3, "consolidation": 2,
                  "distribution": 1, "bear": 0}
    _REGIME_MAP = {"bull": 2, "range": 1, "bear": 0}
    for col, mapping in [("stock_phase", _PHASE_MAP),
                          ("market_phase", _PHASE_MAP),
                          ("regime",       _REGIME_MAP)]:
        if col in base.columns:
            base[col] = base[col].map(mapping).astype("Int8")

    return base.replace([np.inf, -np.inf], np.nan)


def _add_cross_sectional_ranks(base: pd.DataFrame) -> pd.DataFrame:
    """Add per-day cross-sectional rank features (0..1, 1 = highest).

    Lets the model say "this stock is the strongest momentum in the universe
    today" — much more informative than the raw return number alone.
    Ranks are pct-ranked within each date (and additionally within sector).

    All new columns are accumulated in a single dict and concatenated once at
    the end (avoids the DataFrame-fragmentation perf warnings that arise from
    repeated `base[col] = ...` insertions).
    """
    rank_cols = [c for c in [
        "ret_5d", "ret_20d", "ret_20d_vol_scaled",
        "vol_expansion", "breakout_quality", "vol_ratio_20d",
        "delivery_ratio_20d", "pcr_oi", "atr_compression",
        "dist_52w_high", "dist_above_50ma",
        "fut_oi_chg_5d",
        "avg_trade_value", "avg_trade_value_ratio_20d",   # institutional participation
        "beta_nifty_20d", "sharpe_20d",                   # market sensitivity + quality
        "consecutive_green_days", "dist_from_bb_upper",   # dn reversal setups
        # dn_5_xl cross-sectional ranks
        "short_buildup_streak",      # most consecutive shorts = highest-conviction short target
        "put_call_vol_ratio",        # most put activity today relative to peers
        "fut_oi_z_60d",              # most crowded long OI relative to own history
        "stretch_beta",              # most overbought × high-beta = most vulnerable
    ] if c in base.columns]

    if not rank_cols:
        return base

    new_cols: dict[str, pd.Series] = {}

    # Universe-wide rank within each day
    grp = base.groupby("date", sort=False)
    for col in rank_cols:
        new_cols[f"{col}_rank"] = grp[col].rank(pct=True, method="average")

    # Sector-relative rank
    sector_map = base["symbol"].map(TICKER_SECTOR).fillna("other")
    sec_grp = base.assign(_sec=sector_map).groupby(["date", "_sec"], sort=False)
    for col in ["ret_5d", "ret_20d", "breakout_quality", "vol_expansion"]:
        if col in base.columns:
            new_cols[f"{col}_rank_sector"] = sec_grp[col].rank(pct=True, method="average")

    # Single concat — no fragmentation
    return pd.concat([base, pd.DataFrame(new_cols, index=base.index)], axis=1)


def build(include_zones: bool = True, force_zones: bool = False) -> pd.DataFrame:
    """Full rebuild — computes all features for all dates and saves features.parquet."""
    print("[features] loading silver tables …")
    stock, deriv, indices, extras = _load_silver()
    print(f"  rows={len(stock):,}  symbols={stock['symbol'].nunique()}  "
          f"date_range={stock['date'].min().date()}→{stock['date'].max().date()}")

    base = _assemble(stock, deriv, indices, extras,
                     force_zones=force_zones if include_zones else False)
    if not include_zones:
        # drop zone cols if they snuck in from a cached file
        zone_cols = [c for c in base.columns if any(
            c.startswith(p) for p in ("nearest_", "resistance_", "support_",
                                      "consolidating_", "polarity_", "zone_", "weeks_",
                                      "lt_"))]
        base = base.drop(columns=zone_cols, errors="ignore")

    print("[features] group 11: path quality (lagged realized swings) …")
    pq = _group_path_quality(stock)
    base = base.merge(pq, on=["date", "symbol"], how="left")

    print("[features] group 12: path asymmetry (clean-day Layer-2 features) …")
    pa = _group_path_asym(stock)
    base = base.merge(pa, on=["date", "symbol"], how="left")

    present, dropped = _select_features(base)

    keep = ["date", "symbol"] + present
    panel = base[keep].sort_values(["symbol", "date"]).reset_index(drop=True)
    panel.to_parquet(GOLD_FEATURES, index=False)
    print(f"\n[features] saved {len(panel):,} rows × {len(present)} features → {GOLD_FEATURES}")
    if dropped:
        print(f"  [skipped] {len(dropped)} features with no data: {dropped}")
    _report(panel, present)
    return panel


def build_for_date(date_str: str) -> pd.DataFrame:
    """
    Daily inference path — compute features for a single date without
    touching features.parquet.  Full silver history is loaded for correct
    rolling-window lookbacks; zones are always served from cache.
    Returns a DataFrame with one row per active symbol for date_str.
    """
    print(f"[features] daily build for {date_str} …")
    stock, deriv, indices, extras = _load_silver()

    # Trim to date (inclusive) so rolling windows only see past data
    asof = pd.Timestamp(date_str)
    stock   = stock[stock["date"] <= asof].copy()
    deriv_t = deriv.copy()
    deriv_t["date"] = pd.to_datetime(deriv_t["date"])
    deriv_t = deriv_t[deriv_t["date"] <= asof]

    base = _assemble(stock, deriv_t, indices, extras, force_zones=False)

    # Keep only today's rows
    day = base[base["date"] == asof].copy()
    if day.empty:
        raise ValueError(f"No feature rows found for {date_str} — is that a trading day?")

    # Group 11: path quality features from past OHLC swing windows
    pq_today = _path_quality_for_date(stock, asof)
    if not pq_today.empty:
        day = day.merge(pq_today, on=["date", "symbol"], how="left")

    # Group 12: path asymmetry features (clean-day Layer-2)
    pa = _group_path_asym(stock)
    pa_today = pa[pa["date"] == asof]
    if not pa_today.empty:
        day = day.merge(pa_today, on=["date", "symbol"], how="left")

    present, _ = _select_features(day)
    print(f"[features] {len(day)} symbols × {len(present)} features for {date_str}")
    return day[["date", "symbol"] + present]


def _select_features(panel: pd.DataFrame,
                     null_threshold: float = 0.99) -> tuple[list, list]:
    """
    Return (present, dropped).
    Writes ALL computed columns to the parquet — feature selection happens at
    training time via TARGET_FEATURE_COLS in config.py, not here.
    Excluded: index columns (date, symbol), non-feature routing columns, and
    any column with >= null_threshold null rate (no data source yet).
    FEATURE_COLS is kept as a documentation registry only — removing a feature
    from it does NOT remove it from the parquet.
    """
    _NON_FEATURE = {"date", "symbol", "close",
                    "combined_phase"}   # routing-only, not a model input
    present, dropped = [], []
    for c in panel.columns:
        if c in _NON_FEATURE:
            continue
        if panel[c].isna().mean() >= null_threshold:
            dropped.append(c)
        else:
            present.append(c)
    return present, dropped


def _maybe_read(path) -> Optional[pd.DataFrame]:
    from pathlib import Path
    if not Path(path).exists():
        return None
    try:
        return pd.read_parquet(path)
    except Exception as e:
        print(f"  [warn] could not read {path}: {e}")
        return None


def _report(df: pd.DataFrame, feat_cols: list):
    null_pct = df[feat_cols].isna().mean().max() if feat_cols else 0.0
    print(f"  rows={len(df):,}  symbols={df['symbol'].nunique()}  "
          f"dates={df['date'].nunique()}  max_null={null_pct:.1%}")


def load() -> pd.DataFrame:
    return pd.read_parquet(GOLD_FEATURES)


if __name__ == "__main__":
    args = sys.argv[1:]
    include_zones = "--no-zones" not in args
    force_zones   = "--force-zones" in args
    build(include_zones=include_zones, force_zones=force_zones)
