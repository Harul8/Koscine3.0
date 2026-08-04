from __future__ import annotations

import numpy as np
import pandas as pd

from koscine.config import HORIZON_DAYS, MOVE_THRESHOLDS

# Stage-1 expansion supports tier-specific thresholds (4% liquid, 7% rest)
EXPANSION_THRESHOLDS = (0.04, 0.05, 0.07)


def _future_rolling(series: pd.Series, horizon: int, reducer: str) -> pd.Series:
    reversed_series = series.iloc[::-1]
    shifted = reversed_series.shift(1)
    roller = shifted.rolling(horizon, min_periods=horizon)
    if reducer == "max":
        return roller.max().iloc[::-1]
    if reducer == "min":
        return roller.min().iloc[::-1]
    raise ValueError(f"Unsupported reducer: {reducer}")


def add_forward_labels(
    df: pd.DataFrame,
    horizon: int = HORIZON_DAYS,
    thresholds: tuple[float, ...] = EXPANSION_THRESHOLDS,
) -> pd.DataFrame:
    out = df.sort_values(["symbol", "date"]).copy()
    grouped = out.groupby("symbol", group_keys=False)
    out[f"future_{horizon}d_high"] = grouped["high"].apply(_future_rolling, horizon, "max")
    out[f"future_{horizon}d_low"] = grouped["low"].apply(_future_rolling, horizon, "min")
    out[f"future_{horizon}d_close"] = grouped["close"].shift(-horizon)
    out[f"future_{horizon}d_date"] = grouped["date"].shift(-horizon)
    out["entry_1d_date"] = grouped["date"].shift(-1)
    out["entry_1d_open"] = grouped["open"].shift(-1)

    entry_open = out["entry_1d_open"]
    out[f"up_move_{horizon}d"] = out[f"future_{horizon}d_high"] / entry_open - 1.0
    out[f"down_move_{horizon}d"] = 1.0 - out[f"future_{horizon}d_low"] / entry_open
    out[f"fwd_return_{horizon}d"] = out[f"future_{horizon}d_close"] / entry_open - 1.0
    out[f"long_adverse_move_{horizon}d"] = 1.0 - out[f"future_{horizon}d_low"] / entry_open
    out[f"short_adverse_move_{horizon}d"] = out[f"future_{horizon}d_high"] / entry_open - 1.0
    valid_future = (
        out[f"future_{horizon}d_high"].notna()
        & out[f"future_{horizon}d_low"].notna()
        & entry_open.notna()
    )

    # Vol-adjusted adverse limit: use 0.5 * ATR (clamped) so cleaner labels
    # adapt to per-stock realized volatility instead of one global threshold.
    atr_pct = out.get("atr_pct_14")
    if atr_pct is not None:
        out["vol_adj_adverse_limit"] = (0.5 * atr_pct).clip(lower=0.005, upper=0.025)
    else:
        out["vol_adj_adverse_limit"] = 0.0091

    for threshold in thresholds:
        pct = int(round(threshold * 100))
        up_col = f"label_up_{pct}pct_{horizon}d"
        down_col = f"label_down_{pct}pct_{horizon}d"
        quality_up_col = f"label_quality_up_{pct}pct_{horizon}d_adverse2pct"
        expansion_col = f"label_expansion_{pct}pct_{horizon}d"
        bucket_col = f"label_bucket_{pct}pct_{horizon}d"
        direction_col = f"label_direction_{pct}pct_{horizon}d"
        vol_clean_up = f"label_volclean_up_{pct}pct_{horizon}d"
        vol_clean_down = f"label_volclean_down_{pct}pct_{horizon}d"

        out[up_col] = np.where(valid_future, out[f"up_move_{horizon}d"] >= threshold, np.nan)
        out[down_col] = np.where(valid_future, out[f"down_move_{horizon}d"] >= threshold, np.nan)
        out[quality_up_col] = np.where(
            valid_future,
            (out[f"up_move_{horizon}d"] >= threshold)
            & (out[f"long_adverse_move_{horizon}d"] <= 0.02),
            np.nan,
        )
        out[expansion_col] = np.where(
            valid_future,
            (out[up_col] == 1) | (out[down_col] == 1),
            np.nan,
        )
        out[direction_col] = np.where(
            out[up_col].eq(1) & out[down_col].eq(0),
            1,
            np.where(out[down_col].eq(1) & out[up_col].eq(0), 0, np.nan),
        )
        out[bucket_col] = np.select(
            [out[up_col].eq(1) & out[down_col].eq(0), out[down_col].eq(1) & out[up_col].eq(0)],
            ["UP", "DOWN"],
            default="RANGE_OR_BOTH",
        )
        out[vol_clean_up] = np.where(
            valid_future,
            (out[f"up_move_{horizon}d"] >= threshold)
            & (out[f"long_adverse_move_{horizon}d"] <= out["vol_adj_adverse_limit"]),
            np.nan,
        )
        out[vol_clean_down] = np.where(
            valid_future,
            (out[f"down_move_{horizon}d"] >= threshold)
            & (out[f"short_adverse_move_{horizon}d"] <= out["vol_adj_adverse_limit"]),
            np.nan,
        )
        # Stage-1 expansion label: clean move in EITHER direction. This is the
        # primary target for the unified expansion model.
        expansion_clean_col = f"label_expansion_clean_{pct}pct_{horizon}d"
        out[expansion_clean_col] = np.where(
            valid_future,
            ((out[vol_clean_up] == 1) | (out[vol_clean_down] == 1)).astype(float),
            np.nan,
        )
    return out


def _add_symbol_features(group: pd.DataFrame) -> pd.DataFrame:
    group = group.sort_values("date").copy()
    close = group["close"]
    high = group["high"]
    low = group["low"]
    volume = group["volume"].replace(0, np.nan)

    group["ret_1d"] = close.pct_change()
    group["ret_3d"] = close.pct_change(3)
    group["ret_5d"] = close.pct_change(5)
    group["ret_10d"] = close.pct_change(10)
    group["ret_20d"] = close.pct_change(20)

    prev_close = close.shift(1)
    true_range = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    group["atr_5"] = true_range.rolling(5, min_periods=3).mean()
    group["atr_14"] = true_range.rolling(14, min_periods=7).mean()
    group["atr_pct_14"] = group["atr_14"] / close
    group["range_pct"] = (high - low) / close

    for window in (5, 10, 20, 50):
        ma = close.rolling(window, min_periods=max(3, window // 2)).mean()
        group[f"close_sma{window}_dist"] = close / ma - 1.0
        group[f"vol_sma{window}_ratio"] = volume / volume.rolling(
            window, min_periods=max(3, window // 2)
        ).mean()

    rolling_high_20 = high.rolling(20, min_periods=10).max()
    rolling_low_20 = low.rolling(20, min_periods=10).min()
    group["donchian_width_20"] = (rolling_high_20 - rolling_low_20) / close
    group["bb_width_20"] = (
        4.0 * close.rolling(20, min_periods=10).std() / close.rolling(20, min_periods=10).mean()
    )
    group["realized_vol_20"] = group["ret_1d"].rolling(20, min_periods=10).std()
    group["delivery_pct_chg_5"] = group["delivery_pct"].diff(5)
    group["delivery_qty_ratio_20"] = group["delivery_qty"] / group["delivery_qty"].rolling(
        20, min_periods=10
    ).mean()
    group["turnover_ratio_20"] = group["turnover_lacs"] / group["turnover_lacs"].rolling(
        20, min_periods=10
    ).mean()

    deriv_rolling_cols = ["fut_oi", "fut_chg_oi", "fut_vol", "atm_iv", "pcr_oi", "pcr_vol"]
    for col in deriv_rolling_cols:
        if col in group:
            group[f"{col}_chg_5"] = group[col].diff(5)
            group[f"{col}_ratio_20"] = group[col] / group[col].rolling(20, min_periods=10).mean()

    for col in ("fut_close", "max_pain", "call_wall_1", "put_wall_1"):
        if col in group:
            group[f"{col}_dist"] = group[col] / close - 1.0
    return group


def add_features(
    cash: pd.DataFrame,
    index_daily: pd.DataFrame | None = None,
    enrich: bool = True,
) -> pd.DataFrame:
    out = cash.sort_values(["symbol", "date"]).copy()
    out = out.groupby("symbol", group_keys=False).apply(_add_symbol_features)

    if index_daily is not None and not index_daily.empty:
        idx = index_daily.sort_values("date").copy()
        idx["nifty_ret_1d"] = idx["nifty_close"].pct_change()
        idx["nifty_ret_5d"] = idx["nifty_close"].pct_change(5)
        idx["nifty_realized_vol_20"] = idx["nifty_ret_1d"].rolling(20, min_periods=10).std()
        out = out.merge(idx[["date", "nifty_ret_1d", "nifty_ret_5d", "nifty_realized_vol_20"]], on="date", how="left")
        out["rel_ret_5d_vs_nifty"] = out["ret_5d"] - out["nifty_ret_5d"]

    cross_cols = ["ret_5d", "ret_20d", "vol_sma20_ratio", "atr_pct_14", "bb_width_20"]
    for col in cross_cols:
        if col in out:
            out[f"{col}_cs_rank"] = out.groupby("date")[col].rank(pct=True)

    out["day_of_week"] = out["date"].dt.dayofweek
    out["month"] = out["date"].dt.month
    out = out.sort_values(["date", "symbol"]).reset_index(drop=True)

    if enrich:
        from koscine.enhanced_features import enrich_dataset
        from koscine.expansion_features import enrich_with_expansion_features

        out = enrich_dataset(out)
        out = enrich_with_expansion_features(out)
    return out
