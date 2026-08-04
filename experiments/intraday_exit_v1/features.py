"""Leakage-safe 5-minute features and forward labels for exit modelling."""
from __future__ import annotations

import numpy as np
import pandas as pd


REQUIRED = {"trade_id", "timestamp", "close", "ce_close", "pe_close"}


def _forward_extreme(values: pd.Series, horizon: int, kind: str) -> pd.Series:
    future = pd.concat(
        [values.shift(-step) for step in range(1, horizon + 1)],
        axis=1,
    )
    complete = future.notna().sum(axis=1).eq(horizon)
    extreme = future.max(axis=1) if kind == "max" else future.min(axis=1)
    return extreme.where(complete)


def build_exit_training_frame(
    bars_5m: pd.DataFrame,
    *,
    label_horizon_bars: int = 6,
    minimum_remaining_upside: float = 0.03,
) -> pd.DataFrame:
    """Build features known at bar close and separately named future targets.

    The default target asks whether the next 30 minutes (six 5-minute bars)
    offer neither positive terminal return nor at least 3% additional upside.
    It is a research label, not yet a production exit rule.
    """
    missing = sorted(REQUIRED - set(bars_5m.columns))
    if missing:
        raise ValueError(f"missing 5-minute training columns: {missing}")
    if label_horizon_bars < 1:
        raise ValueError("label_horizon_bars must be positive")

    bars = bars_5m.copy()
    bars["timestamp"] = pd.to_datetime(bars["timestamp"])
    bars = bars.sort_values(["trade_id", "timestamp"]).reset_index(drop=True)
    frames: list[pd.DataFrame] = []
    for _, group in bars.groupby("trade_id", sort=False):
        out = group.copy()
        close = pd.to_numeric(out["close"], errors="coerce")
        ce_close = pd.to_numeric(out["ce_close"], errors="coerce")
        pe_close = pd.to_numeric(out["pe_close"], errors="coerce")
        returns = close.pct_change()

        out["feature_elapsed_5m_bars"] = np.arange(1, len(out) + 1)
        out["feature_return_from_first_close"] = close / close.iloc[0] - 1
        out["feature_return_1bar"] = returns
        out["feature_return_3bar"] = close.pct_change(3)
        out["feature_return_6bar"] = close.pct_change(6)
        out["feature_realized_vol_6bar"] = returns.rolling(6, min_periods=3).std()
        out["feature_realized_vol_12bar"] = returns.rolling(12, min_periods=6).std()
        running_peak = close.cummax()
        out["feature_drawdown_from_peak"] = close / running_peak - 1
        out["feature_ce_value_share"] = ce_close / close
        out["feature_ce_return_1bar"] = ce_close.pct_change()
        out["feature_pe_return_1bar"] = pe_close.pct_change()
        out["feature_leg_return_spread"] = (
            out["feature_ce_return_1bar"] - out["feature_pe_return_1bar"]
        )
        minutes = out["timestamp"].dt.hour * 60 + out["timestamp"].dt.minute
        out["feature_session_fraction"] = (minutes - (9 * 60 + 15)) / 375
        if "volume" in out:
            volume = pd.to_numeric(out["volume"], errors="coerce")
            rolling_mean = volume.rolling(12, min_periods=3).mean()
            rolling_std = volume.rolling(12, min_periods=3).std()
            out["feature_volume_zscore_12bar"] = (volume - rolling_mean) / rolling_std
        if "oi" in out:
            out["feature_oi_change_1bar"] = pd.to_numeric(
                out["oi"], errors="coerce"
            ).pct_change()
        if "expiry" in out:
            expiry = pd.to_datetime(out["expiry"], errors="coerce")
            out["feature_dte"] = (
                expiry.dt.normalize() - out["timestamp"].dt.tz_localize(None).dt.normalize()
            ).dt.days

        future_max = _forward_extreme(close, label_horizon_bars, "max")
        future_min = _forward_extreme(close, label_horizon_bars, "min")
        future_close = close.shift(-label_horizon_bars)
        out["target_future_max_return"] = future_max / close - 1
        out["target_future_min_return"] = future_min / close - 1
        out["target_future_close_return"] = future_close / close - 1
        label_known = future_close.notna()
        exit_now = (
            (out["target_future_close_return"] <= 0)
            & (out["target_future_max_return"] < minimum_remaining_upside)
        )
        out["target_exit_now"] = exit_now.where(label_known).astype("boolean")
        frames.append(out)
    return pd.concat(frames, ignore_index=True) if frames else bars


def feature_columns(frame: pd.DataFrame) -> list[str]:
    return sorted(column for column in frame.columns if column.startswith("feature_"))


def target_columns(frame: pd.DataFrame) -> list[str]:
    return sorted(column for column in frame.columns if column.startswith("target_"))
