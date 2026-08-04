"""Canonical 1-minute option bars and leakage-safe 5-minute decision bars."""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


IST = "Asia/Kolkata"
LEG_REQUIRED = {
    "trade_id",
    "option_type",
    "timestamp",
    "open",
    "high",
    "low",
    "close",
}
PRICE_COLUMNS = ("open", "high", "low", "close")


def _to_ist(values: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(values, errors="coerce")
    if parsed.dt.tz is None:
        return parsed.dt.tz_localize(IST)
    return parsed.dt.tz_convert(IST)


def validate_leg_bars(bars: pd.DataFrame) -> pd.DataFrame:
    """Validate long-form exact-contract CE/PE 1-minute candles."""
    missing = sorted(LEG_REQUIRED - set(bars.columns))
    if missing:
        raise ValueError(f"missing option leg columns: {missing}")

    out = bars.copy()
    out["timestamp"] = _to_ist(out["timestamp"])
    out["option_type"] = out["option_type"].astype(str).str.upper()
    invalid_sides = sorted(set(out["option_type"].dropna()) - {"CE", "PE"})
    if invalid_sides:
        raise ValueError(f"option_type must be CE or PE, got {invalid_sides}")
    for column in (*PRICE_COLUMNS, "volume", "oi"):
        if column in out:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    out = out.dropna(subset=["trade_id", "timestamp", "option_type", "open", "close"])
    out = out.sort_values(["trade_id", "option_type", "timestamp"])
    out = out.drop_duplicates(["trade_id", "option_type", "timestamp"], keep="last")
    if (out[["open", "close"]] <= 0).any().any():
        raise ValueError("option leg open/close prices must be positive")
    return out.reset_index(drop=True)


def build_straddle_1m(leg_bars: pd.DataFrame) -> pd.DataFrame:
    """Synchronize fixed CE/PE legs into an executable 1-minute straddle path.

    `open` and `close` are executable synchronized sums. A true straddle high
    or low cannot be recovered from two OHLC candles because the two leg
    extrema may occur at different instants. The bounds are retained only for
    diagnostics and are never used by the simulator or feature builder.
    """
    legs = validate_leg_bars(leg_bars)
    counts = legs.groupby(["trade_id", "timestamp"]).option_type.nunique()
    valid_index = counts[counts == 2].index
    if len(valid_index) == 0:
        raise ValueError("no synchronized CE/PE minute bars were found")

    indexed = legs.set_index(["trade_id", "timestamp"])
    indexed = indexed.loc[indexed.index.isin(valid_index)].reset_index()
    value_columns = [column for column in (*PRICE_COLUMNS, "volume", "oi") if column in indexed]
    wide = indexed.pivot(
        index=["trade_id", "timestamp"],
        columns="option_type",
        values=value_columns,
    )
    wide.columns = [f"{side.lower()}_{field}" for field, side in wide.columns]
    wide = wide.reset_index().sort_values(["trade_id", "timestamp"])

    wide["open"] = wide["ce_open"] + wide["pe_open"]
    wide["close"] = wide["ce_close"] + wide["pe_close"]
    wide["high_upper_bound"] = wide["ce_high"] + wide["pe_high"]
    wide["low_lower_bound"] = wide["ce_low"] + wide["pe_low"]
    if {"ce_volume", "pe_volume"} <= set(wide.columns):
        wide["volume"] = wide["ce_volume"].fillna(0) + wide["pe_volume"].fillna(0)
    if {"ce_oi", "pe_oi"} <= set(wide.columns):
        wide["oi"] = wide["ce_oi"].fillna(0) + wide["pe_oi"].fillna(0)

    metadata_columns = [
        column
        for column in (
            "symbol",
            "signal_date",
            "group",
            "pred",
            "expiry",
            "strike",
            "ce_instrument_key",
            "pe_instrument_key",
        )
        if column in legs
    ]
    if metadata_columns:
        metadata = legs.groupby("trade_id", as_index=False)[metadata_columns].first()
        wide = wide.merge(metadata, on="trade_id", how="left", validate="many_to_one")
    return wide.reset_index(drop=True)


def _first_valid(values: pd.Series) -> object:
    valid = values.dropna()
    return valid.iloc[0] if not valid.empty else np.nan


def resample_to_5m(
    straddle_1m: pd.DataFrame,
    *,
    require_complete: bool = True,
    session_start: str = "09:15",
    session_end: str = "15:30",
) -> pd.DataFrame:
    """Aggregate start-labelled 1-minute bars into completed 5-minute bars.

    The output timestamp is the decision time (bar end), so the 09:15--09:19
    source candles produce one row stamped 09:20. A model using that row may
    fill no earlier than the 09:20 1-minute open.
    """
    required = {"trade_id", "timestamp", "open", "close", "ce_close", "pe_close"}
    missing = sorted(required - set(straddle_1m.columns))
    if missing:
        raise ValueError(f"missing straddle columns: {missing}")

    bars = straddle_1m.copy()
    bars["timestamp"] = _to_ist(bars["timestamp"])
    bars = bars.dropna(subset=["trade_id", "timestamp", "open", "close"])
    local_time = bars["timestamp"].dt.strftime("%H:%M")
    bars = bars[(local_time >= session_start) & (local_time < session_end)]
    bars["bar_start"] = bars["timestamp"].dt.floor("5min")

    aggregation: dict[str, str | callable] = {
        "timestamp": "count",
        "open": "first",
        "close": "last",
        "ce_close": "last",
        "pe_close": "last",
    }
    for column in ("volume", "ce_volume", "pe_volume"):
        if column in bars:
            aggregation[column] = "sum"
    for column in ("oi", "ce_oi", "pe_oi"):
        if column in bars:
            aggregation[column] = "last"
    for column in ("high_upper_bound",):
        if column in bars:
            aggregation[column] = "max"
    for column in ("low_lower_bound",):
        if column in bars:
            aggregation[column] = "min"
    reserved = {"trade_id", "timestamp", "bar_start", *aggregation}
    for column in bars.columns:
        if column not in reserved:
            aggregation[column] = _first_valid

    grouped = bars.groupby(["trade_id", "bar_start"], as_index=False).agg(aggregation)
    grouped = grouped.rename(columns={"timestamp": "minute_count"})
    if require_complete:
        grouped = grouped[grouped["minute_count"] == 5]
    grouped["timestamp"] = grouped["bar_start"] + pd.Timedelta(minutes=5)
    grouped["bar_minutes"] = 5
    grouped = grouped.drop(columns="bar_start")
    return grouped.sort_values(["trade_id", "timestamp"]).reset_index(drop=True)


def iter_trade_frames(frame: pd.DataFrame) -> Iterable[tuple[str, pd.DataFrame]]:
    if "trade_id" not in frame:
        raise ValueError("bars must contain trade_id")
    for trade_id, bars in frame.groupby("trade_id", sort=False):
        yield str(trade_id), bars.reset_index(drop=True)
