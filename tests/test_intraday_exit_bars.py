from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]


def load(name: str):
    path = ROOT / "experiments" / "intraday_exit_v1" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"intraday_exit_{name}", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


bars_module = load("bars")
features_module = load("features")
universe_module = load("universe")


def leg_bars() -> pd.DataFrame:
    timestamps = pd.date_range(
        "2026-01-05 09:15", periods=10, freq="1min", tz="Asia/Kolkata"
    )
    rows = []
    for option_type, offset in (("CE", 10), ("PE", 20)):
        for index, timestamp in enumerate(timestamps):
            value = offset + index
            rows.append(
                {
                    "trade_id": "T1",
                    "option_type": option_type,
                    "timestamp": timestamp,
                    "open": value,
                    "high": value + 2,
                    "low": value - 2,
                    "close": value + 1,
                    "volume": 100 + index,
                    "oi": 1000 + index,
                    "symbol": "RELIANCE",
                }
            )
    return pd.DataFrame(rows)


def test_build_1m_and_end_stamp_complete_5m_bars() -> None:
    one_minute = bars_module.build_straddle_1m(leg_bars())
    five_minute = bars_module.resample_to_5m(one_minute)
    assert len(one_minute) == 10
    assert list(five_minute["minute_count"]) == [5, 5]
    assert five_minute.iloc[0].timestamp == pd.Timestamp(
        "2026-01-05 09:20", tz="Asia/Kolkata"
    )
    assert five_minute.iloc[0].open == pytest.approx(30)
    assert five_minute.iloc[0].close == pytest.approx(40)
    assert five_minute.iloc[1].open == pytest.approx(40)
    assert five_minute.iloc[1].close == pytest.approx(50)


def test_training_features_do_not_mix_with_forward_targets() -> None:
    timestamps = pd.date_range(
        "2026-01-05 09:20", periods=8, freq="5min", tz="Asia/Kolkata"
    )
    frame = pd.DataFrame(
        {
            "trade_id": "T1",
            "timestamp": timestamps,
            "close": [100, 102, 104, 103, 101, 100, 99, 98],
            "ce_close": [50, 53, 56, 55, 52, 51, 50, 49],
            "pe_close": [50, 49, 48, 48, 49, 49, 49, 49],
        }
    )
    training = features_module.build_exit_training_frame(frame, label_horizon_bars=2)
    feature_names = features_module.feature_columns(training)
    target_names = features_module.target_columns(training)
    assert feature_names
    assert target_names
    assert set(feature_names).isdisjoint(target_names)
    assert training.loc[0, "target_future_max_return"] == pytest.approx(0.04)
    assert pd.isna(training.loc[len(training) - 1, "target_exit_now"])


def test_universe_enforces_ranked_top30_and_four_indices() -> None:
    assert len(universe_module.TOP30_STOCKS) == 30
    assert len(set(universe_module.TOP30_STOCKS)) == 30
    universe_module.require_in_universe(
        ["NIFTY", "BANKNIFTY", "SENSEX", "FINNIFTY", "RELIANCE", "BSE"]
    )
    with pytest.raises(ValueError, match="outside"):
        universe_module.require_in_universe(["APOLLOHOSP"])
