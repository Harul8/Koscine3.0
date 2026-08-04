from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pandas as pd
import pytest


MODULE = Path(__file__).resolve().parents[1] / "experiments" / "intraday_exit_v1" / "simulator.py"
SPEC = importlib.util.spec_from_file_location("intraday_exit_simulator", MODULE)
simulator = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = simulator
SPEC.loader.exec_module(simulator)


def bars(values: list[tuple[float, float]]) -> pd.DataFrame:
    timestamps = pd.date_range("2026-01-05 09:30", periods=len(values), freq="15min", tz="Asia/Kolkata")
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": [value[0] for value in values],
            "high": [max(value) for value in values],
            "low": [min(value) for value in values],
            "close": [value[1] for value in values],
        }
    )


def test_trailing_trigger_executes_at_next_bar_open() -> None:
    path = bars([(100, 100), (105, 140), (138, 110), (107, 106)])
    rule = simulator.ExitRule("trail", activation_return=0.20, trailing_drawdown=0.20)
    result = simulator.simulate(
        path, rule, entry_slippage_bps=0, exit_slippage_bps=0, round_trip_cost_pct=0
    )
    assert result.exit_reason == "trailing_stop"
    assert result.exit_timestamp == path.iloc[3].timestamp
    assert result.exit_value == pytest.approx(107)
    assert result.gross_return == pytest.approx(0.07)


def test_hold_uses_final_close_and_applies_costs() -> None:
    path = bars([(100, 100), (110, 115), (120, 125)])
    result = simulator.simulate(
        path,
        simulator.ExitRule("hold"),
        entry_slippage_bps=0,
        exit_slippage_bps=0,
        round_trip_cost_pct=0.03,
    )
    assert result.exit_reason == "time_exit"
    assert result.gross_return == pytest.approx(0.25)
    assert result.net_return == pytest.approx(0.22)


def test_same_bar_low_cannot_trigger_before_activation_close() -> None:
    path = bars([(100, 100), (100, 130), (129, 128)])
    path.loc[1, "low"] = 70
    rule = simulator.ExitRule("trail", activation_return=0.20, trailing_drawdown=0.20)
    result = simulator.simulate(
        path, rule, entry_slippage_bps=0, exit_slippage_bps=0, round_trip_cost_pct=0
    )
    assert result.exit_reason == "time_exit"
    assert result.exit_value == pytest.approx(128)


def test_5m_trigger_fills_at_same_timestamp_1m_open() -> None:
    decision_timestamps = pd.date_range(
        "2026-01-05 09:20", periods=3, freq="5min", tz="Asia/Kolkata"
    )
    decisions = pd.DataFrame(
        {
            "timestamp": decision_timestamps,
            "close": [100, 140, 110],
        }
    )
    execution_timestamps = pd.date_range(
        "2026-01-05 09:20", periods=12, freq="1min", tz="Asia/Kolkata"
    )
    executions = pd.DataFrame(
        {
            "timestamp": execution_timestamps,
            "open": [100, 102, 104, 106, 108, 139, 135, 130, 120, 115, 107, 106],
            "close": [101, 103, 105, 107, 109, 140, 136, 131, 121, 116, 108, 105],
        }
    )
    rule = simulator.ExitRule("trail", activation_return=0.20, trailing_drawdown=0.20)
    result = simulator.simulate_multiresolution(
        decisions,
        executions,
        rule,
        entry_slippage_bps=0,
        exit_slippage_bps=0,
        round_trip_cost_pct=0,
    )
    assert result.entry_timestamp == decision_timestamps[0]
    assert result.exit_timestamp == decision_timestamps[2]
    assert result.exit_value == pytest.approx(107)
    assert result.gross_return == pytest.approx(0.07)


def test_first_completed_5m_bar_cannot_trigger_post_entry_exit() -> None:
    decisions = pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2026-01-05 09:20", periods=2, freq="5min", tz="Asia/Kolkata"
            ),
            "close": [150, 100],
        }
    )
    executions = pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2026-01-05 09:20", periods=7, freq="1min", tz="Asia/Kolkata"
            ),
            "open": [100, 101, 102, 103, 104, 105, 106],
            "close": [101, 102, 103, 104, 105, 106, 107],
        }
    )
    result = simulator.simulate_multiresolution(
        decisions,
        executions,
        simulator.ExitRule("trail", activation_return=0.20, trailing_drawdown=0.20),
        entry_slippage_bps=0,
        exit_slippage_bps=0,
        round_trip_cost_pct=0,
    )
    assert result.exit_reason == "time_exit"
