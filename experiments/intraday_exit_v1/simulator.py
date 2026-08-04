"""Path-safe simulator for fixed-contract intraday straddle bars.

Triggers are evaluated on completed bar closes and filled at the next bar open.
This avoids the common look-ahead error of triggering and filling from the same
OHLC bar, whose price ordering is unknowable.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {"timestamp", "open", "high", "low", "close"}
DECISION_REQUIRED_COLUMNS = {"timestamp", "close"}
EXECUTION_REQUIRED_COLUMNS = {"timestamp", "open", "close"}


@dataclass(frozen=True)
class ExitRule:
    name: str
    activation_return: float | None = None
    trailing_drawdown: float | None = None

    @property
    def is_hold(self) -> bool:
        return self.activation_return is None or self.trailing_drawdown is None


@dataclass(frozen=True)
class SimulationResult:
    rule: str
    entry_timestamp: pd.Timestamp
    exit_timestamp: pd.Timestamp
    entry_value: float
    exit_value: float
    gross_return: float
    net_return: float
    observed_peak_return: float
    capture_ratio: float | None
    exit_reason: str
    bars_held: int


def default_rules() -> list[ExitRule]:
    rules = [ExitRule("hold_5d")]
    for activation in (0.20, 0.35, 0.50, 0.75):
        for drawdown in (0.15, 0.20, 0.25, 0.30):
            rules.append(
                ExitRule(
                    f"trail_a{int(activation * 100)}_d{int(drawdown * 100)}",
                    activation_return=activation,
                    trailing_drawdown=drawdown,
                )
            )
    return rules


def validate_bars(bars: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(REQUIRED_COLUMNS - set(bars.columns))
    if missing:
        raise ValueError(f"missing intraday bar columns: {missing}")
    out = bars.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"])
    out = out.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    for column in ("open", "high", "low", "close"):
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out = out.dropna(subset=["timestamp", "open", "close"])
    if len(out) < 2:
        raise ValueError("at least two valid bars are required for next-bar execution")
    if (out[["open", "close"]] <= 0).any().any():
        raise ValueError("option values must be positive")
    if not out["timestamp"].is_monotonic_increasing:
        raise ValueError("timestamps must be strictly ordered")
    return out


def _validate_resolution_bars(
    bars: pd.DataFrame,
    required: set[str],
    *,
    name: str,
) -> pd.DataFrame:
    missing = sorted(required - set(bars.columns))
    if missing:
        raise ValueError(f"missing {name} columns: {missing}")
    out = bars.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"])
    out = out.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    for column in required - {"timestamp"}:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out = out.dropna(subset=list(required))
    if out.empty:
        raise ValueError(f"no valid {name} remain")
    if (out[list(required - {"timestamp"})] <= 0).any().any():
        raise ValueError(f"{name} prices must be positive")
    return out


def simulate(
    bars: pd.DataFrame,
    rule: ExitRule,
    *,
    entry_slippage_bps: float = 10.0,
    exit_slippage_bps: float = 10.0,
    round_trip_cost_pct: float = 0.01,
) -> SimulationResult:
    path = validate_bars(bars)
    entry_value = float(path.iloc[0].open) * (1 + entry_slippage_bps / 10_000)
    peak_close = float(path.iloc[0].close)
    trigger_index: int | None = None
    activated = False

    if not rule.is_hold:
        assert rule.activation_return is not None
        assert rule.trailing_drawdown is not None
        for index in range(len(path) - 1):
            close = float(path.iloc[index].close)
            peak_close = max(peak_close, close)
            if close / entry_value - 1 >= rule.activation_return:
                activated = True
            if activated and close / peak_close - 1 <= -rule.trailing_drawdown:
                trigger_index = index
                break

    if trigger_index is None:
        exit_index = len(path) - 1
        raw_exit = float(path.iloc[exit_index].close)
        reason = "time_exit"
    else:
        exit_index = trigger_index + 1
        raw_exit = float(path.iloc[exit_index].open)
        reason = "trailing_stop"

    exit_value = raw_exit * (1 - exit_slippage_bps / 10_000)
    gross_return = exit_value / entry_value - 1
    net_return = gross_return - round_trip_cost_pct
    observed_peak_return = float(path.close.max()) / entry_value - 1
    capture_ratio = None
    if observed_peak_return > 0:
        capture_ratio = float(np.clip(gross_return / observed_peak_return, -5.0, 1.0))

    return SimulationResult(
        rule=rule.name,
        entry_timestamp=pd.Timestamp(path.iloc[0].timestamp),
        exit_timestamp=pd.Timestamp(path.iloc[exit_index].timestamp),
        entry_value=entry_value,
        exit_value=exit_value,
        gross_return=gross_return,
        net_return=net_return,
        observed_peak_return=observed_peak_return,
        capture_ratio=capture_ratio,
        exit_reason=reason,
        bars_held=int(exit_index),
    )


def simulate_multiresolution(
    decision_bars: pd.DataFrame,
    execution_bars: pd.DataFrame,
    rule: ExitRule,
    *,
    entry_slippage_bps: float = 10.0,
    exit_slippage_bps: float = 10.0,
    round_trip_cost_pct: float = 0.01,
) -> SimulationResult:
    """Evaluate on completed 5-minute closes and fill on 1-minute opens.

    Decision timestamps denote bar ends. If a trigger is stamped 10:05, the
    fill is the first 1-minute candle whose start timestamp is at or after
    10:05. The first decision bar is reserved for entry and cannot also trigger
    an exit.
    """
    decisions = _validate_resolution_bars(
        decision_bars, DECISION_REQUIRED_COLUMNS, name="decision-bar"
    )
    executions = _validate_resolution_bars(
        execution_bars, EXECUTION_REQUIRED_COLUMNS, name="execution-bar"
    )
    entry_decision_timestamp = pd.Timestamp(decisions.iloc[0].timestamp)
    entry_candidates = executions[executions.timestamp >= entry_decision_timestamp]
    if entry_candidates.empty:
        raise ValueError("no 1-minute execution bar exists at or after the first decision")

    entry_row = entry_candidates.iloc[0]
    entry_timestamp = pd.Timestamp(entry_row.timestamp)
    entry_value = float(entry_row.open) * (1 + entry_slippage_bps / 10_000)
    eligible_decisions = decisions[
        (decisions.timestamp > entry_timestamp)
        & (decisions.timestamp <= executions.iloc[-1].timestamp)
    ]

    peak_close = entry_value
    trigger_timestamp: pd.Timestamp | None = None
    activated = False
    if not rule.is_hold:
        assert rule.activation_return is not None
        assert rule.trailing_drawdown is not None
        for row in eligible_decisions.itertuples(index=False):
            close = float(row.close)
            peak_close = max(peak_close, close)
            if close / entry_value - 1 >= rule.activation_return:
                activated = True
            if activated and close / peak_close - 1 <= -rule.trailing_drawdown:
                trigger_timestamp = pd.Timestamp(row.timestamp)
                break

    if trigger_timestamp is None:
        exit_row = executions.iloc[-1]
        raw_exit = float(exit_row.close)
        reason = "time_exit"
    else:
        exit_candidates = executions[executions.timestamp >= trigger_timestamp]
        if exit_candidates.empty:
            exit_row = executions.iloc[-1]
            raw_exit = float(exit_row.close)
            reason = "time_exit"
        else:
            exit_row = exit_candidates.iloc[0]
            raw_exit = float(exit_row.open)
            reason = "trailing_stop"

    exit_timestamp = pd.Timestamp(exit_row.timestamp)
    exit_value = raw_exit * (1 - exit_slippage_bps / 10_000)
    gross_return = exit_value / entry_value - 1
    net_return = gross_return - round_trip_cost_pct
    observable_path = executions[
        (executions.timestamp >= entry_timestamp) & (executions.timestamp <= exit_timestamp)
    ]
    observed_peak_return = float(observable_path.close.max()) / entry_value - 1
    capture_ratio = None
    if observed_peak_return > 0:
        capture_ratio = float(np.clip(gross_return / observed_peak_return, -5.0, 1.0))
    bars_held = int(
        (
            (decisions.timestamp > entry_timestamp)
            & (decisions.timestamp <= exit_timestamp)
        ).sum()
    )
    return SimulationResult(
        rule=rule.name,
        entry_timestamp=entry_timestamp,
        exit_timestamp=exit_timestamp,
        entry_value=entry_value,
        exit_value=exit_value,
        gross_return=gross_return,
        net_return=net_return,
        observed_peak_return=observed_peak_return,
        capture_ratio=capture_ratio,
        exit_reason=reason,
        bars_held=bars_held,
    )


def run_rule_grid(
    trades: Iterable[tuple[str, pd.DataFrame]],
    rules: Iterable[ExitRule] | None = None,
    **simulation_kwargs: float,
) -> pd.DataFrame:
    selected_rules = list(rules or default_rules())
    rows: list[dict[str, object]] = []
    for trade_id, bars in trades:
        for rule in selected_rules:
            result = asdict(simulate(bars, rule, **simulation_kwargs))
            result["trade_id"] = trade_id
            rows.append(result)
    return pd.DataFrame(rows)


def run_multiresolution_rule_grid(
    trades: Iterable[tuple[str, pd.DataFrame, pd.DataFrame]],
    rules: Iterable[ExitRule] | None = None,
    **simulation_kwargs: float,
) -> pd.DataFrame:
    """Run rules for `(trade_id, decision_5m, execution_1m)` inputs."""
    selected_rules = list(rules or default_rules())
    rows: list[dict[str, object]] = []
    for trade_id, decision_bars, execution_bars in trades:
        for rule in selected_rules:
            result = asdict(
                simulate_multiresolution(
                    decision_bars,
                    execution_bars,
                    rule,
                    **simulation_kwargs,
                )
            )
            result["trade_id"] = trade_id
            rows.append(result)
    return pd.DataFrame(rows)


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    required = {"rule", "trade_id", "net_return", "capture_ratio", "bars_held"}
    missing = sorted(required - set(results.columns))
    if missing:
        raise ValueError(f"missing simulation result columns: {missing}")
    return (
        results.groupby("rule", as_index=False)
        .agg(
            trades=("trade_id", "nunique"),
            mean_net_return=("net_return", "mean"),
            median_net_return=("net_return", "median"),
            win_rate=("net_return", lambda x: float((x > 0).mean())),
            mean_peak_capture=("capture_ratio", "mean"),
            mean_bars_held=("bars_held", "mean"),
            p05_net_return=("net_return", lambda x: float(x.quantile(0.05))),
        )
        .sort_values("mean_net_return", ascending=False)
    )
