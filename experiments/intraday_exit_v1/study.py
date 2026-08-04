"""Run exit rules using 5-minute decisions and 1-minute executions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from simulator import run_multiresolution_rule_grid, summarize


HERE = Path(__file__).resolve().parent


def _read(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--execution-bars",
        type=Path,
        default=Path("data/intraday/options_straddle_1m.parquet"),
    )
    parser.add_argument(
        "--decision-bars",
        type=Path,
        default=Path("data/intraday/options_straddle_5m.parquet"),
    )
    parser.add_argument("--cost-pct", type=float, default=0.01)
    parser.add_argument("--entry-slippage-bps", type=float, default=10)
    parser.add_argument("--exit-slippage-bps", type=float, default=10)
    args = parser.parse_args()

    execution_bars = _read(args.execution_bars)
    decision_bars = _read(args.decision_bars)
    for name, bars in (("execution", execution_bars), ("decision", decision_bars)):
        if "trade_id" not in bars:
            raise ValueError(f"{name} bars must contain trade_id")
    decisions_by_trade = {
        str(trade_id): group
        for trade_id, group in decision_bars.groupby("trade_id", sort=False)
    }
    executions_by_trade = {
        str(trade_id): group
        for trade_id, group in execution_bars.groupby("trade_id", sort=False)
    }
    common_ids = sorted(decisions_by_trade.keys() & executions_by_trade.keys())
    if not common_ids:
        raise ValueError("execution and decision inputs contain no common trade_id values")
    trades = (
        (trade_id, decisions_by_trade[trade_id], executions_by_trade[trade_id])
        for trade_id in common_ids
    )
    results = run_multiresolution_rule_grid(
        trades,
        entry_slippage_bps=args.entry_slippage_bps,
        exit_slippage_bps=args.exit_slippage_bps,
        round_trip_cost_pct=args.cost_pct,
    )
    summary = summarize(results)
    output = HERE / "results"
    output.mkdir(exist_ok=True)
    results.to_parquet(output / "exit_rule_trades.parquet", index=False)
    summary.to_csv(output / "exit_rule_summary.csv", index=False)
    metadata = {
        "execution_bars": str(args.execution_bars),
        "decision_bars": str(args.decision_bars),
        "decision_resolution": "5minute",
        "execution_resolution": "1minute",
        "trades": int(results.trade_id.nunique()),
        "cost_pct": args.cost_pct,
        "entry_slippage_bps": args.entry_slippage_bps,
        "exit_slippage_bps": args.exit_slippage_bps,
    }
    (output / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
