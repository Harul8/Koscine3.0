"""Writes locks/prod_nifty_intraday/signals.json -- the file api/main.py's /prod2/nifty_intraday_*
endpoints serve to the Indices tab's Trades panel, date-picker, and backtest banner.

Reuses validate_recent.py's exact fit (train <= 2026-03-31, test from 2026-04-01 to the latest
available bar) and backtest_signals() (the SAME fire/non-overlap/pattern-exit state machine
production/nifty_live_poller.py runs live) -- this REPLACES the old straddle-premium classifier
this file used to be built from. That old approach had two problems this version fixes:
  (a) it gated fires on data/intraday/nifty_option_chain_5m (the downloaded option-chain
      snapshots, used to price a straddle entry/exit), which lags spot by however far that slow,
      rate-limited downloader happens to be behind -- so the Trades tab silently stopped at
      whatever date the chain data last reached, not the latest actual trading day.
  (b) it flagged EVERY bar in the top-decile predicted-probability quantile as its own
      independent trade, with no "don't fire while a signal is open" state machine -- producing
      stacked, overlapping "trades" a few minutes apart on volatile days.

This version reports predicted-vs-realized MOVE %, not straddle premium P&L -- the strategy
itself changed this session (regime-switching sell-spread / naked-buy off the live option
chain, see production/nifty_live_poller.py's recommend_sell_spread()/recommend_buy()), and a
spot-only backtest isn't gated by the option-chain downloader's lag, so the date range always
reaches the latest available 5-min bar in data/intraday/nifty_5m.parquet.

Usage:
    python experiments/nifty_intraday_breakout_v1/gen_indices_signals.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import validate_recent as vr  # noqa: E402

OUT = Path(r"C:\Users\rahul\Koscine 3.0\locks\prod_nifty_intraday\signals.json")


def _direction_label(d: float | None) -> str | None:
    if d is None or pd.isna(d):
        return None
    if d > 0:
        return "up"
    if d < 0:
        return "down"
    return None


def main() -> None:
    preds = vr.fit_and_predict()
    sig_df = vr.backtest_signals(preds)

    trades: list[dict] = []
    for _, row in sig_df.iterrows():
        entry_ts = pd.Timestamp(row["entry_ts"])
        exit_underlying = row.get("exit_underlying")
        realized = row.get("realized_move_pct")
        exit_reason = row.get("exit_reason")
        trades.append({
            "date": entry_ts.date().isoformat(),
            "entry_ts": row["entry_ts"],
            "exit_ts": row.get("exit_ts") if pd.notna(row.get("exit_ts")) else None,
            "entry_underlying": round(float(row["entry_underlying"]), 1),
            "exit_underlying": round(float(exit_underlying), 1) if pd.notna(exit_underlying) else None,
            "predicted_move_pct": round(float(row["predicted_move_pct"]), 3),
            "realized_move_pct": round(float(realized), 3) if pd.notna(realized) else None,
            "direction": _direction_label(row.get("direction")),
            "exit_reason": exit_reason if pd.notna(exit_reason) else None,
        })

    resolved = [t for t in trades if t["realized_move_pct"] is not None]
    summary = None
    if resolved:
        hits = sum(1 for t in resolved if abs(t["realized_move_pct"]) >= vr.MOVE_THRESHOLD_PCT)
        exit_reasons: dict[str, int] = {}
        for t in resolved:
            exit_reasons[t["exit_reason"]] = exit_reasons.get(t["exit_reason"], 0) + 1
        summary = {
            "n": len(trades), "n_resolved": len(resolved),
            "hit_rate": round(hits / len(resolved), 3),
            "mean_abs_realized_move_pct": round(sum(abs(t["realized_move_pct"]) for t in resolved) / len(resolved), 3),
            "exit_reasons": exit_reasons,
        }

    out = {
        "as_of": pd.Timestamp.now(tz="Asia/Kolkata").isoformat(),
        "horizon_bars": vr.HORIZON, "move_threshold_pct": vr.MOVE_THRESHOLD_PCT,
        "train_cutoff": vr.TRAIN_CUTOFF.date().isoformat(),
        "exit_rule": "pattern/zone-based (direction-lock + erosion + S/R zone + session/max-hold)",
        "backtest_summary": summary,
        "trades": trades,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[gen_indices_signals] wrote {len(trades)} trades ({len(resolved)} resolved) -> {OUT}")
    if summary:
        print(f"  hit_rate={summary['hit_rate']:.1%} mean_abs_move={summary['mean_abs_realized_move_pct']}% n={summary['n']}")
        print(f"  exit_reasons={summary['exit_reasons']}")


if __name__ == "__main__":
    main()
