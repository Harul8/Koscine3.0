"""Recomputes fire events + simulates the FINALIZED exit rule over the full option-chain
window, writing locks/prod_nifty_intraday/signals.json -- the file api/main.py's
/prod2/nifty_intraday_* endpoints serve to the Indices tab. Swapping in a live 5-min poller
later means replacing what writes this file, not the API/frontend reading it.

IMPORTANT -- fire events come from train.py's WALK-FORWARD OUT-OF-SAMPLE PREDICTED
PROBABILITY (results/v2_option_chain_h{HORIZON}_preds.csv's `p` column), NOT the raw
ground-truth label -- see entry_exit_backtest.py's docstring for why using the label
directly is circular (it's computed by looking at bars T+1..T+H, so it can only be known
in hindsight). This means the trades shown in the tab are HISTORICAL walk-forward-honest
signals, not a live feed -- there is no single deployed model yet to score new/live bars
with (train.py retrains a fresh model every walk-forward fold, by design, for evaluation --
that's a different thing from a persisted production model). Standing up genuinely live
scoring is a later step, not this one.

EDIT EXIT_RULE below to whichever rule won in entry_exit_backtest.py's
results/FINDINGS_exit.md -- this script does not pick it for you. It ships with the plain
hold_to_window rule as a defensible, non-overfit default until you've looked at that
comparison.

Usage:
    python experiments/nifty_intraday_breakout_v1/gen_indices_signals.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(r"C:\Users\rahul\Koscine 3.0")
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "experiments" / "intraday_exit_v1"))
import option_features  # noqa: E402
import straddle_path  # noqa: E402
from simulator import ExitRule, simulate_multiresolution  # noqa: E402

HORIZON = 12   # EDIT together with PREDS_FILE if you finalized a different horizon
RESULTS_DIR = Path(__file__).resolve().parent / "results"
PREDS_FILE = RESULTS_DIR / f"v2_option_chain_h{HORIZON}_preds.csv"
DATA_DIR = Path(__file__).resolve().parent / "data"
OPTION_CHAIN_DIR = ROOT / "data" / "intraday" / "nifty_option_chain_5m"
OUT = ROOT / "locks" / "prod_nifty_intraday" / "signals.json"
MAX_HOLD_BARS = 24
ROUND_TRIP_COST_PCT = 0.02
ENTRY_QUANTILE = 0.90   # matches entry_exit_backtest.py / train.py's precision_at_top10pct

# From entry_exit_backtest.py's corrected results/FINDINGS_exit.md: every rule tested was
# net-negative on average (mean_net_return -3.0% to -4.9% across the grid) -- this is the
# LEAST-BAD rule, not a validated winner. See the tab's own banner/methodology text for the
# honest caveat; this is populated for visualization of the research finding, not a live signal.
EXIT_RULE = ExitRule("trail_a10_d15", activation_return=0.10, trailing_drawdown=0.15)


def main() -> None:
    if not PREDS_FILE.exists():
        raise SystemExit(f"{PREDS_FILE} not found -- run train.py first (it writes this)")
    preds = pd.read_csv(PREDS_FILE)
    preds["timestamp"] = pd.to_datetime(preds["timestamp"], utc=True).dt.tz_convert("Asia/Kolkata")
    cutoff = preds["p"].quantile(ENTRY_QUANTILE)
    fires = preds.loc[preds["p"] >= cutoff, "timestamp"].sort_values()

    v2_path = DATA_DIR / "v2_option_chain.parquet"
    if not v2_path.exists():
        raise SystemExit(f"{v2_path} not found -- run dataset.py first")
    v2 = pd.read_parquet(v2_path)
    v2["timestamp"] = pd.to_datetime(v2["timestamp"])
    underlying_by_ts = v2.set_index("timestamp")["close"]

    chain = option_features.load_chain(OPTION_CHAIN_DIR)

    trades: list[dict] = []
    for t in fires:
        underlying_entry = underlying_by_ts.get(t)
        if underlying_entry is None:
            continue
        path = straddle_path.build_straddle_path(chain, t, MAX_HOLD_BARS)
        if path is None or len(path) < 2:
            continue
        result = simulate_multiresolution(path, path, EXIT_RULE, round_trip_cost_pct=ROUND_TRIP_COST_PCT)
        exit_underlying = underlying_by_ts.get(result.exit_timestamp)
        trades.append({
            "date": t.date().isoformat(),
            "entry_ts": result.entry_timestamp.isoformat(),
            "exit_ts": result.exit_timestamp.isoformat(),
            "entry_underlying": round(float(underlying_entry), 1),
            "exit_underlying": round(float(exit_underlying), 1) if exit_underlying is not None else None,
            "straddle_entry": round(result.entry_value, 2),
            "straddle_exit": round(result.exit_value, 2),
            "pnl_pct": round(result.net_return * 100, 2),
            "exit_reason": result.exit_reason,
        })

    summary = None
    if trades:
        pnls = [t["pnl_pct"] for t in trades]
        summary = {
            "n": len(trades),
            "win_rate": round(sum(1 for p in pnls if p > 0) / len(pnls), 3),
            "avg_pnl_pct": round(sum(pnls) / len(pnls), 2),
            "median_pnl_pct": round(sorted(pnls)[len(pnls) // 2], 2),
            "worst_pnl_pct": round(min(pnls), 2),
        }

    out = {
        "as_of": pd.Timestamp.now().isoformat(),
        "horizon_bars": HORIZON, "entry_quantile": ENTRY_QUANTILE, "p_cutoff": round(float(cutoff), 4),
        "exit_rule": EXIT_RULE.name,
        "backtest_summary": summary,
        "trades": trades,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[gen_indices_signals] wrote {len(trades)} trades -> {OUT}")
    if summary:
        print(f"  win_rate={summary['win_rate']:.1%} avg_pnl={summary['avg_pnl_pct']}% n={summary['n']}")


if __name__ == "__main__":
    main()
