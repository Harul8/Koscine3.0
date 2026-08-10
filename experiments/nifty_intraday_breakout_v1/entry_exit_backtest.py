"""Runs the NIFTY ATM long-straddle exit-rule grid against every historical fire event of
the entry signal, over the option-chain window (Oct 2024+, same as dataset.py's v2).
Reuses experiments/intraday_exit_v1/simulator.py's ExitRule machinery directly (imported,
not copied -- it's fully generic, no straddle coupling; see README/plan).

IMPORTANT -- fire events come from train.py's WALK-FORWARD OUT-OF-SAMPLE PREDICTED
PROBABILITY (results/v2_option_chain_h{HORIZON}_preds.csv's `p` column), NOT the raw
ground-truth label. An earlier version of this script used the label directly, which is
circular: the label is computed by looking at bars T+1..T+H, so "firing" on
label==1 means "this is a day a big move already happened" -- backtesting a straddle
bought on those days isn't testing a signal, it's testing hindsight. The predictions CSV
is the actually-honest artifact: at each historical timestamp, `p` was produced by a model
trained only on data strictly before that timestamp's walk-forward fold (see
train.py::walk_forward's embargo). Entries here fire on the top ENTRY_QUANTILE of `p`
across that file, matching how train.py's own precision_at_top10pct/lift_over_base_rate
were computed -- so this backtest's win rate should land in the same ballpark as that
metric, not near the (much higher, illegitimate) numbers hindsight-based firing produced.

Every rule below -- including the hold_to_window baseline -- is evaluated over the SAME
fixed MAX_HOLD_BARS window from entry, so the comparison is apples-to-apples: how much of
that window's move can each rule actually capture, not "which rule got a longer runway."

Usage:
    python experiments/nifty_intraday_breakout_v1/entry_exit_backtest.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(r"C:\Users\rahul\Koscine 3.0")
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "experiments" / "intraday_exit_v1"))
import option_features  # noqa: E402
import straddle_path  # noqa: E402
from simulator import ExitRule, run_multiresolution_rule_grid, summarize  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent / "results"
PREDS_FILE = RESULTS_DIR / "v2_option_chain_h12_preds.csv"   # EDIT if you finalized a different horizon
OPTION_CHAIN_DIR = ROOT / "data" / "intraday" / "nifty_option_chain_5m"
MAX_HOLD_BARS = 24   # fixed 2hr window every rule below is evaluated over
ENTRY_QUANTILE = 0.90   # fire on the top 10% of predicted probability -- matches
                         # train.py's own precision_at_top10pct definition

# NIFTY-appropriate grid: intraday_exit_v1's own grid (20-75% activation / 15-30% drawdown)
# was tuned for a 5-DAY stock-option hold -- a same-day (<=2hr) ATM straddle move is a much
# smaller/faster premium swing, so this is a fresh (lower) grid, same rule *shape*.
EXIT_RULES = [ExitRule("hold_to_window")] + [
    ExitRule(f"trail_a{int(a * 100)}_d{int(d * 100)}", activation_return=a, trailing_drawdown=d)
    for a in (0.10, 0.20, 0.30, 0.50) for d in (0.15, 0.20, 0.25)
]
# 2% round-trip cost, wider than intraday_exit_v1's 1% default -- a same-day ATM straddle's
# bid/ask spread + STT/brokerage is a bigger fraction of a smaller, faster premium move.
ROUND_TRIP_COST_PCT = 0.02


def main() -> None:
    if not PREDS_FILE.exists():
        raise SystemExit(f"{PREDS_FILE} not found -- run train.py first (it writes this)")
    preds = pd.read_csv(PREDS_FILE)
    # train.py saved these tz-naive (a since-fixed bug -- .values stripped the Asia/Kolkata
    # offset); they're UTC-equivalent, not wrong data, so recover the original IST instants
    # rather than re-running train.py.
    preds["timestamp"] = pd.to_datetime(preds["timestamp"], utc=True).dt.tz_convert("Asia/Kolkata")

    cutoff = preds["p"].quantile(ENTRY_QUANTILE)
    fires = preds.loc[preds["p"] >= cutoff, "timestamp"].sort_values()
    print(f"[entry_exit_backtest] {len(fires)} historical fire events "
          f"(top {(1 - ENTRY_QUANTILE):.0%} of predicted probability, p>={cutoff:.3f}, "
          f"out of {len(preds)} walk-forward OOS rows)")
    if fires.empty:
        raise SystemExit("no fire events -- nothing to backtest")

    chain = option_features.load_chain(OPTION_CHAIN_DIR)
    trades = []
    skipped = 0
    for t in fires:
        path = straddle_path.build_straddle_path(chain, t, MAX_HOLD_BARS)
        if path is None or len(path) < 2:
            skipped += 1
            continue
        trades.append((t.isoformat(), path, path))   # decision == execution (5-min only)
    print(f"[entry_exit_backtest] {len(trades)} usable straddle paths ({skipped} skipped -- "
          f"no contract/insufficient bars, e.g. very-late-session signals)")
    if not trades:
        raise SystemExit("no usable straddle paths -- nothing to backtest")

    results = run_multiresolution_rule_grid(trades, EXIT_RULES, round_trip_cost_pct=ROUND_TRIP_COST_PCT)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(RESULTS_DIR / "entry_exit_trades.csv", index=False)
    summary = summarize(results)
    summary.to_csv(RESULTS_DIR / "entry_exit_summary.csv", index=False)
    print("\n" + summary.to_string(index=False))

    lines = [
        "# NIFTY intraday breakout -- entry+exit findings", "",
        f"{len(trades)} straddle trades, {MAX_HOLD_BARS}-bar (2hr) fixed window, "
        f"entries = top {(1 - ENTRY_QUANTILE):.0%} predicted probability (p>={cutoff:.3f}).",
        "",
        "`hold_to_window` is the baseline every trailing-stop rule must beat after costs "
        f"({ROUND_TRIP_COST_PCT:.0%} round-trip assumed -- adjust to your real fees before trusting this).",
        "", "```", summary.to_string(index=False), "```", "",
    ]
    (RESULTS_DIR / "FINDINGS_exit.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
