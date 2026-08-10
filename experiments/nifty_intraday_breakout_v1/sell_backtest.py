"""Tests whether the breakout model's predicted probability is useful as a RISK FILTER for
an intraday NIFTY premium-selling (Broken-Wing credit spread) strategy -- the flip side of
the long-straddle buying approach that backtested net-negative. The model correctly
over-samples actual big-move days at the margin (20-26% hit rate vs 14% base rate at
tighter conviction) -- exactly the days a premium SELLER needs to avoid, even though that
same edge wasn't enough to justify BUYING premium (buying needs to overcome theta decay on
every trade; selling only needs to avoid the rare bad day).

Runs non-overlapping entry checkpoints every CHECKPOINT_BARS through the trading session,
across the SAME checkpoint set for two variants so the comparison is apples-to-apples:
  - unconditional: sell the spread at every checkpoint, no ML filter
  - filtered:       skip checkpoints where predicted probability p is in the top decile
                     (results/v2_option_chain_h12_preds.csv, the same honest walk-forward
                     out-of-sample artifact used throughout -- see entry_exit_backtest.py's
                     docstring for why the raw label can't be used here)
The actual test is whether FILTERED beats UNCONDITIONAL, not whether either alone is
profitable -- that's what tells you whether the model adds value to a seller.

One fixed Broken-Wing structure (short_otm=0.3%, wing_ce=0.3%, wing_pe=0.6% -- tight,
intraday-appropriate distances, an order of magnitude smaller than the EOD version's
2%/5-10% since this is a ~1-2hr hold, not 5 days) against a small stop-loss exit-rule grid.
Structure-grid tuning is a follow-up if this shows any life; the goal here is the
filtered-vs-unconditional question, not premature optimization.

Usage:
    python experiments/nifty_intraday_breakout_v1/sell_backtest.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(r"C:\Users\rahul\Koscine 3.0")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import credit_spread_path  # noqa: E402
import option_features  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
PREDS_FILE = RESULTS_DIR / "v2_option_chain_h12_preds.csv"
OPTION_CHAIN_DIR = ROOT / "data" / "intraday" / "nifty_option_chain_5m"

CHECKPOINT_BARS = 12     # 1hr between candidate entries, non-overlapping
MAX_HOLD_BARS = 24       # 2hr fixed window, same as the buying backtest
SHORT_OTM, WING_CE, WING_PE = 0.003, 0.003, 0.006   # tight, asymmetric (narrow call/wide put)
STOP_LOSS_GRID = [None, -0.50, -0.75, -1.00]        # None = hold_to_window baseline
FILTER_QUANTILE = 0.90   # skip the top 10% of predicted probability (matches
                          # entry_exit_backtest.py's ENTRY_QUANTILE, opposite direction)


def main() -> None:
    if not PREDS_FILE.exists():
        raise SystemExit(f"{PREDS_FILE} not found -- run train.py first (it writes this)")
    preds = pd.read_csv(PREDS_FILE)
    preds["timestamp"] = pd.to_datetime(preds["timestamp"], utc=True).dt.tz_convert("Asia/Kolkata")
    p_by_ts = preds.set_index("timestamp")["p"]
    high_risk_cutoff = preds["p"].quantile(FILTER_QUANTILE)

    v2_path = DATA_DIR / "v2_option_chain.parquet"
    if not v2_path.exists():
        raise SystemExit(f"{v2_path} not found -- run dataset.py first")
    v2 = pd.read_parquet(v2_path)
    v2["timestamp"] = pd.to_datetime(v2["timestamp"])

    # Checkpoints: every CHECKPOINT_BARS-th bar per day, restricted to timestamps that have
    # a walk-forward prediction (so unconditional and filtered draw from the identical set).
    v2["date"] = v2["timestamp"].dt.date
    checkpoints = []
    for _date, g in v2.sort_values("timestamp").groupby("date", sort=False):
        ts_list = g["timestamp"].tolist()
        for i in range(0, max(0, len(ts_list) - MAX_HOLD_BARS), CHECKPOINT_BARS):
            t = ts_list[i]
            if t in p_by_ts.index:
                checkpoints.append(t)
    print(f"[sell_backtest] {len(checkpoints)} candidate checkpoints "
          f"(every {CHECKPOINT_BARS} bars, with a matching walk-forward prediction)")
    if not checkpoints:
        raise SystemExit("no checkpoints -- nothing to backtest")

    chain = option_features.load_chain(OPTION_CHAIN_DIR)

    def run(label: str, checkpoint_list: list) -> pd.DataFrame:
        rows = []
        skipped = 0
        for t in checkpoint_list:
            spread = credit_spread_path.open_credit_spread(
                chain, t, SHORT_OTM, WING_CE, WING_PE, MAX_HOLD_BARS)
            if spread is None:
                skipped += 1
                continue
            for sl in STOP_LOSS_GRID:
                r = credit_spread_path.simulate_exit(spread, sl)
                r["variant"] = label
                r["rule"] = "hold_to_window" if sl is None else f"stop_{int(-sl * 100)}"
                r["signal_ts"] = t
                rows.append(r)
        print(f"[sell_backtest] {label}: {len(checkpoint_list)} checkpoints, "
              f"{skipped} skipped (no contract/insufficient bars)")
        return pd.DataFrame(rows)

    filtered_checkpoints = [t for t in checkpoints if p_by_ts.loc[t] < high_risk_cutoff]
    print(f"[sell_backtest] filtered: {len(filtered_checkpoints)}/{len(checkpoints)} checkpoints "
          f"survive excluding top {(1 - FILTER_QUANTILE):.0%} predicted probability (p>={high_risk_cutoff:.3f})")

    results_unconditional = run("unconditional", checkpoints)
    results_filtered = run("filtered", filtered_checkpoints)
    results = pd.concat([results_unconditional, results_filtered], ignore_index=True)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(RESULTS_DIR / "sell_backtest_trades.csv", index=False)

    summary = (results.groupby(["variant", "rule"])
               .agg(trades=("pnl", "size"),
                    win_rate=("outcome", lambda s: float((s == "win").mean())),
                    mean_ror_pct=("ror_pct", "mean"), median_ror_pct=("ror_pct", "median"),
                    worst_ror_pct=("ror_pct", "min"), worst_dd_pct=("max_dd_pct", "min"),
                    total_pnl=("pnl", "sum"))
               .round(2).reset_index()
               .sort_values(["rule", "variant"]))
    summary.to_csv(RESULTS_DIR / "sell_backtest_summary.csv", index=False)
    print("\n" + summary.to_string(index=False))

    lines = [
        "# NIFTY intraday premium-selling -- filtered vs unconditional findings", "",
        f"Broken-Wing structure: short_otm={SHORT_OTM:.2%}, wing_ce={WING_CE:.2%}, wing_pe={WING_PE:.2%}. "
        f"{CHECKPOINT_BARS}-bar (1hr) checkpoint cadence, {MAX_HOLD_BARS}-bar (2hr) max hold.",
        "",
        "The number that matters is FILTERED vs UNCONDITIONAL at the same rule -- not whether either",
        "alone is net-positive. If filtering out top-decile predicted-probability checkpoints doesn't",
        "measurably improve win_rate/mean_ror_pct/worst_dd_pct over unconditional, the model isn't",
        "adding value here either.",
        "", "```", summary.to_string(index=False), "```", "",
    ]
    (RESULTS_DIR / "FINDINGS_sell.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
