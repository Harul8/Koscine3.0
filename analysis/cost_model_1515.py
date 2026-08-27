"""What is left of the 15:15 Broken-Wing system after real trading costs?

Every prior number in this investigation is GROSS. A 4-leg spread is 8 executed orders (4 in,
4 out) on a mean credit of ~Rs.69, so costs are not a rounding error -- they are plausibly the
whole result. This prices them explicitly.

Statutory costs are exact (NSE F&O options, 2026 rates). Slippage is not knowable from traded
prints alone -- we have last-traded prices, not bid/ask -- so it is swept as a per-leg
assumption rather than guessed at, and the break-even slippage is reported.

Slippage matters more here than statutory cost: the pilot showed the thinnest leg of these
spreads often does not trade for minutes at a time, which is exactly the profile where the
touch is wide.

Usage:
    python analysis/cost_model_1515.py
    python analysis/cost_model_1515.py --file locks/prod_sell_strategies/skew_1515_signal_history.csv --legs 2
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"C:\Users\rahul\Koscine 3.0")

BROKERAGE_PER_ORDER = 20.0      # flat, typical discount broker (Zerodha/Upstox)
STT_SELL_PCT = 0.0625 / 100     # options: on premium, sell side only
EXCH_TXN_PCT = 0.03503 / 100    # NSE F&O options, on premium turnover
SEBI_PCT = 10.0 / 1e7           # Rs.10 per crore
STAMP_PCT = 0.003 / 100         # buy side only
GST_PCT = 0.18                  # on brokerage + exchange + SEBI


def costs_per_lot(sell_prem, buy_prem, exit_val, lot, n_legs):
    """Round-turn cost for one lot, in rupees.

    Entry premium turnover is known exactly. At exit only the NET spread value is recorded, not
    each leg, so exit turnover is approximated by the entry's gross premium scaled by how the
    spread revalued -- conservative in the sense that a spread that collapsed to near zero costs
    little to close, which is what actually happens on the winners.
    """
    gross_entry = (sell_prem + buy_prem) * lot
    scale = np.clip(exit_val / np.maximum(sell_prem - buy_prem, 1e-9), 0.0, 3.0)
    gross_exit = gross_entry * scale
    turnover = gross_entry + gross_exit

    brokerage = BROKERAGE_PER_ORDER * n_legs * 2
    # sell side: the short legs at entry, the long legs at exit -- about half of turnover
    stt = STT_SELL_PCT * turnover * 0.5
    exch = EXCH_TXN_PCT * turnover
    sebi = SEBI_PCT * turnover
    stamp = STAMP_PCT * turnover * 0.5
    gst = GST_PCT * (brokerage + exch + sebi)
    return brokerage + stt + exch + sebi + stamp + gst


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="locks/prod_sell_strategies/bw_1515_signal_history.csv")
    ap.add_argument("--legs", type=int, default=4)
    args = ap.parse_args()

    t = pd.read_csv(ROOT / args.file)
    t = t[t["outcome"] != "pending"].copy()
    lot = t["lot_size"].astype(float)
    t["stat_cost"] = costs_per_lot(t["sell_premium"], t["buy_premium"],
                                   t["exit_value"], lot, args.legs)

    gross = t["pnl_per_lot"].sum()
    risk = t["max_risk_per_lot"].sum()
    print("=" * 92)
    print(f"COST MODEL -- {Path(args.file).name}  ({args.legs} legs, {len(t)} trades)")
    print("=" * 92)
    print(f"  gross total PnL/lot        Rs.{gross:>12,.0f}   ({gross/risk*100:.2f}% of risk)")
    print(f"  mean gross PnL/lot         Rs.{t['pnl_per_lot'].mean():>12,.0f}")
    print(f"  mean statutory cost/lot    Rs.{t['stat_cost'].mean():>12,.0f}   "
          f"({t['stat_cost'].mean()/t['pnl_per_lot'].mean()*100:.0f}% of mean gross PnL)")
    net_stat = gross - t["stat_cost"].sum()
    print(f"  net of statutory only      Rs.{net_stat:>12,.0f}   ({net_stat/risk*100:.2f}% of risk)")

    print()
    print("  Adding SLIPPAGE (cost per leg per side, as % of that leg's premium):")
    print(f"  {'slippage/leg':>13} {'slip cost/lot':>14} {'NET PnL/lot':>14} {'net/risk':>10} "
          f"{'win rate':>9}")
    print("  " + "-" * 66)
    for slip in (0.0, 0.005, 0.01, 0.02, 0.03, 0.05):
        # paid on every leg, both directions, against the premium transacted
        gross_turn = (t["sell_premium"] + t["buy_premium"]) * lot
        scale = np.clip(t["exit_value"] / np.maximum(t["credit"], 1e-9), 0.0, 3.0)
        slip_cost = slip * (gross_turn + gross_turn * scale)
        net = t["pnl_per_lot"] - t["stat_cost"] - slip_cost
        print(f"  {slip*100:>12.1f}% {slip_cost.mean():>14,.0f} {net.sum():>14,.0f} "
              f"{net.sum()/risk*100:>9.2f}% {(net > 0).mean()*100:>8.1f}%")

    # break-even slippage
    lo, hi = 0.0, 0.5
    for _ in range(60):
        mid = (lo + hi) / 2
        gross_turn = (t["sell_premium"] + t["buy_premium"]) * lot
        scale = np.clip(t["exit_value"] / np.maximum(t["credit"], 1e-9), 0.0, 3.0)
        net = (t["pnl_per_lot"] - t["stat_cost"] - mid * (gross_turn + gross_turn * scale)).sum()
        if net > 0:
            lo = mid
        else:
            hi = mid
    print(f"\n  BREAK-EVEN slippage: {lo*100:.2f}% per leg per side")
    print(f"  (above this the strategy loses money; a single tick on a Rs.5 option is 1-2%)")

    print()
    print("=" * 92)
    print("WHAT THE EOD HISTORY TABLE CLAIMS, FOR THE SAME TRADES' WORTH OF ACTIVITY")
    print("=" * 92)
    eod = pd.read_csv(ROOT / "locks/prod_sell_strategies/broken_wing_signal_history.csv")
    eod = eod[(eod["outcome"] != "pending") & (eod["entry_date"] >= t["entry_date"].min())]
    print(f"  EOD table: {len(eod)} trades, win {(eod['outcome']=='win').mean()*100:.1f}%, "
          f"total Rs.{eod['pnl_per_lot'].sum():,.0f}")
    print(f"  realistic: {len(t)} trades, win {(t['outcome']=='win').mean()*100:.1f}%, "
          f"gross Rs.{gross:,.0f}, net-of-statutory Rs.{net_stat:,.0f}")


if __name__ == "__main__":
    main()
