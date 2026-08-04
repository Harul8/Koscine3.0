"""Analyze the 3-day variant: held-to-3 vs peak exit, by group (A ATM+1% / B ATM+2%), top stocks, timing.

    python experiments/option_gain_study_v1/option_gain_3d_analyze.py
Reads results/option_gain_3d_trades.csv. Cheap.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
d = pd.read_csv(HERE / "results" / "option_gain_3d_trades.csv", parse_dates=["entry_date"])
d["abs_move"] = d.stock_move.abs() * 100
COST = 0.30  # ~30% round-trip on premium? NO — express EV as ratio; show net of a flat 3% later
print(f"trades {len(d):,} | {d.entry_date.min().date()}..{d.entry_date.max().date()} | symbols {d.symbol.nunique()} "
      f"| A=ATM+1%, B=ATM+2%, hold 3d, entry@open, pennies dropped\n")

EXITS = [("held day1", "close1_ratio"), ("held day2", "close2_ratio"), ("held day3", "held_ratio"),
         ("PEAK (oracle)", "peak_ratio")]


def tbl(df):
    rows = {}
    for lab, col in EXITS:
        s = df[col].dropna()
        rows[lab] = {"mean": round(s.mean(), 2), "median": round(s.median(), 2),
                     "P>=2x": round((s >= 2).mean(), 3), "P>=3x": round((s >= 3).mean(), 3),
                     "P(loss<0.5x)": round((s < 0.5).mean(), 3)}
    return pd.DataFrame(rows).T


print("=== 1) EXIT COMPARISON — all options (both sides pooled) ===")
print(tbl(d).to_string())
print("\n=== 2) BY GROUP (A ATM+1% vs B ATM+2%) ===")
for g, sub in d.groupby("group"):
    print(f"\n-- {g} (n={len(sub)}) --")
    print(tbl(sub).to_string())

print("\n=== 3) CONVEXITY AVAILABLE: best option per stock-day (oracle side+strike) ===")
best = d.groupby(["symbol", "entry_date"]).agg(best_peak=("peak_ratio", "max"), best_held=("held_ratio", "max"),
                                               abs_move=("abs_move", "first")).reset_index()
print(f"   stock-days {len(best):,} | PEAK: median {best.best_peak.median():.2f}x  P>=2x {(best.best_peak>=2).mean():.3f}"
      f"  >=3x {(best.best_peak>=3).mean():.3f}  >=5x {(best.best_peak>=5).mean():.3f}")
print(f"   stock-days HELD-day3: median {best.best_held.median():.2f}x  P>=2x {(best.best_held>=2).mean():.3f}  >=3x {(best.best_held>=3).mean():.3f}")

print("\n=== 4) TOP STOCKS (rate best-of-day PEAK >= 3x, 3d) ===")
sym = best.groupby("symbol").agg(days=("best_peak", "size"), rate3x_peak=("best_peak", lambda s: (s >= 3).mean()),
                                 rate2x_held=("best_held", lambda s: (s >= 2).mean()), med_peak=("best_peak", "median"))
sym = sym.merge(d.groupby("symbol").group.first(), on="symbol").sort_values("rate3x_peak", ascending=False)
print(sym.head(22).round(3).to_string())

print("\n=== 5) PEAK-DAY timing within the 3d window ===")
print("   all:", d.peak_day.value_counts(normalize=True).round(2).sort_index().to_dict())
print("   winners (peak>=3x):", d[d.peak_ratio >= 3].peak_day.value_counts(normalize=True).round(2).sort_index().to_dict())

print("\n=== 6) held day3 NET of 3% cost, by group (realistic, no timing) ===")
for g, sub in d.groupby("group"):
    r = sub.held_ratio.dropna() - 1 - 0.03
    print(f"   {g}: mean {r.mean()*100:+.2f}%  win {(r>0).mean():.3f}  (vs PEAK gross {sub.peak_ratio.mean():.2f}x)")
print("\nNOTE: per-option, both sides (oracle side not chosen). Realistic book still needs the side (direction)")
print("and either held-3 (no timing) or catching the day-2/3 peak. Compare held day3 here (3d) vs 5d study held 0.95x.")
