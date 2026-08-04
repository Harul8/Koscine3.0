"""Analyze the option-gain trades: where do the large gains come from (strike, IV, DTE, stock, timing)?

    python experiments/option_gain_study_v1/option_gain_analyze.py
Reads results/option_gain_trades.csv (from option_gain_study.py). Cheap; no bhavcopy.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
d = pd.read_csv(HERE / "results" / "option_gain_trades.csv", parse_dates=["entry_date"])
d["abs_move"] = d.stock_move.abs() * 100
ORDER = ["ATM", "ATM+1%", "ATM+2%", "ATM+3%", "ATM+4%", "ATM+5%", "ATM+7%", "ATM+10%"]
d["strike_label"] = pd.Categorical(d.strike_label, ORDER, ordered=True)
print(f"trades {len(d):,} | {d.entry_date.min().date()}..{d.entry_date.max().date()} | symbols {d.symbol.nunique()} | entry_prem>=Rs2\n")


def block(title): print("\n" + "=" * 3, title)


block("1) GAIN BY STRIKE (peak=high_ratio, held=close_ratio)")
g = d.groupby("strike_label", observed=True)
t = pd.DataFrame({
    "n": g.size(),
    "med_peak": g.high_ratio.median().round(2), "mean_peak": g.high_ratio.mean().round(2),
    "P(peak>=2x)": g.high_ratio.apply(lambda s: (s >= 2).mean()).round(3),
    "P(peak>=3x)": g.high_ratio.apply(lambda s: (s >= 3).mean()).round(3),
    "P(peak>=5x)": g.high_ratio.apply(lambda s: (s >= 5).mean()).round(3),
    "med_held": g.close_ratio.median().round(2), "mean_held": g.close_ratio.mean().round(2),
    "P(total loss<0.5x)": g.close_ratio.apply(lambda s: (s < 0.5).mean()).round(3),
})
print(t.to_string())

block("2) BEST OPTION PER STOCK-DAY (any strike/side = convexity available)")
best = d.groupby(["symbol", "entry_date"]).agg(best_peak=("high_ratio", "max"), best_held=("close_ratio", "max"),
                                               abs_move=("abs_move", "first")).reset_index()
print(f"   stock-days {len(best):,} | median best peak {best.best_peak.median():.2f}x | "
      f"P(some opt >=2x) {(best.best_peak>=2).mean():.3f} | >=3x {(best.best_peak>=3).mean():.3f} | "
      f">=5x {(best.best_peak>=5).mean():.3f}")

block("3) TOP STOCKS likely to give large gains (rate best-option-of-day peak >=3x)")
sym = best.groupby("symbol").agg(days=("best_peak", "size"), rate_3x=("best_peak", lambda s: (s >= 3).mean()),
                                 rate_5x=("best_peak", lambda s: (s >= 5).mean()), med_peak=("best_peak", "median"),
                                 mean_abs_move=("abs_move", "mean"))
sym = sym.merge(d.groupby("symbol").group.first(), on="symbol")
sym = sym.sort_values("rate_3x", ascending=False)
print(sym.head(25).round(3).to_string())
print("\n   by group (mean rate >=3x):", d.merge(best, on=["symbol", "entry_date"]).groupby("group").apply(
    lambda x: round((x.best_peak >= 3).mean(), 3), include_groups=False).to_dict())

block("4) WHAT DISTINGUISHES BIG WINNERS (peak>=3x) vs all")
w = d[d.high_ratio >= 3]
print(f"   {'metric':16s}{'winners(>=3x)':>16s}{'all':>12s}")
for col, lab in [("atm_iv", "entry atm_iv"), ("abs_move", "abs stock move%"), ("dte", "days-to-expiry"),
                 ("entry_open", "entry prem Rs"), ("peak_day", "peak day(0-4)")]:
    print(f"   {lab:16s}{w[col].mean():>16.2f}{d[col].mean():>12.2f}")
print("   winner strike mix:", (w.strike_label.value_counts(normalize=True).round(2).reindex(ORDER).dropna()).to_dict())
print("   winner peak-day mix:", w.peak_day.value_counts(normalize=True).round(2).sort_index().to_dict())

block("5) IV REGIME -> gain (entry atm_iv terciles)")
d["iv_bucket"] = pd.qcut(d.atm_iv, 3, labels=["low_iv", "mid_iv", "high_iv"])
gi = d.groupby("iv_bucket", observed=True)
print(pd.DataFrame({"n": gi.size(), "med_atm_iv": gi.atm_iv.median().round(3), "med_peak": gi.high_ratio.median().round(2),
                    "P(peak>=3x)": gi.high_ratio.apply(lambda s: (s >= 3).mean()).round(3),
                    "mean_abs_move%": gi.abs_move.mean().round(2)}).to_string())

block("6) DTE -> gain")
d["dte_bucket"] = pd.cut(d.dte, [0, 10, 20, 40], labels=["<=10d", "11-20d", "21-40d"])
gd = d.groupby("dte_bucket", observed=True)
print(pd.DataFrame({"n": gd.size(), "med_peak": gd.high_ratio.median().round(2),
                    "P(peak>=3x)": gd.high_ratio.apply(lambda s: (s >= 3).mean()).round(3)}).to_string())

block("7) LEVERAGE: option peak multiple per 1% favorable stock move (favorable side only)")
fav = d[((d.side == "CALL") & (d.stock_move > 0)) | ((d.side == "PUT") & (d.stock_move < 0))].copy()
fav = fav[fav.abs_move >= 1]
fav["lev"] = (fav.high_ratio - 1) / fav.abs_move
gl = fav.groupby("strike_label", observed=True).lev.median().round(2)
print("   median (peak%-1)/move% by strike:", gl.reindex(ORDER).dropna().to_dict())
print(f"   corr(abs stock move, best peak per stock-day) = {best.abs_move.corr(best.best_peak):.3f}")
