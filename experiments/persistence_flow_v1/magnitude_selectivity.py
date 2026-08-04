"""Improve large-mover precision via SELECTIVITY (the only lever left — no model beats atm_iv).

Idea: the IV top pick isn't equally trustworthy every day. It should be more precise when IV is
SPIKING (fresh catalyst), when the pick clearly stands out from peers (high IV-gap), or around
earnings. Bucket the IV top-1 pick by these conviction signals, then build a precision-vs-coverage
curve: keep only the highest-conviction days and see how high precision climbs.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np
import pandas as pd

pd.set_option("display.width", 200)
ev = pd.read_parquet(HERE / "magnitude_oos.parquet")
ev["date"] = pd.to_datetime(ev["date"])

g = ev.groupby(["date", "group"])
ev["iv_rank"] = g["atm_iv"].rank(ascending=False, method="first")
ev["actual_rank"] = g["move_mag"].rank(ascending=False, method="first")
ev["day_max"] = g["move_mag"].transform("max")
ev["iv_gap"] = ev["atm_iv"] - g["atm_iv"].transform("median")     # standout vs peers
n_days = ev[["date", "group"]].drop_duplicates().shape[0]
years = ev.date.dt.year.nunique()


def prec(d):
    return dict(trades=len(d),
                in_top3=round((d.actual_rank <= 3).mean() * 100, 1),
                in_top5=round((d.actual_rank <= 5).mean() * 100, 1),
                capture=round((d.move_mag / d.day_max).mean() * 100, 1),
                avg_move=round(d.move_mag.mean() * 100, 1))


pick1 = ev[ev.iv_rank <= 1].copy()
print(f"IV top-1 pick / group / day — baseline: {prec(pick1)}\n")

print("=" * 76)
print("PRECISION by conviction bucket (IV top-1 pick, quartiles)")
print("=" * 76)
for sig in ["atm_iv", "atm_iv_ratio_20", "atm_iv_chg_5", "iv_gap"]:
    pick1["q"] = pd.qcut(pick1[sig].rank(method="first"), 4, labels=["q1", "q2", "q3", "q4"])
    rows = [{"bucket": b, **prec(d)} for b, d in pick1.groupby("q", observed=True)]
    print(f"\n[{sig}]")
    print(pd.DataFrame(rows)[["bucket", "trades", "in_top3", "in_top5", "capture", "avg_move"]].to_string(index=False))

if "earnings_within_5d" in pick1:
    print("\n[earnings_within_5d]")
    rows = [{"earn": int(b), **prec(d)} for b, d in pick1.groupby(pick1.earnings_within_5d.fillna(0) > 0)]
    print(pd.DataFrame(rows)[["earn", "trades", "in_top3", "in_top5", "capture", "avg_move"]].to_string(index=False))

print("\n" + "=" * 76)
print("PRECISION vs COVERAGE — keep only highest-conviction IV top-1 picks")
print("=" * 76)
for sig in ["atm_iv_ratio_20", "iv_gap", "atm_iv"]:
    print(f"\nrank days by [{sig}], keep top fraction:")
    rows = []
    for f in (1.0, 0.5, 0.3, 0.2, 0.1, 0.05):
        bar = pick1[sig].quantile(1 - f)
        d = pick1[pick1[sig] >= bar]
        rows.append({"keep": f"{int(f*100)}%", **prec(d), "trades_per_yr": round(len(d) / years)})
    print(pd.DataFrame(rows)[["keep", "trades", "trades_per_yr", "in_top3", "in_top5", "capture", "avg_move"]].to_string(index=False))
