"""Squeeze the real-premium trades (results/premium_ev_trades.csv) — no new bhavcopy loads.
Does the cheap-convexity edge concentrate (selectivity), survive costs, and improve after stripping pennies?
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
tr = pd.read_csv(HERE / "results" / "premium_ev_trades.csv", parse_dates=["date"])
pk = pd.read_csv(HERE / "results" / "picks.csv", parse_dates=["date"])
m = tr.merge(pk[["selector", "date", "symbol", "group", "pred", "atm_iv", "realized_cc"]],
             on=["selector", "date", "symbol"], how="left")
cc = m[m.selector.eq("cheap_convexity")].copy()
bl = m[m.selector.eq("atm_iv_baseline")].copy()


def line(name, d):
    r = d.ret
    net = {f"net@{c}%": round((r - c / 100).mean() * 100, 2) for c in (2, 3, 5)}
    print(f"{name:26s} n={len(d):5d} mean={r.mean()*100:+6.2f}% med={r.median()*100:+6.2f}% win={ (r>0).mean():.3f} "
          f"| {net}")


print("=== overall (gross + net of round-trip cost on premium) ===")
line("cheap_convexity", cc); line("atm_iv_baseline", bl)

print("\n=== cheap_convexity EV by predicted-surprise quintile (1=low,5=high) ===")
cc["q"] = pd.qcut(cc.pred, 5, labels=[1, 2, 3, 4, 5], duplicates="drop")
for q, d in cc.groupby("q", observed=True):
    line(f"  pred Q{q}", d)

print("\n=== cheap_convexity by entry-premium floor (strip pennies; premium in Rs) ===")
for fl in (0, 5, 10, 20, 40):
    line(f"  entry_prem>={fl}", cc[cc.entry_prem >= fl])

print("\n=== cheap_convexity by year (stability) ===")
cc["yr"] = cc.date.dt.year
for y, d in cc.groupby("yr"):
    line(f"  {y}", d)

print("\n=== cheap_convexity by group ===")
for g, d in cc.groupby("group"):
    line(f"  {g}", d)

print("\n=== BEST combo: high-pred (Q5) AND non-penny (>=20) ===")
line("cheap_convexity Q5 & >=20", cc[(cc.q == 5) & (cc.entry_prem >= 20)])
# tail contribution
r = cc.ret.sort_values()
print(f"\ntail: top-5% trades contribute {r.tail(int(len(r)*0.05)).sum()/ r.sum()*100:.0f}% of total gross return; "
      f"worst-50% mean {r.head(len(r)//2).mean()*100:+.1f}%")
