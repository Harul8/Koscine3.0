"""Precision vs VOLUME frontier for the IV mover-picker.

Direction is a coin flip -> need MANY trades so it averages out. So the key metric isn't just rank;
it's how often a pick makes a tradeable big move EITHER way (what a straddle / exit-at-peak monetizes):
  pct_ge6 / pct_ge8 = % of picks whose 5-day |move| >= 6% / 8%.
Menu of operating points from ~100 to ~1100 trades/yr.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np
import pandas as pd

pd.set_option("display.width", 220)
ev = pd.read_parquet(HERE / "magnitude_oos.parquet")
ev["date"] = pd.to_datetime(ev["date"])
ev["year"] = ev.date.dt.year
NY = ev.year.nunique()
G = ev.groupby(["date", "group"])
ev["iv_rank"] = G["atm_iv"].rank(ascending=False, method="first")
ev["actual_rank"] = G["move_mag"].rank(ascending=False, method="first")
ev["day_max"] = G["move_mag"].transform("max")
ev["iv_gap"] = ev["atm_iv"] - G["atm_iv"].transform("median")


def met(name, d):
    return {"operating point": name, "trades/yr": round(len(d) / NY),
            "avg_move%": round(d.move_mag.mean() * 100, 1),
            "ge6%": round((d.move_mag >= 0.06).mean() * 100, 1),
            "ge8%": round((d.move_mag >= 0.08).mean() * 100, 1),
            "in_top5%": round((d.actual_rank <= 5).mean() * 100, 1),
            "capture%": round((d.move_mag / d.day_max).mean() * 100, 1)}


rows = []
for K in (1, 2, 3):
    rows.append(met(f"top-{K}/group, every day", ev[ev.iv_rank <= K]))
for K, f in [(1, 0.5), (1, 0.3), (2, 0.6), (2, 0.4), (3, 0.5)]:
    d = ev[ev.iv_rank <= K]
    d = d[d.iv_gap >= d.iv_gap.quantile(1 - f)]
    rows.append(met(f"top-{K}/group, IV-gap top {int(f*100)}%", d))

out = pd.DataFrame(rows).sort_values("trades/yr").reset_index(drop=True)
print("=" * 100)
print("IV mover-picker — PRECISION vs VOLUME frontier  (universe base rate: |move|>=6% ~ 38%, >=8% ~ 22%)")
print("=" * 100)
print(out.to_string(index=False))
print("\navg_move/ge6/ge8 = the pick's 5-day |move| (direction-agnostic, what a straddle/peak-exit captures).")
print("in_top5 = pick is among day's 5 biggest movers in group. capture = % of day's best move captured.")

# universe base rates for reference
print(f"\nreference: all eligible A/B picks |move|>=6% = {(ev.move_mag>=0.06).mean()*100:.1f}%, "
      f">=8% = {(ev.move_mag>=0.08).mean()*100:.1f}%, avg = {ev.move_mag.mean()*100:.1f}%")
