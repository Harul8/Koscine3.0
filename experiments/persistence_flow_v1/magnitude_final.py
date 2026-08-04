"""Final: per-year robustness of the selective IV-rank mover-picker + recommended operating points.

Selector = rank by atm_iv (best single signal); be selective on conviction (IV-gap vs peers / IV level
/ earnings). Confirm the precision lift is stable across 2024/2025/2026, not one lucky year.
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
ev["year"] = ev.date.dt.year
g = ev.groupby(["date", "group"])
ev["iv_rank"] = g["atm_iv"].rank(ascending=False, method="first")
ev["actual_rank"] = g["move_mag"].rank(ascending=False, method="first")
ev["day_max"] = g["move_mag"].transform("max")
ev["iv_gap"] = ev["atm_iv"] - g["atm_iv"].transform("median")
pick1 = ev[ev.iv_rank <= 1].copy()
YEARS = sorted(pick1.year.unique())
nyears = len(YEARS)


def prec(d):
    if not len(d):
        return dict(trades=0, t3=0.0, t5=0.0, cap=0.0)
    return dict(trades=len(d),
                t3=round((d.actual_rank <= 3).mean() * 100, 1),
                t5=round((d.actual_rank <= 5).mean() * 100, 1),
                cap=round((d.move_mag / d.day_max).mean() * 100, 1))


def line(label, d):
    row = {"rule": label, "trades/yr": round(len(d) / nyears), **prec(d)}
    for y in YEARS:
        row[f"t3_{y}"] = prec(d[d.year == y])["t3"]
    return row


rows = [line("ALL (2/day baseline)", pick1)]
for f in (0.30, 0.20, 0.10):
    bar = pick1["iv_gap"].quantile(1 - f)
    rows.append(line(f"IV-gap top {int(f*100)}%", pick1[pick1.iv_gap >= bar]))
for f in (0.20, 0.10):
    bar = pick1["atm_iv"].quantile(1 - f)
    rows.append(line(f"atm_iv top {int(f*100)}%", pick1[pick1.atm_iv >= bar]))
if "earnings_within_5d" in pick1:
    earn = pick1[pick1.earnings_within_5d.fillna(0) > 0]
    rows.append(line("earnings-window only", earn))
    bar = pick1["iv_gap"].quantile(0.70)
    combo = pick1[(pick1.iv_gap >= bar) | (pick1.earnings_within_5d.fillna(0) > 0)]
    rows.append(line("IV-gap top30% OR earnings", combo))

out = pd.DataFrame(rows)
cols = ["rule", "trades/yr", "t3", "t5", "cap"] + [f"t3_{y}" for y in YEARS]
print("=" * 96)
print("SELECTIVE IV-rank mover-picker — precision (in_top3 / in_top5 / capture) + per-year in_top3")
print("=" * 96)
print(out[cols].to_string(index=False))
print("\nt3=in top-3 mover, t5=in top-5, cap=% of day's best move captured. Random: t3=11%, t5=18%.")
print("Baseline 'all' = take the IV top pick in each group every day.")
