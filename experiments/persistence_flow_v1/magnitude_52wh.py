"""Does 52-week-high distance ADD to the atm_iv mover-precision selector? (test years 2024-26)"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))

import numpy as np
import pandas as pd
from koscine3.data.sources import load_market_data

pd.set_option("display.width", 200)
groups = json.loads((HERE / "universe_groups.json").read_text())
g2 = {s: g for g, v in groups.items() for s in v}
UNIV = set(g2)

m = load_market_data(columns=["date", "symbol", "open", "high", "low", "close", "atm_iv"])
m["symbol"] = m["symbol"].astype(str)
m = m[m.symbol.isin(UNIV)].sort_values(["symbol", "date"]).reset_index(drop=True)
gb = m.groupby("symbol", sort=False)
entry = gb["open"].shift(-1)
H = pd.concat([gb["high"].shift(-i) for i in range(1, 6)], axis=1).max(axis=1)
L = pd.concat([gb["low"].shift(-i) for i in range(1, 6)], axis=1).min(axis=1)
m["move_mag"] = np.maximum((H - entry) / entry, (entry - L) / entry)
hi252 = gb["close"].transform(lambda s: s.rolling(252, min_periods=120).max())
m["dist_52wh"] = m.close / hi252 - 1.0     # nearer 0 = nearer high
m["group"] = m.symbol.map(g2)
m = m[(m.close >= 100) & m.atm_iv.notna() & m.move_mag.notna() & m.date.dt.year.isin([2024, 2025, 2026])].copy()

G = m.groupby(["date", "group"])
m["actual_rank"] = G["move_mag"].rank(ascending=False, method="first")
m["day_max"] = G["move_mag"].transform("max")
m["iv_r"] = G["atm_iv"].rank(pct=True)
m["wh_r"] = G["dist_52wh"].rank(pct=True)               # higher = nearer 52wk high
m["blend"] = m.iv_r + m.wh_r
m["rank_iv"] = G["atm_iv"].rank(ascending=False, method="first")
m["rank_blend"] = G["blend"].rank(ascending=False, method="first")


def prec(d):
    return dict(trades=len(d), t3=round((d.actual_rank <= 3).mean()*100, 1),
                t5=round((d.actual_rank <= 5).mean()*100, 1),
                cap=round((d.move_mag/d.day_max).mean()*100, 1))


print("=" * 70)
print("Does 52wk-high ADD to atm_iv for mover precision? (top-1 pick/group/day)")
print("=" * 70)
rows = []
for name, rcol in [("atm_iv alone", "rank_iv"), ("IV+52wh blend", "rank_blend")]:
    rows.append({"ranker": name, **prec(m[m[rcol] <= 1])})
print(pd.DataFrame(rows).to_string(index=False))

print("\n" + "=" * 70)
print("Selectivity combo: IV top-1 pick, split by 52wk-high proximity")
print("=" * 70)
p1 = m[m.rank_iv <= 1].copy()
p1["wh_bucket"] = pd.qcut(p1.dist_52wh.rank(method="first"), 3, labels=["far", "mid", "near_high"])
rows = [{"52wh": b, **prec(d)} for b, d in p1.groupby("wh_bucket", observed=True)]
print(pd.DataFrame(rows).to_string(index=False))

print("\n" + "=" * 70)
print("Best selective rule: IV top-20% AND near 52wk-high")
print("=" * 70)
p1["iv_hi"] = p1.atm_iv >= p1.atm_iv.quantile(0.80)
combo = p1[p1.iv_hi & (p1.dist_52wh >= p1.dist_52wh.quantile(0.50))]
print(f"IV-top20% & nearer-half-to-high: {prec(combo)}  (~{round(len(combo)/3)}/yr)")
