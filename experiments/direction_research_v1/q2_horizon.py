"""1-day vs 5-day: is MAGNITUDE (movement) easier to predict at h=1 or h=5? + direction at each (for context)."""
from __future__ import annotations
import sys, json
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parents[1] / "src"))
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from koscine3.data.sources import load_market_data

g2 = {s: grp for grp, syms in json.loads((HERE.parents[1] / "locks/prod_largemove_v2/universe_groups.json").read_text()).items() for s in syms}
m = load_market_data(columns=["date", "symbol", "open", "high", "low", "close", "atm_iv"])
m["symbol"] = m["symbol"].astype(str)
m = m[m.symbol.isin(g2)].sort_values(["symbol", "date"]).reset_index(drop=True)
m["group"] = m.symbol.map(g2)
g = m.groupby("symbol", sort=False)
entry = g["open"].shift(-1)
for h in (1, 5):
    H = pd.concat([g["high"].shift(-i) for i in range(1, h + 1)], axis=1).max(axis=1)
    L = pd.concat([g["low"].shift(-i) for i in range(1, h + 1)], axis=1).min(axis=1)
    m[f"mag{h}"] = np.maximum((H - entry) / entry, (entry - L) / entry)
    m[f"sclose{h}"] = (g["close"].shift(-h) - entry) / entry
    m[f"dir{h}"] = (m[f"sclose{h}"] > 0).astype(float)
m = m[(m.close >= 100) & m.atm_iv.notna() & (m.date.dt.year >= 2022)].dropna(subset=["mag1", "mag5"]).copy()

print(f"rows={len(m)}  (A+B universe, 2022-26)\n")
print("MAGNITUDE — does implied vol rank the big movers? (the mover-book question)")
for h, thr in [(1, 0.02), (1, 0.03), (5, 0.04), (5, 0.06)]:
    big = (m[f"mag{h}"] >= thr).astype(int)
    auc = roc_auc_score(big, m.atm_iv)
    print(f"  h={h}d  P(move>={int(thr*100)}%)={big.mean()*100:4.1f}%   atm_iv AUC for that move = {auc:.3f}")

print("\nMover precision: rank by atm_iv within group/day, are top-3 the actual top-3 movers?")
for h in (1, 5):
    e = m.copy()
    e["ar"] = e.groupby(["date", "group"])[f"mag{h}"].rank(ascending=False, method="first")
    e["ivr"] = e.groupby(["date", "group"])["atm_iv"].rank(ascending=False, method="first")
    top = e[e.ivr <= 3]
    e["daymax"] = e.groupby(["date", "group"])[f"mag{h}"].transform("max")
    print(f"  h={h}d  top-3 IV picks in actual top-3 movers: {(top.ar <= 3).mean()*100:.1f}%  | "
          f"avg move of top-IV pick: {e[e.ivr==1][f'mag{h}'].mean()*100:.1f}%  | "
          f"capture {(e[e.ivr<=1][f'mag{h}']/e[e.ivr<=1].daymax).mean()*100:.0f}%")

print(f"\nDIRECTION base rate P(up): h1={m.dir1.mean()*100:.1f}%  h5={m.dir5.mean()*100:.1f}%  "
      "(direction itself is ~coin flip at both horizons per prior research)")
