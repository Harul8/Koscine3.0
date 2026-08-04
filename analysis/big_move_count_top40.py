"""Top-40: how many times did a stock move >10% in a 5-day window, deduped.
A move = max(up, down) over the 5-day window >= 10%. Overlapping windows of the same move
are collapsed via a 5-trading-day cooldown PER STOCK (so 12-16/13-17/15-19 = 1 instance).
Reports per year: total distinct >10% moves, up vs down split, distinct stocks.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from koscine3.data.sources import load_market_data
from koscine3.data.universe import UniverseConfig, build_universe
from koscine3.outcomes.clean_move_contract import CleanMoveContract, compute_clean_move_outcomes

THRESH, COOLDOWN = 0.10, 5

market = load_market_data()
uni = build_universe(market, UniverseConfig(cutoff_date="2025-12-31", top_n=40))
syms = set(uni["symbol"].astype(str))
oc = compute_clean_move_outcomes(market, universe=uni, contract=CleanMoveContract())
oc = oc[oc.status.eq("evaluated")].copy(); oc["symbol"] = oc["symbol"].astype(str)

up = oc[oc.side.eq("long")][["date", "symbol", "ceiling"]].rename(columns={"ceiling": "up"})
dn = oc[oc.side.eq("short")][["date", "symbol", "ceiling"]].rename(columns={"ceiling": "down"})
m = up.merge(dn, on=["date", "symbol"])
m["mx"] = m[["up", "down"]].max(axis=1)
m["dir"] = np.where(m["up"] >= m["down"], "up", "down")

cal = np.array(sorted(market["date"].unique())); pos = {pd.Timestamp(d): i for i, d in enumerate(cal)}
ev = m[m["mx"] >= THRESH].copy()
ev["idx"] = ev["date"].map(lambda d: pos[pd.Timestamp(d)])
ev = ev.sort_values(["symbol", "idx"])
# greedy 5-day cooldown PER STOCK
keep, last = [], {}
for r in ev.itertuples(index=False):
    if r.idx - last.get(r.symbol, -10**9) > COOLDOWN:
        keep.append(r); last[r.symbol] = r.idx
ded = pd.DataFrame(keep)
ded["year"] = pd.to_datetime(ded["date"]).dt.year

# stocks actually trading each year (for context on look-ahead)
m["year"] = pd.to_datetime(m["date"]).dt.year
avail = m.groupby("year")["symbol"].nunique()

rows = []
for yr, g in ded.groupby("year"):
    rows.append({"year": yr, ">10% moves": len(g), "up": int((g["dir"] == "up").sum()),
                 "down": int((g["dir"] == "down").sum()), "stocks_with_move": g["symbol"].nunique(),
                 "stocks_trading": int(avail.get(yr, 0)),
                 "per_stock": round(len(g) / max(1, g["symbol"].nunique()), 1)})
res = pd.DataFrame(rows).sort_values("year")
pd.set_option("display.width", 200)
print("=== TOP-40: distinct >10% moves per year (5-day window, 5-day per-stock cooldown) ===")
print(res.to_string(index=False))
print(f"\nTOTAL distinct >10% moves (all years): {len(ded)}")
print(f"recent-year average (2024-2025): {res[res.year.isin([2024,2025])]['>10% moves'].mean():.0f}/yr "
      f"over ~{res[res.year.isin([2024,2025])]['stocks_trading'].mean():.0f} trading stocks")
