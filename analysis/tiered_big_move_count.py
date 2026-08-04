"""Tiered big-move supply (2024-2026), top-50, deduped (5-day window, 5-day per-stock cooldown):
  tier A rank  1-10  : >5%
  tier B rank 11-30  : >10%
  tier C rank 31-50  : >15%
A move = max(up, down) over the 5-day window >= the tier threshold.
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

COOLDOWN = 5
YEARS = [2024, 2025, 2026]


def tier_of(rank):
    if rank <= 20: return ("A_top20", 0.05)
    return ("B_21-50", 0.10)


market = load_market_data()
uni = build_universe(market, UniverseConfig(cutoff_date="2025-12-31", top_n=50))
rk = uni.set_index(uni["symbol"].astype(str))["rank"]
tinfo = {s: tier_of(r) for s, r in rk.items()}
tier_map = {s: t for s, (t, _) in tinfo.items()}
thr_map = {s: th for s, (_, th) in tinfo.items()}

oc = compute_clean_move_outcomes(market, universe=uni, contract=CleanMoveContract())
oc = oc[oc.status.eq("evaluated")].copy(); oc["symbol"] = oc["symbol"].astype(str)
up = oc[oc.side.eq("long")][["date", "symbol", "ceiling"]].rename(columns={"ceiling": "up"})
dn = oc[oc.side.eq("short")][["date", "symbol", "ceiling"]].rename(columns={"ceiling": "down"})
m = up.merge(dn, on=["date", "symbol"])
m["mx"] = m[["up", "down"]].max(axis=1)
m["dir"] = np.where(m["up"] >= m["down"], "up", "down")
m["tier"] = m["symbol"].map(tier_map); m["thresh"] = m["symbol"].map(thr_map)
m["year"] = pd.to_datetime(m["date"]).dt.year

cal = np.array(sorted(market["date"].unique())); pos = {pd.Timestamp(d): i for i, d in enumerate(cal)}
ev = m[(m["mx"] >= m["thresh"]) & (m["year"].isin(YEARS))].copy()
ev["idx"] = ev["date"].map(lambda d: pos[pd.Timestamp(d)])
ev = ev.sort_values(["symbol", "idx"])
keep, last = [], {}
for r in ev.itertuples(index=False):
    if r.idx - last.get(r.symbol, -10**9) > COOLDOWN:
        keep.append(r); last[r.symbol] = r.idx
ded = pd.DataFrame(keep)

tdays = m[m.year.isin(YEARS)].groupby("year")["date"].nunique()
tier_stocks = {"A_top20": 20, "B_21-50": 30}
rows = []
for (tier, yr), g in ded.groupby(["tier", "year"]):
    rows.append({"tier": tier, "thresh": {"A_top20": ">5%", "B_21-50": ">10%"}[tier],
                 "year": yr, "moves": len(g), "up": int((g.dir == "up").sum()), "down": int((g.dir == "down").sum()),
                 "stocks": g.symbol.nunique(), "of": tier_stocks[tier],
                 "per_day": round(len(g) / tdays[yr], 2), "per_stock": round(len(g) / tier_stocks[tier], 1)})
res = pd.DataFrame(rows).sort_values(["tier", "year"])
pd.set_option("display.width", 200)
print("=== TIERED >threshold moves, top-50, 2024-2026 (5-day window, 5-day per-stock cooldown) ===")
print(res.to_string(index=False))
print("\nper-year totals (all tiers combined):")
print(ded.groupby(ded.date.map(lambda d: pd.Timestamp(d).year)).size().to_string())
