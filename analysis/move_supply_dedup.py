"""Distinct move episodes per bucket per year, de-duplicated: once a (stock,side) move
>= threshold is counted, skip that stock-side for the next 5 trading days (same move).
Top-30 threshold 5%, Next-35 threshold 10%, both directions.
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

THRESH = {"A_top30": 0.05, "B_next35": 0.10}
COOLDOWN = 5

market = load_market_data()
uni = build_universe(market, UniverseConfig(cutoff_date="2025-12-31", top_n=65))
rk = uni.set_index(uni["symbol"].astype(str))["rank"]
bucket_of = {s: ("A_top30" if r <= 30 else "B_next35") for s, r in rk.items()}
cal = np.array(sorted(market["date"].unique()))
pos = {pd.Timestamp(d): i for i, d in enumerate(cal)}

oc = compute_clean_move_outcomes(market, universe=uni, contract=CleanMoveContract())
oc = oc[oc["status"].eq("evaluated")].copy()
oc["symbol"] = oc["symbol"].astype(str)
oc["bucket"] = oc["symbol"].map(bucket_of)
oc["thr"] = oc["bucket"].map(THRESH)
qual = oc[oc["ceiling"] >= oc["thr"]].copy()
qual["idx"] = qual["date"].map(lambda d: pos[pd.Timestamp(d)])
qual = qual.sort_values(["symbol", "side", "idx"])

# Greedy dedup per (stock, side): keep a move only if >5 trading days since the last kept one.
keep = []
last = {}
for row in qual.itertuples(index=False):
    key = (row.symbol, row.side)
    if row.idx - last.get(key, -10**9) > COOLDOWN:
        keep.append(row)
        last[key] = row.idx
ded = pd.DataFrame(keep)
ded["year"] = pd.to_datetime(ded["date"]).dt.year
ded = ded[ded["year"].isin([2024, 2025, 2026])]

rows = []
for (year, bucket), g in ded.groupby(["year", "bucket"]):
    days = oc[(pd.to_datetime(oc["date"]).dt.year == year)]["date"].nunique()
    up = (g["side"] == "long").sum()
    dn = (g["side"] == "short").sum()
    rows.append({"year": year, "bucket": bucket, "thresh": f"{int(THRESH[bucket]*100)}%",
                 "UP": int(up), "DOWN": int(dn), "DISTINCT_MOVES": len(g),
                 "per_day": round(len(g)/days, 1), "stocks": g["symbol"].nunique(),
                 "of": 30 if bucket == "A_top30" else 35})
res = pd.DataFrame(rows).sort_values(["bucket", "year"])
pd.set_option("display.width", 200)
print("\n===== DISTINCT MOVE EPISODES (5-day dedup per stock+side) =====")
print(res.to_string(index=False))
print("\n2026 = partial (Jan-May). Each move counted once; re-entries into the same move suppressed for 5 trading days.")
