"""Raw opportunity supply: how many favourable >=threshold moves per year, by bucket.
Top-30 bucket threshold = 5%, Next-35 bucket threshold = 10%. Both directions (call/put).
A 'move' = a (stock, signal-date, side) whose 5-day forward favourable peak (ceiling) >= threshold.
"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from koscine3.data.sources import load_market_data
from koscine3.data.universe import UniverseConfig, build_universe
from koscine3.outcomes.clean_move_contract import CleanMoveContract, compute_clean_move_outcomes

THRESH = {"A_top30": 0.05, "B_next35": 0.10}

market = load_market_data()
uni = build_universe(market, UniverseConfig(cutoff_date="2025-12-31", top_n=65))
rk = uni.set_index(uni["symbol"].astype(str))["rank"]
bucket_of = {s: ("A_top30" if r <= 30 else "B_next35") for s, r in rk.items()}

oc = compute_clean_move_outcomes(market, universe=uni, contract=CleanMoveContract())
oc = oc[oc["status"].eq("evaluated")].copy()
oc["symbol"] = oc["symbol"].astype(str)
oc["bucket"] = oc["symbol"].map(bucket_of)
oc["year"] = pd.to_datetime(oc["date"]).dt.year
oc = oc[oc["year"].isin([2024, 2025, 2026])]

rows = []
for (year, bucket), g in oc.groupby(["year", "bucket"]):
    thr = THRESH[bucket]
    days = g["date"].nunique()
    up = g[g["side"].eq("long") & (g["ceiling"] >= thr)]
    dn = g[g["side"].eq("short") & (g["ceiling"] >= thr)]
    tot = len(up) + len(dn)
    rows.append({
        "year": year, "bucket": bucket, "thresh": f"{int(thr*100)}%", "trading_days": days,
        "UP_moves": len(up), "DOWN_moves": len(dn), "TOTAL": tot,
        "per_day": round(tot / days, 1),
        "distinct_stocks": pd.concat([up, dn])["symbol"].nunique(),
        "of_bucket": f"{pd.concat([up, dn])['symbol'].nunique()}/{30 if bucket=='A_top30' else 35}",
    })
res = pd.DataFrame(rows).sort_values(["bucket", "year"])
pd.set_option("display.width", 200)
print("\n===== FAVOURABLE-MOVE SUPPLY BY BUCKET & YEAR (5-day window, both directions) =====")
print(res.to_string(index=False))
print("\nNote: 2026 is partial (Jan-May). A 'move' is a signal-date entry whose forward 5d window")
print("reaches the threshold; consecutive entries into the same move are counted separately (each is a tradeable entry).")
