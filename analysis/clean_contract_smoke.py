"""Integration smoke: does the clean_move_contract module match the standalone base-rate scan?"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pandas as pd

from koscine3.data.sources import load_market_data
from koscine3.data.universe import UniverseConfig, build_universe
from koscine3.outcomes.clean_move_contract import CleanMoveContract, compute_clean_move_outcomes

market = load_market_data()
universe = build_universe(market, UniverseConfig(cutoff_date="2025-12-31", top_n=100))
out = compute_clean_move_outcomes(market, universe=universe, contract=CleanMoveContract())
ev = out[out["status"].eq("evaluated")].copy()
ev["year"] = pd.to_datetime(ev["date"]).dt.year

print(f"evaluated rows: {len(ev):,}  (top100 universe, both sides)")
print("\nclean rate + mean ceiling by side (expect long~0.35 clean, ceiling~5%):")
print(ev.groupby("side").agg(clean_rate=("clean", "mean"),
                             ceiling_mean=("ceiling", "mean"),
                             ceiling_med=("ceiling", "median"),
                             reaches5_rate=("reaches_big", "mean"),
                             med_days_to_peak=("days_to_peak", "median")).round(4).to_string())
print("\nlong clean rate by year (expect 0.29-0.39 every year):")
lng = ev[ev["side"].eq("long")]
print(lng.groupby("year")["clean"].mean().round(4).to_string())
