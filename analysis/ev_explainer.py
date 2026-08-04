"""Clarify: is IDEA in the +37% EV run? what drives EV? is it outlier-dependent?"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from koscine3.data.sources import load_market_data
from koscine3.data.universe import UniverseConfig, build_universe

tr = pd.read_excel(ROOT / "reports" / "option_strategy_top65_buckets_2026-06-11.xlsx", sheet_name="Trades")
print(f"+37% EV run: {len(tr)} trades, {tr['symbol'].nunique()} stocks")
print("IDEA present?", "IDEA" in set(tr["symbol"]))
print("\ntop 8 symbols in the EV run:")
print(tr["symbol"].value_counts().head(8).to_string())

# Is IDEA even in the top-65 universe?
market = load_market_data()
uni = build_universe(market, UniverseConfig(cutoff_date="2025-12-31", top_n=65))
print("\nIDEA in top-65 universe?", "IDEA" in set(uni["symbol"].astype(str)))

# EV = mean return per trade on the premium. Show the full shape.
m = tr["mult_peakclose"].dropna()           # realistic exit
ret = m - 1.0                                 # return on premium (option floored at 0 -> min -1)
print("\n===== EV anatomy (realistic peak-close exit) =====")
print(f"trades: {len(m)}")
print(f"WIN rate (made any profit, mult>1): {(m>1).mean()*100:.0f}%")
print(f"LOSE rate (mult<1):                 {(m<1).mean()*100:.0f}%")
print(f"total-loss-ish (mult<0.5):          {(m<0.5).mean()*100:.0f}%")
print(f"median mult: {m.median():.2f}x   mean mult: {m.mean():.2f}x  -> EV = {ret.mean()*100:+.0f}% per trade")
print(f"P>=2x: {(m>=2).mean()*100:.0f}%   P>=3x: {(m>=3).mean()*100:.0f}%   P>=5x: {(m>=5).mean()*100:.0f}%")

# How concentrated is the profit? contribution of the biggest winners.
gains = ret.sort_values(ascending=False).reset_index(drop=True)
total = gains.sum()
for pct in (0.05, 0.10, 0.20):
    k = max(1, int(len(gains) * pct))
    print(f"top {int(pct*100)}% of trades ({k}) contribute {gains.iloc[:k].sum()/total*100:.0f}% of total gain")
# EV if we drop the single best trade
print(f"\nEV excluding the single best trade: {(gains.iloc[1:].mean())*100:+.0f}%")
print(f"EV excluding the best 5 trades:     {(gains.iloc[5:].mean())*100:+.0f}%")
