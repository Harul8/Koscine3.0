"""Leak hunt: a legit backward-looking feature has ~0 correlation with the NEXT-DAY return (direction is a
coin flip). Any feature strongly correlated with the future is a forward-looking (target/label) leak.

    set PYTHONPATH=src && python experiments/macro_direction_v1/diag_leak.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from koscine3.data.sources import load_market_data  # noqa: E402

LOCK_V2 = ROOT / "locks" / "prod_largemove_v2"
g2 = {s: g for g, syms in json.loads((LOCK_V2 / "universe_groups.json").read_text()).items() for s in syms}

m = load_market_data()
m["symbol"] = m["symbol"].astype(str)
m = m[m.symbol.isin(g2)].copy()
m["date"] = pd.to_datetime(m["date"])
m = m.sort_values(["symbol", "date"]).reset_index(drop=True)
g = m.groupby("symbol", sort=False)
fwd1 = (g["close"].shift(-1) - m["close"]) / m["close"]   # next-day signed return
fwd5 = (g["close"].shift(-5) - m["close"]) / m["close"]   # 5-day signed return

num = [c for c in m.columns if pd.api.types.is_numeric_dtype(m[c]) and c not in
       ("open", "high", "low", "close", "volume")]
rows = []
for c in num:
    s = m[c].replace([np.inf, -np.inf], np.nan)
    if s.notna().sum() < 5000:
        continue
    rows.append((c, abs(s.corr(fwd1)), abs(s.corr(fwd5))))
d = pd.DataFrame(rows, columns=["feature", "abs_corr_next1d", "abs_corr_fwd5d"]).sort_values(
    "abs_corr_next1d", ascending=False)

print(f"features scanned: {len(d)}")
print("\n=== TOP 30 by |corr with NEXT-DAY return| (anything >> 0 is a forward-looking leak) ===")
print(d.head(30).to_string(index=False))
leaky = d[(d.abs_corr_next1d > 0.05) | (d.abs_corr_fwd5d > 0.10)]["feature"].tolist()
print(f"\nLEAKY (|corr_next1d|>0.05 or |corr_fwd5d|>0.10): {len(leaky)}")
print(leaky)
print(f"\nSAFE backward features: {len(d) - len(leaky)}")
