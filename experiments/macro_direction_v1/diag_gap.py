"""Gap-and-go test: does the overnight GAP (known at the 9:15 open) predict the open->close direction you can
still capture entering at the open? close[t]->close[t+1] = gap (close[t]->open[t+1]) + intraday (open->close).

  - corr(gap, intraday) > 0  => continuation (gap-and-go): enter at open in the gap's direction, ride it.
  - corr(gap, intraday) < 0  => fade: the day reverses the gap.
  - ~ 0                       => the gap does NOT clarify the tradeable direction (post-open is a coin flip).

Conditioned on gap size, because the claim is specifically about BIG overnight moves.

    set PYTHONPATH=src && python experiments/macro_direction_v1/diag_gap.py
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

g2 = {s: g for g, syms in json.loads((ROOT / "locks/prod_largemove_v2/universe_groups.json").read_text()).items() for s in syms}
m = load_market_data(columns=["date", "symbol", "open", "close", "atm_iv"])
m["symbol"] = m["symbol"].astype(str)
m = m[m.symbol.isin(g2)].copy()
m["date"] = pd.to_datetime(m["date"])
m = m.sort_values(["symbol", "date"]).reset_index(drop=True)
g = m.groupby("symbol", sort=False)
nopen, nclose = g["open"].shift(-1), g["close"].shift(-1)
m["gap"] = (nopen - m["close"]) / m["close"]          # close[t] -> open[t+1], observed at the open (entry)
m["intra"] = (nclose - nopen) / nopen                 # open[t+1] -> close[t+1], capturable entering at open
m["cc"] = (nclose - m["close"]) / m["close"]          # close -> close (gap + intra)

e = m[(m.close >= 100) & m.atm_iv.notna() & m.gap.notna() & m.intra.notna() & (m.date >= "2024-01-01")].copy()
print(f"=== Gap-and-go test | A/B universe | eval 2024-26 | n={len(e):,} ===\n")
print(f"corr(gap, intraday open->close) = {e.gap.corr(e.intra):+.4f}   <- ~0 means gap does NOT predict the day")
print(f"corr(gap, close->close)         = {e.gap.corr(e.cc):+.4f}   (high: the gap IS most of close-to-close)")
same = (np.sign(e.intra) == np.sign(e.gap)).mean()
print(f"P(intraday same sign as gap)    = {same:.3f}   (0.50 = coin flip / no continuation)")
print(f"mean intraday in gap direction  = {(e.intra * np.sign(e.gap)).mean()*1e4:+.1f} bps  (>0 continuation, <0 fade)\n")

print("by gap size (the claim is about BIG overnight moves):")
print(f"{'|gap| bucket':14s} {'n':>7s} {'mean gap%':>9s} {'mean intra in gap dir (bps)':>28s} {'P(same sign)':>13s}")
e["ag"] = e.gap.abs()
for lo, hi, lab in [(0, .005, '0-0.5%'), (.005, .01, '0.5-1%'), (.01, .02, '1-2%'), (.02, .05, '2-5%'), (.05, 9, '5%+')]:
    b = e[(e.ag >= lo) & (e.ag < hi)]
    if b.empty:
        continue
    md = (b.intra * np.sign(b.gap)).mean() * 1e4
    ss = (np.sign(b.intra) == np.sign(b.gap)).mean()
    print(f"{lab:14s} {len(b):7d} {b.gap.abs().mean()*100:9.2f} {md:28.1f} {ss:13.3f}")

print("\nReading: if 'mean intra in gap dir' ~0/negative and P(same sign) ~0.50 across buckets, the gap does not")
print("clarify the capturable direction -- the remaining day is a coin flip (or fades), so there's no open-entry edge.")
