"""Is the US-overnight macro lift TRADEABLE, or is it the un-capturable open gap?

close[d]->close[d+1]  =  gap (close[d]->open[d+1])  +  intraday (open[d+1]->close[d+1])
The US overnight (US close dated d, prints ~02:00 IST d+1) can drive the d+1 OPEN gap. But:
  - at India EOD d (when our book emits the signal) that US close hasn't happened -> use US_LAG=1 (already known)
  - even at d+1 pre-open, the gap is already in the open price -> only intraday (open->close) is capturable
If corr(overnight, gap) is large but corr(overnight, intraday) ~ 0, the edge is real but NOT tradeable.

    set PYTHONPATH=src && python experiments/macro_direction_v1/diag_macro.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
from koscine3.data.sources import load_market_data  # noqa: E402

g2 = {s: g for g, syms in json.loads((ROOT / "locks/prod_largemove_v2/universe_groups.json").read_text()).items() for s in syms}
m = load_market_data(columns=["date", "symbol", "open", "high", "low", "close", "atm_iv"])
m["symbol"] = m["symbol"].astype(str)
m = m[m.symbol.isin(g2)].copy()
m["date"] = pd.to_datetime(m["date"])
m = m.sort_values(["symbol", "date"]).reset_index(drop=True)
g = m.groupby("symbol", sort=False)
nopen, nclose = g["open"].shift(-1), g["close"].shift(-1)
m["fwd_cc"] = (nclose - m["close"]) / m["close"]      # close->close (what the experiment targets)
m["fwd_gap"] = (nopen - m["close"]) / m["close"]      # close->next open (overnight gap; NOT tradeable)
m["fwd_oc"] = (nclose - nopen) / nopen                # next open->next close (tradeable from the open)

raw = pd.read_parquet(HERE / "data/macro_raw.parquet")
raw["date"] = pd.to_datetime(raw["date"]).dt.normalize()
raw = raw.set_index("date").sort_index()
nse = pd.DatetimeIndex(m["date"].unique()).sort_values()


def overnight(col: str, lag: int) -> pd.Series:
    s = raw[col].dropna()
    if lag:
        s = s.shift(lag)
    full = s.reindex(s.index.union(nse)).sort_index().ffill(limit=2)
    return full.reindex(nse).pct_change(1, fill_method=None)


ov = {}
for col, pre in [("spx", "spx"), ("dji", "dji"), ("ixic", "ndx")]:
    ov[f"{pre}_overnight_LAG0"] = overnight(col, 0)   # used by the experiment (act at d+1 open)
    ov[f"{pre}_prevnight_LAG1"] = overnight(col, 1)   # known at India EOD d (act at EOD d)
ovdf = pd.DataFrame(ov); ovdf["date"] = nse
mm = m.merge(ovdf, on="date", how="left")
e = mm[mm.close.ge(100) & mm.atm_iv.notna() & mm.fwd_cc.notna() & (mm.date >= "2024-01-01")]

print(f"eval rows (2024-26): {len(e):,}\n")
print(f"{'US feature':22s} {'-> close->close':>15s} {'-> overnight gap':>16s} {'-> open->close':>15s}")
for k in ov:
    print(f"{k:22s} {e[k].corr(e.fwd_cc):15.3f} {e[k].corr(e.fwd_gap):16.3f} {e[k].corr(e.fwd_oc):15.3f}")

print("\nReading: LAG0 strong on close->close & gap but ~0 on open->close  => edge lives in the un-capturable")
print("gap. LAG1 (what's actually known at our EOD-d signal time) ~0 everywhere => nothing tradeable at EOD.")
