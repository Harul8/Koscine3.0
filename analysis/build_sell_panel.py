"""Cache the A/B ATM..+/-10% option panel to parquet so the EV / sell-history passes can iterate
without re-loading bhavcopy each time.
Usage: python analysis/build_sell_panel.py 2024-08-01 2026-08-04 panel.parquet
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import pandas as pd

ROOT = Path(r"C:\Users\rahul\Koscine 3.0")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "analysis"))
from options_bhavcopy import load_bhavcopy  # noqa: E402
from koscine3.largemove.mover_v2 import LOCK_V2  # noqa: E402

START, END, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
CE_LO, CE_HI, PE_LO, PE_HI = 0.99, 1.10, 0.90, 1.01

g2 = {s: g for g, syms in json.loads((LOCK_V2 / "universe_groups.json").read_text()).items() for s in syms}
# Panel is cached WIDE (static A/B groups UNION the liquid-universe allowlist) and the actual
# top-50-liquid gate is applied per-day downstream in gen_sell_signal_history.py -- so retuning
# the universe (core_n/daily_n/lookback) never costs a ~15min panel rebuild. Without the union
# the panel carried only the 65 static names and could not see 11 of today's top-50 liquid
# stocks (GRASIM, WIPRO, DLF, TATAPOWER, ...).
from koscine import liquid_universe as _lu  # noqa: E402
# Backtest universe is the fixed Nifty50 (no daily rotation, see liquid_universe.py's module
# docstring for why) -- union static groups too so nothing already relied upon disappears.
names = set(g2) | _lu.backtest_universe()
frames = []
dates = list(pd.bdate_range(START, END))
for k, d in enumerate(dates):
    bc = load_bhavcopy(d)
    if bc is None or len(bc) == 0:
        continue
    bc = bc[bc["symbol"].isin(names) & bc["opt_type"].isin(["CE", "PE"])
            & bc["strike"].notna() & bc["underlying"].notna() & (bc["underlying"] > 0)].copy()
    if bc.empty:
        continue
    m = bc["strike"] / bc["underlying"]
    keep = ((bc["opt_type"] == "CE") & m.between(CE_LO, CE_HI)) | ((bc["opt_type"] == "PE") & m.between(PE_LO, PE_HI))
    bc = bc[keep]
    frames.append(bc[["date", "symbol", "expiry", "strike", "opt_type", "open", "high", "low", "close", "oi", "vol", "underlying"]])
    if k % 60 == 0:
        print(f"  {k}/{len(dates)}", flush=True)

panel = pd.concat(frames, ignore_index=True)
panel["date"] = pd.to_datetime(panel["date"]); panel["expiry"] = pd.to_datetime(panel["expiry"])
panel["group"] = panel["symbol"].map(g2)
panel.to_parquet(OUT, index=False)
print(f"wrote {len(panel)} rows -> {OUT}")
