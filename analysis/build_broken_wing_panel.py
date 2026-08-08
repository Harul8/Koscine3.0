"""Cache the option panel for the Broken-Wing Condor strategy: UNION of the static A/B
universe_groups.json (65 symbols) and each day's DYNAMIC top-50-by-open-interest-in-lots
stocks (fut_oi / that month's lot size -- NOT raw share OI, which is dominated by cheap
high-share-count names like IDEA/YESBANK/SUZLON; OI-in-lots correctly surfaces mega-caps
since NSE sizes lots to normalize notional value per contract).

Why the union, not just top-50-by-OI alone: tested top-50-only and it actually REDUCED both
signal frequency (95/yr vs the static universe's 150/yr) and mega-cap representation stayed
thin (16-20% share) -- the entry-ror gate structurally favors higher-IV "mover" names over
stable mega-caps regardless of which universe they're drawn from, so swapping the universe
alone doesn't fix under-representation, it just shrinks the pool. The union keeps the B_turn35
"movers" that drive frequency AND adds the broader liquid/OI-heavy universe for diversity;
combined with a stratified entry gate (see gen_broken_wing_signal_history.py) mega-cap share
rises to 13-28%/tier without sacrificing the 150-200/yr frequency target.

KNOWN GAP: does NOT exclude F&O-ban-listed stocks (NSE's daily MWPL-based ban list) -- no data
source for this exists anywhere in the codebase (checked). Cross-check live picks against
NSE's published ban list before trading until this is added.

Uses a wider put-side moneyness band than the symmetric condor's (long put targets ~90% of
spot for the widest tier).
Usage: python analysis/build_broken_wing_panel.py 2024-08-01 2026-08-05 bw_panel.parquet
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import pandas as pd

ROOT = Path(r"C:\Users\rahul\Koscine 3.0")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "analysis"))
from options_bhavcopy import load_bhavcopy  # noqa: E402
from koscine3.largemove.mover_v2 import LOCK_V2  # noqa: E402
from pipeline.config import SILVER_TABLES  # noqa: E402
from koscine.config import SILVER_DATA_ROOT  # noqa: E402

START, END, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
CE_LO, CE_HI, PE_LO, PE_HI = 0.99, 1.10, 0.85, 1.01
TOP_N = 50

g2 = {s: g for g, syms in json.loads((LOCK_V2 / "universe_groups.json").read_text()).items() for s in syms}
STATIC_NAMES = set(g2)

print("loading daily OI-in-lots ranking (top-50 dynamic universe)...", flush=True)
oi = pd.read_parquet(SILVER_TABLES["eod_deriv_daily"], columns=["date", "symbol", "fut_oi"])
oi["date"] = pd.to_datetime(oi["date"])
oi = oi[oi["fut_oi"].notna() & (oi["fut_oi"] > 0)].copy()
lot_df = pd.read_parquet(SILVER_DATA_ROOT / "lot_size.parquet", columns=["symbol", "expiry_month", "lot"])
lot_df["period"] = pd.to_datetime(lot_df["expiry_month"]).dt.to_period("M")
lot_df = lot_df.drop_duplicates(["symbol", "period"], keep="last")
oi["period"] = oi["date"].dt.to_period("M")
oi = oi.merge(lot_df[["symbol", "period", "lot"]], on=["symbol", "period"], how="left").dropna(subset=["lot"])
oi["oi_lots"] = oi["fut_oi"] / oi["lot"]
TOP50_BY_DATE: dict[pd.Timestamp, set] = {d: set(g.nlargest(TOP_N, "oi_lots")["symbol"]) for d, g in oi.groupby("date")}
print(f"  {len(TOP50_BY_DATE)} trading days with an OI-in-lots ranking", flush=True)

frames = []
dates = list(pd.bdate_range(START, END))
for k, d in enumerate(dates):
    names = STATIC_NAMES | TOP50_BY_DATE.get(d, set())
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
    if bc.empty:
        continue
    frames.append(bc[["date", "symbol", "expiry", "strike", "opt_type", "open", "high", "low", "close", "oi", "vol", "underlying"]])
    if k % 60 == 0:
        print(f"  {k}/{len(dates)}", flush=True)

panel = pd.concat(frames, ignore_index=True)
panel["date"] = pd.to_datetime(panel["date"]); panel["expiry"] = pd.to_datetime(panel["expiry"])
panel["group"] = panel["symbol"].map(lambda s: g2.get(s, "C_top50oi"))
panel.to_parquet(OUT, index=False)
print(f"wrote {len(panel)} rows, {panel['symbol'].nunique()} unique symbols -> {OUT}")
