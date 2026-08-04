"""Extract ATM+2% option liquidity (traded contracts) + premium per (stock, day) for the 65-universe, 2024-2026.
Needed for the v3 liquidity gate (signal only if ATM+2% has >=1000 contracts). Light snapshot pass (no window).

    set PYTHONPATH=src && python experiments/mover_precision_v1/extract_atm2_liquidity.py [START END]
Writes results/atm2_liquidity.csv. Read-only; PROD untouched.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "analysis"))
from koscine3.data.sources import load_market_data       # noqa: E402
from options_bhavcopy import load_bhavcopy                # noqa: E402

LOCK_V2 = ROOT / "locks" / "prod_largemove_v2"
MIN_DTE = 6


def main(start, end):
    g2 = {s: g for g, syms in json.loads((LOCK_V2 / "universe_groups.json").read_text()).items() for s in syms}
    f = load_market_data(columns=["date", "symbol", "close", "atm_iv"])
    f["symbol"] = f["symbol"].astype(str); f["date"] = pd.to_datetime(f["date"])
    f = f[f.symbol.isin(g2) & (f.date >= start) & (f.date <= end)]
    px = {(r.date, r.symbol): r.close for r in f.itertuples(index=False)}
    cal = list(np.sort(f.date.unique()))

    rows = []
    for d in cal:
        d = pd.Timestamp(d)
        bc = load_bhavcopy(d)
        if bc is None or bc.empty:
            continue
        bcx = bc.dropna(subset=["strike", "expiry"]).copy()
        bcx["expiry"] = pd.to_datetime(bcx["expiry"])
        for sym, sub in bcx.groupby("symbol"):
            if sym not in g2:
                continue
            U = px.get((d, sym))
            if U is None or U < 100:
                continue
            exps = sorted(e for e in sub.expiry.dropna().unique() if (pd.Timestamp(e) - d).days >= MIN_DTE)
            if not exps:
                continue
            ch = sub[sub.expiry.eq(pd.Timestamp(exps[0]))]
            ce, pe = ch[ch.opt_type.eq("CE")], ch[ch.opt_type.eq("PE")]
            if ce.empty or pe.empty:
                continue
            ck = min(ce.strike.unique(), key=lambda s: abs(s - U * 1.02))
            pk = min(pe.strike.unique(), key=lambda s: abs(s - U * 0.98))
            cr, pr = ce[ce.strike.eq(ck)], pe[pe.strike.eq(pk)]
            if cr.empty or pr.empty:
                continue
            rows.append({"date": d, "symbol": sym, "U": U,
                         "c_vol": float(cr.vol.iloc[0]), "p_vol": float(pr.vol.iloc[0]),
                         "c_oi": float(cr.oi.iloc[0]), "p_oi": float(pr.oi.iloc[0]),
                         "c_prem": float(cr.close.iloc[0]), "p_prem": float(pr.close.iloc[0])})
    df = pd.DataFrame(rows)
    out = HERE / "results"; out.mkdir(exist_ok=True)
    df.to_csv(out / "atm2_liquidity.csv", index=False)
    liq = (df[["c_vol", "p_vol"]].min(axis=1) >= 1000).mean()
    print(f"rows {len(df):,} | {df.date.min().date()}..{df.date.max().date()} | both-legs>=1000 contracts: {liq:.1%}")
    print(f"saved -> {out/'atm2_liquidity.csv'}")


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0] if len(a) >= 1 else "2024-01-01", a[1] if len(a) >= 2 else "2026-06-12")
