"""Build the per (stock, day) STRADDLE & STRANGLE premium PATH (combined call+put close each of 5 days) on real
bhavcopy — so any exit rule (held / day-3 / day-4 / peak / trailing) is derivable for modeling. Entry at OPEN.
Straddle = ATM call + ATM put; Strangle = ATM+3% call + ATM-3% put (3% OTM both). Streaming (each date once).

    set PYTHONPATH=src && python experiments/options_ev_model_v1/build_straddle_path.py [START END]
Writes results/straddle_paths.csv. Read-only; PROD untouched.
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
HOLD = 5
FLOOR = 2.0
MIN_DTE = 6
STRUCTS = [("straddle", 0, 0), ("strangle", 3, 3)]   # (name, call %OTM, put %OTM)


def main(start, end):
    g2 = {s: g for g, syms in json.loads((LOCK_V2 / "universe_groups.json").read_text()).items() for s in syms}
    f = load_market_data(columns=["date", "symbol", "close", "atm_iv"])
    f["symbol"] = f["symbol"].astype(str); f["date"] = pd.to_datetime(f["date"])
    f = f[f.symbol.isin(g2) & (f.date >= start) & (f.date <= end)]
    px = {(r.date, r.symbol): (r.close, r.atm_iv) for r in f.itertuples(index=False)}
    cal = list(np.sort(f.date.unique()))

    pos, rows = [], []
    for d in cal:
        d = pd.Timestamp(d)
        bc = load_bhavcopy(d)
        if bc is None or bc.empty:
            continue
        bcx = bc.dropna(subset=["strike", "expiry"]).copy()
        bcx["expiry"] = pd.to_datetime(bcx["expiry"])
        lk = bcx.set_index(["symbol", "strike", "expiry", "opt_type"])[["open", "high", "close"]]
        lk = lk[~lk.index.duplicated()].to_dict("index")

        still = []
        for p in pos:
            vc = lk.get((p["symbol"], p["cstrike"], p["expiry"], "CE"))
            vp = lk.get((p["symbol"], p["pstrike"], p["expiry"], "PE"))
            if vc is not None and vp is not None:
                p["r"][p["days"]] = (vc["close"] + vp["close"]) / p["entry"] - 1.0
            p["days"] += 1
            if p["days"] >= HOLD:
                rows.append(p)
            else:
                still.append(p)
        pos = still

        bcg = {s: sub for s, sub in bcx.groupby("symbol")}
        for sym in g2:
            info = px.get((d, sym))
            if info is None:
                continue
            U, iv = info
            if not (U and U >= 100) or pd.isna(iv):
                continue
            sub = bcg.get(sym)
            if sub is None:
                continue
            exps = sorted(e for e in sub.expiry.dropna().unique() if (pd.Timestamp(e) - d).days >= MIN_DTE)
            if not exps:
                continue
            exp = pd.Timestamp(exps[0])
            ch = sub[sub.expiry.eq(exp)]
            ce, pe = ch[ch.opt_type.eq("CE")], ch[ch.opt_type.eq("PE")]
            if ce.empty or pe.empty:
                continue
            cks, pks = sorted(ce.strike.unique()), sorted(pe.strike.unique())
            for name, c_otm, p_otm in STRUCTS:
                ck = min(cks, key=lambda s: abs(s - U * (1 + c_otm / 100)))
                pk = min(pks, key=lambda s: abs(s - U * (1 - p_otm / 100)))
                cr, pr = ce[ce.strike.eq(ck)], pe[pe.strike.eq(pk)]
                if cr.empty or pr.empty:
                    continue
                entry = float(cr.open.iloc[0]) + float(pr.open.iloc[0])
                if entry < FLOOR:
                    continue
                r = [np.nan] * HOLD
                r[0] = (float(cr.close.iloc[0]) + float(pr.close.iloc[0])) / entry - 1.0
                pos.append({"symbol": sym, "group": g2[sym], "structure": name, "expiry": exp,
                            "cstrike": ck, "pstrike": pk, "dte": (exp - d).days, "entry_date": d,
                            "atm_iv": iv, "entry": entry, "r": r, "days": 1})

    out = []
    for p in rows:
        rec = {k: p[k] for k in ("symbol", "group", "structure", "dte", "entry_date", "atm_iv", "entry")}
        for i in range(HOLD):
            rec[f"r{i+1}"] = p["r"][i]
        out.append(rec)
    df = pd.DataFrame(out)
    res = HERE / "results"; res.mkdir(exist_ok=True)
    df.to_csv(res / "straddle_paths.csv", index=False)
    print(f"paths {len(df):,} | {df.entry_date.min().date()}..{df.entry_date.max().date()} | "
          f"straddle {sum(df.structure=='straddle'):,} strangle {sum(df.structure=='strangle'):,}")
    print(f"saved -> {res/'straddle_paths.csv'}")


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0] if len(a) >= 1 else "2024-01-01", a[1] if len(a) >= 2 else "2026-06-12")
