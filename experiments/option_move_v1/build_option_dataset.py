"""Contract-level dataset for option_move_v1 — model which OPTION CONTRACTS give big premium moves (vega+gamma,
not just the underlying). Strike ladder = ATM +/-5% (calls & puts), real bhavcopy. Entry at close[d], 5-day fwd.

Per (stock, day d, strike, side): record entry premium (close[d]), OI, volume, underlying U, strike, moneyness,
DTE  +  the forward peak/held option gain (over d+1..d+5). Per-strike IV is added offline (add_iv.py: BS inversion).
Streaming (each bhavcopy date once). Read-only; PROD untouched.

    set PYTHONPATH=src && python experiments/option_move_v1/build_option_dataset.py [START END]
Writes results/option_contracts.csv.
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
W = 5
MIN_DTE = 6
MIN_UNDERLYING = 100.0
MNY = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]   # strike ladder: ATM +/- 5% (per strike we take BOTH call & put)


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
        lk = bcx.set_index(["symbol", "strike", "expiry", "opt_type"])[["high", "close"]]
        lk = lk[~lk.index.duplicated()].to_dict("index")

        still = []
        for p in pos:
            v = lk.get((p["symbol"], p["strike"], p["expiry"], p["ot"]))
            if v is not None:
                p["max_high"] = max(p["max_high"], v["high"]); p["last_close"] = v["close"]
            p["days"] += 1
            (rows if p["days"] >= W else still).append(p)
        pos = still

        for sym, sub in bcx.groupby("symbol"):
            if sym not in g2:
                continue
            info = px.get((d, sym))
            if info is None or info[0] < MIN_UNDERLYING:
                continue
            U, atm = info
            exps = sorted(e for e in sub.expiry.dropna().unique() if (pd.Timestamp(e) - d).days >= MIN_DTE)
            if not exps:
                continue
            exp = pd.Timestamp(exps[0]); dte = (exp - d).days
            ch = sub[sub.expiry.eq(exp)]
            strikes_all = sorted(ch.strike.unique())
            used = set()
            for m in MNY:
                k = min(strikes_all, key=lambda s: abs(s - U * (1 + m / 100)))
                if k in used:
                    continue
                used.add(k)
                for ot in ("CE", "PE"):
                    r = ch[(ch.strike.eq(k)) & (ch.opt_type.eq(ot))]
                    if r.empty:
                        continue
                    prem = float(r.close.iloc[0])
                    if prem <= 0:
                        continue
                    pos.append({"symbol": sym, "group": g2[sym], "date": d, "ot": ot, "strike": k, "expiry": exp,
                                "dte": dte, "U": U, "atm_iv": atm, "moneyness": round((k / U - 1) * 100, 2),
                                "entry": prem, "oi": float(r.oi.iloc[0]), "vol": float(r.vol.iloc[0]),
                                "max_high": 0.0, "last_close": np.nan, "days": 0})  # peak over FORWARD highs only

    out = []
    for p in rows:
        out.append({**{k: p[k] for k in ("symbol", "group", "date", "ot", "strike", "expiry", "dte", "U",
                                         "atm_iv", "moneyness", "entry", "oi", "vol")},
                    "peak_ratio": p["max_high"] / p["entry"], "held_ratio": p["last_close"] / p["entry"]})
    df = pd.DataFrame(out)
    df = df[df.peak_ratio > 0]                       # drop contracts with no forward trades
    res = HERE / "results"; res.mkdir(parents=True, exist_ok=True)
    df.to_csv(res / "option_contracts.csv", index=False)
    print(f"contracts {len(df):,} | {df.date.min().date()}..{df.date.max().date()} | symbols {df.symbol.nunique()} "
          f"| CE {sum(df.ot=='CE'):,} PE {sum(df.ot=='PE'):,} | mean peak {df.peak_ratio.mean():.2f}x")
    print(f"saved -> {res/'option_contracts.csv'}")


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0] if len(a) >= 1 else "2024-01-01", a[1] if len(a) >= 2 else "2026-06-12")
