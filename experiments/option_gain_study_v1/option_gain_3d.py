"""3-day variant: A_mcap30 -> ATM+1%, B_turn35 -> ATM+2%, hold 3 trading days.
Captures both exits: peak (max high over the 3d window) and held (close at day 3). Entry at OPEN.
Saves per-day ratios so 'held to day 1/2/3' and 'peak' are all derivable.

    set PYTHONPATH=src && python experiments/option_gain_study_v1/option_gain_3d.py [START END]
Writes results/option_gain_3d_trades.csv. Streaming bhavcopy (each date once). Read-only; PROD untouched.
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
HOLD = 3
FLOOR = 2.0
MIN_DTE = 4
GROUP_LVL = {"A_mcap30": 1, "B_turn35": 2}    # A mega-cap -> ATM+1%, B movers -> ATM+2%


def main(start: str, end: str) -> None:
    g2 = {s: g for g, syms in json.loads((LOCK_V2 / "universe_groups.json").read_text()).items() for s in syms}
    f = load_market_data(columns=["date", "symbol", "close", "atm_iv"])
    f["symbol"] = f["symbol"].astype(str)
    f = f[f.symbol.isin(g2)]
    f["date"] = pd.to_datetime(f["date"])
    f = f[(f.date >= start) & (f.date <= end)]
    px = {(r.date, r.symbol): (r.close, r.atm_iv) for r in f.itertuples(index=False)}
    cal = list(np.sort(f.date.unique()))

    positions: list[dict] = []
    rows: list[dict] = []
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
        for p in positions:
            v = lk.get((p["symbol"], p["strike"], p["expiry"], p["ot"]))
            if v is not None:
                if v["high"] > p["max_high"]:
                    p["max_high"] = v["high"]; p["peak_day"] = p["days"]
                p["last_close"] = v["close"]
                p[f"c{p['days']}"] = v["close"]
            p["days"] += 1
            if p["days"] > HOLD:
                p["exit_close_u"] = px.get((d, p["symbol"]), (np.nan, np.nan))[0]
                rows.append(p)
            else:
                still.append(p)
        positions = still

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
            lvl = GROUP_LVL[g2[sym]]
            for side, ot in (("CALL", "CE"), ("PUT", "PE")):
                legs = ch[ch.opt_type.eq(ot)]
                if legs.empty:
                    continue
                ks = sorted(legs.strike.unique())
                tgt = U * (1 + lvl / 100) if ot == "CE" else U * (1 - lvl / 100)
                k = min(ks, key=lambda s: abs(s - tgt))
                r = legs[legs.strike.eq(k)]
                if r.empty:
                    continue
                o, h, c = float(r.open.iloc[0]), float(r.high.iloc[0]), float(r.close.iloc[0])
                if not (o >= FLOOR):
                    continue
                positions.append({"symbol": sym, "group": g2[sym], "side": side, "ot": ot,
                                  "strike_label": f"ATM+{lvl}%", "otm_pct": round((k / U - 1) * 100 if ot == "CE" else (1 - k / U) * 100, 2),
                                  "strike": k, "expiry": exp, "dte": (exp - d).days, "entry_date": d, "U": U,
                                  "atm_iv": iv, "entry_open": o, "max_high": h, "last_close": c, "peak_day": 1,
                                  "c1": c, "c2": np.nan, "c3": np.nan, "days": 1})

    df = pd.DataFrame(rows)
    df["peak_ratio"] = df.max_high / df.entry_open
    df["held_ratio"] = df.last_close / df.entry_open
    for k in (1, 2, 3):
        df[f"close{k}_ratio"] = df[f"c{k}"] / df.entry_open
    df["stock_move"] = df.exit_close_u / df.U - 1.0
    df = df.drop(columns=["ot", "days", "c1", "c2", "c3", "max_high", "last_close"])
    out = HERE / "results"; out.mkdir(exist_ok=True)
    df.to_csv(out / "option_gain_3d_trades.csv", index=False)
    print(f"trades {len(df):,} | {df.entry_date.min().date()}..{df.entry_date.max().date()} | symbols {df.symbol.nunique()} | "
          f"mean peak {df.peak_ratio.mean():.2f}x held {df.held_ratio.mean():.2f}x")
    print(f"saved -> {out/'option_gain_3d_trades.csv'}")


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0] if len(a) >= 1 else "2024-01-01", a[1] if len(a) >= 2 else "2026-06-12")
