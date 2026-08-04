"""Save the full straddle-value PATH per pick (entry + daily returns t+1..t+W) and sweep exit rules.
One heavy bhavcopy pass -> results/premium_ev_paths.csv (reusable for ANY future exit rule, no more loads).
Sweeps: held, peak(oracle), trailing-stop 20/30/40%, fixed exit on day k. Trailing stop = let winners run,
exit when straddle value falls X% from its running peak (does NOT cap the upside, unlike a profit target).

    set PYTHONPATH=src && python experiments/cheap_convexity_v1/premium_ev_paths.py
Read-only; PROD untouched.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(HERE))
from koscine3.data.sources import load_market_data       # noqa: E402
import premium_ev as pe                                   # noqa: E402

W = pe.WINDOW
COST = 0.03


def build_paths() -> pd.DataFrame:
    picks = pd.read_csv(HERE / "results" / "picks.csv", parse_dates=["date"])
    cal = list(np.sort(pd.to_datetime(load_market_data(columns=["date"]).date.unique())))
    pos = {d: i for i, d in enumerate(cal)}
    rows = []
    for t, day_picks in picks.groupby("date"):
        ti = pos.get(pd.Timestamp(t).normalize())
        if ti is None or ti + 1 >= len(cal):
            continue
        fwd = cal[ti + 1: ti + 1 + W]
        for r in day_picks.itertuples(index=False):
            ent = pe.atm_straddle(pd.Timestamp(t), r.symbol)
            if not ent or ent[2] <= 0:
                continue
            strike, expiry, ep = ent
            rets = [(v / ep - 1.0) for d in fwd if (v := pe.exit_premium(pd.Timestamp(d), r.symbol, strike, expiry)) is not None]
            if not rets:
                continue
            row = {"selector": r.selector, "date": t, "symbol": r.symbol, "group": r.group,
                   "pred": r.pred, "entry_prem": ep}
            for i in range(W):
                row[f"r{i+1}"] = rets[i] if i < len(rets) else np.nan
            rows.append(row)
        for k in [d for d in list(pe._BC) if d < pd.Timestamp(t).normalize()]:
            pe._BC.pop(k, None)
    return pd.DataFrame(rows)


def path_of(row) -> list[float]:
    return [row[f"r{i+1}"] for i in range(W) if pd.notna(row[f"r{i+1}"])]


def held(p): return p[-1]
def peak(p): return max(p)
def fixed(p, k): return p[min(k, len(p)) - 1]
def trail(p, x):
    pk = -9.9
    for r in p:
        pk = max(pk, r)
        if (1 + r) / (1 + pk) - 1 <= -x:        # value fell x% from running peak -> exit at this close
            return r
    return p[-1]


RULES = {"held": held, "peak(oracle)": peak, "exit_d1": lambda p: fixed(p, 1), "exit_d2": lambda p: fixed(p, 2),
         "exit_d3": lambda p: fixed(p, 3), "trail20": lambda p: trail(p, .20),
         "trail30": lambda p: trail(p, .30), "trail40": lambda p: trail(p, .40)}


def sweep(df, label):
    paths = df.apply(path_of, axis=1)
    print(f"\n--- {label} (n={len(df)}) ---  net of {COST*100:.0f}% cost")
    for name, fn in RULES.items():
        r = paths.apply(fn).to_numpy() - COST
        print(f"   {name:13s} mean={r.mean()*100:+6.2f}%  win={(r>0).mean():.3f}  median={np.median(r)*100:+6.2f}%")


def main():
    df = build_paths()
    df.to_csv(HERE / "results" / "premium_ev_paths.csv", index=False)
    print(f"saved paths: {len(df)} picks -> results/premium_ev_paths.csv")
    cc = df[df.selector.eq("cheap_convexity")].copy()
    cc["q"] = pd.qcut(cc.pred, 5, labels=[1, 2, 3, 4, 5], duplicates="drop")
    sweep(df[df.selector.eq("atm_iv_baseline")], "atm_iv baseline")
    sweep(cc, "cheap_convexity ALL")
    sweep(cc[cc.q.isin([4, 5])], "cheap_convexity Q4-Q5")
    sweep(cc[cc.group.eq("A_mcap30")], "cheap_convexity A_mcap30")
    sweep(cc[cc.q.isin([4, 5]) & cc.group.eq("A_mcap30")], "cheap_convexity Q4-Q5 & A")


if __name__ == "__main__":
    main()
