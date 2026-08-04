"""Exit-timing test for the cheap_convexity straddle book on REAL premiums.
Held-to-close gives the convexity back (median -9.5%). Compare exit rules over t+1..t+WINDOW:
  held   = exit at last available close (what premium_ev.py measured)
  peak   = exit at the best close (ORACLE upper bound — not tradeable, shows the ceiling)
  pt25   = exit first day straddle close is up >= 25% (realistic standing limit), else last close
  pt50   = same at +50%
Reports each rule per selector, per predicted-surprise quintile, per group, net of cost.

    set PYTHONPATH=src && python experiments/cheap_convexity_v1/premium_ev_exits.py

Heavier (loads t+1..t+WINDOW bhavcopy per entry date; bounded rolling cache). Read-only; PROD untouched.
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
import premium_ev as pe                                   # noqa: E402  (reuse atm_straddle/exit_premium/bc/WINDOW)

W = pe.WINDOW


def main():
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
            path = [v for d in fwd if (v := pe.exit_premium(pd.Timestamp(d), r.symbol, strike, expiry)) is not None]
            if not path:
                continue
            rets = [v / ep - 1.0 for v in path]
            held = rets[-1]
            peak = max(rets)
            def pt(tgt):
                for x in rets:
                    if x >= tgt:
                        return tgt
                return rets[-1]
            rows.append({"selector": r.selector, "date": t, "symbol": r.symbol,
                         "held": held, "peak": peak, "pt25": pt(0.25), "pt50": pt(0.50)})
        for k in [d for d in list(pe._BC) if d < pd.Timestamp(t).normalize()]:   # bound memory: evict past dates
            pe._BC.pop(k, None)

    res = pd.DataFrame(rows)
    pk = picks[["selector", "date", "symbol", "group", "pred"]].copy()
    m = res.merge(pk, on=["selector", "date", "symbol"], how="left")
    m.to_csv(HERE / "results" / "premium_ev_exits.csv", index=False)
    print(f"priced picks: {len(m)}\n")

    def report(df, label):
        print(f"--- {label} (n={len(df)}) ---")
        for rule in ("held", "peak", "pt25", "pt50"):
            r = df[rule]
            print(f"   {rule:5s} mean={r.mean()*100:+6.2f}%  net@3%={ (r-0.03).mean()*100:+6.2f}%  win={(r>0).mean():.3f}")

    for sel, d in m.groupby("selector"):
        report(d, sel)
    cc = m[m.selector.eq("cheap_convexity")].copy()
    cc["q"] = pd.qcut(cc.pred, 5, labels=[1, 2, 3, 4, 5], duplicates="drop")
    report(cc[cc.q.isin([4, 5])], "cheap_convexity  Q4-Q5 (selective)")
    report(cc[cc.group.eq("A_mcap30")], "cheap_convexity  A_mcap30")
    report(cc[cc.q.isin([4, 5]) & cc.group.eq("A_mcap30")], "cheap_convexity  Q4-Q5 & A_mcap30")
    print("\nsaved -> results/premium_ev_exits.csv")


if __name__ == "__main__":
    main()
