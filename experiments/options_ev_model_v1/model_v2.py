"""Options-gain ML model v2 — managed-exit selection, raw-return vs SURPRISE target.

v1: held-5d is theta-killed AND predicting the raw straddle return mis-ranks (picks expensive straddles).
v2 fixes both: (1) day-by-day path -> managed exits (d3/d4/trailing); (2) compare two model targets —
  A) the managed-exit straddle return (raw), and
  B) the cheap_convexity SURPRISE = |realized 5d move| − atm_iv·sqrt(5/252)  (residual; ranks well, proven IC).
Both are eval'd on the SAME realized managed-exit return: top-K/group + top-quintile, net of 3% cost, vs atm_iv.

    set PYTHONPATH=src && python experiments/options_ev_model_v1/model_v2.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))
from model_v1 import add_features, COST, EMBARGO, QUARTERS, CB  # noqa: E402
from koscine3.data.sources import load_market_data             # noqa: E402

RCOLS = ["r1", "r2", "r3", "r4", "r5"]


def trail_exit(row, x=0.30):
    pk, last = -9.0, np.nan
    for c in RCOLS:
        v = row[c]
        if pd.isna(v):
            continue
        last, pk = v, max(pk, v)
        if (1 + v) / (1 + pk) - 1 <= -x:
            return v
    return last


def load_exits():
    df = pd.read_csv(HERE / "results" / "straddle_paths.csv", parse_dates=["entry_date"])
    df = df.rename(columns={"entry": "entry_prem"})
    Rf = df[RCOLS].ffill(axis=1)
    df["held"], df["d3"], df["d4"] = Rf["r5"], Rf["r3"], Rf["r4"]
    df["peak"] = df[RCOLS].max(axis=1)
    df["trail30"] = df.apply(lambda r: trail_exit(r, 0.30), axis=1)
    # realized 5d stock move + cheap_convexity surprise target
    f = load_market_data(columns=["date", "symbol", "close"]).sort_values(["symbol", "date"])
    f["symbol"] = f["symbol"].astype(str); f["date"] = pd.to_datetime(f["date"])
    f["fwd5"] = f.groupby("symbol").close.shift(-5) / f.close - 1.0
    df = df.merge(f[["date", "symbol", "fwd5"]], left_on=["entry_date", "symbol"], right_on=["date", "symbol"], how="left").drop(columns=["date"])
    df["surprise"] = df.fwd5.abs() - df.atm_iv * np.sqrt(5 / 252)
    return df


def wf_score(d, feats, target):
    from catboost import CatBoostRegressor
    out = []
    for q in QUARTERS:
        cut = q.start_time - pd.Timedelta(days=5 + EMBARGO)
        tr = d[(d.entry_date < cut) & d[target].notna()]
        te = d[(d.entry_date >= q.start_time) & (d.entry_date <= q.end_time) & d[target].notna()].copy()
        if len(tr) < 3000 or te.empty:
            continue
        clip = (-1, 8) if target != "surprise" else (-0.5, 0.5)
        mdl = CatBoostRegressor(**CB, loss_function="RMSE").fit(tr[feats], tr[target].clip(*clip))
        te["score"] = mdl.predict(te[feats])
        out.append(te)
    return pd.concat(out, ignore_index=True)


def ev_topk(ev, score, ret_col, k):
    picks = ev.sort_values(score, ascending=False).groupby(["entry_date", "group"], sort=False).head(k)
    r = picks[ret_col].to_numpy() - COST
    return round(r.mean() * 100, 2), round((r > 0).mean(), 3), round(len(picks) / 2.45)


def ev_quintile(ev, score, ret_col, q=5):
    ev = ev.copy(); ev["qb"] = pd.qcut(ev[score], q, labels=False, duplicates="drop")
    top = ev[ev.qb == ev.qb.max()]
    r = top[ret_col].to_numpy() - COST
    return round(r.mean() * 100, 2), round((r > 0).mean(), 3), len(top)


def main():
    raw = load_exits()
    print(f"paths {len(raw):,} | surprise non-null {raw.surprise.notna().mean():.2f}")
    EXITS = ["held", "d3", "d4", "trail30", "peak"]
    print("\n=== universe EV by exit (net@3%), per structure ===")
    for s in ("straddle", "strangle"):
        u = raw[raw.structure == s]
        print("  " + s + ": " + "  ".join(f"{e} {(u[e].mean()-COST)*100:+.1f}%(win{(u[e]>0).mean():.2f})" for e in EXITS))

    summary = []
    for s in ("straddle", "strangle"):
        d0 = raw[raw.structure == s].copy()
        d, feats = add_features(d0)
        print(f"\n===== {s} | rows {len(d):,} | feats {len(feats)} =====")
        for ret_col in ("d3", "d4", "trail30"):
            for target in (ret_col, "surprise"):
                ev = wf_score(d, feats, target)
                t2 = ev_topk(ev, "score", ret_col, 2)
                qn = ev_quintile(ev, "score", ret_col)
                tag = "raw " if target == ret_col else "surp"
                print(f"  exit={ret_col:7s} target={tag} | top2/grp net {t2[0]:+.2f}% win{t2[1]} ({t2[2]}/yr) | topQ net {qn[0]:+.2f}% win{qn[1]} n{qn[2]}")
                summary.append((s, ret_col, tag, t2[0], qn[0]))
    print("\n=== SUMMARY (top2/grp net EV %, best first) ===")
    for r in sorted(summary, key=lambda x: -x[3]):
        print(f"  {r[0]:9s} exit={r[1]:7s} target={r[2]} : top2 {r[3]:+.2f}%  topQ {r[4]:+.2f}%")


if __name__ == "__main__":
    main()
