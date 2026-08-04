"""Mover-precision model — high-precision, low-volume, direction-agnostic LARGE-MOVE signals for option buying.

OBJECTIVE (per user): emit ~2-3 signals/day; maximize the chance they are among the day's actual TOP-3/TOP-5
movers (by 5-day magnitude, EITHER direction = what an option buyer captures, direction taken offline). Quality
over quantity; conviction-gate to lift precision as volume drops. Quarterly-retrain walk-forward 2024->2026.

Target: move_mag = max( max(high[t+1..t+5])/close-1 , close/min(low[t+1..t+5])-1 )   (exit-at-peak magnitude).
Eval: per day rank the tradeable 65-universe by model score; top-K signals; precision = in actual top-3/top-5;
P(>=1 of my top-3 in actual top-3/5); mean move_mag of signals; hit>=6/8%; wasted(<3%). vs atm_iv-rank + random.
CatBoost (reg move_mag / clf big-move), leak-safe (forward K2 label cols excluded). PROD untouched.

    set PYTHONPATH=src && python experiments/mover_precision_v1/mover_precision.py
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
from koscine3.data.sources import load_market_data  # noqa: E402

LOCK_V2 = ROOT / "locks" / "prod_largemove_v2"
W = 5
TRAIN_DAYS = 1100          # rolling ~4.5yr train window (recency + speed)
QUARTERS = pd.period_range("2024Q1", "2026Q2", freq="Q")
ID = {"date", "symbol", "open", "high", "low", "close", "volume", "move_mag", "up", "dn", "in_univ", "eligible"}
LEAK = ("future", "fwd", "next", "label", "tomorrow", "ahead", "adverse", "move_5d", "move_1d", "move_3d",
        "move_10d", "expansion", "volclean", "outcome")
CB = dict(iterations=600, depth=6, learning_rate=0.03, l2_leaf_reg=6.0, random_seed=7,
          allow_writing_files=False, verbose=False, task_type="GPU", devices="0")


def load():
    g2 = {s: g for g, syms in json.loads((LOCK_V2 / "universe_groups.json").read_text()).items() for s in syms}
    m = load_market_data()
    m["symbol"] = m["symbol"].astype(str)
    m["date"] = pd.to_datetime(m["date"])
    m = m.sort_values(["symbol", "date"]).reset_index(drop=True)
    g = m.groupby("symbol", sort=False)
    hi = pd.concat([g["high"].shift(-i) for i in range(1, W + 1)], axis=1).max(axis=1)
    lo = pd.concat([g["low"].shift(-i) for i in range(1, W + 1)], axis=1).min(axis=1)
    m["move_mag"] = np.maximum(hi / m.close - 1.0, m.close / lo - 1.0)
    m["in_univ"] = m.symbol.isin(g2)
    m["group"] = m.symbol.map(g2)
    m["eligible"] = m.close.ge(100) & m.atm_iv.notna()
    feats = [c for c in m.columns if c not in ID and c != "group" and pd.api.types.is_numeric_dtype(m[c])
             and not any(h in c.lower() for h in LEAK)]
    # cross-sectional rank-within-day features (help ranking)
    for c in ["atm_iv", "realized_vol_20", "atr_pct_14", "donchian_width_20"]:
        if c in m.columns:
            m[c + "_xrank"] = m.groupby("date")[c].rank(pct=True)
            feats.append(c + "_xrank")
    return m, feats


def score_walkforward(m, feats, target_fn, clf=False):
    from catboost import CatBoostClassifier, CatBoostRegressor
    parts = []
    for q in QUARTERS:
        qs, qe = q.start_time, q.end_time
        cut = qs - pd.Timedelta(days=W + 5)
        tr = m[(m.date < cut) & (m.date >= cut - pd.Timedelta(days=TRAIN_DAYS * 1.6)) & m.eligible & m.move_mag.notna()].dropna(subset=["atm_iv"])
        ev = m[(m.date >= qs) & (m.date <= qe) & m.eligible & m.in_univ & m.move_mag.notna()].copy()
        if len(tr) < 5000 or ev.empty:
            continue
        y = target_fn(tr)
        if clf:
            mdl = CatBoostClassifier(**CB, loss_function="Logloss").fit(tr[feats], y)
            ev["score"] = mdl.predict_proba(ev[feats])[:, 1]
        else:
            mdl = CatBoostRegressor(**CB, loss_function="RMSE").fit(tr[feats], y)
            ev["score"] = mdl.predict(ev[feats])
        parts.append(ev)
    return pd.concat(parts, ignore_index=True)


def precision_eval(ev, score, K=3, gate_frac=1.0):
    """Per day rank 65-univ by score, take top-K signals. Report precision vs actual top-3/top-5 movers."""
    d = ev.copy()
    d["arank"] = d.groupby("date").move_mag.rank(ascending=False, method="first")   # actual mover rank
    d["srank"] = d.groupby("date")[score].rank(ascending=False, method="first")     # model signal rank
    if gate_frac < 1.0:                                # conviction gate: keep only the strongest signal-days
        thr = d[d.srank <= K][score].quantile(1 - gate_frac)
        keep_days = d[(d.srank <= K) & (d[score] >= thr)].date.unique()
        d = d[d.date.isin(keep_days)]
    sig = d[d.srank <= K]
    if sig.empty:
        return {}
    per_day = sig.groupby("date").apply(lambda x: pd.Series({
        "in3": (x.arank <= 3).sum(), "in5": (x.arank <= 5).sum(), "n": len(x)}), include_groups=False)
    return {
        "signals/yr": round(len(sig) / 2.45),
        "prec_in_top3": round((sig.arank <= 3).mean(), 3),
        "prec_in_top5": round((sig.arank <= 5).mean(), 3),
        "P(>=1 in top3)/day": round((per_day.in3 >= 1).mean(), 3),
        "P(>=1 in top5)/day": round((per_day.in5 >= 1).mean(), 3),
        "mean_move%": round(sig.move_mag.mean() * 100, 2),
        "hit>=6%": round((sig.move_mag >= 0.06).mean(), 3),
        "hit>=8%": round((sig.move_mag >= 0.08).mean(), 3),
        "wasted<3%": round((sig.move_mag < 0.03).mean(), 3),
    }


def main():
    m, feats = load()
    nday_univ = m[m.eligible & m.in_univ].groupby("date").size()
    print(f"rows {len(m):,} | feats {len(feats)} | 65-univ eligible/day ~{nday_univ.median():.0f} | "
          f"univ move_mag median {m[m.eligible & m.in_univ].move_mag.median()*100:.2f}% mean {m[m.eligible & m.in_univ].move_mag.mean()*100:.2f}%")

    selectors = {}
    selectors["atm_iv"] = m[m.eligible & m.in_univ & m.move_mag.notna() & (m.date >= "2024-01-01")].copy()
    selectors["atm_iv"]["score"] = selectors["atm_iv"]["atm_iv"]
    np.random.seed(1); selectors["random"] = selectors["atm_iv"].copy(); selectors["random"]["score"] = np.random.rand(len(selectors["atm_iv"]))
    selectors["reg_movemag"] = score_walkforward(m, feats, lambda tr: tr.move_mag.clip(0, 0.4))
    selectors["clf_big6"] = score_walkforward(m, feats, lambda tr: (tr.move_mag >= 0.06).astype(int), clf=True)
    selectors["clf_big8"] = score_walkforward(m, feats, lambda tr: (tr.move_mag >= 0.08).astype(int), clf=True)

    print("\n=== PRECISION @ top-3 signals/day (pooled 65-univ) ===")
    hdr = f"{'selector':12s} {'sig/yr':>7s} {'in_top3':>8s} {'in_top5':>8s} {'>=1top3/d':>10s} {'>=1top5/d':>10s} {'move%':>6s} {'hit6%':>6s} {'wast':>5s}"
    print(hdr)
    rows = []
    for name, ev in selectors.items():
        r = precision_eval(ev, "score", K=3)
        if r:
            print(f"{name:12s} {r['signals/yr']:>7d} {r['prec_in_top3']:>8.3f} {r['prec_in_top5']:>8.3f} {r['P(>=1 in top3)/day']:>10.3f} {r['P(>=1 in top5)/day']:>10.3f} {r['mean_move%']:>6.2f} {r['hit>=6%']:>6.3f} {r['wasted<3%']:>5.3f}")
            rows.append((name, r))

    print("\n=== CONVICTION GATING (best model, fewer signal-days -> higher precision) ===")
    best = max([r for r in rows if r[0].startswith(("reg", "clf"))], key=lambda x: x[1]["prec_in_top5"])[0]
    print(f"  gating model = {best}")
    for gf in (1.0, 0.5, 0.3, 0.15):
        r = precision_eval(selectors[best], "score", K=3, gate_frac=gf)
        if r:
            print(f"   keep top {int(gf*100):3d}% conviction: {r['signals/yr']:>4d}/yr  in_top3 {r['prec_in_top3']}  in_top5 {r['prec_in_top5']}  >=1top5/d {r['P(>=1 in top5)/day']}  move% {r['mean_move%']}  hit6 {r['hit>=6%']}")

    # save the best model's forward book
    bk = selectors[best][["date", "symbol", "group", "score", "move_mag", "atm_iv"]].copy()
    bk["arank"] = bk.groupby("date").move_mag.rank(ascending=False, method="first")
    bk["srank"] = bk.groupby("date")["score"].rank(ascending=False, method="first")
    out = HERE / "results"; out.mkdir(exist_ok=True)
    bk.sort_values(["date", "srank"]).to_csv(out / f"book_{best}.csv", index=False)
    print(f"\nsaved forward book -> results/book_{best}.csv")


if __name__ == "__main__":
    main()
