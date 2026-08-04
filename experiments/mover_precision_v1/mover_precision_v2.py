"""Mover-precision v2 — push precision past v1 (atm_iv ~ ceiling). Levers: learning-to-rank (YetiRank),
ensemble (rank-avg), dual-gate (model AND atm_iv agree), per-group selection, lean features, and a
hit-rate table at the user's 1-3 signals/day volume (magnitude is what pays an option buyer).

    set PYTHONPATH=src && python experiments/mover_precision_v1/mover_precision_v2.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))
from mover_precision import load, score_walkforward, precision_eval, CB, QUARTERS, W  # noqa: E402

LEAN = ["atm_iv", "atm_iv_chg_5", "atm_iv_ratio_20", "realized_vol_20", "atr_pct_14", "atr_pct_14_rank_60d",
        "donchian_width_20", "range_contraction_5v20", "volume_dryup_score", "vol_sma20_ratio",
        "days_to_earnings", "earnings_within_5d", "is_expiry_week", "nifty_realized_vol_20", "mkt_pct_above_sma50",
        "put_call_iv_skew", "pcr_oi", "fut_chg_oi", "atm_iv_xrank", "realized_vol_20_xrank", "atr_pct_14_xrank"]


def rank_wf(m, feats):
    from catboost import CatBoostRanker, Pool
    parts = []
    cbr = {k: v for k, v in CB.items()}
    for q in QUARTERS:
        cut = q.start_time - pd.Timedelta(days=W + 5)
        tr = m[(m.date < cut) & (m.date >= cut - pd.Timedelta(days=1760)) & m.eligible & m.move_mag.notna()].dropna(subset=["atm_iv"]).sort_values("date")
        ev = m[(m.date >= q.start_time) & (m.date <= q.end_time) & m.eligible & m.in_univ & m.move_mag.notna()].copy()
        if len(tr) < 5000 or ev.empty:
            continue
        gid = pd.factorize(tr.date)[0]
        rel = (tr.move_mag.clip(0, 0.4) * 100).round().astype(int)     # integer relevance for YetiRank
        mdl = CatBoostRanker(loss_function="YetiRank", **cbr).fit(Pool(tr[feats], label=rel, group_id=gid))
        ev["score"] = mdl.predict(ev[feats])
        parts.append(ev)
    return pd.concat(parts, ignore_index=True)


def add_rank(df):
    df = df.copy()
    df["srank"] = df.groupby("date")["score"].rank(ascending=False, method="first")
    return df


def pergroup_eval(ev, kg=2):
    d = ev.copy()
    d["arank"] = d.groupby("date").move_mag.rank(ascending=False, method="first")
    d["gr"] = d.groupby(["date", "group"])["score"].rank(ascending=False, method="first")
    sig = d[d.gr <= kg]
    pd_ = sig.groupby("date").apply(lambda x: (x.arank <= 5).sum(), include_groups=False)
    return {"sig/yr": round(len(sig) / 2.45), "in_top3": round((sig.arank <= 3).mean(), 3),
            "in_top5": round((sig.arank <= 5).mean(), 3), ">=1top5/d": round((pd_ >= 1).mean(), 3),
            "move%": round(sig.move_mag.mean() * 100, 2), "hit6": round((sig.move_mag >= .06).mean(), 3)}


def main():
    m, feats = load()
    print(f"loaded rows {len(m):,} feats {len(feats)}")
    base_iv = m[m.eligible & m.in_univ & m.move_mag.notna() & (m.date >= "2024-01-01")].copy()
    base_iv["score"] = base_iv["atm_iv"]

    print("\n--- scoring models (walk-forward) ---", flush=True)
    reg = score_walkforward(m, feats, lambda tr: tr.move_mag.clip(0, 0.4))
    clf8 = score_walkforward(m, feats, lambda tr: (tr.move_mag >= 0.08).astype(int), clf=True)
    yeti = rank_wf(m, feats)
    lean = score_walkforward(m, [c for c in LEAN if c in m.columns], lambda tr: (tr.move_mag >= 0.08).astype(int), clf=True)

    scored = {"atm_iv": base_iv, "reg": reg, "clf8": clf8, "yetirank": yeti, "clf8_lean": lean}

    # ensemble: rank-average of clf8 + reg + atm_iv (align on date,symbol)
    key = ["date", "symbol"]
    ens = clf8[key + ["score", "move_mag", "group", "atm_iv"]].rename(columns={"score": "s_clf"})
    ens = ens.merge(reg[key + ["score"]].rename(columns={"score": "s_reg"}), on=key)
    ens = ens.merge(base_iv[key + ["score"]].rename(columns={"score": "s_iv"}), on=key)
    for c in ("s_clf", "s_reg", "s_iv"):
        ens[c + "r"] = ens.groupby("date")[c].rank(pct=True)
    ens["score"] = ens[["s_clfr", "s_regr", "s_ivr"]].mean(axis=1)
    scored["ensemble"] = ens

    print(f"\n=== top-3/day precision (K=3 pooled) ===")
    print(f"{'selector':12s} {'in3':>6s} {'in5':>6s} {'>=1t5/d':>8s} {'move%':>6s} {'hit6':>6s} {'hit8':>6s}")
    for name, ev in scored.items():
        r = precision_eval(ev, "score", K=3)
        print(f"{name:12s} {r['prec_in_top3']:>6.3f} {r['prec_in_top5']:>6.3f} {r['P(>=1 in top5)/day']:>8.3f} {r['mean_move%']:>6.2f} {r['hit>=6%']:>6.3f} {r['hit>=8%']:>6.3f}")

    # dual-gate: clf8 AND atm_iv both top-5 that day (agreement) -> intersection signal
    print("\n=== dual-gate (clf8 top-N AND atm_iv top-N agree) ===")
    dg = clf8[key + ["move_mag", "group"]].copy()
    dg["r_clf"] = clf8.groupby("date")["score"].rank(ascending=False, method="first").values
    dg = dg.merge(base_iv[key].assign(r_iv=base_iv.groupby("date")["score"].rank(ascending=False, method="first").values), on=key)
    dg["arank"] = dg.groupby("date").move_mag.rank(ascending=False, method="first")
    for N in (3, 5, 8):
        sig = dg[(dg.r_clf <= N) & (dg.r_iv <= N)]
        pday = sig.groupby("date").apply(lambda x: (x.arank <= 5).sum(), include_groups=False)
        print(f"   both top-{N}: {round(len(sig)/2.45):>4d}/yr  in3 {(sig.arank<=3).mean():.3f}  in5 {(sig.arank<=5).mean():.3f}  >=1t5/d {(pday>=1).mean():.3f}  hit6 {(sig.move_mag>=.06).mean():.3f}  move% {sig.move_mag.mean()*100:.2f}")

    # per-group selection (A,B separate top-2 each = ~4/day)
    print("\n=== per-group selection (clf8, top-2 per group) ===")
    print("  ", pergroup_eval(clf8, 2))

    # hit-rate (option-buyer metric) by conviction, best clf8
    print("\n=== clf8: HIT-RATE by signals/day (the option-buyer metric: did it move big?) ===")
    for K in (1, 2, 3):
        r = precision_eval(clf8, "score", K=K)
        print(f"   top-{K}/day ({r['signals/yr']}/yr): hit>=6% {r['hit>=6%']}  hit>=8% {r['hit>=8%']}  move% {r['mean_move%']}  in5 {r['prec_in_top5']}  >=1t5/d {r['P(>=1 in top5)/day']}")
    for gf in (0.5, 0.3, 0.15):
        r = precision_eval(clf8, "score", K=3, gate_frac=gf)
        print(f"   top-3 gate{int(gf*100)}% ({r['signals/yr']}/yr): hit>=6% {r['hit>=6%']}  hit>=8% {r['hit>=8%']}  move% {r['mean_move%']}")


if __name__ == "__main__":
    main()
