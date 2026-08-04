"""Large-mover ranker v2 — optimize the precision@k objective DIRECTLY.

v1 finding: MSE regression < PROD confidence. Fix: train classifiers on the actual target
(is this a top-3 / top-5 mover within its group that day), walk-forward, rank by P(top-k).
Also rank by atm_iv alone (how much is just implied vol?) and a rank-blend of PROD conf + P(top5).
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from largemove import pipeline as P
from largemove.config import PROD
from koscine3.data.sources import load_market_data

PROD_PRED = HERE.parents[1] / "locks" / "prod_largemove_v1" / "predictions"
pd.set_option("display.width", 200)

EXTRA = ["atm_iv_chg_5", "earnings_within_5d", "gap_pct", "fut_chg_oi", "oi_buildup_ratio",
         "vol_sma20_ratio", "pcr_oi_chg_5", "ret_5d", "ret_20d", "fut_chg_oi_ratio_20",
         "delivery_qty_ratio_20", "adx_14"]
FEATS = ["atm_iv", "atr_pct_14", "realized_vol_20", "donchian_width_20", "sector_vol_20",
         "nifty_realized_vol_20", "atm_iv_ratio_20", "atm_iv_chg_5", "atr_pct_14_rank_60d",
         "atr_pct_14_cs_rank", "days_to_earnings", "earnings_within_5d", "gap_pct",
         "vol_sma20_ratio", "vol_5v20_ratio", "abs_ret_5d", "abs_ret_20d", "abs_fut_chg_oi",
         "oi_buildup_ratio", "pcr_oi_chg_5", "ret_20d_cs_rank", "delivery_qty_ratio_20"]
CLF = dict(n_estimators=400, max_depth=6, learning_rate=0.04, subsample=0.85,
           colsample_bytree=0.85, tree_method="hist", device="cuda", verbosity=0)


def build():
    df = P.load_dataset(PROD)
    mag = df.groupby(["date", "symbol"])["ceiling"].max().reset_index(name="move_mag")
    base = df[df.side.eq("long")].drop(columns=["side", "ceiling"]).merge(mag, on=["date", "symbol"])
    extra = load_market_data(columns=["date", "symbol", *EXTRA])
    extra["symbol"] = extra["symbol"].astype(str)
    base = base.merge(extra.drop_duplicates(["date", "symbol"]), on=["date", "symbol"], how="left")
    base["abs_ret_5d"] = base["ret_5d"].abs()
    base["abs_ret_20d"] = base["ret_20d"].abs()
    base["abs_fut_chg_oi"] = base["fut_chg_oi"].abs()

    prod = pd.concat([pd.read_csv(PROD_PRED / f"group_{b}_predictions.csv", parse_dates=["date"])
                      for b in dict(PROD.group_thresholds)], ignore_index=True)
    prod["symbol"] = prod["symbol"].astype(str)
    conf = prod.groupby(["date", "symbol"])["confidence"].max().reset_index(name="prod_conf")
    base = base.merge(conf, on=["date", "symbol"], how="left")

    ab = base[base.group.notna() & base.eligible].dropna(subset=["move_mag"]).copy()
    ab["mrank"] = ab.groupby(["date", "group"])["move_mag"].rank(ascending=False, method="first")
    ab["is_top3"] = (ab.mrank <= 3).astype(int)
    ab["is_top5"] = (ab.mrank <= 5).astype(int)
    return ab


def walk_forward(ab):
    parts = []
    for T in PROD.test_years:
        tr = ab[ab.year < T]                 # XGBoost handles NaN natively — keep full universe
        ev = ab[ab.year == T].copy()
        if ev.empty:
            continue
        for tgt in ("is_top3", "is_top5"):
            y = tr[tgt]
            spw = (len(y) - y.sum()) / max(1, y.sum())
            clf = XGBClassifier(scale_pos_weight=spw, **CLF).fit(tr[FEATS], y)
            ev[f"p_{tgt}"] = clf.predict_proba(ev[FEATS])[:, 1]
        parts.append(ev)
    ev = pd.concat(parts, ignore_index=True)
    # rank-blend of PROD conf + P(top5)
    for c in ("prod_conf", "p_is_top5"):
        ev[c + "_r"] = ev.groupby(["date", "group"])[c].rank(pct=True)
    ev["blend"] = ev["prod_conf_r"] + ev["p_is_top5_r"]
    return ev


def precision(ev, score):
    ev = ev.copy()
    ev["actual_rank"] = ev.groupby(["date", "group"])["move_mag"].rank(ascending=False, method="first")
    ev["day_max"] = ev.groupby(["date", "group"])["move_mag"].transform("max")
    ev["pred_rank"] = ev.groupby(["date", "group"])[score].rank(ascending=False, method="first")
    out = {}
    for n in (1, 2):
        p = ev[ev.pred_rank <= n]
        out[f"top{n}"] = {"in_top3_%": round((p.actual_rank <= 3).mean() * 100, 1),
                          "in_top5_%": round((p.actual_rank <= 5).mean() * 100, 1),
                          "capture_%": round((p.move_mag / p.day_max).mean() * 100, 1)}
    return out


def main():
    ab = build()
    ev = walk_forward(ab)
    print(f"eval rows={len(ev)} | avg group size/day={ev.groupby(['date','group']).size().mean():.0f}\n")
    print("=" * 78)
    print("PRECISION @ topN — fraction of picks among day's biggest movers (per group)")
    print("=" * 78)
    scorers = [("PROD confidence (baseline)", "prod_conf"), ("atm_iv alone", "atm_iv"),
               ("P(top3) classifier", "p_is_top3"), ("P(top5) classifier", "p_is_top5"),
               ("blend conf+P(top5)", "blend")]
    rows = []
    for name, s in scorers:
        d = ev.dropna(subset=[s])
        r = precision(d, s)
        rows.append({"ranker": name, **{f"{k}_{m}": v for k, kk in r.items() for m, v in kk.items()}})
    out = pd.DataFrame(rows)
    out.columns = [c.replace("in_top", "t").replace("_%", "").replace("capture", "cap") for c in out.columns]
    print(out.to_string(index=False))
    print("\n(top1_t3 = top-1 pick in day's top-3; top2_t5 = top-2 picks in top-5; cap = % of day's best move captured)")


if __name__ == "__main__":
    main()
