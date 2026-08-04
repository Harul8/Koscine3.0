"""Large-mover ranker v3 — can anything beat raw implied vol?

v2: atm_iv alone (31.4/45.1) > PROD conf > 22-feat classifier. Try the proper ranking tool
(LambdaMART / XGBRanker, query = date x group) and a minimal IV+catalyst model, vs atm_iv.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))

import numpy as np
import pandas as pd
from xgboost import XGBRanker, XGBClassifier

from magnitude_ranker_v2 import build, FEATS, precision
from largemove.config import PROD

pd.set_option("display.width", 200)
PARAMS = dict(n_estimators=400, max_depth=6, learning_rate=0.04, subsample=0.85,
              colsample_bytree=0.85, tree_method="hist", device="cuda", verbosity=0)
MINI = ["atm_iv", "atm_iv_chg_5", "atm_iv_ratio_20", "earnings_within_5d", "atr_pct_14"]


def main():
    ab = build()
    parts = []
    for T in PROD.test_years:
        tr = ab[ab.year < T].sort_values(["date", "group"]).copy()
        ev = ab[ab.year == T].copy()
        if ev.empty:
            continue
        # LambdaMART: graded relevance = global quintile of move_mag, query groups = (date, group)
        tr["grade"] = pd.qcut(tr["move_mag"].rank(method="first"), 5, labels=False).astype(int)
        gsizes = tr.groupby(["date", "group"], sort=True).size().to_numpy()
        rk = XGBRanker(objective="rank:ndcg", **PARAMS).fit(tr[FEATS], tr["grade"], group=gsizes)
        ev["ltr"] = rk.predict(ev[FEATS])
        # minimal IV+catalyst classifier on is_top5
        y = tr["is_top5"]
        spw = (len(y) - y.sum()) / max(1, y.sum())
        clf = XGBClassifier(scale_pos_weight=spw, **PARAMS).fit(tr[MINI], y)
        ev["p_mini"] = clf.predict_proba(ev[MINI])[:, 1]
        parts.append(ev)
    ev = pd.concat(parts, ignore_index=True)

    print(f"eval rows={len(ev)} | group size/day={ev.groupby(['date','group']).size().mean():.0f}\n")
    print("=" * 78)
    print("CAN ANYTHING BEAT RAW IMPLIED VOL?  precision @ topN (per group/day)")
    print("=" * 78)
    rows = []
    for name, s in [("atm_iv alone", "atm_iv"), ("LambdaMART (LTR)", "ltr"),
                    ("minimal IV+catalyst", "p_mini")]:
        r = precision(ev.dropna(subset=[s]), s)
        rows.append({"ranker": name, **{f"{k}_{m}": v for k, kk in r.items() for m, v in kk.items()}})
    out = pd.DataFrame(rows)
    out.columns = [c.replace("in_top", "t").replace("_%", "").replace("capture", "cap") for c in out.columns]
    print(out.to_string(index=False))

    # how concentrated is the move? distribution of where atm_iv top-1 lands
    e = ev.copy()
    e["actual_rank"] = e.groupby(["date", "group"])["move_mag"].rank(ascending=False, method="first")
    e["iv_rank"] = e.groupby(["date", "group"])["atm_iv"].rank(ascending=False, method="first")
    top = e[e.iv_rank <= 1]
    print("\nwhere the #1 IV pick actually lands (move-rank distribution):")
    print((top.actual_rank.value_counts(normalize=True).sort_index().head(8) * 100).round(1).to_string())


if __name__ == "__main__":
    main()
