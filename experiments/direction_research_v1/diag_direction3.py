"""Refinement: is the multivariate direction edge DURABLE? Quarterly walk-forward (retrain each quarter),
report OOS AUC + confident-decile directional accuracy BY YEAR, overall and on the high-IV mover subset."""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parents[1] / "src"))
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from diag_direction2 import FEAT_COLS, PARAMS
from koscine3.data.sources import load_market_data

pd.set_option("display.width", 200)


def main():
    cols = ["date", "symbol", "close"] + FEAT_COLS
    m = load_market_data(columns=cols)
    m["symbol"] = m["symbol"].astype(str)
    m = m.sort_values(["symbol", "date"]).reset_index(drop=True)
    g = m.groupby("symbol", sort=False)
    m["fwd5"] = g["close"].shift(-5) / m["close"] - 1.0
    m["vol_spread"] = m.atm_ce_iv - m.atm_pe_iv
    hi252 = g["close"].transform(lambda s: s.rolling(252, min_periods=120).max())
    m["dist_52wh"] = m.close / hi252 - 1.0
    feats = FEAT_COLS + ["vol_spread", "dist_52wh"]
    m[feats] = m[feats].replace([np.inf, -np.inf], np.nan)
    m = m[(m.close >= 100) & m.fwd5.notna()].copy()
    m["dir5"] = (m.fwd5 > 0).astype(int)

    parts = []
    for q in pd.period_range("2024Q1", "2026Q2", freq="Q"):
        tr = m[m.date < q.start_time]
        te = m[(m.date >= q.start_time) & (m.date <= q.end_time)].copy()
        if te.empty:
            continue
        clf = XGBClassifier(**PARAMS).fit(tr[feats], tr.dir5)
        te["p"] = clf.predict_proba(te[feats])[:, 1]
        parts.append(te)
    oos = pd.concat(parts, ignore_index=True)
    oos["yr"] = oos.date.dt.year
    oos["iv_hi"] = oos.atm_iv >= oos.atm_iv.quantile(0.66)

    print(f"walk-forward OOS rows={len(oos)} (2024-26)\n")
    print("=" * 70)
    print("Durability: OOS AUC + confident-decile UP-rate, BY YEAR")
    print("=" * 70)
    rows = []
    for y in (2024, 2025, 2026):
        d = oos[oos.yr == y]
        d = d.assign(dec=pd.qcut(d.p.rank(method="first"), 10, labels=False))
        top, bot = d[d.dec == 9], d[d.dec == 0]
        acc = ((top.dir5.mean()) + (1 - bot.dir5.mean())) / 2
        rows.append({"year": y, "AUC": round(roc_auc_score(d.dir5, d.p), 4),
                     "P(up)": round(d.dir5.mean(), 3),
                     "top_dec_up%": round(top.dir5.mean() * 100, 1),
                     "bot_dec_up%": round(bot.dir5.mean() * 100, 1),
                     "extreme_dir_acc%": round(acc * 100, 1)})
    print(pd.DataFrame(rows).to_string(index=False))

    print("\n" + "=" * 70)
    print("Does it help the HIGH-IV mover subset (what the book trades)?")
    print("=" * 70)
    rows = []
    for label, d0 in [("all", oos), ("high_IV only", oos[oos.iv_hi])]:
        d = d0.assign(dec=pd.qcut(d0.p.rank(method="first"), 10, labels=False))
        top, bot = d[d.dec == 9], d[d.dec == 0]
        rows.append({"subset": label, "AUC": round(roc_auc_score(d.dir5, d.p), 4),
                     "top_dec_up%": round(top.dir5.mean() * 100, 1),
                     "bot_dec_up%": round(bot.dir5.mean() * 100, 1),
                     "n": len(d)})
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
