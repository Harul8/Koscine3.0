"""Final refinement: recency-adaptive model (trailing ~1yr window) — does adapting to the recent regime
recover any 2025-26 direction edge that the expanding-window model lost? Per-quarter OOS AUC + extreme-decile acc."""
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

    print("recency-adaptive (train on trailing 365 calendar days only), per quarter:\n")
    rows = []
    for q in pd.period_range("2025Q1", "2026Q2", freq="Q"):
        win_start = q.start_time - pd.Timedelta(days=365)
        tr = m[(m.date >= win_start) & (m.date < q.start_time)]
        te = m[(m.date >= q.start_time) & (m.date <= q.end_time)].copy()
        if te.empty or len(tr) < 5000:
            continue
        clf = XGBClassifier(**PARAMS).fit(tr[feats], tr.dir5)
        te["p"] = clf.predict_proba(te[feats])[:, 1]
        d = te.assign(dec=pd.qcut(te.p.rank(method="first"), 10, labels=False))
        top, bot = d[d.dec == 9], d[d.dec == 0]
        rows.append({"quarter": str(q), "AUC": round(roc_auc_score(te.dir5, te.p), 4),
                     "top_dec_up%": round(top.dir5.mean() * 100, 1),
                     "bot_dec_up%": round(bot.dir5.mean() * 100, 1),
                     "extreme_acc%": round(((top.dir5.mean()) + (1 - bot.dir5.mean())) / 2 * 100, 1),
                     "n_test": len(te)})
    print(pd.DataFrame(rows).to_string(index=False))
    aucs = [r["AUC"] for r in rows if r["quarter"].startswith("2026")]
    print(f"\n2026 mean AUC (recency model) = {np.mean(aucs):.4f}" if aucs else "")


if __name__ == "__main__":
    main()
