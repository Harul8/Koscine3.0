"""How good is 1-DAY direction prediction? Multivariate XGBoost, walk-forward, AUC + extreme-decile accuracy by year."""
from __future__ import annotations
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parents[1] / "src"))
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from diag_direction2 import FEAT_COLS, PARAMS
from koscine3.data.sources import load_market_data

m = load_market_data(columns=["date", "symbol", "close"] + FEAT_COLS)
m["symbol"] = m["symbol"].astype(str)
m = m.sort_values(["symbol", "date"]).reset_index(drop=True)
g = m.groupby("symbol", sort=False)
m["fwd1"] = g["close"].shift(-1) / m["close"] - 1.0
m["vol_spread"] = m.atm_ce_iv - m.atm_pe_iv
hi252 = g["close"].transform(lambda s: s.rolling(252, min_periods=120).max())
m["dist_52wh"] = m.close / hi252 - 1.0
feats = FEAT_COLS + ["vol_spread", "dist_52wh"]
m[feats] = m[feats].replace([np.inf, -np.inf], np.nan)
m = m[(m.close >= 100) & m.fwd1.notna()].copy()
m["dir1"] = (m.fwd1 > 0).astype(int)

parts = []
for q in pd.period_range("2024Q1", "2026Q2", freq="Q"):
    tr = m[m.date < q.start_time]
    te = m[(m.date >= q.start_time) & (m.date <= q.end_time)].copy()
    if te.empty:
        continue
    clf = XGBClassifier(**PARAMS).fit(tr[feats], tr.dir1)
    te["p"] = clf.predict_proba(te[feats])[:, 1]
    parts.append(te)
oos = pd.concat(parts, ignore_index=True)
oos["yr"] = oos.date.dt.year

print(f"1-DAY direction, multivariate walk-forward (OOS {len(oos)} rows)\n")
print(f"overall OOS AUC = {roc_auc_score(oos.dir1, oos.p):.4f}   (0.5 = coin flip; P(up)={oos.dir1.mean():.3f})\n")
rows = []
for y in (2024, 2025, 2026):
    d = oos[oos.yr == y].assign(dec=lambda x: pd.qcut(x.p.rank(method="first"), 10, labels=False))
    top, bot = d[d.dec == 9], d[d.dec == 0]
    rows.append({"year": y, "AUC": round(roc_auc_score(d.dir1, d.p), 4), "P(up)": round(d.dir1.mean(), 3),
                 "top_dec_up%": round(top.dir1.mean() * 100, 1), "bot_dec_up%": round(bot.dir1.mean() * 100, 1),
                 "extreme_dir_acc%": round(((top.dir1.mean()) + (1 - bot.dir1.mean())) / 2 * 100, 1)})
print(pd.DataFrame(rows).to_string(index=False))
