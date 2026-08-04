"""Definitive ceiling test: full multivariate XGBoost for 5-day direction, broad universe, OOS time-split.
If interactions hide an edge, this finds it. Reports OOS AUC, by-year, and confidence-decile lift
(does the most-confident decile actually go the called way?)."""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parents[1] / "src"))
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from koscine3.data.sources import load_market_data

pd.set_option("display.width", 200)
FEAT_COLS = ["atm_iv", "atm_ce_iv", "atm_pe_iv", "atm_iv_chg_5", "atm_iv_ratio_20", "put_call_iv_skew",
             "iv_skew_ce_minus_pe", "pcr_oi", "pcr_vol", "pcr_oi_chg_5", "pcr_vol_chg_5", "fut_oi_ratio_20",
             "fut_chg_oi", "fut_chg_oi_ratio_20", "oi_buildup_ratio", "fut_oi_chg_5", "delivery_pct",
             "delivery_pct_chg_5", "delivery_qty_ratio_20", "ret_1d", "ret_3d", "ret_5d", "ret_10d", "ret_20d",
             "ret_5d_cs_rank", "ret_20d_cs_rank", "rel_ret_5d_vs_nifty", "close_sma5_dist", "close_sma20_dist",
             "close_sma50_dist", "adx_14", "donchian_width_20", "atr_pct_14", "realized_vol_20",
             "vol_sma20_ratio", "gap_pct", "sector_ret_5d", "stock_rel_sector_ret_5d",
             "stock_rel_sector_ret_20d", "mkt_pct_above_sma50", "days_to_earnings"]
PARAMS = dict(n_estimators=500, max_depth=5, learning_rate=0.03, subsample=0.8,
              colsample_bytree=0.7, tree_method="hist", device="cuda", verbosity=0,
              reg_lambda=5.0, min_child_weight=20)


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
    m = m[(m.close >= 100) & m.fwd5.notna()].copy()
    m["dir5"] = (m.fwd5 > 0).astype(int)
    m["yr"] = m.date.dt.year

    m[feats] = m[feats].replace([np.inf, -np.inf], np.nan)
    tr = m[m.yr < 2024]
    te = m[m.yr >= 2024].copy()
    print(f"train rows={len(tr)} ({tr.yr.min()}-{tr.yr.max()}) | test rows={len(te)} (2024-26) | feats={len(feats)}")
    clf = XGBClassifier(**PARAMS).fit(tr[feats], tr.dir5)
    te["p"] = clf.predict_proba(te[feats])[:, 1]

    auc = roc_auc_score(te.dir5, te.p)
    print(f"\nFULL MULTIVARIATE OOS AUC (dir5, 2024-26) = {auc:.4f}")
    for y in (2024, 2025, 2026):
        d = te[te.yr == y]
        print(f"  {y}: AUC {roc_auc_score(d.dir5, d.p):.4f}  (P(up)={d.dir5.mean():.3f}, n={len(d)})")

    print("\nconfidence-decile lift (test): does the most-confident decile go the called way?")
    te["dec"] = pd.qcut(te.p.rank(method="first"), 10, labels=False)
    lift = te.groupby("dec").agg(p_mean=("p", "mean"), actual_up=("dir5", "mean"), n=("dir5", "size"))
    print(lift.round(3).to_string())
    top, bot = te[te.dec == 9], te[te.dec == 0]
    print(f"\ntop decile (most bullish): actual up = {top.dir5.mean()*100:.1f}%  | "
          f"bottom decile (most bearish): actual up = {bot.dir5.mean()*100:.1f}%")
    print(f"=> if you traded the extreme deciles, directional accuracy ~ "
          f"{((top.dir5.mean()) + (1-bot.dir5.mean()))/2*100:.1f}%")
    imp = pd.Series(clf.feature_importances_, index=feats).sort_values(ascending=False)
    print("\ntop features:", ", ".join(f"{k}={v:.2f}" for k, v in imp.head(10).items()))


if __name__ == "__main__":
    main()
