"""Round-2 research signals — NEW direction candidates not tested before.

  - 52-week-high proximity (George-Hwang 2004): near high -> continuation up
  - Close-Location-Value / Chaikin money flow: daily buying-pressure / order-flow proxy
  - option-VOLUME skew + pcr_vol (Pan-Poteshman 2006): call-heavy volume -> up
Tested CONDITIONAL ON A BIG MOVE (dir_peak = which side spikes more; dir_close = closes up).
Plus: do these add to a combined direction model beyond the prior 0.548 AUC?
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from koscine3.data.sources import load_market_data

pd.set_option("display.width", 200)
UNIV = {s for v in json.loads((HERE / "universe_groups.json").read_text()).values() for s in v}

COLS = ["date", "symbol", "open", "high", "low", "close", "volume",
        "pcr_vol", "pcr_vol_chg_5", "opt_call_vol", "opt_put_vol",
        "delivery_pct_chg_5", "ret_5d", "ret_20d", "atm_iv", "atm_iv_chg_5",
        "iv_skew_ce_minus_pe", "fut_chg_oi", "oi_buildup_ratio", "fut_chg_oi_ratio_20",
        "close_sma50_dist", "adx_14", "gap_pct"]


def auc_safe(y, x):
    d = pd.DataFrame({"y": y, "x": x}).replace([np.inf, -np.inf], np.nan).dropna()
    if d.y.nunique() < 2 or len(d) < 300:
        return np.nan, len(d)
    return roc_auc_score(d.y, d.x), len(d)


def main():
    m = load_market_data(columns=COLS)
    m["symbol"] = m["symbol"].astype(str)
    m = m[m.symbol.isin(UNIV)].sort_values(["symbol", "date"]).reset_index(drop=True)
    g = m.groupby("symbol", sort=False)

    # forward 5-day labels
    entry = g["open"].shift(-1)
    H = pd.concat([g["high"].shift(-i) for i in range(1, 6)], axis=1).max(axis=1)
    L = pd.concat([g["low"].shift(-i) for i in range(1, 6)], axis=1).min(axis=1)
    c5 = g["close"].shift(-5)
    m["up_move"] = (H - entry) / entry
    m["down_move"] = (entry - L) / entry
    m["signed_close"] = (c5 - entry) / entry

    # NEW signals (all known at EOD t)
    hi252 = g["close"].transform(lambda s: s.rolling(252, min_periods=120).max())
    lo252 = g["close"].transform(lambda s: s.rolling(252, min_periods=120).min())
    m["dist_52wh"] = m.close / hi252 - 1.0                       # near 0 = at high -> continuation up
    m["dist_52wl"] = m.close / lo252 - 1.0
    rng = (m.high - m.low).replace(0, np.nan)
    m["clv"] = ((m.close - m.low) - (m.high - m.close)) / rng     # intraday buying pressure [-1,1]
    m["clv_5"] = m.groupby("symbol")["clv"].transform(lambda s: s.rolling(5, min_periods=3).mean())
    mf = (m["clv"] * m["volume"])
    m["cmf_20"] = (mf.groupby(m.symbol).transform(lambda s: s.rolling(20, min_periods=10).sum())
                   / m.groupby("symbol")["volume"].transform(lambda s: s.rolling(20, min_periods=10).sum()))
    m["opt_vol_skew"] = (m.opt_call_vol - m.opt_put_vol) / (m.opt_call_vol + m.opt_put_vol)

    m = m[m.close >= 100].dropna(subset=["up_move", "down_move", "signed_close"]).copy()
    m["dir_peak"] = (m.up_move > m.down_move).astype(int)
    m["dir_close"] = (m.signed_close > 0).astype(int)
    m["big4"] = (np.maximum(m.up_move, m.down_move) >= 0.04).astype(int)
    m["yr"] = m.date.dt.year

    big = m[m.big4 == 1].copy()
    print(f"rows={len(m)} | big-move rows={len(big)} | dir_peak base={big.dir_peak.mean()*100:.1f}% up\n")

    print("=" * 74)
    print("NEW DIRECTION SIGNALS — univariate AUC (conditional on big >=4% move)")
    print("=" * 74)
    new = ["dist_52wh", "dist_52wl", "clv", "clv_5", "cmf_20", "opt_vol_skew", "pcr_vol", "pcr_vol_chg_5"]
    rows = []
    for s in new:
        ap, n = auc_safe(big.dir_peak, big[s])
        ac, _ = auc_safe(big.dir_close, big[s])
        rows.append({"signal": s, "AUC_dir_peak": round(ap, 3), "AUC_dir_close": round(ac, 3), "n": n})
    print(pd.DataFrame(rows).to_string(index=False))
    print("(AUC<0.5 => predicts DOWN; strength = |AUC-0.5|)")

    # combined direction model with NEW + prior signals
    feats = new + ["iv_skew_ce_minus_pe", "fut_chg_oi_ratio_20", "oi_buildup_ratio",
                   "ret_20d", "close_sma50_dist", "gap_pct", "atm_iv_chg_5", "delivery_pct_chg_5"]
    params = dict(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
                  colsample_bytree=0.8, tree_method="hist", device="cuda", verbosity=0)
    print("\n" + "=" * 74)
    print("COMBINED direction model (NEW+prior) — train<2023, test>=2023")
    print("=" * 74)
    for label in ("dir_peak", "dir_close"):
        d = big.dropna(subset=feats + [label])
        tr, te = d[d.yr < 2023], d[d.yr >= 2023]
        clf = XGBClassifier(**params).fit(tr[feats], tr[label])
        auc = roc_auc_score(te[label], clf.predict_proba(te[feats])[:, 1])
        print(f"  {label}: OOS AUC = {auc:.3f}  (was 0.548 without new signals; test n={len(te)})")
        if label == "dir_peak":
            imp = pd.Series(clf.feature_importances_, index=feats).sort_values(ascending=False)
            print("   top:", ", ".join(f"{k}={v:.2f}" for k, v in imp.head(8).items()))

    # MAGNITUDE add-on: does 52wk-high proximity / CLV help predict a big move (beyond noise)?
    print("\n" + "=" * 74)
    print("MAGNITUDE — do new signals predict P(big>=4%)? univariate AUC")
    print("=" * 74)
    for s in ["dist_52wh", "dist_52wl", "clv", "cmf_20", "opt_vol_skew"]:
        a, n = auc_safe(m.big4, m[s].abs() if s in ("dist_52wh", "clv") else m[s])
        print(f"  {s:14s} AUC={a:.3f}  n={n}")


if __name__ == "__main__":
    main()
