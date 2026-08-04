"""Test research-backed pre-move patterns on our data (A+B universe, EOD t -> t+1..t+5 forward).

MAGNITUDE (will a big move happen?) — established precursors:
  - volatility contraction (NR7/VCP/squeeze): low Donchian width / low ATR percentile
  - volume dry-up during the base
DIRECTION (which way? — the unsolved problem) — academically-supported leak signals:
  - options: call-minus-put IV "volatility spread" (Cremers-Weinbaum 2010, +ve -> up),
             IV skew (Xing-Zhang 2010, steep put skew -> down)
  - F&O positioning: OI build-up (price up + OI up = long build-up = sustainable up)
  - delivery%: rising delivery on up-day = accumulation
  - momentum / relative strength

Labels (per stock-day, entry t+1 open, 5-day window):
  up_move/down_move = max favorable excursion each side; big = max(.)>=thr
  dir_peak  = up_move > down_move   (which side spikes more — relevant for option side)
  dir_close = 5-day close is up     (the harder close-direction)
DIRECTION is tested CONDITIONAL ON A BIG MOVE (our actual decision: gate to movers, then pick side).
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
groups = json.loads((HERE / "universe_groups.json").read_text())
UNIV = {s for v in groups.values() for s in v}

COLS = ["date", "symbol", "open", "high", "low", "close",
        "atm_ce_iv", "atm_pe_iv", "atm_iv", "put_call_iv_skew", "iv_skew_ce_minus_pe", "iv_skew_norm", "atm_iv_chg_5",
        "fut_chg_oi", "oi_buildup_ratio", "fut_oi_ratio_20", "fut_chg_oi_ratio_20", "pcr_oi", "pcr_oi_chg_5",
        "delivery_pct", "delivery_pct_chg_5", "delivery_qty_ratio_20",
        "ret_1d", "ret_5d", "ret_20d", "ret_20d_cs_rank", "rel_ret_5d_vs_nifty", "close_sma50_dist",
        "adx_14", "donchian_width_20", "atr_pct_14", "atr_pct_14_rank_60d", "volume_dryup_score",
        "vol_sma20_ratio", "gap_pct"]


def auc_safe(y, x):
    d = pd.DataFrame({"y": y, "x": x}).replace([np.inf, -np.inf], np.nan).dropna()
    if d.y.nunique() < 2 or len(d) < 200:
        return np.nan, len(d)
    return roc_auc_score(d.y, d.x), len(d)


def load():
    m = load_market_data(columns=COLS)
    m["symbol"] = m["symbol"].astype(str)
    m = m[m.symbol.isin(UNIV)].sort_values(["symbol", "date"]).reset_index(drop=True)
    g = m.groupby("symbol", sort=False)
    entry = g["open"].shift(-1)
    H = pd.concat([g["high"].shift(-i) for i in range(1, 6)], axis=1).max(axis=1)
    L = pd.concat([g["low"].shift(-i) for i in range(1, 6)], axis=1).min(axis=1)
    c5 = g["close"].shift(-5)
    m["up_move"] = (H - entry) / entry
    m["down_move"] = (entry - L) / entry
    m["signed_close"] = (c5 - entry) / entry
    m = m[(m.close >= 100)].dropna(subset=["up_move", "down_move", "signed_close"]).copy()
    m["dir_peak"] = (m.up_move > m.down_move).astype(int)
    m["dir_close"] = (m.signed_close > 0).astype(int)
    # engineered research signals
    m["vol_spread"] = m.atm_ce_iv - m.atm_pe_iv                       # +ve = expensive calls -> up
    m["oi_long_score"] = np.sign(m.ret_1d) * m.fut_chg_oi            # long build-up positive
    m["deliv_dir"] = m.delivery_pct_chg_5 * np.sign(m.ret_5d)        # accumulation vs distribution
    m["yr"] = m.date.dt.year
    return m


def main():
    m = load()
    print(f"universe={len(UNIV)} stocks | rows={len(m)} | years {m.yr.min()}-{m.yr.max()}")

    # ---------------- MAGNITUDE: do contraction / dry-up precede big moves? -------------
    for thr in (0.04, 0.05):
        m[f"big{int(thr*100)}"] = (np.maximum(m.up_move, m.down_move) >= thr).astype(int)
    print(f"\nbase rates: big>=4% = {m.big4.mean()*100:.1f}%   big>=5% = {m.big5.mean()*100:.1f}%")
    print("\n" + "=" * 70)
    print("MAGNITUDE — univariate AUC for P(big move >=4% in next 5d)")
    print("=" * 70)
    for s in ["donchian_width_20", "atr_pct_14", "atr_pct_14_rank_60d", "volume_dryup_score",
              "vol_sma20_ratio", "adx_14", "atm_iv", "atm_iv_chg_5"]:
        a, n = auc_safe(m.big4, m[s])
        tag = "  (inverse: low->big)" if a < 0.5 else ""
        print(f"  {s:24s} AUC={a:.3f}  n={n}{tag}")

    # ---------------- DIRECTION conditional on a big move ------------------------------
    big = m[m.big4 == 1].copy()
    print("\n" + "=" * 70)
    print(f"DIRECTION | conditional on big>=4% move (n={len(big)}, dir_peak base={big.dir_peak.mean()*100:.1f}% up)")
    print("=" * 70)
    dir_signals = ["vol_spread", "iv_skew_ce_minus_pe", "put_call_iv_skew", "iv_skew_norm",
                   "fut_chg_oi", "oi_buildup_ratio", "oi_long_score", "fut_chg_oi_ratio_20",
                   "delivery_pct_chg_5", "deliv_dir", "delivery_qty_ratio_20",
                   "ret_5d", "ret_20d", "ret_20d_cs_rank", "rel_ret_5d_vs_nifty", "close_sma50_dist",
                   "pcr_oi", "pcr_oi_chg_5", "gap_pct"]
    rows = []
    for s in dir_signals:
        ap, _ = auc_safe(big.dir_peak, big[s])
        ac, n = auc_safe(big.dir_close, big[s])
        rows.append({"signal": s, "AUC_dir_peak": round(ap, 3), "AUC_dir_close": round(ac, 3), "n": n})
    res = pd.DataFrame(rows).sort_values("AUC_dir_peak", key=lambda c: (c - 0.5).abs(), ascending=False)
    print(res.to_string(index=False))
    print("(AUC<0.5 = the signal predicts DOWN; strength = |AUC-0.5|)")

    # ---------------- combined direction model (time-split, honest) -------------------
    feats = ["vol_spread", "iv_skew_ce_minus_pe", "iv_skew_norm", "atm_iv_chg_5",
             "fut_chg_oi", "oi_buildup_ratio", "oi_long_score", "fut_chg_oi_ratio_20", "pcr_oi_chg_5",
             "delivery_pct_chg_5", "deliv_dir", "delivery_qty_ratio_20",
             "ret_5d", "ret_20d", "ret_20d_cs_rank", "rel_ret_5d_vs_nifty", "close_sma50_dist", "gap_pct", "adx_14"]
    params = dict(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
                  colsample_bytree=0.8, tree_method="hist", device="cuda", verbosity=0)
    print("\n" + "=" * 70)
    print("COMBINED direction model — train<2023, test>=2023 (OOS)")
    print("=" * 70)
    for label in ("dir_peak", "dir_close"):
        d = big.dropna(subset=feats + [label])
        tr, te = d[d.yr < 2023], d[d.yr >= 2023]
        clf = XGBClassifier(**params).fit(tr[feats], tr[label])
        p = clf.predict_proba(te[feats])[:, 1]
        auc = roc_auc_score(te[label], p)
        print(f"  {label}: OOS AUC = {auc:.3f}  (train {len(tr)}, test {len(te)}, base {te[label].mean()*100:.0f}% )")
        if label == "dir_peak":
            imp = pd.Series(clf.feature_importances_, index=feats).sort_values(ascending=False)
            print("   top features:", ", ".join(f"{k}={v:.2f}" for k, v in imp.head(8).items()))

    # tertile lift for the single best direction signal
    best = res.iloc[0]["signal"]
    print(f"\nTERTILE LIFT — best signal '{best}' on dir_peak (conditional on big):")
    big["_t"] = pd.qcut(big[best].rank(method="first"), 3, labels=["low", "mid", "high"])
    print(big.groupby("_t", observed=True).dir_peak.mean().mul(100).round(1).to_string())


if __name__ == "__main__":
    main()
