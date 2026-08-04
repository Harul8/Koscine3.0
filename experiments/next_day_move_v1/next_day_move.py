"""Predict NEXT-DAY (t+1) movement MAGNITUDE — direction-agnostic.

Decision at EOD t. Target = the largest distance t+1 travels from today's close, either way:
    next_move = max( (high[t+1]-close[t])/close[t] , (close[t]-low[t+1])/close[t] )
Walk-forward (quarterly retrain) XGBoost regressor on magnitude/catalyst features, vs the atm_iv baseline
(implied vol = the market's own daily-move forecast). Metrics: cross-sectional IC, AUC for move>=thr,
top-k mover precision within group, calibration. Broad F&O training; A/B ranking eval. PROD untouched.

    python next_day_move.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parents[1] / "src"))
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from xgboost import XGBRegressor
from koscine3.data.sources import load_market_data

pd.set_option("display.width", 200)
G2 = {s: g for g, syms in json.loads((HERE / "universe_groups.json").read_text()).items() for s in syms}
RAW = ["date", "symbol", "open", "high", "low", "close", "atm_iv", "atm_iv_chg_5", "atm_iv_ratio_20",
       "atr_pct_14", "atr_pct_14_rank_60d", "realized_vol_20", "donchian_width_20", "sector_vol_20",
       "nifty_realized_vol_20", "vol_sma20_ratio", "days_to_earnings", "earnings_within_5d", "gap_pct",
       "ret_1d", "ret_5d", "fut_chg_oi", "oi_buildup_ratio", "pcr_oi_chg_5", "turnover_ratio_20"]
FEATS = ["atm_iv", "atm_iv_chg_5", "atm_iv_ratio_20", "atr_pct_14", "atr_pct_14_rank_60d", "realized_vol_20",
         "donchian_width_20", "sector_vol_20", "nifty_realized_vol_20", "vol_sma20_ratio", "days_to_earnings",
         "earnings_within_5d", "gap_pct", "abs_ret_1d", "abs_ret_5d", "abs_fut_chg_oi", "oi_buildup_ratio",
         "pcr_oi_chg_5", "turnover_ratio_20"]
REG = dict(n_estimators=500, max_depth=5, learning_rate=0.03, subsample=0.8, colsample_bytree=0.7,
           reg_lambda=5.0, min_child_weight=20, tree_method="hist", device="cuda", verbosity=0)


def build():
    m = load_market_data(columns=RAW)
    m["symbol"] = m["symbol"].astype(str)
    m = m.sort_values(["symbol", "date"]).reset_index(drop=True)
    g = m.groupby("symbol", sort=False)
    up = (g["high"].shift(-1) - m["close"]) / m["close"]
    dn = (m["close"] - g["low"].shift(-1)) / m["close"]
    m["next_move"] = np.maximum(up, dn)
    m["c2c"] = (g["close"].shift(-1) - m["close"]).abs() / m["close"]
    m["abs_ret_1d"] = m["ret_1d"].abs()
    m["abs_ret_5d"] = m["ret_5d"].abs()
    m["abs_fut_chg_oi"] = m["fut_chg_oi"].abs()
    m[FEATS] = m[FEATS].replace([np.inf, -np.inf], np.nan)
    m["group"] = m["symbol"].map(G2)
    m["eligible"] = m["close"].ge(100) & m["atm_iv"].notna()
    return m[m.next_move.notna()].copy()


def daily_ic(df, pred):
    return df.groupby("date").apply(lambda d: d[[pred, "next_move"]].corr(method="spearman").iloc[0, 1]
                                    if len(d) > 5 else np.nan).mean()


def main():
    m = build()
    parts = []
    for q in pd.period_range("2024Q1", "2026Q2", freq="Q"):
        tr = m[(m.date < q.start_time) & m.eligible].dropna(subset=FEATS)
        ev = m[(m.date >= q.start_time) & (m.date <= q.end_time) & m.eligible].copy()
        if tr.empty or ev.empty:
            continue
        reg = XGBRegressor(**REG).fit(tr[FEATS], tr.next_move.clip(0, 0.25))
        ev["pred"] = reg.predict(ev[FEATS])
        parts.append(ev)
    oos = pd.concat(parts, ignore_index=True)
    oos["yr"] = oos.date.dt.year
    print(f"OOS rows={len(oos)} 2024-26 | mean next-day move={oos.next_move.mean()*100:.2f}% "
          f"median={oos.next_move.median()*100:.2f}%\n")

    print("=" * 70)
    print("MODEL vs atm_iv baseline — cross-sectional rank IC (Spearman, per-day mean)")
    print("=" * 70)
    print(f"  model pred : IC = {daily_ic(oos, 'pred'):.3f}")
    print(f"  atm_iv     : IC = {daily_ic(oos, 'atm_iv'):.3f}")
    for y in (2024, 2025, 2026):
        d = oos[oos.yr == y]
        print(f"    {y}: model IC {daily_ic(d, 'pred'):.3f} | atm_iv IC {daily_ic(d, 'atm_iv'):.3f}")

    print("\n" + "=" * 70)
    print("AUC for 'next-day move >= thr'  (model pred vs atm_iv)")
    print("=" * 70)
    for thr in (0.015, 0.02, 0.03, 0.04):
        y = (oos.next_move >= thr).astype(int)
        if y.nunique() < 2:
            continue
        print(f"  >={int(thr*1000)/10}% (base {y.mean()*100:4.1f}%): model AUC {roc_auc_score(y, oos.pred):.3f} | "
              f"atm_iv AUC {roc_auc_score(y, oos.atm_iv):.3f}")

    print("\n" + "=" * 70)
    print("Top-k mover precision within group (are top-pred the actual top movers next day?)")
    print("=" * 70)
    ab = oos[oos.group.notna()].copy()
    rows = []
    for scorer in ("pred", "atm_iv"):
        e = ab.copy()
        e["ar"] = e.groupby(["date", "group"])["next_move"].rank(ascending=False, method="first")
        e["sr"] = e.groupby(["date", "group"])[scorer].rank(ascending=False, method="first")
        e["daymax"] = e.groupby(["date", "group"])["next_move"].transform("max")
        t1, t3 = e[e.sr <= 1], e[e.sr <= 3]
        rows.append({"scorer": scorer,
                     "top1_in_top3%": round((t1.ar <= 3).mean() * 100, 1),
                     "top3_in_top3%": round((t3.ar <= 3).mean() * 100, 1),
                     "top3_in_top5%": round((t3.ar <= 5).mean() * 100, 1),
                     "avg_move_top1%": round(t1.next_move.mean() * 100, 2),
                     "capture_top1%": round((t1.next_move / t1.daymax).mean() * 100, 1)})
    print(pd.DataFrame(rows).to_string(index=False))

    print("\n" + "=" * 70)
    print("Calibration — predicted-move decile vs realized")
    print("=" * 70)
    oos["dec"] = pd.qcut(oos.pred.rank(method="first"), 10, labels=False)
    cal = oos.groupby("dec").agg(pred_mean=("pred", "mean"), realized_mean=("next_move", "mean"),
                                 n=("next_move", "size"))
    cal[["pred_mean", "realized_mean"]] = (cal[["pred_mean", "realized_mean"]] * 100).round(2)
    print(cal.to_string())

    imp = pd.Series(reg.feature_importances_, index=FEATS).sort_values(ascending=False)
    print("\ntop features:", ", ".join(f"{k}={v:.2f}" for k, v in imp.head(8).items()))
    oos[["date", "group", "symbol", "atm_iv", "pred", "next_move", "eligible"]].to_parquet(HERE / "next_move_oos.parquet")
    print(f"saved OOS -> next_move_oos.parquet")


if __name__ == "__main__":
    main()
