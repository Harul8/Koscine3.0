"""megacap_direction_v1 / Phase 5 — NEXT-DAY OUTLOOK for Nifty + top-15: expected MOVE size (magnitude, the
predictable part) + a marginal P(up) (direction, ~coin flip). (contained; read-only; no PROD touch; CPU)

Validates the magnitude forecast (walk-forward rank-IC of predicted vs realized next-day |move|, vs the atm_iv
baseline), then prints the latest-date outlook table: expected ±move% and P(up) for Nifty, basket, and each name.

    set PYTHONPATH=src && python experiments/megacap_direction_v1/next_day_outlook.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from koscine3.data.sources import load_market_data  # noqa: E402

FII = ROOT / "data" / "silver" / "fii_dii_cash.parquet"
TOP = ["RELIANCE", "HDFCBANK", "ICICIBANK", "KOTAKBANK", "SBIN", "AXISBANK", "BAJFINANCE", "BAJAJFINSV",
       "MARUTI", "TCS", "INFY", "HINDUNILVR", "ITC", "LT", "BHARTIARTL"]
EVAL_MONTHS = pd.period_range("2024-01", "2026-06", freq="M")
MAGF = ["atm_iv", "realized_vol_20", "atr_pct_14", "bb_width_20", "abs_ret1", "abs_ret5", "vol_5v20_ratio",
        "gap_abs", "atm_iv_chg_5", "day_of_week", "range_pct"]
DIRF = ["ret_1d", "ret_5d", "ret_20d", "nifty_ret_1d", "nifty_ret_5d", "sector_ret_5d", "stock_rel_sector_ret_5d",
        "mkt_pct_above_sma50", "pcr_oi", "pcr_oi_chg_5", "iv_skew_norm", "gap_pct", "fii_net", "fii_net_5d"]
CBR = dict(iterations=400, depth=5, learning_rate=0.03, l2_leaf_reg=6.0, random_seed=7, allow_writing_files=False, verbose=False, thread_count=-1)


def main():
    from catboost import CatBoostRegressor, CatBoostClassifier
    m = load_market_data()
    m["symbol"] = m.symbol.astype(str); m["date"] = pd.to_datetime(m["date"])
    m = m.sort_values(["symbol", "date"]).reset_index(drop=True)
    g = m.groupby("symbol", sort=False)
    m["fwd1"] = g["close"].shift(-1) / m.close - 1.0
    m["mag"] = m["fwd1"].abs()                     # next-day move size
    m["ydir"] = (m.fwd1 > 0).astype(float); m.loc[m.fwd1.isna(), "ydir"] = np.nan
    m["abs_ret1"] = m.ret_1d.abs(); m["abs_ret5"] = m.ret_5d.abs(); m["gap_abs"] = m.gap_pct.abs()
    f = pd.read_parquet(FII)[["date", "fii_buy", "fii_sell", "fii_net"]]; f["date"] = pd.to_datetime(f.date); f = f.sort_values("date")
    f["fii_net_5d"] = f.fii_net.rolling(5).sum()
    m = m.merge(f[["date", "fii_net", "fii_net_5d"]], on="date", how="left")
    m["eligible"] = m.close.ge(100) & m.atm_iv.notna()
    have = [s for s in TOP if s in set(m.symbol.unique())]

    # ---- validate MAGNITUDE forecast (walk-forward, eval top-15) vs atm_iv baseline ----
    magf = [c for c in MAGF if c in m.columns]
    parts = []
    for mo in EVAL_MONTHS:
        ms, me = mo.start_time, mo.end_time; cut = ms - pd.Timedelta(days=2)
        tr = m[(m.date < cut) & m.eligible & m.mag.notna()]
        ev = m[(m.date >= ms) & (m.date <= me) & m.eligible & m.symbol.isin(have) & m.mag.notna()]
        if len(tr) < 5000 or ev.empty:
            continue
        mdl = CatBoostRegressor(**CBR, loss_function="RMSE").fit(tr[magf], tr.mag.clip(0, 0.2))
        e = ev[["date", "symbol", "mag", "atm_iv"]].copy(); e["pred"] = mdl.predict(ev[magf]); e["per"] = e.date.dt.year.astype(str)
        parts.append(e)
    mp = pd.concat(parts, ignore_index=True)
    def ic(d, a): return d[a].corr(d.mag, "spearman")
    print("=== MAGNITUDE (next-day |move|) forecast quality — rank-IC vs realized, eval top-15 ===")
    print(f"  ALL : model {ic(mp,'pred'):+.3f} | atm_iv baseline {ic(mp,'atm_iv'):+.3f}  n={len(mp)}")
    for per in ("2025", "2026"):
        d = mp[mp.per == per]
        if len(d) > 100:
            print(f"  {per}: model {ic(d,'pred'):+.3f} | atm_iv baseline {ic(d,'atm_iv'):+.3f}  n={len(d)}")

    # ---- marginal DIRECTION P(up) (we know ~coin flip) walk-forward, eval top-15 ----
    dirf = [c for c in DIRF if c in m.columns]
    dparts = []
    for mo in EVAL_MONTHS:
        ms, me = mo.start_time, mo.end_time; cut = ms - pd.Timedelta(days=2)
        tr = m[(m.date < cut) & m.eligible & m.ydir.notna()]
        ev = m[(m.date >= ms) & (m.date <= me) & m.eligible & m.symbol.isin(have) & m.ydir.notna()]
        if len(tr) < 5000 or ev.empty:
            continue
        mdl = CatBoostClassifier(**CBR, loss_function="Logloss").fit(tr[dirf], tr.ydir.astype(int))
        e = ev[["date", "symbol", "ydir", "fwd1"]].copy(); e["pup"] = mdl.predict_proba(ev[dirf])[:, 1]; e["per"] = e.date.dt.year.astype(str)
        dparts.append(e)
    dp = pd.concat(dparts, ignore_index=True)
    from sklearn.metrics import roc_auc_score
    print("\n=== DIRECTION P(up) next-day — eval top-15 (expect ~coin flip) ===")
    for per in ("ALL", "2025", "2026"):
        d = dp if per == "ALL" else dp[dp.per == per]
        if len(d) > 100 and d.ydir.nunique() > 1:
            print(f"  {per}: AUC {roc_auc_score(d.ydir,d.pup):.3f}  hit {(((d.pup>0.5)&(d.ydir==1))|((d.pup<=0.5)&(d.ydir==0))).mean():.3f}  n={len(d)}")

    # ---- LATEST-date outlook ----
    last = m[m.eligible & m.symbol.isin(have)].date.max()
    print(f"\n=== NEXT-DAY OUTLOOK (from {last.date()}) — expected move & direction ===")
    # train final models on all data < last (embargo 2d)
    cut = last - pd.Timedelta(days=2)
    trm = m[(m.date < cut) & m.eligible & m.mag.notna()]
    magmdl = CatBoostRegressor(**CBR, loss_function="RMSE").fit(trm[magf], trm.mag.clip(0, 0.2))
    dirmdl = CatBoostClassifier(**CBR, loss_function="Logloss").fit(trm[dirf], trm.ydir.astype(int))
    cur = m[(m.date == last) & m.symbol.isin(have)].copy()
    cur["exp_move_pct"] = magmdl.predict(cur[magf]) * 100
    cur["iv_implied_pct"] = cur.atm_iv / np.sqrt(252) * 100
    cur["p_up"] = dirmdl.predict_proba(cur[dirf])[:, 1]
    cur = cur.sort_values("exp_move_pct", ascending=False)
    print(f"{'symbol':12s} {'exp_move%':>9s} {'iv_impl%':>9s} {'P(up)':>6s} {'lean':>5s}")
    for r in cur.itertuples():
        lean = "CALL" if r.p_up >= 0.54 else ("PUT" if r.p_up <= 0.46 else "~flat")
        print(f"{r.symbol:12s} {r.exp_move_pct:>9.2f} {r.iv_implied_pct:>9.2f} {r.p_up:>6.3f} {lean:>5s}")
    cur[["symbol", "exp_move_pct", "iv_implied_pct", "p_up"]].to_csv(ROOT / "experiments/megacap_direction_v1/next_day_outlook.csv", index=False)
    print(f"\n(top-15 basket avg expected move ≈ {cur.exp_move_pct.mean():.2f}% ; avg P(up) {cur.p_up.mean():.3f})")
    print("saved next_day_outlook.csv")


if __name__ == "__main__":
    main()
