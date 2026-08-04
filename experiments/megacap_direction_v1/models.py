"""megacap_direction_v1 / Phase 2 — DIRECTION models for 1/2/3-day on (a) individual top-15 mega-caps and
(b) the AGGREGATE (Nifty index & equal-weight top-15 basket). (contained; no PROD touch)

Walk-forward (expanding, monthly retrain, embargo by horizon). Reports OOS AUC / hit / IC by horizon and period,
plus index-model feature importance. Train: ALL eligible for individual; index frame for the aggregate.

    set PYTHONPATH=src && python experiments/megacap_direction_v1/models.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from koscine3.data.sources import load_market_data  # noqa: E402

TOP = ["RELIANCE", "HDFCBANK", "ICICIBANK", "KOTAKBANK", "SBIN", "AXISBANK", "BAJFINANCE", "BAJAJFINSV",
       "MARUTI", "TCS", "INFY", "HINDUNILVR", "ITC", "LT", "BHARTIARTL"]
EVAL_MONTHS = pd.period_range("2024-01", "2026-06", freq="M")
LEAK = ("future", "fwd", "next", "ahead", "label", "adverse", "up_move", "down_move", "expansion",
        "volclean", "outcome", "entry_1d", "_date", "tomorrow")
ID = {"date", "symbol", "open", "high", "low", "close", "last", "prev_close", "volume", "group",
      "eligible", "y", "fwd_ret"}
CB = dict(iterations=350, depth=5, learning_rate=0.03, l2_leaf_reg=6.0, random_seed=7,
          allow_writing_files=False, verbose=False, task_type="GPU", devices="0")


def auc_hit_ic(d, ycol="y", pcol="p", retcol="fwd_ret"):
    from sklearn.metrics import roc_auc_score
    d = d.dropna(subset=[pcol, ycol])
    if len(d) < 60 or d[ycol].nunique() < 2:
        return None
    call = d[pcol] > 0.5
    hit = float(((call & (d[ycol] == 1)) | (~call & (d[ycol] == 0))).mean())
    ic = float(d[pcol].corr(d[retcol], "spearman")) if retcol in d else np.nan
    return dict(auc=round(roc_auc_score(d[ycol], d[pcol]), 4), hit=round(hit, 4), ic=round(ic, 4), n=int(len(d)))


def wf_individual(m, feats, horizon, eval_syms):
    """Expanding monthly walk-forward; train all eligible, eval the given symbols. Returns preds."""
    from catboost import CatBoostClassifier
    emb = horizon + 1
    parts = []
    for mo in EVAL_MONTHS:
        ms, me = mo.start_time, mo.end_time
        cut = ms - pd.Timedelta(days=emb)
        tr = m[(m.date < cut) & m.eligible & m[f"y{horizon}"].notna()]
        ev = m[(m.date >= ms) & (m.date <= me) & m.eligible & m.symbol.isin(eval_syms) & m[f"y{horizon}"].notna()]
        if len(tr) < 5000 or ev.empty:
            continue
        mdl = CatBoostClassifier(**CB, loss_function="Logloss").fit(tr[feats], tr[f"y{horizon}"].astype(int))
        e = ev[["date", "symbol", f"y{horizon}", f"fwd{horizon}"]].copy()
        e.columns = ["date", "symbol", "y", "fwd_ret"]; e["p"] = mdl.predict_proba(ev[feats])[:, 1]
        e["per"] = e.date.dt.year.astype(str); parts.append(e)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def build_index_frame(m):
    g = m.groupby("date")
    idx = g.agg(nifty_ret_1d=("nifty_ret_1d", "first"), nifty_ret_5d=("nifty_ret_5d", "first"),
               nifty_rvol=("nifty_realized_vol_20", "first"), breadth20=("mkt_pct_above_sma20", "first"),
               breadth50=("mkt_pct_above_sma50", "first"), adv_ratio=("mkt_advance_ratio", "first"),
               avg_pcr=("pcr_oi", "mean"), avg_pcr_chg=("pcr_oi_chg_5", "mean"), avg_skew=("iv_skew_norm", "mean"),
               avg_gap=("gap_pct", "mean"), avg_deliv_chg=("delivery_pct_chg_5", "mean"),
               avg_oiz=("fut_oi_z_60d", "mean"), avg_atm_iv=("atm_iv", "mean"),
               avg_atm_iv_chg=("atm_iv_chg_5", "mean"), dow=("day_of_week", "first"), month=("month", "first")).reset_index()
    idx = idx.dropna(subset=["nifty_ret_1d"]).sort_values("date").reset_index(drop=True)
    idx["lr"] = np.log1p(idx.nifty_ret_1d)
    idx["mom2"] = idx.lr.rolling(2).sum(); idx["mom3"] = idx.lr.rolling(3).sum(); idx["mom10"] = idx.lr.rolling(10).sum()
    for h in (1, 2, 3):
        idx[f"f{h}"] = sum(idx.lr.shift(-k) for k in range(1, h + 1))
        idx[f"y{h}"] = (idx[f"f{h}"] > 0).astype(float); idx.loc[idx[f"f{h}"].isna(), f"y{h}"] = np.nan
    return idx


def wf_index(idx, feats, horizon, label):
    from catboost import CatBoostClassifier
    emb = horizon + 1
    parts = []
    for mo in EVAL_MONTHS:
        ms, me = mo.start_time, mo.end_time
        cut = ms - pd.Timedelta(days=emb)
        tr = idx[(idx.date < cut) & idx[f"y{horizon}"].notna()].dropna(subset=feats)
        ev = idx[(idx.date >= ms) & (idx.date <= me) & idx[f"y{horizon}"].notna()].dropna(subset=feats)
        if len(tr) < 800 or ev.empty:
            continue
        mdl = CatBoostClassifier(**CB, loss_function="Logloss").fit(tr[feats], tr[f"y{horizon}"].astype(int))
        e = ev[["date", f"y{horizon}", f"f{horizon}"]].copy()
        e.columns = ["date", "y", "fwd_ret"]; e["p"] = mdl.predict_proba(ev[feats])[:, 1]
        e["per"] = e.date.dt.year.astype(str); parts.append(e)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def main():
    m = load_market_data()
    m["symbol"] = m.symbol.astype(str); m["date"] = pd.to_datetime(m["date"])
    m = m.sort_values(["symbol", "date"]).reset_index(drop=True)
    g = m.groupby("symbol", sort=False)
    for h in (1, 2, 3):
        fr = sum(np.log1p(g["close"].shift(-k) / g["close"].shift(-k + 1) - 1) for k in range(1, h + 1)) if False else None
    # forward returns (simple cumulative) per stock
    for h in (1, 2, 3):
        m[f"fwd{h}"] = g["close"].shift(-h) / m.close - 1.0
        m[f"y{h}"] = (m[f"fwd{h}"] > 0).astype(float); m.loc[m[f"fwd{h}"].isna(), f"y{h}"] = np.nan
    m["eligible"] = m.close.ge(100.0) & m.atm_iv.notna()
    feats = [c for c in m.columns if c not in ID and not c.startswith(("y1", "y2", "y3", "fwd"))
             and pd.api.types.is_numeric_dtype(m[c]) and not any(h in c.lower() for h in LEAK)]
    have = [s for s in TOP if s in set(m.symbol.unique())]

    print(f"feats {len(feats)} | top {len(have)}")
    print("\n=== INDIVIDUAL top-15 mega-caps (expanding monthly WF, train all eligible) ===")
    print(f"{'horizon':8s} | {'ALL auc/hit/ic':>22s} | {'2026 auc/hit/ic':>22s}")
    for h in (1, 2, 3):
        pr = wf_individual(m, feats, h, set(have))
        if pr.empty:
            continue
        a, a26 = auc_hit_ic(pr), auc_hit_ic(pr[pr.per == "2026"])
        f = lambda x: f"{x['auc']:.3f}/{x['hit']:.3f}/{x['ic']:+.3f}" if x else "  -  "
        print(f"{h}d       | {f(a):>22s} | {f(a26):>22s}   n={a['n'] if a else 0}")

    # AGGREGATE
    idx = build_index_frame(m)
    idxf = ["nifty_ret_1d", "nifty_ret_5d", "nifty_rvol", "breadth20", "breadth50", "adv_ratio", "avg_pcr",
            "avg_pcr_chg", "avg_skew", "avg_gap", "avg_deliv_chg", "avg_oiz", "avg_atm_iv", "avg_atm_iv_chg",
            "dow", "month", "mom2", "mom3", "mom10"]
    print("\n=== AGGREGATE — Nifty index next 1/2/3d direction (expanding monthly WF) ===")
    print(f"{'horizon':8s} | {'ALL auc/hit/ic':>22s} | {'2025 auc/hit':>14s} | {'2026 auc/hit/ic':>22s}")
    last_pr = None
    for h in (1, 2, 3):
        pr = wf_index(idx, idxf, h, "nifty")
        if pr.empty:
            continue
        a, a25, a26 = auc_hit_ic(pr), auc_hit_ic(pr[pr.per == "2025"]), auc_hit_ic(pr[pr.per == "2026"])
        f = lambda x: f"{x['auc']:.3f}/{x['hit']:.3f}/{x['ic']:+.3f}" if x else "  -  "
        f2 = lambda x: f"{x['auc']:.3f}/{x['hit']:.3f}" if x else "  -  "
        print(f"{h}d       | {f(a):>22s} | {f2(a25):>14s} | {f(a26):>22s}   n={a['n'] if a else 0}")
        if h == 1:
            last_pr = pr

    # index feature importance (fwd1, trained <=2025)
    from catboost import CatBoostClassifier
    tr = idx[(idx.date < "2026-01-01") & idx.y1.notna()].dropna(subset=idxf)
    mdl = CatBoostClassifier(**CB, loss_function="Logloss").fit(tr[idxf], tr.y1.astype(int))
    imp = pd.Series(mdl.feature_importances_, index=idxf).sort_values(ascending=False)
    print("\nNifty fwd1 model — top features:")
    print(imp.head(10).round(2).to_string())
    if last_pr is not None:
        last_pr.to_csv(ROOT / "experiments/megacap_direction_v1/nifty_fwd1_preds.csv", index=False)
        print("\nsaved nifty_fwd1_preds.csv")


if __name__ == "__main__":
    main()
