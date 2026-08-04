"""megacap_direction_v1 / Phase 3 — AGGREGATE next 1/2/3-day direction with FII flow features.
(contained; read-only; no PROD touch; CPU to avoid GPU contention)

Targets the user's core ask: next-day movement of Nifty and the top-10 basket. Tests whether daily FII net cash
flow (silver/fii_dii_cash.parquet, the dominant 2026 driver per the research) improves aggregate direction over a
market/momentum/breadth/PCR/IV base. Walk-forward monthly expanding 2024-2026.

    set PYTHONPATH=src && python experiments/megacap_direction_v1/aggregate_fii.py
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
CB = dict(iterations=400, depth=4, learning_rate=0.03, l2_leaf_reg=6.0, random_seed=7,
          allow_writing_files=False, verbose=False, thread_count=-1)   # CPU
BASE = ["nifty_ret_1d", "nifty_ret_5d", "nifty_rvol", "breadth20", "breadth50", "adv_ratio", "avg_pcr",
        "avg_pcr_chg", "avg_skew", "avg_gap", "avg_deliv_chg", "avg_oiz", "avg_atm_iv", "avg_atm_iv_chg",
        "dow", "month", "mom2", "mom3", "mom10"]
FIIF = ["fii_net", "fii_net_3d", "fii_net_5d", "fii_net_20d", "fii_net_z60", "fii_ratio", "fii_streak"]


def auc_hit_ic(d):
    from sklearn.metrics import roc_auc_score
    d = d.dropna(subset=["p", "y"])
    if len(d) < 40 or d.y.nunique() < 2:
        return None
    call = d.p > 0.5
    hit = float(((call & (d.y == 1)) | (~call & (d.y == 0))).mean())
    return dict(auc=round(roc_auc_score(d.y, d.p), 4), hit=round(hit, 4),
                ic=round(float(d.p.corr(d.fwd, "spearman")), 4), n=int(len(d)))


def build_frame(target="nifty"):
    m = load_market_data()
    m["symbol"] = m.symbol.astype(str); m["date"] = pd.to_datetime(m["date"])
    g = m.groupby("date")
    idx = g.agg(nifty_ret_1d=("nifty_ret_1d", "first"), nifty_ret_5d=("nifty_ret_5d", "first"),
               nifty_rvol=("nifty_realized_vol_20", "first"), breadth20=("mkt_pct_above_sma20", "first"),
               breadth50=("mkt_pct_above_sma50", "first"), adv_ratio=("mkt_advance_ratio", "first"),
               avg_pcr=("pcr_oi", "mean"), avg_pcr_chg=("pcr_oi_chg_5", "mean"), avg_skew=("iv_skew_norm", "mean"),
               avg_gap=("gap_pct", "mean"), avg_deliv_chg=("delivery_pct_chg_5", "mean"),
               avg_oiz=("fut_oi_z_60d", "mean"), avg_atm_iv=("atm_iv", "mean"),
               avg_atm_iv_chg=("atm_iv_chg_5", "mean"), dow=("day_of_week", "first"), month=("month", "first")).reset_index()
    # basket return = equal-weight top-15 ret_1d
    bk = m[m.symbol.isin(TOP)].groupby("date")["ret_1d"].mean().rename("basket_ret").reset_index()
    idx = idx.merge(bk, on="date", how="left").dropna(subset=["nifty_ret_1d"]).sort_values("date").reset_index(drop=True)
    idx["ret"] = idx.nifty_ret_1d if target == "nifty" else idx.basket_ret
    idx["lr"] = np.log1p(idx.ret)
    idx["mom2"] = idx.lr.rolling(2).sum(); idx["mom3"] = idx.lr.rolling(3).sum(); idx["mom10"] = idx.lr.rolling(10).sum()
    # FII features
    f = pd.read_parquet(FII)[["date", "fii_buy", "fii_sell", "fii_net"]]
    f["date"] = pd.to_datetime(f["date"]); f = f.sort_values("date")
    f["fii_net_3d"] = f.fii_net.rolling(3).sum(); f["fii_net_5d"] = f.fii_net.rolling(5).sum()
    f["fii_net_20d"] = f.fii_net.rolling(20).sum()
    f["fii_net_z60"] = (f.fii_net - f.fii_net.rolling(60).mean()) / f.fii_net.rolling(60).std()
    f["fii_ratio"] = f.fii_buy / (f.fii_sell.abs() + 1.0) - 1.0
    sign = np.sign(f.fii_net.fillna(0))
    f["fii_streak"] = sign.groupby((sign != sign.shift()).cumsum()).cumcount().add(1) * sign
    idx = idx.merge(f[["date"] + FIIF], on="date", how="left")
    for h in (1, 2, 3):
        idx[f"f{h}"] = sum(idx.lr.shift(-k) for k in range(1, h + 1))
        idx[f"y{h}"] = (idx[f"f{h}"] > 0).astype(float); idx.loc[idx[f"f{h}"].isna(), f"y{h}"] = np.nan
    return idx


def wf(idx, feats, h):
    from catboost import CatBoostClassifier
    parts = []
    for mo in EVAL_MONTHS:
        ms, me = mo.start_time, mo.end_time
        cut = ms - pd.Timedelta(days=h + 1)
        tr = idx[(idx.date < cut) & idx[f"y{h}"].notna()].dropna(subset=feats)
        ev = idx[(idx.date >= ms) & (idx.date <= me) & idx[f"y{h}"].notna()].dropna(subset=feats)
        if len(tr) < 800 or ev.empty:
            continue
        mdl = CatBoostClassifier(**CB, loss_function="Logloss").fit(tr[feats], tr[f"y{h}"].astype(int))
        e = ev[["date", f"y{h}", f"f{h}"]].copy(); e.columns = ["date", "y", "fwd"]
        e["p"] = mdl.predict_proba(ev[feats])[:, 1]; e["per"] = e.date.dt.year.astype(str)
        parts.append(e)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def run(target):
    idx = build_frame(target)
    print(f"\n################  {target.upper()}  (rows {len(idx)}, FII coverage {idx.fii_net.notna().mean():.0%})  ################")
    for name, feats in [("BASE", BASE), ("BASE+FII", BASE + FIIF), ("FII-only", FIIF)]:
        print(f"\n--- {name} ({len(feats)} feats) ---")
        print(f"{'h':3s} | {'ALL auc/hit/ic':>22s} | {'2025 auc/hit':>13s} | {'2026 auc/hit/ic':>22s}")
        for h in (1, 2, 3):
            pr = wf(idx, feats, h)
            if pr.empty:
                continue
            a, a25, a26 = auc_hit_ic(pr), auc_hit_ic(pr[pr.per == "2025"]), auc_hit_ic(pr[pr.per == "2026"])
            f = lambda x: f"{x['auc']:.3f}/{x['hit']:.3f}/{x['ic']:+.3f}" if x else "  -  "
            f2 = lambda x: f"{x['auc']:.3f}/{x['hit']:.3f}" if x else "  -  "
            print(f"{h}d  | {f(a):>22s} | {f2(a25):>13s} | {f(a26):>22s}  n={a['n'] if a else 0}")
    # FII importance for fwd1
    from catboost import CatBoostClassifier
    tr = idx[(idx.date < "2026-01-01") & idx.y1.notna()].dropna(subset=BASE + FIIF)
    mdl = CatBoostClassifier(**CB, loss_function="Logloss").fit(tr[BASE + FIIF], tr.y1.astype(int))
    imp = pd.Series(mdl.feature_importances_, index=BASE + FIIF).sort_values(ascending=False)
    print(f"\n{target} fwd1 BASE+FII top features:\n{imp.head(10).round(2).to_string()}")


def main():
    for t in ("nifty", "basket"):
        run(t)


if __name__ == "__main__":
    main()
