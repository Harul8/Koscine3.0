"""megacap_direction_v1 / Phase 4 — data-driven REGROUPING of the 65 universe by behaviour
(vol / IV / beta / move magnitude / direction-predictability). (contained; read-only; no PROD touch)

Goal: instead of A=mcap30 / B=turn35, propose groups that separate where DIRECTION is predictable (high-beta,
momentum-responsive) from where it is not (efficient mega-caps), and where MAGNITUDE is large. KMeans on the
behaviour profile + per-stock 2026 momentum IC.

    set PYTHONPATH=src && python experiments/megacap_direction_v1/regroup.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from koscine3.data.sources import load_market_data  # noqa: E402

LOCK_V2 = ROOT / "locks" / "prod_largemove_v2"
TOP = {"RELIANCE", "HDFCBANK", "ICICIBANK", "KOTAKBANK", "SBIN", "AXISBANK", "BAJFINANCE", "BAJAJFINSV",
       "MARUTI", "TCS", "INFY", "HINDUNILVR", "ITC", "LT", "BHARTIARTL"}


def sic(s, a, b):
    x = s[[a, b]].replace([np.inf, -np.inf], np.nan).dropna()
    return x[a].corr(x[b], "spearman") if len(x) > 60 else np.nan


def main():
    g2 = {s: g for g, syms in json.loads((LOCK_V2 / "universe_groups.json").read_text()).items() for s in syms}
    m = load_market_data(columns=["date", "symbol", "close", "high", "low", "ret_1d", "ret_5d", "atm_iv",
                                  "realized_vol_20", "nifty_ret_1d", "turnover_lacs"])
    m["symbol"] = m.symbol.astype(str); m["date"] = pd.to_datetime(m["date"])
    m = m[m.symbol.isin(g2)].sort_values(["symbol", "date"]).reset_index(drop=True)
    g = m.groupby("symbol", sort=False)
    hi = pd.concat([g["high"].shift(-i) for i in range(1, 6)], axis=1).max(axis=1)
    lo = pd.concat([g["low"].shift(-i) for i in range(1, 6)], axis=1).min(axis=1)
    m["move5"] = np.maximum(hi / m.close - 1, m.close / lo - 1)
    m["fwd5"] = g["close"].shift(-5) / m.close - 1
    m["per"] = np.where(m.date < "2026-01-01", "hist", "2026")

    rows = []
    for sym, s in m.groupby("symbol"):
        s26 = s[s.per == "2026"]
        v = s.nifty_ret_1d.var()
        beta = (s[["ret_1d", "nifty_ret_1d"]].cov().iloc[0, 1] / v) if v and v > 0 else np.nan
        rows.append(dict(symbol=sym, group=g2[sym], top=sym in TOP, n=len(s),
                         vol=s.realized_vol_20.mean(), iv=s.atm_iv.mean(), beta=beta,
                         move5=s.move5.mean(), turn=s.turnover_lacs.mean(),
                         mom_ic_hist=sic(s[s.per == "hist"], "ret_5d", "fwd5"),
                         mom_ic_2026=sic(s26, "ret_5d", "fwd5"), upr_2026=(s26.fwd5 > 0).mean()))
    d = pd.DataFrame(rows)

    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    X = d[["vol", "iv", "beta", "move5"]].fillna(d[["vol", "iv", "beta", "move5"]].median())
    Xs = StandardScaler().fit_transform(X)
    for k in (2, 3):
        d[f"k{k}"] = KMeans(n_clusters=k, random_state=7, n_init=10).fit_predict(Xs)

    print(f"universe {len(d)} | TOP-in-universe {d.top.sum()}")
    print("\n=== current A/B vs behaviour ===")
    print(d.groupby("group")[["vol", "iv", "beta", "move5", "mom_ic_2026", "upr_2026"]].mean().round(3).to_string())

    print("\n=== KMeans k=2 clusters (on vol/iv/beta/move5) ===")
    print(d.groupby("k2").agg(n=("symbol", "size"), vol=("vol", "mean"), iv=("iv", "mean"), beta=("beta", "mean"),
                              move5=("move5", "mean"), mom_ic26=("mom_ic_2026", "mean"),
                              n_top=("top", "sum")).round(3).to_string())
    print("\n=== KMeans k=3 clusters ===")
    print(d.groupby("k3").agg(n=("symbol", "size"), vol=("vol", "mean"), iv=("iv", "mean"), beta=("beta", "mean"),
                              move5=("move5", "mean"), mom_ic26=("mom_ic_2026", "mean"),
                              n_top=("top", "sum")).round(3).to_string())

    # rank by 2026 direction predictability (momentum IC) to see where direction lives
    print("\n=== most direction-predictable in 2026 (top/bottom by mom_ic_2026) ===")
    dd = d.dropna(subset=["mom_ic_2026"]).sort_values("mom_ic_2026", ascending=False)
    cols = ["symbol", "group", "top", "vol", "iv", "beta", "move5", "mom_ic_2026", "upr_2026"]
    print("TOP 12:\n" + dd.head(12)[cols].round(3).to_string(index=False))
    print("\nBOTTOM 8:\n" + dd.tail(8)[cols].round(3).to_string(index=False))

    # proposed split: high-beta/high-move ("traders", direction-tilt candidates) vs low ("efficient", agnostic)
    d["proposed"] = np.where((d.beta >= d.beta.median()) & (d.move5 >= d.move5.median()), "G1_highbeta_move", "G2_lowbeta")
    print("\n=== proposed 2-group split: beta & move5 above median = G1 ===")
    print(d.groupby("proposed").agg(n=("symbol", "size"), beta=("beta", "mean"), move5=("move5", "mean"),
                                    iv=("iv", "mean"), mom_ic26=("mom_ic_2026", "mean"),
                                    n_top=("top", "sum")).round(3).to_string())
    d.sort_values(["k2", "beta"]).to_csv(ROOT / "experiments/megacap_direction_v1/stock_profile.csv", index=False)
    print("\nsaved stock_profile.csv")


if __name__ == "__main__":
    main()
