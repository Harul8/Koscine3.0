"""Deep direction diagnostic — find ANY pocket where direction beats ~0.52 AUC.

Broad F&O universe, 2022-26. Tests: (1) absolute close-direction AUC by horizon 1/2/3/5/10d,
(2) RELATIVE (cross-sectional, market-neutral) direction, (3) turnover interaction (reversal vs momentum),
(4) volatility-regime conditioning, (5) PEAD (post-earnings drift). Univariate AUC = no fitting, no leakage
(signals at t, outcomes forward).
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from koscine3.data.sources import load_market_data

pd.set_option("display.width", 220)
COLS = ["date", "symbol", "open", "high", "low", "close", "atm_iv", "atm_ce_iv", "atm_pe_iv",
        "realized_vol_20", "ret_1d", "ret_3d", "ret_5d", "ret_10d", "ret_20d", "ret_5d_cs_rank",
        "ret_20d_cs_rank", "close_sma20_dist", "close_sma50_dist", "adx_14", "gap_pct",
        "days_to_earnings", "earnings_within_5d", "turnover_ratio_20", "vol_sma20_ratio",
        "fut_chg_oi", "delivery_pct_chg_5", "rel_ret_5d_vs_nifty", "mkt_pct_above_sma50"]


def auc(df, ycol, xcol):
    d = df[[ycol, xcol]].replace([np.inf, -np.inf], np.nan).dropna()
    if d[ycol].nunique() < 2 or len(d) < 500:
        return np.nan, len(d)
    return roc_auc_score(d[ycol], d[xcol]), len(d)


def main():
    m = load_market_data(columns=COLS)
    m["symbol"] = m["symbol"].astype(str)
    m = m.sort_values(["symbol", "date"]).reset_index(drop=True)
    g = m.groupby("symbol", sort=False)
    for h in (1, 2, 3, 5, 10):
        m[f"fwd{h}"] = g["close"].shift(-h) / m["close"] - 1.0
        m[f"dir{h}"] = (m[f"fwd{h}"] > 0).astype(float)
    for h in (1, 5, 10):
        mk = m.groupby("date")[f"fwd{h}"].transform("mean")
        m[f"reldir{h}"] = (m[f"fwd{h}"] - mk > 0).astype(float)
    m["vol_spread"] = m.atm_ce_iv - m.atm_pe_iv
    hi252 = g["close"].transform(lambda s: s.rolling(252, min_periods=120).max())
    m["dist_52wh"] = m.close / hi252 - 1.0
    m["rev_1d"] = -m.ret_1d
    m["rev_5d"] = -m.ret_5d
    m = m[(m.close >= 100) & (m.date.dt.year >= 2022)].copy()
    print(f"rows={len(m)} symbols={m.symbol.nunique()} {m.date.min().date()}..{m.date.max().date()}")

    print("\n" + "=" * 78)
    print("(1) ABSOLUTE close-direction AUC by horizon  (base rate = P(up))")
    print("=" * 78)
    sigs = ["ret_5d", "ret_20d", "ret_20d_cs_rank", "close_sma50_dist", "dist_52wh", "adx_14",
            "rev_1d", "rev_5d", "vol_spread", "fut_chg_oi", "delivery_pct_chg_5", "gap_pct"]
    rows = []
    for h in (1, 2, 3, 5, 10):
        base = m[f"dir{h}"].mean()
        r = {"horizon": f"{h}d", "P(up)": round(base, 3)}
        for s in sigs:
            a, _ = auc(m, f"dir{h}", s)
            r[s] = round(a, 3)
        rows.append(r)
    print(pd.DataFrame(rows).to_string(index=False))

    print("\n" + "=" * 78)
    print("(2) RELATIVE (cross-sectional, market-neutral) direction AUC")
    print("=" * 78)
    rows = []
    for h in (1, 5, 10):
        r = {"horizon": f"{h}d", "P(rel up)": round(m[f"reldir{h}"].mean(), 3)}
        for s in ["ret_20d_cs_rank", "ret_5d_cs_rank", "ret_20d", "close_sma50_dist", "dist_52wh", "rev_5d"]:
            a, _ = auc(m, f"reldir{h}", s)
            r[s] = round(a, 3)
        rows.append(r)
    print(pd.DataFrame(rows).to_string(index=False))

    print("\n" + "=" * 78)
    print("(3) TURNOVER interaction — reversal(rev_1d) vs momentum(ret_20d), h=1 & h=5")
    print("=" * 78)
    m["turn_b"] = pd.qcut(m.turnover_ratio_20.rank(method="first"), 3, labels=["low", "mid", "high"])
    rows = []
    for tb in ["low", "mid", "high"]:
        d = m[m.turn_b == tb]
        rows.append({"turnover": tb,
                     "rev_1d->dir1": round(auc(d, "dir1", "rev_1d")[0], 3),
                     "ret_20d->dir1": round(auc(d, "dir1", "ret_20d")[0], 3),
                     "rev_1d->dir5": round(auc(d, "dir5", "rev_1d")[0], 3),
                     "ret_20d->dir5": round(auc(d, "dir5", "ret_20d")[0], 3)})
    print(pd.DataFrame(rows).to_string(index=False))

    print("\n" + "=" * 78)
    print("(4) VOLATILITY-REGIME conditioning — direction AUC in low vs high IV (h=5)")
    print("=" * 78)
    m["iv_b"] = pd.qcut(m.atm_iv.rank(method="first"), 3, labels=["low_iv", "mid_iv", "high_iv"])
    rows = []
    for ib in ["low_iv", "mid_iv", "high_iv"]:
        d = m[m.iv_b == ib]
        rows.append({"iv_regime": ib,
                     "ret_20d->dir5": round(auc(d, "dir5", "ret_20d")[0], 3),
                     "ret_20d_cs_rank->dir5": round(auc(d, "dir5", "ret_20d_cs_rank")[0], 3),
                     "rev_5d->dir5": round(auc(d, "dir5", "rev_5d")[0], 3),
                     "close_sma50_dist->dir5": round(auc(d, "dir5", "close_sma50_dist")[0], 3)})
    print(pd.DataFrame(rows).to_string(index=False))

    print("\n" + "=" * 78)
    print("(5) PEAD — post-earnings drift (earnings within last/next 5d): gap & momentum -> dir5")
    print("=" * 78)
    em = m[m.earnings_within_5d.fillna(0) > 0]
    print(f"earnings-window rows={len(em)}  P(up,5d)={em.dir5.mean():.3f}")
    for s in ["gap_pct", "ret_5d", "ret_1d", "ret_20d"]:
        a, n = auc(em, "dir5", s)
        print(f"  {s:14s} AUC={a:.3f}  n={n}")


if __name__ == "__main__":
    main()
