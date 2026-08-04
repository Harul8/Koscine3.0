"""megacap_direction_v1 / Phase 1 — characterize 1/2/3-day direction for the TOP mega-caps, Nifty, and the
top-10 basket. (contained; read-only; no PROD touch)

Questions:
  - At 1/2/3-day horizons, are mega-caps momentum or mean-reverting? (autocorr of returns, by period)
  - Which features carry SIGNED direction (IC) for the top-10 at 1/2/3d — and is it different in 2026?
  - Is the AGGREGATE (Nifty / equal-weight top-10 basket) more predictable than individual names?
  - Base rates + per-stock predictability.

    set PYTHONPATH=src && python experiments/megacap_direction_v1/characterize.py
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
CAND = ["ret_1d", "ret_3d", "ret_5d", "ret_20d", "gap_pct", "close_sma5_dist", "close_sma20_dist",
        "consec_up_days", "consec_down_days", "pos_day_share_10d", "rsi_proxy", "nifty_ret_1d", "nifty_ret_5d",
        "mkt_pct_above_sma20", "mkt_pct_above_sma50", "mkt_advance_ratio", "pcr_oi", "pcr_oi_chg_5",
        "oi_buildup_ratio", "fut_oi_z_60d", "price_oi_divergence", "iv_skew_norm", "iv_skew_chg_5d",
        "atm_iv_chg_5", "delivery_pct", "delivery_pct_chg_5", "vol_5v20_ratio", "bb_width_20", "adx_14", "di_diff"]


def ic(df, sig, fwd):
    s = df[[sig, fwd]].replace([np.inf, -np.inf], np.nan).dropna()
    return s[sig].corr(s[fwd], "spearman") if len(s) > 150 else np.nan


def main():
    m = load_market_data()
    m["symbol"] = m.symbol.astype(str); m["date"] = pd.to_datetime(m["date"])
    have = [s for s in TOP if s in set(m.symbol.unique())]
    print(f"TOP present ({len(have)}): {have}")
    e = m[m.symbol.isin(have)].sort_values(["symbol", "date"]).reset_index(drop=True)
    g = e.groupby("symbol", sort=False)
    for h in (1, 2, 3, 5):
        e[f"fwd{h}"] = g["close"].shift(-h) / e.close - 1.0
    # a simple overbought proxy if rsi not present
    if "rsi_proxy" in CAND and "rsi_proxy" not in e.columns:
        e["rsi_proxy"] = e["close_sma5_dist"] if "close_sma5_dist" in e.columns else np.nan
    e = e[(e.atm_iv.notna()) & (e.close >= 100)].copy()
    e["per"] = np.where(e.date < "2025-01-01", "pre2025", np.where(e.date < "2026-01-01", "2025", "2026"))
    avail = [c for c in CAND if c in e.columns]

    print(f"\nrows {len(e):,} | base P(up): " + " ".join(f"{h}d {(e[f'fwd{h}']>0).mean():.3f}" for h in (1, 2, 3, 5)))

    # 1) autocorrelation: momentum (+) vs mean-reversion (-) at each horizon, by period
    print("\n=== return autocorrelation (corr ret_Nd vs fwd_Nd) — +momentum / -mean-reversion ===")
    print(f"{'period':8s} {'1d(r1->f1)':>11s} {'2d(r2->f2)':>11s} {'3d(r3->f3)':>11s} {'5d(r5->f5)':>11s}")
    e["ret_2d"] = g["close"].transform(lambda s: s / s.shift(2) - 1)
    for per in ("pre2025", "2025", "2026"):
        d = e[e.per == per]
        row = []
        for h, rcol in ((1, "ret_1d"), (2, "ret_2d"), (3, "ret_3d"), (5, "ret_5d")):
            row.append(ic(d, rcol, f"fwd{h}"))
        print(f"{per:8s} " + " ".join(f"{v:>11.3f}" for v in row))

    # 2) signed IC of features for fwd1 / fwd3, by period (top movers of directional content)
    for h in (1, 3):
        print(f"\n=== signed IC vs fwd{h}d (TOP mega-caps), by period — top 8 by |2026 IC| ===")
        rows = []
        for sig in avail:
            r = {p: ic(e[e.per == p], sig, f"fwd{h}") for p in ("pre2025", "2025", "2026")}
            rows.append((sig, r))
        rows.sort(key=lambda x: -abs(x[1]["2026"] if np.isfinite(x[1]["2026"]) else 0))
        print(f"{'signal':20s} {'pre2025':>9s} {'2025':>9s} {'2026':>9s}")
        for sig, r in rows[:10]:
            print(f"{sig:20s} {r['pre2025']:>9.3f} {r['2025']:>9.3f} {r['2026']:>9.3f}")

    # 3) per-stock 1d & 3d autocorr in 2026 (which names are tradeable?)
    print("\n=== per-stock autocorr in 2026 (ret_1d->fwd1, ret_3d->fwd3) + base P(up1d) ===")
    d26 = e[e.per == "2026"]
    print(f"{'symbol':12s} {'n':>4s} {'P(up1)':>7s} {'ac1d':>7s} {'ac3d':>7s}")
    for sym in have:
        s = d26[d26.symbol == sym]
        if len(s) < 30:
            continue
        print(f"{sym:12s} {len(s):>4d} {(s.fwd1>0).mean():>7.3f} {ic(s,'ret_1d','fwd1'):>7.3f} {ic(s,'ret_3d','fwd3'):>7.3f}")

    # 4) AGGREGATE: Nifty next-day + equal-weight top-10 basket
    print("\n=== AGGREGATE predictability (Nifty index & equal-weight TOP basket) ===")
    nf = (m.dropna(subset=["nifty_ret_1d"]).groupby("date", as_index=False)["nifty_ret_1d"].first()
          .sort_values("date").reset_index(drop=True))
    nf["lr"] = np.log1p(nf.nifty_ret_1d)
    for h in (1, 2, 3):
        nf[f"fwd{h}"] = nf.lr[::-1].rolling(h).sum()[::-1].shift(-1)  # next h-day cum log ret
    nf["mom5"] = nf.lr.rolling(5).sum(); nf["mom1"] = nf.lr
    nf["per"] = np.where(nf.date < "2025-01-01", "pre2025", np.where(nf.date < "2026-01-01", "2025", "2026"))
    print("Nifty: autocorr mom1->fwd1, mom5->fwd1/2/3 by period")
    for per in ("pre2025", "2025", "2026"):
        d = nf[nf.per == per]
        print(f"  {per:8s} mom1->fwd1 {ic(d,'mom1','fwd1'):+.3f} | mom5->fwd1 {ic(d,'mom5','fwd1'):+.3f} "
              f"fwd2 {ic(d,'mom5','fwd2'):+.3f} fwd3 {ic(d,'mom5','fwd3'):+.3f} | base P(up1) {(d.fwd1>0).mean():.3f} n={len(d)}")
    # equal-weight basket
    bk = (e.groupby("date").agg(bret=("ret_1d", "mean")).reset_index().sort_values("date").reset_index(drop=True))
    bk["lr"] = np.log1p(bk.bret)
    bk["mom5"] = bk.lr.rolling(5).sum(); bk["mom1"] = bk.lr
    for h in (1, 2, 3):
        bk[f"fwd{h}"] = bk.lr[::-1].rolling(h).sum()[::-1].shift(-1)
    bk["per"] = np.where(bk.date < "2025-01-01", "pre2025", np.where(bk.date < "2026-01-01", "2025", "2026"))
    print("TOP-10 equal-weight basket: same")
    for per in ("pre2025", "2025", "2026"):
        d = bk[bk.per == per]
        print(f"  {per:8s} mom1->fwd1 {ic(d,'mom1','fwd1'):+.3f} | mom5->fwd1 {ic(d,'mom5','fwd1'):+.3f} "
              f"fwd2 {ic(d,'mom5','fwd2'):+.3f} fwd3 {ic(d,'mom5','fwd3'):+.3f} | base P(up1) {(d.fwd1>0).mean():.3f} n={len(d)}")


if __name__ == "__main__":
    main()
