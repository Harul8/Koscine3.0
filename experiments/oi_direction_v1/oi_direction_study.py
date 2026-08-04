"""OI-positioning -> DIRECTION study (contained; reads data read-only; touches no PROD).

Tests the practitioner belief: do the 4 OI regimes predict forward direction?
  Long Buildup  (price^ OI^)  -> bullish continuation
  Short Buildup (price_ OI^)  -> bearish continuation
  Short Covering(price^ OI_)  -> up, often reversal-prone
  Long Unwinding(price_ OI_)  -> down, often reversal-prone
Computed from futures OI change + price, at 1-day and 5-day horizons. Target = forward 1d & 5d SIGNED direction.
Also: aggressiveness cut (large |OI z|), per-year stability, and a multivariate OI direction classifier (AUC vs 0.50).
Direction is a coin flip on EOD price/macro/candles (established) -> this checks if OI positioning is the exception.

    set PYTHONPATH=src && python experiments/oi_direction_v1/oi_direction_study.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from koscine3.data.sources import load_market_data  # noqa: E402

QUARTERS = pd.period_range("2024Q1", "2026Q2", freq="Q")


def regime(dp, doi):
    # dp, doi: arrays of price-change sign and OI-change sign
    r = np.full(len(dp), "flat", dtype=object)
    r[(dp > 0) & (doi > 0)] = "LongBuildup"
    r[(dp < 0) & (doi > 0)] = "ShortBuildup"
    r[(dp > 0) & (doi < 0)] = "ShortCovering"
    r[(dp < 0) & (doi < 0)] = "LongUnwinding"
    return r


def main():
    cols = ["date", "symbol", "close", "ret_1d", "ret_5d", "fut_chg_oi", "fut_oi_z_60d", "atm_iv"]
    m = load_market_data(columns=[c for c in cols])
    m["symbol"] = m["symbol"].astype(str); m["date"] = pd.to_datetime(m["date"])
    m = m.sort_values(["symbol", "date"]).reset_index(drop=True)
    g = m.groupby("symbol", sort=False)
    m["fwd1"] = g["close"].shift(-1) / m.close - 1
    m["fwd5"] = g["close"].shift(-5) / m.close - 1
    m["cum_oi_chg_5"] = g["fut_chg_oi"].transform(lambda s: s.rolling(5, min_periods=3).sum())
    m["reg1"] = regime(np.sign(m.ret_1d.values), np.sign(m.fut_chg_oi.values))
    m["reg5"] = regime(np.sign(m.ret_5d.values), np.sign(m.cum_oi_chg_5.values))
    e = m[(m.atm_iv.notna()) & (m.close >= 100) & m.fwd5.notna() & m.fut_chg_oi.notna()].copy()
    print(f"rows {len(e):,} | symbols {e.symbol.nunique()} | base P(up 1d) {(e.fwd1>0).mean():.3f}  P(up 5d) {(e.fwd5>0).mean():.3f}")

    bull = {"LongBuildup", "ShortCovering"}
    for col, lab in [("reg1", "1-DAY regime (price1d x dOI1d)"), ("reg5", "5-DAY regime (price5d x cumOI5d)")]:
        print(f"\n=== {lab} -> forward direction ===")
        print(f"{'regime':16s} {'n':>8s} {'P(up 1d)':>9s} {'P(up 5d)':>9s} {'mean fwd5%':>11s} {'expect':>9s}")
        for rg in ["LongBuildup", "ShortCovering", "ShortBuildup", "LongUnwinding"]:
            b = e[e[col] == rg]
            if b.empty:
                continue
            print(f"{rg:16s} {len(b):>8d} {(b.fwd1>0).mean():>9.3f} {(b.fwd5>0).mean():>9.3f} {b.fwd5.mean()*100:>11.2f} {'UP' if rg in bull else 'DOWN':>9s}")

    print("\n=== AGGRESSIVE cut (|fut_oi_z_60d| top tercile) — 5d regime ===")
    agg = e[e.fut_oi_z_60d.abs() >= e.fut_oi_z_60d.abs().quantile(0.667)]
    for rg in ["LongBuildup", "ShortCovering", "ShortBuildup", "LongUnwinding"]:
        b = agg[agg.reg5 == rg]
        if len(b) > 50:
            print(f"   {rg:16s} n={len(b):>6d}  P(up 5d) {(b.fwd5>0).mean():.3f}  mean fwd5% {b.fwd5.mean()*100:+.2f}")

    print("\n=== per-year P(up 5d) by 5d-regime (stability) ===")
    e["yr"] = e.date.dt.year; e["up5"] = (e.fwd5 > 0).astype(int)
    piv = e.pivot_table(index="reg5", columns="yr", values="up5", aggfunc="mean")
    print(piv.reindex(["LongBuildup", "ShortCovering", "ShortBuildup", "LongUnwinding"]).round(3).to_string())
    print("aggressive-only per-year (|OI z| top tercile):")
    pivA = e[e.fut_oi_z_60d.abs() >= e.fut_oi_z_60d.abs().quantile(0.667)].pivot_table(index="reg5", columns="yr", values="up5", aggfunc="mean")
    print(pivA.reindex(["LongBuildup", "ShortCovering", "ShortBuildup", "LongUnwinding"]).round(3).to_string())

    # multivariate: can OI features predict 5d direction beyond 0.50?
    print("\n=== multivariate OI direction classifier (5d), purged walk-forward ===")
    from catboost import CatBoostClassifier
    from sklearn.metrics import roc_auc_score
    feats = ["ret_1d", "ret_5d", "fut_chg_oi", "fut_oi_z_60d", "cum_oi_chg_5"]
    e["y5"] = (e.fwd5 > 0).astype(int)
    ev = []
    for q in QUARTERS:
        cut = q.start_time - pd.Timedelta(days=10)
        tr = e[(e.date < cut)].dropna(subset=feats)
        te = e[(e.date >= q.start_time) & (e.date <= q.end_time)].dropna(subset=feats)
        if len(tr) < 5000 or te.empty:
            continue
        clf = CatBoostClassifier(iterations=300, depth=4, learning_rate=0.03, verbose=False,
                                 allow_writing_files=False).fit(tr[feats], tr.y5)
        t = te.copy(); t["p"] = clf.predict_proba(te[feats])[:, 1]; ev.append(t)
    ev = pd.concat(ev, ignore_index=True)
    auc = roc_auc_score(ev.y5, ev.p)
    k = max(1, int(len(ev) * 0.1)); top = ev.nlargest(k, "p"); bot = ev.nsmallest(k, "p")
    print(f"   OOS AUC {auc:.4f} (0.50 = coin flip) | top-decile P(up) {top.y5.mean():.3f} | bottom-decile P(up) {bot.y5.mean():.3f}")
    print("\nVERDICT: if regimes' P(up) ~0.50 across the board and AUC ~0.50 -> OI positioning does NOT crack direction.")


if __name__ == "__main__":
    main()
