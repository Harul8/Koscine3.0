"""Instant next-day snapshot (no training): expected move size (atm_iv-implied + trailing realized) and a
direction context (momentum/sector/FII) for Nifty + top-15, as of the latest data date. (read-only; no PROD touch)

    set PYTHONPATH=src && python experiments/megacap_direction_v1/latest_snapshot.py
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


def main():
    m = load_market_data(columns=["date", "symbol", "close", "ret_1d", "ret_5d", "ret_20d", "atm_iv",
                                  "realized_vol_20", "nifty_ret_1d", "nifty_ret_5d", "nifty_realized_vol_20",
                                  "sector_ret_5d", "stock_rel_sector_ret_5d", "pcr_oi", "iv_skew_norm"])
    m["symbol"] = m.symbol.astype(str); m["date"] = pd.to_datetime(m["date"])
    m = m.sort_values(["symbol", "date"]).reset_index(drop=True)
    g = m.groupby("symbol", sort=False)
    m["realmove20"] = g["ret_1d"].transform(lambda s: s.abs().rolling(20).mean())  # avg daily |move| last 20d
    last = m[m.symbol.isin(TOP) & m.atm_iv.notna()].date.max()
    f = pd.read_parquet(FII)[["date", "fii_net"]]; f["date"] = pd.to_datetime(f.date)
    fii_recent = f[f.date <= last].tail(5)

    print(f"=== NEXT-DAY SNAPSHOT as of {last.date()} ===")
    nrv = m[m.date == last]["nifty_realized_vol_20"].dropna()
    nmom5 = m[m.date == last]["nifty_ret_5d"].dropna()
    nifty_daily = (nrv.iloc[0] / np.sqrt(252) * 100) if len(nrv) else np.nan
    print(f"NIFTY: expected next-day move ~ ±{nifty_daily:.2f}%  (from realized vol; direction ≈ coin flip)")
    print(f"  recent FII net (last 5d, Rs cr): {[round(x) for x in fii_recent.fii_net.tolist()]}  "
          f"sum {fii_recent.fii_net.sum():+.0f}  | nifty 5d momentum {nmom5.iloc[0]*100:+.2f}%" if len(nmom5) else "")
    print(f"\n{'symbol':12s} {'close':>9s} {'atm_iv':>6s} {'exp_move±%':>10s} {'real20±%':>9s} {'ret5%':>7s} {'relsec5%':>8s} {'dir_hint':>9s}")
    cur = m[(m.date == last) & m.symbol.isin(TOP)].copy()
    cur["impl"] = cur.atm_iv / np.sqrt(252) * 100
    cur["real"] = cur.realmove20 * 100
    cur = cur.sort_values("impl", ascending=False)
    for r in cur.itertuples():
        # direction hint: momentum (ret5) aligned with sector; honest -> mostly weak
        hint = "neutral"
        if pd.notna(r.ret_5d) and pd.notna(r.stock_rel_sector_ret_5d):
            if r.ret_5d > 0.02 and r.sector_ret_5d > 0:
                hint = "up-tilt"
            elif r.ret_5d < -0.02 and r.sector_ret_5d < 0:
                hint = "down-tilt"
        print(f"{r.symbol:12s} {r.close:>9.1f} {r.atm_iv*100:>5.0f}% {r.impl:>9.2f} {r.real:>8.2f} "
              f"{(r.ret_5d or 0)*100:>6.1f} {(r.stock_rel_sector_ret_5d or 0)*100:>7.1f} {hint:>9s}")
    print(f"\nbasket avg expected next-day move ~ ±{cur.impl.mean():.2f}% | avg real20 ±{cur.real.mean():.2f}%")
    print("DIRECTION: individual 1-3d ≈ coin flip (model AUC 0.48-0.51, 2026); use expected-move for straddle/sizing.")


if __name__ == "__main__":
    main()
