"""Why did the direction signal break in 2026, and what works now? (contained; read-only; no PROD touch)

Tests, per period (pre-2025 / 2025 / 2026) x horizon (fwd 1/2/5 day), the signed IC (spearman of signal vs
forward return) for many candidate direction signals. Key questions:
  - did MOMENTUM vs MEAN-REVERSION flip (regime change / SEBI Nov-2024 retail squeeze)?
  - what (if anything) carries directional content in 2026?
  - is direction more predictable at 1-2 day than 5 day (the short-lived overlay idea)?
Includes a 2026 vol-regime conditioning.

    set PYTHONPATH=src && python experiments/oi_direction_v1/direction_2026_diag.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from koscine3.data.sources import load_market_data  # noqa: E402

# candidate signed signals (hypothesis: higher -> more likely UP). Sign interpreted in the IC.
CAND = ["ret_1d", "ret_5d", "ret_20d", "gap_pct", "fut_chg_oi", "oi_buildup_ratio", "price_oi_divergence",
        "pcr_oi", "pcr_oi_chg_5", "iv_skew_norm", "iv_skew_chg_5d", "atm_iv_chg_5", "delivery_pct",
        "delivery_pct_chg_5", "mkt_pct_above_sma50", "nifty_ret_5d", "fut_oi_z_60d"]


def ic(df, sig, fwd):
    s = df[[sig, fwd]].replace([np.inf, -np.inf], np.nan).dropna()
    return s[sig].corr(s[fwd], "spearman") if len(s) > 200 else np.nan


def main():
    cols = ["date", "symbol", "close", "high", "low", "atm_iv", "realized_vol_20", "max_pain"] + CAND
    m = load_market_data()
    keep = [c for c in cols if c in m.columns]
    m = m[keep].copy()
    m["symbol"] = m["symbol"].astype(str); m["date"] = pd.to_datetime(m["date"])
    m = m.sort_values(["symbol", "date"]).reset_index(drop=True)
    g = m.groupby("symbol", sort=False)
    for h in (1, 2, 5):
        m[f"fwd{h}"] = g["close"].shift(-h) / m.close - 1
    m = m[(m.atm_iv.notna()) & (m.close >= 100)].copy()
    avail = [c for c in CAND if c in m.columns]
    m["per"] = np.where(m.date < "2025-01-01", "pre2025", np.where(m.date < "2026-01-01", "2025", "2026"))

    # maxpain distance (pin/GEX proxy): >0 means price above max_pain (mean-rev expects pull DOWN)
    if "max_pain" in m.columns:
        m["maxpain_gap"] = (m.close - m.max_pain) / m.close
        avail.append("maxpain_gap")

    for h in (2, 5):
        print(f"\n=== signed IC (spearman signal vs fwd{h}d return) by period ===")
        print(f"{'signal':20s} {'pre2025':>9s} {'2025':>9s} {'2026':>9s}   {'flip?':>6s}")
        rows = []
        for sig in avail:
            r = {p: ic(m[m.per == p], sig, f"fwd{h}") for p in ("pre2025", "2025", "2026")}
            flip = "FLIP" if (np.sign(r["pre2025"]) != np.sign(r["2026"]) and abs(r["pre2025"]) > 0.01 and abs(r["2026"]) > 0.01) else ""
            print(f"{sig:20s} {r['pre2025']:>9.3f} {r['2025']:>9.3f} {r['2026']:>9.3f}   {flip:>6s}")
            rows.append((sig, abs(r["2026"]), r["2026"]))
        print(f"  strongest 2026 |IC| (fwd{h}): " + ", ".join(f"{s}({v:+.3f})" for s, _, v in sorted(rows, key=lambda x: -x[1])[:5]))

    # momentum vs mean-reversion summary (ret_5d IC sign tells the regime)
    print("\n=== regime: ret_5d IC (continuation if +, mean-reversion if -) ===")
    for h in (1, 2, 5):
        print(f"  fwd{h}: " + " ".join(f"{p} {ic(m[m.per==p],'ret_5d',f'fwd{h}'):+.3f}" for p in ("pre2025", "2025", "2026")))

    # 2026 vol conditioning: does anything work in low-vol vs high-vol 2026?
    print("\n=== 2026 only: top signals by |IC fwd2| in LOW vs HIGH realized_vol_20 ===")
    e26 = m[m.per == "2026"].copy()
    e26["volbk"] = np.where(e26.realized_vol_20 <= e26.realized_vol_20.median(), "lowvol", "highvol")
    for vb in ("lowvol", "highvol"):
        d = e26[e26.volbk == vb]
        rk = sorted(((s, ic(d, s, "fwd2")) for s in avail), key=lambda x: -abs(x[1] if np.isfinite(x[1]) else 0))[:5]
        print(f"  {vb}: " + ", ".join(f"{s}({v:+.3f})" for s, v in rk))


if __name__ == "__main__":
    main()
