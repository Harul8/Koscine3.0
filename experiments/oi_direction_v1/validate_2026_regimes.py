"""Validate (again): do the OI regimes — Long Buildup / Short Covering / Short Buildup / Long Unwinding —
give ANY directional signal in 2026, or none at all? (contained; read-only; no PROD touch)

For each period (pre2025 / 2025 / 2026) and each regime, report P(up) at fwd 1/2/5d, the EDGE vs that
period's base rate, sample size, and a 2*SE significance flag. Then the key question two ways:
  (1) SEPARATION: do the 4 regimes still spread apart in 2026 (a signal exists even if the SIGN flipped —
      a flipped-but-consistent regime is still tradeable, just invert the rule)?
  (2) PERSISTENCE: does each regime keep the SAME sign of edge it had in 2024-25 (the original rule still works)?
Plus an aggressive cut (|OI z| top tercile) and a 2026 low/high-vol split (finding: high-vol=momentum).

    set PYTHONPATH=src && python experiments/oi_direction_v1/validate_2026_regimes.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from koscine3.data.sources import load_market_data  # noqa: E402

REGS = ["LongBuildup", "ShortCovering", "ShortBuildup", "LongUnwinding"]
# original practitioner rule: bullish = LongBuildup + ShortCovering ; bearish = ShortBuildup + LongUnwinding
BULL = {"LongBuildup", "ShortCovering"}


def regime(dp, doi):
    r = np.full(len(dp), "flat", dtype=object)
    r[(dp > 0) & (doi > 0)] = "LongBuildup"
    r[(dp < 0) & (doi > 0)] = "ShortBuildup"
    r[(dp > 0) & (doi < 0)] = "ShortCovering"
    r[(dp < 0) & (doi < 0)] = "LongUnwinding"
    return r


def pup(s):
    s = s.dropna()
    n = len(s)
    if n < 100:
        return np.nan, np.nan, n
    p = (s > 0).mean()
    return p, np.sqrt(p * (1 - p) / n), n


def main():
    cols = ["date", "symbol", "close", "ret_1d", "ret_5d", "fut_chg_oi", "fut_oi_z_60d",
            "atm_iv", "realized_vol_20"]
    m = load_market_data(columns=cols)
    m["symbol"] = m["symbol"].astype(str); m["date"] = pd.to_datetime(m["date"])
    m = m.sort_values(["symbol", "date"]).reset_index(drop=True)
    g = m.groupby("symbol", sort=False)
    for h in (1, 2, 5):
        m[f"fwd{h}"] = g["close"].shift(-h) / m.close - 1
    m["cum_oi_chg_5"] = g["fut_chg_oi"].transform(lambda s: s.rolling(5, min_periods=3).sum())
    m["reg1"] = regime(np.sign(m.ret_1d.values), np.sign(m.fut_chg_oi.values))   # 1-day regime
    m["reg5"] = regime(np.sign(m.ret_5d.values), np.sign(m.cum_oi_chg_5.values))  # 5-day regime
    e = m[(m.atm_iv.notna()) & (m.close >= 100) & m.fut_chg_oi.notna()].copy()
    e["per"] = np.where(e.date < "2025-01-01", "pre2025", np.where(e.date < "2026-01-01", "2025", "2026"))

    for regcol, fwd, lab in [("reg1", "fwd1", "1-DAY regime -> fwd 1d"),
                             ("reg1", "fwd2", "1-DAY regime -> fwd 2d"),
                             ("reg5", "fwd5", "5-DAY regime -> fwd 5d")]:
        print(f"\n{'='*78}\n{lab}    (edge = P(up) - period base;  * = |edge| > 2*SE)\n{'='*78}")
        for per in ("pre2025", "2025", "2026"):
            d = e[e.per == per]
            base = (d[fwd] > 0).mean()
            print(f"\n  [{per}]  base P(up)={base:.3f}  n={len(d):,}")
            print(f"    {'regime':15s} {'n':>7s} {'P(up)':>7s} {'edge':>7s} {'sig':>4s} {'origRule':>9s}")
            spreads = []
            for rg in REGS:
                p, se, n = pup(d[d[regcol] == rg][fwd])
                if np.isnan(p):
                    continue
                edge = p - base
                sig = "*" if abs(edge) > 2 * se else ""
                rule = "UP" if rg in BULL else "DOWN"
                # does realized edge agree with the original rule's expected sign?
                agree = "ok" if (edge > 0) == (rg in BULL) else "FLIP"
                print(f"    {rg:15s} {n:>7d} {p:>7.3f} {edge:>+7.3f} {sig:>4s} {rule:>5s}/{agree:<4s}")
                spreads.append(p)
            if spreads:
                print(f"    -> regime spread (max-min P(up)) = {max(spreads)-min(spreads):.3f}  "
                      f"(separation = signal exists, regardless of sign)")

    # aggressive cut, 2026
    print(f"\n{'='*78}\nAGGRESSIVE 2026 (|fut_oi_z_60d| top tercile), 5d regime -> fwd5\n{'='*78}")
    e26 = e[e.per == "2026"]
    agg = e26[e26.fut_oi_z_60d.abs() >= e26.fut_oi_z_60d.abs().quantile(0.667)]
    base = (agg.fwd5 > 0).mean()
    print(f"  base P(up5)={base:.3f}  n={len(agg):,}")
    for rg in REGS:
        p, se, n = pup(agg[agg.reg5 == rg].fwd5)
        if not np.isnan(p):
            agree = "ok" if (p - base > 0) == (rg in BULL) else "FLIP"
            print(f"    {rg:15s} n={n:>5d}  P(up5)={p:.3f}  edge={p-base:+.3f} {'*' if abs(p-base)>2*se else ''}  {agree}")

    # 2026 vol split
    print(f"\n{'='*78}\n2026 by VOL (realized_vol_20 median split), 5d regime -> fwd5\n{'='*78}")
    e26 = e26.copy(); e26["vb"] = np.where(e26.realized_vol_20 <= e26.realized_vol_20.median(), "lowvol", "highvol")
    for vb in ("lowvol", "highvol"):
        d = e26[e26.vb == vb]; base = (d.fwd5 > 0).mean()
        print(f"\n  [{vb}] base P(up5)={base:.3f} n={len(d):,}")
        for rg in REGS:
            p, se, n = pup(d[d.reg5 == rg].fwd5)
            if not np.isnan(p):
                agree = "ok" if (p - base > 0) == (rg in BULL) else "FLIP"
                print(f"    {rg:15s} n={n:>5d} P(up5)={p:.3f} edge={p-base:+.3f} {'*' if abs(p-base)>2*se else ''} {agree}")

    print(f"\n{'='*78}\nREAD: 'spread'~0 & no '*' in 2026 => regimes give NO signal. Large spread but 'FLIP' on every"
          f"\nregime => signal SURVIVES but INVERTED (momentum), i.e. invert the old rule. 'ok' => original rule holds.\n{'='*78}")


if __name__ == "__main__":
    main()
