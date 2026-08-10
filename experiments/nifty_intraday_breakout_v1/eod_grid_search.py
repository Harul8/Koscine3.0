"""Grid search over eod_credit_spread.py's structure/hold-window params, optimizing for MEDIAN
pnl_per_lot (the user's stated target metric, not mean -- mean is dominated by rare deep-loss
tails on a short-premium book, median is what a trader actually experiences most of the time).
Benchmark to beat/approach: stock-level median pnl_per_lot -- condor ~Rs 2,475, broken_wing
~Rs 6,630, skew ~Rs 7,660 (locks/prod_sell_strategies/*.csv).

Loads the EOD table ONCE (expensive part), then reuses it across every grid combo.
"""
from __future__ import annotations

import itertools
import time
import warnings
from pathlib import Path

import pandas as pd

import eod_credit_spread as ecs

warnings.filterwarnings("ignore", category=FutureWarning)   # pandas argmin-skipna noise, harmless
RESULTS_DIR = Path(__file__).resolve().parent / "results"

SHORT_OTM_GRID = [0.005, 0.0075, 0.01, 0.0125, 0.015, 0.02]
WING_CE_GRID = [0.01, 0.015, 0.02]
WING_PE_GRID = [0.02, 0.03, 0.04]
FWD_GRID = [3, 5, 7]
DTE_MIN_GRID = [1, 2]
SAFE_DTE = 0

if __name__ == "__main__":
    t0 = time.time()
    eod = ecs.load_eod_table()
    print(f"[grid] EOD table loaded: {len(eod):,} rows, {time.time()-t0:.1f}s")

    rows = []
    combos = list(itertools.product(SHORT_OTM_GRID, WING_CE_GRID, WING_PE_GRID, FWD_GRID, DTE_MIN_GRID))
    print(f"[grid] {len(combos)} combos")
    t0 = time.time()
    for i, (so, wc, wp, fwd, dmin) in enumerate(combos):
        if wc >= wp:   # keep the Broken-Wing asymmetry (narrow call / wide put) -- the direction
            continue   # that already works for stocks; skip symmetric/inverted combos
        df = ecs.build_eod_spreads(eod, short_otm=so, wing_ce=wc, wing_pe=wp,
                                    fwd_days=fwd, dte_min=dmin, safe_dte=SAFE_DTE)
        if len(df) < 30:
            continue
        rows.append({
            "short_otm": so, "wing_ce": wc, "wing_pe": wp, "fwd_days": fwd, "dte_min": dmin,
            "n": len(df),
            "win_rate": round((df["outcome"] == "win").mean(), 3),
            "median_pnl_per_lot": round(df["pnl_per_lot"].median(), 1),
            "mean_pnl_per_lot": round(df["pnl_per_lot"].mean(), 1),
            "median_ror_pct": round(df["ror_pct"].median(), 1),
            "worst_ror_pct": round(df["ror_pct"].min(), 1),
            "worst_dd_pct": round(df["max_dd_pct"].min(), 1),
            "p10_pnl_per_lot": round(df["pnl_per_lot"].quantile(0.10), 1),
        })
        if (i + 1) % 50 == 0:
            print(f"[grid] {i+1}/{len(combos)} combos done, {time.time()-t0:.0f}s elapsed")

    res = pd.DataFrame(rows).sort_values("median_pnl_per_lot", ascending=False)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    res.to_csv(RESULTS_DIR / "eod_grid_search.csv", index=False)
    print(f"\n[grid] done in {time.time()-t0:.0f}s, {len(res)} valid combos\n")
    print("=== Top 15 by median_pnl_per_lot ===")
    print(res.head(15).to_string(index=False))
    print("\n=== Top 15 by median_pnl_per_lot among win_rate>=0.75 ===")
    print(res[res["win_rate"] >= 0.75].head(15).to_string(index=False))
