"""Stage 2: take the strongest structures from eod_grid_search.py's stage-1 sweep and layer a
stop-loss + earlier safe_dte cutoff on top, to see whether the tail risk (worst_dd_pct, p10
pnl_per_lot) can be tightened toward the stock strategies' profile (worst_dd typically 0, only
rare deep excursions) without giving up much of the median pnl_per_lot edge.
"""
from __future__ import annotations

import itertools
import time
import warnings
from pathlib import Path

import pandas as pd

import eod_credit_spread as ecs

warnings.filterwarnings("ignore", category=FutureWarning)
RESULTS_DIR = Path(__file__).resolve().parent / "results"

# Carried over from stage 1's top results (both the unconstrained-median and win_rate>=0.75 winners)
CANDIDATES = [
    dict(short_otm=0.0075, wing_ce=0.020, wing_pe=0.04, fwd_days=5, dte_min=2),
    dict(short_otm=0.0075, wing_ce=0.020, wing_pe=0.03, fwd_days=5, dte_min=2),
    dict(short_otm=0.0150, wing_ce=0.020, wing_pe=0.04, fwd_days=5, dte_min=2),
    dict(short_otm=0.0125, wing_ce=0.020, wing_pe=0.04, fwd_days=5, dte_min=1),
    dict(short_otm=0.0100, wing_ce=0.020, wing_pe=0.04, fwd_days=5, dte_min=2),
]
STOP_LOSS_GRID = [None, -0.5, -0.75, -1.0]
SAFE_DTE_GRID = [0, 1, 2]

if __name__ == "__main__":
    t0 = time.time()
    eod = ecs.load_eod_table()
    print(f"[grid2] EOD table loaded: {len(eod):,} rows, {time.time()-t0:.1f}s")

    rows = []
    combos = list(itertools.product(CANDIDATES, STOP_LOSS_GRID, SAFE_DTE_GRID))
    print(f"[grid2] {len(combos)} combos")
    t0 = time.time()
    for i, (struct, sl, sd) in enumerate(combos):
        df = ecs.build_eod_spreads(eod, short_otm=struct["short_otm"], wing_ce=struct["wing_ce"],
                                    wing_pe=struct["wing_pe"], fwd_days=struct["fwd_days"],
                                    dte_min=struct["dte_min"], safe_dte=sd, stop_loss_frac=sl)
        if len(df) < 30:
            continue
        rows.append({
            **struct, "safe_dte": sd, "stop_loss": sl if sl is not None else "none",
            "n": len(df),
            "win_rate": round((df["outcome"] == "win").mean(), 3),
            "median_pnl_per_lot": round(df["pnl_per_lot"].median(), 1),
            "mean_pnl_per_lot": round(df["pnl_per_lot"].mean(), 1),
            "median_ror_pct": round(df["ror_pct"].median(), 1),
            "worst_ror_pct": round(df["ror_pct"].min(), 1),
            "worst_dd_pct": round(df["max_dd_pct"].min(), 1),
            "p10_pnl_per_lot": round(df["pnl_per_lot"].quantile(0.10), 1),
            "p25_pnl_per_lot": round(df["pnl_per_lot"].quantile(0.25), 1),
        })
        if (i + 1) % 15 == 0:
            print(f"[grid2] {i+1}/{len(combos)} done, {time.time()-t0:.0f}s elapsed")

    res = pd.DataFrame(rows).sort_values("median_pnl_per_lot", ascending=False)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    res.to_csv(RESULTS_DIR / "eod_grid_search2.csv", index=False)
    print(f"\n[grid2] done in {time.time()-t0:.0f}s\n")
    print(res.to_string(index=False))
