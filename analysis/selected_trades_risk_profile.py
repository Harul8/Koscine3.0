"""For the exact 1188 selected trades, report per side (LONG/SHORT):
  - adverse excursion (floor_depth) = how far the underlying moved AGAINST  -> 'stop %'
  - % that moved opposite (adverse excursion exceeded the favourable peak)
  - mean upside target (favourable peak) EXCLUDING the opposite movers
Joins the saved trade list to the clean-move outcomes (which carry floor_depth + ceiling).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from koscine3.data.sources import load_market_data  # noqa: E402
from koscine3.data.universe import UniverseConfig, build_universe  # noqa: E402
from koscine3.outcomes.clean_move_contract import CleanMoveContract, compute_clean_move_outcomes  # noqa: E402

XLSX = ROOT / "reports" / "option_strategy_top65_buckets_2026-06-11.xlsx"


def main() -> None:
    trades = pd.read_excel(XLSX, sheet_name="Trades")
    trades["date"] = pd.to_datetime(trades["date"])
    trades["symbol"] = trades["symbol"].astype(str)

    market = load_market_data()
    universe = build_universe(market, UniverseConfig(cutoff_date="2025-12-31", top_n=65))
    oc = compute_clean_move_outcomes(market, universe=universe, contract=CleanMoveContract())
    oc = oc[oc["status"].eq("evaluated")][["date", "symbol", "side", "ceiling", "floor_depth"]].copy()
    oc["symbol"] = oc["symbol"].astype(str)

    m = trades.merge(oc, on=["date", "symbol", "side"], how="left")
    print(f"trades: {len(m)} | matched to outcomes: {m['ceiling'].notna().sum()}")

    rows = []
    for side in ("long", "short"):
        d = m[m["side"].eq(side)].dropna(subset=["ceiling", "floor_depth"])
        opp = d["floor_depth"] > d["ceiling"]          # moved more against than for
        fav = ~opp
        rows.append({
            "side": side, "n": len(d),
            "avg_adverse_(stop)_%": round(d["floor_depth"].mean() * 100, 2),
            "median_adverse_%": round(d["floor_depth"].median() * 100, 2),
            "p90_adverse_%": round(d["floor_depth"].quantile(0.90) * 100, 2),
            "pct_moved_opposite_%": round(opp.mean() * 100, 1),
            "pct_moved_favourable_%": round(fav.mean() * 100, 1),
            "mean_upside_excl_opp_%": round(d.loc[fav, "ceiling"].mean() * 100, 2),
            "median_upside_excl_opp_%": round(d.loc[fav, "ceiling"].median() * 100, 2),
            "mean_upside_all_%": round(d["ceiling"].mean() * 100, 2),
        })
    out = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print("\n===== SELECTED TRADES — RISK/REWARD BY SIDE =====")
    print(out.to_string(index=False))
    print("\nDefs: adverse = max move against entry over the 5d window (floor_depth); "
          "opposite = adverse excursion exceeded the favourable peak; "
          "upside = favourable peak (ceiling) of the non-opposite movers.")


if __name__ == "__main__":
    main()
