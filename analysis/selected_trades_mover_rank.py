"""Of the 1188 selected trades, how many were among the day's TOP movers?

For each (date, side) rank all top-65 universe stocks by favourable move (ceiling):
  - mover_rank_universe : rank among all 65 (1 = biggest favourable mover of the day, that side)
  - mover_rank_bucket   : rank within the pick's own bucket (top30 / next35)
Then report, per side, the share of selected trades in top-1/3/5/10.
"""

from __future__ import annotations

import sys
from pathlib import Path

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
    rk = universe.set_index(universe["symbol"].astype(str))["rank"]
    bucket_of = {s: ("A_top30" if r <= 30 else "B_next35") for s, r in rk.items()}

    oc = compute_clean_move_outcomes(market, universe=universe, contract=CleanMoveContract())
    oc = oc[oc["status"].eq("evaluated")][["date", "symbol", "side", "ceiling"]].copy()
    oc["symbol"] = oc["symbol"].astype(str)
    oc["bucket"] = oc["symbol"].map(bucket_of)
    oc["rank_universe"] = oc.groupby(["date", "side"])["ceiling"].rank(method="min", ascending=False)
    oc["rank_bucket"] = oc.groupby(["date", "side", "bucket"])["ceiling"].rank(method="min", ascending=False)

    m = trades.merge(oc[["date", "symbol", "side", "rank_universe", "rank_bucket"]],
                     on=["date", "symbol", "side"], how="left")
    print(f"selected trades: {len(m)} | matched: {m['rank_universe'].notna().sum()}")

    def report(scope_col, scope_name, ks):
        rows = []
        for side in ("long", "short"):
            d = m[m["side"].eq(side)].dropna(subset=[scope_col])
            row = {"side": side, "n": len(d)}
            for k in ks:
                cnt = int((d[scope_col] <= k).sum())
                row[f"top{k}"] = f"{cnt} ({cnt/len(d)*100:.0f}%)"
            rows.append(row)
        # combined
        d = m.dropna(subset=[scope_col])
        row = {"side": "BOTH", "n": len(d)}
        for k in ks:
            cnt = int((d[scope_col] <= k).sum())
            row[f"top{k}"] = f"{cnt} ({cnt/len(d)*100:.0f}%)"
        rows.append(row)
        print(f"\n===== rank within {scope_name} (per day, per side) =====")
        print(pd.DataFrame(rows).to_string(index=False))

    report("rank_universe", "ALL 65 (the day's top movers)", [1, 3, 5, 10])
    report("rank_bucket", "OWN BUCKET (top30 / next35)", [1, 3, 5])

    # by year, universe top5
    print("\n===== top-5 of day (universe) share by year =====")
    by = m.dropna(subset=["rank_universe"]).assign(top5=lambda x: x["rank_universe"] <= 5)
    print(by.groupby(by["date"].dt.year)["top5"].mean().mul(100).round(1).to_string())


if __name__ == "__main__":
    main()
