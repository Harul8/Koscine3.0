"""Cross-strategy signal reconciliation. Run AFTER all three gen_*_signal_history.py scripts --
it reads their real-signal CSVs (Condor/Broken-Wing/Skew, bar-clearing candidates only) and their
*_candidates_raw.parquet caches (every structurally-valid candidate, gate or no gate).

1. Profitability-based Broken-Wing vs Skew de-dup: previously (gen_skew_signal_history.py) a
   skew candidate was dropped outright whenever the same symbol also had a Broken-Wing signal on
   the same entry_date -- "BW always wins," an arbitrary category preference. Now whichever
   structure is actually more profitable that symbol/day wins (credit * lot_size, i.e.
   max_profit_per_lot -- the same "most profitable" rule already used elsewhere in this codebase:
   portfolio_sim.py's _dedup_daily, the frontend's per-symbol-per-day pick). Justified by the
   data: Skew's mean/median ROR beats Broken-Wing's at every comparable tier (often ~2x), and
   needs meaningfully less margin (1.6x vs 2.7x max_risk, see portfolio_sim.py's live Upstox
   margin-API calibration) -- Skew was structurally the stronger signal, just always losing this
   arbitrary tie. Condor is not part of this dedup (never was).

2. Cross-strategy "no signal that day" fallback: on a calendar day where NONE of Condor/
   Broken-Wing/Skew has a REAL signal (after step 1's dedup) -- checked across the WHOLE day, not
   per-strategy, so a real Condor signal on a quiet Broken-Wing/Skew day means no fallback fires
   for any of the three -- pool every structurally-valid candidate from ALL 3 strategies/tiers for
   that day with entry_ror_pct > FALLBACK_MIN_ROR, and keep exactly ONE, tagged tier="NS" (no
   real signal that day -- a backstop pick, not a qualifying setup). Ranked by REALIZED
   pnl_per_lot (the candidate's own actual/mark-to-market simulated outcome), not theoretical
   max_profit_per_lot: entry_ror_pct is a quality floor known before the trade, not a forecast of
   what actually happens -- a candidate with a lower entry ratio can still realize MORE than one
   that cleared a much higher bar (concretely verified: 2026-07-30 EICHERMOT skew candidate,
   entry_ror 125.4% -> realized ror 84.6%/pnl_per_lot Rs11,255, vs. 2026-07-28 ADANIENT condor
   real signal, entry_ror 284.1% -> realized ror only 62.9%/pnl_per_lot Rs7,586 -- entry ROR and
   realized outcome are different measurements, not guaranteed to rank the same way). Ranking the
   fallback by what a trade actually made, not its theoretical ceiling, is what "maximizes profit
   realized" means here.

Idempotent: strips any previously-added tier="NS" rows from each CSV before recomputing, so this
can be safely re-run after any of the 3 base generators are re-run (e.g. a partial-range regen).

Usage:
    python analysis/gen_signal_dedup_and_fallback.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(r"C:\Users\rahul\Koscine 3.0")
OUTDIR = ROOT / "locks" / "prod_sell_strategies"
FALLBACK_MIN_ROR = 100.0

# (strategy, real-signal CSV, raw-candidates cache, sort columns)
STRATEGIES = [
    ("condor", "signal_history.csv", "condor_candidates_raw.parquet", ["entry_date"]),
    ("broken_wing", "broken_wing_signal_history.csv", "broken_wing_candidates_raw.parquet", ["tier", "entry_date"]),
    ("skew", "skew_signal_history.csv", "skew_candidates_raw.parquet", ["tier", "entry_date"]),
]


def _load(sig_file: str, cand_file: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    sp, cp = OUTDIR / sig_file, OUTDIR / cand_file
    sig = pd.read_csv(sp) if sp.exists() else pd.DataFrame()
    cand = pd.read_parquet(cp) if cp.exists() else pd.DataFrame()
    if len(sig) and "tier" in sig.columns:
        sig = sig[sig["tier"] != "NS"].copy()   # strip prior fallback rows -- recomputed fresh each run
    return sig, cand


def _dedup_bw_vs_skew(bw: pd.DataFrame, sk: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not len(bw) or not len(sk):
        return bw, sk
    bw = bw.copy(); sk = sk.copy()
    bw["_profit"] = bw["credit"] * bw["lot_size"]
    sk["_profit"] = sk["credit"] * sk["lot_size"]
    bw_keys = set(zip(bw["symbol"], bw["entry_date"]))
    sk_keys = set(zip(sk["symbol"], sk["entry_date"]))
    shared = bw_keys & sk_keys
    if shared:
        bw_shared = bw[bw[["symbol", "entry_date"]].apply(tuple, axis=1).isin(shared)]
        sk_shared = sk[sk[["symbol", "entry_date"]].apply(tuple, axis=1).isin(shared)]
        bw_best = bw_shared.groupby(["symbol", "entry_date"])["_profit"].max()
        sk_best = sk_shared.groupby(["symbol", "entry_date"])["_profit"].max()
        drop_from_bw, drop_from_sk = set(), set()
        for key in shared:
            if bw_best.get(key, -1.0) >= sk_best.get(key, -1.0):
                drop_from_sk.add(key)
            else:
                drop_from_bw.add(key)
        bw = bw[~bw[["symbol", "entry_date"]].apply(tuple, axis=1).isin(drop_from_bw)]
        sk = sk[~sk[["symbol", "entry_date"]].apply(tuple, axis=1).isin(drop_from_sk)]
        print(f"BW-vs-skew dedup: {len(drop_from_sk)} symbol/days kept for BW (more profitable), "
              f"{len(drop_from_bw)} kept for skew", flush=True)
    return bw.drop(columns=["_profit"]), sk.drop(columns=["_profit"])


def main() -> None:
    sig: dict[str, pd.DataFrame] = {}
    cand: dict[str, pd.DataFrame] = {}
    for strat, sig_file, cand_file, _ in STRATEGIES:
        sig[strat], cand[strat] = _load(sig_file, cand_file)

    sig["broken_wing"], sig["skew"] = _dedup_bw_vs_skew(sig["broken_wing"], sig["skew"])

    days_with_signal: set[str] = set()
    for strat, _, _, _ in STRATEGIES:
        if len(sig[strat]):
            days_with_signal |= set(sig[strat]["entry_date"])
    all_days: set[str] = set()
    for strat, _, _, _ in STRATEGIES:
        if len(cand[strat]):
            all_days |= set(cand[strat]["entry_date"])
    no_signal_days = sorted(all_days - days_with_signal)
    print(f"{len(no_signal_days)} days with no real signal from any strategy (of {len(all_days)} candidate days)", flush=True)

    added: dict[str, list[pd.Series]] = {strat: [] for strat, _, _, _ in STRATEGIES}
    for day in no_signal_days:
        pool: list[tuple[str, pd.Series]] = []
        for strat, _, _, _ in STRATEGIES:
            cdf = cand[strat]
            if not len(cdf):
                continue
            day_rows = cdf[(cdf["entry_date"] == day) & (cdf["entry_ror_pct"] > FALLBACK_MIN_ROR)]
            pool.extend((strat, row) for _, row in day_rows.iterrows())
        if not pool:
            continue
        best_strat, best_row = max(pool, key=lambda sr: sr[1]["pnl_per_lot"] if pd.notna(sr[1].get("pnl_per_lot")) else float("-inf"))
        best_row = best_row.copy()
        best_row["tier"] = "NS"
        if "clears_primary_bar" in best_row.index:
            best_row = best_row.drop("clears_primary_bar")
        added[best_strat].append(best_row)

    for strat, sig_file, _, sort_cols in STRATEGIES:
        rows, df = added[strat], sig[strat]
        if rows:
            add_df = pd.DataFrame(rows)
            df = pd.concat([df, add_df], ignore_index=True) if len(df) else add_df
        if len(df):
            df = df.sort_values(sort_cols)
        df.to_csv(OUTDIR / sig_file, index=False)
        print(f"{strat}: {len(df)} total signals ({len(rows)} NS fallback) -> {sig_file}", flush=True)


if __name__ == "__main__":
    main()
