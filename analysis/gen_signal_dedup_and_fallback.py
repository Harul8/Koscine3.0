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
   real signal that day -- a backstop pick, not a qualifying setup). Ranked by theoretical
   max_profit_per_lot (credit * lot_size), the SAME entry-time-only criterion real signals are
   selected by -- NOT realized/mark-to-market pnl_per_lot (tried first; reverted, 2026-08): a
   selection rule that depends on ex-post outcome data means the SAME day's fallback pick can
   change identity on a later re-run purely because more forward-price data became available for
   an already-open trade, which is exactly the "history re-deciding its own picks" behavior this
   whole reconciliation step exists to eliminate. Selection is now 100% a function of entry-day
   data alone -- once written, a day's pick (real or NS) never changes. Actual realized PnL for a
   recent pick is a SEPARATE, ongoing concern: see analysis/append_daily_sell_signals.py, which
   marks already-logged trades from the last 5 trading days to market without re-picking anything.

Idempotent: strips any previously-added tier="NS" rows in the target range from each CSV before
recomputing (rows outside the range are always left untouched -- see the optional range args
below), so this can be safely re-run after any of the 3 base generators are re-run (e.g. a
partial-range regen). Since selection is now entry-time-only/deterministic (see the base
generators' 2026-08 unification), re-running over the SAME range always reproduces the SAME
picks -- this is not "history re-deciding its own signals," it's a no-op recomputation that
happens to also be cheap to skip via the range args below.

Usage:
    python analysis/gen_signal_dedup_and_fallback.py                          # full history (one-time)
    python analysis/gen_signal_dedup_and_fallback.py 2026-08-08 2026-08-12    # only reconsider
        no-signal-days in [start, end] -- for analysis/append_daily_sell_signals.py's daily use,
        so a routine daily run doesn't re-scan/rewrite the full multi-year history every time.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(r"C:\Users\rahul\Koscine 3.0")
OUTDIR = ROOT / "locks" / "prod_sell_strategies"
FALLBACK_MIN_ROR = 100.0   # 100 -> 130 -> 300 -> 500 -> back to 100 (2026-08, explicit user
# decision). The 300/500 steps were kept in lockstep with Condor's own primary bar specifically
# because tightening Condor without also raising this just pushed its volume into the fallback
# instead of actually cutting it (verified: Condor's real-signal rate had collapsed enough that
# the fallback fired ~101/yr just to cover its gap). Condor has since been killed entirely (see
# gen_sell_signal_history.py), so that lockstep no longer applies -- with BW/Skew now each firing
# ~120-130/yr of real signals on their own, a genuine no-signal day is rare, so the fallback can
# safely sit BELOW their primary bars again as a true backstop, gated by MIN_PROFIT_PER_LOT below.
MIN_PROFIT_PER_LOT = 10_000.0   # 2026-08, explicit user decision -- see
                                 # gen_sell_signal_history.py's identical constant for rationale.
RANGE_MIN = sys.argv[1] if len(sys.argv) > 1 else None
RANGE_MAX = sys.argv[2] if len(sys.argv) > 2 else None

# (strategy, real-signal CSV, raw-candidates cache, sort columns)
STRATEGIES = [
    ("condor", "signal_history.csv", "condor_candidates_raw.parquet", ["entry_date"]),
    ("broken_wing", "broken_wing_signal_history.csv", "broken_wing_candidates_raw.parquet", ["tier", "entry_date"]),
    ("skew", "skew_signal_history.csv", "skew_candidates_raw.parquet", ["tier", "entry_date"]),
]


def _in_range(dates) -> pd.Series:
    d = pd.Series(dates)
    lo = d >= RANGE_MIN if RANGE_MIN else pd.Series(True, index=d.index)
    hi = d <= RANGE_MAX if RANGE_MAX else pd.Series(True, index=d.index)
    return lo & hi


def _load(sig_file: str, cand_file: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    sp, cp = OUTDIR / sig_file, OUTDIR / cand_file
    sig = pd.read_csv(sp) if sp.exists() else pd.DataFrame()
    cand = pd.read_parquet(cp) if cp.exists() else pd.DataFrame()
    if len(sig) and "tier" in sig.columns:
        # strip prior fallback rows ONLY inside the target range -- an NS pick outside it is
        # never touched (see module docstring: selection is entry-time-only and permanent).
        drop = (sig["tier"] == "NS") & _in_range(sig["entry_date"])
        sig = sig[~drop].copy()
    return sig, cand


def _dedup_bw_vs_skew(bw: pd.DataFrame, sk: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not len(bw) or not len(sk):
        return bw, sk
    bw = bw.copy(); sk = sk.copy()
    bw["_profit"] = bw["credit"] * bw["lot_size"]
    sk["_profit"] = sk["credit"] * sk["lot_size"]
    # only symbol/day pairs INSIDE the target range are eligible to be reconsidered -- a pair
    # outside it keeps whatever this dedup already decided for it, permanently.
    bw_in = bw[_in_range(bw["entry_date"])]
    sk_in = sk[_in_range(sk["entry_date"])]
    bw_keys = set(zip(bw_in["symbol"], bw_in["entry_date"]))
    sk_keys = set(zip(sk_in["symbol"], sk_in["entry_date"]))
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
            cdf = cand[strat]
            all_days |= set(cdf.loc[_in_range(cdf["entry_date"]), "entry_date"])
    no_signal_days = sorted(all_days - days_with_signal)
    print(f"{len(no_signal_days)} days with no real signal from any strategy (of {len(all_days)} candidate days"
          f"{f' in {RANGE_MIN}..{RANGE_MAX}' if (RANGE_MIN or RANGE_MAX) else ''})", flush=True)

    added: dict[str, list[pd.Series]] = {strat: [] for strat, _, _, _ in STRATEGIES}
    for day in no_signal_days:
        pool: list[tuple[str, pd.Series]] = []
        for strat, _, _, _ in STRATEGIES:
            if strat == "condor":
                # Condor was killed (2026-08, MIN_ENTRY_ROR set unreachable) specifically to
                # produce ZERO signals, real or otherwise -- letting it re-enter through the
                # fallback pool (its raw candidates clear FALLBACK_MIN_ROR=100 easily, since that
                # bar was never raised in lockstep once Condor's own primary bar went to
                # infinity) silently defeated that. Confirmed: without this exclusion Condor
                # picked up 168 of 196 no-signal-days as NS fallback, i.e. it became the DOMINANT
                # fallback source instead of contributing nothing. Excluded entirely.
                continue
            if strat == "broken_wing":
                # 2026-08, explicit user decision: BW's NS fallback had grown to 94/232 (41%) of
                # its total signal count -- its wide-OTM-wing candidates have structurally high
                # max_profit_per_lot even on days its own (now tighter) primary bar wouldn't fire
                # for real, so it kept winning the cross-strategy fallback pool regardless of
                # quality. Skew is both the stronger performer (mean/median ROR ~2x BW's, see
                # gen_skew_signal_history.py) and now the sole fallback source -- a genuine
                # no-signal day gets Skew's best leftover candidate or nothing, never BW's.
                continue
            cdf = cand[strat]
            if not len(cdf):
                continue
            day_rows = cdf[(cdf["entry_date"] == day) & (cdf["entry_ror_pct"] > FALLBACK_MIN_ROR)
                           & (cdf["max_profit_per_lot"] >= MIN_PROFIT_PER_LOT)]
            pool.extend((strat, row) for _, row in day_rows.iterrows())
        if not pool:
            continue
        best_strat, best_row = max(pool, key=lambda sr: sr[1]["max_profit_per_lot"] if pd.notna(sr[1].get("max_profit_per_lot")) else float("-inf"))
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
