"""Universe for the premium-SELLING strategies (condor / broken-wing / skew), shared by the
offline backtests (analysis/gen_*_signal_history.py) and the live API endpoints (api/main.py) so
both see the SAME universe -- previously they didn't: live was gated by the stale ~97-symbol
ML-pipeline active_universe.txt while the backtest ran over 190-208 bhavcopy-derived names, so
the live tab could show a track record for a strategy it couldn't actually surface picks for.

Two different universes, by design (explicit user decision, not a default):

  - BACKTEST universe = NIFTY50 only, held FIXED across the whole backtest window. No daily
    rotation. This trades a small amount of backtest purity (a handful of the current 50 names
    -- e.g. JIOFIN, ETERNAL, MAXHEALTH -- are relatively recent NIFTY50 additions, so the
    backtest treats them as "always in the index" rather than tracking true point-in-time
    membership, which NSE does not publish as a clean historical feed) for a universe that's
    simple, reproducible, and exactly matches what a trader today would recognize as "Nifty 50".

  - LIVE universe = NIFTY50 (the same fixed 50) UNION that day's top DAILY_N=15 most liquid
    NON-Nifty50 names by futures OI-in-lots. The daily tail exists only for live/current use,
    not backtesting.

NIFTY50 below is a SNAPSHOT (see fetched-on date), not live-refetched on every use -- refetching
NSE's constituent list inside a hot API path is fragile (anti-bot headers, rate limits) and would
silently change backtest results run-to-run. Refresh it deliberately (see refresh_nifty50()) when
NSE next rebalances the index, not automatically.
"""
from __future__ import annotations

import os

import pandas as pd

from koscine.config import SILVER_DATA_ROOT

# Fetched from https://archives.nseindia.com/content/indices/ind_nifty50list.csv on 2026-08-11.
# 50 symbols, verified count. Re-run refresh_nifty50() (fetches the same URL) after NSE's next
# semi-annual index rebalance to pick up additions/removals; this constant will not update itself.
NIFTY50 = frozenset({
    "ADANIENT", "ADANIPORTS", "APOLLOHOSP", "ASIANPAINT", "AXISBANK", "BAJAJ-AUTO", "BAJFINANCE",
    "BAJAJFINSV", "BEL", "BHARTIARTL", "CIPLA", "COALINDIA", "DRREDDY", "EICHERMOT", "ETERNAL",
    "GRASIM", "HCLTECH", "HDFCBANK", "HDFCLIFE", "HINDALCO", "HINDUNILVR", "ICICIBANK", "ITC",
    "INFY", "INDIGO", "JSWSTEEL", "JIOFIN", "KOTAKBANK", "LT", "M&M", "MARUTI", "MAXHEALTH",
    "NTPC", "NESTLEIND", "ONGC", "POWERGRID", "RELIANCE", "SBILIFE", "SHRIRAMFIN", "SBIN",
    "SUNPHARMA", "TCS", "TATACONSUM", "TMPV", "TATASTEEL", "TECHM", "TITAN", "TRENT",
    "ULTRACEMCO", "WIPRO",
})
assert len(NIFTY50) == 50, f"expected 50 Nifty constituents, got {len(NIFTY50)}"

DAILY_N = 15   # rotating tail, LIVE universe only -- today's most liquid non-Nifty50 names

# NSE index derivatives -- excluded from the daily tail because these are STOCK-option
# strategies (index options are cash-settled, a separate book under the Indices tab). They
# would otherwise dominate the ranking: index futures OI-in-lots dwarfs every single stock.
# Verified against the bhavcopy IDXOPT/STKOPT split -- exactly these 5, no overlap with stocks.
INDEX_SYMBOLS = frozenset({"NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "NIFTYNXT50"})

_CACHE: dict = {"mtime": None, "matrix": None}


def _oi_lots_matrix() -> pd.DataFrame:
    """date x symbol matrix of futures OI-in-lots (excluding Nifty50, since the daily tail only
    ever ranks NON-Nifty50 names), cached on eod_deriv_daily's mtime."""
    f = SILVER_DATA_ROOT / "eod_deriv_daily.parquet"
    mt = os.path.getmtime(f)
    if _CACHE["mtime"] == mt:
        return _CACHE["matrix"]

    dd = pd.read_parquet(f, columns=["date", "symbol", "fut_oi"])
    dd["date"] = pd.to_datetime(dd["date"])
    dd["symbol"] = dd["symbol"].astype(str)
    dd = dd[dd["fut_oi"].notna() & (dd["fut_oi"] > 0)
            & ~dd["symbol"].isin(NIFTY50) & ~dd["symbol"].isin(INDEX_SYMBOLS)]

    lot_f = SILVER_DATA_ROOT / "lot_size.parquet"
    if lot_f.exists():
        lots = pd.read_parquet(lot_f, columns=["symbol", "expiry_month", "lot"])
        lots["symbol"] = lots["symbol"].astype(str)
        lots["period"] = pd.to_datetime(lots["expiry_month"]).dt.to_period("M")
        by_month = lots.drop_duplicates(["symbol", "period"]).set_index(["symbol", "period"])["lot"]
        latest = lots.sort_values("expiry_month").groupby("symbol")["lot"].last()
        idx = pd.MultiIndex.from_arrays([dd["symbol"], dd["date"].dt.to_period("M")])
        dd["lot"] = by_month.reindex(idx).to_numpy()
        dd["lot"] = dd["lot"].fillna(dd["symbol"].map(latest))
    else:
        dd["lot"] = pd.NA
    dd["lot"] = pd.to_numeric(dd["lot"], errors="coerce").fillna(1.0).replace(0, 1.0)
    dd["oi_lots"] = dd["fut_oi"] / dd["lot"]

    matrix = dd.pivot_table(index="date", columns="symbol", values="oi_lots", aggfunc="last").sort_index()
    _CACHE.update(mtime=mt, matrix=matrix)
    return matrix


def backtest_universe() -> set:
    """Fixed universe for offline backtests (analysis/gen_*_signal_history.py): NIFTY50 only,
    same set on every date. No look-ahead concern since it's not derived from any per-date
    ranking -- see the module docstring for the point-in-time-membership trade-off this implies."""
    return set(NIFTY50)


def live_daily_tail(core_n_placeholder: int = DAILY_N) -> set:
    """Today's top-N most liquid NON-Nifty50 names by futures OI-in-lots (no look-ahead: ranks
    on the prior available session, same as every other liquidity ranking in this repo)."""
    m = _oi_lots_matrix()
    if m.empty:
        return set()
    last = m.index.max()
    row = m.loc[last].dropna().sort_values(ascending=False)
    return set(row.head(core_n_placeholder).index)


def live_universe_today() -> set:
    """Live universe for the API endpoints: NIFTY50 (fixed 50) + today's top-15 most liquid
    non-Nifty50 names. This is intentionally WIDER than backtest_universe() -- the daily tail
    exists only to give live picks some coverage beyond the index while still keeping Nifty50
    as the backtested core."""
    try:
        return set(NIFTY50) | live_daily_tail()
    except Exception:
        return set(NIFTY50)


def live_daily_tail_for_date(date, core_n_placeholder: int = DAILY_N) -> set:
    """Same top-N-by-OI-in-lots ranking as live_daily_tail(), but for a specific historical date
    instead of always the latest available session. Ranks on the latest OI data available ON OR
    BEFORE `date` (no look-ahead, same as live_daily_tail() ranking on "the prior available
    session"). Lets the offline signal-history generators use the EXACT SAME universe-selection
    rule the live endpoints use, date-by-date, instead of a separately-fixed backtest_universe()
    -- unified by explicit user decision (2026-08), superseding the split documented in this
    module's header."""
    m = _oi_lots_matrix()
    if m.empty:
        return set()
    d = pd.Timestamp(date).normalize()
    avail = m.index[m.index <= d]
    if len(avail) == 0:
        return set()
    row = m.loc[avail.max()].dropna().sort_values(ascending=False)
    return set(row.head(core_n_placeholder).index)


def live_universe_for_date(date) -> set:
    """live_universe_today(), but for a specific historical date -- NIFTY50 (fixed 50) + that
    date's top-15 most liquid non-Nifty50 names. Used by the offline signal-history generators
    so history is built from the exact same universe rule as the live picks panel."""
    try:
        return set(NIFTY50) | live_daily_tail_for_date(date)
    except Exception:
        return set(NIFTY50)


ALLOWLIST_FILE = SILVER_DATA_ROOT / "liquid_universe.txt"


def write_allowlist(days: int = 520) -> list[str]:
    """Persist NIFTY50 UNION the daily tail's names over the last `days` sessions (~2 trading
    years, matching the backtest window) to silver/liquid_universe.txt.
    pipeline/silver.py's build_eod_deriv_contracts unions this with active_universe.txt when
    deciding which symbols to keep per-strike option rows for -- the daily tail rotates, so the
    silver contracts table needs the historical union, not just today's live set, to keep the
    live endpoints from going blind on days the tail differs from today's."""
    m = _oi_lots_matrix()
    tail: set = set()
    if not m.empty:
        recent = m.loc[m.index.to_series().tail(days).index]
        for _, row in recent.iterrows():
            r = row.dropna().sort_values(ascending=False)
            tail |= set(r.head(DAILY_N).index)
    syms = sorted(set(NIFTY50) | tail)
    ALLOWLIST_FILE.write_text("\n".join(syms), encoding="utf-8")
    return syms


def load_allowlist() -> list[str]:
    if ALLOWLIST_FILE.exists():
        return [s.strip() for s in ALLOWLIST_FILE.read_text(encoding="utf-8").splitlines() if s.strip()]
    return []


def refresh_nifty50() -> None:
    """Not run automatically. Re-fetch https://archives.nseindia.com/content/indices/
    ind_nifty50list.csv (needs the pipeline/fetch.py-style NSE session handshake -- a plain
    request will likely 403) and hand-update the NIFTY50 constant above after NSE's next
    semi-annual rebalance. Deliberately manual: silently changing this set would change
    backtest results without anyone noticing."""
    raise NotImplementedError("update NIFTY50 by hand after checking NSE's latest constituent list")


if __name__ == "__main__":
    live = live_universe_today()
    tail = live_daily_tail()
    print(f"[liquid_universe] backtest universe: {len(NIFTY50)} (fixed Nifty50)")
    print(f"[liquid_universe] live universe: {len(live)} = 50 Nifty50 + {len(tail)} daily-liquid tail")
    print(f"  daily tail: {', '.join(sorted(tail))}")
    syms = write_allowlist()
    print(f"[liquid_universe] wrote {len(syms)} symbols (Nifty50 + rolling daily-tail union) -> {ALLOWLIST_FILE}")
