"""Builds a single NIFTY ATM long-straddle price path (CE+PE summed) for one entry event,
from the chain-shaped nifty_option_chain_5m data. Feeds directly into
experiments/intraday_exit_v1/simulator.py's ExitRule machinery (imported there, not copied
-- that module is fully generic, see plan/README).

No 1-minute NIFTY option execution data exists (unlike intraday_exit_v1's per-trade-sliced
stock legs), so decision and execution bars passed to simulate_multiresolution end up being
the SAME 5-min series -- its execution-bar requirement is just (timestamp, open, close), so
no fabricated straddle high/low is needed (the same reason intraday_exit_v1 itself never
feeds its diagnostic-only high_upper_bound/low_lower_bound into the simulator).
"""
from __future__ import annotations

import pandas as pd


def _front_expiry_and_atm(chain: pd.DataFrame, at: pd.Timestamp) -> tuple[pd.Timestamp, float] | None:
    """Front (nearest unexpired) expiry and its ATM strike, as of the entry timestamp's OWN
    chain snapshot -- i.e. what a trader actually sees at the moment of placing the trade."""
    snap = chain[chain["timestamp"] == at]
    snap = snap[snap["expiry"].dt.date >= at.date()]
    if snap.empty:
        return None
    front_expiry = snap["expiry"].min()
    snap = snap[snap["expiry"] == front_expiry].copy()
    snap["strike_dist"] = (snap["strike"] - snap["underlying_close"]).abs()
    atm_row = snap.sort_values("strike_dist").iloc[0]
    return front_expiry, float(atm_row["strike"])


def build_straddle_path(chain: pd.DataFrame, entry_signal_time: pd.Timestamp,
                         max_hold_bars: int) -> pd.DataFrame | None:
    """`chain`: option_features.load_chain() output (all expiry files concatenated).
    `entry_signal_time`: the bar whose CLOSE generated the breakout signal -- the trade
    enters at the NEXT bar's open (no look-ahead), so the returned path starts one bar after
    this timestamp. Never carries the position past the same trading day (no overnight
    NIFTY option risk in this model -- see labels.py's same-day-only forward windows).

    Returns a (timestamp, open, close) DataFrame -- straddle open/close = CE+PE summed --
    or None if the contract/data isn't available (e.g. very-late-session signal with no room
    left, or a gap in the chain download for that day)."""
    picked = _front_expiry_and_atm(chain, entry_signal_time)
    if picked is None:
        return None
    expiry, strike = picked

    leg = chain[(chain["expiry"] == expiry) & (chain["strike"] == strike)
                & (chain["timestamp"] > entry_signal_time)].sort_values("timestamp")
    same_day = leg[leg["timestamp"].dt.date == entry_signal_time.date()]
    if same_day.empty:
        return None
    same_day = same_day.head(max_hold_bars)

    # NOTE: build via .reset_index(drop=True) on each Series, not .values -- .values on a
    # tz-aware datetime Series silently strips the Asia/Kolkata offset (converts to naive
    # UTC), which would then fail to match against tz-aware timestamps everywhere else in
    # the pipeline (e.g. looking up the exit bar's underlying price by timestamp).
    out = pd.DataFrame({
        "timestamp": same_day["timestamp"].reset_index(drop=True),
        "open": (same_day["ce_open"].fillna(0) + same_day["pe_open"].fillna(0)).reset_index(drop=True),
        "close": (same_day["ce_close"].fillna(0) + same_day["pe_close"].fillna(0)).reset_index(drop=True),
    })
    out = out[(out["open"] > 0) & (out["close"] > 0)]
    if len(out) < 2:
        return None
    return out.reset_index(drop=True)
