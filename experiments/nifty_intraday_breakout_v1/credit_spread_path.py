"""Builds a NIFTY intraday defined-risk credit-spread P&L path (Broken-Wing style:
asymmetric narrow-call/wide-put wings -- the same direction that already won for stocks,
see BW_NOTE in api/main.py) from the chain-shaped nifty_option_chain_5m data.

Mirrors analysis/gen_sell_signal_history.py's EOD mark-to-market convention (credit, value
clamped to [0, width], pnl = credit - value, ror_pct = pnl/max_risk*100) at 5-min
granularity, rather than intraday_exit_v1.simulator's long-position ratio-return math --
that's the wrong shape for a credit spread: the seller wants value to DECREASE, and the
meaningful metric is P&L as a fraction of max risk, not a price ratio.
"""
from __future__ import annotations

import pandas as pd


def _front_expiry(chain: pd.DataFrame, at: pd.Timestamp) -> pd.Timestamp | None:
    # Filter sequentially (timestamp equality first) rather than one combined boolean mask --
    # pandas evaluates both sides of `&` on the FULL frame before combining, so `.dt.date`
    # (a slow per-row Python-object conversion, not vectorized) would otherwise run across
    # all ~14.5M rows on every call. Narrowing to `snap` first makes it run on a handful of
    # rows instead. Same reasoning applied throughout this module -- see open_credit_spread.
    snap = chain[chain["timestamp"] == at]
    if snap.empty:
        return None
    snap = snap[snap["expiry"].dt.date >= at.date()]
    return snap["expiry"].min() if not snap.empty else None


def open_credit_spread(chain: pd.DataFrame, entry_signal_time: pd.Timestamp,
                        short_otm: float, wing_ce: float, wing_pe: float,
                        max_hold_bars: int) -> dict | None:
    """Sells the ~short_otm% OTM CE+PE, buys the wing_ce%/wing_pe% wings (asymmetric --
    narrow call / wide put), strikes chosen from the entry_signal_time snapshot (the bar
    whose close generated the entry decision), filled at the NEXT bar's open (no
    look-ahead). Returns None if strikes/contract data are unavailable, else a dict with
    entry metadata (credit, width, max_risk, strikes, entry_ts) plus a `path` DataFrame
    (timestamp, open_value, close_value) of the spread's cost-to-close, clamped to
    [0, width], for up to max_hold_bars bars starting at entry."""
    expiry = _front_expiry(chain, entry_signal_time)
    if expiry is None:
        return None
    entry_snap = chain[(chain["timestamp"] == entry_signal_time) & (chain["expiry"] == expiry)]
    if entry_snap.empty:
        return None
    u = float(entry_snap["underlying_close"].iloc[0])

    def nearest_strike(target: float) -> float | None:
        d = entry_snap.assign(dist=(entry_snap["strike"] - target).abs()).sort_values("dist")
        return float(d.iloc[0]["strike"]) if not d.empty else None

    short_ce = nearest_strike(u * (1 + short_otm))
    long_ce = nearest_strike(u * (1 + short_otm + wing_ce))
    short_pe = nearest_strike(u * (1 - short_otm))
    long_pe = nearest_strike(u * (1 - short_otm - wing_pe))
    if any(k is None for k in (short_ce, long_ce, short_pe, long_pe)):
        return None
    if long_ce <= short_ce or long_pe >= short_pe:   # degenerate/too-tight strike spacing
        return None

    # Same sequential-narrowing reasoning as _front_expiry: filter by the two selective,
    # vectorized-and-cheap equality checks (expiry, then the 4-strike isin) FIRST, so the
    # `.dt.date` day-boundary check below only ever runs on a tiny already-narrowed slice
    # instead of the full ~14.5M-row chain on every checkpoint call.
    day_chain = chain[chain["expiry"] == expiry]
    day_chain = day_chain[day_chain["strike"].isin({short_ce, long_ce, short_pe, long_pe})]
    day_chain = day_chain[day_chain["timestamp"] > entry_signal_time]
    day_chain = day_chain[day_chain["timestamp"].dt.date == entry_signal_time.date()]
    if day_chain.empty:
        return None

    piv = day_chain.pivot_table(index="timestamp", columns="strike",
                                 values=["ce_open", "ce_close", "pe_open", "pe_close"])
    ts_all = sorted(piv.index.unique())[:max_hold_bars]
    if not ts_all:
        return None
    piv = piv.loc[ts_all]

    def col(field: str, strike: float):
        try:
            return piv[(field, strike)]
        except KeyError:
            return None

    ce_o_s, ce_o_l = col("ce_open", short_ce), col("ce_open", long_ce)
    pe_o_s, pe_o_l = col("pe_open", short_pe), col("pe_open", long_pe)
    ce_c_s, ce_c_l = col("ce_close", short_ce), col("ce_close", long_ce)
    pe_c_s, pe_c_l = col("pe_close", short_pe), col("pe_close", long_pe)
    if any(s is None for s in (ce_o_s, ce_o_l, pe_o_s, pe_o_l, ce_c_s, ce_c_l, pe_c_s, pe_c_l)):
        return None

    call_width = long_ce - short_ce
    put_width = short_pe - long_pe
    width = max(call_width, put_width)

    entry_row = ts_all[0]
    sell_premium = float(ce_o_s.loc[entry_row]) + float(pe_o_s.loc[entry_row])
    buy_premium = float(ce_o_l.loc[entry_row]) + float(pe_o_l.loc[entry_row])
    if pd.isna(sell_premium) or pd.isna(buy_premium) or sell_premium <= buy_premium:
        return None
    credit = sell_premium - buy_premium
    max_risk = width - credit
    if max_risk <= 0:
        return None

    open_value = ((ce_o_s - ce_o_l) + (pe_o_s - pe_o_l)).clip(lower=0.0, upper=width)
    close_value = ((ce_c_s - ce_c_l) + (pe_c_s - pe_c_l)).clip(lower=0.0, upper=width)
    path = pd.DataFrame({"timestamp": ts_all, "open_value": open_value.values,
                          "close_value": close_value.values}).dropna()
    if path.empty:
        return None

    return {"expiry": expiry,
            "strikes": {"short_ce": short_ce, "long_ce": long_ce, "short_pe": short_pe, "long_pe": long_pe},
            "credit": credit, "width": width, "max_risk": max_risk, "entry_ts": ts_all[0], "path": path}


def simulate_exit(spread: dict, stop_loss_frac: float | None) -> dict:
    """stop_loss_frac: e.g. -0.5 means close if unrealized P&L drops to -50% of max_risk.
    None = hold to the end of the path (window), no stop. This is a seller's risk control
    (cap losses, let theta decay do the rest) -- not a trailing-activation rule, which is a
    buyer's tool and the wrong shape for a credit position. Trigger evaluated on a
    completed bar's close_value, filled at the NEXT bar's open_value (no look-ahead); the
    entry bar itself (path row 0) can't also trigger an exit, same discipline as
    intraday_exit_v1.simulator's "first decision bar reserved for entry."""
    path = spread["path"].reset_index(drop=True)
    credit, max_risk = spread["credit"], spread["max_risk"]
    dd = 0.0
    trigger_idx = None
    for i in range(1, len(path)):
        unrealized = credit - float(path.loc[i, "close_value"])
        dd = min(dd, unrealized)
        if stop_loss_frac is not None and unrealized <= stop_loss_frac * max_risk:
            trigger_idx = i
            break
    if trigger_idx is None:
        exit_value = float(path["close_value"].iloc[-1])
        exit_ts = path["timestamp"].iloc[-1]
        reason = "time_exit"
    else:
        exit_idx = min(trigger_idx + 1, len(path) - 1)
        exit_value = float(path.loc[exit_idx, "open_value"])
        exit_ts = path.loc[exit_idx, "timestamp"]
        reason = "stop_loss"
    pnl = credit - exit_value
    return {
        "entry_ts": spread["entry_ts"], "exit_ts": exit_ts,
        "credit": round(credit, 2), "max_risk": round(max_risk, 2), "exit_value": round(exit_value, 2),
        "pnl": round(pnl, 2), "ror_pct": round(pnl / max_risk * 100, 2),
        "max_dd_pct": round(dd / max_risk * 100, 2),
        "exit_reason": reason, "outcome": "win" if pnl > 0 else "loss",
    }
