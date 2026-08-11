"""Zerodha F&O OPTIONS charges (not equity, not futures -- those have different STT/brokerage
formulas). Rates fetched live from https://zerodha.com/charges/ on 2026-08-11, not from
training-data memory -- these change periodically via government notification (most recently
the Oct-2024 STT hike from 0.0625%/0.1% to the current rate). Re-check the live page before
trusting these numbers far into the future.

Per-leg, per-side charges:
  - Brokerage:  min(Rs 20, 0.03% of premium turnover) per executed order
  - STT:        0.15% of premium, SELL side only (charged whether selling to open a short or
                selling to close a long -- not charged on any buy)
  - Transaction charges (NSE): 0.03553% of premium, both sides
  - GST:        18% on (brokerage + SEBI charges + transaction charges) -- NOT on STT or stamp
  - SEBI charges: Rs 10 / crore of turnover (= 0.0001%), both sides
  - Stamp duty: 0.003% of premium, BUY side only

A multi-leg spread (Condor/Broken-Wing: 4 legs; Skew: 2 legs) pays these PER LEG PER EXECUTION --
a 4-leg round trip is 8 executed orders (4 to open, 4 to close), not 1. This roughly doubles
Condor/Broken-Wing's brokerage drag vs Skew's for the same trade count, which matters when
comparing strategies net of costs, not just gross.
"""
from __future__ import annotations

BROKERAGE_FLAT = 20.0
BROKERAGE_PCT = 0.0003          # 0.03% -- min() with the flat fee below
STT_SELL_PCT = 0.0015           # 0.15%, sell side only
TXN_CHARGE_PCT = 0.0003553      # 0.03553%, NSE, both sides
GST_PCT = 0.18                  # on (brokerage + SEBI + transaction charges) only
SEBI_PCT = 0.000001             # Rs 10 / crore = 0.0001% = 1e-6, both sides
STAMP_DUTY_PCT = 0.00003        # 0.003%, buy side only


def leg_cost(premium: float, lot_size: int, is_buy: bool) -> float:
    """Total charges for ONE executed order (one leg, one side, one lot) at the given premium.
    `premium` is the per-unit option price (e.g. `sce["open"]`), not already multiplied by
    lot_size -- turnover = premium * lot_size is computed here."""
    turnover = premium * lot_size
    brokerage = min(BROKERAGE_FLAT, BROKERAGE_PCT * turnover)
    stt = 0.0 if is_buy else STT_SELL_PCT * turnover
    txn = TXN_CHARGE_PCT * turnover
    sebi = SEBI_PCT * turnover
    gst = GST_PCT * (brokerage + sebi + txn)
    stamp = STAMP_DUTY_PCT * turnover if is_buy else 0.0
    return brokerage + stt + txn + sebi + gst + stamp


def round_trip_cost(entry_legs: list[tuple[float, bool]], exit_legs: list[tuple[float, bool]],
                     lot_size: int) -> float:
    """Sums leg_cost() across every entry and exit leg. Each leg is (premium, is_buy) --
    e.g. a Condor entry is [(sell_ce_premium, False), (buy_ce_premium, True),
    (sell_pe_premium, False), (buy_pe_premium, True)], and its exit is the mirror image
    (buy back the short legs = is_buy True, sell the long legs = is_buy False) at the exit
    day's premiums."""
    return sum(leg_cost(p, lot_size, is_buy) for p, is_buy in entry_legs + exit_legs)


def condor_round_trip_cost(sell_ce_entry: float, buy_ce_entry: float, sell_pe_entry: float, buy_pe_entry: float,
                            sell_ce_exit: float, buy_ce_exit: float, sell_pe_exit: float, buy_pe_exit: float,
                            lot_size: int) -> float:
    """4-leg (Condor/Broken-Wing) round-trip cost. Exit closes each leg with the OPPOSITE side:
    the short legs are bought back (is_buy=True), the long legs are sold to close (is_buy=False)."""
    entry = [(sell_ce_entry, False), (buy_ce_entry, True), (sell_pe_entry, False), (buy_pe_entry, True)]
    exit_ = [(sell_ce_exit, True), (buy_ce_exit, False), (sell_pe_exit, True), (buy_pe_exit, False)]
    return round_trip_cost(entry, exit_, lot_size)


def vertical_round_trip_cost(short_entry: float, long_entry: float,
                              short_exit: float, long_exit: float, lot_size: int) -> float:
    """2-leg (Skew) round-trip cost. Same opposite-side-on-exit convention as condor_round_trip_cost."""
    entry = [(short_entry, False), (long_entry, True)]
    exit_ = [(short_exit, True), (long_exit, False)]
    return round_trip_cost(entry, exit_, lot_size)
