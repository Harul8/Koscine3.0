"""Choose which candidates trade (requirements S-5 to S-13, DR-7).

One function is used by the harness and by the daily run. It never adds a trade to fill a
slot: fewer than five, or none, is a valid week. Every refusal carries its reason.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping

import pandas as pd

INDEX_BOOK = ("NIFTY", "BANKNIFTY")


@dataclass
class BookState:
    """What is already on, for the limits to count against."""

    open_instruments: set = field(default_factory=set)
    week_counts: dict = field(default_factory=dict)        # (iso year, iso week) -> {"index": n, "stock": n}

    def started(self, date, book: str | None = None) -> int:
        y, w, _ = pd.Timestamp(date).isocalendar()
        counts = self.week_counts.get((y, w), {})
        return sum(counts.values()) if book is None else counts.get(book, 0)

    def record_start(self, date, book: str) -> None:
        y, w, _ = pd.Timestamp(date).isocalendar()
        self.week_counts.setdefault((y, w), {}).setdefault(book, 0)
        self.week_counts[(y, w)][book] += 1


def group_of(instrument: str, groups: Mapping[str, list]) -> str | None:
    for name, members in groups.items():
        if instrument in members:
            return name
    return None


def select(candidates, state: BookState, entry_date, limits: Mapping, groups: Mapping[str, list] | None = None,
           gate_passed: Callable[[str], bool] | None = None, direction_passed: Callable[[str], bool] | None = None,
           apply_weekly_cap: bool = True, max_per_day: int | None = None):
    """(accepted, refusals). Candidates are taken best score first until a limit stops them.

    limits: max_new_trades_per_week, max_open_per_instrument, optional index_book_trades_per_week,
            stock_book_trades_per_week, max_correlated_trades.
    gate_passed(strategy) is checked when given (live lists only, S-11).
    direction_passed(strategy) is checked for any candidate that uses a direction forecast (S-12, DR-7).
    """
    groups = groups or {}
    accepted, refused = [], []
    open_now = set(state.open_instruments)
    weekly = int(limits["max_new_trades_per_week"])
    per_book = {"index": limits.get("index_book_trades_per_week"), "stock": limits.get("stock_book_trades_per_week")}
    max_corr = limits.get("max_correlated_trades")
    started = state.started(entry_date)
    started_book = {b: state.started(entry_date, b) for b in ("index", "stock")}
    group_open: dict[str, int] = {}
    for inst in open_now:
        g = group_of(inst, groups)
        if g:
            group_open[g] = group_open.get(g, 0) + 1

    def refuse(c, reason):
        refused.append({"instrument": c.instrument, "strategy": c.strategy, "score": c.score, "reason": reason})

    for c in sorted(candidates, key=lambda c: (-c.score, c.instrument, c.strategy)):
        if gate_passed is not None and not gate_passed(c.strategy):
            refuse(c, "strategy_has_no_gate_2_record")
            continue
        if c.uses_direction and direction_passed is not None and not direction_passed(c.strategy):
            refuse(c, "direction_model_has_not_passed_its_gate")
            continue
        if c.instrument in open_now:
            refuse(c, "instrument_already_has_an_open_trade")
            continue
        if max_per_day is not None and len(accepted) >= int(max_per_day):
            refuse(c, "daily_limit")
            continue
        if apply_weekly_cap:
            if started >= weekly:
                refuse(c, "weekly_limit")
                continue
            cap = per_book.get(c.book)
            if cap is not None and started_book[c.book] >= int(cap):
                refuse(c, f"weekly_limit_{c.book}_book")
                continue
        g = group_of(c.instrument, groups)
        if g and max_corr is not None and group_open.get(g, 0) >= int(max_corr):
            refuse(c, f"correlated_limit:{g}")
            continue
        accepted.append(c)
        open_now.add(c.instrument)
        started += 1
        started_book[c.book] += 1
        if g:
            group_open[g] = group_open.get(g, 0) + 1
    return accepted, refused
