"""Walk-forward splits (requirements V-2, V-3, M-4).

Train up to a cutoff, test the following month, roll forward one month. The cutoff sits a set
number of sessions before the first test date, and no training outcome may end after it. There
is no random split anywhere in this package.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from koscine4.store.calendar import TradingCalendar


class GapError(RuntimeError):
    """Raised when training outcomes come closer to the test dates than the required gap."""


@dataclass(frozen=True)
class Fold:
    cutoff: pd.Timestamp          # the last date a training outcome's window may end
    test_start: pd.Timestamp      # first session of the test month (the day the stock list is rebuilt)
    test_end: pd.Timestamp        # last session of the test month


def monthly_folds(calendar: TradingCalendar, start, end, gap_sessions: int, min_gap: int) -> list[Fold]:
    """One fold per calendar month whose first session lies in [start, end]."""
    if gap_sessions < min_gap:
        raise GapError(f"The gap is {gap_sessions} sessions; at least {min_gap} are required (V-3).")
    days = calendar.days
    in_range = days[(days >= pd.Timestamp(start)) & (days <= pd.Timestamp(end))]
    folds = []
    months = pd.Series(in_range, index=in_range).groupby(in_range.to_period("M"))
    for _, sessions in months:
        first = sessions.iloc[0]
        if first != days[days.to_period("M") == first.to_period("M")][0]:
            continue                                   # a partial first month: not a rebuild date
        i = calendar.index_on_or_before(first) - gap_sessions
        if i < 0:
            continue
        folds.append(Fold(days[i], first, sessions.iloc[-1]))
    return folds


def check_fold(fold: Fold, train_window_ends: pd.Series, calendar: TradingCalendar, min_gap: int) -> None:
    """Stop if any training outcome ends after the cutoff, or the cutoff is too close to the test start."""
    gap = calendar.index_on_or_before(fold.test_start) - calendar.index_on_or_before(fold.cutoff)
    if gap < min_gap:
        raise GapError(f"Fold {fold.test_start.date()}: gap of {gap} sessions is below the required {min_gap} (V-3).")
    if len(train_window_ends) and pd.Timestamp(train_window_ends.max()) > fold.cutoff:
        raise GapError(f"Fold {fold.test_start.date()}: a training outcome ends after the cutoff {fold.cutoff.date()} (M-4).")
