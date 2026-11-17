"""Trading calendar: the ordered list of sessions, taken from the data itself."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


class CalendarError(ValueError):
    """Raised when a date falls outside the calendar."""


@dataclass(frozen=True)
class TradingCalendar:
    days: pd.DatetimeIndex

    @classmethod
    def from_dates(cls, dates) -> "TradingCalendar":
        idx = pd.DatetimeIndex(pd.to_datetime(pd.Index(dates))).normalize().unique().sort_values()
        if len(idx) == 0:
            raise CalendarError("A trading calendar needs at least one session.")
        return cls(idx)

    @classmethod
    def from_prices(cls, prices: pd.DataFrame, min_symbols: int) -> "TradingCalendar":
        """A date is a session when at least `min_symbols` stocks traded on it."""
        stocks = prices[prices["kind"] == "stock"] if "kind" in prices.columns else prices
        counts = stocks.groupby("date")["instrument"].nunique()
        return cls.from_dates(counts[counts >= min_symbols].index)

    def __len__(self) -> int:
        return len(self.days)

    def is_session(self, date) -> bool:
        return pd.Timestamp(date).normalize() in self.days

    def index_on_or_before(self, date) -> int:
        """Position of the last session on or before `date`; -1 if there is none."""
        return int(np.searchsorted(self.days.values, np.datetime64(pd.Timestamp(date).normalize()), side="right")) - 1

    def on_or_before(self, date) -> pd.Timestamp:
        i = self.index_on_or_before(date)
        if i < 0:
            raise CalendarError(f"{pd.Timestamp(date).date()} is before the first session {self.days[0].date()}.")
        return self.days[i]

    def shift(self, date, sessions: int) -> pd.Timestamp:
        """The session `sessions` steps from the last session on or before `date`."""
        i = self.index_on_or_before(date) + sessions
        if i < 0 or i >= len(self.days):
            raise CalendarError(
                f"Shifting {pd.Timestamp(date).date()} by {sessions} sessions leaves the calendar "
                f"({self.days[0].date()} to {self.days[-1].date()})."
            )
        return self.days[i]

    def between(self, start, end) -> pd.DatetimeIndex:
        return self.days[(self.days >= pd.Timestamp(start)) & (self.days <= pd.Timestamp(end))]
