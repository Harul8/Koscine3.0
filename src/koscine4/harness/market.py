"""Market data for the harness and the view a strategy is allowed to see (requirements V-4, V-5, V-6).

`MarketData` holds history for a run. A strategy never touches it. It gets a `MarketView`,
which answers questions only as of one decision (a date and a time) and raises `LookAheadError`
if asked for anything later. The harness itself reads later bars to fill and mark trades.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from koscine4.features.registry import REGISTRY
from koscine4.settings import Settings
from koscine4.store.api import Store
from koscine4.store.availability import Decision
from koscine4.store.tables import PRICES_RAW, StoreError, read_table


class LookAheadError(RuntimeError):
    """Raised when a strategy asks for data later than its decision can see."""


class IntradayMarks:
    """Option prices at one clock time from the 5-minute index chain files, where they exist."""

    FOLDERS = {"NIFTY": "nifty_option_chain_5m", "BANKNIFTY": "banknifty_option_chain_5m"}

    def __init__(self, settings: Settings, instrument: str, hhmm: str = "15:15"):
        self.table: dict[tuple, tuple[float, float]] = {}
        self.dates: set[pd.Timestamp] = set()
        folder = settings.path("intraday") / self.FOLDERS.get(instrument, "_none_")
        cols = ["timestamp", "expiry", "strike", "ce_close", "ce_volume", "pe_close", "pe_volume"]
        for f in sorted(folder.glob("*.parquet")) if folder.exists() else []:
            try:
                x = pd.read_parquet(f, columns=cols)
            except Exception:
                continue
            x = x[x["timestamp"].dt.strftime("%H:%M") == hhmm]
            if x.empty:
                continue
            day = x["timestamp"].dt.tz_localize(None).dt.normalize() if x["timestamp"].dt.tz is not None else x["timestamp"].dt.normalize()
            expiry = pd.to_datetime(x["expiry"])
            for d, e, k, cc, cv, pc, pv in zip(day, expiry, x["strike"], x["ce_close"], x["ce_volume"], x["pe_close"], x["pe_volume"]):
                self.dates.add(d)
                if cc == cc and cc > 0:
                    self.table[(d, e, float(k), "CE")] = (float(cc), float(cv) if cv == cv else 0.0)
                if pc == pc and pc > 0:
                    self.table[(d, e, float(k), "PE")] = (float(pc), float(pv) if pv == pv else 0.0)

    def has_day(self, date) -> bool:
        return pd.Timestamp(date) in self.dates

    def price(self, date, expiry, strike, opt_type):
        return self.table.get((pd.Timestamp(date), pd.Timestamp(expiry), float(strike), opt_type))


class MarketData:
    """History for one run, loaded once."""

    def __init__(self, settings: Settings, decision_time: str = "evening", predictions: dict[str, pd.DataFrame] | None = None,
                 features: pd.DataFrame | None = None, use_intraday: bool = True):
        self.settings = settings
        self.store = Store(settings)
        self.calendar = self.store.calendar
        self.decision_time = decision_time
        raw = read_table(settings, PRICES_RAW, columns=["date", "instrument", "open", "close"])
        self._raw = {k: v.set_index("date") for k, v in raw.groupby("instrument", sort=False)}
        self._chains: dict[str, dict] = {}
        self._intraday: dict[str, IntradayMarks] = {}
        self.use_intraday = use_intraday
        self.predictions = {k: v.set_index(["date", "instrument"])["prediction"] for k, v in (predictions or {}).items()}
        if features is None:
            from koscine4.features.build import table_path

            path = table_path(settings, decision_time)
            features = pd.read_parquet(path) if path.exists() else None
        self.features = None if features is None else features.set_index(["date", "instrument"]).sort_index()
        ev = self.store.all_events()
        self._events = {k: v for k, v in ev.assign(known=ev["announced_on"].fillna(ev["event_date"])).groupby("instrument", sort=False)} if len(ev) else {}

    # ---- raw access for the harness (fills and marks). Not given to strategies.
    def chain(self, instrument: str, date) -> pd.DataFrame:
        if instrument not in self._chains:
            try:
                q = self.store._quotes(instrument)
            except StoreError:
                q = pd.DataFrame(columns=["date"])
            self._chains[instrument] = {d: g for d, g in q.groupby("date", sort=False)}
        return self._chains[instrument].get(pd.Timestamp(date), pd.DataFrame())

    def raw_price(self, instrument: str, date, field: str = "close") -> float:
        df = self._raw.get(instrument)
        if df is None or pd.Timestamp(date) not in df.index:
            return float("nan")
        return float(df.at[pd.Timestamp(date), field])

    def intraday(self, instrument: str) -> IntradayMarks | None:
        if not self.use_intraday or instrument not in IntradayMarks.FOLDERS:
            return None
        if instrument not in self._intraday:
            self._intraday[instrument] = IntradayMarks(self.settings, instrument)
        return self._intraday[instrument]

    def events_for(self, instrument: str) -> pd.DataFrame | None:
        return self._events.get(instrument)

    def view(self, date, time: str | None = None) -> "MarketView":
        return MarketView(self, Decision(date, time or self.decision_time))


@dataclass
class MarketView:
    """What a strategy may know at one decision. Every answer is as of `decision`."""

    data: MarketData
    decision: Decision

    def __post_init__(self):
        self.visible = self.data.store.last_visible("prices_raw", self.decision)       # last full day it can read

    @property
    def date(self) -> pd.Timestamp:
        return self.decision.date

    def _guard(self, date) -> pd.Timestamp:
        date = pd.Timestamp(date)
        if self.visible is None or date > self.visible:
            raise LookAheadError(f"A decision at {self.decision.date.date()} {self.decision.time} cannot read {date.date()}.")
        return date

    def spot(self, instrument: str) -> float:
        """The last full-day close the decision can see (raw price, to match strikes)."""
        return float("nan") if self.visible is None else self.data.raw_price(instrument, self.visible)

    def chain(self, instrument: str) -> pd.DataFrame:
        """The last full-day option chain the decision can see."""
        return pd.DataFrame() if self.visible is None else self.data.chain(instrument, self.visible)

    def chain_on(self, instrument: str, date) -> pd.DataFrame:
        return self.data.chain(instrument, self._guard(date))

    def close_on(self, instrument: str, date) -> float:
        return self.data.raw_price(instrument, self._guard(date))

    def feature(self, instrument: str, name: str) -> float:
        """A registered feature for this decision. Unregistered names raise (F-1)."""
        REGISTRY.require([name])
        f = self.data.features
        if f is None:
            raise StoreError(f"No feature table for '{self.data.decision_time}'. Run: python -m koscine4.features.build --time {self.data.decision_time}")
        try:
            return float(f.at[(self.decision.date, instrument), name])
        except KeyError:
            return float("nan")

    def prediction(self, model: str, instrument: str) -> float:
        p = self.data.predictions.get(model)
        if p is None:
            raise StoreError(f"No predictions loaded for model '{model}'.")
        try:
            return float(p.at[(self.decision.date, instrument)])
        except KeyError:
            return float("nan")

    def results_within(self, instrument: str, sessions: int) -> bool:
        """True if an announced result date falls in the next `sessions` sessions (today included at 14:30)."""
        ev = self.data.events_for(instrument)
        if ev is None or self.visible is None:
            return False
        cal = self.data.calendar
        i = cal.index_on_or_before(self.decision.date)
        last = cal.days[min(i + sessions, len(cal.days) - 1)] if i + sessions < len(cal.days) else self.decision.date + pd.Timedelta(days=int(sessions * 1.6) + 1)
        start = self.decision.date if self.decision.time != "evening" else self.decision.date + pd.Timedelta(days=1)
        known = ev[ev["known"] <= self.visible]
        return bool(((known["event_date"] >= start) & (known["event_date"] <= last)).any())

    def stock_list(self) -> list[str]:
        from koscine4.universe.build import stock_list

        return stock_list(self.data.settings, self.decision.date)
