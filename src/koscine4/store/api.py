"""The one way to read the store. Every read names the decision it is made for (design rule 2)."""
from __future__ import annotations

from functools import cached_property

import pandas as pd

from koscine4.settings import Settings, load_settings
from koscine4.store import events as _events
from koscine4.store import options as _options
from koscine4.store.availability import Decision, last_visible_date, source_availability
from koscine4.store.calendar import TradingCalendar
from koscine4.store.prices import adjust_prices
from koscine4.store import surface as _surface
from koscine4.store.symbols import canonicalize, load_changes
from koscine4.store.market_value import MARKET_VALUE
from koscine4.store.tables import ADJUSTMENTS, PRICES_RAW, read_table


class Store:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or load_settings()

    # ---- cached tables -------------------------------------------------------------------
    @cached_property
    def _raw(self) -> pd.DataFrame:
        return read_table(self.settings, PRICES_RAW)

    @cached_property
    def _adjustments(self) -> pd.DataFrame:
        return read_table(self.settings, ADJUSTMENTS)

    @cached_property
    def calendar(self) -> TradingCalendar:
        return TradingCalendar.from_prices(self._raw, int(self.settings.section("store")["min_symbols_for_session"]))

    @cached_property
    def _events(self) -> pd.DataFrame:
        return _events.load_events(self.settings)

    # ---- availability --------------------------------------------------------------------
    def last_visible(self, source: str, asof: Decision) -> pd.Timestamp | None:
        """Latest row date of `source` that a decision at `asof` may read."""
        rule = source_availability(self.settings.section("availability"), source)
        return last_visible_date(rule, asof, self.calendar)

    # ---- reads ---------------------------------------------------------------------------
    def prices(self, instruments=None, start=None, end=None, *, asof: Decision, adjusted: bool = True) -> pd.DataFrame:
        """Daily prices visible at `asof`. Adjusted prices use only corporate actions up to the visible date."""
        visible = self.last_visible("prices_raw", asof)
        raw = self._raw
        if visible is None:
            return raw.iloc[0:0].copy()
        if instruments is not None:
            raw = raw[raw["instrument"].isin([instruments] if isinstance(instruments, str) else list(instruments))]
        raw = raw[raw["date"] <= visible]
        out = adjust_prices(raw, self._adjustments, visible) if adjusted else raw.copy()
        if start is not None:
            out = out[out["date"] >= pd.Timestamp(start)]
        if end is not None:
            out = out[out["date"] <= pd.Timestamp(end)]
        return out.reset_index(drop=True)

    def adjustments(self, *, asof: Decision) -> pd.DataFrame:
        visible = self.last_visible("prices_raw", asof)
        adj = self._adjustments
        return adj.iloc[0:0].copy() if visible is None else adj[adj["ex_date"] <= visible].reset_index(drop=True)

    def market_value(self, *, asof: Decision) -> pd.DataFrame:
        """Market value per stock and day, up to the last price date visible at `asof` (D-11)."""
        visible = self.last_visible("prices_raw", asof)
        mv = read_table(self.settings, MARKET_VALUE)
        return mv.iloc[0:0] if visible is None else mv[mv["date"] <= visible].reset_index(drop=True)

    def universe(self, *, asof: Decision, open_instruments=()) -> list[str]:
        """Instruments tradeable at `asof`: the indices, the month's stock list, and open trades (U-1 to U-5)."""
        from koscine4.universe.build import tradeable

        return tradeable(self.settings, asof.date, open_instruments)

    def option_chain(self, instrument: str, *, asof: Decision) -> pd.DataFrame:
        """The latest full-day option and futures chain visible at `asof`."""
        visible = self.last_visible("option_quotes", asof)
        if visible is None:
            raise LookupError(f"No option data is visible at {asof}.")
        return _options.chain_on(self.settings, instrument, visible)

    def _flow(self, file: str, source: str, asof: Decision) -> pd.DataFrame:
        visible = self.last_visible(source, asof)
        df = pd.read_parquet(self.settings.path("silver") / file)
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        return df.iloc[0:0] if visible is None else df[df["date"] <= visible].sort_values("date").reset_index(drop=True)

    def fii_dii(self, *, asof: Decision) -> pd.DataFrame:
        return self._flow("fii_dii_cash.parquet", "flows_fii_dii", asof)

    def participant_oi(self, *, asof: Decision) -> pd.DataFrame:
        return self._flow("participant_oi.parquet", "flows_participant_oi", asof)

    def all_events(self) -> pd.DataFrame:
        """Every event with its announcement date. For builders that apply visibility row by row."""
        return self._events

    @cached_property
    def _deriv(self) -> pd.DataFrame:
        cols = ["date", "symbol", "fut_close", "fut_oi", "opt_call_oi", "opt_put_oi", "opt_call_vol", "opt_put_vol"]
        d = pd.read_parquet(self.settings.path("silver") / "eod_deriv_daily.parquet", columns=cols)
        d["date"] = pd.to_datetime(d["date"]).dt.normalize()
        d["instrument"] = canonicalize(d["symbol"], d["date"], load_changes(self.settings))
        return d.drop(columns="symbol").drop_duplicates(["date", "instrument"], keep="last")

    def deriv_daily(self, *, asof: Decision) -> pd.DataFrame:
        """Per-instrument daily futures and option totals visible at `asof`."""
        visible = self.last_visible("option_quotes", asof)
        d = self._deriv
        return d.iloc[0:0] if visible is None else d[d["date"] <= visible].reset_index(drop=True)

    @cached_property
    def _surface(self) -> pd.DataFrame:
        return _surface.read_surface(self.settings)

    def surface(self, *, asof: Decision) -> pd.DataFrame:
        """Option surface rows (one per instrument, day and expiry) visible at `asof` (D-6)."""
        visible = self.last_visible("option_quotes", asof)
        s = self._surface
        return s.iloc[0:0] if visible is None else s[s["date"] <= visible].reset_index(drop=True)

    def quotes(self, instrument: str, *, asof: Decision) -> pd.DataFrame:
        """Every stored option and futures quote of one instrument up to the day visible at `asof`."""
        visible = self.last_visible("option_quotes", asof)
        q = self._quotes(instrument)
        return q.iloc[0:0] if visible is None else q[q["date"] <= visible]

    def _quotes(self, instrument: str) -> pd.DataFrame:
        cache = self.__dict__.setdefault("_quote_cache", {})
        if instrument not in cache:
            cache[instrument] = _surface.read_quotes(self.settings, instrument)
        return cache[instrument]

    def intraday_bars(self, *, asof: Decision) -> pd.DataFrame:
        """5-minute index bars that had finished by `asof`: at 14:30, today's bars stamped before 14:30."""
        frames = []
        for instrument, file in (("NIFTY", "nifty_5m.parquet"), ("BANKNIFTY", "banknifty_5m.parquet")):
            path = self.settings.path("intraday") / file
            if path.exists():
                b = pd.read_parquet(path, columns=["timestamp", "open", "high", "low", "close"])
                b["instrument"] = instrument
                frames.append(b)
        if not frames:
            return pd.DataFrame(columns=["timestamp", "instrument", "open", "high", "low", "close"])
        bars = pd.concat(frames, ignore_index=True)
        if bars["timestamp"].dt.tz is not None:
            bars["timestamp"] = bars["timestamp"].dt.tz_localize(None)
        day = bars["timestamp"].dt.normalize()
        if asof.time == "evening":
            return bars[day <= asof.date]
        if asof.time == "preopen":
            return bars[day < asof.date]
        return bars[(day < asof.date) | ((day == asof.date) & (bars["timestamp"].dt.strftime("%H:%M") < "14:30"))]

    def events(self, *, asof: Decision) -> pd.DataFrame:
        return _events.visible_events(self._events, self.last_visible("events", asof)).reset_index(drop=True)
