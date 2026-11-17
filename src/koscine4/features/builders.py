"""Feature formulas. Each builder returns values keyed by (date, instrument), where `date` is the
date of the source rows used: a value on date D depends on rows dated D or earlier and on nothing
else. Shifting to a decision time happens afterwards, in `features.build`, in one place.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from koscine4.store.calendar import TradingCalendar

ANNUAL = np.sqrt(252.0)
INDEX_NAMES = ("NIFTY", "BANKNIFTY")
VIX = "INDIAVIX"


@dataclass(frozen=True)
class Inputs:
    """Everything the builders read. `truncated` gives the same inputs with later rows removed."""

    prices: pd.DataFrame            # adjusted daily prices: date, instrument, kind, open, high, low, close, prev_close, volume
    surface: pd.DataFrame           # near expiry per (date, instrument): atm_iv, next_iv, skew, straddle_pct, dte
    deriv: pd.DataFrame             # date, instrument, fut_close, fut_oi, opt_call_oi, opt_put_oi, opt_call_vol, opt_put_vol
    fii_dii: pd.DataFrame           # date, fii_net, dii_net
    participant: pd.DataFrame       # date, participant, fut_idx_long, fut_idx_short, opt_idx_*
    events: pd.DataFrame            # instrument, event_date, announced_on
    calendar: TradingCalendar
    intraday: pd.DataFrame | None = None   # timestamp, instrument, open, high, low, close (5-minute bars)

    def truncated(self, date) -> "Inputs":
        date = pd.Timestamp(date)
        cut = lambda df: df[df["date"] <= date] if df is not None and len(df) else df
        ev = self.events
        if len(ev):
            ev = ev[ev["announced_on"].fillna(ev["event_date"]) <= date]
        intraday = self.intraday
        if intraday is not None and len(intraday):
            intraday = intraday[intraday["timestamp"].dt.normalize() <= date]
        return replace(self, prices=cut(self.prices), surface=cut(self.surface), deriv=cut(self.deriv),
                       fii_dii=cut(self.fii_dii), participant=cut(self.participant), events=ev,
                       calendar=TradingCalendar(self.calendar.days[self.calendar.days <= date]), intraday=intraday)


def _roll(g, col: str, n: int, how: str, min_periods: int | None = None) -> pd.Series:
    r = g[col].rolling(n, min_periods=min_periods or n)
    return getattr(r, how)().reset_index(level=0, drop=True)


def _pctile(g, col: str, n: int = 252, min_periods: int = 60) -> pd.Series:
    return g[col].rolling(n, min_periods=min_periods).rank(pct=True).reset_index(level=0, drop=True)


def _lag(g, col: str, n: int) -> pd.Series:
    return g[col].shift(n)


def price_features(inp: Inputs) -> pd.DataFrame:
    p = inp.prices[inp.prices["instrument"] != VIX].sort_values(["instrument", "date"]).reset_index(drop=True)
    p = p[["date", "instrument", "open", "high", "low", "close", "prev_close", "volume"]].copy()
    with np.errstate(divide="ignore", invalid="ignore"):
        p["lr"] = np.log(p["close"] / p["prev_close"])
        p["hl2"] = np.log(p["high"] / p["low"]) ** 2
    p["tr"] = np.maximum(p["high"] - p["low"], np.maximum((p["high"] - p["prev_close"]).abs(), (p["low"] - p["prev_close"]).abs()))
    g = p.groupby("instrument", sort=False)
    out = p[["date", "instrument"]].copy()
    out["ret_1d"] = p["close"] / p["prev_close"] - 1.0
    out["ret_5d"] = p["close"] / _lag(g, "close", 5) - 1.0
    out["ret_20d"] = p["close"] / _lag(g, "close", 20) - 1.0
    for n in (20, 50, 200):
        out[f"dist_ma{n}"] = p["close"] / _roll(g, "close", n, "mean") - 1.0
    out["gap_open"] = p["open"] / p["prev_close"] - 1.0
    for n in (5, 20, 60):
        out[f"rv_{n}"] = _roll(g, "lr", n, "std") * ANNUAL
    out["rv_ratio_5_20"] = out["rv_5"] / out["rv_20"]
    out["parkinson_10"] = np.sqrt(_roll(g, "hl2", 10, "mean") / (4.0 * np.log(2.0))) * ANNUAL
    out["atr_pct_14"] = _roll(g, "tr", 14, "mean") / p["close"]
    out["range_pct"] = (p["high"] - p["low"]) / p["close"]
    vol_mean = _roll(g, "volume", 20, "mean")
    out["volume_ratio_20"] = (p["volume"] / vol_mean).where(vol_mean > 0)
    out["rv_20_pctile"] = _pctile(out.groupby("instrument", sort=False), "rv_20")
    return out


def option_features(inp: Inputs, price: pd.DataFrame | None = None) -> pd.DataFrame:
    price = price_features(inp) if price is None else price
    base = price[["date", "instrument", "rv_20"]]
    s = inp.surface[["date", "instrument", "atm_iv", "next_iv", "skew", "straddle_pct", "dte"]] if len(inp.surface) else \
        pd.DataFrame(columns=["date", "instrument", "atm_iv", "next_iv", "skew", "straddle_pct", "dte"])
    d = inp.deriv
    # futures are quoted in raw terms, so the basis needs the raw close: adjusted close / adjustment factor
    close = inp.prices[["date", "instrument", "close"]].copy()
    if "adj_factor" in inp.prices.columns:
        close["close"] = close["close"] / inp.prices["adj_factor"].to_numpy()
    m = base.merge(s, on=["date", "instrument"], how="left").merge(d, on=["date", "instrument"], how="left")
    m = m.merge(close, on=["date", "instrument"], how="left").sort_values(["instrument", "date"]).reset_index(drop=True)
    g = m.groupby("instrument", sort=False)
    out = m[["date", "instrument"]].copy()
    out["atm_iv"] = m["atm_iv"]
    out["iv_rv_ratio"] = (m["atm_iv"] / m["rv_20"]).where(m["rv_20"] > 0)
    out["iv_minus_rv"] = m["atm_iv"] - m["rv_20"]
    out["implied_move_5d"] = m["atm_iv"] * np.sqrt(5.0 / 252.0)
    out["iv_chg_5d"] = m["atm_iv"] - _lag(g, "atm_iv", 5)
    out["iv_pctile"] = _pctile(g, "atm_iv")
    out["term_slope"] = (m["next_iv"] / m["atm_iv"] - 1.0).where(m["atm_iv"] > 0)
    out["skew"] = m["skew"]
    m["pcr_oi_"] = (m["opt_put_oi"] / m["opt_call_oi"]).where(m["opt_call_oi"] > 0)
    out["pcr_oi"] = m["pcr_oi_"]
    out["pcr_oi_chg_5d"] = m["pcr_oi_"] - m.groupby("instrument", sort=False)["pcr_oi_"].shift(5)
    out["pcr_vol"] = (m["opt_put_vol"] / m["opt_call_vol"]).where(m["opt_call_vol"] > 0)
    prev_oi = _lag(g, "fut_oi", 5)
    out["fut_oi_chg_5d"] = (m["fut_oi"] / prev_oi - 1.0).where(prev_oi > 0)
    out["fut_basis"] = (m["fut_close"] / m["close"] - 1.0).where((m["close"] > 0) & (m["fut_close"] > 0))
    out["straddle_pct"] = m["straddle_pct"]
    out["dte_near"] = m["dte"] / 30.0
    m["optvol"] = m["opt_call_vol"].fillna(0) + m["opt_put_vol"].fillna(0)
    vmean = _roll(m.groupby("instrument", sort=False), "optvol", 20, "mean")
    out["opt_volume_ratio_20"] = (m["optvol"] / vmean).where(vmean > 0)
    return out


def event_features(inp: Inputs, index: pd.DataFrame) -> pd.DataFrame:
    """Result-date features for each (date, instrument) row of `index`.

    On date D an upcoming result counts only if it had been announced by D. An event with no
    known announcement date counts from its own date, so it is never seen coming.
    """
    out = index[["date", "instrument"]].copy()
    cal = inp.calendar
    pos = pd.Series(np.arange(len(cal.days)), index=cal.days)
    out["i"] = out["date"].map(pos)
    days_to = np.full(len(out), 30.0)
    since = np.full(len(out), 60.0)
    ev = inp.events
    if len(ev):
        ev = ev.assign(known=ev["announced_on"].fillna(ev["event_date"]))
        rows_by = {k: v.index.to_numpy() for k, v in out.groupby("instrument", sort=False)}
        loc = pd.Series(np.arange(len(out)), index=out.index)
        for instrument, e in ev.groupby("instrument", sort=False):
            ridx = rows_by.get(instrument)
            if ridx is None:
                continue
            at = loc.loc[ridx].to_numpy()
            d = out.loc[ridx, "date"].to_numpy()
            i_row = out.loc[ridx, "i"].to_numpy(dtype=float)
            for ed, known in zip(e["event_date"].to_numpy(), e["known"].to_numpy()):
                i_event = cal.index_on_or_before(ed)
                if i_event < 0:
                    continue
                if cal.days[i_event] != pd.Timestamp(ed):
                    i_event += 1                              # an event on a holiday acts on the next session
                ahead = (d >= known) & (d < ed)               # announced, and still to come
                days_to[at[ahead]] = np.minimum(days_to[at[ahead]], i_event - i_row[ahead])
                past = d >= ed
                since[at[past]] = np.minimum(since[at[past]], i_row[past] - i_event)
    out["days_to_results"] = np.clip(days_to, 0, 30) / 30.0
    out["results_in_5d"] = (days_to <= 5).astype(float)
    out["days_since_results"] = np.clip(since, 0, 60) / 60.0
    return out.drop(columns="i")


def regime_features(inp: Inputs, price: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per date; the caller repeats it for every instrument."""
    price = price_features(inp) if price is None else price
    nifty = price[price["instrument"] == "NIFTY"].set_index("date")
    out = pd.DataFrame(index=pd.Index(sorted(inp.prices["date"].unique()), name="date"))
    out["nifty_ret_5d"] = nifty["ret_5d"]
    out["nifty_rv_20"] = nifty["rv_20"]
    out["nifty_dist_ma200"] = nifty["dist_ma200"]
    vix = inp.prices[inp.prices["instrument"] == VIX].sort_values("date").set_index("date")["close"]
    out["vix_pctile"] = vix.rolling(252, min_periods=60).rank(pct=True)
    out["vix_chg_5d"] = vix / vix.shift(5) - 1.0
    stocks = inp.prices[inp.prices["kind"] == "stock"][["date", "instrument"]].merge(
        price[["date", "instrument", "dist_ma50"]], on=["date", "instrument"])
    stocks = stocks[stocks["dist_ma50"].notna()]
    out["breadth_ma50"] = (stocks["dist_ma50"] > 0).groupby(stocks["date"]).mean()
    return out.reset_index()


def flow_features(inp: Inputs) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(cash-flow features by date, participant-position features by date): two sources, two availability times."""
    f = inp.fii_dii.sort_values("date").set_index("date")
    cash = pd.DataFrame(index=f.index)
    for who in ("fii", "dii"):
        s5 = f[f"{who}_net"].rolling(5, min_periods=5).sum()
        mean, std = s5.rolling(252, min_periods=60).mean(), s5.rolling(252, min_periods=60).std()
        cash[f"{who}_net_5d_z"] = ((s5 - mean) / std).where(std > 0)
        if who == "fii":
            cash["fii_net_5d_pctile"] = s5.rolling(252, min_periods=60).rank(pct=True)
    p = inp.participant
    p = p[p["participant"].astype(str).str.upper() == "FII"].sort_values("date").set_index("date")
    pos = pd.DataFrame(index=p.index)
    tot = p["fut_idx_long"] + p["fut_idx_short"]
    pos["fii_idx_fut_ratio"] = ((p["fut_idx_long"] - p["fut_idx_short"]) / tot).where(tot > 0)
    pos["fii_idx_fut_ratio_chg_5d"] = pos["fii_idx_fut_ratio"] - pos["fii_idx_fut_ratio"].shift(5)
    need = ["opt_idx_call_long", "opt_idx_put_long", "opt_idx_call_short", "opt_idx_put_short"]
    if all(c in p.columns for c in need):
        total = p[need].sum(axis=1)
        tilt = p["opt_idx_call_long"] + p["opt_idx_put_short"] - p["opt_idx_put_long"] - p["opt_idx_call_short"]
        pos["fii_idx_opt_tilt"] = (tilt / total).where(total > 0)
    else:
        pos["fii_idx_opt_tilt"] = np.nan
    return cash.reset_index(), pos.reset_index()


def intraday_features(inp: Inputs, cutoff: str = "14:30") -> pd.DataFrame:
    """State of the day up to `cutoff`, from 5-minute bars. A bar stamped 14:25 ends at 14:30."""
    cols = ["date", "instrument", "intraday_ret_1430", "intraday_range_1430", "gap_open_today"]
    bars = inp.intraday
    if bars is None or bars.empty:
        return pd.DataFrame(columns=cols)
    b = bars.copy()
    b["date"] = b["timestamp"].dt.tz_localize(None).dt.normalize() if b["timestamp"].dt.tz is not None else b["timestamp"].dt.normalize()
    hhmm = b["timestamp"].dt.strftime("%H:%M")
    b = b[hhmm < cutoff].sort_values(["instrument", "timestamp"])
    g = b.groupby(["instrument", "date"], sort=False)
    day = g.agg(open=("open", "first"), high=("high", "max"), low=("low", "min"), last=("close", "last")).reset_index()
    prev = inp.prices[["date", "instrument", "close"]].sort_values(["instrument", "date"])
    prev["prev_close"] = prev.groupby("instrument", sort=False)["close"].shift(1)
    day = day.merge(prev[["date", "instrument", "prev_close"]], on=["date", "instrument"], how="left")
    # the previous close must come from the session before, even when today's daily row is not in yet
    last_close = prev.groupby("instrument", sort=False).tail(1).set_index("instrument")
    missing = day["prev_close"].isna()
    if missing.any():
        after = day.loc[missing, "date"].to_numpy() > day.loc[missing, "instrument"].map(last_close["date"]).to_numpy()
        fill = day.loc[missing, "instrument"].map(last_close["close"]).where(after)
        day.loc[missing, "prev_close"] = fill
    day["intraday_ret_1430"] = day["last"] / day["open"] - 1.0
    day["intraday_range_1430"] = (day["high"] - day["low"]) / day["open"]
    day["gap_open_today"] = day["open"] / day["prev_close"] - 1.0
    return day[cols]
