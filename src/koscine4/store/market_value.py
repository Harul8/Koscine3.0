"""Market value per stock and day (requirement D-11, decision A1).

Shares in issue come from the exchange's own files, which exist for two stretches only:
index-constituent files up to 2013 and the market-capitalisation file from 2024. For any
other day the count is carried from the nearest observation and corrected for every split
and bonus in between. Each row says how its share count was obtained:

    observed          an exchange figure no older than `observed_max_age_days`
    carried_forward   an older exchange figure, corrected for splits and bonuses since.
                      Uses nothing from the future, but misses share issues, buybacks and mergers.
    back_cast         no earlier figure exists (the stock listed after the early files ended),
                      so a later figure is carried backwards. This uses later information and
                      is the approximation D-11 allows; it is marked so it can be reported.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from koscine4.settings import Settings
from koscine4.sources.exchange_reports import read_reports
from koscine4.store.symbols import canonical_frame, load_changes
from koscine4.store.tables import write_table

MARKET_VALUE = "market_value"
MARKET_VALUE_COLUMNS = ["date", "instrument", "shares", "close", "market_value", "method", "anchor_date"]
_SHARE_CHANGING = ("split_or_bonus", "inferred_split")


def share_multiplier(dates: np.ndarray, events: pd.DataFrame) -> np.ndarray:
    """For each date, the product of share-count ratios (1 / factor) of splits and bonuses with ex-date <= date."""
    if events is None or events.empty:
        return np.ones(len(dates))
    ev = events.sort_values("ex_date")
    cum = np.concatenate([[1.0], np.cumprod(1.0 / ev["factor"].to_numpy(dtype=float))])
    return cum[np.searchsorted(ev["ex_date"].to_numpy(), dates, side="right")]


def clean_anchors(anchors: pd.DataFrame, events: pd.DataFrame, window: int) -> pd.DataFrame:
    """Share counts with splits taken out and short glitches removed.

    Around an ex-date the exchange file can show the old count for a day or two. Dividing by the
    split multiplier makes the series flat across a split, and a trailing median over `window`
    observations then drops a short disagreement without looking ahead.
    """
    a = anchors.sort_values("date").drop_duplicates("date", keep="last").copy()
    a["norm"] = a["shares"] / share_multiplier(a["date"].to_numpy(), events)
    a["norm"] = a["norm"].rolling(window, min_periods=1).median()
    return a[["date", "norm"]]


def shares_for(dates: np.ndarray, anchors: pd.DataFrame, events: pd.DataFrame, cfg) -> pd.DataFrame:
    """Share count on each date for one stock, with the method used."""
    out = pd.DataFrame({"date": dates, "shares": np.nan, "method": None, "anchor_date": pd.NaT})
    out["anchor_date"] = pd.to_datetime(out["anchor_date"])
    if anchors is None or anchors.empty:
        return out
    a = clean_anchors(anchors, events, int(cfg["share_glitch_window"]))
    a_dates, a_norm = a["date"].to_numpy(), a["norm"].to_numpy()
    mult = share_multiplier(dates, events)
    before = np.searchsorted(a_dates, dates, side="right") - 1          # last anchor on or before the date
    has_before = before >= 0
    idx = np.where(has_before, before, 0)                                 # else the first anchor after it
    anchor_date = a_dates[idx]
    out["shares"] = a_norm[idx] * mult
    out["anchor_date"] = anchor_date
    age = (pd.to_datetime(dates) - pd.to_datetime(anchor_date)).days
    fresh = has_before & (age <= int(cfg["observed_max_age_days"]))
    out["method"] = np.where(fresh, "observed", np.where(has_before, "carried_forward", "back_cast"))
    return out


def build_market_value(settings: Settings, prices_raw: pd.DataFrame, adjustments: pd.DataFrame,
                       share_counts: pd.DataFrame | None = None) -> pd.DataFrame:
    cfg = settings.section("store")
    counts = read_reports(settings, "share_counts") if share_counts is None else share_counts
    stocks = prices_raw[prices_raw["kind"] == "stock"][["date", "instrument", "close"]]
    if counts is None or counts.empty:
        df = pd.DataFrame(columns=MARKET_VALUE_COLUMNS)
        write_table(settings, MARKET_VALUE, df)
        return df
    counts = counts.rename(columns={"symbol": "instrument"})
    counts["date"] = pd.to_datetime(counts["date"]).dt.normalize()
    counts = canonical_frame(counts, load_changes(settings)).drop_duplicates(["instrument", "date"], keep="last")
    adj = adjustments[adjustments["applied"].astype(bool) & adjustments["kind"].isin(_SHARE_CHANGING)]
    events = {k: v for k, v in adj.groupby("instrument", sort=False)}
    anchors = {k: v[["date", "shares"]] for k, v in counts.groupby("instrument", sort=False)}
    parts = []
    for instrument, rows in stocks.groupby("instrument", sort=False):
        rows = rows.sort_values("date")
        s = shares_for(rows["date"].to_numpy(), anchors.get(instrument), events.get(instrument), cfg)
        s["instrument"] = instrument
        s["close"] = rows["close"].to_numpy()
        parts.append(s)
    df = pd.concat(parts, ignore_index=True)
    df = df[df["shares"].notna()]
    df["market_value"] = df["shares"] * df["close"]
    df = df[MARKET_VALUE_COLUMNS].sort_values(["instrument", "date"]).reset_index(drop=True)
    write_table(settings, MARKET_VALUE, df)
    return df
