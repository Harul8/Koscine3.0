"""Raw prices (requirements D-4, D-7) and adjusted prices as of a date (D-1).

Raw prices are stored once and never rewritten. An adjusted price as of date T is the raw
price multiplied by the factor of every corporate action with a row date < ex-date <= T.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from koscine4.settings import Settings
from koscine4.sources.exchange_reports import read_reports
from koscine4.store.symbols import canonical_frame, canonicalize, load_changes
from koscine4.store.tables import PRICE_COLUMNS, PRICES_RAW, StoreError, write_table

# Exchange index names, old and new, mapped to one instrument name each.
INDEX_NAMES = {
    "NIFTY": ("CNX Nifty", "Nifty 50"),
    "BANKNIFTY": ("CNX Bank", "Nifty Bank"),
    "INDIAVIX": ("India VIX",),
}
PRICE_FIELDS = ("open", "high", "low", "close", "prev_close")


def build_prices_raw(settings: Settings) -> pd.DataFrame:
    """Stocks that ever had derivatives, plus Nifty, Bank Nifty and India VIX, from the silver layer.

    Stocks that later left F&O or were delisted are kept (D-7).
    """
    silver = settings.path("silver")
    changes = load_changes(settings)
    deriv = pd.read_parquet(silver / "eod_deriv_daily.parquet", columns=["date", "symbol"]).drop_duplicates()
    fo_symbols = set(canonicalize(deriv["symbol"], deriv["date"], changes).unique())

    stock = pd.read_parquet(
        silver / "eod_stock.parquet",
        columns=["date", "symbol", "open", "high", "low", "close", "prev_close",
                 "volume", "turnover", "n_trades", "deliv_qty", "deliv_pct"],
    )
    stock["date"] = pd.to_datetime(stock["date"]).dt.normalize()
    stock = stock.rename(columns={"symbol": "instrument"})
    stock["instrument"] = canonicalize(stock["instrument"], stock["date"], changes)   # one name per company (D-7)
    stock = stock[stock["instrument"].isin(fo_symbols)].drop_duplicates(["instrument", "date"], keep="last")
    stock["kind"] = "stock"
    cfg = settings.section("store")
    stock = restore_raw(stock, int(cfg["min_symbols_for_session"]), float(cfg["restore_tolerance"]), int(cfg["restore_min_run"]))

    idx = pd.read_parquet(silver / "indices.parquet", columns=["date", "index_name", "open", "high", "low", "close", "volume", "turnover"])
    name_to_instrument = {n: inst for inst, names in INDEX_NAMES.items() for n in names}
    idx = idx[idx["index_name"].isin(name_to_instrument)].copy()
    idx["instrument"] = idx["index_name"].map(name_to_instrument)
    idx["date"] = pd.to_datetime(idx["date"]).dt.normalize()
    idx = idx.drop(columns=["index_name"])
    idx = _fill_from_reports(idx, read_reports(settings, "index_prices"))
    idx = idx.sort_values(["instrument", "date"]).drop_duplicates(["instrument", "date"], keep="first")
    idx["prev_close"] = idx.groupby("instrument")["close"].shift(1)
    idx["kind"] = "index"

    # Rows for sessions when a stock traded outside the main series (trade-for-trade), which silver drops.
    other = canonical_frame(read_reports(settings, "other_series"), changes)
    if len(other):
        other = other[other["instrument"].isin(fo_symbols)].drop(columns=["series"]).assign(kind="stock", restore_factor=1.0)
        stock = _fill_from_reports(stock, other)

    out = pd.concat([stock, idx], ignore_index=True)
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    out["instrument"] = out["instrument"].astype(str)
    for col in PRICE_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan
    out["restore_factor"] = out["restore_factor"].fillna(1.0) if "restore_factor" in out.columns else 1.0
    for col in ("open", "high", "low", "close", "prev_close", "volume", "turnover", "n_trades", "deliv_qty", "deliv_pct"):
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("float64")
    out = out[PRICE_COLUMNS].sort_values(["instrument", "date"]).reset_index(drop=True)
    write_table(settings, PRICES_RAW, out)
    return out


def _fill_from_reports(base: pd.DataFrame, extra: pd.DataFrame) -> pd.DataFrame:
    """Add rows from the exchange report files for (instrument, date) pairs that `base` lacks. `base` wins on overlap."""
    if extra is None or extra.empty:
        return base
    extra = extra.copy()
    extra["date"] = pd.to_datetime(extra["date"]).dt.normalize()
    extra = extra[(extra["close"] > 0) & (extra["open"] > 0)]
    have = set(zip(base["instrument"], base["date"]))
    new = extra[[k not in have for k in zip(extra["instrument"], extra["date"])]]
    new = new.drop_duplicates(["instrument", "date"])
    return pd.concat([base, new], ignore_index=True) if len(new) else base


def restore_raw(stock: pd.DataFrame, min_symbols: int, tolerance: float, min_run: int) -> pd.DataFrame:
    """Undo the back-adjustment the silver layer applied to some symbols, using the exchange's own field.

    The silver builder multiplies the history of some symbols by a split factor, but leaves
    `prev_close` exactly as the exchange published it. So for two rows on consecutive sessions,
    `prev_close[next] / close[this]` is 1 when the row is untouched and the reciprocal of the
    applied factor when it was back-adjusted. Multiplying by that ratio restores the raw price.

    A ratio is trusted only when at least `min_run` consecutive rows agree on it, so a single odd
    row cannot rescale anything. Adds a `restore_factor` column (1.0 for untouched rows).
    """
    s = stock.sort_values(["instrument", "date"]).reset_index(drop=True)
    counts = s.groupby("date")["instrument"].nunique()
    sessions = counts[counts >= min_symbols].index.sort_values()
    s["_i"] = s["date"].map(pd.Series(np.arange(len(sessions)), index=sessions))
    g = s.groupby("instrument", sort=False)
    adjacent = g["_i"].shift(-1) == s["_i"] + 1
    k = (g["prev_close"].shift(-1) / s["close"]).where(adjacent & (s["close"] > 0))
    k = k.mask((k - 1.0).abs() < tolerance, 1.0)        # a missing ratio stays missing here; it is filled below

    # Keep a ratio other than 1 only where it persists. Rows with no adjacent next session carry
    # no ratio and must not break a run, or a thinly traded stock would never be restored.
    known = k[k.notna() & (k > 0)]
    step = np.round(np.log(known), 2)
    inst = s.loc[known.index, "instrument"]
    run_id = (step != step.groupby(inst, sort=False).shift(1)).cumsum()
    run_len = step.groupby(run_id).transform("size")
    drop = known.index[(known != 1.0) & (run_len < min_run)]
    k.loc[drop] = np.nan

    k = k.groupby(s["instrument"], sort=False).ffill()
    k = k.groupby(s["instrument"], sort=False).bfill().fillna(1.0)
    for col in ("open", "high", "low", "close"):
        s[col] = np.where(k == 1.0, s[col], (s[col] * k).round(2))
    for col in ("volume", "n_trades"):
        if col in s.columns:
            s[col] = np.where(k == 1.0, s[col], (s[col] / k).round(0))
    s["restore_factor"] = k
    return s.drop(columns=["_i"])


def adjustment_multiplier(raw: pd.DataFrame, adjustments: pd.DataFrame, asof_date) -> pd.Series:
    """For each raw row, the product of factors of applied events with row date < ex_date <= asof_date."""
    mult = pd.Series(1.0, index=raw.index)
    if adjustments is None or adjustments.empty:
        return mult
    asof_date = pd.Timestamp(asof_date)
    adj = adjustments[(adjustments["ex_date"] <= asof_date)]
    if "applied" in adj.columns:
        adj = adj[adj["applied"].astype(bool)]
    if adj.empty:
        return mult
    events = {k: g.sort_values("ex_date") for k, g in adj.groupby("instrument")}
    for instrument, rows in raw[raw["instrument"].isin(events)].groupby("instrument", sort=False):
        ev = events[instrument]
        ex = ev["ex_date"].values
        # cum[i] = product of factors of events i..end; a row dated d takes events with ex_date > d
        cum = np.append(np.cumprod(ev["factor"].values[::-1])[::-1], 1.0)
        pos = np.searchsorted(ex, rows["date"].values, side="right")
        mult.loc[rows.index] = cum[pos]
    return mult


def adjust_prices(raw: pd.DataFrame, adjustments: pd.DataFrame, asof_date) -> pd.DataFrame:
    """Adjusted copy of `raw` as of `asof_date`. Rows after `asof_date` are dropped, never adjusted."""
    asof_date = pd.Timestamp(asof_date)
    out = raw[raw["date"] <= asof_date].copy()
    mult = adjustment_multiplier(out, adjustments, asof_date)
    for col in ("open", "high", "low", "close"):
        out[col] = out[col] * mult
    out["volume"] = out["volume"] / mult
    out["adj_factor"] = mult
    out = out.sort_values(["instrument", "date"])
    # previous close in adjusted terms, so close / prev_close is a true return across an ex-date
    out["prev_close"] = out.groupby("instrument", sort=False)["close"].shift(1)
    return out


def validate_raw(raw: pd.DataFrame) -> None:
    missing = [c for c in PRICE_COLUMNS if c not in raw.columns]
    if missing:
        raise StoreError(f"prices_raw is missing columns: {missing}")
