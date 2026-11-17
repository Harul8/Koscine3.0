"""One name per company across symbol changes.

The exchange files carry the symbol in use on each day, so a renamed company (HEROHONDA to
HEROMOTOCO in 2011) looks like one stock ending and another starting. The store names every
row by the company's latest symbol, using the exchange's own list of symbol changes. The name
is only a label: no price or date is taken from the future.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from koscine4.settings import Settings
from koscine4.sources.exchange_reports import read_symbol_changes

CHANGE_COLUMNS = ["old", "new", "date"]


def load_changes(settings: Settings) -> pd.DataFrame:
    return read_symbol_changes(settings)


def canonicalize(symbols: pd.Series, dates: pd.Series, changes: pd.DataFrame | None) -> pd.Series:
    """The latest symbol for each (symbol, date) row. A change applies to rows dated before it took effect."""
    cur = symbols.astype(str).str.upper()
    if changes is None or changes.empty or cur.empty:
        return cur
    values = cur.to_numpy(dtype=object).copy()
    day = pd.to_datetime(dates).to_numpy()
    rows = {k: np.asarray(v) for k, v in pd.Series(np.arange(len(values))).groupby(values).indices.items()}
    for old, new, when in changes.sort_values("date").itertuples(index=False):
        idx = rows.get(old)
        if idx is None or old == new:
            continue
        move = idx[day[idx] < np.datetime64(when)]
        if len(move) == 0:
            continue
        values[move] = new
        rest = idx[day[idx] >= np.datetime64(when)]
        if len(rest):
            rows[old] = rest
        else:
            rows.pop(old)
        rows[new] = np.concatenate([rows[new], move]) if new in rows else move
    return pd.Series(values, index=symbols.index)


def canonical_frame(df: pd.DataFrame, changes: pd.DataFrame | None, symbol: str = "instrument", date: str = "date") -> pd.DataFrame:
    """Copy of `df` with its symbol column renamed to the latest symbol."""
    if df is None or df.empty or changes is None or changes.empty:
        return df
    out = df.copy()
    out[symbol] = canonicalize(out[symbol], out[date], changes)
    return out


def names_over_time(changes: pd.DataFrame | None) -> dict[str, list[tuple[pd.Timestamp, str]]]:
    """For each latest symbol: [(date the old name stopped, old name), ...] in date order."""
    out: dict[str, list[tuple[pd.Timestamp, str]]] = {}
    if changes is None or changes.empty:
        return out
    final: dict[str, str] = {}
    for old, new, when in changes.sort_values("date", ascending=False).itertuples(index=False):
        if old == new:
            continue
        canon = final.get(new, new)
        final.setdefault(old, canon)
        out.setdefault(canon, []).append((pd.Timestamp(when), old))
    for canon in out:
        out[canon].sort()
    return out


def symbol_on(instrument: str, date, timeline: dict[str, list[tuple[pd.Timestamp, str]]]) -> str:
    """The symbol the exchange used for `instrument` on `date`."""
    date = pd.Timestamp(date)
    for stopped, old in timeline.get(instrument, []):
        if date < stopped:
            return old
    return instrument
