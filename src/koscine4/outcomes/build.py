"""What happened after each decision (requirements T-1 to T-3, D-2).

Outcomes live in their own folder, apart from every input table, and are read only through
`read` and `read_structure`, which take a cutoff and return nothing whose window ends later.

Two tables per entry time:

    moves_<entry_time>       per instrument and day: the move from the entry price over 1, 3 and
                             5 sessions, the move minus the implied move, and whether bands held
    structure_<strategy>     the result of a standard option structure after costs, written by
                             the harness when that structure is run on every signal

Entry prices (T-2). For a decision at the close, entry is the next session's open, so the
overnight gap is not part of the outcome. For a 14:30 decision, entry is 15:15; with daily
data the same day's close stands in for it.

    python -m koscine4.outcomes.build          # the owner runs this; about a minute
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from koscine4.settings import Settings, load_settings
from koscine4.store.api import Store
from koscine4.store.availability import Decision
from koscine4.store.surface import near_and_next
from koscine4.store.tables import StoreError

HORIZONS = (1, 3, 5)
# Forward-looking columns are named fwd_*, so none can be mistaken for a trailing feature of a similar name.
ENTRY_TIMES = ("next_open", "1515")
BANDS_PCT = (2.0, 3.0)
KEY = ["date", "instrument"]


class OutcomeError(RuntimeError):
    """Raised when an outcome table is missing or asked for without a cutoff."""


def moves_path(settings: Settings, entry_time: str):
    return settings.path("outcomes") / f"moves_{entry_time}.parquet"


def structure_path(settings: Settings, strategy: str):
    return settings.path("outcomes") / f"structure_{strategy}.parquet"


def compute_moves(prices: pd.DataFrame, implied: pd.DataFrame, entry_time: str) -> pd.DataFrame:
    """Outcome rows from adjusted prices. `implied` holds atm_iv per (date, instrument) as the decision saw it."""
    if entry_time not in ENTRY_TIMES:
        raise OutcomeError(f"Unknown entry time '{entry_time}'. Use one of {ENTRY_TIMES}.")
    p = prices[prices["instrument"] != "INDIAVIX"].sort_values(["instrument", "date"]).reset_index(drop=True)
    g = p.groupby("instrument", sort=False)
    out = p[KEY].copy()
    out["entry_time"] = entry_time
    if entry_time == "next_open":
        entry, out["entry_date"] = g["open"].shift(-1), g["date"].shift(-1)
    else:
        entry, out["entry_date"] = p["close"], p["date"]
    out["entry_price"] = entry
    out = out.merge(implied[KEY + ["atm_iv"]], on=KEY, how="left")
    for h in HORIZONS:
        exit_close = g["close"].shift(-h)
        end = g["date"].shift(-h)
        # the highest high and lowest low over the h sessions after the decision day
        hi = pd.concat([g["high"].shift(-k) for k in range(1, h + 1)], axis=1).max(axis=1, skipna=False)
        lo = pd.concat([g["low"].shift(-k) for k in range(1, h + 1)], axis=1).min(axis=1, skipna=False)
        ret = exit_close / entry - 1.0
        implied_move = out["atm_iv"] * np.sqrt(h / 252.0)
        worst = np.maximum((hi / entry - 1.0).abs(), (lo / entry - 1.0).abs())
        out[f"fwd_ret_{h}d"] = ret
        out[f"fwd_abs_{h}d"] = ret.abs()
        out[f"fwd_up_{h}d"] = (ret > 0).astype(float).where(ret.notna())
        out[f"implied_{h}d"] = implied_move
        out[f"excess_{h}d"] = ret.abs() - implied_move
        out[f"beat_implied_{h}d"] = (ret.abs() > implied_move).astype(float).where(ret.notna() & implied_move.notna())
        out[f"within_implied_{h}d"] = (worst <= implied_move).astype(float).where(worst.notna() & implied_move.notna())
        for band in BANDS_PCT:
            out[f"within_{band:g}pct_{h}d"] = (worst <= band / 100.0).astype(float).where(worst.notna())
        out[f"window_end_{h}d"] = end                     # T-3: missing until the whole window has passed
    return out.drop(columns=["atm_iv"])


def build_moves(settings: Settings | None = None) -> dict:
    settings = settings or load_settings()
    store = Store(settings)
    last = store.calendar.days[-1]
    asof = Decision(last, "evening")
    prices = store.prices(asof=asof)
    try:
        cfg = settings.section("features")
        s = near_and_next(store.surface(asof=asof), int(cfg["min_days_to_expiry"]))[KEY + ["atm_iv"]]
    except StoreError:
        s = pd.DataFrame(columns=KEY + ["atm_iv"])
    out_dir = settings.path("outcomes")
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for entry_time in ENTRY_TIMES:
        implied = s
        if entry_time == "1515":                          # a 14:30 decision sees yesterday's surface
            days = store.calendar.days
            pos = pd.Series(np.arange(len(days)), index=days)
            i = s["date"].map(pos) + 1
            keep = i.notna() & (i < len(days))
            implied = s[keep].assign(date=days[i[keep].astype(int).to_numpy()])
        table = compute_moves(prices, implied, entry_time)
        table.to_parquet(moves_path(settings, entry_time), index=False)
        summary[entry_time] = {"rows": len(table), "complete_5d": int(table["window_end_5d"].notna().sum())}
    return summary


def read(settings: Settings, entry_time: str, cutoff, horizon: int, instruments=None) -> pd.DataFrame:
    """Outcomes for one horizon whose window ended on or before `cutoff`. The only way in (D-2, T-3)."""
    if cutoff is None:
        raise OutcomeError("Outcomes can only be read with a cutoff date.")
    if horizon not in HORIZONS:
        raise OutcomeError(f"Horizon must be one of {HORIZONS}.")
    path = moves_path(settings, entry_time)
    if not path.exists():
        raise OutcomeError(f"No outcome table for entry '{entry_time}'. Run: python -m koscine4.outcomes.build")
    suffix = f"_{horizon}d"
    table = pd.read_parquet(path)
    cols = KEY + ["entry_time", "entry_date", "entry_price"] + [c for c in table.columns if c.endswith(suffix)]
    table = table[cols].rename(columns={c: c[: -len(suffix)] for c in cols if c.endswith(suffix)})
    table = table[table["window_end"].notna() & (table["window_end"] <= pd.Timestamp(cutoff))]
    if instruments is not None:
        table = table[table["instrument"].isin(list(instruments))]
    return table.reset_index(drop=True)


def write_structure(settings: Settings, strategy: str, priced_trades: pd.DataFrame) -> int:
    """Store the after-cost result of a structure taken on every signal (written by the harness)."""
    cols = ["decision_date", "instrument", "entry_date", "exit_date", "result_pct", "pnl_per_lot", "max_loss_per_lot", "exit_reason", "entry_trust"]
    t = priced_trades[cols].rename(columns={"decision_date": "date", "exit_date": "window_end"})
    path = structure_path(settings, strategy)
    path.parent.mkdir(parents=True, exist_ok=True)
    t.to_parquet(path, index=False)
    return len(t)


def read_structure(settings: Settings, strategy: str, cutoff) -> pd.DataFrame:
    """Structure results whose trade had closed on or before `cutoff`."""
    if cutoff is None:
        raise OutcomeError("Outcomes can only be read with a cutoff date.")
    path = structure_path(settings, strategy)
    if not path.exists():
        raise OutcomeError(f"No structure outcomes for '{strategy}'. Run it first: python -m koscine4.harness.run --strategy {strategy}")
    t = pd.read_parquet(path)
    return t[t["window_end"] <= pd.Timestamp(cutoff)].reset_index(drop=True)


def overlap_weight(horizon: int) -> float:
    """Weight for one row of an h-session outcome, so consecutive overlapping rows count as about one event (M-5)."""
    return 1.0 / float(horizon)


def main() -> None:
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    for key, value in build_moves().items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
