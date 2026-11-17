"""Data checks run on every refresh (requirements D-1, D-8, D-10).

Each check returns a list of issues. An "error" stops the refresh; a "review" is listed for
the owner and does not stop it.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import pandas as pd

from koscine4.store.calendar import TradingCalendar
from koscine4.store.prices import adjust_prices


@dataclass(frozen=True)
class Issue:
    check: str
    severity: str          # "error" or "review"
    instrument: str
    date: str
    detail: str


class DataCheckError(RuntimeError):
    def __init__(self, issues: list[Issue]):
        self.issues = issues
        head = "; ".join(f"{i.check} {i.instrument} {i.date}: {i.detail}" for i in issues[:5])
        more = f" (+{len(issues) - 5} more)" if len(issues) > 5 else ""
        super().__init__(f"{len(issues)} data check error(s): {head}{more}")


def _d(ts) -> str:
    return str(pd.Timestamp(ts).date())


def check_duplicates(prices: pd.DataFrame) -> list[Issue]:
    dup = prices[prices.duplicated(["instrument", "date"], keep=False)]
    return [Issue("duplicate_row", "error", r.instrument, _d(r.date), "more than one row for this instrument and day")
            for r in dup.drop_duplicates(["instrument", "date"]).itertuples(index=False)]


def check_price_sanity(prices: pd.DataFrame) -> list[Issue]:
    p = prices
    bad_sign = p[(p[["open", "high", "low", "close"]] <= 0).any(axis=1)]
    bad_range = p[(p["high"] < p["low"]) | (p["high"] < p[["open", "close"]].max(axis=1) - 1e-9) | (p["low"] > p[["open", "close"]].min(axis=1) + 1e-9)]
    out = [Issue("non_positive_price", "error", r.instrument, _d(r.date), "a price is zero or negative") for r in bad_sign.itertuples(index=False)]
    # the volatility index is a computed figure, not a traded price: a bad bar there is listed, not fatal
    out += [Issue("ohlc_out_of_order", "review" if r.instrument == "INDIAVIX" else "error", r.instrument, _d(r.date),
                  "high/low do not bracket open/close") for r in bad_range.itertuples(index=False)]
    return out


def check_missing_sessions(prices: pd.DataFrame, calendar: TradingCalendar, recent_sessions: int = 60) -> list[Issue]:
    """Sessions with no row between an instrument's first and last date.

    An index missing one of the last `recent_sessions` sessions is an error: a refresh must not
    pass with a new hole. Older index holes and all stock holes are listed for review (D-4).
    """
    out: list[Issue] = []
    recent_from = calendar.days[max(0, len(calendar) - recent_sessions)]
    for (instrument, kind), g in prices.groupby(["instrument", "kind"], sort=False):
        expected = calendar.between(g["date"].min(), g["date"].max())
        missing = expected.difference(pd.DatetimeIndex(g["date"]))
        if len(missing) == 0:
            continue
        new = missing[missing >= recent_from]
        if kind == "index" and len(new):
            out.append(Issue("missing_sessions", "error", instrument, _d(new[0]),
                             f"{len(new)} recent session(s) without a row, first {_d(new[0])}, last {_d(new[-1])}"))
        old = missing if kind != "index" else missing[missing < recent_from]
        if len(old):
            out.append(Issue("missing_sessions", "review", instrument, _d(old[0]),
                             f"{len(old)} session(s) without a row, first {_d(old[0])}, last {_d(old[-1])}"))
    return out


def check_stale_runs(prices: pd.DataFrame, run: int) -> list[Issue]:
    """`run` or more sessions in a row with an unchanged close and no volume."""
    p = prices[prices["kind"] == "stock"].sort_values(["instrument", "date"])
    same = (p["close"] == p.groupby("instrument", sort=False)["close"].shift(1)) & (p["volume"].fillna(0) == 0)
    block = (~same).groupby(p["instrument"], sort=False).cumsum()
    length = same.groupby([p["instrument"], block], sort=False).cumsum()
    hits = p[same & (length == run)]
    return [Issue("stale_price", "error", r.instrument, _d(r.date), f"{run} sessions in a row with no volume and an unchanged close")
            for r in hits.itertuples(index=False)]


def check_jumps(prices: pd.DataFrame, adjustments: pd.DataFrame, jump_pct: float) -> list[Issue]:
    """D-1: after adjustment, one-day changes above the threshold are listed for review."""
    asof = prices["date"].max()
    adj = adjust_prices(prices[prices["kind"] == "stock"], adjustments, asof)
    ret = adj["close"] / adj["prev_close"] - 1.0
    days = (adj["date"] - adj.groupby("instrument", sort=False)["date"].shift(1)).dt.days
    adj = adj.assign(ret=ret, days=days)          # before filtering: assigning to an empty frame would add rows
    big = adj[adj["ret"].abs() > jump_pct / 100.0]
    return [Issue("large_move", "review", r.instrument, _d(r.date),
                  f"adjusted change of {r.ret:+.1%}" + (f" across a {int(r.days)}-day break in the data" if r.days > 5 else " in one session"))
            for r in big.itertuples(index=False)]


def check_freshness(folder: Path, today, max_days: int) -> list[Issue]:
    """D-10: the newest file in an intraday folder must be recent."""
    folder = Path(folder)
    files = [f for f in folder.glob("*.parquet")] if folder.exists() else []
    if not files:
        return [Issue("intraday_missing", "error", folder.name, _d(today), "no intraday files found")]
    newest = max(pd.Timestamp(f.stem) for f in files if _is_date(f.stem))
    age = (pd.Timestamp(today).normalize() - newest).days
    if age > max_days:
        return [Issue("intraday_stale", "error", folder.name, _d(today), f"newest file is {age} days old ({_d(newest)})")]
    return []


def _is_date(text: str) -> bool:
    try:
        pd.Timestamp(text)
        return True
    except (ValueError, TypeError):
        return False


def run_price_checks(prices: pd.DataFrame, adjustments: pd.DataFrame, calendar: TradingCalendar,
                     cfg: Mapping[str, float], raise_on_error: bool = True) -> list[Issue]:
    issues = (
        check_duplicates(prices)
        + check_price_sanity(prices)
        + check_missing_sessions(prices, calendar, int(cfg["missing_error_recent_sessions"]))
        + check_stale_runs(prices, int(cfg["stale_run_sessions"]))
        + check_jumps(prices, adjustments, float(cfg["jump_review_pct"]))
    )
    errors = [i for i in issues if i.severity == "error"]
    if errors and raise_on_error:
        raise DataCheckError(errors)
    return issues


def issues_frame(issues: list[Issue]) -> pd.DataFrame:
    return pd.DataFrame([i.__dict__ for i in issues], columns=["check", "severity", "instrument", "date", "detail"])
