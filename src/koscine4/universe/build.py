"""Build and read the monthly stock list (requirements U-1 to U-6, decision A1).

On the first trading session of a month the list is rebuilt from data up to the previous close:

1. A stock is eligible if its options traded on at least `min_option_days_share` of the last
   `option_days_window` sessions.
2. Eligible stocks are ranked by market value at the previous close. The top `top_n` are the list.

Each month is saved once under `<pit>/universe/<YYYY-MM>.parquet`. A saved month is never
changed by a normal build. Nifty and Bank Nifty are always tradeable and are not ranked.

    python -m koscine4.universe.build                 # build months not yet saved
    python -m koscine4.universe.build --rebuild       # rebuild all (old files are kept under _superseded)
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd

from koscine4.settings import Settings, load_settings
from koscine4.store.calendar import TradingCalendar
from koscine4.store.market_value import MARKET_VALUE
from koscine4.store.symbols import canonicalize, load_changes
from koscine4.store.tables import PRICES_RAW, StoreError, read_table

INDEX_BOOK = ("NIFTY", "BANKNIFTY")
UNIVERSE_COLUMNS = [
    "month", "rebuild_date", "data_up_to", "instrument", "rank", "market_value",
    "shares_method", "option_days_share", "in_list",
]
SUMMARY_FILE = Path("universe") / "top_list.csv"


class UniverseError(RuntimeError):
    """Raised when a month's list is missing or is asked for before its rebuild date."""


def universe_dir(settings: Settings) -> Path:
    return settings.path("pit") / "universe"


def month_key(date) -> str:
    return pd.Timestamp(date).strftime("%Y-%m")


def rebuild_dates(calendar: TradingCalendar) -> pd.DatetimeIndex:
    """The first session of every month in the calendar."""
    days = pd.Series(calendar.days, index=calendar.days)
    return pd.DatetimeIndex(days.groupby(days.dt.to_period("M")).min().values)


def option_activity(settings: Settings) -> pd.DataFrame:
    """One row per (date, instrument) with whether any option traded that day."""
    d = pd.read_parquet(settings.path("silver") / "eod_deriv_daily.parquet",
                        columns=["date", "symbol", "opt_call_vol", "opt_put_vol"])
    d["date"] = pd.to_datetime(d["date"]).dt.normalize()
    traded = (d["opt_call_vol"].fillna(0) + d["opt_put_vol"].fillna(0)) > 0
    out = pd.DataFrame({"date": d["date"], "instrument": canonicalize(d["symbol"], d["date"], load_changes(settings)), "traded": traded})
    return out.groupby(["date", "instrument"], as_index=False)["traded"].max()


def build_month(rebuild_date, calendar: TradingCalendar, market_value: pd.DataFrame, activity: pd.DataFrame, cfg) -> pd.DataFrame:
    """The ranked list for the month starting at `rebuild_date`, using rows dated before it only."""
    rebuild_date = pd.Timestamp(rebuild_date).normalize()
    i = calendar.index_on_or_before(rebuild_date)
    window = int(cfg["option_days_window"])
    if i < window or calendar.days[i] != rebuild_date:
        return pd.DataFrame(columns=UNIVERSE_COLUMNS)
    prev = calendar.days[i - 1]
    sessions = calendar.days[i - window:i]
    act = activity[activity["date"].isin(sessions) & ~activity["instrument"].isin(INDEX_BOOK)]
    share = act.groupby("instrument")["traded"].sum() / float(window)
    eligible = share[share >= float(cfg["min_option_days_share"])]

    mv = market_value[(market_value["date"] <= prev) & (market_value["date"] >= calendar.days[max(0, i - 1 - int(cfg["max_value_age_sessions"]))])]
    mv = mv.sort_values("date").drop_duplicates("instrument", keep="last").set_index("instrument")
    rows = pd.DataFrame({"option_days_share": eligible}).join(mv[["market_value", "method"]], how="inner")
    rows = rows[rows["market_value"] > 0].sort_values("market_value", ascending=False)
    rows["rank"] = range(1, len(rows) + 1)
    out = rows.reset_index().rename(columns={"index": "instrument", "method": "shares_method"})
    out["month"] = month_key(rebuild_date)
    out["rebuild_date"] = rebuild_date
    out["data_up_to"] = prev
    out["in_list"] = out["rank"] <= int(cfg["top_n"])
    return out[UNIVERSE_COLUMNS]


def build_universe(settings: Settings | None = None, start=None, end=None, rebuild: bool = False) -> dict:
    settings = settings or load_settings()
    cfg = settings.section("universe")
    prices = read_table(settings, PRICES_RAW, columns=["date", "instrument", "kind"])
    calendar = TradingCalendar.from_prices(prices, int(settings.section("store")["min_symbols_for_session"]))
    market_value = read_table(settings, MARKET_VALUE)
    activity = option_activity(settings)
    folder = universe_dir(settings)
    folder.mkdir(parents=True, exist_ok=True)
    built = skipped = empty = 0
    for rd in rebuild_dates(calendar):
        if (start is not None and rd < pd.Timestamp(start)) or (end is not None and rd > pd.Timestamp(end)):
            continue
        path = folder / f"{month_key(rd)}.parquet"
        if path.exists() and not rebuild:
            skipped += 1
            continue
        month = build_month(rd, calendar, market_value, activity, cfg)
        if month.empty:
            empty += 1
            continue
        if path.exists():                               # an explicit rebuild keeps what it replaces
            keep = folder / "_superseded"
            keep.mkdir(exist_ok=True)
            n = len(list(keep.glob(f"{path.stem}__*.parquet")))
            shutil.copy2(path, keep / f"{path.stem}__{n + 1}.parquet")
        month.to_parquet(path, index=False)
        built += 1
    summary = write_summary(settings)
    return {"months_built": built, "months_kept": skipped, "months_without_data": empty, **summary}


def read_all(settings: Settings) -> pd.DataFrame:
    files = sorted(universe_dir(settings).glob("*.parquet")) if universe_dir(settings).exists() else []
    if not files:
        return pd.DataFrame(columns=UNIVERSE_COLUMNS)
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def write_summary(settings: Settings) -> dict:
    """A small tracked copy of every month's list, and counts for the build report."""
    allm = read_all(settings)
    top = allm[allm["in_list"]] if len(allm) else allm
    path = settings.path("records") / SUMMARY_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    top.to_csv(path, index=False)
    if top.empty:
        return {"months_saved": 0}
    later = top[top["shares_method"] == "back_cast"].groupby("month").size()
    return {
        "months_saved": int(top["month"].nunique()),
        "first_month": str(top["month"].min()),
        "last_month": str(top["month"].max()),
        "months_with_a_back_cast_name": int(len(later)),
        "share_method_counts": top["shares_method"].value_counts().to_dict(),
    }


def read_month(settings: Settings, date) -> pd.DataFrame:
    """The saved ranking for the month containing `date`. Refused before that month's rebuild date."""
    date = pd.Timestamp(date).normalize()
    path = universe_dir(settings) / f"{month_key(date)}.parquet"
    if not path.exists():
        raise UniverseError(f"No stock list has been built for {month_key(date)}. Run: python -m koscine4.universe.build")
    month = pd.read_parquet(path)
    rebuilt = pd.Timestamp(month["rebuild_date"].iloc[0])
    if date < rebuilt:
        raise UniverseError(f"The list for {month_key(date)} is built on {rebuilt.date()}; {date.date()} is before that.")
    return month


def stock_list(settings: Settings, date) -> list[str]:
    """The top stocks tradeable on `date`, in rank order (U-1, U-2)."""
    month = read_month(settings, date)
    return month[month["in_list"]].sort_values("rank")["instrument"].tolist()


def tradeable(settings: Settings, date, open_instruments=()) -> list[str]:
    """Everything that may be traded on `date`: the indices (U-5), the month's stocks, and any
    instrument with an open trade, which stays tradeable until that trade exits (U-4)."""
    names = list(INDEX_BOOK) + stock_list(settings, date)
    return names + [x for x in open_instruments if x not in names]


def fo_members(activity: pd.DataFrame, calendar: TradingCalendar, date, window: int = 20) -> list[str]:
    """Stocks with a derivatives row in the `window` sessions before `date`: the training population (U-6)."""
    i = calendar.index_on_or_before(pd.Timestamp(date))
    sessions = calendar.days[max(0, i - window):i]
    act = activity[activity["date"].isin(sessions) & ~activity["instrument"].isin(INDEX_BOOK)]
    return sorted(act["instrument"].unique())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rebuild", action="store_true", help="rebuild months already saved (old files are kept)")
    parser.add_argument("--start")
    parser.add_argument("--end")
    args = parser.parse_args()
    try:
        for key, value in build_universe(start=args.start, end=args.end, rebuild=args.rebuild).items():
            print(f"{key}: {value}")
    except StoreError as exc:
        raise SystemExit(f"{exc}\nBuild the store first: python -m koscine4.store.build")


if __name__ == "__main__":
    main()
