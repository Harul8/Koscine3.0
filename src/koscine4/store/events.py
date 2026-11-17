"""Scheduled events calendar (requirement D-9).

An event may be used ahead of time only if the date it was announced is known.

Result dates come from two places:

1. The exchange's board-meeting file, published each evening. The file date is the day the
   meeting was announced, so a decision can know about a result date from that evening on.
2. The silver earnings table, for meetings the file does not show. It has no announcement
   dates, so those result dates are visible only on the day itself.
"""
from __future__ import annotations

import pandas as pd

from koscine4.settings import Settings
from koscine4.sources.exchange_reports import read_reports
from koscine4.store.symbols import canonical_frame, load_changes

EVENT_COLUMNS = ["instrument", "event_date", "kind", "announced_on"]
_RESULTS = r"result"
# a notice that moves or cancels a meeting still names "results"; it does not create a meeting on that date
_NOT_A_MEETING = r"postpon|cancel|defer|adjourn|resched"


def _from_board_meetings(settings: Settings) -> pd.DataFrame:
    bm = read_reports(settings, "board_meetings")
    if bm.empty:
        return pd.DataFrame(columns=EVENT_COLUMNS)
    bm = bm[bm["meeting_date"].notna() & bm["purpose"].str.contains(_RESULTS, case=False, na=False)]
    moved = bm["purpose"].str.slice(0, 60).str.contains(_NOT_A_MEETING, case=False, na=False)
    bm = bm[~moved | (bm["meeting_date"] > bm["file_date"])]      # a postponement to a new date is itself a notice
    first = bm.groupby(["symbol", "meeting_date"], as_index=False)["file_date"].min()
    first = first[first["file_date"] <= first["meeting_date"]]
    return pd.DataFrame({
        "instrument": first["symbol"].astype(str).str.upper(),
        "event_date": pd.to_datetime(first["meeting_date"]).dt.normalize(),
        "kind": "results",
        "announced_on": pd.to_datetime(first["file_date"]).dt.normalize(),
    })


def _from_silver(settings: Settings) -> pd.DataFrame:
    path = settings.path("silver") / "earnings.parquet"
    if not path.exists():
        return pd.DataFrame(columns=EVENT_COLUMNS)
    e = pd.read_parquet(path)
    text = pd.Series("", index=e.index)
    for col in ("event_type", "period"):
        if col in e.columns:
            text = text + " " + e[col].fillna("").astype(str)
    if text.str.strip().ne("").any():
        e = e[text.str.contains(_RESULTS, case=False)]
    return pd.DataFrame({
        "instrument": e["symbol"].astype(str).str.upper(),
        "event_date": pd.to_datetime(e["date"]).dt.normalize(),
        "kind": "results",
        "announced_on": pd.to_datetime(e["announced_on"]).dt.normalize() if "announced_on" in e.columns else pd.NaT,
    })


def load_events(settings: Settings) -> pd.DataFrame:
    frames = [f for f in (_from_board_meetings(settings), _from_silver(settings)) if len(f)]
    if not frames:
        return pd.DataFrame(columns=EVENT_COLUMNS)
    out = pd.concat(frames, ignore_index=True)
    out = canonical_frame(out, load_changes(settings), date="event_date")
    out["announced_on"] = pd.to_datetime(out["announced_on"])
    # the same meeting from both sources: keep the row that knows when it was announced
    out = out.sort_values(["instrument", "event_date", "announced_on"], na_position="last")
    return out.drop_duplicates(["instrument", "event_date", "kind"]).reset_index(drop=True)


def visible_events(events: pd.DataFrame, visible_date) -> pd.DataFrame:
    """Events a decision may know about, given the last date it can see.

    Known announcement date: visible from that date. Unknown: visible only once the event date arrives.
    """
    if visible_date is None:
        return events.iloc[0:0]
    visible_date = pd.Timestamp(visible_date)
    known_from = events["announced_on"].fillna(events["event_date"])
    return events[known_from <= visible_date]
