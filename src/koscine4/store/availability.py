"""When data becomes usable (requirement D-3, design rule 2).

Every read of the store names a `Decision`: a session date and a point in that day.
A row dated D from a source is readable only once that source has published it.

Points in a day, in order:   preopen  <  1430  <  evening
Source availability values:  evening        the evening of D
                             next_preopen   before the open of the session after D
                             next_evening   the evening of the session after D
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import pandas as pd

from koscine4.store.calendar import TradingCalendar

DECISION_SLOTS = {"preopen": 0, "1430": 1, "evening": 2}

# source availability -> (sessions after D, slot in that day)
_READY = {"evening": (0, 2), "next_preopen": (1, 0), "next_evening": (1, 2)}


class AvailabilityError(ValueError):
    """Raised for an unknown decision time, availability value or source."""


@dataclass(frozen=True)
class Decision:
    """The moment a question is asked: a session date and a point within it."""

    date: pd.Timestamp
    time: str = "evening"

    def __post_init__(self):
        if self.time not in DECISION_SLOTS:
            raise AvailabilityError(f"Unknown decision time '{self.time}'. Use one of {sorted(DECISION_SLOTS)}.")
        object.__setattr__(self, "date", pd.Timestamp(self.date).normalize())


def last_visible_date(availability: str, decision: Decision, calendar: TradingCalendar) -> pd.Timestamp | None:
    """The latest row date from a source that `decision` may read, or None if nothing is readable yet.

    `decision.date` need not be a session: a question asked on a holiday sees what the
    previous session's evening saw.
    """
    if availability not in _READY:
        raise AvailabilityError(f"Unknown availability '{availability}'. Use one of {sorted(_READY)}.")
    lag, ready_slot = _READY[availability]
    i_now = calendar.index_on_or_before(decision.date)
    if i_now < 0:
        return None
    slot_now = DECISION_SLOTS[decision.time] if calendar.is_session(decision.date) else DECISION_SLOTS["evening"]
    # row D (session index i_d) is ready at (i_d + lag, ready_slot); readable when that is <= (i_now, slot_now)
    i_max = i_now - lag if ready_slot <= slot_now else i_now - lag - 1
    if i_max < 0:
        return None
    return calendar.days[i_max]


def source_availability(settings_section: Mapping[str, str], source: str) -> str:
    try:
        value = settings_section[source]
    except KeyError:
        raise AvailabilityError(
            f"Source '{source}' has no entry in [availability]. Every source must record when it becomes usable."
        ) from None
    if value not in _READY:
        raise AvailabilityError(f"[availability].{source} = '{value}' is not one of {sorted(_READY)}.")
    return value
