from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class RetrainWindow:
    label: str
    train_end: pd.Timestamp
    prediction_start: pd.Timestamp
    prediction_end: pd.Timestamp

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "train_end": str(self.train_end.date()),
            "prediction_start": str(self.prediction_start.date()),
            "prediction_end": str(self.prediction_end.date()),
        }


def monthly_retrain_windows(
    prediction_start: pd.Timestamp,
    prediction_end: pd.Timestamp,
    cutoff_day: int = 20,
) -> list[RetrainWindow]:
    prediction_start = pd.Timestamp(prediction_start).normalize()
    prediction_end = pd.Timestamp(prediction_end).normalize()
    windows: list[RetrainWindow] = []
    month_start = pd.Timestamp(prediction_start.year, prediction_start.month, 1)
    while month_start <= prediction_end:
        window_start = max(prediction_start, month_start)
        window_end = min(prediction_end, month_start + pd.offsets.MonthEnd(0))
        previous_month = month_start - pd.DateOffset(months=1)
        train_end = pd.Timestamp(previous_month.year, previous_month.month, cutoff_day).normalize()
        windows.append(
            RetrainWindow(
                label=month_start.strftime("%Y-%m"),
                train_end=train_end,
                prediction_start=window_start,
                prediction_end=window_end,
            )
        )
        month_start = month_start + pd.offsets.MonthBegin(1)
    return windows
