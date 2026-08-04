from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class WalkForwardSplit:
    name: str
    base_train_end: str
    calibration_start: str
    calibration_end: str
    prediction_start: str
    prediction_end: str


DEFAULT_SPLITS = [
    WalkForwardSplit(
        name="validate_2024",
        base_train_end="2023-12-31",
        calibration_start="2023-01-01",
        calibration_end="2023-12-31",
        prediction_start="2024-01-01",
        prediction_end="2024-12-31",
    ),
    WalkForwardSplit(
        name="validate_2025",
        base_train_end="2024-12-31",
        calibration_start="2024-01-01",
        calibration_end="2024-12-31",
        prediction_start="2025-01-01",
        prediction_end="2025-12-31",
    ),
    WalkForwardSplit(
        name="test_2026_jan_jun5",
        base_train_end="2025-12-31",
        calibration_start="2025-01-01",
        calibration_end="2025-12-31",
        prediction_start="2026-01-01",
        prediction_end="2026-06-05",
    ),
]


def between_dates(df: pd.DataFrame, start: str | None = None, end: str | None = None) -> pd.Series:
    dates = pd.to_datetime(df["date"])
    mask = pd.Series(True, index=df.index)
    if start is not None:
        mask &= dates.ge(pd.Timestamp(start))
    if end is not None:
        mask &= dates.le(pd.Timestamp(end))
    return mask
