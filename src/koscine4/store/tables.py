"""Where the point-in-time tables live, and how they are read and written."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from koscine4.settings import Settings

PRICES_RAW = "prices_raw"
ADJUSTMENTS = "adjustments"
LOT_SIZES = "lot_sizes"

PRICE_COLUMNS = [
    "date", "instrument", "kind", "open", "high", "low", "close", "prev_close",
    "volume", "turnover", "n_trades", "deliv_qty", "deliv_pct", "restore_factor",
]
ADJUSTMENT_COLUMNS = [
    "instrument", "ex_date", "factor", "kind", "source", "observed_gap", "declared_ratio", "applied", "note",
]


class StoreError(RuntimeError):
    """Raised when a store table is missing or malformed."""


def table_path(settings: Settings, name: str) -> Path:
    return settings.path("pit") / f"{name}.parquet"


def write_table(settings: Settings, name: str, df: pd.DataFrame) -> Path:
    path = table_path(settings, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def read_table(settings: Settings, name: str, columns: list[str] | None = None) -> pd.DataFrame:
    path = table_path(settings, name)
    if not path.exists():
        raise StoreError(f"Store table '{name}' has not been built yet (expected {path}).")
    return pd.read_parquet(path, columns=columns)
