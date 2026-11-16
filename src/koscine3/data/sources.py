from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from koscine3.paths import DATA_SOURCES_CONFIG


@dataclass(frozen=True)
class DataSource:
    name: str
    path: Path
    rows: int | None = None
    columns: int | None = None
    symbols: int | None = None
    date_min: str | None = None
    date_max: str | None = None


def read_data_source(config_path: Path = DATA_SOURCES_CONFIG) -> DataSource:
    with config_path.open("rb") as f:
        payload = tomllib.load(f)
    source = payload["source"]
    return DataSource(
        name=str(source["name"]),
        path=Path(source["path"]),
        rows=int(source["rows"]) if "rows" in source else None,
        columns=int(source["columns"]) if "columns" in source else None,
        symbols=int(source["symbols"]) if "symbols" in source else None,
        date_min=str(source["date_min"]) if "date_min" in source else None,
        date_max=str(source["date_max"]) if "date_max" in source else None,
    )


def load_market_data(
    columns: Iterable[str] | None = None,
    source: DataSource | None = None,
    sort: bool = False,
) -> pd.DataFrame:
    source = source or read_data_source()
    if not source.path.exists():
        raise FileNotFoundError(f"Market data source not found: {source.path}")
    df = pd.read_parquet(source.path, columns=list(columns) if columns is not None else None)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    if sort and {"date", "symbol"}.issubset(df.columns):
        df = df.sort_values(["date", "symbol"]).reset_index(drop=True)
    return df


def inspect_source(source: DataSource | None = None) -> dict[str, object]:
    source = source or read_data_source()
    df = load_market_data(source=source)
    return {
        "name": source.name,
        "path": str(source.path),
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "symbols": int(df["symbol"].nunique()) if "symbol" in df.columns else None,
        "date_min": str(df["date"].min().date()) if "date" in df.columns else None,
        "date_max": str(df["date"].max().date()) if "date" in df.columns else None,
        "column_names": list(df.columns),
    }
