"""Option and futures contract quotes with lot sizes (requirement D-5).

The silver layer already holds every contract from 2010, one Parquet file per year. This module
reads it as of a decision and supplies a lot size for every contract: the exchange's own figure
where the silver lot table has one, and otherwise one inferred from open interest.

Open interest is quoted in shares and is always a whole multiple of the lot, so the greatest
common divisor of a contract month's open interest is the lot. Known limit: if a lot size is
changed part-way through a contract month (after a split), the inferred value is the common
divisor of the old and new lots.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from koscine4.settings import Settings
from koscine4.store.symbols import canonicalize, load_changes, names_over_time, symbol_on
from koscine4.store.tables import LOT_SIZES, StoreError, read_table, write_table

CONTRACT_COLUMNS = [
    "date", "symbol", "instrument", "expiry", "strike", "opt_type",
    "open", "high", "low", "close", "settle", "contracts", "open_int",
]


def contracts_folder(settings: Settings) -> Path:
    return settings.path("silver") / "eod_deriv_contracts"


def available_years(settings: Settings) -> list[int]:
    return sorted(int(p.name.split("=")[1]) for p in contracts_folder(settings).glob("year=*") if p.is_dir())


def read_contracts(settings: Settings, year: int, columns: list[str] | None = None, symbols: list[str] | None = None) -> pd.DataFrame:
    files = sorted((contracts_folder(settings) / f"year={year}").glob("*.parquet"))
    if not files:
        raise StoreError(f"No contract data for {year} under {contracts_folder(settings)}.")
    filters = [("symbol", "in", list(symbols))] if symbols else None
    return pd.concat([pd.read_parquet(f, columns=columns, filters=filters) for f in files], ignore_index=True)


def infer_lot_sizes(contracts: pd.DataFrame) -> pd.DataFrame:
    """Greatest common divisor of positive open interest per (symbol, expiry)."""
    c = contracts[contracts["open_int"] > 0]
    lots = c.groupby(["symbol", "expiry"], sort=False)["open_int"].agg(lambda s: int(np.gcd.reduce(s.to_numpy(dtype=np.int64))))
    return lots.rename("lot").reset_index()


def build_lot_sizes(settings: Settings) -> pd.DataFrame:
    """One lot size per (instrument, expiry) for the whole history."""
    parts = []
    for year in available_years(settings):
        parts.append(infer_lot_sizes(read_contracts(settings, year, columns=["symbol", "expiry", "open_int"])))
    inferred = pd.concat(parts, ignore_index=True)
    inferred["symbol"] = canonicalize(inferred["symbol"], inferred["expiry"], load_changes(settings))   # one name per company
    # a contract month can span two calendar years: combine its two partial answers
    inferred = inferred.groupby(["symbol", "expiry"], sort=False)["lot"].agg(lambda s: int(np.gcd.reduce(s.to_numpy(dtype=np.int64)))).reset_index()
    inferred["source"] = "inferred_from_open_interest"

    exch_path = settings.path("silver") / "lot_size.parquet"
    if exch_path.exists():
        exch = pd.read_parquet(exch_path, columns=["symbol", "expiry_month", "lot"]).rename(columns={"lot": "exchange_lot"})
        inferred["expiry_month"] = inferred["expiry"].values.astype("datetime64[M]").astype("datetime64[ns]")
        exch["expiry_month"] = pd.to_datetime(exch["expiry_month"]).values.astype("datetime64[M]").astype("datetime64[ns]")
        exch = exch.drop_duplicates(["symbol", "expiry_month"], keep="last")
        inferred = inferred.merge(exch, on=["symbol", "expiry_month"], how="left").drop(columns=["expiry_month"])
        has = inferred["exchange_lot"].notna()
        inferred["inferred_lot"] = inferred["lot"]
        inferred.loc[has, "lot"] = inferred.loc[has, "exchange_lot"].astype(int)
        inferred.loc[has, "source"] = "exchange"
        inferred = inferred.drop(columns=["exchange_lot"])
    else:
        inferred["inferred_lot"] = inferred["lot"]
    out = inferred.rename(columns={"symbol": "instrument"})[["instrument", "expiry", "lot", "inferred_lot", "source"]]
    out = out.sort_values(["instrument", "expiry"]).reset_index(drop=True)
    write_table(settings, LOT_SIZES, out)
    return out


def chain_on(settings: Settings, instrument: str, date) -> pd.DataFrame:
    """Every option and futures contract of one instrument on one day, with lot sizes."""
    date = pd.Timestamp(date).normalize()
    then = symbol_on(instrument, date, names_over_time(load_changes(settings)))     # the symbol in use that day
    df = read_contracts(settings, date.year, columns=CONTRACT_COLUMNS, symbols=[then])
    df = df[df["date"] == date].rename(columns={"symbol": "instrument", "instrument": "contract_type", "contracts": "volume"})
    df["instrument"] = instrument
    try:
        lots = read_table(settings, LOT_SIZES, columns=["instrument", "expiry", "lot"])
        df = df.merge(lots, on=["instrument", "expiry"], how="left")
    except StoreError:
        df["lot"] = np.nan
    return df.sort_values(["expiry", "opt_type", "strike"]).reset_index(drop=True)
