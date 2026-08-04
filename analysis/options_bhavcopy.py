"""Loader for NSE F&O bhavcopy (strike-wise option OHLC), unifying the old (zipped,
2010-mid2024) and new UDiFF (Aug2024+) formats into one schema:

    date, symbol, kind, expiry, strike, opt_type, open, high, low, close, settle, oi, vol, underlying

kind in {STKOPT, IDXOPT, STKFUT, IDXFUT}. Stock options = STKOPT.
"""

from __future__ import annotations

import os
import re
from datetime import datetime
from pathlib import Path

import pandas as pd

BASE = Path(r"C:\Users\rahul\Koscine 3.0\data\raw\derivatives_bhavcopy")

_OLD_KIND = {"OPTSTK": "STKOPT", "OPTIDX": "IDXOPT", "FUTSTK": "STKFUT", "FUTIDX": "IDXFUT"}
_NEW_KIND = {"STO": "STKOPT", "IDO": "IDXOPT", "STF": "STKFUT", "IDF": "IDXFUT"}
_MONTHS = "JAN FEB MAR APR MAY JUN JUL AUG SEP OCT NOV DEC".split()


def _old_path_for(date: pd.Timestamp) -> Path | None:
    fn = f"fo{date.day:02d}{_MONTHS[date.month-1]}{date.year}bhav.csv.zip"
    p = BASE / str(date.year) / fn
    return p if p.exists() else None


def _new_path_for(date: pd.Timestamp) -> Path | None:
    p = BASE / f"BhavCopy_NSE_FO_0_0_0_{date:%Y%m%d}_F_0000.csv"
    return p if p.exists() else None


def _new_zip_path_for(date: pd.Timestamp) -> Path | None:
    # New UDiFF format, zipped inside the year folder (used for 2026).
    p = BASE / str(date.year) / f"BhavCopy_NSE_FO_0_0_0_{date:%Y%m%d}_F_0000.csv.zip"
    return p if p.exists() else None


def _norm_old(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({
        "date": pd.to_datetime(df["TIMESTAMP"], format="%d-%b-%Y", errors="coerce"),
        "symbol": df["SYMBOL"].astype(str),
        "kind": df["INSTRUMENT"].map(_OLD_KIND),
        "expiry": pd.to_datetime(df["EXPIRY_DT"], format="%d-%b-%Y", errors="coerce"),
        "strike": pd.to_numeric(df["STRIKE_PR"], errors="coerce"),
        "opt_type": df["OPTION_TYP"].astype(str),
        "open": df["OPEN"], "high": df["HIGH"], "low": df["LOW"], "close": df["CLOSE"],
        "settle": df["SETTLE_PR"], "oi": df["OPEN_INT"], "vol": df["CONTRACTS"],
        "underlying": pd.NA,
    })
    return out


def _norm_new(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({
        "date": pd.to_datetime(df["TradDt"], errors="coerce"),
        "symbol": df["TckrSymb"].astype(str),
        "kind": df["FinInstrmTp"].map(_NEW_KIND),
        "expiry": pd.to_datetime(df["XpryDt"], errors="coerce"),
        "strike": pd.to_numeric(df["StrkPric"], errors="coerce"),
        "opt_type": df["OptnTp"].astype(str),
        "open": df["OpnPric"], "high": df["HghPric"], "low": df["LwPric"], "close": df["ClsPric"],
        "settle": df["SttlmPric"], "oi": df["OpnIntrst"], "vol": df["TtlTradgVol"],
        "underlying": pd.to_numeric(df["UndrlygPric"], errors="coerce"),
    })
    return out


def load_bhavcopy(date: pd.Timestamp, kinds: tuple[str, ...] = ("STKOPT",)) -> pd.DataFrame:
    """Load one trading day's F&O bhavcopy, normalized; prefers new format if both exist."""
    date = pd.Timestamp(date)
    p_new, p_newzip, p_old = _new_path_for(date), _new_zip_path_for(date), _old_path_for(date)
    if p_new is not None:
        df = _norm_new(pd.read_csv(p_new, low_memory=False))
    elif p_newzip is not None:
        df = _norm_new(pd.read_csv(p_newzip, compression="zip", low_memory=False))
    elif p_old is not None:
        df = _norm_old(pd.read_csv(p_old, compression="zip", low_memory=False))
    else:
        return pd.DataFrame()
    df = df[df["kind"].isin(kinds)].copy()
    return df


def available_dates() -> dict:
    """Quick inventory of how many daily files exist per year (both formats)."""
    inv = {}
    for entry in sorted(os.listdir(BASE)):
        full = BASE / entry
        if full.is_dir():
            inv[entry] = len([f for f in os.listdir(full) if f.endswith(".zip")])
        elif (m := re.search(r"_(\d{8})_", entry)):
            y = m.group(1)[:4]
            inv[f"{y}(new)"] = inv.get(f"{y}(new)", 0) + 1
    return inv


if __name__ == "__main__":
    print("inventory (files per year):")
    for k, v in available_dates().items():
        print(f"  {k}: {v}")

    # Validate both formats on a real stock option chain.
    for d in [pd.Timestamp("2025-12-31"), pd.Timestamp("2024-03-15")]:
        bc = load_bhavcopy(d)
        print(f"\n=== {d.date()} | STKOPT rows: {len(bc):,} | symbols: {bc['symbol'].nunique()} ===")
        if bc.empty:
            continue
        sym = "RELIANCE" if "RELIANCE" in set(bc["symbol"]) else bc["symbol"].iloc[0]
        chain = bc[bc["symbol"].eq(sym)]
        near_exp = sorted(chain["expiry"].dropna().unique())[0]
        ce = chain[chain["expiry"].eq(near_exp) & chain["opt_type"].eq("CE")].sort_values("strike")
        print(f"  {sym} nearest expiry {pd.Timestamp(near_exp).date()} | CE strikes: {len(ce)} | "
              f"underlying={chain['underlying'].dropna().iloc[0] if chain['underlying'].notna().any() else 'NA'}")
        cols = ["strike", "open", "high", "low", "close", "settle", "oi", "vol"]
        print(ce[cols].head(8).to_string(index=False))
