"""Fetch EOD macro / cross-asset series for macro_direction_v1 (EXTERNAL — run from terminal, needs internet).

Pulls daily close for USD/INR, US indices (S&P 500, Nasdaq, Dow), crude (Brent, WTI) and the dollar index
via yfinance -> experiments/macro_direction_v1/data/macro_raw.parquet. India VIX / Nifty IT come from our
LOCAL silver (see macro.py) and are NOT fetched here.

    python experiments/macro_direction_v1/fetch_macro.py

Idempotent (overwrites macro_raw.parquet). Touches no PROD artifact.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "data"
START = "2010-01-01"

# yfinance ticker -> our column name
TICKERS = {
    "INR=X": "usdinr",      # USD/INR spot
    "^GSPC": "spx",         # S&P 500
    "^IXIC": "ixic",        # Nasdaq Composite (-> ndx_* features)
    "^DJI": "dji",          # Dow Jones Industrial Average
    "BZ=F": "brent",        # Brent crude (India-relevant)
    "CL=F": "wti",          # WTI crude
    "DX-Y.NYB": "dxy",      # US Dollar Index
}


def fetch_one(ticker: str) -> "pd.Series | None":
    import yfinance as yf
    try:
        h = yf.Ticker(ticker).history(start=START, auto_adjust=False)
        if h is None or h.empty:
            h = yf.download(ticker, start=START, progress=False, auto_adjust=False)
        if h is None or h.empty:
            return None
        s = h["Close"]
        if isinstance(s, pd.DataFrame):          # multi-ticker download edge case
            s = s.iloc[:, 0]
        s = s.copy()
        s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
        s = s[~s.index.duplicated(keep="last")].dropna()
        return s
    except Exception as e:  # noqa: BLE001
        print(f"  !! {ticker}: {type(e).__name__}: {str(e)[:110]}")
        return None


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cols: dict[str, pd.Series] = {}
    for tk, name in TICKERS.items():
        s = fetch_one(tk)
        if s is None or s.empty:
            print(f"  MISS {name:7s} ({tk})")
            continue
        cols[name] = s
        print(f"  OK   {name:7s} ({tk:10s}) {len(s):5d} rows  {s.index.min().date()}..{s.index.max().date()}")
    if not cols:
        raise SystemExit("no series fetched — check internet / `pip install -U yfinance`")
    df = pd.DataFrame(cols).sort_index()
    df.index.name = "date"
    p = OUT / "macro_raw.parquet"
    df.reset_index().to_parquet(p, index=False)
    print(f"\nwrote {p}   shape={df.shape}   series={list(df.columns)}")


if __name__ == "__main__":
    main()
