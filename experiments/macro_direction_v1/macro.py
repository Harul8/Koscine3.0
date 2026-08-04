"""Leak-safe macro feature builder for macro_direction_v1.

Combines externally-fetched series (data/macro_raw.parquet from fetch_macro.py) with India VIX / Nifty IT
from LOCAL silver, then derives STATIONARY, properly-LAGGED features aligned onto NSE trading dates.

Alignment (predict India move d -> d+1 from the info set available at d+1 pre-open):
  USD/INR, Brent, WTI, DXY, India VIX  : EOD of day d (known by d evening)            -> attach to date d
  US indices (S&P / Nasdaq / Dow)      : US close on calendar date d (the overnight   -> attach to date d
                                         ending ~02:00 IST d+1, before NSE opens d+1)
Every feature on NSE date d is knowable before the d->d+1 move it predicts. US_LAG=1 = ultra-conservative
(only US closes already known at India EOD d). Holiday gaps: reindex onto NSE dates, ffill <= FFILL_LIMIT,
never backfill. Returns/changes only (except VIX level).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]                                  # .../Koscine 3.0
SILVER = ROOT / "data" / "silver" / "indices.parquet"
MACRO_RAW = HERE / "data" / "macro_raw.parquet"

US_LAG = 0            # 0 = US close dated d, legitimately known by a ~7AM IST d+1 pre-open refresh; 1 = only US closes known at India EOD d
FFILL_LIMIT = 2       # max days to carry a stale macro value across holiday gaps

MACRO_COLS = [
    "usdinr_ret_1", "usdinr_ret_5",
    "spx_overnight", "spx_ret_5", "ndx_overnight", "ndx_ret_5", "dji_overnight", "dji_ret_5",
    "brent_ret_1", "brent_ret_5", "wti_ret_1", "wti_ret_5", "dxy_ret_1", "dxy_ret_5",
    "vix_in", "vix_in_chg_1", "vix_in_ratio_5", "niftyit_ret_1",
]


def _india_index(name: str) -> pd.Series:
    idx = pd.read_parquet(SILVER, columns=["date", "index_name", "close"])
    s = idx[idx.index_name.eq(name)].set_index("date")["close"].sort_index()
    s.index = pd.to_datetime(s.index).normalize()
    return s[~s.index.duplicated(keep="last")].dropna()


def _nifty_it() -> pd.Series:
    # NSE renamed CNX IT -> Nifty IT (~2015); union both labels into one continuous series.
    parts = []
    for nm in ("CNX IT", "Nifty IT"):
        try:
            parts.append(_india_index(nm))
        except Exception:  # noqa: BLE001
            pass
    if not parts:
        return pd.Series(dtype=float)
    s = pd.concat(parts).sort_index()
    return s[~s.index.duplicated(keep="last")]


def macro_features(nse_dates) -> pd.DataFrame:
    """DataFrame[date, <macro features>] — one leak-safe row per NSE trading day in `nse_dates`."""
    nse = pd.DatetimeIndex(pd.to_datetime(pd.Index(nse_dates)).unique()).normalize().sort_values()

    raw = pd.DataFrame()
    if MACRO_RAW.exists():
        raw = pd.read_parquet(MACRO_RAW)
        raw["date"] = pd.to_datetime(raw["date"]).dt.normalize()
        raw = raw.set_index("date").sort_index()
    else:
        print(f"  [macro] note: {MACRO_RAW.name} not found — run fetch_macro.py; yfinance features will be NaN")

    # local India series (always available)
    raw["vix_in"] = _india_index("India VIX")
    raw["niftyit"] = _nifty_it()

    def aligned(col: str, lag: int = 0) -> "pd.Series | None":
        if col not in raw.columns:
            return None
        s = raw[col].dropna()
        if s.empty:
            return None
        if lag:
            s = s.shift(lag)
        full = s.reindex(s.index.union(nse)).sort_index().ffill(limit=FFILL_LIMIT)
        return full.reindex(nse)

    out = pd.DataFrame(index=nse)

    inr = aligned("usdinr")
    if inr is not None:
        out["usdinr_ret_1"] = inr.pct_change(1, fill_method=None)
        out["usdinr_ret_5"] = inr.pct_change(5, fill_method=None)

    for col, pre in [("spx", "spx"), ("ixic", "ndx"), ("dji", "dji")]:
        s = aligned(col, lag=US_LAG)
        if s is not None:
            out[f"{pre}_overnight"] = s.pct_change(1, fill_method=None)   # return since the last actionable session
            out[f"{pre}_ret_5"] = s.pct_change(5, fill_method=None)

    for col in ("brent", "wti", "dxy"):
        s = aligned(col)
        if s is not None:
            out[f"{col}_ret_1"] = s.pct_change(1, fill_method=None)
            out[f"{col}_ret_5"] = s.pct_change(5, fill_method=None)

    v = aligned("vix_in")
    if v is not None:
        out["vix_in"] = v
        out["vix_in_chg_1"] = v.diff(1)
        out["vix_in_ratio_5"] = v / v.rolling(5, min_periods=3).mean()

    it = aligned("niftyit")
    if it is not None:
        out["niftyit_ret_1"] = it.pct_change(1, fill_method=None)

    out = out.replace([np.inf, -np.inf], np.nan)
    out.index.name = "date"
    return out.reset_index()


if __name__ == "__main__":
    # light self-test (read-only; uses local VIX even without macro_raw)
    idx = pd.read_parquet(SILVER, columns=["date"])
    dts = pd.to_datetime(idx["date"]).sort_values().unique()[-300:]
    mf = macro_features(dts)
    present = [c for c in MACRO_COLS if c in mf.columns and mf[c].notna().any()]
    print(f"macro_features rows={len(mf)}  cols with data={len(present)}/{len(MACRO_COLS)}")
    print("populated:", present)
    print(mf.tail(3).to_string(index=False))
