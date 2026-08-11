"""NIFTY support/resistance zones, computed with the EXACT SAME algorithm this repo already
uses for stocks (pipeline/zones.py -- swing-high/low clustering on daily->weekly resampled OHLC,
5% retreat valid-touch rule, consolidation/polarity-flip flags) rather than a separate
reimplementation. NIFTY is just treated as one more "symbol" fed through
zones.py::_build_symbol_zones_forward(), reusing its O(N) forward-pass builder directly.

Daily OHLC source: silver/indices.parquet's "Nifty 50" row (NSE's own official daily index
series, 2015-present -- ~11 years, comfortably past zones.py's own 104-week/2yr "long-term
zone" threshold) rather than resampling the shorter 2022-2026 nifty_5m.parquet, so the zone
history is as deep as what stocks get from their own multi-year daily history.

Output: locks/prod_nifty_intraday/nifty_zones.parquet, one row per trading day with the same
column layout pipeline/zones.py produces for stocks (resistance_level, support_level,
nearest_resistance_dist, nearest_support_dist, resistance/support_valid_touches,
zone_strength, consolidating_at_*, zone_breakout/breakdown, etc.) -- see that module's
docstring for the full field list. Sampled weekly (cadence_days=5, zones.py's own default --
zones move slowly, no reason to recompute daily) and forward-filled to daily resolution.

Usage:
    python -m koscine.nifty_zones
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from koscine.config import SILVER_DATA_ROOT
from pipeline.zones import _build_symbol_zones_forward

ROOT = Path(r"C:\Users\rahul\Koscine 3.0")
OUT_PATH = ROOT / "locks" / "prod_nifty_intraday" / "nifty_zones.parquet"


def load_nifty_daily() -> pd.DataFrame:
    """NIFTY 50's own daily OHLC from silver/indices.parquet, in the (date, open, high, low,
    close) shape zones.py's builder expects (same shape as silver/eod_stock.parquet per-symbol)."""
    idx = pd.read_parquet(SILVER_DATA_ROOT / "indices.parquet",
                          columns=["date", "index_name", "open", "high", "low", "close"])
    n = idx[idx["index_name"] == "Nifty 50"].copy()
    n["date"] = pd.to_datetime(n["date"])
    return n.sort_values("date")[["date", "open", "high", "low", "close"]].reset_index(drop=True)


def build_nifty_zones(cadence_days: int = 5) -> pd.DataFrame:
    daily = load_nifty_daily()
    sampled = _build_symbol_zones_forward(daily, cadence_days=cadence_days)
    if sampled.empty:
        raise SystemExit("zones.py returned no rows -- check silver/indices.parquet's Nifty 50 history")
    feat_cols = [c for c in sampled.columns if c != "date"]
    daily_idx = daily[["date"]].drop_duplicates().sort_values("date")
    out = daily_idx.merge(sampled, on="date", how="left").sort_values("date")
    out[feat_cols] = out[feat_cols].ffill()
    return out.reset_index(drop=True)


def latest_zones() -> dict:
    """Most recent day's zone snapshot -- what the live poller reads for its exit logic."""
    if not OUT_PATH.exists():
        return {}
    df = pd.read_parquet(OUT_PATH)
    if df.empty:
        return {}
    return df.iloc[-1].to_dict()


if __name__ == "__main__":
    out = build_nifty_zones()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_PATH, index=False)
    last = out.iloc[-1]
    print(f"[nifty_zones] {len(out)} daily rows ({out['date'].min().date()} -> {out['date'].max().date()}) "
          f"-> {OUT_PATH}")
    print(f"  latest ({last['date'].date()}): support={last['support_level']:.1f} "
          f"(dist {last['nearest_support_dist']:.3%}, touches={last['support_valid_touches']:.0f})  "
          f"resistance={last['resistance_level']:.1f} "
          f"(dist {last['nearest_resistance_dist']:.3%}, touches={last['resistance_valid_touches']:.0f})")
