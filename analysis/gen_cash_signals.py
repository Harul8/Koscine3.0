"""Generate today's live Cash Signals picks: stocks breaking out of a resistance zone,
breaking down through a support zone, or consolidating tightly inside one -- scanned across
the full F&O universe (every symbol pipeline/zones.py has ever built a zone for). Reuses
pipeline/zones.py's zone engine (gold/zones.parquet) for the S/R levels and flags, joined to
silver/eod_stock.parquet for close/volume (same source zones.py itself used, so dates always
line up) and daily_features.parquet for a couple of already-live compression columns as
secondary/confirmatory context only. Writes locks/prod_cash_signals/cash_signals.json for the
API (/prod2/cash_signals) to serve.

Pipeline:
  1. python -m pipeline.zones --full       # one-time: builds gold/zones.parquet (long-running)
  2. python analysis/gen_cash_signals.py   # -> locks/prod_cash_signals/cash_signals.json
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(r"C:\Users\rahul\Koscine 3.0")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))
from pipeline.config import GOLD_ZONES, SILVER_TABLES  # noqa: E402
from koscine3.data.sources import load_market_data  # noqa: E402

TOP_N = 10

if not GOLD_ZONES.exists():
    print(f"[cash_signals] {GOLD_ZONES} does not exist yet -- run: python -m pipeline.zones --full")
    sys.exit(1)

zones = pd.read_parquet(GOLD_ZONES)
zones["date"] = pd.to_datetime(zones["date"])
last_date = zones["date"].max()
latest = zones[zones["date"] == last_date].copy()
print(f"[cash_signals] zones as of {last_date.date()}, {zones['symbol'].nunique()} symbols in universe")

# close/volume: same source zones.py itself used to build gold/zones.parquet, so the dates
# always line up (no risk of a stale/ahead mismatch against a separately-refreshed table).
stock = pd.read_parquet(SILVER_TABLES["eod_stock"], columns=["date", "symbol", "close", "volume"])
stock["date"] = pd.to_datetime(stock["date"]); stock["symbol"] = stock["symbol"].astype(str)
stock = stock[stock["symbol"].isin(latest["symbol"].unique())].sort_values(["symbol", "date"])
stock["vol20"] = stock.groupby("symbol")["volume"].transform(lambda s: s.rolling(20, min_periods=5).mean())
sday = stock[stock["date"] == last_date].set_index("symbol")

# compression columns: secondary/confirmatory context only, sourced from the separately-refreshed
# daily_features.parquet, so may lag zones by a few days -- best-available join, not primary logic.
try:
    mk = load_market_data(columns=["date", "symbol", "bb_width_20_rank_60d", "atr_pct_14_rank_60d", "compression_composite"])
    mk["date"] = pd.to_datetime(mk["date"]); mk["symbol"] = mk["symbol"].astype(str)
    mkday = mk[mk["date"] == mk["date"].max()].set_index("symbol")
except FileNotFoundError:
    mkday = pd.DataFrame()


def _ctx(sym: str) -> dict:
    close = vol_ratio = compression = None
    if sym in sday.index:
        row = sday.loc[sym]
        close = float(row["close"]) if pd.notna(row.get("close")) else None
        if pd.notna(row.get("volume")) and pd.notna(row.get("vol20")) and row.get("vol20", 0) > 0:
            vol_ratio = round(float(row["volume"]) / float(row["vol20"]), 2)
    if not mkday.empty and sym in mkday.index and pd.notna(mkday.loc[sym].get("compression_composite")):
        compression = round(float(mkday.loc[sym]["compression_composite"]), 3)
    return {"close": round(close, 1) if close is not None else None, "vol_ratio_20d": vol_ratio, "compression_composite": compression}


def _row(r: pd.Series, category: str) -> dict:
    base = {"symbol": r["symbol"], "as_of": r["date"].date().isoformat(), **_ctx(r["symbol"])}
    close = base["close"]
    if category == "breakout":
        level = r["resistance_level"]
        base.update({
            "resistance_level": round(float(level), 1) if pd.notna(level) else None,
            "pct_beyond_level": round((close - level) / level * 100, 2) if close is not None and pd.notna(level) and level else None,
            "resistance_valid_touches": int(r["resistance_valid_touches"]) if pd.notna(r["resistance_valid_touches"]) else 0,
            "resistance_zone_strength": round(float(r["resistance_zone_strength"]), 2) if pd.notna(r["resistance_zone_strength"]) else 0.0,
            "resistance_zone_age_weeks": round(float(r["resistance_zone_age_weeks"]), 1) if pd.notna(r["resistance_zone_age_weeks"]) else None,
        })
    elif category == "breakdown":
        level = r["support_level"]
        base.update({
            "support_level": round(float(level), 1) if pd.notna(level) else None,
            "pct_beyond_level": round((level - close) / level * 100, 2) if close is not None and pd.notna(level) and level else None,
            "support_valid_touches": int(r["support_valid_touches"]) if pd.notna(r["support_valid_touches"]) else 0,
            "support_zone_strength": round(float(r["support_zone_strength"]), 2) if pd.notna(r["support_zone_strength"]) else 0.0,
            "support_zone_age_weeks": round(float(r["support_zone_age_weeks"]), 1) if pd.notna(r["support_zone_age_weeks"]) else None,
        })
    else:  # consolidation
        base.update({
            "resistance_level": round(float(r["resistance_level"]), 1) if pd.notna(r["resistance_level"]) else None,
            "support_level": round(float(r["support_level"]), 1) if pd.notna(r["support_level"]) else None,
            "zone_box_width_pct": round(float(r["zone_box_width_pct"]) * 100, 2) if pd.notna(r["zone_box_width_pct"]) else None,
            "weeks_in_box": int(r["weeks_in_box"]) if pd.notna(r["weeks_in_box"]) else None,
        })
    return base


breakout = latest[latest["zone_breakout"] == 1]
breakdown = latest[latest["zone_breakdown"] == 1]
consolidating = latest[((latest["consolidating_at_resistance"] == 1) | (latest["consolidating_at_support"] == 1))
                        & latest["zone_box_width_pct"].notna()]

breakout_rows = sorted((_row(r, "breakout") for _, r in breakout.iterrows()),
                        key=lambda r: r["resistance_zone_strength"], reverse=True)[:TOP_N]
breakdown_rows = sorted((_row(r, "breakdown") for _, r in breakdown.iterrows()),
                         key=lambda r: r["support_zone_strength"], reverse=True)[:TOP_N]
consolidation_rows = sorted((_row(r, "consolidation") for _, r in consolidating.iterrows()),
                             key=lambda r: (r["zone_box_width_pct"], -(r["weeks_in_box"] or 0)))[:TOP_N]

out = {
    "as_of": last_date.date().isoformat(),
    "universe_size": int(zones["symbol"].nunique()),
    "breakout": breakout_rows,
    "breakdown": breakdown_rows,
    "consolidation": consolidation_rows,
}

outdir = ROOT / "locks" / "prod_cash_signals"
outdir.mkdir(parents=True, exist_ok=True)
(outdir / "cash_signals.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
print(f"[cash_signals] wrote {len(breakout_rows)} breakout / {len(breakdown_rows)} breakdown / "
      f"{len(consolidation_rows)} consolidation picks -> {outdir / 'cash_signals.json'}")
