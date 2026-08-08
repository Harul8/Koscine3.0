"""Generate the historical Cash Signals track record: every historical date a breakout /
breakdown / consolidation signal actually FIRED (a zone_breakout / zone_breakdown /
consolidating_at_* flag flipping 0->1, not a sustained flag -- otherwise one multi-week
breakout would be double/triple counted), with the realized outcome over fixed forward
horizons. Reuses gold/zones.parquet (pipeline/zones.py) + silver/eod_stock.parquet for
forward closes/highs/lows (same source zones.py itself used, so every zone row has a match).
Writes locks/prod_cash_signals/cash_signal_history.csv for the API (/prod2/cash_signal_history)
to serve.

Signal / outcome definitions:
  breakout      : zone_breakout      0->1. win = fwd_return_10d > 0.  held = close[t+10] > resistance_level.
  breakdown     : zone_breakdown     0->1. win = fwd_return_10d < 0.  held = close[t+10] < support_level.
  consolidation : (consolidating_at_resistance | consolidating_at_support) 0->1.
                  win = no zone_breakout/zone_breakdown fired for that symbol within the 10d window
                  (i.e. it actually stayed contained). realized_range_10d_pct recorded for context.

Pipeline:
  1. python -m pipeline.zones --full            # one-time: builds gold/zones.parquet (long-running)
  2. python analysis/gen_cash_signal_history.py  # -> locks/prod_cash_signals/cash_signal_history.csv
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(r"C:\Users\rahul\Koscine 3.0")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))
from pipeline.config import GOLD_ZONES, SILVER_TABLES  # noqa: E402

HORIZONS = (5, 10, 20)
PRIMARY = 10

if not GOLD_ZONES.exists():
    print(f"[cash_signal_history] {GOLD_ZONES} does not exist -- run: python -m pipeline.zones --full")
    sys.exit(1)

zones = pd.read_parquet(GOLD_ZONES)
zones["date"] = pd.to_datetime(zones["date"]); zones = zones.sort_values(["symbol", "date"])

stock = pd.read_parquet(SILVER_TABLES["eod_stock"], columns=["date", "symbol", "close", "high", "low"])
stock["date"] = pd.to_datetime(stock["date"]); stock["symbol"] = stock["symbol"].astype(str)
stock = stock[stock["symbol"].isin(zones["symbol"].unique())].sort_values(["symbol", "date"])

tdays = np.array(sorted(stock["date"].unique()))
tpos = {d: i for i, d in enumerate(tdays)}
by_sym = {s: g.set_index("date") for s, g in stock.groupby("symbol", sort=False)}


def fwd_window(sym: str, t: pd.Timestamp, n: int):
    """Rows for `sym` from t (exclusive) through the n-th following trading day on the GLOBAL
    calendar, using whatever dates that symbol actually has (handles listing gaps without
    misaligning the horizon). None if the symbol has no data in that window (e.g. delisted)."""
    p = tpos.get(t)
    g = by_sym.get(sym)
    if p is None or g is None:
        return None
    end = tdays[min(p + n, len(tdays) - 1)]
    w = g[(g.index > t) & (g.index <= end)]
    return w if not w.empty else None


def close_at(sym: str, t: pd.Timestamp):
    g = by_sym.get(sym)
    if g is None or t not in g.index:
        return None
    return float(g.loc[t, "close"])


zones["prev_breakout"] = zones.groupby("symbol")["zone_breakout"].shift(1).fillna(0)
zones["fire_breakout"] = (zones["zone_breakout"] == 1) & (zones["prev_breakout"] == 0)
zones["prev_breakdown"] = zones.groupby("symbol")["zone_breakdown"].shift(1).fillna(0)
zones["fire_breakdown"] = (zones["zone_breakdown"] == 1) & (zones["prev_breakdown"] == 0)
zones["cons_flag"] = ((zones["consolidating_at_resistance"] == 1) | (zones["consolidating_at_support"] == 1)).astype(int)
zones["prev_cons"] = zones.groupby("symbol")["cons_flag"].shift(1).fillna(0)
zones["fire_cons"] = (zones["cons_flag"] == 1) & (zones["prev_cons"] == 0)

out: list[dict] = []

for _, r in zones[zones["fire_breakout"]].iterrows():
    sym, t = r["symbol"], r["date"]
    c0 = close_at(sym, t)
    w10 = fwd_window(sym, t, PRIMARY)
    if c0 is None or c0 <= 0 or w10 is None or pd.isna(r["resistance_level"]):
        continue
    c10 = float(w10["close"].iloc[-1])
    fwd10 = c10 / c0 - 1
    row = {
        "signal_type": "breakout", "symbol": sym, "signal_date": t.date().isoformat(), "entry_close": round(c0, 1),
        "resistance_level": round(float(r["resistance_level"]), 1),
        "resistance_zone_strength": round(float(r["resistance_zone_strength"]), 2),
        "fwd_return_10d": round(fwd10 * 100, 2), "held_10d": bool(c10 > r["resistance_level"]),
        "outcome": "win" if fwd10 > 0 else "loss",
    }
    for h in HORIZONS:
        wh = fwd_window(sym, t, h)
        row[f"fwd_return_{h}d"] = round((float(wh["close"].iloc[-1]) / c0 - 1) * 100, 2) if wh is not None else None
    out.append(row)

for _, r in zones[zones["fire_breakdown"]].iterrows():
    sym, t = r["symbol"], r["date"]
    c0 = close_at(sym, t)
    w10 = fwd_window(sym, t, PRIMARY)
    if c0 is None or c0 <= 0 or w10 is None or pd.isna(r["support_level"]):
        continue
    c10 = float(w10["close"].iloc[-1])
    fwd10 = c10 / c0 - 1
    row = {
        "signal_type": "breakdown", "symbol": sym, "signal_date": t.date().isoformat(), "entry_close": round(c0, 1),
        "support_level": round(float(r["support_level"]), 1),
        "support_zone_strength": round(float(r["support_zone_strength"]), 2),
        "fwd_return_10d": round(fwd10 * 100, 2), "held_10d": bool(c10 < r["support_level"]),
        "outcome": "win" if fwd10 < 0 else "loss",
    }
    for h in HORIZONS:
        wh = fwd_window(sym, t, h)
        row[f"fwd_return_{h}d"] = round((float(wh["close"].iloc[-1]) / c0 - 1) * 100, 2) if wh is not None else None
    out.append(row)

# consolidation success = no breakout/breakdown fired for this symbol within the 10d window
brk_dates = zones[zones["fire_breakout"] | zones["fire_breakdown"]].groupby("symbol")["date"].apply(list).to_dict()
for _, r in zones[zones["fire_cons"]].iterrows():
    sym, t = r["symbol"], r["date"]
    c0 = close_at(sym, t)
    w10 = fwd_window(sym, t, PRIMARY)
    if c0 is None or c0 <= 0 or w10 is None or pd.isna(r["zone_box_width_pct"]):
        continue
    end10 = w10.index[-1]
    later_fires = [d for d in brk_dates.get(sym, []) if t < d <= end10]
    realized_range = (float(w10["high"].max()) - float(w10["low"].min())) / c0
    out.append({
        "signal_type": "consolidation", "symbol": sym, "signal_date": t.date().isoformat(), "entry_close": round(c0, 1),
        "resistance_level": round(float(r["resistance_level"]), 1) if pd.notna(r["resistance_level"]) else None,
        "support_level": round(float(r["support_level"]), 1) if pd.notna(r["support_level"]) else None,
        "zone_box_width_pct": round(float(r["zone_box_width_pct"]) * 100, 2),
        "realized_range_10d_pct": round(realized_range * 100, 2),
        "outcome": "win" if not later_fires else "loss",
    })

df = pd.DataFrame(out).sort_values("signal_date")
outdir = ROOT / "locks" / "prod_cash_signals"
outdir.mkdir(parents=True, exist_ok=True)
df.to_csv(outdir / "cash_signal_history.csv", index=False)
print(f"wrote {len(df)} signals -> {outdir / 'cash_signal_history.csv'}")
for cat, g in df.groupby("signal_type"):
    wr = (g["outcome"] == "win").mean()
    if "fwd_return_10d" in g.columns and g["fwd_return_10d"].notna().any():
        extra = f"avg fwd10d {g['fwd_return_10d'].mean():.2f}%  held_rate {g['held_10d'].mean():.1%}"
    else:
        extra = f"avg realized range {g['realized_range_10d_pct'].mean():.2f}%"
    print(f"  {cat}: n={len(g)}  win_rate={wr:.1%}  {extra}")
