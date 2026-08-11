"""Do large NIFTY moves have any relation to short-term (trailing-5-trading-day) support/
resistance/consolidation zones on 15-min candles -- specifically, do moves accelerate when
price BREAKS one of these zones (makes a new 5-day high/low)? Separate question from
koscine/nifty_zones.py's weekly-swing zones (that's the long-horizon version reused from
stocks); this is a much shorter, faster-moving range detector purpose-built for this question.

Method: Donchian-channel-style box -- trailing 5-trading-day rolling high/low on 15-min bars
(~125 bars). A "breakout" bar closes above the trailing high (a new 5-day high); a "breakdown"
bar closes below the trailing low. Compares the forward move magnitude (same direction-agnostic
metric as labels.py, just recomputed at 15-min resolution) immediately after a break vs. the
unconditional baseline, and separately checks whether a TIGHTER box beforehand (more compressed
range = more "coiled") predicts a bigger release.

Usage:
    python experiments/nifty_intraday_breakout_v1/short_term_zone_analysis.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"C:\Users\rahul\Koscine 3.0")
NIFTY_5M = ROOT / "data" / "intraday" / "nifty_5m.parquet"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

BOX_TRADING_DAYS = 5
BARS_PER_DAY_15M = 25          # 09:15-15:30 in 15-min steps
BOX_BARS = BOX_TRADING_DAYS * BARS_PER_DAY_15M
FWD_BARS = 4                    # ~1hr forward window at 15-min resolution, same-day only


def resample_15m(bars: pd.DataFrame) -> pd.DataFrame:
    df = bars.set_index("timestamp").sort_index()
    df["date"] = df.index.date
    out = []
    for _, day in df.groupby("date", sort=False):
        r = day.resample("15min", origin=day.index[0]).agg(
            {"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
        r["date"] = day["date"].iloc[0]
        out.append(r)
    result = pd.concat(out).reset_index()
    return result


def compute_box_and_breaks(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["trailing_high"] = df["high"].rolling(BOX_BARS, min_periods=BOX_BARS).max().shift(1)
    df["trailing_low"] = df["low"].rolling(BOX_BARS, min_periods=BOX_BARS).min().shift(1)
    df["box_width_pct"] = (df["trailing_high"] - df["trailing_low"]) / df["close"]
    df["breakout"] = df["close"] > df["trailing_high"]
    df["breakdown"] = df["close"] < df["trailing_low"]
    df["break_event"] = df["breakout"] | df["breakdown"]
    return df


def add_forward_move(df: pd.DataFrame, fwd_bars: int = FWD_BARS) -> pd.DataFrame:
    day = df.groupby("date", sort=False)
    highs = pd.concat([day["high"].transform(lambda s, k=k: s.shift(-k)) for k in range(1, fwd_bars + 1)], axis=1)
    lows = pd.concat([day["low"].transform(lambda s, k=k: s.shift(-k)) for k in range(1, fwd_bars + 1)], axis=1)
    complete = highs.notna().sum(axis=1).eq(fwd_bars)
    up = (highs.max(axis=1) - df["close"]) / df["close"]
    down = (df["close"] - lows.min(axis=1)) / df["close"]
    mag = pd.concat([up, down], axis=1).max(axis=1)
    df["fwd_move_mag"] = mag.where(complete)
    return df


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    bars = pd.read_parquet(NIFTY_5M)
    bars["timestamp"] = pd.to_datetime(bars["timestamp"])
    df15 = resample_15m(bars)
    print(f"[short_term_zone] {len(df15):,} 15-min bars, {df15['date'].nunique()} trading days")

    df15 = compute_box_and_breaks(df15)
    df15 = add_forward_move(df15)
    d = df15.dropna(subset=["fwd_move_mag", "trailing_high", "trailing_low"])
    print(f"[short_term_zone] {len(d):,} bars with valid box + forward-move data")

    baseline = d["fwd_move_mag"].mean()
    breakout = d.loc[d["breakout"], "fwd_move_mag"]
    breakdown = d.loc[d["breakdown"], "fwd_move_mag"]
    no_break = d.loc[~d["break_event"], "fwd_move_mag"]

    print(f"\nunconditional mean fwd_move_mag: {baseline:.5f}")
    print(f"no-break bars    (n={len(no_break):6d}): mean={no_break.mean():.5f}")
    print(f"breakout bars    (n={len(breakout):6d}): mean={breakout.mean():.5f}  "
          f"ratio_vs_no_break={breakout.mean()/no_break.mean():.2f}x")
    print(f"breakdown bars   (n={len(breakdown):6d}): mean={breakdown.mean():.5f}  "
          f"ratio_vs_no_break={breakdown.mean()/no_break.mean():.2f}x")

    # tighter box beforehand -> bigger release? bucket by box_width_pct quintile among break events
    breaks = d[d["break_event"]].copy()
    breaks["width_quintile"] = pd.qcut(breaks["box_width_pct"], 5, labels=False, duplicates="drop")
    by_width = breaks.groupby("width_quintile")["fwd_move_mag"].agg(["mean", "count"])
    print("\nfwd_move_mag by box-width quintile among break events (0=tightest box beforehand):")
    print(by_width)

    lines = [
        "# Do large NIFTY moves relate to trailing-5-day 15-min S/R zones?", "",
        f"Unconditional mean fwd_move_mag ({FWD_BARS} bars / ~1hr): {baseline:.5f}", "",
        f"- No-break bars: mean={no_break.mean():.5f} (n={len(no_break)})",
        f"- Breakout bars (new 5-day high): mean={breakout.mean():.5f}, "
        f"{breakout.mean()/no_break.mean():.2f}x no-break (n={len(breakout)})",
        f"- Breakdown bars (new 5-day low): mean={breakdown.mean():.5f}, "
        f"{breakdown.mean()/no_break.mean():.2f}x no-break (n={len(breakdown)})",
        "", "## fwd_move_mag by box-width quintile among break events (0=tightest beforehand)", "",
        "```", by_width.to_string(), "```", "",
    ]
    (RESULTS_DIR / "FINDINGS_short_term_zones.md").write_text("\n".join(lines), encoding="utf-8")
    d.to_parquet(RESULTS_DIR / "short_term_zone_data.parquet", index=False)
    print(f"\n[short_term_zone] wrote results -> {RESULTS_DIR}")


if __name__ == "__main__":
    main()
