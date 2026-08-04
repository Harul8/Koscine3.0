"""Read-only base-rate scan for the clean-move contract.

Clean (long)  : min_low over t+1..t+5 stays >= entry_open * (1 - tol)
Ceiling (long): (max_high over t+1..t+5 - entry_open) / entry_open
Short mirrors with high/low swapped.

tol variants: fixed {0.3%, 0.5%, 1.0%} and volatility-scaled {0.3*ATR%, 0.5*ATR%}.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

PATH = r"C:\Users\rahul\Koscine 3.0\data\processed\daily_features.parquet"
WINDOW = 5


def pick_turnover_col(all_cols: list[str]) -> str | None:
    for c in ["turnover", "traded_value", "value", "delivery_value"]:
        if c in all_cols:
            return c
    return None


def main() -> None:
    schema_cols = pq.ParquetFile(PATH).schema.names
    tcol = pick_turnover_col(schema_cols)
    cols = ["date", "symbol", "open", "high", "low", "close"]
    if tcol:
        cols.append(tcol)
    df = pd.read_parquet(PATH, columns=cols)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    g = df.groupby("symbol", sort=False)

    df["entry_open"] = g["open"].shift(-1)
    high_next = pd.concat([g["high"].shift(-i) for i in range(1, WINDOW + 1)], axis=1)
    low_next = pd.concat([g["low"].shift(-i) for i in range(1, WINDOW + 1)], axis=1)
    close_next = pd.concat([g["close"].shift(-i) for i in range(1, WINDOW + 1)], axis=1)
    df["win_high"] = high_next.max(axis=1)
    df["win_low"] = low_next.min(axis=1)
    df["n_obs"] = close_next.notna().sum(axis=1)

    # ATR(14) known at signal day t
    prev_close = g["close"].shift(1)
    tr = pd.concat(
        [(df["high"] - df["low"]).abs(),
         (df["high"] - prev_close).abs(),
         (df["low"] - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    df["tr"] = tr
    df["atr14"] = df.groupby("symbol")["tr"].transform(lambda s: s.rolling(14, min_periods=14).mean())
    df["atr_pct"] = df["atr14"] / df["close"]

    ev = df[
        df["entry_open"].notna()
        & (df["n_obs"] == WINDOW)
        & df["win_high"].notna()
        & df["win_low"].notna()
        & df["atr_pct"].notna()
        & (df["entry_open"] > 0)
    ].copy()
    ev["year"] = ev["date"].dt.year

    # Long contract
    ev["long_floor_depth"] = (ev["entry_open"] - ev["win_low"]) / ev["entry_open"]
    ev["long_ceiling"] = (ev["win_high"] - ev["entry_open"]) / ev["entry_open"]
    # Short contract
    ev["short_floor_depth"] = (ev["win_high"] - ev["entry_open"]) / ev["entry_open"]
    ev["short_ceiling"] = (ev["entry_open"] - ev["win_low"]) / ev["entry_open"]

    print(f"Loaded {len(df):,} rows | evaluated stock-days: {len(ev):,} | turnover col: {tcol}")
    print(f"date range: {ev['date'].min().date()} .. {ev['date'].max().date()}")
    print(f"median ATR%: {ev['atr_pct'].median():.4f}  p25={ev['atr_pct'].quantile(.25):.4f}  p75={ev['atr_pct'].quantile(.75):.4f}")
    print()

    tol_specs = [
        ("fix_0.3%", lambda d: 0.003),
        ("fix_0.5%", lambda d: 0.005),
        ("fix_1.0%", lambda d: 0.010),
        ("0.3xATR", lambda d: 0.3 * d["atr_pct"]),
        ("0.5xATR", lambda d: 0.5 * d["atr_pct"]),
    ]

    for side, fd, ceil in [("long", "long_floor_depth", "long_ceiling"),
                           ("short", "short_floor_depth", "short_ceiling")]:
        print(f"===== SIDE = {side.upper()} =====")
        rows = []
        for name, tolfn in tol_specs:
            tol = tolfn(ev)
            clean = ev[fd] <= tol
            n_clean = int(clean.sum())
            clean_rate = n_clean / len(ev)
            cl = ev.loc[clean, ceil]
            rows.append({
                "tol": name,
                "clean_rate": round(clean_rate, 4),
                "n_clean": n_clean,
                "ceil_med": round(float(cl.median()), 4) if n_clean else 0,
                "ceil_p75": round(float(cl.quantile(.75)), 4) if n_clean else 0,
                "ceil_p90": round(float(cl.quantile(.90)), 4) if n_clean else 0,
                "clean&>=3%": round(float((clean & (ev[ceil] >= 0.03)).sum()) / len(ev), 4),
                "clean&>=4%": round(float((clean & (ev[ceil] >= 0.04)).sum()) / len(ev), 4),
                "clean&>=5%": round(float((clean & (ev[ceil] >= 0.05)).sum()) / len(ev), 4),
            })
        print(pd.DataFrame(rows).to_string(index=False))
        print()

        # By-year stability for the 0.5xATR tolerance
        print(f"  by-year clean rate & clean ceiling median ({side}, tol=0.5xATR):")
        tol = 0.5 * ev["atr_pct"]
        clean = ev[fd] <= tol
        yr = ev.assign(_clean=clean).groupby("year").apply(
            lambda d: pd.Series({
                "evaluated": len(d),
                "clean_rate": round(d["_clean"].mean(), 4),
                "clean_ceil_med": round(float(d.loc[d["_clean"], ceil].median()), 4) if d["_clean"].any() else 0,
                "clean&>=4%_rate": round(float((d["_clean"] & (d[ceil] >= 0.04)).mean()), 4),
            }),
            include_groups=False,
        )
        print(yr.to_string())
        print()

    # Daily opportunity view: among clean-long (0.5xATR), how many per day and top ceiling
    tol = 0.5 * ev["atr_pct"]
    clean_long = ev[ev["long_floor_depth"] <= tol]
    per_day = clean_long.groupby("date").agg(
        n_clean=("long_ceiling", "size"),
        top_ceiling=("long_ceiling", "max"),
    )
    print("===== DAILY CLEAN-LONG OPPORTUNITY (tol=0.5xATR, full universe) =====")
    print(f"  trading days with >=1 clean long: {len(per_day):,}")
    print(f"  median clean-longs per day: {per_day['n_clean'].median():.0f}")
    print(f"  median of daily TOP ceiling: {per_day['top_ceiling'].median():.4f}")
    print(f"  p25/p75 of daily top ceiling: {per_day['top_ceiling'].quantile(.25):.4f} / {per_day['top_ceiling'].quantile(.75):.4f}")


if __name__ == "__main__":
    main()
