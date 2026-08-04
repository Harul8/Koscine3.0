"""Clean-move base rates restricted to the tradeable predict universe.

Universe = top-N F&O symbols by median turnover_lacs over the 252 trading days
up to 2025-12-31 (same ranking the production universe builder uses).
Tolerances centered on the user's tradeable range: fix 2.0% and 0.5/0.55/0.6 x ATR.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PATH = r"C:\Users\rahul\Koscine 3.0\data\processed\daily_features.parquet"
WINDOW = 5
CUTOFF = pd.Timestamp("2025-12-31")
LOOKBACK = 252


def build_ranked_universe(df: pd.DataFrame, top_n: int) -> list[str]:
    elig = df[df["date"].le(CUTOFF)]
    dates = sorted(elig["date"].dropna().unique())[-LOOKBACK:]
    recent = elig[elig["date"].isin(dates)]
    exp_days = len(dates)
    m = recent.groupby("symbol").agg(
        observed=("date", "nunique"),
        med_turnover=("turnover_lacs", "median"),
        med_volume=("volume", "median"),
    )
    m["ohlc_missing"] = recent.groupby("symbol")[["open", "high", "low", "close"]].apply(
        lambda f: float(f.isna().any(axis=1).mean())
    )
    m["coverage"] = m["observed"] / exp_days
    m = m[(m["coverage"] >= 0.80) & (m["ohlc_missing"] <= 0.02)
          & (m["med_turnover"].fillna(0) >= 1.0) & (m["med_volume"].fillna(0) >= 1.0)]
    ranked = m.sort_values(["med_turnover", "med_volume"], ascending=False)
    return list(ranked.head(top_n).index)


def main() -> None:
    df = pd.read_parquet(PATH, columns=["date", "symbol", "open", "high", "low", "close",
                                        "turnover_lacs", "volume"])
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    g = df.groupby("symbol", sort=False)

    df["entry_open"] = g["open"].shift(-1)
    df["win_high"] = pd.concat([g["high"].shift(-i) for i in range(1, WINDOW + 1)], axis=1).max(axis=1)
    df["win_low"] = pd.concat([g["low"].shift(-i) for i in range(1, WINDOW + 1)], axis=1).min(axis=1)
    df["n_obs"] = pd.concat([g["close"].shift(-i) for i in range(1, WINDOW + 1)], axis=1).notna().sum(axis=1)
    prev_close = g["close"].shift(1)
    tr = pd.concat([(df["high"] - df["low"]).abs(), (df["high"] - prev_close).abs(),
                    (df["low"] - prev_close).abs()], axis=1).max(axis=1)
    df["atr_pct"] = df.assign(tr=tr).groupby("symbol")["tr"].transform(
        lambda s: s.rolling(14, min_periods=14).mean()) / df["close"]

    ev = df[df["entry_open"].notna() & (df["n_obs"] == WINDOW) & df["win_high"].notna()
            & df["win_low"].notna() & df["atr_pct"].notna() & (df["entry_open"] > 0)].copy()
    ev["year"] = ev["date"].dt.year
    ev["long_floor"] = (ev["entry_open"] - ev["win_low"]) / ev["entry_open"]
    ev["long_ceil"] = (ev["win_high"] - ev["entry_open"]) / ev["entry_open"]
    ev["short_floor"] = (ev["win_high"] - ev["entry_open"]) / ev["entry_open"]
    ev["short_ceil"] = (ev["entry_open"] - ev["win_low"]) / ev["entry_open"]

    universes = {"top65": build_ranked_universe(df, 65),
                 "top90": build_ranked_universe(df, 90),
                 "top100": build_ranked_universe(df, 100),
                 "all447": sorted(ev["symbol"].unique())}
    for k, v in universes.items():
        print(f"{k}: {len(v)} symbols")
    print(f"median ATR% (all): {ev['atr_pct'].median():.4f}")
    print()

    tol_specs = [("fix_2.0%", lambda d: 0.020),
                 ("0.5xATR", lambda d: 0.50 * d["atr_pct"]),
                 ("0.55xATR", lambda d: 0.55 * d["atr_pct"]),
                 ("0.6xATR", lambda d: 0.60 * d["atr_pct"])]

    for uname, syms in universes.items():
        sub = ev[ev["symbol"].isin(syms)]
        print(f"################## UNIVERSE = {uname} ({sub['symbol'].nunique()} symbols, {len(sub):,} stock-days) ##################")
        for side, fcol, ccol in [("long", "long_floor", "long_ceil"), ("short", "short_floor", "short_ceil")]:
            rows = []
            for tname, tolfn in tol_specs:
                tol = tolfn(sub)
                clean = sub[fcol] <= tol
                n = int(clean.sum())
                cl = sub.loc[clean, ccol]
                rows.append({
                    "tol": tname, "clean_rate": round(n / len(sub), 4), "n_clean": n,
                    "ceil_med": round(float(cl.median()), 4) if n else 0,
                    "ceil_p75": round(float(cl.quantile(.75)), 4) if n else 0,
                    "cln&>=4%": round(float((clean & (sub[ccol] >= 0.04)).mean()), 4),
                    "cln&>=6%": round(float((clean & (sub[ccol] >= 0.06)).mean()), 4),
                    "cln&>=8%": round(float((clean & (sub[ccol] >= 0.08)).mean()), 4),
                })
            print(f"  --- {side} ---")
            print(pd.DataFrame(rows).to_string(index=False).replace("\n", "\n  "))
        print()

    # Daily oracle opportunity: top-ceiling per day when ranking clean candidates (0.6xATR)
    print("########## DAILY ORACLE OPPORTUNITY (clean, tol=0.6xATR) ##########")
    print("For each day: pick clean candidates, look at the realized ceiling of the best ones.")
    for uname, syms in universes.items():
        sub = ev[ev["symbol"].isin(syms)].copy()
        tol = 0.6 * sub["atr_pct"]
        for side, fcol, ccol in [("long", "long_floor", "long_ceil"), ("short", "short_floor", "short_ceil")]:
            clean = sub[sub[fcol] <= tol]
            per_day = clean.groupby("date")[ccol].agg(["size", "max",
                                                       lambda s: s.nlargest(3).mean(),
                                                       lambda s: s.nlargest(5).mean()])
            per_day.columns = ["n_clean", "top1", "top3_mean", "top5_mean"]
            print(f"  {uname:7s} {side:5s} | days={len(per_day):4d} "
                  f"| med clean/day={per_day['n_clean'].median():5.1f} "
                  f"| med top1={per_day['top1'].median():.4f} "
                  f"| med top3={per_day['top3_mean'].median():.4f} "
                  f"| med top5={per_day['top5_mean'].median():.4f}")
        print()

    # Recent-year stability for top100 at 0.6xATR
    print("########## BY-YEAR (top100, tol=0.6xATR) ##########")
    sub = ev[ev["symbol"].isin(universes["top100"])].copy()
    tol = 0.6 * sub["atr_pct"]
    for side, fcol, ccol in [("long", "long_floor", "long_ceil"), ("short", "short_floor", "short_ceil")]:
        clean = sub[fcol] <= tol
        yr = sub.assign(_c=clean).groupby("year").apply(
            lambda d: pd.Series({
                "clean_rate": round(d["_c"].mean(), 4),
                "clean_ceil_med": round(float(d.loc[d["_c"], ccol].median()), 4) if d["_c"].any() else 0,
                "cln&>=6%": round(float((d["_c"] & (d[ccol] >= 0.06)).mean()), 4),
            }), include_groups=False)
        print(f"  --- {side} ---")
        print(yr.to_string().replace("\n", "\n  "))
        print()


if __name__ == "__main__":
    main()
