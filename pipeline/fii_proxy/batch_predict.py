"""
Batch-score a date range using the FII accumulation proxy model.

Reads all dates in one parquet scan (fast), applies regime routing per date,
and writes a single CSV with columns:
  date, symbol, regime, fii_accum_prob, score_type, model_used

Run:
    python -m pipeline.fii_proxy.batch_predict 2025-03-01 2025-12-31
    python -m pipeline.fii_proxy.batch_predict 2025-03-01 2026-01-13 --out gold/fii_proxy_2025.csv
"""
from __future__ import annotations
import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

GOLD_FEATURES = Path(r"C:\Users\rahul\Koscine 3.0\gold\features.parquet")
MODEL_PATH    = Path(r"C:\Users\rahul\Koscine 3.0\models\fii_proxy_lgbm.pkl")
GOLD_DIR      = Path(r"C:\Users\rahul\Koscine 3.0\gold")


def batch_score(start: str, end: str, out_path: Path | None = None) -> pd.DataFrame:
    print(f"[fii_proxy.batch_predict] Scoring {start} to {end} ...")

    with open(MODEL_PATH, "rb") as f:
        artifact = pickle.load(f)

    features        = artifact["features"]
    trained_symbols = set(artifact.get("trained_symbols", []))
    regime_models   = artifact.get("regime_models", {})

    # Load all features for the date range in one pass
    df = pd.read_parquet(
        GOLD_FEATURES,
        filters=[("date", ">=", pd.Timestamp(start)),
                 ("date", "<=", pd.Timestamp(end))],
    )
    df["date"] = pd.to_datetime(df["date"])

    if df.empty:
        raise ValueError(f"No feature data between {start} and {end}.")

    print(f"  Loaded {len(df):,} rows  |  {df['date'].nunique()} dates  "
          f"|  {df['symbol'].nunique()} symbols")

    X_all = df.reindex(columns=features)

    # Score per regime in vectorised batches
    prob = np.full(len(df), np.nan)
    model_used_col = pd.array([""] * len(df), dtype=object)

    for regime in df["regime"].unique():
        mask = (df["regime"] == regime).values
        rm = regime_models.get(regime)
        if rm is not None:
            m, cal = rm["model"], rm["calibrator"]
            label = f"{regime}-specific"
        else:
            m, cal = artifact["model"], artifact["calibrator"]
            label = "fallback(all)"
        raw = m.predict_proba(X_all[mask])[:, 1]
        prob[mask] = cal.predict(raw)
        model_used_col[mask] = label

    result = df[["date", "symbol"]].copy()
    result["regime"]        = df["regime"].values if "regime" in df.columns else "unknown"
    result["fii_accum_prob"] = prob
    result["score_type"]    = result["symbol"].apply(
        lambda s: "trained" if s in trained_symbols else "extrap"
    )
    result["model_used"]    = model_used_col

    result = result.sort_values(["date", "fii_accum_prob"], ascending=[True, False])
    result = result.reset_index(drop=True)

    # Summary table: top-5 per date
    print(f"\n  Top-5 stocks per date (sample — last 10 dates):\n")
    for dt in sorted(result["date"].unique())[-10:]:
        day = result[result["date"] == dt].head(5)
        regime_val = day["regime"].iloc[0]
        print(f"  {dt.date()}  [{regime_val}]")
        for _, r in day.iterrows():
            bar = "#" * int(r["fii_accum_prob"] * 20)
            print(f"    {r['symbol']:<15} {r['fii_accum_prob']:.3f}  {r['score_type']:<7}  {bar}")

    if out_path is None:
        out_path = GOLD_DIR / f"fii_proxy_{start}_to_{end}.csv"
    result.to_csv(out_path, index=False)
    print(f"\n  Saved {len(result):,} rows -> {out_path}")

    # Monthly summary
    result["month"] = result["date"].dt.to_period("M").astype(str)
    summary = (
        result.groupby("month")["fii_accum_prob"]
        .agg(mean="mean", p75=lambda x: x.quantile(0.75), p90=lambda x: x.quantile(0.90))
        .round(3)
    )
    print(f"\n  Monthly probability summary:")
    print(summary.to_string())

    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("start", help="Start date YYYY-MM-DD")
    parser.add_argument("end",   help="End date YYYY-MM-DD")
    parser.add_argument("--out", default=None, help="Output CSV path")
    args = parser.parse_args()
    out = Path(args.out) if args.out else None
    batch_score(args.start, args.end, out)


if __name__ == "__main__":
    main()
