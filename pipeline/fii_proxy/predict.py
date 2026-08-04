"""
Score today's stocks with the FII accumulation proxy model.

Reads features from gold/features.parquet for the given date, detects
the market regime, routes to the matching regime-specific LightGBM model
(bull / bear / range), and outputs per-stock fii_accum_prob (0-1).

Regime routing:
  bull  -> bull model
  bear  -> bear model
  range -> range model
  unknown / missing regime model -> fallback model (all-regime)

Scores are marked 'trained' for symbols the model saw FII ground truth
for, and 'extrap' for others (model is generalising; treat with lower
confidence).

Run:
    python -m pipeline.fii_proxy.predict 2026-05-18
    python -m pipeline.fii_proxy.predict 2026-05-18 --save
    python -m pipeline.fii_proxy.predict 2026-05-18 --trained-only
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


def score(date_str: str, save: bool = False,
          trained_only: bool = False) -> pd.DataFrame:
    target_date = pd.Timestamp(date_str)

    with open(MODEL_PATH, "rb") as f:
        artifact = pickle.load(f)

    features        = artifact["features"]
    trained_symbols = set(artifact.get("trained_symbols", []))
    regime_models   = artifact.get("regime_models", {})

    day = pd.read_parquet(
        GOLD_FEATURES,
        filters=[("date", "=", target_date)],
    )
    if day.empty:
        raise ValueError(f"No feature data for {date_str}. Run pipeline.features first.")

    day["date"] = pd.to_datetime(day["date"])

    if trained_only and trained_symbols:
        day = day[day["symbol"].isin(trained_symbols)].copy()

    # ── Regime routing ────────────────────────────────────────────────────────
    regime = day["regime"].iloc[0] if "regime" in day.columns else None
    regime_entry = regime_models.get(regime) if regime else None

    if regime_entry is not None:
        model      = regime_entry["model"]
        calibrator = regime_entry["calibrator"]
        model_used = f"{regime}-specific"
    else:
        model      = artifact["model"]
        calibrator = artifact["calibrator"]
        model_used = "fallback(all)" if regime_entry is None else "fallback(no-regime)"

    X = day.reindex(columns=features)
    raw_prob = model.predict_proba(X)[:, 1]
    cal_prob = calibrator.predict(raw_prob)

    result = day[["date", "symbol"]].copy()
    result["fii_accum_prob"] = cal_prob
    result["regime"]     = regime or "unknown"
    result["model_used"] = model_used
    result["score_type"] = result["symbol"].apply(
        lambda s: "trained" if s in trained_symbols else "extrap"
    )
    result = result.sort_values("fii_accum_prob", ascending=False).reset_index(drop=True)

    val_auc = artifact.get("val_auc", {})
    auc_str = val_auc.get(regime, val_auc) if isinstance(val_auc, dict) else val_auc

    print(f"\n[fii_proxy] FII Accumulation Probability -- {date_str}")
    print(f"  Regime: {regime or 'unknown'}  |  Model: {model_used}  |  Val AUC: {auc_str}")
    print(f"{'Symbol':<15} {'Prob':>6}  {'Type':>7}  {'Bar'}")
    print("-" * 50)
    for _, row in result.iterrows():
        bar = "#" * int(row["fii_accum_prob"] * 20)
        print(f"{row['symbol']:<15} {row['fii_accum_prob']:>6.3f}  "
              f"{row['score_type']:>7}  {bar}")

    if save:
        out = GOLD_DIR / f"fii_proxy_{date_str}.csv"
        result.to_csv(out, index=False)
        print(f"\nSaved -> {out}")

    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("date", help="Prediction date YYYY-MM-DD")
    parser.add_argument("--save", action="store_true",
                        help="Save output CSV to gold/")
    parser.add_argument("--trained-only", action="store_true",
                        help="Only score symbols with FII ground truth")
    args = parser.parse_args()
    score(args.date, save=args.save, trained_only=args.trained_only)


if __name__ == "__main__":
    main()
