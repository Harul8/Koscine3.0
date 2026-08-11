"""Does adding koscine/nifty_zones.py's support/resistance zone info as MODEL FEATURES improve
the magnitude regressor, on top of what it already sees? The finalized model (finalize_magnitude.py)
was NOT given zone features -- this tests whether it should be.

Joins nifty_zones.parquet (one row per trading DAY -- support_level/resistance_level are stable
through the day) onto every 5-min bar by date, then computes PER-BAR zone distances using that
bar's own close (not the day's forward-filled EOD distance, which zones.py computes relative to
the day's closing price -- wrong for an intraday bar mid-session).

Same walk-forward harness as train_magnitude.py (same folds, same LGBM params) run twice --
price-only baseline vs price+zone features -- so any IC delta is attributable to the zone
features alone, not to a different train/test split.

Usage:
    python experiments/nifty_intraday_breakout_v1/zone_features_ablation.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_magnitude as tm  # noqa: E402

ROOT = Path(r"C:\Users\rahul\Koscine 3.0")
DATA = Path(__file__).resolve().parent / "data" / "v1_price_only.parquet"
ZONES = ROOT / "locks" / "prod_nifty_intraday" / "nifty_zones.parquet"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
HORIZON = 6


def add_zone_features(df: pd.DataFrame) -> pd.DataFrame:
    zones = pd.read_parquet(ZONES)
    zones["date"] = pd.to_datetime(zones["date"]).dt.date
    keep = zones[["date", "support_level", "resistance_level", "support_valid_touches",
                 "resistance_valid_touches", "support_zone_strength", "resistance_zone_strength",
                 "consolidating_at_support", "consolidating_at_resistance", "zone_box_width_pct"]]
    out = df.merge(keep, on="date", how="left")
    out["feature_zone_support_dist"] = (out["close"] - out["support_level"]) / out["close"]
    out["feature_zone_resistance_dist"] = (out["resistance_level"] - out["close"]) / out["close"]
    out["feature_zone_support_touches"] = out["support_valid_touches"]
    out["feature_zone_resistance_touches"] = out["resistance_valid_touches"]
    out["feature_zone_support_strength"] = out["support_zone_strength"]
    out["feature_zone_resistance_strength"] = out["resistance_zone_strength"]
    out["feature_zone_consolidating_support"] = out["consolidating_at_support"]
    out["feature_zone_consolidating_resistance"] = out["consolidating_at_resistance"]
    out["feature_zone_box_width_pct"] = out["zone_box_width_pct"]
    return out


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(DATA)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"] = pd.to_datetime(df["date"]).dt.date if df["date"].dtype != object else df["date"]

    price_feats = tm.feature_columns(df)
    print(f"[ablation] baseline: {len(price_feats)} price-only features")
    preds_base = tm.walk_forward(df, HORIZON, price_feats)
    summ_base = tm.summarize(preds_base, "price_only")
    print(f"  {summ_base}")

    df_z = add_zone_features(df)
    zone_feats = price_feats + [c for c in df_z.columns if c.startswith("feature_zone_")]
    print(f"\n[ablation] with zones: {len(zone_feats)} features "
          f"({len(zone_feats) - len(price_feats)} zone features added)")
    preds_zone = tm.walk_forward(df_z, HORIZON, zone_feats)
    summ_zone = tm.summarize(preds_zone, "price_plus_zones")
    print(f"  {summ_zone}")

    summary = pd.DataFrame([summ_base, summ_zone])
    summary.to_csv(RESULTS_DIR / "zone_ablation_summary.csv", index=False)
    print("\n" + summary.to_string(index=False))

    delta_ic = summ_zone["spearman_ic"] - summ_base["spearman_ic"]
    verdict = ("zones measurably help" if delta_ic > 0.01
               else "zones make no real difference" if abs(delta_ic) <= 0.01
               else "zones measurably HURT")
    lines = ["# Does adding S/R zone features help the NIFTY magnitude model?", "",
             f"IC delta (with zones - price only): {delta_ic:+.4f} -- {verdict}", "",
             "```", summary.to_string(index=False), "```", ""]
    (RESULTS_DIR / "FINDINGS_zone_ablation.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[ablation] verdict: {verdict} (IC delta {delta_ic:+.4f})")


if __name__ == "__main__":
    main()
