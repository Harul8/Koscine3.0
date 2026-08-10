"""Assemble the NIFTY intraday breakout dataset(s):

  data/v1_price_only.parquet   -- full 2022-2026 history, price-only features
  data/v2_option_chain.parquet -- Oct 2024+ window, price features + option-chain features
      (v2's price-only feature_* columns are identical in construction to v1's, for that
      same window -- train.py uses this one file to build BOTH the matched-window
      price-only baseline and the price+option comparison, so the two are evaluated on
      literally the same rows.)

Usage:
    python experiments/nifty_intraday_breakout_v1/dataset.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(r"C:\Users\rahul\Koscine 3.0")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import features as price_features  # noqa: E402
import labels  # noqa: E402
import option_features  # noqa: E402

NIFTY_5M = ROOT / "data" / "intraday" / "nifty_5m.parquet"
OPTION_CHAIN_DIR = ROOT / "data" / "intraday" / "nifty_option_chain_5m"
OUT_DIR = Path(__file__).resolve().parent / "data"
OPTION_CHAIN_CUTOFF = pd.Timestamp("2024-10-01", tz="Asia/Kolkata")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    bars = pd.read_parquet(NIFTY_5M)
    bars["timestamp"] = pd.to_datetime(bars["timestamp"])
    feat = price_features.build_price_features(bars)
    feat = labels.add_forward_extremes(feat)
    feat.to_parquet(OUT_DIR / "v1_price_only.parquet", index=False)
    print(f"[dataset] v1 price-only: {len(feat):,} bars, {feat['date'].min()} -> {feat['date'].max()}")
    for h in labels.HORIZONS:
        col = f"fwd_move_mag_{h}"
        n_valid = int(feat[col].notna().sum())
        print(f"  horizon={h}bar  valid_rows={n_valid:,}  median_move_mag={feat[col].median():.4f}")

    if not OPTION_CHAIN_DIR.exists():
        print(f"[dataset] {OPTION_CHAIN_DIR} not found -- skipping v2 (option-chain) dataset")
        return

    window = feat[feat["timestamp"] >= OPTION_CHAIN_CUTOFF].copy()
    print(f"[dataset] building option-chain features (this scans all files under {OPTION_CHAIN_DIR}, "
          f"a few minutes for a Black-Scholes solve per ATM row) ...")
    opt = option_features.build_option_features(OPTION_CHAIN_DIR)
    v2 = window.merge(opt, on="timestamp", how="left")
    v2.to_parquet(OUT_DIR / "v2_option_chain.parquet", index=False)
    coverage = v2["feature_atm_iv_avg"].notna().mean() if "feature_atm_iv_avg" in v2.columns else 0.0
    print(f"[dataset] v2 option-chain window: {len(v2):,} bars, {v2['date'].min()} -> {v2['date'].max()}, "
          f"option feature coverage={coverage:.1%}")


if __name__ == "__main__":
    main()
