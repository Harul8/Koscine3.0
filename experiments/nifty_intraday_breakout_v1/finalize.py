"""Computes ONE production breakout threshold K from the FULL available price-only
history -- not per-fold (that's what train.py's walk-forward is for; this is the "ship it"
step, meant to be run only after train.py's results show real out-of-sample lift for the
chosen horizon).

Writes locks/prod_nifty_intraday/config.json: {horizon_bars, target_rate, k, calibrated_on}.

EDIT HORIZON below once train.py's per-horizon results are in -- pick whichever horizon
(6/12/24 bars) had the best out-of-sample lift_over_base_rate, not necessarily the 12-bar
default this file ships with. This script does not pick it for you.

Usage:
    python experiments/nifty_intraday_breakout_v1/finalize.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(r"C:\Users\rahul\Koscine 3.0")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import labels  # noqa: E402

HORIZON = 12   # EDIT once train.py's results are in
TARGET_RATE = labels.TARGET_RATE

DATA = Path(__file__).resolve().parent / "data" / "v1_price_only.parquet"
OUT_DIR = ROOT / "locks" / "prod_nifty_intraday"


def main() -> None:
    if not DATA.exists():
        raise SystemExit(f"{DATA} not found -- run dataset.py first")
    df = pd.read_parquet(DATA)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    k = labels.calibrate_k(df, HORIZON, target_rate=TARGET_RATE)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    config = {
        "horizon_bars": HORIZON,
        "target_rate": TARGET_RATE,
        "k": k,
        "calibrated_on": {
            "n_rows": int(len(df)),
            "date_min": str(df["timestamp"].min()),
            "date_max": str(df["timestamp"].max()),
        },
    }
    (OUT_DIR / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"[finalize] horizon={HORIZON}bar target_rate={TARGET_RATE} k={k:.3f} "
          f"-> {OUT_DIR / 'config.json'}")


if __name__ == "__main__":
    main()
