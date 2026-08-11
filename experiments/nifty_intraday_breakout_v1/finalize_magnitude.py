"""Fits ONE production magnitude-regression model on the FULL available price-only history --
not per-fold (train_magnitude.py's walk-forward is for validation; this is the "ship it" step,
run only after train_magnitude.py showed real out-of-sample skill: Spearman IC 0.53, top-decile
3.1x bottom-decile, at h=6/30-min on the full 2022-2026 history).

h=6 and price-only features (not the option-chain-augmented v2) by design: train_magnitude.py's
matched-window comparison showed option-chain features add ~nothing over price-only at every
horizon (IC 0.4756 vs 0.4784 at h6) while requiring live option-chain data the poller would
otherwise not need -- shipping price-only is a real simplification, not a corner cut.

Unlike finalize.py (which only ever persisted a calibrated threshold K, because the classifier's
walk-forward `p` was only consumed for offline backtesting), this DOES persist a real, loadable
model -- live inference needs to score brand-new bars, not just replay a precomputed CSV column.
Saved via LightGBM's native format (not pickle) so it can be loaded without unpickling arbitrary
code: locks/prod_nifty_intraday/magnitude_model.txt + a companion magnitude_config.json
recording the feature list/order and training metadata a live scorer needs.

Usage:
    python experiments/nifty_intraday_breakout_v1/finalize_magnitude.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(r"C:\Users\rahul\Koscine 3.0")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_magnitude as tm  # noqa: E402

HORIZON = 6   # EDIT only if a later train_magnitude.py run shows a different horizon winning

DATA = Path(__file__).resolve().parent / "data" / "v1_price_only.parquet"
OUT_DIR = ROOT / "locks" / "prod_nifty_intraday"


def main() -> None:
    import lightgbm as lgb

    if not DATA.exists():
        raise SystemExit(f"{DATA} not found -- run dataset.py first")
    df = pd.read_parquet(DATA)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    feats = tm.feature_columns(df)
    target_col = f"fwd_move_mag_{HORIZON}"
    d = df.dropna(subset=[target_col] + feats, how="any")

    model = lgb.LGBMRegressor(**tm.LGBM_PARAMS)
    model.fit(d[feats], d[target_col])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(str(OUT_DIR / "magnitude_model.txt"))
    config = {
        "horizon_bars": HORIZON,
        "features": feats,
        "trained_on": {
            "n_rows": int(len(d)),
            "date_min": str(d["timestamp"].min()),
            "date_max": str(d["timestamp"].max()),
        },
        "validation_summary": {   # from train_magnitude.py's walk-forward, not this full-data fit
            "spearman_ic": 0.5323, "mae": 0.00074, "top_decile_vs_bottom_decile": 3.10,
        },
    }
    (OUT_DIR / "magnitude_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"[finalize_magnitude] horizon={HORIZON}bar, {len(feats)} features, {len(d)} rows "
          f"-> {OUT_DIR / 'magnitude_model.txt'} + magnitude_config.json")


if __name__ == "__main__":
    main()
