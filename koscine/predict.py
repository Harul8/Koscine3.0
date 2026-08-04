from __future__ import annotations

from pathlib import Path

import pandas as pd

from koscine.config import HORIZON_DAYS, MODEL_DIR, PREDICTIONS_DIR, TARGET_UNIVERSE
from koscine.training import load_model_bundle


def model_name(side: str, threshold: float, train_end_year: int) -> str:
    pct = int(round(threshold * 100))
    return f"lgbm_{side}_{pct}pct_{HORIZON_DAYS}d_train_{train_end_year}"


def predict_bucket_year(
    dataset_path: Path,
    side: str,
    threshold: float,
    test_year: int,
    train_end_year: int = 2024,
    model_dir: Path = MODEL_DIR,
) -> pd.DataFrame:
    name = model_name(side, threshold, train_end_year)
    model, metadata = load_model_bundle(name, model_dir=model_dir)
    features = metadata["features"]
    pct = int(round(threshold * 100))
    label_col = f"label_{side}_{pct}pct_{HORIZON_DAYS}d"
    pred_col = f"pred_{side}_{pct}pct_{HORIZON_DAYS}d"

    df = pd.read_parquet(dataset_path)
    mask = df["date"].dt.year.eq(test_year) & df["symbol"].isin(TARGET_UNIVERSE)
    out_cols = [
        "date",
        "symbol",
        "close",
        "entry_1d_date",
        "entry_1d_open",
        f"future_{HORIZON_DAYS}d_high",
        f"future_{HORIZON_DAYS}d_low",
        f"future_{HORIZON_DAYS}d_close",
        f"future_{HORIZON_DAYS}d_date",
        f"up_move_{HORIZON_DAYS}d",
        f"down_move_{HORIZON_DAYS}d",
        f"fwd_return_{HORIZON_DAYS}d",
        f"long_adverse_move_{HORIZON_DAYS}d",
        f"short_adverse_move_{HORIZON_DAYS}d",
        label_col,
    ]
    read_cols = list(dict.fromkeys(out_cols + features))
    scored = df.loc[mask, read_cols].dropna(subset=[label_col]).copy()
    scored[pred_col] = model.predict(scored[features], num_iteration=model.best_iteration)
    scored["side"] = side
    scored["threshold"] = threshold
    scored["label_col"] = label_col
    scored["pred_col"] = pred_col
    return scored[out_cols + [pred_col, "side", "threshold", "label_col", "pred_col"]].sort_values(
        ["date", pred_col],
        ascending=[True, False],
    )


def predict_buckets_year(
    dataset_path: Path,
    test_year: int,
    train_end_year: int = 2024,
    thresholds: tuple[float, ...] = (0.05, 0.07),
    output_dir: Path = PREDICTIONS_DIR,
) -> pd.DataFrame:
    frames = []
    for threshold in thresholds:
        for side in ("up", "down"):
            frames.append(
                predict_bucket_year(
                    dataset_path=dataset_path,
                    side=side,
                    threshold=threshold,
                    test_year=test_year,
                    train_end_year=train_end_year,
                )
            )
    predictions = pd.concat(frames, ignore_index=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(output_dir / f"bucket_predictions_{test_year}.parquet", index=False)
    return predictions
