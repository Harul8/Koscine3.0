from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from koscine.config import HORIZON_DAYS, RUNS_DIR, TARGET_UNIVERSE
from koscine.experiments import select_features


@dataclass(frozen=True)
class RollingConfig:
    train_start_year: int = 2012
    first_prediction_month: str = "2025-01"
    last_prediction_month: str = "2025-12"
    feature_profile: str = "all"
    sides: tuple[str, ...] = ("up", "down")
    thresholds: tuple[float, ...] = (0.05, 0.07)
    validation_days: int = 365
    train_cutoff_day: int = 20


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def month_windows(first_month: str, last_month: str) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    starts = pd.date_range(
        pd.Timestamp(first_month + "-01"),
        pd.Timestamp(last_month + "-01"),
        freq="MS",
    )
    return [(start, start + pd.offsets.MonthEnd(0)) for start in starts]


def lgbm_params(y: pd.Series, seed: int = 42) -> dict:
    positives = int(y.sum())
    negatives = int(len(y) - positives)
    return {
        "objective": "binary",
        "metric": "average_precision",
        "boosting_type": "gbdt",
        "learning_rate": 0.025,
        "num_leaves": 31,
        "min_data_in_leaf": 600,
        "feature_fraction": 0.75,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l1": 2.0,
        "lambda_l2": 15.0,
        "min_gain_to_split": 0.01,
        "scale_pos_weight": negatives / max(positives, 1),
        "verbosity": -1,
        "seed": seed,
    }


def known_training_rows(
    df: pd.DataFrame,
    train_start_year: int,
    train_end: pd.Timestamp,
    label_col: str,
) -> pd.DataFrame:
    future_date_col = f"future_{HORIZON_DAYS}d_date"
    mask = df["date"].dt.year.ge(train_start_year)
    mask &= df["date"].le(train_end)
    mask &= df[future_date_col].le(train_end)
    out = df[mask].dropna(subset=[label_col]).copy()
    return out


def fit_model(train: pd.DataFrame, features: list[str], label_col: str, validation_days: int) -> lgb.Booster:
    valid_cutoff = train["date"].max() - pd.Timedelta(days=validation_days)
    inner = train[train["date"] < valid_cutoff].copy()
    valid = train[train["date"] >= valid_cutoff].copy()
    train_set = lgb.Dataset(inner[features], label=inner[label_col])
    valid_set = lgb.Dataset(valid[features], label=valid[label_col], reference=train_set)
    return lgb.train(
        lgbm_params(inner[label_col]),
        train_set,
        valid_sets=[valid_set],
        num_boost_round=2500,
        callbacks=[lgb.early_stopping(150), lgb.log_evaluation(0)],
    )


def score_month(frame: pd.DataFrame, score_col: str, label_col: str, side: str) -> dict:
    y = frame[label_col].astype(int)
    pred = frame[score_col]
    side_mult = -1.0 if side == "down" else 1.0
    metrics = {
        "rows": float(len(frame)),
        "actual_events": float(y.sum()),
        "base_rate": float(y.mean()) if len(frame) else np.nan,
        "average_precision": float(average_precision_score(y, pred)) if y.nunique() > 1 else np.nan,
        "roc_auc": float(roc_auc_score(y, pred)) if y.nunique() > 1 else np.nan,
    }
    ranked = frame.sort_values(score_col, ascending=False)
    for k in (10, 25, 50, 100):
        top = ranked.head(k)
        if len(top):
            metrics[f"precision_at_{k}"] = float(top[label_col].mean())
            metrics[f"avg_close_strategy_return_at_{k}"] = float(
                side_mult * top[f"fwd_return_{HORIZON_DAYS}d"].mean()
            )
    return metrics


def run_monthly_retrain(
    dataset_path: Path,
    config: RollingConfig,
    run_name: str | None = None,
) -> Path:
    run_id = run_name or f"monthly_retrain_{timestamp()}"
    run_dir = RUNS_DIR / run_id
    (run_dir / "models").mkdir(parents=True, exist_ok=True)
    (run_dir / "predictions").mkdir(parents=True, exist_ok=True)
    (run_dir / "reports").mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(dataset_path)
    df["date"] = pd.to_datetime(df["date"])
    df[f"future_{HORIZON_DAYS}d_date"] = pd.to_datetime(df[f"future_{HORIZON_DAYS}d_date"])

    metrics_rows = []
    run_stamp = run_dir.name
    manifest = {
        "run_dir": str(run_dir),
        "dataset_path": str(dataset_path),
        "config": config.__dict__,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "leakage_rule": f"training rows require future_{HORIZON_DAYS}d_date <= train_end",
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))

    for month_start, month_end in month_windows(
        config.first_prediction_month,
        config.last_prediction_month,
    ):
        previous_month_start = month_start - pd.offsets.MonthBegin(1)
        train_end = pd.Timestamp(
            year=previous_month_start.year,
            month=previous_month_start.month,
            day=config.train_cutoff_day,
        )
        month_tag = month_start.strftime("%Y%m")
        test_mask = df["date"].between(month_start, month_end)
        test_mask &= df["symbol"].isin(TARGET_UNIVERSE)
        test_df = df[test_mask].copy()

        for threshold in config.thresholds:
            pct = int(round(threshold * 100))
            for side in config.sides:
                label_col = f"label_{side}_{pct}pct_{HORIZON_DAYS}d"
                features = select_features(df, side, config.feature_profile)
                train = known_training_rows(df, config.train_start_year, train_end, label_col)
                model = fit_model(train, features, label_col, config.validation_days)
                model_id = f"{run_stamp}_{month_tag}_{side}_{pct}pct_{config.feature_profile}"
                model_path = run_dir / "models" / f"{model_id}.txt"
                model.save_model(model_path)
                metadata = {
                    "model_id": model_id,
                    "month": month_tag,
                    "prediction_start": str(month_start.date()),
                    "prediction_end": str(month_end.date()),
                    "train_end": str(train_end.date()),
                    "side": side,
                    "threshold": threshold,
                    "label_col": label_col,
                    "feature_profile": config.feature_profile,
                    "features": features,
                    "best_iteration": model.best_iteration,
                    "train_rows": len(train),
                    "model_file": model_path.name,
                }
                (run_dir / "models" / f"{model_id}.json").write_text(
                    json.dumps(metadata, indent=2, default=str),
                    encoding="utf-8",
                )

                scored = test_df.dropna(subset=[label_col]).copy()
                score_col = f"score_{model_id}"
                scored[score_col] = model.predict(scored[features], num_iteration=model.best_iteration)
                scored["side"] = side
                scored["threshold"] = threshold
                scored["label_col"] = label_col
                scored["pred_col"] = score_col
                pred_cols = [
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
                    score_col,
                    "side",
                    "threshold",
                    "label_col",
                    "pred_col",
                ]
                pred_path = (
                    run_dir
                    / "predictions"
                    / f"{run_stamp}_{month_tag}_{side}_{pct}pct_predictions.parquet"
                )
                scored[pred_cols].to_parquet(pred_path, index=False)
                metrics = score_month(scored, score_col, label_col, side)
                metrics.update(
                    {
                        "month": month_tag,
                        "prediction_start": str(month_start.date()),
                        "prediction_end": str(month_end.date()),
                        "train_end": str(train_end.date()),
                        "side": side,
                        "threshold": threshold,
                        "label_col": label_col,
                        "feature_profile": config.feature_profile,
                        "best_iteration": model.best_iteration,
                        "train_rows": len(train),
                        "prediction_file": pred_path.name,
                        "model_file": model_path.name,
                    }
                )
                metrics_rows.append(metrics)

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_path = run_dir / "reports" / f"{run_stamp}_monthly_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    comparison = (
        metrics_df.groupby(["side", "threshold"])
        .agg(
            avg_ap=("average_precision", "mean"),
            min_ap=("average_precision", "min"),
            avg_precision_at_25=("precision_at_25", "mean"),
            min_precision_at_25=("precision_at_25", "min"),
            avg_close_strategy_at_25=("avg_close_strategy_return_at_25", "mean"),
        )
        .reset_index()
    )
    comparison.to_csv(run_dir / "reports" / f"{run_stamp}_monthly_comparison.csv", index=False)
    return run_dir
