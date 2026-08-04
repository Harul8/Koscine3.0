from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from koscine.config import HORIZON_DAYS, PREDICTIONS_DIR, RUNS_DIR, TARGET_UNIVERSE
from koscine.experiments import select_features
from koscine.rolling import known_training_rows, month_windows


@dataclass(frozen=True)
class QualityLongConfig:
    train_start_year: int = 2012
    first_prediction_month: str = "2025-01"
    last_prediction_month: str = "2025-12"
    feature_profile: str = "side_compact"
    thresholds: tuple[float, ...] = (0.05, 0.07)
    train_cutoff_day: int = 20
    validation_days: int = 365
    adverse_limit: float = 0.02
    annual_signal_target: int = 500


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def quality_label_col(threshold: float, adverse_limit: float) -> str:
    pct = int(round(threshold * 100))
    adverse_pct = int(round(adverse_limit * 100))
    return f"label_quality_up_{pct}pct_{HORIZON_DAYS}d_adverse{adverse_pct}pct"


def ensure_quality_labels(df: pd.DataFrame, config: QualityLongConfig) -> pd.DataFrame:
    out = df.copy()
    valid = (
        out[f"up_move_{HORIZON_DAYS}d"].notna()
        & out[f"long_adverse_move_{HORIZON_DAYS}d"].notna()
    )
    for threshold in config.thresholds:
        col = quality_label_col(threshold, config.adverse_limit)
        out[col] = np.where(
            valid,
            (out[f"up_move_{HORIZON_DAYS}d"] >= threshold)
            & (out[f"long_adverse_move_{HORIZON_DAYS}d"] <= config.adverse_limit),
            np.nan,
        )
    return out


def quality_lgbm_params(y: pd.Series, seed: int = 42) -> dict:
    positives = int(y.sum())
    negatives = int(len(y) - positives)
    return {
        "objective": "binary",
        "metric": "average_precision",
        "boosting_type": "gbdt",
        "learning_rate": 0.02,
        "num_leaves": 31,
        "min_data_in_leaf": 900,
        "feature_fraction": 0.72,
        "bagging_fraction": 0.78,
        "bagging_freq": 1,
        "lambda_l1": 4.0,
        "lambda_l2": 25.0,
        "min_gain_to_split": 0.01,
        "scale_pos_weight": negatives / max(positives, 1),
        "verbosity": -1,
        "seed": seed,
    }


def fit_quality_model(
    train: pd.DataFrame,
    features: list[str],
    label_col: str,
    validation_days: int,
) -> lgb.Booster:
    valid_cutoff = train["date"].max() - pd.Timedelta(days=validation_days)
    inner = train[train["date"] < valid_cutoff].copy()
    valid = train[train["date"] >= valid_cutoff].copy()
    train_set = lgb.Dataset(inner[features], label=inner[label_col])
    valid_set = lgb.Dataset(valid[features], label=valid[label_col], reference=train_set)
    return lgb.train(
        quality_lgbm_params(inner[label_col]),
        train_set,
        valid_sets=[valid_set],
        num_boost_round=2500,
        callbacks=[lgb.early_stopping(150), lgb.log_evaluation(0)],
    )


def quality_metrics(frame: pd.DataFrame, score_col: str, label_col: str, top_n: int) -> dict:
    y = frame[label_col].astype(int)
    pred = frame[score_col]
    ranked = frame.sort_values(score_col, ascending=False)
    top = ranked.head(top_n)
    metrics = {
        "rows": float(len(frame)),
        "actual_quality_events": float(y.sum()),
        "quality_base_rate": float(y.mean()) if len(y) else np.nan,
        "average_precision": float(average_precision_score(y, pred)) if y.nunique() > 1 else np.nan,
        "roc_auc": float(roc_auc_score(y, pred)) if y.nunique() > 1 else np.nan,
    }
    for k in (50, 100, 250, 500):
        sample = ranked.head(k)
        if sample.empty:
            continue
        metrics[f"quality_precision_at_{k}"] = float(sample[label_col].mean())
        metrics[f"target_hit_at_{k}"] = float(sample["target_hit"].mean())
        metrics[f"safe_adverse_at_{k}"] = float(sample["safe_adverse"].mean())
        metrics[f"fav_ge_2pct_at_{k}"] = float(sample["fav_ge_2pct"].mean())
        metrics[f"fav_ge_4pct_at_{k}"] = float(sample["fav_ge_4pct"].mean())
        metrics[f"avg_up_move_at_{k}"] = float(sample[f"up_move_{HORIZON_DAYS}d"].mean())
        metrics[f"avg_adverse_at_{k}"] = float(sample[f"long_adverse_move_{HORIZON_DAYS}d"].mean())
    metrics["annual_target_calls"] = float(len(top))
    metrics["annual_quality_precision"] = float(top[label_col].mean()) if len(top) else np.nan
    metrics["annual_target_hit_rate"] = float(top["target_hit"].mean()) if len(top) else np.nan
    metrics["annual_safe_adverse_rate"] = float(top["safe_adverse"].mean()) if len(top) else np.nan
    metrics["annual_avg_up_move"] = float(top[f"up_move_{HORIZON_DAYS}d"].mean()) if len(top) else np.nan
    metrics["annual_avg_adverse"] = float(top[f"long_adverse_move_{HORIZON_DAYS}d"].mean()) if len(top) else np.nan
    return metrics


def run_quality_long_monthly(
    dataset_path: Path,
    config: QualityLongConfig,
    run_name: str | None = None,
) -> Path:
    run_id = run_name or f"quality_long_{timestamp()}"
    run_dir = RUNS_DIR / run_id
    (run_dir / "models").mkdir(parents=True, exist_ok=True)
    (run_dir / "predictions").mkdir(parents=True, exist_ok=True)
    (run_dir / "reports").mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(dataset_path)
    df["date"] = pd.to_datetime(df["date"])
    df[f"future_{HORIZON_DAYS}d_date"] = pd.to_datetime(df[f"future_{HORIZON_DAYS}d_date"])
    df = ensure_quality_labels(df, config)
    features = select_features(df, "up", config.feature_profile)

    manifest = {
        "run_dir": str(run_dir),
        "dataset_path": str(dataset_path),
        "config": config.__dict__,
        "features": features,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "objective": (
            f"long quality = up_move >= threshold and long_adverse_move <= "
            f"{config.adverse_limit:.2%}"
        ),
        "leakage_rule": f"training rows require future_{HORIZON_DAYS}d_date <= train_end",
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))

    metrics_rows = []
    predictions = []
    run_stamp = run_dir.name
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
        test = df[
            df["date"].between(month_start, month_end) & df["symbol"].isin(TARGET_UNIVERSE)
        ].copy()

        for threshold in config.thresholds:
            label_col = quality_label_col(threshold, config.adverse_limit)
            pct = int(round(threshold * 100))
            train = known_training_rows(df, config.train_start_year, train_end, label_col)
            model = fit_quality_model(train, features, label_col, config.validation_days)
            model_id = f"{run_stamp}_{month_tag}_quality_up_{pct}pct"
            model_path = run_dir / "models" / f"{model_id}.txt"
            model.save_model(model_path)
            metadata = {
                "model_id": model_id,
                "month": month_tag,
                "prediction_start": str(month_start.date()),
                "prediction_end": str(month_end.date()),
                "train_end": str(train_end.date()),
                "side": "up",
                "threshold": threshold,
                "adverse_limit": config.adverse_limit,
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

            scored = test.dropna(subset=[label_col]).copy()
            score_col = f"score_{model_id}"
            scored[score_col] = model.predict(scored[features], num_iteration=model.best_iteration)
            scored["side"] = "up"
            scored["threshold"] = threshold
            scored["adverse_limit"] = config.adverse_limit
            scored["label_col"] = label_col
            scored["pred_col"] = score_col
            scored["model_id"] = model_id
            scored["score"] = scored[score_col]
            scored["target_hit"] = scored[f"up_move_{HORIZON_DAYS}d"].ge(threshold)
            scored["safe_adverse"] = scored[f"long_adverse_move_{HORIZON_DAYS}d"].le(
                config.adverse_limit
            )
            scored["fav_ge_2pct"] = scored[f"up_move_{HORIZON_DAYS}d"].ge(0.02)
            scored["fav_ge_4pct"] = scored[f"up_move_{HORIZON_DAYS}d"].ge(0.04)
            scored["actual_quality"] = scored[label_col]
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
                label_col,
                "actual_quality",
                "target_hit",
                "safe_adverse",
                "fav_ge_2pct",
                "fav_ge_4pct",
                "score",
                score_col,
                "side",
                "threshold",
                "adverse_limit",
                "label_col",
                "pred_col",
                "model_id",
            ]
            predictions.append(scored[pred_cols].copy())
            metrics = quality_metrics(scored, score_col, label_col, config.annual_signal_target)
            metrics.update(
                {
                    "month": month_tag,
                    "prediction_start": str(month_start.date()),
                    "prediction_end": str(month_end.date()),
                    "train_end": str(train_end.date()),
                    "threshold": threshold,
                    "adverse_limit": config.adverse_limit,
                    "label_col": label_col,
                    "feature_profile": config.feature_profile,
                    "best_iteration": model.best_iteration,
                    "train_rows": len(train),
                    "model_file": model_path.name,
                }
            )
            metrics_rows.append(metrics)

    predictions_df = pd.concat(predictions, ignore_index=True)
    predictions_path = run_dir / "predictions" / f"{run_stamp}_all_predictions.parquet"
    predictions_df.to_parquet(predictions_path, index=False)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    stable_predictions_path = PREDICTIONS_DIR / "quality_long_monthly2025_predictions.parquet"
    predictions_df.to_parquet(stable_predictions_path, index=False)

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(run_dir / "reports" / f"{run_stamp}_monthly_metrics.csv", index=False)

    annual_rows = []
    selected_frames = []
    for threshold, group in predictions_df.groupby("threshold"):
        ranked = group.sort_values("score", ascending=False)
        selected = ranked.head(config.annual_signal_target).copy()
        selected["selection_mode"] = f"year_top_{config.annual_signal_target}"
        selected_frames.append(selected)
        annual_rows.append(
            {
                "threshold": threshold,
                "selection_mode": f"year_top_{config.annual_signal_target}",
                "calls": len(selected),
                "actual_quality_events_available": int(group["actual_quality"].sum()),
                "quality_hits": int(selected["actual_quality"].sum()),
                "quality_precision": float(selected["actual_quality"].mean()),
                "target_hit_rate": float(selected["target_hit"].mean()),
                "safe_adverse_rate": float(selected["safe_adverse"].mean()),
                "fav_ge_4pct_rate": float(selected["fav_ge_4pct"].mean()),
                "fav_ge_2pct_rate": float(selected["fav_ge_2pct"].mean()),
                "avg_up_move": float(selected[f"up_move_{HORIZON_DAYS}d"].mean()),
                "median_up_move": float(selected[f"up_move_{HORIZON_DAYS}d"].median()),
                "avg_adverse_move": float(selected[f"long_adverse_move_{HORIZON_DAYS}d"].mean()),
                "median_adverse_move": float(
                    selected[f"long_adverse_move_{HORIZON_DAYS}d"].median()
                ),
                "score_cutoff": float(selected["score"].min()),
            }
        )
    selected_df = pd.concat(selected_frames, ignore_index=True)
    selected_df.to_parquet(
        run_dir / "predictions" / f"{run_stamp}_year_top_{config.annual_signal_target}.parquet",
        index=False,
    )
    selected_df.to_csv(
        PREDICTIONS_DIR / f"quality_long_year_top_{config.annual_signal_target}.csv",
        index=False,
    )
    annual_df = pd.DataFrame(annual_rows)
    annual_df.to_csv(run_dir / "reports" / f"{run_stamp}_annual_top_quality.csv", index=False)
    annual_df.to_csv(
        Path("reports") / f"quality_long_annual_top_{config.annual_signal_target}.csv",
        index=False,
    )
    return run_dir
