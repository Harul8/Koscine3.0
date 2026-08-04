from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from koscine.config import HORIZON_DAYS, RUNS_DIR, TARGET_UNIVERSE
from koscine.training import feature_columns


QUARTERS_2025 = {
    "2025Q1": ("2025-01-01", "2025-03-31"),
    "2025Q2": ("2025-04-01", "2025-06-30"),
    "2025Q3": ("2025-07-01", "2025-09-30"),
    "2025Q4": ("2025-10-01", "2025-12-31"),
}


@dataclass(frozen=True)
class ModelSpec:
    name: str
    feature_profile: str
    params: dict


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def make_run_dir(run_name: str | None = None) -> Path:
    run_id = run_name or f"quarter_validation_{timestamp()}"
    run_dir = RUNS_DIR / run_id
    (run_dir / "models").mkdir(parents=True, exist_ok=True)
    (run_dir / "predictions").mkdir(parents=True, exist_ok=True)
    (run_dir / "reports").mkdir(parents=True, exist_ok=True)
    return run_dir


def _contains_any(col: str, terms: tuple[str, ...]) -> bool:
    return any(term in col for term in terms)


def select_features(df: pd.DataFrame, side: str, profile: str) -> list[str]:
    all_features = feature_columns(df)
    if profile == "all":
        return all_features

    common_terms = (
        "volume",
        "turnover",
        "delivery",
        "atr",
        "range",
        "bb_width",
        "donchian",
        "realized_vol",
        "nifty",
        "cs_rank",
        "day_of_week",
        "month",
    )
    up_terms = (
        "ret_",
        "close_sma",
        "vol_sma",
        "rel_ret",
        "fut_oi",
        "fut_vol",
        "pcr",
        "call_wall",
        "put_wall",
        "max_pain",
        "atm_iv",
    )
    down_terms = (
        "ret_",
        "atr",
        "range",
        "bb_width",
        "realized_vol",
        "pcr",
        "put_call",
        "put_wall",
        "call_wall",
        "max_pain",
        "atm_iv",
        "fut_chg_oi",
        "fut_oi",
        "fut_vol",
    )
    strict_terms = common_terms + (up_terms if side == "up" else down_terms)

    selected = [col for col in all_features if _contains_any(col, strict_terms)]
    if profile == "side_curated":
        return selected

    if profile == "side_compact":
        compact_exclude = ("open", "high", "low", "last", "prev_close", "fut_settle")
        selected = [col for col in selected if col not in compact_exclude]
        selected = [col for col in selected if not col.endswith("_2") and not col.endswith("_3")]
        return selected

    raise ValueError(f"Unknown feature profile: {profile}")


def specs_for_side(side: str) -> list[ModelSpec]:
    base = {
        "objective": "binary",
        "metric": "average_precision",
        "boosting_type": "gbdt",
        "learning_rate": 0.03,
        "num_leaves": 31,
        "min_data_in_leaf": 300,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l1": 1.0,
        "lambda_l2": 8.0,
        "min_gain_to_split": 0.01,
        "verbosity": -1,
        "seed": 42,
    }
    conservative = {
        **base,
        "learning_rate": 0.02,
        "num_leaves": 31,
        "min_data_in_leaf": 800,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.75,
        "lambda_l1": 3.0,
        "lambda_l2": 20.0,
    }
    expressive = {
        **base,
        "learning_rate": 0.025,
        "num_leaves": 63,
        "min_data_in_leaf": 400,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.85,
        "lambda_l1": 0.5,
        "lambda_l2": 8.0,
    }
    if side == "down":
        conservative = {**conservative, "min_data_in_leaf": 1200, "lambda_l2": 30.0}
        expressive = {**expressive, "num_leaves": 31, "lambda_l2": 15.0}
    return [
        ModelSpec("all_regularized", "all", base),
        ModelSpec("side_curated_conservative", "side_curated", conservative),
        ModelSpec("side_compact_expressive", "side_compact", expressive),
    ]


def _prepare_split(
    df: pd.DataFrame,
    train_start_year: int,
    train_end_year: int,
    label_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = df[
        df["date"].dt.year.ge(train_start_year) & df["date"].dt.year.le(train_end_year)
    ].copy()
    train = train.dropna(subset=[label_col])
    valid_cutoff = train["date"].max() - pd.Timedelta(days=365)
    inner = train[train["date"] < valid_cutoff].copy()
    valid = train[train["date"] >= valid_cutoff].copy()
    return inner, valid


def _params_with_weight(params: dict, y: pd.Series) -> dict:
    positives = int(y.sum())
    negatives = int(len(y) - positives)
    return {**params, "scale_pos_weight": negatives / max(positives, 1)}


def train_one_model(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    features: list[str],
    label_col: str,
    params: dict,
) -> lgb.Booster:
    train_set = lgb.Dataset(train[features], label=train[label_col])
    valid_set = lgb.Dataset(valid[features], label=valid[label_col], reference=train_set)
    model = lgb.train(
        _params_with_weight(params, train[label_col]),
        train_set,
        valid_sets=[valid_set],
        num_boost_round=2500,
        callbacks=[lgb.early_stopping(150), lgb.log_evaluation(0)],
    )
    return model


def score_predictions(frame: pd.DataFrame, score_col: str, label_col: str, side: str) -> dict:
    y = frame[label_col].astype(int)
    pred = frame[score_col]
    side_mult = -1.0 if side == "down" else 1.0
    metrics: dict[str, float] = {
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
            threshold = float(top["threshold"].iloc[0])
            event_proxy = np.where(
                top[label_col].eq(1),
                threshold,
                side_mult * top[f"fwd_return_{HORIZON_DAYS}d"],
            )
            metrics[f"avg_event_proxy_return_at_{k}"] = float(np.nanmean(event_proxy))
    return metrics


def score_prediction_group(frame: pd.DataFrame) -> dict:
    pred_col = str(frame["pred_col"].iloc[0])
    label_col = str(frame["label_col"].iloc[0])
    side = str(frame["side"].iloc[0])
    scored = frame.copy()
    if "actual" not in scored:
        scored["actual"] = scored.apply(lambda row: row[row["label_col"]], axis=1)
    score_col = "score"
    scored[score_col] = scored[pred_col]

    y = scored["actual"].astype(int)
    pred = scored[score_col]
    side_mult = -1.0 if side == "down" else 1.0
    metrics: dict[str, float] = {
        "rows": float(len(scored)),
        "actual_events": float(y.sum()),
        "base_rate": float(y.mean()) if len(scored) else np.nan,
        "average_precision": float(average_precision_score(y, pred)) if y.nunique() > 1 else np.nan,
        "roc_auc": float(roc_auc_score(y, pred)) if y.nunique() > 1 else np.nan,
    }
    ranked = scored.sort_values(score_col, ascending=False)
    threshold = float(scored["threshold"].iloc[0])
    for k in (10, 25, 50, 100):
        top = ranked.head(k)
        if len(top):
            metrics[f"precision_at_{k}"] = float(top["actual"].mean())
            metrics[f"avg_close_strategy_return_at_{k}"] = float(
                side_mult * top[f"fwd_return_{HORIZON_DAYS}d"].mean()
            )
            event_proxy = np.where(
                top["actual"].eq(1),
                threshold,
                side_mult * top[f"fwd_return_{HORIZON_DAYS}d"],
            )
            metrics[f"avg_event_proxy_return_at_{k}"] = float(np.nanmean(event_proxy))
    return metrics


def _copy_model_artifacts(model_family: str, run_dir: Path) -> None:
    source_dir = Path("models")
    target_dir = run_dir / "models"
    target_dir.mkdir(parents=True, exist_ok=True)
    if not source_dir.exists():
        return
    patterns = []
    if model_family == "lgbm":
        patterns = ["lgbm_*"]
    elif model_family == "catboost":
        patterns = ["catboost_*"]
    elif model_family == "ensemble":
        patterns = ["lgbm_*", "catboost_*"]
    for pattern in patterns:
        for path in source_dir.glob(pattern):
            if path.is_file():
                shutil.copy2(path, target_dir / path.name)


def evaluate_predictions_by_quarter(
    predictions_path: Path,
    model_family: str,
    run_name: str | None = None,
    year: int = 2025,
) -> Path:
    run_dir = make_run_dir(run_name or f"{model_family}_quarter_eval_{timestamp()}")
    _copy_model_artifacts(model_family, run_dir)
    run_stamp = run_dir.name
    predictions = pd.read_parquet(predictions_path)
    predictions["date"] = pd.to_datetime(predictions["date"])

    metrics_rows = []
    for quarter, (start, end) in QUARTERS_2025.items():
        if not quarter.startswith(str(year)):
            continue
        mask = predictions["date"].between(pd.Timestamp(start), pd.Timestamp(end))
        quarter_predictions = predictions[mask].copy()
        pred_path = (
            run_dir
            / "predictions"
            / f"{run_stamp}_{model_family}_{quarter}_predictions.parquet"
        )
        quarter_predictions.to_parquet(pred_path, index=False)
        for (side, threshold), group in quarter_predictions.groupby(["side", "threshold"]):
            metrics = score_prediction_group(group)
            metrics.update(
                {
                    "model_family": model_family,
                    "quarter": quarter,
                    "side": side,
                    "threshold": threshold,
                    "prediction_file": pred_path.name,
                }
            )
            metrics_rows.append(metrics)

    metrics_df = pd.DataFrame(metrics_rows)
    report_path = run_dir / "reports" / f"{run_stamp}_{model_family}_quarter_metrics.csv"
    metrics_df.to_csv(report_path, index=False)
    comparison = (
        metrics_df.groupby(["side", "threshold"])
        .agg(
            avg_ap=("average_precision", "mean"),
            min_ap=("average_precision", "min"),
            avg_precision_at_25=("precision_at_25", "mean"),
            min_precision_at_25=("precision_at_25", "min"),
            avg_close_strategy_at_25=("avg_close_strategy_return_at_25", "mean"),
            avg_event_proxy_at_25=("avg_event_proxy_return_at_25", "mean"),
        )
        .reset_index()
    )
    comparison.to_csv(
        run_dir / "reports" / f"{run_stamp}_{model_family}_quarter_comparison.csv",
        index=False,
    )
    manifest = {
        "run_dir": str(run_dir),
        "predictions_path": str(predictions_path),
        "model_family": model_family,
        "year": year,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    return run_dir


def _quarter_frame(df: pd.DataFrame, quarter: str) -> pd.DataFrame:
    start, end = QUARTERS_2025[quarter]
    mask = df["date"].between(pd.Timestamp(start), pd.Timestamp(end))
    mask &= df["symbol"].isin(TARGET_UNIVERSE)
    return df[mask].copy()


def run_quarter_experiment(
    dataset_path: Path,
    train_start_year: int = 2012,
    train_end_year: int = 2024,
    run_name: str | None = None,
    thresholds: tuple[float, ...] = (0.05, 0.07),
) -> Path:
    run_dir = make_run_dir(run_name)
    df = pd.read_parquet(dataset_path)
    df["date"] = pd.to_datetime(df["date"])

    all_metrics = []
    feature_rows = []
    manifest = {
        "run_dir": str(run_dir),
        "dataset_path": str(dataset_path),
        "train_start_year": train_start_year,
        "train_end_year": train_end_year,
        "thresholds": thresholds,
        "quarters": QUARTERS_2025,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))

    run_stamp = run_dir.name
    for threshold in thresholds:
        pct = int(round(threshold * 100))
        for side in ("up", "down"):
            label_col = f"label_{side}_{pct}pct_{HORIZON_DAYS}d"
            inner, valid = _prepare_split(df, train_start_year, train_end_year, label_col)
            for spec in specs_for_side(side):
                features = select_features(df, side, spec.feature_profile)
                model = train_one_model(inner, valid, features, label_col, spec.params)
                model_id = f"{run_stamp}_{side}_{pct}pct_{spec.name}"
                model_path = run_dir / "models" / f"{model_id}.txt"
                model.save_model(model_path)
                metadata = {
                    "model_id": model_id,
                    "side": side,
                    "threshold": threshold,
                    "label_col": label_col,
                    "spec": spec.name,
                    "feature_profile": spec.feature_profile,
                    "features": features,
                    "params": spec.params,
                    "best_iteration": model.best_iteration,
                    "model_file": model_path.name,
                }
                (run_dir / "models" / f"{model_id}.json").write_text(
                    json.dumps(metadata, indent=2, default=str)
                )
                importance = model.feature_importance(importance_type="gain")
                for feature, gain in zip(features, importance, strict=False):
                    feature_rows.append(
                        {
                            "model_id": model_id,
                            "side": side,
                            "threshold": threshold,
                            "feature": feature,
                            "gain": gain,
                        }
                    )

                for quarter in QUARTERS_2025:
                    quarter_df = _quarter_frame(df, quarter).dropna(subset=[label_col]).copy()
                    score_col = f"score_{model_id}"
                    quarter_df[score_col] = model.predict(
                        quarter_df[features],
                        num_iteration=model.best_iteration,
                    )
                    quarter_df["side"] = side
                    quarter_df["threshold"] = threshold
                    quarter_df["label_col"] = label_col
                    quarter_df["pred_col"] = score_col
                    pred_cols = [
                        "date",
                        "symbol",
                        "close",
                        f"future_{HORIZON_DAYS}d_high",
                        f"future_{HORIZON_DAYS}d_low",
                        f"future_{HORIZON_DAYS}d_close",
                        f"future_{HORIZON_DAYS}d_date",
                        f"up_move_{HORIZON_DAYS}d",
                        f"down_move_{HORIZON_DAYS}d",
                        f"fwd_return_{HORIZON_DAYS}d",
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
                        / f"{run_stamp}_{quarter}_{side}_{pct}pct_{spec.name}_predictions.parquet"
                    )
                    quarter_df[pred_cols].to_parquet(pred_path, index=False)
                    metrics = score_predictions(quarter_df, score_col, label_col, side)
                    metrics.update(
                        {
                            "model_id": model_id,
                            "quarter": quarter,
                            "side": side,
                            "threshold": threshold,
                            "spec": spec.name,
                            "feature_profile": spec.feature_profile,
                            "feature_count": len(features),
                            "best_iteration": model.best_iteration,
                            "prediction_file": pred_path.name,
                        }
                    )
                    all_metrics.append(metrics)

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(run_dir / "reports" / f"{run_stamp}_quarter_metrics.csv", index=False)
    pd.DataFrame(feature_rows).to_csv(
        run_dir / "reports" / f"{run_stamp}_feature_importance.csv",
        index=False,
    )
    best = (
        metrics_df.groupby(["side", "threshold", "spec"])
        .agg(
            avg_ap=("average_precision", "mean"),
            avg_precision_at_25=("precision_at_25", "mean"),
            avg_event_proxy_at_25=("avg_event_proxy_return_at_25", "mean"),
            avg_close_strategy_at_25=("avg_close_strategy_return_at_25", "mean"),
        )
        .reset_index()
        .sort_values(["avg_event_proxy_at_25", "avg_precision_at_25"], ascending=False)
    )
    best.to_csv(run_dir / "reports" / f"{run_stamp}_model_comparison.csv", index=False)
    return run_dir
