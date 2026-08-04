from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import average_precision_score, roc_auc_score

from koscine.config import HORIZON_DAYS, MODEL_DIR, PREDICTIONS_DIR, REPORTS_DIR, TARGET_UNIVERSE
from koscine.training import feature_columns


def _catboost_params(y: pd.Series, seed: int = 42) -> dict:
    positives = int(y.sum())
    negatives = int(len(y) - positives)
    pos_weight = negatives / max(positives, 1)
    return {
        "loss_function": "Logloss",
        "eval_metric": "PRAUC",
        "iterations": 1500,
        "learning_rate": 0.03,
        "depth": 6,
        "l2_leaf_reg": 10,
        "random_strength": 1.0,
        "bagging_temperature": 1.0,
        "class_weights": [1.0, pos_weight],
        "od_type": "Iter",
        "od_wait": 100,
        "allow_writing_files": False,
        "random_seed": seed,
        "verbose": 100,
    }


def catboost_model_name(side: str, threshold: float, train_end_year: int) -> str:
    pct = int(round(threshold * 100))
    return f"catboost_{side}_{pct}pct_{HORIZON_DAYS}d_train_{train_end_year}"


def train_catboost_binary_model(
    dataset_path: Path,
    label_col: str,
    train_end_year: int = 2024,
    train_start_year: int | None = 2012,
    validation_days: int = 365,
) -> dict:
    df = pd.read_parquet(dataset_path)
    dates = pd.to_datetime(df["date"])
    train = df[dates.dt.year <= train_end_year].copy()
    if train_start_year is not None:
        train = train[train["date"].dt.year >= train_start_year].copy()
    train = train.dropna(subset=[label_col])

    valid_cutoff = train["date"].max() - pd.Timedelta(days=validation_days)
    inner_train = train[train["date"] < valid_cutoff].copy()
    valid = train[train["date"] >= valid_cutoff].copy()
    features = feature_columns(df)

    model = CatBoostClassifier(**_catboost_params(inner_train[label_col]))
    model.fit(
        Pool(inner_train[features], label=inner_train[label_col]),
        eval_set=Pool(valid[features], label=valid[label_col]),
        use_best_model=True,
    )

    valid_pred = model.predict_proba(valid[features])[:, 1]
    y = valid[label_col].astype(int)
    metadata = {
        "name": f"catboost_{label_col.removeprefix('label_')}_train_{train_end_year}",
        "label": label_col,
        "train_start_year": train_start_year,
        "train_end_year": train_end_year,
        "train_rows": len(train),
        "feature_count": len(features),
        "best_iteration": model.get_best_iteration(),
        "average_precision": average_precision_score(y, valid_pred) if y.nunique() > 1 else np.nan,
        "roc_auc": roc_auc_score(y, valid_pred) if y.nunique() > 1 else np.nan,
        "features": features,
        "model_file": f"catboost_{label_col.removeprefix('label_')}_train_{train_end_year}.cbm",
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(str(MODEL_DIR / metadata["model_file"]))
    (MODEL_DIR / f"{metadata['name']}.json").write_text(
        json.dumps(metadata, indent=2, default=str),
        encoding="utf-8",
    )
    return metadata


def train_catboost_bucket_models(
    dataset_path: Path,
    thresholds: tuple[float, ...] = (0.05, 0.07),
    train_end_year: int = 2024,
    train_start_year: int | None = 2012,
) -> pd.DataFrame:
    rows = []
    for threshold in thresholds:
        pct = int(round(threshold * 100))
        for side in ("up", "down"):
            label_col = f"label_{side}_{pct}pct_{HORIZON_DAYS}d"
            rows.append(
                train_catboost_binary_model(
                    dataset_path=dataset_path,
                    label_col=label_col,
                    train_end_year=train_end_year,
                    train_start_year=train_start_year,
                )
            )
    summary = pd.DataFrame(rows)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(REPORTS_DIR / f"catboost_training_summary_{train_end_year}.csv", index=False)
    return summary


def _load_catboost(name: str) -> tuple[CatBoostClassifier, dict]:
    metadata = json.loads((MODEL_DIR / f"{name}.json").read_text(encoding="utf-8"))
    model = CatBoostClassifier()
    model.load_model(str(MODEL_DIR / metadata["model_file"]))
    return model, metadata


def predict_catboost_buckets_year(
    dataset_path: Path,
    test_year: int,
    train_end_year: int = 2024,
    thresholds: tuple[float, ...] = (0.05, 0.07),
    output_dir: Path = PREDICTIONS_DIR,
) -> pd.DataFrame:
    df = pd.read_parquet(dataset_path)
    frames = []
    for threshold in thresholds:
        pct = int(round(threshold * 100))
        for side in ("up", "down"):
            name = catboost_model_name(side, threshold, train_end_year)
            model, metadata = _load_catboost(name)
            features = metadata["features"]
            label_col = f"label_{side}_{pct}pct_{HORIZON_DAYS}d"
            pred_col = f"pred_catboost_{side}_{pct}pct_{HORIZON_DAYS}d"
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
            mask = df["date"].dt.year.eq(test_year) & df["symbol"].isin(TARGET_UNIVERSE)
            scored = df.loc[mask, list(dict.fromkeys(out_cols + features))].dropna(
                subset=[label_col]
            )
            scored = scored.copy()
            scored[pred_col] = model.predict_proba(scored[features])[:, 1]
            scored["side"] = side
            scored["threshold"] = threshold
            scored["label_col"] = label_col
            scored["pred_col"] = pred_col
            frames.append(scored[out_cols + [pred_col, "side", "threshold", "label_col", "pred_col"]])
    predictions = pd.concat(frames, ignore_index=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(output_dir / f"catboost_bucket_predictions_{test_year}.parquet", index=False)
    return predictions


def ensemble_predictions(
    lgbm_predictions: pd.DataFrame,
    catboost_predictions: pd.DataFrame,
    lgbm_weight: float = 0.55,
    output_dir: Path = PREDICTIONS_DIR,
    test_year: int = 2025,
) -> pd.DataFrame:
    base_cols = [
        "date",
        "symbol",
        "side",
        "threshold",
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
        "label_col",
    ]
    lgbm = lgbm_predictions.copy()
    lgbm["lgbm_score"] = lgbm.apply(lambda row: row[row["pred_col"]], axis=1)
    lgbm["actual"] = lgbm.apply(lambda row: row[row["label_col"]], axis=1)
    cat = catboost_predictions.copy()
    cat["catboost_score"] = cat.apply(lambda row: row[row["pred_col"]], axis=1)
    merged = lgbm[base_cols + ["lgbm_score", "actual"]].merge(
        cat[["date", "symbol", "side", "threshold", "catboost_score"]],
        on=["date", "symbol", "side", "threshold"],
        how="inner",
    )
    merged["pred_ensemble"] = (
        lgbm_weight * merged["lgbm_score"] + (1.0 - lgbm_weight) * merged["catboost_score"]
    )
    merged["pred_col"] = "pred_ensemble"
    output_dir.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(output_dir / f"ensemble_bucket_predictions_{test_year}.parquet", index=False)
    return merged
