from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ModelConfig:
    learning_rate: float = 0.035
    num_leaves: int = 47
    min_data_in_leaf: int = 180
    feature_fraction: float = 0.78
    bagging_fraction: float = 0.84
    bagging_freq: int = 1
    lambda_l1: float = 0.5
    lambda_l2: float = 12.0
    num_boost_round: int = 360
    early_stopping_rounds: int = 45
    num_threads: int = 0
    seed: int = 76031


@dataclass
class SwingModelBundle:
    feature_names: list[str]
    feature_medians: dict[str, float]
    config: ModelConfig
    models: dict[str, lgb.Booster | None]
    constants: dict[str, float]


def _prepare_x(frame: pd.DataFrame, features: list[str], medians: dict[str, float]) -> pd.DataFrame:
    x = frame[features].replace([np.inf, -np.inf], np.nan).copy()
    return x.fillna(medians)


def _medians(train: pd.DataFrame, features: list[str]) -> dict[str, float]:
    values = train[features].replace([np.inf, -np.inf], np.nan).median(numeric_only=True)
    return {col: float(values.get(col, 0.0)) if np.isfinite(values.get(col, 0.0)) else 0.0 for col in features}


def _binary_params(y: pd.Series, config: ModelConfig, seed_offset: int) -> dict:
    positives = int(y.sum())
    negatives = int(len(y) - positives)
    return {
        "objective": "binary",
        "metric": "average_precision",
        "boosting_type": "gbdt",
        "learning_rate": config.learning_rate,
        "num_leaves": config.num_leaves,
        "min_data_in_leaf": config.min_data_in_leaf,
        "feature_fraction": config.feature_fraction,
        "bagging_fraction": config.bagging_fraction,
        "bagging_freq": config.bagging_freq,
        "lambda_l1": config.lambda_l1,
        "lambda_l2": config.lambda_l2,
        "scale_pos_weight": negatives / max(positives, 1),
        "verbosity": -1,
        "num_threads": config.num_threads,
        "seed": config.seed + seed_offset,
    }


def _regression_params(config: ModelConfig, seed_offset: int) -> dict:
    return {
        "objective": "huber",
        "metric": "l2",
        "boosting_type": "gbdt",
        "learning_rate": config.learning_rate,
        "num_leaves": config.num_leaves,
        "min_data_in_leaf": config.min_data_in_leaf,
        "feature_fraction": config.feature_fraction,
        "bagging_fraction": config.bagging_fraction,
        "bagging_freq": config.bagging_freq,
        "lambda_l1": config.lambda_l1,
        "lambda_l2": config.lambda_l2,
        "verbosity": -1,
        "num_threads": config.num_threads,
        "seed": config.seed + seed_offset,
    }


def _fit_binary(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    features: list[str],
    medians: dict[str, float],
    target: str,
    config: ModelConfig,
    seed_offset: int,
) -> tuple[lgb.Booster | None, float]:
    y_train = train[target].astype(int)
    y_valid = valid[target].astype(int)
    constant = float(y_train.mean()) if len(y_train) else 0.0
    if y_train.nunique() < 2 or y_valid.nunique() < 2:
        return None, constant
    train_set = lgb.Dataset(_prepare_x(train, features, medians), label=y_train)
    valid_set = lgb.Dataset(_prepare_x(valid, features, medians), label=y_valid, reference=train_set)
    model = lgb.train(
        _binary_params(y_train, config, seed_offset),
        train_set,
        valid_sets=[valid_set],
        num_boost_round=config.num_boost_round,
        callbacks=[lgb.early_stopping(config.early_stopping_rounds), lgb.log_evaluation(0)],
    )
    return model, constant


def _fit_regression(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    features: list[str],
    medians: dict[str, float],
    target: str,
    config: ModelConfig,
    seed_offset: int,
) -> tuple[lgb.Booster | None, float]:
    y_train = pd.to_numeric(train[target], errors="coerce").fillna(0.0)
    y_valid = pd.to_numeric(valid[target], errors="coerce").fillna(0.0)
    constant = float(y_train.mean()) if len(y_train) else 0.0
    if y_train.nunique() < 3:
        return None, constant
    train_set = lgb.Dataset(_prepare_x(train, features, medians), label=y_train)
    valid_set = lgb.Dataset(_prepare_x(valid, features, medians), label=y_valid, reference=train_set)
    model = lgb.train(
        _regression_params(config, seed_offset),
        train_set,
        valid_sets=[valid_set],
        num_boost_round=config.num_boost_round,
        callbacks=[lgb.early_stopping(config.early_stopping_rounds), lgb.log_evaluation(0)],
    )
    return model, constant


def train_swing_models(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    features: list[str],
    config: ModelConfig | None = None,
) -> SwingModelBundle:
    config = config or ModelConfig()
    medians = _medians(train, features)
    models: dict[str, lgb.Booster | None] = {}
    constants: dict[str, float] = {}
    for idx, target in enumerate(["hit_near_target", "opposite_target", "top5_target"], start=1):
        model, constant = _fit_binary(train, valid, features, medians, target, config, idx * 101)
        models[target] = model
        constants[target] = constant
    model, constant = _fit_regression(train, valid, features, medians, "utility_target", config, 509)
    models["utility_target"] = model
    constants["utility_target"] = constant
    return SwingModelBundle(features, medians, config, models, constants)


def _predict_head(bundle: SwingModelBundle, frame: pd.DataFrame, head: str) -> np.ndarray:
    model = bundle.models.get(head)
    if model is None:
        return np.full(len(frame), float(bundle.constants.get(head, 0.0)))
    return model.predict(_prepare_x(frame, bundle.feature_names, bundle.feature_medians), num_iteration=model.best_iteration)


def score_swing_candidates(frame: pd.DataFrame, bundle: SwingModelBundle) -> pd.DataFrame:
    out = frame.copy()
    out["p_hit_near"] = np.clip(_predict_head(bundle, out, "hit_near_target"), 0.0, 1.0)
    out["p_opposite"] = np.clip(_predict_head(bundle, out, "opposite_target"), 0.0, 1.0)
    out["p_top5"] = np.clip(_predict_head(bundle, out, "top5_target"), 0.0, 1.0)
    out["pred_utility"] = _predict_head(bundle, out, "utility_target")
    threshold = pd.to_numeric(out["threshold"], errors="coerce").replace(0, np.nan)
    out["edge_score"] = (
        0.90 * out["p_hit_near"]
        + 0.34 * out["p_top5"]
        + 0.26 * out["pred_utility"].clip(lower=-1.0, upper=2.0)
        - 1.18 * out["p_opposite"]
        + 0.06 * pd.to_numeric(out.get("threshold_pct", threshold), errors="coerce").fillna(threshold)
    )
    out["edge_score"] = out["edge_score"].replace([np.inf, -np.inf], np.nan)
    return out


def save_bundle(bundle: SwingModelBundle, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "config": asdict(bundle.config),
        "feature_names": bundle.feature_names,
        "feature_medians": bundle.feature_medians,
        "constants": bundle.constants,
        "heads": {},
    }
    for head, model in bundle.models.items():
        if model is None:
            manifest["heads"][head] = {"type": "constant", "value": bundle.constants.get(head, 0.0)}
            continue
        filename = f"{head}.lgbm.txt"
        model.save_model(str(output_dir / filename))
        manifest["heads"][head] = {"type": "lightgbm", "file": filename, "best_iteration": model.best_iteration}
    (output_dir / "model_manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
