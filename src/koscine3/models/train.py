from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from koscine3.datasets.splits import WalkForwardSplit, between_dates


try:
    from lightgbm import LGBMClassifier, LGBMRegressor

    HAS_LIGHTGBM = True
except Exception:
    LGBMClassifier = None
    LGBMRegressor = None
    HAS_LIGHTGBM = False


@dataclass
class ConstantProbabilityModel:
    probability: float

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        p = np.full(len(x), self.probability, dtype=float)
        return np.column_stack([1.0 - p, p])


@dataclass
class ConstantRegressor:
    value: float

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        return np.full(len(x), self.value, dtype=float)


@dataclass
class SideModelBundle:
    side: str
    hit_near_model: Any
    clean_success_model: Any
    opposite_model: Any
    favorable_move_model: Any
    signed_close_model: Any
    train_rows: int
    calibration_rows: int


@dataclass
class SwingModel:
    model_id: str
    split: WalkForwardSplit
    feature_columns: list[str]
    algorithm: str
    long_bundle: SideModelBundle
    short_bundle: SideModelBundle
    params: dict[str, Any]


def _clean_x(frame: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    return frame[feature_columns].replace([np.inf, -np.inf], np.nan)


def _classifier(n_estimators: int, random_state: int) -> Any:
    if HAS_LIGHTGBM:
        return LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=0.045,
            num_leaves=31,
            subsample=0.85,
            colsample_bytree=0.85,
            class_weight="balanced",
            random_state=random_state,
            verbosity=-1,
        )
    return HistGradientBoostingClassifier(
        max_iter=max(50, n_estimators),
        learning_rate=0.045,
        random_state=random_state,
        class_weight="balanced",
    )


def _regressor(n_estimators: int, random_state: int) -> Any:
    if HAS_LIGHTGBM:
        return LGBMRegressor(
            n_estimators=n_estimators,
            learning_rate=0.045,
            num_leaves=31,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=random_state,
            verbosity=-1,
        )
    return HistGradientBoostingRegressor(
        max_iter=max(50, n_estimators),
        learning_rate=0.045,
        random_state=random_state,
    )


def _fit_prefit_calibrator(
    base_model: Any,
    x_calibration: pd.DataFrame,
    y_calibration: pd.Series,
    method: str = "sigmoid",
) -> Any:
    try:
        from sklearn.frozen import FrozenEstimator

        calibrated = CalibratedClassifierCV(FrozenEstimator(base_model), method=method)
        calibrated.fit(x_calibration, y_calibration)
        return calibrated
    except Exception:
        pass
    try:
        calibrated = CalibratedClassifierCV(estimator=base_model, method=method, cv="prefit")
    except TypeError:
        calibrated = CalibratedClassifierCV(base_estimator=base_model, method=method, cv="prefit")
    calibrated.fit(x_calibration, y_calibration)
    return calibrated


def _fit_probability_model(
    train: pd.DataFrame,
    calibration: pd.DataFrame,
    feature_columns: list[str],
    target: str,
    n_estimators: int,
    random_state: int,
) -> Any:
    y_train = train[target].astype(int)
    y_cal = calibration[target].astype(int)
    if y_train.nunique() < 2 or y_cal.nunique() < 2:
        return ConstantProbabilityModel(float(y_train.mean()))

    base = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("model", _classifier(n_estimators=n_estimators, random_state=random_state)),
        ]
    )
    base.fit(_clean_x(train, feature_columns), y_train)
    return _fit_prefit_calibrator(base, _clean_x(calibration, feature_columns), y_cal)


def _fit_regression_model(
    train: pd.DataFrame,
    feature_columns: list[str],
    target: str,
    n_estimators: int,
    random_state: int,
) -> Any:
    y = train[target].astype(float)
    if y.nunique(dropna=True) < 2:
        return ConstantRegressor(float(y.mean()))
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("model", _regressor(n_estimators=n_estimators, random_state=random_state)),
        ]
    )
    model.fit(_clean_x(train, feature_columns), y)
    return model


def _train_side_bundle(
    dataset: pd.DataFrame,
    side: str,
    split: WalkForwardSplit,
    feature_columns: list[str],
    n_estimators: int,
    random_state: int,
) -> SideModelBundle:
    side_df = dataset[dataset["side"].eq(side) & dataset["status"].eq("evaluated")].copy()
    train = side_df[between_dates(side_df, end=split.base_train_end)]
    calibration = side_df[between_dates(side_df, split.calibration_start, split.calibration_end)]
    if not calibration.empty:
        calibration_dates = pd.to_datetime(calibration["date"])
        train_dates = pd.to_datetime(train["date"])
        overlap_mask = train_dates.between(calibration_dates.min(), calibration_dates.max())
        train_fit = train[~overlap_mask].copy()
        if not train_fit.empty:
            train = train_fit
    if calibration.empty:
        calibration = train
    if train.empty:
        raise ValueError(f"No training rows for side={side} split={split.name}")

    hit_model = _fit_probability_model(
        train,
        calibration,
        feature_columns,
        "hit_or_near",
        n_estimators=n_estimators,
        random_state=random_state,
    )
    clean_model = _fit_probability_model(
        train,
        calibration,
        feature_columns,
        "clean_success",
        n_estimators=n_estimators,
        random_state=random_state + 5,
    )
    opposite_model = _fit_probability_model(
        train,
        calibration,
        feature_columns,
        "opposite",
        n_estimators=n_estimators,
        random_state=random_state + 11,
    )
    favorable_model = _fit_regression_model(
        train,
        feature_columns,
        "favorable_move",
        n_estimators=n_estimators,
        random_state=random_state + 23,
    )
    signed_model = _fit_regression_model(
        train,
        feature_columns,
        "signed_close_return",
        n_estimators=n_estimators,
        random_state=random_state + 37,
    )
    return SideModelBundle(
        side=side,
        hit_near_model=hit_model,
        clean_success_model=clean_model,
        opposite_model=opposite_model,
        favorable_move_model=favorable_model,
        signed_close_model=signed_model,
        train_rows=int(len(train)),
        calibration_rows=int(len(calibration)),
    )


def train_swing_model(
    dataset: pd.DataFrame,
    feature_columns: list[str],
    split: WalkForwardSplit,
    model_id: str,
    n_estimators: int = 120,
    random_state: int = 17,
) -> SwingModel:
    long_bundle = _train_side_bundle(
        dataset, "long", split, feature_columns, n_estimators, random_state
    )
    short_bundle = _train_side_bundle(
        dataset, "short", split, feature_columns, n_estimators, random_state + 101
    )
    return SwingModel(
        model_id=model_id,
        split=split,
        feature_columns=feature_columns,
        algorithm="lightgbm" if HAS_LIGHTGBM else "hist_gradient_boosting",
        long_bundle=long_bundle,
        short_bundle=short_bundle,
        params={"n_estimators": n_estimators, "random_state": random_state},
    )


def _positive_probability(model: Any, x: pd.DataFrame) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="X does not have valid feature names.*")
        proba = model.predict_proba(x)
    return np.asarray(proba[:, 1], dtype=float)


def _predict_regression(model: Any, x: pd.DataFrame) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="X does not have valid feature names.*")
        return np.asarray(model.predict(x), dtype=float)


def predict_swing_model(model: SwingModel, dataset: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    passthrough = [
        "date",
        "symbol",
        "side",
        "band",
        "threshold",
        "entry_date",
        "entry_open",
        "window_end_date",
        "status",
        "verdict",
        "hit",
        "near",
        "hit_or_near",
        "clean_success",
        "opposite",
        "small",
        "favorable_move",
        "signed_close_return",
        "symbol_return_20d_rank_pct",
        "relative_return_20d",
        "hist_opposite_rate_63",
    ]
    for side, bundle in [("long", model.long_bundle), ("short", model.short_bundle)]:
        part = dataset[dataset["side"].eq(side)].copy()
        if part.empty:
            continue
        x = _clean_x(part, model.feature_columns)
        out = part[[c for c in passthrough if c in part.columns]].copy()
        out["p_hit_near"] = np.clip(_positive_probability(bundle.hit_near_model, x), 0, 1)
        out["p_clean_success"] = np.clip(
            _positive_probability(bundle.clean_success_model, x), 0, 1
        )
        out["p_opposite"] = np.clip(_positive_probability(bundle.opposite_model, x), 0, 1)
        out["expected_favorable_move"] = np.maximum(
            _predict_regression(bundle.favorable_move_model, x), 0
        )
        out["expected_signed_close_return"] = _predict_regression(bundle.signed_close_model, x)
        out["model_id"] = model.model_id
        frames.append(out)
    return pd.concat(frames, ignore_index=True).sort_values(["date", "symbol", "side"])


def save_model(model: SwingModel, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_dir / "model.joblib")
    manifest = {
        "model_id": model.model_id,
        "split": model.split.__dict__,
        "feature_count": len(model.feature_columns),
        "feature_columns": model.feature_columns,
        "algorithm": model.algorithm,
        "params": model.params,
        "long_train_rows": model.long_bundle.train_rows,
        "long_calibration_rows": model.long_bundle.calibration_rows,
        "short_train_rows": model.short_bundle.train_rows,
        "short_calibration_rows": model.short_bundle.calibration_rows,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def load_model(model_dir: Path) -> SwingModel:
    return joblib.load(model_dir / "model.joblib")
