"""Model types and the walk-forward loop (requirements M-2 to M-7, V-2, V-3).

Version 1 has two model types only: regularised linear models and gradient-boosted trees (M-3).
Every model is fitted with a fixed seed, so retraining from its record gives the same predictions.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from koscine4.harness.walkforward import Fold, check_fold
from koscine4.store.calendar import TradingCalendar

ALGORITHMS = ("ridge", "logistic", "gbt")
KINDS = ("regression", "classification")


class ModelError(ValueError):
    """Raised for a model configuration outside what version 1 allows."""


@dataclass(frozen=True)
class ModelConfig:
    name: str
    kind: str                         # regression | classification
    algorithm: str                    # ridge | logistic | gbt
    features: tuple[str, ...]
    decision_time: str = "evening"
    target: dict = field(default_factory=dict)       # table, column, horizon  (or table="structure", strategy)
    instruments: tuple[str, ...] = ()                # empty = every F&O stock that existed at the time (U-6)
    train_window_months: int = 0                     # 0 = all history; above 0 = recent data only
    refit_every_months: int = 1                      # M-4: monthly
    min_train_rows: int = 500
    seed: int = 7
    params: dict = field(default_factory=dict)
    start: str = ""

    def validate(self) -> "ModelConfig":
        from koscine4.features.registry import MAX_MODEL_FEATURES, REGISTRY

        if self.algorithm not in ALGORITHMS or self.kind not in KINDS:
            raise ModelError(f"Version 1 models are {ALGORITHMS} of kind {KINDS} (M-3).")
        if self.algorithm == "ridge" and self.kind != "regression" or self.algorithm == "logistic" and self.kind != "classification":
            raise ModelError("ridge is for regression and logistic is for classification.")
        REGISTRY.require(self.features)
        for n in self.features:
            if self.decision_time not in REGISTRY.get(n).decision_times:
                raise ModelError(f"Feature '{n}' is not valid for a {self.decision_time} decision (F-5).")
        if len(self.features) > MAX_MODEL_FEATURES:
            raise ModelError(f"A version 1 model uses at most {MAX_MODEL_FEATURES} features (F-10); this one lists {len(self.features)}.")
        return self


def make_model(cfg: ModelConfig):
    """An unfitted model with `fit(X, y, sample_weight)` and `predict(X)` returning a score per row."""
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    p = dict(cfg.params)
    if cfg.algorithm == "ridge":
        from sklearn.linear_model import Ridge

        return _Wrapped(make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=float(p.get("alpha", 10.0)), random_state=cfg.seed)), False, "ridge__sample_weight")
    if cfg.algorithm == "logistic":
        from sklearn.linear_model import LogisticRegression

        return _Wrapped(make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                      LogisticRegression(C=float(p.get("C", 0.1)), max_iter=500, random_state=cfg.seed)), True, "logisticregression__sample_weight")
    import lightgbm as lgb

    base = dict(n_estimators=int(p.get("n_estimators", 200)), learning_rate=float(p.get("learning_rate", 0.03)),
                num_leaves=int(p.get("num_leaves", 15)), min_child_samples=int(p.get("min_child_samples", 50)),
                subsample=float(p.get("subsample", 0.8)), subsample_freq=1, colsample_bytree=float(p.get("colsample_bytree", 0.8)),
                reg_lambda=float(p.get("reg_lambda", 5.0)), random_state=cfg.seed, deterministic=True, force_row_wise=True,
                n_jobs=int(p.get("n_jobs", 4)), verbose=-1)
    model = lgb.LGBMClassifier(**base) if cfg.kind == "classification" else lgb.LGBMRegressor(**base)
    return _Wrapped(model, cfg.kind == "classification", "sample_weight")


class _Wrapped:
    def __init__(self, model, proba: bool, weight_arg: str):
        self.model, self.proba, self.weight_arg = model, proba, weight_arg

    def fit(self, X, y, sample_weight=None):
        kw = {self.weight_arg: sample_weight} if sample_weight is not None else {}
        self.model.fit(X, y, **kw)
        return self

    def predict(self, X) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1] if self.proba else self.model.predict(X)


def walk_forward(cfg: ModelConfig, features: pd.DataFrame, outcomes: pd.DataFrame, folds: list[Fold], calendar: TradingCalendar,
                 min_gap: int, weight: float = 1.0, model_dir: Path | None = None, score_instruments=None,
                 fit_predict=None) -> tuple[pd.DataFrame, list[dict]]:
    """Train on outcomes that ended by each cutoff, predict the following month.

    features: date, instrument, <feature columns>.  outcomes: date, instrument, y, window_end.
    Returns (predictions [date, instrument, prediction, cutoff], one record per fitted model).
    """
    names = list(cfg.features)
    labelled = features.merge(outcomes[["date", "instrument", "y", "window_end"]], on=["date", "instrument"], how="inner")
    labelled = labelled[labelled["y"].notna()].sort_values("date")
    preds, records, model, fitted_cutoff = [], [], None, None
    for k, fold in enumerate(folds):
        if model is None or k % max(1, cfg.refit_every_months) == 0:
            train = labelled[labelled["window_end"] <= fold.cutoff]
            if cfg.train_window_months > 0:
                train = train[train["date"] > fold.cutoff - pd.DateOffset(months=cfg.train_window_months)]
            check_fold(fold, train["window_end"], calendar, min_gap)                 # V-3, M-4
            if len(train) < cfg.min_train_rows or train["y"].nunique() < 2:
                continue
            w = np.full(len(train), weight)
            if fit_predict is None:
                model = make_model(cfg).fit(train[names], train["y"], sample_weight=w)
            fitted_cutoff = fold.cutoff
            record = {"model": cfg.name, "cutoff": str(fold.cutoff.date()), "train_rows": int(len(train)),
                      "train_first": str(train["date"].min().date()), "train_last_window_end": str(train["window_end"].max().date()),
                      "features": names, "seed": cfg.seed, "algorithm": cfg.algorithm, "kind": cfg.kind, "params": dict(cfg.params)}
            records.append(record)
            if model_dir is not None and fit_predict is None:
                save_model(model_dir / str(fold.cutoff.date()), model, record)
        test = features[(features["date"] >= fold.test_start) & (features["date"] <= fold.test_end)]
        if score_instruments is not None:
            test = test[test["instrument"].isin(score_instruments(fold.test_start))]   # U-6: score only the tradeable list
        if test.empty or (model is None and fit_predict is None):
            continue
        if fit_predict is not None:
            train = labelled[labelled["window_end"] <= fold.cutoff]
            if len(train) < cfg.min_train_rows:
                continue
            p = fit_predict(train[names], train["y"], test[names])
            fitted_cutoff = fold.cutoff
        else:
            p = model.predict(test[names])
        preds.append(pd.DataFrame({"date": test["date"].to_numpy(), "instrument": test["instrument"].to_numpy(),
                                   "prediction": np.asarray(p, dtype=float), "cutoff": fitted_cutoff}))
    out = pd.concat(preds, ignore_index=True) if preds else pd.DataFrame(columns=["date", "instrument", "prediction", "cutoff"])
    return out, records


def save_model(folder: Path, model: _Wrapped, record: dict) -> None:
    import joblib

    folder.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, folder / "model.joblib")
    (folder / "record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")


def load_latest_model(model_root: Path, name: str, on_or_before) -> tuple[_Wrapped, dict]:
    """The saved model with the latest cutoff on or before a date, and its record."""
    import joblib

    folders = sorted(p for p in (model_root / name).glob("*") if (p / "model.joblib").exists() and pd.Timestamp(p.name) <= pd.Timestamp(on_or_before))
    if not folders:
        raise ModelError(f"No saved model '{name}' with a cutoff on or before {pd.Timestamp(on_or_before).date()}.")
    record = json.loads((folders[-1] / "record.json").read_text(encoding="utf-8"))
    return joblib.load(folders[-1] / "model.joblib"), record


def calibration_table(pred: pd.Series, y: pd.Series, bins: int = 10) -> list[dict]:
    """Predicted probability against realised rate, by tenth of the predictions (M-6)."""
    df = pd.DataFrame({"p": pred, "y": y}).dropna()
    if len(df) < bins * 5:
        return []
    df["bin"] = pd.qcut(df["p"], bins, labels=False, duplicates="drop")
    return [{"bin": int(b), "mean_prediction": float(g["p"].mean()), "realised_rate": float(g["y"].mean()), "rows": int(len(g))}
            for b, g in df.groupby("bin")]
