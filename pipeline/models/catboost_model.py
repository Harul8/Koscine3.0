"""
CatBoost model wrapper.

Mirrors the LGBMModel interface:
  - Per-target hyperparameter overrides from config
  - GPU auto-detection (tries GPU, falls back to CPU)
  - Early stopping on val AUC
  - Isotonic or Platt calibration on val fold
  - Pickle-safe save / load

CatBoost handles missing values natively — no imputation needed.
"""
from __future__ import annotations

import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from .base import BaseModel
from ..config import (
    CATBOOST_BASE_PARAMS, CATBOOST_TARGET_PARAMS, CALIBRATION_METHOD,
)
from ..calibration import make_calibrator


def _catboost_device() -> str:
    """Return 'GPU' if a CUDA device is available, else 'CPU'."""
    forced = os.environ.get("ESN_CATBOOST_DEVICE")
    if forced:
        return forced.upper()
    try:
        from catboost import CatBoostClassifier
        import numpy as np
        # Quick canary fit — if GPU is available CatBoost won't raise
        m = CatBoostClassifier(iterations=5, task_type="GPU", verbose=0)
        rng = np.random.default_rng(0)
        m.fit(rng.standard_normal((200, 5)), rng.integers(0, 2, 200))
        return "GPU"
    except Exception:
        return "CPU"


_CB_DEVICE: str | None = None


def _device() -> str:
    global _CB_DEVICE
    if _CB_DEVICE is None:
        _CB_DEVICE = _catboost_device()
        print(f"[catboost] device = {_CB_DEVICE}")
    return _CB_DEVICE


class CatBoostModel(BaseModel):
    """CatBoost binary classifier matching the BaseModel interface."""

    model_type = "catboost"

    def __init__(
        self,
        target: str,
        feature_cols: list[str] | None = None,
        **kwargs,
    ):
        """
        target       : label name (e.g. "up_liq", "bad_dn")
        feature_cols : explicit feature list; None → auto-detect
        **kwargs     : optional hyperparam overrides
        """
        self.target = target
        self._feature_cols_override = feature_cols
        self._extra_params = kwargs
        self._model = None
        self._calibrator = None
        self._feat_cols: list[str] = []
        self._base_rate: float = float("nan")

    # ── Training ───────────────────────────────────────────────────────────────

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val:   pd.DataFrame,
        y_val:   np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> dict:
        from catboost import CatBoostClassifier, Pool

        self._feat_cols = self._numeric_feat_cols(X_train, self._feature_cols_override)
        self._base_rate = float(np.asarray(y_train).mean())

        X_tr = self._clean(X_train[self._feat_cols]).astype(float).values.astype(np.float32)
        X_va = self._clean(X_val[self._feat_cols]).astype(float).values.astype(np.float32)
        y_tr = np.asarray(y_train, dtype=np.float32)
        y_va = np.asarray(y_val,   dtype=np.float32)

        # Build params: base → target override → caller override
        p = dict(CATBOOST_BASE_PARAMS)
        p.update(CATBOOST_TARGET_PARAMS.get(self.target, {}))
        p.update(self._extra_params)

        # Class balance via scale_pos_weight equivalent (class_weights)
        pos_w = (1.0 - self._base_rate) / (self._base_rate + 1e-7)
        p["class_weights"] = [1.0, pos_w]
        # task_type: respect per-target override (e.g. "CPU" for dn_liq/overlay
        # targets where GPU falls back to Logloss and early-stops at iter 1-17).
        # Only auto-detect if not already set.
        if "task_type" not in p:
            p["task_type"] = _device()

        print(f"  [{self.target}] base_rate={self._base_rate:.3%}  "
              f"pos_weight={pos_w:.2f}  iterations={p.get('iterations')}  "
              f"device={p['task_type']}")

        train_pool = Pool(X_tr, label=y_tr, weight=sample_weight)
        val_pool   = Pool(X_va, label=y_va)

        self._model = CatBoostClassifier(**p)
        self._model.fit(train_pool, eval_set=val_pool, use_best_model=True)

        raw_val = self._model.predict_proba(X_va)[:, 1]

        # Calibrate
        self._calibrator = make_calibrator(self.target, CALIBRATION_METHOD)
        self._calibrator.fit(raw_val, y_va)
        cal_val = self._calibrator.predict(raw_val)

        metrics = self._metrics(y_va, cal_val)
        print(f"  [{self.target}] catboost  val_auc={metrics['val_auc']:.3f}  "
              f"val_ap={metrics['val_ap']:.3f}  "
              f"prec@10%={metrics['val_prec10']:.2%}  "
              f"best_iter={self._model.get_best_iteration()}")
        return metrics

    # ── Inference ──────────────────────────────────────────────────────────────

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X_in = self._clean(X[self._feat_cols]).astype(float).values.astype(np.float32)
        raw  = self._model.predict_proba(X_in)[:, 1]
        return self._calibrator.predict(raw)

    # ── Persistence ────────────────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        payload = {
            "model":                 self._model,
            "calibrator":            self._calibrator,
            "feat_cols":             self._feat_cols,
            "base_rate":             self._base_rate,
            "target":                self.target,
            "model_type":            self.model_type,
            "feature_cols_override": self._feature_cols_override,
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(payload, fh)

    @classmethod
    def load(cls, path: Path) -> "CatBoostModel":
        with open(path, "rb") as fh:
            p = pickle.load(fh)
        m = cls(target=p["target"])
        m._model                  = p["model"]
        m._calibrator             = p["calibrator"]
        m._feat_cols              = p["feat_cols"]
        m._base_rate              = p["base_rate"]
        m._feature_cols_override  = p.get("feature_cols_override")
        return m

    @property
    def base_rate(self) -> float:
        return self._base_rate

    @property
    def feat_cols(self) -> list[str]:
        return self._feat_cols
