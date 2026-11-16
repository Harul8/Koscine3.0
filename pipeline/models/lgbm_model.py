"""
LightGBM model wrapper.

Wraps LGBMClassifier with:
  - Per-target hyperparameter overrides from config
  - GPU auto-detection (tries cuda, falls back to cpu)
  - Early stopping on val AUC
  - Isotonic or Platt calibration on val fold
  - Pickle-safe save / load
"""
from __future__ import annotations
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .base import BaseModel
from ..config import (
    LGBM_BASE_PARAMS, LGBM_TARGET_PARAMS, CALIBRATION_METHOD,
)
from ..calibration import make_calibrator


def _try_gpu() -> str:
    """
    Benchmark CPU vs GPU (OpenCL) on a representative dummy dataset and return
    whichever is faster.  CUDA tree learner requires a custom build and is
    skipped if not available.

    Benchmark: 5,000 rows × 100 features, 200 trees — fast enough to run at
    startup, representative of tree-building cost at our actual feature count.
    """
    import time
    import lightgbm as lgb
    import numpy as np

    rng  = np.random.default_rng(0)
    X_bm = rng.standard_normal((5_000, 100)).astype(np.float32)
    y_bm = rng.integers(0, 2, size=5_000).astype(np.float32)

    base_params = dict(n_estimators=200, num_leaves=31, verbose=-1,
                       n_jobs=-1, random_state=0)

    def _time(dev: str) -> float | None:
        try:
            m = lgb.LGBMClassifier(**base_params, device_type=dev)
            t0 = time.perf_counter()
            m.fit(X_bm, y_bm)
            return time.perf_counter() - t0
        except Exception:
            return None

    timings: dict[str, float] = {}
    for dev in ("cpu", "cuda", "gpu"):
        t = _time(dev)
        if t is not None:
            timings[dev] = t

    best_dev = min(timings, key=timings.get)

    summary = "  ".join(
        f"{d}={t:.2f}s{'*' if d == best_dev else ''}"
        for d, t in timings.items()
    )
    print(f"[lgbm] device benchmark — {summary}")
    return best_dev


_DEVICE: str | None = None   # cached after first call


def _device() -> str:
    """Return the best available device, checking env var ESN_LGBM_DEVICE first."""
    global _DEVICE
    if _DEVICE is None:
        import os
        forced = os.environ.get("ESN_LGBM_DEVICE")
        if forced:
            _DEVICE = forced
            print(f"[lgbm] device = {_DEVICE}  (ESN_LGBM_DEVICE override)")
        else:
            _DEVICE = _try_gpu()
            print(f"[lgbm] device = {_DEVICE}")
    return _DEVICE


class LGBMModel(BaseModel):
    model_type = "lgbm"

    def __init__(self, target: str, feature_cols: list[str] | None = None, **kwargs):
        """
        target       : one of MODEL_TARGETS (e.g. "up_5")
        feature_cols : explicit list of features to use.  None = auto-detect
                       from the dataset (uses all numeric non-identity cols).
                       Set via config.TARGET_FEATURE_COLS[target] for per-target
                       feature independence.
        **kwargs     : optional hyperparam overrides (used by Optuna search).
        """
        self.target = target
        self._feature_cols_override = feature_cols   # None → auto
        self._extra_params = kwargs
        self._model = None
        self._calibrator = None
        self._feat_cols: list[str] = []
        self._base_rate: float = float("nan")

    # ── Training ───────────────────────────────────────────────────────────────

    def fit(self, X_train, y_train, X_val, y_val) -> dict:
        import lightgbm as lgb

        self._feat_cols = self._numeric_feat_cols(X_train, self._feature_cols_override)
        self._base_rate = float(np.asarray(y_train).mean())

        X_train = self._clean(X_train[self._feat_cols])
        X_val   = self._clean(X_val[self._feat_cols])

        # Build params: base → target override → caller override
        p = dict(LGBM_BASE_PARAMS)
        p.update(LGBM_TARGET_PARAMS.get(self.target, {}))
        p.update(self._extra_params)
        # Compute pos_weight from actual training data so it is correct for any
        # label config (0.18/0.09 gives ~9% xl rate, not the 11.78% in config).
        # TabM and AE-MLP already do this dynamically.
        pos_w = (1 - self._base_rate) / (self._base_rate + 1e-7)
        p["scale_pos_weight"] = pos_w
        p["device_type"] = _device()
        print(f"  [{self.target}] base_rate={self._base_rate:.3%}  "
              f"pos_weight={pos_w:.2f}  n_est={p.get('n_estimators')}")

        self._model = lgb.LGBMClassifier(**p)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[
                    lgb.early_stopping(50, verbose=False),
                    lgb.log_evaluation(500),
                ],
            )

        raw_val = self._model.predict_proba(X_val)[:, 1]

        # Calibrate
        self._calibrator = make_calibrator(self.target, CALIBRATION_METHOD)
        self._calibrator.fit(raw_val, y_val)
        cal_val = self._calibrator.predict(raw_val)

        metrics = self._metrics(np.asarray(y_val), cal_val)
        print(f"  [{self.target}] val_auc={metrics['val_auc']:.3f}  "
              f"val_ap={metrics['val_ap']:.3f}  "
              f"prec@10%={metrics['val_prec10']:.2%}  "
              f"lift={metrics['val_lift']:.2f}x")
        return metrics

    # ── Inference ──────────────────────────────────────────────────────────────

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X_in = self._clean(X[self._feat_cols])
        raw  = self._model.predict_proba(X_in)[:, 1]
        return self._calibrator.predict(raw)

    # ── Persistence ────────────────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        payload = {
            "model":               self._model,
            "calibrator":          self._calibrator,
            "feat_cols":           self._feat_cols,
            "base_rate":           self._base_rate,
            "target":              self.target,
            "model_type":          self.model_type,
            "feature_cols_override": self._feature_cols_override,
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(payload, fh)

    @classmethod
    def load(cls, path: Path) -> "LGBMModel":
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
