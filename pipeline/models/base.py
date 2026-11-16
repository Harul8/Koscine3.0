"""
Abstract base class that all model types must implement.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd


class BaseModel(ABC):
    """Common interface for LightGBM, AE-MLP, and TabM models."""

    model_type: str = "base"   # overridden by subclasses

    # ── Required interface ─────────────────────────────────────────────────────

    @abstractmethod
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: pd.DataFrame,
        y_val: np.ndarray,
    ) -> dict:
        """Train on (X_train, y_train), early-stop on (X_val, y_val).

        Returns a metrics dict containing at minimum:
            val_ap   — Average Precision on val (primary objective)
            val_auc  — ROC-AUC on val
            base_rate — fraction of positives in training set
        """

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return 1-D array of positive-class probabilities, shape (n,)."""

    @abstractmethod
    def save(self, path: Path) -> None:
        """Serialize model + preprocessing to disk."""

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> "BaseModel":
        """Deserialize from disk and return a ready-to-predict instance."""

    # ── Shared helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _clean(X: pd.DataFrame) -> pd.DataFrame:
        """Replace inf with NaN (model-specific NaN handling varies)."""
        return X.replace([float("inf"), float("-inf")], float("nan"))

    @staticmethod
    def _numeric_feat_cols(X: pd.DataFrame,
                           override: list[str] | None = None) -> list[str]:
        """
        Return column names that are numeric and not identity columns.
        Excludes 'date', 'symbol', and any object/categorical/datetime columns.
        Called once during fit() to determine which columns the model will use.

        override : if supplied, use only these columns (filtered to those that
                   are actually present and numeric in X).  This enables per-target
                   feature selection via TARGET_FEATURE_COLS in config.py.
        """
        skip = {"date", "symbol"}
        if override is not None:
            return [
                c for c in override
                if c not in skip
                and c in X.columns
                and pd.api.types.is_numeric_dtype(X[c])
            ]
        return [
            c for c in X.columns
            if c not in skip and pd.api.types.is_numeric_dtype(X[c])
        ]

    @staticmethod
    def _metrics(y_true: np.ndarray, probs: np.ndarray) -> dict:
        from sklearn.metrics import roc_auc_score, average_precision_score
        n = len(y_true)
        base = float(y_true.mean())
        order = np.argsort(probs)[::-1]
        top_n = max(1, int(0.10 * n))
        prec10 = float(y_true[order[:top_n]].mean()) if top_n else float("nan")
        try:
            auc = float(roc_auc_score(y_true, probs)) if len(np.unique(y_true)) > 1 else float("nan")
            ap  = float(average_precision_score(y_true, probs)) if len(np.unique(y_true)) > 1 else float("nan")
        except Exception:
            auc = ap = float("nan")
        return {
            "val_auc":  auc,
            "val_ap":   ap,
            "val_prec10": prec10,
            "val_lift": prec10 / base if base > 0 else float("nan"),
            "base_rate": base,
            "n_val": n,
        }
