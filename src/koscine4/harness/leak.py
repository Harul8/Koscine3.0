"""Leak tests for models (requirement V-13).

Two checks that a pipeline is not learning from the future:

1. Shuffled labels. Train and test with the outcomes shuffled within each date. Any real link
   between inputs and outcomes is destroyed, so the score must be about zero. A score well
   above zero means information is reaching the model some other way.
2. Too-good columns. An input that tracks the future outcome almost perfectly is not a
   feature, it is the answer. Such a column is reported.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd


def rank_correlation(a, b) -> float:
    a, b = pd.Series(np.asarray(a, dtype=float)), pd.Series(np.asarray(b, dtype=float))
    ok = a.notna() & b.notna()
    if ok.sum() < 3 or a[ok].nunique() < 2 or b[ok].nunique() < 2:
        return 0.0
    return float(a[ok].rank().corr(b[ok].rank()))


def shuffle_within_dates(y: pd.Series, dates: pd.Series, seed: int) -> pd.Series:
    rng = np.random.default_rng(seed)
    out = y.copy()
    for _, idx in y.groupby(dates.values, sort=False).groups.items():
        out.loc[idx] = rng.permutation(y.loc[idx].to_numpy())
    return out


def shuffled_label_score(fit_predict: Callable, X: pd.DataFrame, y: pd.Series, dates: pd.Series, folds, seed: int = 7) -> float:
    """Out-of-sample rank correlation when the model is trained on shuffled outcomes. Should be near zero.

    fit_predict(X_train, y_train, X_test) -> predictions.  folds: iterable of (train_mask, test_mask).
    """
    shuffled = shuffle_within_dates(y, dates, seed)
    preds, truth = [], []
    for train, test in folds:
        if train.sum() == 0 or test.sum() == 0:
            continue
        preds.append(np.asarray(fit_predict(X[train], shuffled[train], X[test])))
        truth.append(y[test].to_numpy())
    if not preds:
        return 0.0
    return rank_correlation(np.concatenate(preds), np.concatenate(truth))


def too_good_columns(X: pd.DataFrame, y: pd.Series, max_abs_corr: float = 0.5) -> list[dict]:
    """Columns whose rank correlation with the outcome is implausibly high for a real feature."""
    out = []
    for col in X.columns:
        c = rank_correlation(X[col], y)
        if abs(c) > max_abs_corr:
            out.append({"column": col, "rank_correlation": c})
    return out
