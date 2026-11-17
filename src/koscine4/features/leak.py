"""Leak tests for features (requirements F-3, F-4).

Truncation test: compute features, then compute them again from inputs with every row after a
date T removed. A feature that uses only the past gives the same values up to T both times. A
feature that looks ahead does not, because what it looked at is gone.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from koscine4.features.builders import Inputs
from koscine4.features.registry import Feature, Registry, RegistryError

KEY = ["date", "instrument"]


def truncation_test(compute: Callable[[Inputs], pd.DataFrame], inputs: Inputs, dates, names=None,
                    tolerance: float = 1e-9, window: int = 5) -> list[dict]:
    """Rows where a feature changes when later data is removed. Empty means no leak was found.

    Compares the `window` sessions ending at each test date, where a look-ahead would show.
    """
    full = compute(inputs)
    names = [c for c in full.columns if c not in KEY + ["decision_time"]] if names is None else list(names)
    problems: list[dict] = []
    for t in pd.to_datetime(list(dates)):
        cut = compute(inputs.truncated(t))
        days = inputs.calendar.days
        recent = days[(days <= t)][-window:]
        a = full[full["date"].isin(recent)].set_index(KEY)[names].sort_index()
        b = cut[cut["date"].isin(recent)].set_index(KEY)[names].sort_index()
        b = b.reindex(a.index)
        for n in names:
            x, y = a[n].to_numpy(dtype=float), b[n].to_numpy(dtype=float)
            differs = ~((np.isnan(x) & np.isnan(y)) | (np.abs(x - y) <= tolerance * (1.0 + np.abs(x))))
            if differs.any():
                problems.append({"feature": n, "date": t, "rows": int(differs.sum())})
    return problems


def register_checked(registry: Registry, feature: Feature, compute: Callable[[Inputs], pd.DataFrame],
                     inputs: Inputs, dates) -> Feature:
    """Register a feature only if it passes the truncation test on `inputs` (F-4)."""
    leaks = truncation_test(compute, inputs, dates, names=[feature.name])
    if leaks:
        raise RegistryError(f"Feature '{feature.name}' uses data from after the decision date "
                            f"(changed on {len(leaks)} test date(s) when later rows were removed). Not registered.")
    return registry.register(feature)
