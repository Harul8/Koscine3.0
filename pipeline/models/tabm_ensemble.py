"""
TabMSeedEnsemble — averages predictions from N TabMModel instances trained
with different seeds. Variance reduction by ~1/sqrt(N).

Standard production pattern:
    tabm_seed0.pkl, tabm_seed1.pkl, ..., tabm_seed4.pkl  →  one prediction

Files are saved by train.py as: {target}_tabm_seed{i}.pkl
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd

from .tabm import TabMModel


class TabMSeedEnsemble:
    """Average predict_proba across N seed-trained TabMModel instances."""

    model_type = "tabm"

    def __init__(self, models: list[TabMModel]):
        if not models:
            raise ValueError("TabMSeedEnsemble requires at least 1 model")
        self.models = models
        # Sanity: all members must agree on target + feature_cols
        target = models[0].target
        feat_cols = models[0].feat_cols
        for i, m in enumerate(models[1:], start=1):
            if m.target != target:
                raise ValueError(f"seed {i} target {m.target} != seed 0 target {target}")
            if m.feat_cols != feat_cols:
                raise ValueError(f"seed {i} feat_cols disagrees with seed 0")
        self.target    = target
        self.feat_cols = feat_cols

    # ── Inference ──────────────────────────────────────────────────────────────
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Rank-average across seeds: convert each seed's probs to percentile
        ranks, then average. Better than probability averaging for top-k
        metrics (prec@10%) because it preserves extreme predictions instead
        of shrinking them toward the mean."""
        probs_stack = np.stack([m.predict_proba(X) for m in self.models])
        # Rank each seed's predictions independently (0..1 percentile)
        ranks = np.empty_like(probs_stack)
        for i in range(probs_stack.shape[0]):
            order = probs_stack[i].argsort()
            ranks[i, order] = np.linspace(0.0, 1.0, len(order))
        # Average ranks across seeds, then rescale to a probability-like [0,1]
        avg_rank = ranks.mean(axis=0)
        # Convert back to probability scale matching the calibration of base rate
        # via a simple linear blend: mostly rank, slight prob blend for stability
        mean_prob = probs_stack.mean(axis=0)
        return 0.7 * avg_rank + 0.3 * mean_prob

    def predict_proba_mean(self, X: pd.DataFrame) -> np.ndarray:
        """Pure probability averaging — use when you need calibrated probabilities."""
        return np.stack([m.predict_proba(X) for m in self.models]).mean(axis=0)

    @property
    def base_rate(self) -> float:
        return float(np.mean([m.base_rate for m in self.models]))

    # ── Loaders ────────────────────────────────────────────────────────────────
    @classmethod
    def load_from_dir(cls, run_dir: Path, target: str) -> "TabMSeedEnsemble":
        """Load all `{target}_tabm_seed*.pkl` files from a run directory."""
        run_dir = Path(run_dir)
        pkls = sorted(run_dir.glob(f"{target}_tabm_seed*.pkl"))
        if not pkls:
            # Fall back to non-seeded single model if present
            single = run_dir / f"{target}_tabm.pkl"
            if single.exists():
                return cls([TabMModel.load(single)])
            raise FileNotFoundError(f"No {target}_tabm_seed*.pkl in {run_dir}")
        models = [TabMModel.load(p) for p in pkls]
        print(f"  [tabm-ensemble:{target}] loaded {len(models)} seeds")
        return cls(models)
