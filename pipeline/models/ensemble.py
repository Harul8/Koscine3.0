"""
Ensemble utilities — rank-blending across model types.

Design:
  - Each model type (lgbm, ae_mlp, tabm) produces a probability for each target.
  - Probabilities are converted to percentile ranks within the day's universe.
  - Weighted average of ranks gives the ensemble score.
  - Weights are loaded from prod/ensemble_weights.json if present,
    otherwise uniform across available model types.

Usage:
    from pipeline.models.ensemble import load_prod_models, predict_ensemble

    models = load_prod_models(prod_dir)
    scores = predict_ensemble(models, X_today)   # dict target → np.ndarray
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import get_model_class
from .base import BaseModel

# filename pattern: {target}_{model_type}.pkl
_MODEL_TYPES = ["lgbm", "ae_mlp", "tabm", "catboost"]


def load_prod_models(prod_dir: Path) -> dict[str, dict[str, BaseModel]]:
    """
    Scan prod_dir for all model pkl files.

    For tabm, automatically aggregates `{target}_tabm_seed*.pkl` files into
    a single TabMSeedEnsemble that averages predictions across seeds.

    Returns:
        {target: {model_type: model_instance}}
    e.g. {"up_5": {"lgbm": <LGBMModel>, "tabm": <TabMSeedEnsemble>}}
    """
    prod_dir = Path(prod_dir)
    out: dict[str, dict[str, BaseModel]] = {}

    # ── First pass: gather tabm seeds per target ─────────────────────────────
    from .tabm_ensemble import TabMSeedEnsemble
    tabm_seeds: dict[str, list[Path]] = {}
    for pkl in sorted(prod_dir.glob("*_tabm_seed*.pkl")):
        stem = pkl.stem
        target = stem.split("_tabm_seed")[0]
        tabm_seeds.setdefault(target, []).append(pkl)

    for target, pkls in tabm_seeds.items():
        try:
            from .tabm import TabMModel
            models = [TabMModel.load(p) for p in sorted(pkls)]
            out.setdefault(target, {})["tabm"] = TabMSeedEnsemble(models)
            print(f"[ensemble] {target}/tabm: {len(models)}-seed ensemble loaded")
        except Exception as e:
            print(f"[ensemble] warn: tabm seeds for {target}: {e}")

    # ── Second pass: single-model files ──────────────────────────────────────
    for pkl in sorted(prod_dir.glob("*.pkl")):
        stem = pkl.stem   # e.g. "up_5_lgbm"
        if "_tabm_seed" in stem:
            continue   # already handled above
        for mt in _MODEL_TYPES:
            if stem.endswith(f"_{mt}"):
                target = stem[: -(len(mt) + 1)]
                if target in out and mt in out[target]:
                    break   # already loaded as seed ensemble
                cls = get_model_class(mt)
                try:
                    model = cls.load(pkl)
                    out.setdefault(target, {})[mt] = model
                except Exception as e:
                    print(f"[ensemble] warn: could not load {pkl.name}: {e}")
                break

    if not out:
        raise FileNotFoundError(f"No model pkl files found in {prod_dir}")

    loaded = {t: list(v.keys()) for t, v in out.items()}
    print(f"[ensemble] loaded: {loaded}")
    return out


def predict_ensemble(
    models: dict[str, dict[str, BaseModel]],
    X: pd.DataFrame,
    weights: dict[str, float] | None = None,
    per_target_weights: "dict[str, dict[str, float]] | None" = None,
) -> dict[str, np.ndarray]:
    """
    Run ensemble prediction for all targets.

    For each target:
      1. Get predictions from each available model type.
      2. Convert each to within-day percentile rank (0→1).
      3. Weighted average of ranks → ensemble score.

    weights             : global model-type weights  {model_type: float}
                          applied uniformly across all targets (falls back
                          to 1.0 for any type not listed).
    per_target_weights  : per-target overrides  {target: {model_type: float}}
                          Takes priority over `weights` for listed targets.
                          Loaded from prod/tiered_blend_weights.json for tiered models.

    Returns:
        {target: ensemble_proba_array}  (aligned with X index)
    """
    scores: dict[str, np.ndarray] = {}
    weights = weights or {}
    per_target_weights = per_target_weights or {}

    for target, type_models in models.items():
        probs_by_type: dict[str, np.ndarray] = {}
        base_rates: list[float] = []

        for mt, model in type_models.items():
            try:
                p = model.predict_proba(X)
                probs_by_type[mt] = p
                if hasattr(model, "base_rate"):
                    base_rates.append(model.base_rate)
            except Exception as e:
                print(f"[ensemble] warn: {target}/{mt} predict failed: {e}")

        if not probs_by_type:
            continue

        # Resolve weights: per-target override > global weights > uniform 1.0
        target_type_w = per_target_weights.get(target, {})

        # Rank each model's predictions within today's universe
        ranks: list[np.ndarray] = []
        type_weights: list[float] = []
        for mt, p in probs_by_type.items():
            rank = _to_rank(p)
            ranks.append(rank)
            type_weights.append(target_type_w.get(mt, weights.get(mt, 1.0)))

        w = np.array(type_weights, dtype=float)
        w /= w.sum()
        ensemble = sum(wi * r for wi, r in zip(w, ranks))

        scores[target] = ensemble

    return scores


def load_weights(prod_dir: Path) -> dict[str, float]:
    """Load ensemble weights from prod_dir/ensemble_weights.json if it exists."""
    path = Path(prod_dir) / "ensemble_weights.json"
    if path.exists():
        return json.loads(path.read_text())
    return {}


def save_weights(weights: dict[str, float], prod_dir: Path) -> None:
    path = Path(prod_dir) / "ensemble_weights.json"
    path.write_text(json.dumps(weights, indent=2))
    print(f"[ensemble] saved weights → {path}")


def optimize_weights(
    models: dict[str, dict[str, BaseModel]],
    X_hold: pd.DataFrame,
    y_hold: dict[str, np.ndarray],
) -> dict[str, float]:
    """
    Grid-search blend weights on a holdout set.
    Returns the model_type → weight mapping that maximises composite AP.
    """
    from sklearn.metrics import average_precision_score
    from ..config import SEARCH_AP_WEIGHTS

    best_score = -1.0
    best_weights: dict[str, float] = {}

    # Collect raw probs per model type across all targets
    raw: dict[str, dict[str, np.ndarray]] = {}  # {model_type: {target: probs}}
    all_types: set[str] = set()
    for target, type_models in models.items():
        for mt, model in type_models.items():
            all_types.add(mt)
            raw.setdefault(mt, {})[target] = model.predict_proba(X_hold)

    all_types_list = sorted(all_types)
    if len(all_types_list) == 1:
        return {all_types_list[0]: 1.0}

    # Grid: try weights at 0.0, 0.2, 0.4, ... per type
    import itertools
    steps = np.arange(0.0, 1.01, 0.2)
    for combo in itertools.product(steps, repeat=len(all_types_list)):
        s = sum(combo)
        if s < 0.01:
            continue
        w = {mt: c / s for mt, c in zip(all_types_list, combo)}

        total = 0.0
        for target, target_w in SEARCH_AP_WEIGHTS.items():
            if target not in y_hold:
                continue
            probs_list = [raw[mt][target] * w[mt]
                          for mt in all_types_list if target in raw.get(mt, {})]
            if not probs_list:
                continue
            blend = sum(probs_list)
            try:
                ap = average_precision_score(y_hold[target], blend)
                total += target_w * ap
            except Exception:
                pass

        if total > best_score:
            best_score = total
            best_weights = dict(w)

    print(f"[ensemble] optimal weights: {best_weights}  score={best_score:.4f}")
    return best_weights


def _to_rank(probs: np.ndarray) -> np.ndarray:
    """Convert probabilities to uniform [0,1] percentile ranks."""
    n = len(probs)
    order = np.argsort(probs)
    ranks = np.empty(n)
    ranks[order] = np.arange(n) / max(n - 1, 1)
    return ranks
