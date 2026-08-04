"""
Single experiment runner — one trial in the Optuna / search loop.

Responsibilities:
  1. Build labels with the trial's label config (custom target_rate_base/xl).
     Labels are saved to a per-config temp file so shared features are untouched.
  2. Load shared features.parquet, inner-join with labels.
  3. Split into train / val by date (uses TRAIN_END / VAL_END from config).
  4. Train one model type for all MODEL_TARGETS.
  5. Compute weighted Average Precision on val (SEARCH_AP_WEIGHTS).
  6. Log params + metrics to MLflow.
  7. Return the composite AP score.

Can be imported by search.py (as Optuna objective) or run as a standalone script.

Usage (standalone):
    python -m pipeline.experiment \\
        --model_type lgbm \\
        --label_base 0.20 --label_xl 0.10
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import gc

import numpy as np
import pandas as pd

from .config import (
    GOLD_FEATURES, GOLD_DIR, MODEL_DIR, MLRUNS_DIR, MLFLOW_EXPERIMENT,
    MODEL_TARGETS, SEARCH_AP_WEIGHTS, TRAIN_END, VAL_END,
    TARGET_FEATURE_COLS,
)
from .models import get_model_class


# ── Helpers ────────────────────────────────────────────────────────────────────

def _labels_path(target_rate_base: float, target_rate_xl: float) -> Path:
    """Deterministic per-config label file path in gold/labels_exp/."""
    key = f"b{target_rate_base:.3f}_xl{target_rate_xl:.3f}"
    return GOLD_DIR / "labels_exp" / f"labels_{key}.parquet"


def _ensure_labels(target_rate_base: float, target_rate_xl: float) -> Path:
    """Build labels for this config if not already on disk, return path."""
    path = _labels_path(target_rate_base, target_rate_xl)
    if not path.exists():
        from .labels import build_with_rates
        build_with_rates(target_rate_base, target_rate_xl, out_path=path)
    return path


def _load_panel(labels_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load features + labels, inner-join on (date, symbol).
    Returns (train_df, val_df) — test rows are excluded.
    """
    feat = pd.read_parquet(GOLD_FEATURES)
    feat["date"] = pd.to_datetime(feat["date"])

    labs = pd.read_parquet(labels_path)
    labs["date"] = pd.to_datetime(labs["date"])

    panel = feat.merge(labs[["date", "symbol", "split"] + MODEL_TARGETS],
                       on=["date", "symbol"], how="inner")

    train = panel[panel["split"] == "train"].drop(columns=["split"]).reset_index(drop=True)
    val   = panel[panel["split"] == "val"  ].drop(columns=["split"]).reset_index(drop=True)
    return train, val


def _composite_ap(metrics_by_target: dict[str, dict]) -> float:
    """Weighted average of per-target val AP."""
    total = 0.0
    for target, w in SEARCH_AP_WEIGHTS.items():
        ap = metrics_by_target.get(target, {}).get("val_ap", float("nan"))
        if not np.isnan(ap):
            total += w * ap
    return total


# ── Main entry point ───────────────────────────────────────────────────────────

def run_trial(
    model_type: str,
    model_params: dict[str, Any],
    label_config: dict[str, float] | None = None,
    run_name: str | None = None,
    save_models: bool = False,
    save_dir: Path | None = None,
    search_frac: float = 1.0,
    preloaded_data: tuple | None = None,
    target_filter: str | None = None,
    return_metric: str = "val_ap",
    per_target_feature_cols: dict[str, list[str] | None] | None = None,
) -> float:
    """
    Train one model type across all targets and return composite val AP.

    Parameters
    ----------
    model_type    : "lgbm", "ae_mlp", or "tabm"
    model_params  : hyperparam dict passed to the model constructor
    label_config  : {"target_rate_base": float, "target_rate_xl": float}
                    If None, uses the canonical GOLD_LABELS file.
    run_name      : MLflow run name (auto-generated if None)
    save_models   : if True, save trained models to save_dir
    save_dir      : where to save pkl files (default: MODEL_DIR/runs/<timestamp>)
    search_frac   : fraction of training rows to use (default 1.0 = full data).
                    Set to 0.25 during HPO search — hyperparameter rankings are
                    preserved on a subsample, and trials run 4× faster.
    preloaded_data : (train_df, val_df) already in memory — skips _load_panel.
                    Use this in search to avoid re-reading the parquet per trial.
    target_filter : if set, train only this one target and return its metric
                    directly (not composite). Used by per-target HPO search.
    return_metric : metric to return when target_filter is set.
                    "val_ap"    — average precision (good for ranking quality)
                    "val_prec10"— precision in top-10% decile (good for xl targets
                                  where production use is top-pick precision)
                    "val_lift"  — prec10 / base_rate (normalised version of prec10)
                    Default: "val_ap"
    per_target_feature_cols : dict mapping target → feature list (or None).
                    If provided, overrides TARGET_FEATURE_COLS from config for
                    that trial.  None entry means "use all features".
                    Falls back to config.TARGET_FEATURE_COLS when not provided.

    Returns
    -------
    float — composite_ap (all targets) or the requested metric (single target)
    """
    import mlflow

    label_config = label_config or {}
    target_rate_base = label_config.get("target_rate_base", 0.20)
    target_rate_xl   = label_config.get("target_rate_xl",   0.10)

    if label_config:
        labels_path = _ensure_labels(target_rate_base, target_rate_xl)
    else:
        from .config import GOLD_LABELS
        labels_path = GOLD_LABELS

    if preloaded_data is not None:
        train, val = preloaded_data   # already loaded + subsampled by caller
    else:
        train, val = _load_panel(labels_path)
        # Subsample training rows for HPO speed — val stays full for fair scoring
        if search_frac < 1.0:
            n_full = len(train)
            n_keep = max(1, int(n_full * search_frac))
            train = train.sample(n=n_keep, random_state=42).reset_index(drop=True)
            print(f"  [experiment] search subsample: {n_keep:,} / {n_full:,} "
                  f"train rows ({search_frac:.0%})")

    feat_cols = [c for c in train.columns if c not in (["date", "symbol"] + MODEL_TARGETS)]
    X_train = train[["date", "symbol"] + feat_cols]
    X_val   = val[["date", "symbol"] + feat_cols]

    mlflow.set_tracking_uri(f"sqlite:///{MLRUNS_DIR / 'mlflow.db'}")
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    auto_name = run_name or (
        f"{model_type}_b{target_rate_base:.2f}_xl{target_rate_xl:.2f}_"
        + hashlib.md5(json.dumps(model_params, sort_keys=True).encode()).hexdigest()[:6]
    )

    t0 = time.time()
    metrics_by_target: dict[str, dict] = {}

    with mlflow.start_run(run_name=auto_name):
        # Log configuration
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("target_rate_base", target_rate_base)
        mlflow.log_param("target_rate_xl",   target_rate_xl)
        mlflow.log_param("n_train", len(train))
        mlflow.log_param("n_val",   len(val))
        for k, v in model_params.items():
            mlflow.log_param(k, v)

        cls = get_model_class(model_type)

        if save_models and save_dir is None:
            ts = time.strftime("%Y%m%d_%H%M%S")
            save_dir = MODEL_DIR / "runs" / f"{model_type}_{ts}"
            save_dir.mkdir(parents=True, exist_ok=True)

        targets_to_run = [target_filter] if target_filter else MODEL_TARGETS

        # Resolve per-target feature column override:
        # explicit arg > config.TARGET_FEATURE_COLS > None (use all)
        _feat_override = per_target_feature_cols or {}

        for target in targets_to_run:
            if target not in train.columns:
                print(f"  [experiment] skip {target} — not in labels")
                continue

            y_train = train[target].astype(float).values
            y_val   = val[target].astype(float).values

            # Drop rows where label is NaN
            tr_ok = ~np.isnan(y_train)
            va_ok = ~np.isnan(y_val)

            # Per-target feature list: explicit override > config default
            feat_cols = _feat_override.get(target, TARGET_FEATURE_COLS.get(target))

            model = cls(target=target, feature_cols=feat_cols, **model_params)
            m = model.fit(
                X_train[tr_ok], y_train[tr_ok],
                X_val[va_ok],   y_val[va_ok],
            )
            metrics_by_target[target] = m

            # Log per-target metrics with target prefix
            for metric_name, val_ in m.items():
                if isinstance(val_, (int, float)) and not isinstance(val_, bool):
                    mlflow.log_metric(f"{target}_{metric_name}", val_)

            if save_models:
                pkl_path = save_dir / f"{target}_{model_type}.pkl"
                model.save(pkl_path)

            # Free GPU memory after each target — PyTorch's caching allocator
            # keeps freed tensors reserved; with n_jobs=2, stale allocations
            # from completed targets accumulate and cause OOM on target N+1.
            if hasattr(model, "_net") and model._net is not None:
                model._net = None
            del model
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

        composite = _composite_ap(metrics_by_target)
        elapsed = time.time() - t0

        mlflow.log_metric("composite_ap", composite)
        mlflow.log_metric("elapsed_sec", elapsed)

        if target_filter:
            tgt_m = metrics_by_target.get(target_filter, {})
            for mk in ("val_ap", "val_prec10", "val_lift"):
                if mk in tgt_m:
                    mlflow.log_metric(mk, tgt_m[mk])
            print(f"[experiment] {auto_name}  target={target_filter}  "
                  f"val_ap={tgt_m.get('val_ap', float('nan')):.4f}  "
                  f"prec10={tgt_m.get('val_prec10', float('nan')):.4f}  "
                  f"lift={tgt_m.get('val_lift', float('nan')):.3f}  "
                  f"elapsed={elapsed:.0f}s")
        else:
            print(f"[experiment] {auto_name}  composite_ap={composite:.4f}  "
                  f"elapsed={elapsed:.0f}s")

    if target_filter:
        return metrics_by_target.get(target_filter, {}).get(return_metric, float("nan"))
    return composite


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Run a single experiment trial")
    p.add_argument("--model_type", default="lgbm", choices=["lgbm", "ae_mlp", "tabm"])
    p.add_argument("--label_base", type=float, default=0.20)
    p.add_argument("--label_xl",   type=float, default=0.10)
    p.add_argument("--save", action="store_true", help="Save trained models to disk")
    args = p.parse_args()

    score = run_trial(
        model_type=args.model_type,
        model_params={},
        label_config={"target_rate_base": args.label_base,
                      "target_rate_xl":   args.label_xl},
        save_models=args.save,
    )
    print(f"composite_ap = {score:.4f}")
