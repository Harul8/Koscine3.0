"""
Production training pipeline — trains all MODEL_TARGETS for one model type,
saves to a timestamped run directory, and promotes to prod/ if the composite
val AP beats the current champion.

Usage:
    # Train all three model types (default labels)
    python -m pipeline.train

    # Train a specific model type
    python -m pipeline.train --model lgbm
    python -m pipeline.train --model ae_mlp
    python -m pipeline.train --model tabm

    # Train with custom label config (e.g. from search results)
    python -m pipeline.train --model lgbm --label_base 0.18 --label_xl 0.09

    # Skip promotion check (always saves, never overwrites prod/)
    python -m pipeline.train --model lgbm --no_promote

After training all model types, run ensemble weight optimisation:
    python -m pipeline.train --optimize_ensemble
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    GOLD_FEATURES, GOLD_LABELS, GOLD_DIR, MODEL_DIR, MLRUNS_DIR,
    MLFLOW_EXPERIMENT, MODEL_TARGETS, SEARCH_AP_WEIGHTS,
    TRAIN_END, VAL_END, TARGET_FEATURE_COLS, MODEL_FEATURE_OVERRIDES,
    TABM_TARGET_PARAMS,
    LGBM_BASE_PARAMS,
    # Tiered model config
    GOLD_LABELS_TIERED, TIERED_MODEL_TARGETS, TIERED_OVERLAY_TARGETS,
    LGBM_TIERED_TARGET_PARAMS, CATBOOST_TARGET_PARAMS,
    WRONG_DIR_PENALTY, WRONG_DIR_PENALTY_BY_TARGET, OVERLAY_ALPHA,
    TIERED_BLEND_WEIGHTS,
    # Clean directional model config
    CLEAN_MODEL_TARGETS, LGBM_CLEAN_PARAMS, LGBM_CLEAN_N_SEEDS,
    LGBM_CLEAN_SEED_BUFFER, LGBM_CLEAN_SADDLE_THRESH, LGBM_CLEAN_TARGET_PARAMS,
    LGBM_CLEAN_USE_SPE, LGBM_CLEAN_SPE_ROUNDS,
    LGBM_CLEAN_USE_FOCAL, LGBM_CLEAN_FOCAL_GAMMA,
    CALIBRATION_METHOD,
)
from .models import get_model_class
from .models.ensemble import (
    load_prod_models, predict_ensemble, optimize_weights,
    load_weights, save_weights,
)


# ── Paths ──────────────────────────────────────────────────────────────────────
PROD_DIR    = MODEL_DIR / "prod"
CHAMP_FILE  = MODEL_DIR / "champion.json"   # {model_type: {"score": float, "run_dir": str}}


# ── Data loading ───────────────────────────────────────────────────────────────

def _load_panel(labels_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Join features + labels. Returns (train_df, val_df, test_df).
    All DataFrames include both feature columns and label columns.
    """
    feat = pd.read_parquet(GOLD_FEATURES)
    feat["date"] = pd.to_datetime(feat["date"])

    labs = pd.read_parquet(labels_path)
    labs["date"] = pd.to_datetime(labs["date"])

    panel = feat.merge(labs[["date", "symbol", "split"] + MODEL_TARGETS],
                       on=["date", "symbol"], how="inner")

    train = panel[panel["split"] == "train"].drop(columns=["split"]).reset_index(drop=True)
    val   = panel[panel["split"] == "val"  ].drop(columns=["split"]).reset_index(drop=True)
    test  = panel[panel["split"] == "test" ].drop(columns=["split"]).reset_index(drop=True)

    print(f"[train] panel: train={len(train):,}  val={len(val):,}  test={len(test):,}")
    return train, val, test


# ── Champion tracking ──────────────────────────────────────────────────────────

def _load_champion() -> dict:
    if CHAMP_FILE.exists():
        return json.loads(CHAMP_FILE.read_text())
    return {}


def _save_champion(champ: dict) -> None:
    CHAMP_FILE.parent.mkdir(parents=True, exist_ok=True)
    CHAMP_FILE.write_text(json.dumps(champ, indent=2))


def _composite_ap(metrics_by_target: dict) -> float:
    total = 0.0
    for target, w in SEARCH_AP_WEIGHTS.items():
        ap = metrics_by_target.get(target, {}).get("val_ap", float("nan"))
        if not np.isnan(ap):
            total += w * ap
    return total


# ── Per-type training ──────────────────────────────────────────────────────────

def train_model_type(
    model_type: str,
    labels_path: Path | None = None,
    model_params: dict | None = None,
    per_target_params: dict[str, dict] | None = None,
    per_target_feature_cols: dict[str, list[str] | None] | None = None,
    promote: bool = True,
    targets: list[str] | None = None,
) -> dict:
    """
    Train all MODEL_TARGETS for one model type.

    Parameters
    ----------
    model_type        : "lgbm", "ae_mlp", or "tabm"
    labels_path       : path to labels parquet (defaults to GOLD_LABELS)
    model_params      : shared hyperparam overrides applied to every target
    per_target_params : per-target hyperparam overrides, keyed by target name.
                        If supplied, each target uses
                        ``per_target_params[target]`` (falling back to
                        ``model_params`` for targets not in the dict).
                        The special key ``"label_config"`` inside each target
                        dict is stripped before passing to the model — it is
                        used only to build the correct labels file.
                        Produced by ``pipeline.search.best_params_per_target``.
    per_target_feature_cols : per-target feature column lists, keyed by target.
                        None entry (or absent key) means use config.TARGET_FEATURE_COLS,
                        which itself falls back to all numeric features.
                        Run ``python -m pipeline.analyze_features`` to generate
                        SHAP-based recommendations to populate this.
    promote           : if True, copy models to prod/ when composite AP improves
    targets           : subset of MODEL_TARGETS to train (default: all).
                        When a subset is specified, promotion is skipped
                        automatically (partial composite AP is not comparable).

    Returns
    -------
    result dict with keys: model_type, composite_ap, run_dir, metrics_by_target
    """
    import mlflow

    labels_path = labels_path or GOLD_LABELS
    model_params = model_params or {}
    per_target_params = per_target_params or {}

    # If per_target_params supplies a label_config per target, each target may
    # need a different labels file.  Build a map target → labels_path.
    _target_labels: dict[str, Path] = {}
    if per_target_params:
        from .experiment import _ensure_labels
        for t, tp in per_target_params.items():
            lc = tp.get("label_config")
            if lc:
                _target_labels[t] = _ensure_labels(
                    lc["target_rate_base"], lc["target_rate_xl"]
                )

    train, val, test = _load_panel(labels_path)

    feat_cols = [c for c in train.columns if c not in (["date", "symbol"] + MODEL_TARGETS)]
    X_train = train[["date", "symbol"] + feat_cols]
    X_val   = val[["date", "symbol"] + feat_cols]

    ts = time.strftime("%Y%m%d_%H%M%S")
    run_dir = MODEL_DIR / "runs" / f"{model_type}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(f"sqlite:///{MLRUNS_DIR / 'mlflow.db'}")
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    run_name = f"prod_{model_type}_{ts}"

    metrics_by_target: dict[str, dict] = {}
    cls = get_model_class(model_type)

    print(f"\n{'='*60}")
    print(f"[train] model_type={model_type}  run_dir={run_dir}")
    print(f"{'='*60}")
    t0 = time.time()

    with mlflow.start_run(run_name=run_name, tags={"stage": "prod"}):
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("labels_path", str(labels_path))
        mlflow.log_param("n_train", len(train))
        mlflow.log_param("n_val",   len(val))
        for k, v in model_params.items():
            mlflow.log_param(k, v)
        if per_target_params:
            mlflow.log_param("per_target_params", "true")

        # Per-target feature col resolution:
        #   explicit caller arg > MODEL_FEATURE_OVERRIDES[model_type] > TARGET_FEATURE_COLS
        _feat_override = per_target_feature_cols or {}
        _model_feat_defaults = {
            **TARGET_FEATURE_COLS,
            **MODEL_FEATURE_OVERRIDES.get(model_type, {}),
        }

        _active_targets = targets if targets else MODEL_TARGETS
        for target in _active_targets:
            if target not in train.columns:
                continue

            # Resolve per-target hyperparams:
            #   explicit per_target_params > model-type built-in defaults > shared model_params
            if target in per_target_params:
                t_params = {k: v for k, v in per_target_params[target].items()
                            if k != "label_config"}
            elif model_type == "tabm" and target in TABM_TARGET_PARAMS:
                t_params = {**model_params, **TABM_TARGET_PARAMS[target]}
            else:
                t_params = model_params

            # Resolve per-target feature list
            feat_cols = _feat_override.get(target, _model_feat_defaults.get(target))

            # If this target has its own label file, reload y from it
            if target in _target_labels and _target_labels[target] != labels_path:
                _t_panel = _load_panel(_target_labels[target])
                _t_train, _t_val = _t_panel[0], _t_panel[1]
                y_train = _t_train[target].astype(float).values if target in _t_train.columns else train[target].astype(float).values
                y_val   = _t_val[target].astype(float).values   if target in _t_val.columns   else val[target].astype(float).values
            else:
                y_train = train[target].astype(float).values
                y_val   = val[target].astype(float).values

            tr_ok = ~np.isnan(y_train)
            va_ok = ~np.isnan(y_val)

            print(f"\n  [{target}] train={tr_ok.sum():,}  val={va_ok.sum():,}")
            if t_params != model_params:
                print(f"  [{target}] using per-target params: {t_params}")

            # ── Seed-ensemble training for tabm ────────────────────────────
            # n_seeds is stripped from t_params (TabMModel doesn't accept it).
            # If model_type != "tabm", n_seeds is ignored.
            n_seeds = t_params.pop("n_seeds", 1) if model_type == "tabm" else 1

            if model_type == "tabm" and n_seeds > 1:
                import gc
                import torch
                from sklearn.metrics import average_precision_score, roc_auc_score
                seed_probs = []
                for s in range(n_seeds):
                    print(f"  [{target}] seed {s+1}/{n_seeds}")
                    sub = cls(target=target, feature_cols=feat_cols, seed=s, **t_params)
                    sub.fit(X_train[tr_ok], y_train[tr_ok],
                            X_val[va_ok],   y_val[va_ok])
                    sub.save(run_dir / f"{target}_{model_type}_seed{s}.pkl")
                    seed_probs.append(sub.predict_proba(X_val[va_ok]))
                    # Free GPU memory before next seed — prevents allocator
                    # fragmentation that can stall later seeds.
                    del sub
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                # Averaged metrics across seeds
                avg_probs = np.stack(seed_probs).mean(axis=0)
                yv = y_val[va_ok]
                p10_idx = max(1, int(len(avg_probs) * 0.10))
                top = np.argsort(avg_probs)[::-1][:p10_idx]
                m = {
                    "val_auc":    roc_auc_score(yv, avg_probs),
                    "val_ap":     average_precision_score(yv, avg_probs),
                    "val_prec10": yv[top].mean(),
                }
                print(f"  [{target}] SEED-ENSEMBLE  val_auc={m['val_auc']:.3f}  "
                      f"val_ap={m['val_ap']:.3f}  prec@10%={m['val_prec10']:.2%}")
            else:
                # Single-model path (lgbm, ae_mlp, or tabm with n_seeds=1)
                model = cls(target=target, feature_cols=feat_cols, **t_params)
                m = model.fit(
                    X_train[tr_ok], y_train[tr_ok],
                    X_val[va_ok],   y_val[va_ok],
                )
                pkl_path = run_dir / f"{target}_{model_type}.pkl"
                model.save(pkl_path)

            metrics_by_target[target] = m

            for metric_name, metric_val in m.items():
                if isinstance(metric_val, (int, float)) and not isinstance(metric_val, bool):
                    mlflow.log_metric(f"{target}_{metric_name}", metric_val)

        composite = _composite_ap(metrics_by_target)
        elapsed   = time.time() - t0

        mlflow.log_metric("composite_ap", composite)
        mlflow.log_metric("elapsed_sec", elapsed)

    print(f"\n[train] composite_ap={composite:.4f}  elapsed={elapsed:.0f}s")

    # ── Promotion logic ────────────────────────────────────────────────────────
    if targets and set(targets) != set(MODEL_TARGETS):
        promote = False
        print(f"[train] subset training ({targets}) — skipping promotion, use --no_promote explicitly or train all targets")

    if promote:
        champ = _load_champion()
        prev_score = champ.get(model_type, {}).get("score", -1.0)

        if composite > prev_score:
            print(f"[train] NEW CHAMPION  {model_type}  "
                  f"{prev_score:.4f} → {composite:.4f}  promoting to prod/")
            PROD_DIR.mkdir(parents=True, exist_ok=True)
            for pkl in run_dir.glob(f"*_{model_type}.pkl"):
                dest = PROD_DIR / pkl.name
                dest.write_bytes(pkl.read_bytes())
                print(f"  → {dest.name}")
            champ[model_type] = {"score": composite, "run_dir": str(run_dir)}
            _save_champion(champ)
        else:
            print(f"[train] no improvement  {model_type}  "
                  f"{prev_score:.4f} → {composite:.4f}  prod/ unchanged")

    return {
        "model_type":        model_type,
        "composite_ap":      composite,
        "run_dir":           str(run_dir),
        "metrics_by_target": metrics_by_target,
    }


# ── Ensemble weight optimisation ───────────────────────────────────────────────

def optimise_ensemble_weights(labels_path: Path | None = None) -> None:
    """
    Load all prod models, grid-search blend weights on val data, save to
    prod/ensemble_weights.json.
    """
    labels_path = labels_path or GOLD_LABELS

    if not PROD_DIR.exists() or not any(PROD_DIR.glob("*.pkl")):
        print("[train] no prod models found — run train first")
        return

    _, val, _ = _load_panel(labels_path)
    feat_cols  = [c for c in val.columns if c not in (["date", "symbol"] + MODEL_TARGETS)]
    X_val      = val[["date", "symbol"] + feat_cols]
    y_val      = {t: val[t].astype(float).values for t in MODEL_TARGETS if t in val.columns}

    print("[train] loading prod models for ensemble optimisation …")
    models = load_prod_models(PROD_DIR)

    print("[train] searching ensemble weights …")
    weights = optimize_weights(models, X_val, y_val)
    save_weights(weights, PROD_DIR)

    # Score the ensemble with optimised weights
    scores = predict_ensemble(models, X_val, weights=weights)
    from sklearn.metrics import average_precision_score
    total = 0.0
    for target, w in SEARCH_AP_WEIGHTS.items():
        if target in scores and target in y_val:
            yt = y_val[target]
            ok = ~np.isnan(yt)
            if ok.sum() > 0 and len(np.unique(yt[ok])) > 1:
                ap = average_precision_score(yt[ok], scores[target][ok])
                total += w * ap
    print(f"[train] ensemble composite_ap (val) = {total:.4f}")


# ── Tiered training (v5) ───────────────────────────────────────────────────────

def _load_tiered_panel() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Join features with tiered labels.
    Returns (train_df, val_df, test_df) — each includes feature columns,
    tier, all tiered label cols, bad_up/bad_dn overlay cols, and fwd_close_ret.
    """
    feat = pd.read_parquet(GOLD_FEATURES)
    feat["date"] = pd.to_datetime(feat["date"])

    label_cols = (TIERED_MODEL_TARGETS + TIERED_OVERLAY_TARGETS
                  + CLEAN_MODEL_TARGETS + ["tier", "fwd_close_ret"])
    labs = pd.read_parquet(GOLD_LABELS_TIERED)
    labs["date"] = pd.to_datetime(labs["date"])
    # Only include clean columns that are actually present in the parquet
    # (guards against a stale parquet that pre-dates the clean label build)
    available_label_cols = [c for c in label_cols if c in labs.columns]

    panel = feat.merge(
        labs[["date", "symbol", "split"] + available_label_cols],
        on=["date", "symbol"],
        how="inner",
    )
    train = panel[panel["split"] == "train"].drop(columns=["split"]).reset_index(drop=True)
    val   = panel[panel["split"] == "val"  ].drop(columns=["split"]).reset_index(drop=True)
    test  = panel[panel["split"] == "test" ].drop(columns=["split"]).reset_index(drop=True)

    print(f"[train_tiered] panel: train={len(train):,}  val={len(val):,}  test={len(test):,}")
    liq_tr = (train["tier"] == "liquid").sum()
    print(f"[train_tiered] liquid: train={liq_tr:,}  rest={len(train)-liq_tr:,}")
    return train, val, test


def _wrong_dir_weights(
    y: np.ndarray,
    bad_direction: np.ndarray,
    penalty: float = WRONG_DIR_PENALTY,
    target: str | None = None,
) -> np.ndarray:
    """
    Sample weights that penalise wrong-direction errors:
      - Positive labels (target hit):          weight = 1.0
      - Negative, right-direction miss:        weight = 1.0
      - Negative, wrong direction (bad_close): weight = penalty

    bad_direction is the overlay label (bad_up for up models, bad_dn for dn models).
    It is 1 where the stock closed strongly against the direction — exactly the
    errors we want to suppress.

    Per-target overrides are applied from WRONG_DIR_PENALTY_BY_TARGET
    (e.g. dn_liq gets 8× to suppress its historically high wrong-direction rate).
    """
    if target and target in WRONG_DIR_PENALTY_BY_TARGET:
        penalty = WRONG_DIR_PENALTY_BY_TARGET[target]
    weights = np.ones(len(y), dtype=np.float32)
    # Only penalise false-positive risk: rows that are negative in the target
    # label AND flagged as wrong-direction closes.
    wrong_dir_mask = (y == 0) & (bad_direction == 1)
    weights[wrong_dir_mask] = penalty
    return weights


def _train_tiered_target(
    target: str,
    tier: str,
    train_df: pd.DataFrame,
    val_df:   pd.DataFrame,
    run_dir:  Path,
    feat_cols_override: list[str] | None = None,
    train_overlay: bool = True,
) -> dict:
    """
    Train LGBM + CatBoost ensemble for one (target, tier) pair.
    Optionally also train the bad-close overlay for that direction.

    Returns a metrics dict with keys:
        lgbm_<target>, catboost_<target>,
        ensemble_<target> (averaged calibrated probs),
        [overlay_<bad_lbl>] if train_overlay=True
    """
    from sklearn.metrics import average_precision_score

    all_metrics: dict[str, dict] = {}

    # ── Filter to correct tier ────────────────────────────────────────────────
    tr = train_df[train_df["tier"] == tier].copy()
    va = val_df[val_df["tier"]   == tier].copy()

    feat_cols_meta = ["date", "symbol", "tier"] + TIERED_MODEL_TARGETS + TIERED_OVERLAY_TARGETS + ["fwd_close_ret"]
    non_feat = set(feat_cols_meta)

    all_feat = [c for c in tr.columns if c not in non_feat]
    X_train  = tr[["date", "symbol"] + all_feat]
    X_val    = va[["date", "symbol"] + all_feat]

    y_train = tr[target].astype(float).values
    y_val   = va[target].astype(float).values

    tr_ok = ~np.isnan(y_train)
    va_ok = ~np.isnan(y_val)

    print(f"\n  [{target}][{tier}] train={tr_ok.sum():,}  val={va_ok.sum():,}")

    # ── Wrong-direction sample weights ───────────────────────────────────────
    # bad_up for up models, bad_dn for dn models
    bad_col = "bad_up" if target.startswith("up") else "bad_dn"
    bad_arr = tr[bad_col].fillna(0).astype(float).values
    sw = _wrong_dir_weights(y_train, bad_arr, target=target)
    sw_valid = sw[tr_ok]
    eff_penalty = WRONG_DIR_PENALTY_BY_TARGET.get(target, WRONG_DIR_PENALTY)
    print(f"  [{target}] wrong_dir_penalty={eff_penalty:.1f}x")

    # ── Feature list resolution ───────────────────────────────────────────────
    feat_cols = feat_cols_override or TARGET_FEATURE_COLS.get(target)

    # ── LGBM ─────────────────────────────────────────────────────────────────
    from .models.lgbm_model import LGBMModel
    import lightgbm as lgb
    import warnings

    lgbm_params = dict(LGBM_TIERED_TARGET_PARAMS.get(target, {}))
    lgbm_model = LGBMModel(target=target, feature_cols=feat_cols, **lgbm_params)

    lgbm_model._feat_cols = lgbm_model._numeric_feat_cols(X_train, feat_cols)
    lgbm_model._base_rate = float(np.nanmean(y_train[tr_ok]))
    pos_w = (1 - lgbm_model._base_rate) / (lgbm_model._base_rate + 1e-7)

    p = dict(LGBM_BASE_PARAMS)
    p.update(LGBM_TIERED_TARGET_PARAMS.get(target, {}))
    p["scale_pos_weight"] = pos_w

    from .models.lgbm_model import _device as _lgbm_device
    p["device_type"] = _lgbm_device()

    X_tr_clean = lgbm_model._clean(X_train[lgbm_model._feat_cols])
    X_va_clean = lgbm_model._clean(X_val[lgbm_model._feat_cols])

    lgbm_clf = lgb.LGBMClassifier(**p)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lgbm_clf.fit(
            X_tr_clean[tr_ok], y_train[tr_ok],
            sample_weight=sw_valid,
            eval_set=[(X_va_clean[va_ok], y_val[va_ok])],
            callbacks=[lgb.early_stopping(50, verbose=False),
                       lgb.log_evaluation(500)],
        )

    lgbm_model._model = lgbm_clf
    from .calibration import make_calibrator
    from .config import CALIBRATION_METHOD
    lgbm_model._calibrator = make_calibrator(target, CALIBRATION_METHOD)
    raw_lgbm_val = lgbm_clf.predict_proba(X_va_clean[va_ok])[:, 1]
    lgbm_model._calibrator.fit(raw_lgbm_val, y_val[va_ok])
    lgbm_probs_val = lgbm_model._calibrator.predict(raw_lgbm_val)

    lgbm_model.save(run_dir / f"{target}_lgbm.pkl")
    lgbm_metrics = lgbm_model._metrics(y_val[va_ok], lgbm_probs_val)
    print(f"  [{target}] lgbm     ap={lgbm_metrics['val_ap']:.3f}  "
          f"prec@10%={lgbm_metrics['val_prec10']:.2%}  lift={lgbm_metrics['val_lift']:.2f}x")
    all_metrics[f"lgbm_{target}"] = lgbm_metrics

    # ── CatBoost ──────────────────────────────────────────────────────────────
    from .models.catboost_model import CatBoostModel
    cb_model = CatBoostModel(target=target, feature_cols=feat_cols)
    cb_metrics = cb_model.fit(
        X_train[tr_ok], y_train[tr_ok],
        X_val[va_ok],   y_val[va_ok],
        sample_weight=sw_valid,
    )
    cb_model.save(run_dir / f"{target}_catboost.pkl")
    cb_probs_val = cb_model.predict_proba(X_val[va_ok])
    print(f"  [{target}] catboost ap={cb_metrics['val_ap']:.3f}  "
          f"prec@10%={cb_metrics['val_prec10']:.2%}  lift={cb_metrics['val_lift']:.2f}x")
    all_metrics[f"catboost_{target}"] = cb_metrics

    # ── Ensemble (weighted average of calibrated probs) ───────────────────────
    lgbm_w, cb_w = TIERED_BLEND_WEIGHTS.get(target, (0.5, 0.5))
    total_w = lgbm_w + cb_w
    lgbm_w_n = lgbm_w / total_w
    cb_w_n   = cb_w  / total_w
    ens_probs = lgbm_w_n * lgbm_probs_val + cb_w_n * cb_probs_val
    ens_metrics = lgbm_model._metrics(y_val[va_ok], ens_probs)
    print(f"  [{target}] ensemble ap={ens_metrics['val_ap']:.3f}  "
          f"prec@10%={ens_metrics['val_prec10']:.2%}  lift={ens_metrics['val_lift']:.2f}x  "
          f"blend=lgbm:{lgbm_w_n:.0%}/cb:{cb_w_n:.0%}  ← primary score")
    all_metrics[f"ensemble_{target}"] = ens_metrics

    # ── Overlay (bad-close predictor) ─────────────────────────────────────────
    if train_overlay:
        bad_lbl = "bad_up" if target.startswith("up") else "bad_dn"
        y_bad_tr  = tr[bad_lbl].astype(float).values
        y_bad_va  = va[bad_lbl].astype(float).values
        bad_tr_ok = ~np.isnan(y_bad_tr)
        bad_va_ok = ~np.isnan(y_bad_va)

        overlay_feat = TARGET_FEATURE_COLS.get(bad_lbl)

        print(f"\n  [{bad_lbl}][{tier}] overlay train={bad_tr_ok.sum():,}  val={bad_va_ok.sum():,}")

        # LGBM overlay
        ov_lgbm = LGBMModel(target=bad_lbl, feature_cols=overlay_feat)
        ov_lgbm._feat_cols = ov_lgbm._numeric_feat_cols(X_train, overlay_feat)
        ov_lgbm._base_rate = float(np.nanmean(y_bad_tr[bad_tr_ok]))
        ov_pos_w = (1 - ov_lgbm._base_rate) / (ov_lgbm._base_rate + 1e-7)

        ov_p = dict(LGBM_BASE_PARAMS)
        ov_p.update(LGBM_TIERED_TARGET_PARAMS.get(bad_lbl, {}))
        ov_p["scale_pos_weight"] = ov_pos_w
        ov_p["device_type"] = _lgbm_device()

        X_tr_ov = ov_lgbm._clean(X_train[ov_lgbm._feat_cols])
        X_va_ov = ov_lgbm._clean(X_val[ov_lgbm._feat_cols])

        ov_clf = lgb.LGBMClassifier(**ov_p)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ov_clf.fit(
                X_tr_ov[bad_tr_ok], y_bad_tr[bad_tr_ok],
                eval_set=[(X_va_ov[bad_va_ok], y_bad_va[bad_va_ok])],
                callbacks=[lgb.early_stopping(50, verbose=False),
                           lgb.log_evaluation(500)],
            )
        ov_lgbm._model = ov_clf
        ov_lgbm._calibrator = make_calibrator(bad_lbl, CALIBRATION_METHOD)
        raw_ov = ov_clf.predict_proba(X_va_ov[bad_va_ok])[:, 1]
        ov_lgbm._calibrator.fit(raw_ov, y_bad_va[bad_va_ok])

        # CatBoost overlay
        ov_cb = CatBoostModel(target=bad_lbl, feature_cols=overlay_feat)
        ov_cb_m = ov_cb.fit(
            X_train[bad_tr_ok], y_bad_tr[bad_tr_ok],
            X_val[bad_va_ok],   y_bad_va[bad_va_ok],
        )

        # Save overlay models with tier suffix
        ov_lgbm.save(run_dir / f"{bad_lbl}_{tier}_lgbm.pkl")
        ov_cb.save(  run_dir / f"{bad_lbl}_{tier}_catboost.pkl")

        # Overlay ensemble metrics
        ov_probs = 0.5 * ov_lgbm._calibrator.predict(raw_ov) + 0.5 * ov_cb.predict_proba(X_val[bad_va_ok])
        ov_metrics = ov_lgbm._metrics(y_bad_va[bad_va_ok], ov_probs)
        print(f"  [{bad_lbl}][{tier}] overlay ap={ov_metrics['val_ap']:.3f}  "
              f"prec@10%={ov_metrics['val_prec10']:.2%}")
        all_metrics[f"overlay_{bad_lbl}_{tier}"] = ov_metrics

    return all_metrics


def _train_clean_target(
    target: str,
    train_df: pd.DataFrame,
    val_df:   pd.DataFrame,
    run_dir:  Path,
) -> dict:
    """
    Train an 8-seed LGBM ensemble for one clean directional target.

    Tier-asymmetric design:
      - clean_*_liq  → filter to liquid tier (4% threshold, base rate ~12%)
      - clean_*_rest → filter to rest tier   (7% threshold, base rate ~9%)
    Saddle filter only applied to liquid bull (clean_up_5_liq); other targets
    train cleanly with full 8 seeds (AP metric handles class imbalance).

    Key differences from _train_tiered_target:
      - metric = average_precision (not auc)
      - NO scale_pos_weight (causes iter=2 saddle with ~12% positive rate)
      - 8-seed ensemble
      - LGBM only (no CatBoost) — AP metric + no rebalancing already trains well
      - Platt calibrator on ensemble-averaged val probs

    Parameters
    ----------
    target   : one of CLEAN_MODEL_TARGETS
    train_df : tiered labels panel (already joined with features)
    val_df   : validation panel
    run_dir  : directory to save .pkl files

    Returns
    -------
    dict with keys: val_ap, val_prec10, val_lift, n_seeds_used, tier
    """
    import lightgbm as lgb
    import warnings
    from sklearn.metrics import average_precision_score
    from .models.lgbm_model import LGBMModel
    from .calibration import make_calibrator

    # ── Derive tier + direction from target name ─────────────────────────────
    is_liq   = target.endswith("_liq")
    is_bull  = "_up_" in target
    tier_key = "liquid" if is_liq else "rest"
    # Saddle filter applied to ALL 4 clean targets (threshold=LGBM_CLEAN_SADDLE_THRESH).
    # AP metric with ~8-12% base rate causes early saddle collapses across all targets;
    # the filter was previously on for liquid bull only but the training logs showed
    # collapsed seeds in all other targets too (best_iter < 10 in several cases).
    saddle_filter = True

    # Filter to the relevant tier
    tr = train_df[train_df["tier"] == tier_key].copy()
    va = val_df[val_df["tier"]   == tier_key].copy()

    feat_cols_meta = (
        ["date", "symbol", "tier"] +
        TIERED_MODEL_TARGETS + TIERED_OVERLAY_TARGETS + CLEAN_MODEL_TARGETS +
        ["fwd_close_ret"]
    )
    non_feat = set(feat_cols_meta)
    all_feat = [c for c in tr.columns if c not in non_feat]

    y_train = tr[target].astype(float).values
    y_val   = va[target].astype(float).values

    tr_ok = ~np.isnan(y_train)
    va_ok = ~np.isnan(y_val)

    if tr_ok.sum() == 0 or va_ok.sum() == 0:
        print(f"  [{target}] no valid rows — skipping")
        return {}

    base_tr = float(y_train[tr_ok].mean())
    base_va = float(y_val[va_ok].mean())
    print(f"\n  [{target}] train={tr_ok.sum():,} (pos={base_tr:.2%})  "
          f"val={va_ok.sum():,} (pos={base_va:.2%})")

    # Resolve feature columns (from config.TARGET_FEATURE_COLS)
    feat_cols = TARGET_FEATURE_COLS.get(target)

    # Build a temporary LGBMModel to resolve numeric feature columns
    _proxy = LGBMModel(target=target, feature_cols=feat_cols)
    X_train_df = tr[["date", "symbol"] + all_feat]
    X_val_df   = va[["date", "symbol"] + all_feat]
    resolved   = _proxy._numeric_feat_cols(X_train_df, feat_cols)
    print(f"  [{target}] {len(resolved)} features")

    def _clean(df):
        return df[resolved].astype(float).replace([np.inf, -np.inf], np.nan)

    X_tr = _clean(X_train_df)
    X_va = _clean(X_val_df)

    # ── Per-target param overrides (softer trees for rest targets) ───────────
    target_overrides = LGBM_CLEAN_TARGET_PARAMS.get(target, {})
    base_p = {**LGBM_CLEAN_PARAMS, **target_overrides}

    # ── Focal loss helpers ────────────────────────────────────────────────────
    def _focal_obj(y_pred: np.ndarray, dtrain) -> tuple:
        """Focal loss objective — set via params['objective'] in LightGBM 4.x.
        LightGBM still calls fobj(y_pred, train_dataset) internally even in 4.x;
        only the fobj= kwarg on lgb.train() was removed.
        """
        y_true = dtrain.get_label()
        p = 1.0 / (1.0 + np.exp(-y_pred))
        p_clipped = np.clip(p, 1e-7, 1 - 1e-7)
        gamma = LGBM_CLEAN_FOCAL_GAMMA
        p_t = np.where(y_true == 1, p_clipped, 1 - p_clipped)
        focal_w = (1.0 - p_t) ** gamma
        # Gradient: focal_w × CE gradient
        grad = focal_w * (p - y_true)
        # Hessian: working approximation — focal_w × p(1−p)
        hess = np.maximum(focal_w * p_clipped * (1.0 - p_clipped), 1e-7)
        return grad, hess

    def _ap_eval(y_pred: np.ndarray, dtrain) -> tuple:
        """AP eval metric for lgb.train() feval (needed with custom fobj)."""
        from sklearn.metrics import average_precision_score
        y_true = dtrain.get_label()
        # y_pred are raw logits when using custom fobj
        p = 1.0 / (1.0 + np.exp(-y_pred))
        ap = average_precision_score(y_true, p)
        return "average_precision", ap, True   # higher = better

    # ── Seed ensemble or SPE ──────────────────────────────────────────────────
    n_train = LGBM_CLEAN_N_SEEDS + LGBM_CLEAN_SEED_BUFFER

    raw_seeds: list[tuple[int, np.ndarray]] = []

    if LGBM_CLEAN_USE_SPE:
        # Self-Paced Ensemble: each round trains on balanced subset where
        # "hard" negatives (near decision boundary) are progressively up-weighted.
        # Addresses the 9-12% imbalance without scale_pos_weight saddle collapse.
        pos_idx = np.where(tr_ok & (y_train == 1))[0]
        neg_idx = np.where(tr_ok & (y_train == 0))[0]
        n_pos   = len(pos_idx)
        n_neg   = len(neg_idx)
        # Uniform difficulty weights to start (epoch 0)
        difficulty = np.ones(n_neg, dtype=float) / n_neg
        # Params for SPE: standard binary+AUC (balanced subsets → no saddle)
        spe_p = {k: v for k, v in base_p.items()
                 if k not in ("metric", "objective")}
        spe_p["objective"] = "binary"
        spe_p["metric"]    = "auc"
        print(f"  [{target}] SPE: {n_pos:,} pos  {n_neg:,} neg  "
              f"{LGBM_CLEAN_SPE_ROUNDS + LGBM_CLEAN_SEED_BUFFER} rounds")
        rng = np.random.default_rng(42)
        n_rounds = LGBM_CLEAN_SPE_ROUNDS + LGBM_CLEAN_SEED_BUFFER
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for s in range(n_rounds):
                # Sample n_pos negatives weighted by difficulty
                n_sample = min(n_pos, n_neg)
                sampled_neg = rng.choice(neg_idx, size=n_sample,
                                         replace=False, p=difficulty)
                sample_idx = np.concatenate([pos_idx, sampled_neg])
                rng.shuffle(sample_idx)
                y_s = y_train[sample_idx]
                X_s = X_tr.iloc[sample_idx]
                p_s = dict(spe_p, random_state=42 + s)
                if LGBM_CLEAN_USE_FOCAL:
                    # Use lgb.train() for focal obj; wrap arrays as Dataset
                    ds_tr = lgb.Dataset(X_s, label=y_s, free_raw_data=False)
                    ds_va = lgb.Dataset(X_va.iloc[va_ok], label=y_val[va_ok],
                                        reference=ds_tr, free_raw_data=False)
                    lgb_p = {k: v for k, v in p_s.items()
                             if k not in ("n_estimators", "objective", "metric",
                                          "n_jobs", "verbose", "random_state")}
                    lgb_p.update({"num_threads": -1, "verbose": -1,
                                  "seed": 42 + s})
                    lgb_p["objective"] = _focal_obj   # LightGBM 4.x API
                    booster = lgb.train(
                        lgb_p,
                        ds_tr,
                        num_boost_round=base_p.get("n_estimators", 4000),
                        feval=_ap_eval,
                        valid_sets=[ds_va],
                        callbacks=[
                            lgb.early_stopping(150, verbose=False),
                            lgb.log_evaluation(period=0),
                        ],
                    )
                    best_iter = booster.best_iteration
                    # Predict with sigmoid applied to raw scores
                    raw_scores_va = booster.predict(X_va.iloc[va_ok],
                                                    raw_score=True)
                    val_probs = 1.0 / (1.0 + np.exp(-raw_scores_va))
                    # Update difficulty from predictions on ALL negatives
                    raw_neg = booster.predict(X_tr.iloc[neg_idx], raw_score=True)
                    neg_preds = 1.0 / (1.0 + np.exp(-raw_neg))
                else:
                    clf = lgb.LGBMClassifier(**p_s)
                    clf.fit(
                        X_s, y_s,
                        eval_set=[(X_va.iloc[va_ok], y_val[va_ok])],
                        callbacks=[lgb.early_stopping(150, verbose=False),
                                   lgb.log_evaluation(period=0)],
                    )
                    best_iter  = clf.best_iteration_
                    val_probs  = clf.predict_proba(X_va.iloc[va_ok])[:, 1]
                    neg_preds  = clf.predict_proba(X_tr.iloc[neg_idx])[:, 1]
                # Update difficulty: harder negatives = higher predicted pos prob
                diff_raw   = neg_preds.astype(float)
                diff_raw   = diff_raw - diff_raw.min() + 1e-8   # shift > 0
                difficulty = diff_raw / diff_raw.sum()
                raw_seeds.append((best_iter, val_probs))
        mode_str = "SPE+focal" if LGBM_CLEAN_USE_FOCAL else "SPE"
    else:
        # Original fixed-seed approach (no SPE)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for s in range(n_train):
                if LGBM_CLEAN_USE_FOCAL:
                    p_s = {k: v for k, v in base_p.items()
                           if k not in ("n_estimators", "objective", "metric",
                                        "n_jobs", "verbose")}
                    p_s.update({"num_threads": -1, "verbose": -1, "seed": 42 + s})
                    p_s["objective"] = _focal_obj   # LightGBM 4.x API
                    ds_tr = lgb.Dataset(X_tr.iloc[tr_ok], label=y_train[tr_ok],
                                        free_raw_data=False)
                    ds_va = lgb.Dataset(X_va.iloc[va_ok], label=y_val[va_ok],
                                        reference=ds_tr, free_raw_data=False)
                    booster = lgb.train(
                        p_s,
                        ds_tr,
                        num_boost_round=base_p.get("n_estimators", 4000),
                        feval=_ap_eval,
                        valid_sets=[ds_va],
                        callbacks=[lgb.early_stopping(150, verbose=False),
                                   lgb.log_evaluation(period=0)],
                    )
                    raw_sc = booster.predict(X_va.iloc[va_ok], raw_score=True)
                    val_probs = 1.0 / (1.0 + np.exp(-raw_sc))
                    raw_seeds.append((booster.best_iteration, val_probs))
                else:
                    p = dict(base_p, random_state=42 + s)
                    clf = lgb.LGBMClassifier(**p)
                    clf.fit(
                        X_tr.iloc[tr_ok], y_train[tr_ok],
                        eval_set=[(X_va.iloc[va_ok], y_val[va_ok])],
                        eval_metric="average_precision",
                        callbacks=[
                            lgb.early_stopping(stopping_rounds=150, verbose=False),
                            lgb.log_evaluation(period=0),
                        ],
                    )
                    raw_seeds.append((clf.best_iteration_,
                                      clf.predict_proba(X_va.iloc[va_ok])[:, 1]))
        mode_str = "focal" if LGBM_CLEAN_USE_FOCAL else "standard"

    # Saddle filter: drop seeds that collapsed (best_iter < threshold).
    # Fallback: if fewer than 4 survive, progressively relax threshold.
    N_MIN = 4
    thresh = LGBM_CLEAN_SADDLE_THRESH
    kept = [r for r in raw_seeds if r[0] >= thresh]
    if len(kept) < N_MIN:
        thresh = 10
        kept = [r for r in raw_seeds if r[0] >= thresh]
    if len(kept) < N_MIN:
        kept = sorted(raw_seeds, key=lambda r: r[0], reverse=True)[:N_MIN]
    if len(kept) > LGBM_CLEAN_N_SEEDS:
        kept = sorted(kept, key=lambda r: r[0], reverse=True)[:LGBM_CLEAN_N_SEEDS]
    n_dropped = len(raw_seeds) - len(kept)
    print(f"  [{target}] {mode_str} saddle filter (thresh={thresh}): "
          f"kept {len(kept)}/{len(raw_seeds)} seeds "
          f"(dropped {n_dropped} with best_iter<{thresh})")

    best_iters = [r[0] for r in kept]
    print(f"  [{target}] best_iters: {best_iters}")

    p_val = np.mean([r[1] for r in kept], axis=0)

    # ── Calibration on ensemble-averaged val probs ────────────────────────────
    cal = make_calibrator(target, CALIBRATION_METHOD)
    cal.fit(p_val, y_val[va_ok])
    p_val_cal = cal.predict(p_val)

    # ── Val metrics ───────────────────────────────────────────────────────────
    val_ap   = average_precision_score(y_val[va_ok], p_val_cal)
    k10      = max(1, int(len(p_val_cal) * 0.10))
    top10    = np.argsort(p_val_cal)[::-1][:k10]
    prec10   = float(y_val[va_ok][top10].mean())
    lift10   = prec10 / base_va if base_va > 0 else float("nan")
    print(f"  [{target}] val_ap={val_ap:.3f}  prec@10%={prec10:.2%}  lift={lift10:.2f}x")

    # ── Save: re-train ensemble using ALL seeds (for prod, no saddle filter needed
    #         here because we're saving individual models — load_prod_models will
    #         average them). Save as <target>_clean_lgbm_seed<N>.pkl ────────────
    #
    # We only need one pickled object — build a lightweight wrapper that holds
    # the ensemble of per-seed models + calibrator for predict.py to consume.
    # Re-use LGBMModel as the container; pack all seed classifiers inside.
    container = LGBMModel(target=target, feature_cols=feat_cols)
    container._feat_cols  = resolved
    container._base_rate  = base_tr
    container._calibrator = cal

    # ── Re-train final prod models at fixed depth ─────────────────────────────
    # We retrain each kept seed on the FULL training set (no SPE sampling) with
    # n_estimators fixed at best_iter — this is the saved prod model ensemble.
    # When focal loss is enabled, the final models also use focal loss so that
    # predict.py scores are on the same scale as the calibration was fitted on.
    use_focal_final = LGBM_CLEAN_USE_FOCAL
    final_clfs   = []
    final_is_booster = use_focal_final   # flag for predict_proba patching
    final_target_overrides = LGBM_CLEAN_TARGET_PARAMS.get(target, {})

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i, (best_iter, _) in enumerate(kept):
            if use_focal_final:
                # Re-train as lgb.Booster with focal obj at fixed num_boost_round
                fp = {k: v for k, v in {**LGBM_CLEAN_PARAMS,
                                         **final_target_overrides}.items()
                      if k not in ("n_estimators", "objective", "metric",
                                   "n_jobs", "verbose")}
                fp.update({"num_threads": -1, "verbose": -1, "seed": 42 + i})
                fp["objective"] = _focal_obj   # LightGBM 4.x API
                ds_f = lgb.Dataset(X_tr.iloc[tr_ok], label=y_train[tr_ok],
                                   free_raw_data=False)
                b = lgb.train(
                    fp, ds_f,
                    num_boost_round=max(best_iter, 1),
                )
                final_clfs.append(b)
            else:
                p_final = dict(LGBM_CLEAN_PARAMS, **final_target_overrides,
                               random_state=42 + i,
                               n_estimators=max(best_iter, 1))
                clf_f = lgb.LGBMClassifier(**p_final)
                clf_f.fit(X_tr.iloc[tr_ok], y_train[tr_ok])
                final_clfs.append(clf_f)

    # Monkey-patch predict_proba on the container to average final_clfs.
    # Handles both lgb.Booster (focal; raw_score + sigmoid) and LGBMClassifier.
    import types
    _is_booster = final_is_booster

    def _ensemble_predict_proba(self, X):
        if hasattr(X, "columns"):
            Xc = self._clean(X[self._feat_cols])
        else:
            Xc = X
        if self._is_booster:
            raw = np.mean(
                [c.predict(Xc, raw_score=True) for c in self._clfs], axis=0
            )
            probs = 1.0 / (1.0 + np.exp(-raw))
        else:
            probs = np.mean([c.predict_proba(Xc)[:, 1] for c in self._clfs],
                            axis=0)
        if self._calibrator is not None:
            probs = self._calibrator.predict(probs)
        return probs

    container._clfs = final_clfs
    container._is_booster = _is_booster
    container._clean = lambda df: (
        df[resolved].astype(float).replace([np.inf, -np.inf], np.nan)
    )
    container.predict_proba = types.MethodType(_ensemble_predict_proba, container)

    out_path = run_dir / f"{target}_clean_lgbm.pkl"
    container.save(out_path)
    print(f"  [{target}] saved -> {out_path.name}")

    # ── Save val-tuned thresholds for predict.py Layer-3 gating ─────────────
    # Compute percentile thresholds from val calibrated probs so predict.py can
    # apply the same gates without needing to see the val set at inference time.
    import json as _json
    thresholds = {
        "p95": float(np.percentile(p_val_cal, 95)),
        "p97": float(np.percentile(p_val_cal, 97)),
        "p99": float(np.percentile(p_val_cal, 99)),
        "base_rate_val": float(base_va),
    }
    thr_path = run_dir / f"{target}_thresholds.json"
    thr_path.write_text(_json.dumps(thresholds, indent=2))
    print(f"  [{target}] val thresholds: p95={thresholds['p95']:.4f}  "
          f"p97={thresholds['p97']:.4f}  p99={thresholds['p99']:.4f}")

    return {
        "val_ap":       val_ap,
        "val_prec10":   prec10,
        "val_lift":     lift10,
        "n_seeds_used": len(kept),
        "tier":         tier_key,
    }


def train_tiered(
    labels_path: Path | None = None,
    promote: bool = True,
    train_overlay: bool = True,
) -> dict:
    """
    Train the v5 two-family model architecture:

      Liquid tier  (top 30)  → up_liq / dn_liq  (4% threshold)
      Rest tier    (rest)    → up_rest / dn_rest (8% threshold)

    Each target gets an LGBM + CatBoost ensemble with wrong-direction
    sample weighting.  Optionally trains bad-close overlay models.

    Returns dict with all metrics.
    """
    import mlflow, time

    labels_path = labels_path or GOLD_LABELS_TIERED
    train_df, val_df, test_df = _load_tiered_panel()

    ts      = time.strftime("%Y%m%d_%H%M%S")
    run_dir = MODEL_DIR / "runs" / f"tiered_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(f"sqlite:///{MLRUNS_DIR / 'mlflow.db'}")
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    all_metrics: dict = {}
    t0 = time.time()

    # Map each tiered target to its tier
    _target_tier = {
        "up_liq": "liquid", "dn_liq": "liquid",
        "up_rest": "rest",  "dn_rest": "rest",
    }

    with mlflow.start_run(run_name=f"tiered_{ts}", tags={"stage": "tiered"}):
        mlflow.log_param("labels_path", str(labels_path))
        mlflow.log_param("wrong_dir_penalty", WRONG_DIR_PENALTY)
        mlflow.log_param("overlay_alpha",     OVERLAY_ALPHA)
        mlflow.log_param("n_train", len(train_df))
        mlflow.log_param("n_val",   len(val_df))

        for target in TIERED_MODEL_TARGETS:
            tier = _target_tier[target]
            m = _train_tiered_target(
                target=target,
                tier=tier,
                train_df=train_df,
                val_df=val_df,
                run_dir=run_dir,
                train_overlay=train_overlay,
            )
            all_metrics.update(m)

            # Log primary ensemble metric to MLflow
            ens_key = f"ensemble_{target}"
            if ens_key in m:
                for metric_name, val in m[ens_key].items():
                    if isinstance(val, (int, float)) and not isinstance(val, bool):
                        mlflow.log_metric(f"{target}_{metric_name}", val)

        # ── Clean directional models (Layer 1+2) ──────────────────────────────
        print(f"\n{'='*60}")
        print(" CLEAN DIRECTIONAL MODELS (Layer 1+2 path-asymmetry features)")
        print(f"{'='*60}")
        for target in CLEAN_MODEL_TARGETS:
            if target not in train_df.columns:
                print(f"  [{target}] column not found in panel — skipping "
                      f"(run pipeline.labels tiered first)")
                continue
            cm = _train_clean_target(
                target=target,
                train_df=train_df,
                val_df=val_df,
                run_dir=run_dir,
            )
            if cm:
                all_metrics[f"clean_{target}"] = cm
                for metric_name, val in cm.items():
                    if isinstance(val, (int, float)) and not isinstance(val, bool):
                        mlflow.log_metric(f"{target}_{metric_name}", val)

        elapsed = time.time() - t0
        mlflow.log_metric("elapsed_sec", elapsed)

    print(f"\n[train_tiered] done  elapsed={elapsed:.0f}s  run_dir={run_dir}")
    print(f"\n{'='*60}")
    print(" TIERED ENSEMBLE SUMMARY (val)")
    print(f"{'='*60}")
    for target in TIERED_MODEL_TARGETS:
        ek = f"ensemble_{target}"
        if ek in all_metrics:
            m = all_metrics[ek]
            print(f"  {target:<10}  ap={m['val_ap']:.3f}  "
                  f"prec@10%={m['val_prec10']:.2%}  lift={m['val_lift']:.2f}x")

    print(f"\n{'='*60}")
    print(" CLEAN DIRECTIONAL MODEL SUMMARY (val)")
    print(f"{'='*60}")
    for target in CLEAN_MODEL_TARGETS:
        ck = f"clean_{target}"
        if ck in all_metrics:
            m = all_metrics[ck]
            print(f"  {target:<20}  ap={m['val_ap']:.3f}  "
                  f"prec@10%={m['val_prec10']:.2%}  lift={m['val_lift']:.2f}x  "
                  f"seeds={m['n_seeds_used']}")

    # ── Save per-target blend weights for predict.py ─────────────────────────
    import json
    blend_weights: dict[str, dict[str, float]] = {}
    for t in TIERED_MODEL_TARGETS:
        lgbm_w, cb_w = TIERED_BLEND_WEIGHTS.get(t, (0.5, 0.5))
        total_w = lgbm_w + cb_w
        blend_weights[t] = {"lgbm": round(lgbm_w / total_w, 4),
                             "catboost": round(cb_w  / total_w, 4)}
    (run_dir / "tiered_blend_weights.json").write_text(json.dumps(blend_weights, indent=2))

    # ── Save metadata for predict.py ──────────────────────────────────────────
    meta = {
        "run_dir":       str(run_dir),
        "timestamp":     ts,
        "overlay_alpha": OVERLAY_ALPHA,
        "wrong_dir_penalty": WRONG_DIR_PENALTY,
        "metrics": {k: {mk: float(mv) if isinstance(mv, (int, float)) else mv
                        for mk, mv in v.items()}
                    for k, v in all_metrics.items()},
    }
    (run_dir / "tiered_meta.json").write_text(json.dumps(meta, indent=2))

    if promote:
        PROD_DIR.mkdir(parents=True, exist_ok=True)
        for pkl in run_dir.glob("*.pkl"):
            dest = PROD_DIR / pkl.name
            dest.write_bytes(pkl.read_bytes())
        # Copy metadata, blend weights, and clean model thresholds
        meta_dest = PROD_DIR / "tiered_meta.json"
        meta_dest.write_text(json.dumps(meta, indent=2))
        blend_dest = PROD_DIR / "tiered_blend_weights.json"
        blend_dest.write_text(json.dumps(blend_weights, indent=2))
        for thr_json in run_dir.glob("*_thresholds.json"):
            (PROD_DIR / thr_json.name).write_bytes(thr_json.read_bytes())
        n_pkls = len(list(run_dir.glob("*.pkl")))
        print(f"[train_tiered] promoted {n_pkls} model files → {PROD_DIR}")

    return {"run_dir": str(run_dir), "metrics": all_metrics}


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Production training")
    p.add_argument("--tiered", action="store_true",
                   help="Train the v5 tiered two-family model "
                        "(liquid 4%% / rest 8%%) with LGBM+CatBoost ensemble + overlay")
    p.add_argument("--no_overlay", action="store_true",
                   help="Skip bad-close overlay training (tiered mode only)")
    p.add_argument("--model", nargs="+", default=["lgbm", "ae_mlp", "tabm"],
                   choices=["lgbm", "ae_mlp", "tabm"],
                   help="Model type(s) to train (default: all three)")
    p.add_argument("--label_base", type=float, default=None,
                   help="Custom label target_rate_base")
    p.add_argument("--label_xl",   type=float, default=None,
                   help="Custom label target_rate_xl")
    p.add_argument("--no_promote", action="store_true",
                   help="Do not promote to prod/ even if score improves")
    p.add_argument("--optimize_ensemble", action="store_true",
                   help="After training, optimise ensemble blend weights")
    p.add_argument("--per_target", action="store_true",
                   help="Use per-target best params from Optuna search "
                        "(reads mlruns/optuna.db, picks best val_ap per target)")
    p.add_argument("--targets", nargs="+", default=None,
                   choices=list(MODEL_TARGETS),
                   help="Train only these targets (default: all). "
                        "Promotion to prod/ is skipped automatically for subsets.")
    args = p.parse_args()

    # ── Tiered mode ──────────────────────────────────────────────────────────
    if args.tiered:
        train_tiered(
            promote=not args.no_promote,
            train_overlay=not args.no_overlay,
        )
        if args.optimize_ensemble:
            print("[train] --optimize_ensemble is not applicable in tiered mode; skipped")
        import sys; sys.exit(0)

    labels_path = None
    if args.label_base is not None or args.label_xl is not None:
        from .experiment import _ensure_labels
        from .config import LABEL_TARGET_RATE_BASE, LABEL_TARGET_RATE_XL
        b  = args.label_base or LABEL_TARGET_RATE_BASE
        xl = args.label_xl   or LABEL_TARGET_RATE_XL
        labels_path = _ensure_labels(b, xl)

    for mt in args.model:
        per_target_params = None
        if args.per_target:
            from .search import best_params_per_target
            print(f"[train] extracting per-target best params for {mt} …")
            per_target_params = best_params_per_target(mt)

        train_model_type(
            model_type=mt,
            labels_path=labels_path,
            per_target_params=per_target_params,
            promote=not args.no_promote,
            targets=args.targets,
        )

    if args.optimize_ensemble:
        optimise_ensemble_weights(labels_path=labels_path)
