"""
Optuna-based hyperparameter search across model types and label configs.

Search strategy:
  Outer loop : label configs (SEARCH_LABEL_CONFIGS — 3 configs)
  Inner loop : one Optuna study per (model_type, label_config)
  Objective  : composite val Average Precision (SEARCH_AP_WEIGHTS)

Studies are persisted in SQLite so they resume automatically if interrupted.

Budgets (configurable via CLI):
  lgbm   : 100 trials per label config
  ae_mlp : 50  trials per label config
  tabm   : 30  trials per label config

Usage:
    # Search all model types across all label configs
    python -m pipeline.search

    # Search only LightGBM, 30 trials
    python -m pipeline.search --model lgbm --n_trials 30

    # Resume interrupted study
    python -m pipeline.search --model ae_mlp --resume
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .config import (
    MLRUNS_DIR, SEARCH_LABEL_CONFIGS, SEARCH_AP_WEIGHTS, MODEL_TARGETS,
)
from .experiment import run_trial



# ── Trial budgets ──────────────────────────────────────────────────────────────
DEFAULT_N_TRIALS: dict[str, int] = {
    "lgbm":   100,
    "ae_mlp":  50,
    "tabm":    30,
}


# ── Hyperparameter search spaces ───────────────────────────────────────────────

def _suggest_lgbm(trial) -> dict[str, Any]:
    # n_estimators is NOT searched — with early_stopping(50) on val AUC the
    # actual tree count is determined by convergence, not this cap.
    # n_estimators comes from LGBM_BASE_PARAMS (set to 4000 for headroom).
    return dict(
        learning_rate     = trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
        num_leaves        = trial.suggest_int("num_leaves", 15, 127),
        min_child_samples = trial.suggest_int("min_child_samples", 30, 300),
        reg_lambda        = trial.suggest_float("reg_lambda", 0.5, 15.0, log=True),
        reg_alpha         = trial.suggest_float("reg_alpha", 0.0, 5.0),
        feature_fraction  = trial.suggest_float("feature_fraction", 0.4, 1.0),
        bagging_fraction  = trial.suggest_float("bagging_fraction", 0.4, 1.0),
        min_split_gain    = trial.suggest_float("min_split_gain", 0.0, 0.5),
    )


def _suggest_ae_mlp(trial) -> dict[str, Any]:
    # enc_dim/mlp_dim ≥ 256 ensures CUDA kernels are large enough to saturate
    # the GPU — tiny configs (128-dim) produce <1ms kernels where Python
    # overhead (1ms/batch) dominates and GPU sits idle.
    return dict(
        enc_dim    = trial.suggest_categorical("enc_dim",    [256, 512]),
        latent_dim = trial.suggest_categorical("latent_dim", [16, 32, 64]),
        mlp_dim    = trial.suggest_categorical("mlp_dim",    [256, 512]),
        dropout    = trial.suggest_float("dropout",   0.05, 0.40),
        noise_std  = trial.suggest_float("noise_std", 0.05, 0.30),
        lr         = trial.suggest_float("lr", 3e-4, 3e-3, log=True),
        batch_size = trial.suggest_categorical("batch_size", [4096, 8192]),
        # Reduced budget for search — final train.py run uses full 300/25
        n_epochs   = 80,
        patience   = 20,
    )


def _suggest_tabm(trial) -> dict[str, Any]:
    # k ≥ 8, hidden ≥ 256 ensures CUDA kernels are large enough to saturate
    # the GPU — k=4/hidden=128 produces 0.27ms kernels vs 1ms Python overhead.
    return dict(
        k          = trial.suggest_categorical("k",          [8, 16]),
        hidden_dim = trial.suggest_categorical("hidden_dim", [256, 512]),
        n_layers   = trial.suggest_int("n_layers", 2, 4),
        dropout    = trial.suggest_float("dropout", 0.05, 0.30),
        lr         = trial.suggest_float("lr", 3e-4, 3e-3, log=True),
        batch_size = trial.suggest_categorical("batch_size", [4096, 8192]),
        # Reduced budget for search — final train.py run uses full 300/25
        n_epochs   = 80,
        patience   = 20,
    )


_SUGGEST_FN = {
    "lgbm":   _suggest_lgbm,
    "ae_mlp": _suggest_ae_mlp,
    "tabm":   _suggest_tabm,
}

# Target groupings for search-space selection
_XL_TARGETS = {"up_5_xl", "dn_5_xl"}   # rare positives (~10%) — most constrained
_DN_TARGETS = {"dn_3", "dn_5"}          # harder to predict — moderately constrained

# Default label config for per-target search (no label-config sweep needed)
_PT_LABEL_CONFIG = {"target_rate_base": 0.20, "target_rate_xl": 0.10}

# Fraction of training rows used per trial (keeps each trial fast)
# xl targets use a higher fraction: rare positives (~10% base rate) need more
# samples per trial to get a stable prec10 estimate for TPE to learn from.
_SEARCH_FRAC = 0.20
_SEARCH_FRAC_XL = 0.50

# Objective metric per target — drives what the Optuna study maximises.
# xl targets: val_prec10 (top-decile precision = production metric for rare events)
# base targets: val_ap (full PR curve = better for ensemble ranking)
_OBJECTIVE_METRIC: dict[str, str] = {
    "up_5_xl": "val_prec10",
    "dn_5_xl": "val_prec10",
}


def _suggest_lgbm_dn(trial) -> dict[str, Any]:
    """
    Moderately constrained space for dn_3 / dn_5.
    Down moves are more panic-driven and harder to predict than up moves
    → smaller trees + stronger regularisation reduces overfitting.
    Space centres on the hand-tuned LGBM_TARGET_PARAMS for dn_3/dn_5.
    """
    return dict(
        learning_rate     = trial.suggest_float("learning_rate", 0.004, 0.025, log=True),
        num_leaves        = trial.suggest_int("num_leaves", 8, 63),
        min_child_samples = trial.suggest_int("min_child_samples", 60, 400),
        reg_lambda        = trial.suggest_float("reg_lambda", 2.0, 15.0, log=True),
        reg_alpha         = trial.suggest_float("reg_alpha", 0.2, 5.0),
        feature_fraction  = trial.suggest_float("feature_fraction", 0.4, 0.85),
        bagging_fraction  = trial.suggest_float("bagging_fraction", 0.4, 0.85),
        min_split_gain    = trial.suggest_float("min_split_gain", 0.0, 0.8),
    )


def _suggest_lgbm_xl(trial) -> dict[str, Any]:
    """
    Most constrained space for up_5_xl / dn_5_xl.
    xl positives are ~10% of data → small trees + strong regularisation to avoid
    chasing noise in rare events.  num_leaves raised from 47 → 63 to allow the
    model to use the new compound interaction features (stretch_beta, etc.) which
    require slightly deeper trees to activate.
    """
    return dict(
        learning_rate     = trial.suggest_float("learning_rate", 0.003, 0.02, log=True),
        num_leaves        = trial.suggest_int("num_leaves", 8, 63),
        min_child_samples = trial.suggest_int("min_child_samples", 80, 500),
        reg_lambda        = trial.suggest_float("reg_lambda", 2.0, 20.0, log=True),
        reg_alpha         = trial.suggest_float("reg_alpha", 0.3, 5.0),
        feature_fraction  = trial.suggest_float("feature_fraction", 0.35, 0.85),
        bagging_fraction  = trial.suggest_float("bagging_fraction", 0.35, 0.85),
        min_split_gain    = trial.suggest_float("min_split_gain", 0.0, 1.0),
    )


# ── Study management ───────────────────────────────────────────────────────────

def _study_db_url() -> str:
    db_path = MLRUNS_DIR / "optuna.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{db_path}"


def _study_name(model_type: str, label_config: dict) -> str:
    b  = label_config["target_rate_base"]
    xl = label_config["target_rate_xl"]
    return f"{model_type}_b{b:.2f}_xl{xl:.2f}"


def run_study(
    model_type: str,
    label_config: dict,
    n_trials: int,
    timeout_sec: int | None = None,
) -> dict:
    """
    Create (or resume) one Optuna study and run n_trials.
    Returns the best trial params + score.
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    suggest_fn = _SUGGEST_FN[model_type]
    name = _study_name(model_type, label_config)

    # ── Preload data once — avoids repeated parquet reads across parallel trials ─
    # Each trial loading 400+ MB parquet + merge is the main bottleneck with n_jobs>1.
    # Preloading here and subsampling once gives all trials the same subset,
    # which is better for TPE (less noise in objective comparisons).
    from .experiment import _ensure_labels, _load_panel
    _labels_path = _ensure_labels(
        label_config["target_rate_base"], label_config["target_rate_xl"]
    )
    _train_full, _val = _load_panel(_labels_path)
    _n_full = len(_train_full)
    _n_keep = max(1, int(_n_full * _SEARCH_FRAC))
    _train_sub = _train_full.sample(n=_n_keep, random_state=42).reset_index(drop=True)
    print(f"[search] data preloaded — subsample: {_n_keep:,} / {_n_full:,} train rows ({_SEARCH_FRAC:.0%})")
    del _train_full   # free full copy, keep subsample only
    _preloaded = (_train_sub, _val)

    def objective(trial):
        params = suggest_fn(trial)
        try:
            score = run_trial(
                model_type=model_type,
                model_params=params,
                label_config=label_config,
                run_name=f"{name}_t{trial.number}",
                preloaded_data=_preloaded,  # no parquet re-read per trial
            )
        except Exception as e:
            print(f"  [search] trial {trial.number} FAILED: {e}")
            score = float("nan")
        finally:
            # Release GPU memory between trials (noop if no GPU)
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
        return score

    study = optuna.create_study(
        study_name=name,
        storage=_study_db_url(),
        direction="maximize",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=10),
    )

    # n_jobs=2 for GPU models: while trial A holds GIL doing sklearn preprocessing
    # (~5s), trial B runs GPU training (GIL released) — GPU stays busy.
    # n_jobs=4 causes 3 threads to block on GIL while 1 preprocesses, leaving
    # GPU idle for long stretches. n_jobs=1 works but wastes CPU/GPU overlap.
    # LightGBM stays at 1 — already saturates CPU cores via its own n_jobs=-1.
    parallel = 2 if model_type in ("tabm", "ae_mlp") else 1

    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout_sec,
        show_progress_bar=True,
        n_jobs=parallel,
    )

    best = study.best_trial
    result = {
        "study_name":   name,
        "model_type":   model_type,
        "label_config": label_config,
        "best_score":   best.value,
        "best_params":  best.params,
        "n_trials":     len(study.trials),
    }
    print(f"\n[search] {name}  best_score={best.value:.4f}  "
          f"params={json.dumps(best.params, indent=2)}")
    return result


# ── Top-level search across all configs ───────────────────────────────────────

def run_full_search(
    model_types: list[str] | None = None,
    n_trials_override: dict[str, int] | None = None,
) -> list[dict]:
    """
    Run Optuna search across all label configs and model types.
    Returns list of per-study results sorted by best_score descending.
    """
    model_types = model_types or list(DEFAULT_N_TRIALS.keys())
    n_trials = dict(DEFAULT_N_TRIALS)
    if n_trials_override:
        n_trials.update(n_trials_override)

    results: list[dict] = []

    for label_cfg in SEARCH_LABEL_CONFIGS:
        print(f"\n{'='*60}")
        print(f"[search] label config: {label_cfg}")
        print(f"{'='*60}")

        for mt in model_types:
            print(f"\n[search] model={mt}  trials={n_trials[mt]}")
            r = run_study(mt, label_cfg, n_trials[mt])
            results.append(r)

    results.sort(key=lambda x: x["best_score"], reverse=True)
    _print_summary(results)
    _save_results(results)
    return results


# ── Per-target search ──────────────────────────────────────────────────────────

def run_per_target_study(
    model_type: str,
    target: str,
    n_trials: int,
    label_config: dict | None = None,
    timeout_sec: int | None = None,
) -> dict:
    """
    Run one Optuna study for a single target.

    Objective metric and search space are chosen automatically per target:
      - up_3 / up_5   : standard space,   objective = val_ap
      - dn_3 / dn_5   : constrained space, objective = val_ap
      - up_5_xl / dn_5_xl : xl space,     objective = val_prec10
                            (top-decile precision is the production metric for
                             rare xl events; val_ap optimisation leads to high
                             AUC but anti-predictive top-decile picks)

    Training subsample: 50% for xl (rare positives need more signal per trial),
    20% for all others.

    Study name: ``{model_type}_{target}_{obj}_pt``
    Resumes automatically if interrupted.
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    label_config = label_config or _PT_LABEL_CONFIG

    # Objective metric — drives what Optuna maximises
    obj_metric = _OBJECTIVE_METRIC.get(target, "val_ap")
    obj_suffix  = "p10" if obj_metric == "val_prec10" else "ap"

    # Search space — three tiers for lgbm
    if model_type == "lgbm" and target in _XL_TARGETS:
        suggest_fn = _suggest_lgbm_xl
    elif model_type == "lgbm" and target in _DN_TARGETS:
        suggest_fn = _suggest_lgbm_dn
    else:
        suggest_fn = _SUGGEST_FN[model_type]

    # Study name encodes model, target, and objective so ap-opt and p10-opt
    # studies are stored separately and best_params_per_target picks correctly.
    name = f"{model_type}_{target}_{obj_suffix}_pt"

    # Subsample fraction — xl needs more data per trial for stable prec10 signal
    frac = _SEARCH_FRAC_XL if target in _XL_TARGETS else _SEARCH_FRAC

    # Preload data once — avoid re-reading parquet per trial
    from .experiment import _ensure_labels, _load_panel
    labels_path = _ensure_labels(
        label_config["target_rate_base"], label_config["target_rate_xl"]
    )
    _train_full, _val = _load_panel(labels_path)
    _n_full = len(_train_full)
    _n_keep = max(1, int(_n_full * frac))
    _train_sub = _train_full.sample(n=_n_keep, random_state=42).reset_index(drop=True)
    del _train_full
    _preloaded = (_train_sub, _val)
    print(f"[search] {name}  objective={obj_metric}  "
          f"subsample: {_n_keep:,} / {_n_full:,} ({frac:.0%})")

    def objective(trial):
        params = suggest_fn(trial)
        try:
            score = run_trial(
                model_type=model_type,
                model_params=params,
                label_config=label_config,
                run_name=f"{name}_t{trial.number}",
                preloaded_data=_preloaded,
                target_filter=target,
                return_metric=obj_metric,
            )
        except Exception as e:
            print(f"  [search] trial {trial.number} FAILED: {e}")
            score = float("nan")
        finally:
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
        return score if not np.isnan(score) else 0.0

    study = optuna.create_study(
        study_name=name,
        storage=_study_db_url(),
        direction="maximize",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=10),
    )
    # Store label config so best_params_per_target can retrieve it without MLflow
    study.set_user_attr("label_config", label_config)

    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout_sec,
        show_progress_bar=True,
        n_jobs=1,   # one model per trial → no benefit from parallel for LGBM
    )

    best = study.best_trial
    result = {
        "study_name":   name,
        "model_type":   model_type,
        "target":       target,
        "label_config": label_config,
        "best_score":   best.value,
        "best_params":  best.params,
        "n_trials":     len(study.trials),
    }
    print(f"\n[search] {name}  best_val_ap={best.value:.4f}  "
          f"params={json.dumps(best.params, indent=2)}")
    return result


def run_per_target_search(
    model_type: str,
    n_trials: int = 100,
    label_config: dict | None = None,
    targets: list[str] | None = None,
) -> list[dict]:
    """
    Run one Optuna study per target, each maximising that target's val_ap.

    Parameters
    ----------
    model_type   : "lgbm", "ae_mlp", or "tabm"
    n_trials     : trials per target (default 100; total = n_trials × n_targets)
    label_config : fixed label config (default _PT_LABEL_CONFIG = 0.20/0.10)
    targets      : subset of MODEL_TARGETS to search (default: all)

    Each study resumes if interrupted. Results saved to mlruns/search_results.json.
    """
    targets = targets or list(MODEL_TARGETS)
    results: list[dict] = []

    for target in targets:
        print(f"\n{'='*60}")
        print(f"[search] per-target  model={model_type}  target={target}  "
              f"trials={n_trials}")
        print(f"{'='*60}")
        r = run_per_target_study(model_type, target, n_trials, label_config)
        results.append(r)

    results.sort(key=lambda x: x["best_score"], reverse=True)
    _print_per_target_summary(results)
    _save_results(results)
    return results


def _print_per_target_summary(results: list[dict]) -> None:
    print("\n" + "="*70)
    print(f"{'TARGET':<20} {'BEST_VAL_AP':>12} {'TRIALS':>7}")
    print("-"*70)
    for r in results:
        print(f"{r['target']:<20} {r['best_score']:>12.4f} {r['n_trials']:>7}")
    print("="*70)


def _print_summary(results: list[dict]) -> None:
    print("\n" + "="*70)
    print(f"{'STUDY':<45} {'SCORE':>8} {'TRIALS':>7}")
    print("-"*70)
    for r in results:
        print(f"{r['study_name']:<45} {r['best_score']:>8.4f} {r['n_trials']:>7}")
    print("="*70)


def _save_results(results: list[dict]) -> None:
    out = MLRUNS_DIR / "search_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"[search] results saved → {out}")


# ── Per-target best-params extractor ──────────────────────────────────────────

def best_params_per_target(model_type: str) -> dict[str, dict]:
    """
    Return per-target best hyperparams from Optuna.

    Priority:
      1. Per-target study (``{model_type}_{target}_pt``) — direct best trial,
         objective was that target's val_ap with the right search space.
      2. Composite studies — find the trial with highest ``{target}_val_ap``
         across all composite runs via MLflow (fallback if no per-target study).

    Returns
    -------
    dict keyed by target name, each value is a hyperparam dict with an extra
    ``"label_config"`` key so train.py can build the right labels file.
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    storage = _study_db_url()
    try:
        all_study_names = set(optuna.get_all_study_names(storage))
    except Exception:
        all_study_names = set()

    result: dict[str, dict] = {}
    targets_needing_fallback: list[str] = []

    # ── Pass 1: per-target studies ─────────────────────────────────────────────
    for target in MODEL_TARGETS:
        obj_metric = _OBJECTIVE_METRIC.get(target, "val_ap")
        obj_suffix = "p10" if obj_metric == "val_prec10" else "ap"
        pt_name    = f"{model_type}_{target}_{obj_suffix}_pt"

        if pt_name not in all_study_names:
            targets_needing_fallback.append(target)
            continue

        study = optuna.load_study(study_name=pt_name, storage=storage)
        completed = [t for t in study.trials if t.state.name == "COMPLETE"]
        if not completed:
            targets_needing_fallback.append(target)
            continue

        best = study.best_trial
        label_cfg = study.user_attrs.get("label_config", _PT_LABEL_CONFIG)
        print(f"  [search] {target}: per-target study ({obj_metric})  "
              f"best={best.value:.4f}")
        result[target] = {**best.params, "label_config": label_cfg}

    if not targets_needing_fallback:
        return result

    # ── Pass 2: composite studies + MLflow fallback ────────────────────────────
    composite_studies = [
        s for s in all_study_names
        if s.startswith(f"{model_type}_") and not s.endswith("_pt")
    ]
    if not composite_studies:
        raise RuntimeError(
            f"No per-target or composite studies found for {model_type!r}.\n"
            f"Run:  python -m pipeline.search --model {model_type} --per_target"
        )

    # Gather all completed composite trials
    all_trials: list[dict] = []
    for sname in composite_studies:
        study = optuna.load_study(study_name=sname, storage=storage)
        parts = sname.replace(f"{model_type}_", "").split("_")
        try:
            b  = float(parts[0].lstrip("b"))
            xl = float(parts[1].lstrip("xl"))
        except (IndexError, ValueError):
            b, xl = 0.20, 0.10
        label_cfg = {"target_rate_base": b, "target_rate_xl": xl}
        for t in study.trials:
            if t.state.name != "COMPLETE":
                continue
            entry = dict(t.params)
            entry["_label_config"]  = label_cfg
            entry["_composite_ap"]  = t.value
            entry["_study_name"]    = sname
            entry["_trial_number"]  = t.number
            all_trials.append(entry)

    if not all_trials:
        raise RuntimeError(f"Composite studies for {model_type} have zero COMPLETE trials.")

    # Enrich with per-target val_ap from MLflow
    import mlflow
    from .config import MLFLOW_EXPERIMENT, MLRUNS_DIR
    mlflow.set_tracking_uri(f"sqlite:///{MLRUNS_DIR / 'mlflow.db'}")
    client = mlflow.tracking.MlflowClient()
    try:
        exp = client.get_experiment_by_name(MLFLOW_EXPERIMENT)
        exp_id = exp.experiment_id if exp else None
    except Exception:
        exp_id = None

    if exp_id:
        runs = client.search_runs(
            experiment_ids=[exp_id],
            filter_string=f"params.model_type = '{model_type}'",
            max_results=5000,
        )
        run_metrics = {
            r.info.run_name: {
                k: v for k, v in r.data.metrics.items() if k.endswith("_val_ap")
            }
            for r in runs if r.info.run_name
        }
        for entry in all_trials:
            rname = f"{entry['_study_name']}_t{entry['_trial_number']}"
            if rname in run_metrics:
                entry.update(run_metrics[rname])

    for target in targets_needing_fallback:
        metric_key = f"{target}_val_ap"
        candidates = [e for e in all_trials if metric_key in e]
        if candidates:
            best_entry = max(candidates, key=lambda e: e[metric_key])
            print(f"  [search] {target}: composite fallback  "
                  f"best_val_ap={best_entry[metric_key]:.4f}  "
                  f"(composite={best_entry.get('_composite_ap', float('nan')):.4f})")
        else:
            best_entry = max(all_trials, key=lambda e: e.get("_composite_ap", -1))
            print(f"  [search] {target}: no per-target AP in MLflow — "
                  f"using composite-best params")

        params = {k: v for k, v in best_entry.items()
                  if not k.startswith("_") and not k.endswith("_val_ap")}
        result[target] = {**params, "label_config": best_entry.get("_label_config", {})}

    return result


def print_per_target_params(model_type: str) -> None:
    """Print per-target best params table for inspection."""
    per_target = best_params_per_target(model_type)
    print(f"\n{'='*70}")
    print(f"Per-target best params for {model_type}")
    print(f"{'='*70}")
    for target, params in per_target.items():
        lcfg = params.pop("label_config", {})
        print(f"\n  {target}  label_config={lcfg}")
        for k, v in params.items():
            print(f"    {k:<25} {v}")
        params["label_config"] = lcfg   # put back


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Optuna hyperparameter search")
    p.add_argument("--model", nargs="+", choices=["lgbm", "ae_mlp", "tabm"],
                   help="Model type(s) to search (default: all)")
    p.add_argument("--n_trials", type=int, default=None,
                   help="Trials per study (default: 100 per-target, 150 composite)")
    p.add_argument("--per_target", action="store_true",
                   help="Run one study per target, each maximising that target's val_ap. "
                        "Uses target-appropriate search space (xl models get constrained space). "
                        "Recommended over composite search.")
    p.add_argument("--targets", nargs="+", choices=list(MODEL_TARGETS),
                   help="Subset of targets for --per_target (default: all 6)")
    p.add_argument("--label_base", type=float, default=None,
                   help="Label config base rate (composite mode only)")
    p.add_argument("--label_xl", type=float, default=None,
                   help="Label config xl rate (composite mode only)")
    args = p.parse_args()

    mt_list = args.model or list(DEFAULT_N_TRIALS.keys())

    if args.per_target:
        # ── Per-target search (recommended) ───────────────────────────────────
        nt = args.n_trials or 100
        lc = None
        if args.label_base is not None or args.label_xl is not None:
            lc = {
                "target_rate_base": args.label_base or _PT_LABEL_CONFIG["target_rate_base"],
                "target_rate_xl":   args.label_xl   or _PT_LABEL_CONFIG["target_rate_xl"],
            }
        for mt in mt_list:
            run_per_target_search(
                model_type=mt,
                n_trials=nt,
                label_config=lc,
                targets=args.targets,
            )
    else:
        # ── Composite search (legacy) ──────────────────────────────────────────
        n_override = None
        if args.n_trials is not None:
            n_override = {mt: args.n_trials for mt in mt_list}

        if args.label_base is not None or args.label_xl is not None:
            from .config import SEARCH_LABEL_CONFIGS as _default_cfgs
            cfg = {
                "target_rate_base": args.label_base or _default_cfgs[1]["target_rate_base"],
                "target_rate_xl":   args.label_xl   or _default_cfgs[1]["target_rate_xl"],
            }
            for mt in mt_list:
                nt = (n_override or {}).get(mt, DEFAULT_N_TRIALS[mt])
                run_study(mt, cfg, nt)
        else:
            run_full_search(
                model_types=args.model,
                n_trials_override=n_override,
            )
