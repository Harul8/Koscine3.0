"""
Automated ML pipeline — one command, fully unattended.

Workflow per label config:
  1.  Build labels with that config's target rates (cached if already built)
  2.  Hyperparameter search (Optuna, reduced budget)
  3.  Train all model types with best params + full budget
  4.  Predict on 2025 val data → evaluate (AP, AUC-ROC, bucket hit rate,
      avg peak return T+3/T+5, confusion matrix)
  5.  Print per-config summary

After all configs:
  6.  Comparison table: every config × model type side-by-side
  7.  Auto-select winner (highest ensemble composite AP on val)

Final:
  8.  Promote winner's models to prod/
  9.  Optimise ensemble blend weights (val data)
  10. Full evaluation on test split

Usage:
    python -m pipeline.automl                         # full run, all 3 label configs
    python -m pipeline.automl --skip_label_sweep      # default config only (fastest)
    python -m pipeline.automl --models tabm ae_mlp    # skip lgbm
    python -m pipeline.automl --trials 15 15 25       # trial counts: tabm ae_mlp lgbm
    python -m pipeline.automl --resume                # skip already-completed studies
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np

from .config import (
    GOLD_LABELS, MLRUNS_DIR, MODEL_DIR,
    SEARCH_LABEL_CONFIGS, SEARCH_AP_WEIGHTS,
)
from .experiment import _ensure_labels
from .search import run_study
from .train import train_model_type, optimise_ensemble_weights, PROD_DIR
from .evaluate import run_evaluation


# ── Defaults ───────────────────────────────────────────────────────────────────

AUTOML_N_TRIALS: dict[str, int] = {
    "tabm":   20,
    "ae_mlp": 20,
    "lgbm":   30,
}

_SEARCH_ONLY_PARAMS = {"n_epochs", "patience", "batch_size"}
# batch_size is tuned during search on the 20%-subsample (240k rows).
# Full training uses 1.2M rows where a larger batch is both faster and
# better-calibrated — class defaults (16384) are used instead.
RESULTS_FILE = MLRUNS_DIR / "automl_results.json"


# ── Helpers ────────────────────────────────────────────────────────────────────

def _cfg_key(cfg: dict) -> str:
    return f"b{cfg['target_rate_base']:.2f}_xl{cfg['target_rate_xl']:.2f}"


def _strip_search_params(params: dict) -> dict:
    """Remove search-budget params so final train uses class defaults (300/25)."""
    return {k: v for k, v in params.items() if k not in _SEARCH_ONLY_PARAMS}


def _phase(title: str) -> None:
    w = 70
    print(f"\n{'='*w}\n  {title}\n{'='*w}")


def _load_results() -> dict:
    if RESULTS_FILE.exists():
        return json.loads(RESULTS_FILE.read_text())
    return {}


def _save_results(data: dict) -> None:
    MLRUNS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_FILE.write_text(json.dumps(data, indent=2, default=str))


# ── Per-config pipeline ────────────────────────────────────────────────────────

def _run_one_config(
    cfg: dict,
    model_types: list[str],
    n_trials: dict[str, int],
    resume: bool,
    saved: dict,
) -> dict:
    """
    Run steps 1-5 for a single label config.
    Returns a result dict that gets stored in automl_results.json.
    """
    key = _cfg_key(cfg)

    # ── Step 1: Labels ─────────────────────────────────────────────────────────
    labels_path = _ensure_labels(cfg["target_rate_base"], cfg["target_rate_xl"])

    # ── Step 2: Hyperparameter search ─────────────────────────────────────────
    _phase(f"CONFIG {key} | Step 2 — Hyperparameter Search")

    search: dict[str, dict] = saved.get(key, {}).get("search", {})
    for mt in model_types:
        if mt in search and resume:
            print(f"[automl] {key}/{mt}: cached  score={search[mt]['best_score']:.4f}")
            continue
        print(f"[automl] searching {mt}  trials={n_trials[mt]} …")
        t0 = time.time()
        r  = run_study(mt, cfg, n_trials[mt])
        search[mt] = {
            "best_score":  r["best_score"],
            "best_params": r["best_params"],
            "n_trials":    r["n_trials"],
            "elapsed_sec": round(time.time() - t0),
        }
        # Persist after each study so we can resume
        saved.setdefault(key, {})["search"] = search
        _save_results(saved)

    # ── Step 3: Train with best params + full budget ───────────────────────────
    _phase(f"CONFIG {key} | Step 3 — Full Training")

    run_dirs: dict[str, str] = saved.get(key, {}).get("run_dirs", {})
    train_metrics: dict[str, dict] = {}

    for mt in model_types:
        if mt in run_dirs and resume and Path(run_dirs[mt]).exists():
            print(f"[automl] {key}/{mt}: skipping train, run_dir exists → {run_dirs[mt]}")
            continue

        raw_params   = search.get(mt, {}).get("best_params", {})
        final_params = _strip_search_params(raw_params)
        print(f"\n[automl] training {mt}  best_params={final_params}")

        result = train_model_type(
            model_type=mt,
            labels_path=labels_path,
            model_params=final_params,
            promote=False,          # DO NOT promote yet — wait until winner is chosen
        )
        run_dirs[mt]    = result["run_dir"]
        train_metrics[mt] = result

        saved.setdefault(key, {})["run_dirs"]      = run_dirs
        saved.setdefault(key, {})["train_metrics"] = train_metrics
        _save_results(saved)

    # ── Step 4: Evaluate on 2025 val data ─────────────────────────────────────
    _phase(f"CONFIG {key} | Step 4 — Evaluate on 2025 Val Data")

    eval_dir = MODEL_DIR / "eval_per_config" / key
    metrics  = run_evaluation(
        split       = "val",
        run_dirs    = {mt: Path(d) for mt, d in run_dirs.items()},
        labels_path = labels_path,
        save_dir    = eval_dir,
    )

    saved.setdefault(key, {})["eval"] = metrics
    _save_results(saved)

    # ── Step 5: Per-config summary ─────────────────────────────────────────────
    _phase(f"CONFIG {key} | Summary")
    _print_config_summary(key, cfg, search, metrics)

    return saved[key]


# ── Comparison and selection ───────────────────────────────────────────────────

def _select_winner(saved: dict, label_configs: list[dict]) -> tuple[dict, str]:
    """Score every config and return (best_cfg, best_key)."""
    _phase("Comparison — All Label Configs vs 2025 Val Data")

    scores: dict[str, float] = {}
    for cfg in label_configs:
        key = _cfg_key(cfg)
        m   = saved.get(key, {}).get("eval", {})
        scores[key] = m.get("composite_ap", float("nan"))

    # Print comparison table
    cols = ["composite_ap", "signal_lift_up", "signal_lift_dn"]
    hdr  = f"{'CONFIG':<22}" + "".join(f"{c:>18}" for c in cols)
    print(f"\n{hdr}")
    print("-" * (22 + 18 * len(cols)))

    valid = {k: v for k, v in scores.items() if not np.isnan(v)}
    best_key = max(valid, key=valid.get) if valid else _cfg_key(label_configs[0])

    for cfg in label_configs:
        key = _cfg_key(cfg)
        m   = saved.get(key, {}).get("eval", {})
        marker = " ← WINNER" if key == best_key else ""
        row = f"{key:<22}"
        for c in cols:
            v = m.get(c, float("nan"))
            row += f"{v:>17.4f} " if not np.isnan(v) else f"{'—':>18}"
        print(row + marker)

    best_cfg = next(c for c in label_configs if _cfg_key(c) == best_key)
    print(f"\n[automl] winner: {best_cfg}  composite_ap={scores.get(best_key, 'nan'):.4f}")
    return best_cfg, best_key


# ── Promotion ──────────────────────────────────────────────────────────────────

def _promote_winner(saved: dict, best_key: str) -> None:
    """Copy winner's model pkls to prod/."""
    _phase("Promote Winner → prod/")
    run_dirs = saved[best_key].get("run_dirs", {})
    PROD_DIR.mkdir(parents=True, exist_ok=True)

    promoted = []
    for mt, run_dir in run_dirs.items():
        for pkl in Path(run_dir).glob(f"*_{mt}.pkl"):
            dest = PROD_DIR / pkl.name
            shutil.copy2(pkl, dest)
            promoted.append(dest.name)
            print(f"  → {dest.name}")

    if not promoted:
        print("[automl] WARNING: no pkls found to promote")


# ── Print helpers ──────────────────────────────────────────────────────────────

def _print_config_summary(
    key: str,
    cfg: dict,
    search: dict,
    metrics: dict,
) -> None:
    print(f"\nLabel config : {cfg}")
    print(f"Composite AP : {metrics.get('composite_ap', float('nan')):.4f}")
    print(f"Signal lift  : UP={metrics.get('signal_lift_up', float('nan')):.2f}x  "
          f"DN={metrics.get('signal_lift_dn', float('nan')):.2f}x")
    print(f"\n{'MODEL':<12} {'SEARCH_AP':>10} {'BEST_PARAMS'}")
    print("-"*70)
    for mt, r in search.items():
        params_str = ", ".join(f"{k}={v}" for k, v in
                               _strip_search_params(r.get("best_params", {})).items())
        print(f"{mt:<12} {r.get('best_score', float('nan')):>10.4f}  {params_str}")


# ── Main ───────────────────────────────────────────────────────────────────────

def run_automl(
    model_types: list[str] | None = None,
    n_trials: dict[str, int] | None = None,
    skip_label_sweep: bool = False,
    resume: bool = False,
    label_base: float | None = None,
    label_xl: float | None = None,
) -> None:
    model_types   = model_types or ["tabm", "ae_mlp", "lgbm"]
    n_trials      = n_trials    or dict(AUTOML_N_TRIALS)

    if label_base is not None or label_xl is not None:
        # Single explicit config — overrides sweep/skip flags
        from .config import SEARCH_LABEL_CONFIGS as _cfgs
        _default = _cfgs[1] if len(_cfgs) > 1 else _cfgs[0]
        label_configs = [{
            "target_rate_base": label_base if label_base is not None else _default["target_rate_base"],
            "target_rate_xl":   label_xl   if label_xl   is not None else _default["target_rate_xl"],
        }]
    elif skip_label_sweep:
        label_configs = [{"target_rate_base": 0.20, "target_rate_xl": 0.10}]
    else:
        label_configs = list(SEARCH_LABEL_CONFIGS)

    # Estimate runtime
    trial_secs = {"tabm": 150, "ae_mlp": 120, "lgbm": 115}
    train_secs = {"tabm": 900, "ae_mlp": 720, "lgbm": 120}
    est_search = sum(n_trials.get(mt, 0) * trial_secs.get(mt, 120) for mt in model_types)
    est_train  = sum(train_secs.get(mt, 120) for mt in model_types)
    est_total  = (est_search + est_train) * len(label_configs)

    _phase("ESN AutoML — Full Pipeline")
    print(f"Models        : {model_types}")
    print(f"Label configs : {len(label_configs)}")
    print(f"Trials        : { {mt: n_trials.get(mt) for mt in model_types} }")
    print(f"Est. runtime  : {est_total/3600:.1f}h  "
          f"(search ~{est_search*len(label_configs)/3600:.1f}h  "
          f"+ train ~{est_train*len(label_configs)/3600:.1f}h)")
    print(f"Results file  : {RESULTS_FILE}")

    t_start = time.time()
    saved   = _load_results() if resume else {}

    # ── Per-config loop (steps 1-5) ────────────────────────────────────────────
    for cfg in label_configs:
        _run_one_config(cfg, model_types, n_trials, resume, saved)

    # ── Select winner (step 6-7) ───────────────────────────────────────────────
    best_cfg, best_key = _select_winner(saved, label_configs)
    saved["winner"] = {"key": best_key, "cfg": best_cfg}
    _save_results(saved)

    # ── Promote winner (step 8) ────────────────────────────────────────────────
    _promote_winner(saved, best_key)

    # ── Ensemble weights (step 9) ──────────────────────────────────────────────
    _phase("Ensemble Weight Optimisation")
    best_labels = _ensure_labels(best_cfg["target_rate_base"], best_cfg["target_rate_xl"])
    optimise_ensemble_weights(labels_path=best_labels)

    # ── Final evaluation on test split (step 10) ──────────────────────────────
    _phase("Final Evaluation — Test Split (out-of-sample)")
    test_metrics = run_evaluation(
        split       = "test",
        labels_path = best_labels,
        save_dir    = MODEL_DIR / "eval_final",
    )
    saved["final_test_metrics"] = test_metrics
    _save_results(saved)

    # ── Done ───────────────────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    _phase(f"DONE  ({elapsed/3600:.1f}h total)")
    print(f"Winner label config : {best_cfg}")
    print(f"Val  composite AP   : {saved[best_key]['eval'].get('composite_ap', 'nan'):.4f}")
    print(f"Test composite AP   : {test_metrics.get('composite_ap', 'nan'):.4f}")
    print(f"Prod models         : {PROD_DIR}")
    print(f"Full results        : {RESULTS_FILE}")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Automated ML: per-config search → train → validate → select → promote"
    )
    p.add_argument("--models", nargs="+", default=["tabm", "ae_mlp", "lgbm"],
                   choices=["tabm", "ae_mlp", "lgbm"])
    p.add_argument("--trials", nargs="+", type=int, default=None,
                   help="Trial counts matching --models order")
    p.add_argument("--skip_label_sweep", action="store_true",
                   help="Only use default label config (0.20/0.10)")
    p.add_argument("--label_base", type=float, default=None,
                   help="Use a single explicit label config (base rate)")
    p.add_argument("--label_xl", type=float, default=None,
                   help="Use a single explicit label config (xl rate)")
    p.add_argument("--resume", action="store_true",
                   help="Resume — skip completed studies and existing run dirs")
    args = p.parse_args()

    n_trials_map = None
    if args.trials:
        if len(args.trials) != len(args.models):
            p.error("--trials must have same length as --models")
        n_trials_map = dict(zip(args.models, args.trials))

    run_automl(
        model_types      = args.models,
        n_trials         = n_trials_map,
        skip_label_sweep = args.skip_label_sweep,
        resume           = args.resume,
        label_base       = args.label_base,
        label_xl         = args.label_xl,
    )
