"""
Per-target feature importance analysis.

Loads prod LGBM models, computes LGBM feature importances (gain), and
for each target prints / optionally writes the recommended TARGET_FEATURE_COLS
subset (top-K features by cumulative gain).

Usage:
    # Print recommended feature sets (no file changes)
    python -m pipeline.analyze_features

    # Print + update config.py TARGET_FEATURE_COLS in-place
    python -m pipeline.analyze_features --update_config

    # Use a specific run dir instead of prod/
    python -m pipeline.analyze_features --run_dir models/runs/lgbm_20260522_123456

Options:
    --top_k N        Keep features covering top N% of cumulative gain (default 90)
    --min_features M Minimum features to keep per target (default 20)
    --update_config  Rewrite TARGET_FEATURE_COLS in pipeline/config.py
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


def load_importances(model_dir: Path) -> dict[str, dict[str, float]]:
    """Load LGBM models from model_dir, return {target: {feature: importance}}."""
    import pickle

    importances: dict[str, dict[str, float]] = {}

    for pkl in sorted(model_dir.glob("*_lgbm.pkl")):
        target = pkl.stem.replace("_lgbm", "")
        try:
            with open(pkl, "rb") as fh:
                p = pickle.load(fh)
            model     = p.get("model")
            feat_cols = p.get("feat_cols", [])
            if model is None or not feat_cols:
                print(f"  [skip] {pkl.name}: missing model or feat_cols")
                continue
            raw = model.booster_.feature_importance(importance_type="gain")
            importances[target] = dict(zip(feat_cols, raw.astype(float)))
            print(f"  {target}: {len(feat_cols)} features  "
                  f"top-3: {sorted(importances[target], key=importances[target].get, reverse=True)[:3]}")
        except Exception as e:
            print(f"  [warn] {pkl.name}: {e}")

    return importances


def top_k_features(importance: dict[str, float],
                   cumulative_pct: float = 90.0,
                   min_features: int = 20) -> list[str]:
    """Return features covering `cumulative_pct`% of total gain."""
    total = sum(importance.values())
    if total == 0:
        return list(importance.keys())

    ranked = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    cutoff = total * cumulative_pct / 100.0
    running = 0.0
    selected: list[str] = []
    for feat, imp in ranked:
        selected.append(feat)
        running += imp
        if running >= cutoff and len(selected) >= min_features:
            break

    # Always include at least min_features
    if len(selected) < min_features and len(ranked) >= min_features:
        selected = [f for f, _ in ranked[:min_features]]

    return selected


def update_config(recommendations: dict[str, list[str]]) -> None:
    """Rewrite TARGET_FEATURE_COLS in pipeline/config.py."""
    config_path = Path(__file__).resolve().parent / "config.py"
    text = config_path.read_text(encoding="utf-8")

    # Build new dict literal
    lines = ["TARGET_FEATURE_COLS: dict[str, list[str] | None] = {"]
    for target, feats in recommendations.items():
        feat_str = ", ".join(f'"{f}"' for f in feats)
        lines.append(f'    "{target}": [{feat_str}],')
    lines.append("}")
    new_block = "\n".join(lines)

    # Replace the old block (from TARGET_FEATURE_COLS: ... to closing })
    pattern = (r'TARGET_FEATURE_COLS: dict\[str, list\[str\] \| None\] = \{[^}]*\}')
    if re.search(pattern, text, re.DOTALL):
        updated = re.sub(pattern, new_block, text, flags=re.DOTALL)
        config_path.write_text(updated, encoding="utf-8")
        print(f"\nconfig.py TARGET_FEATURE_COLS updated ({config_path})")
    else:
        print("\n[warn] Could not find TARGET_FEATURE_COLS block in config.py — "
              "copy the output above manually.")


def main():
    p = argparse.ArgumentParser(description="Per-target LGBM feature importance analysis")
    p.add_argument("--run_dir", type=str, default=None,
                   help="Path to model run dir (default: models/prod/)")
    p.add_argument("--top_k", type=float, default=90.0,
                   help="Cumulative gain %% to cover (default 90)")
    p.add_argument("--min_features", type=int, default=20,
                   help="Minimum features per target (default 20)")
    p.add_argument("--update_config", action="store_true",
                   help="Rewrite TARGET_FEATURE_COLS in config.py")
    args = p.parse_args()

    from .config import MODEL_DIR

    if args.run_dir:
        model_dir = Path(args.run_dir)
    else:
        model_dir = MODEL_DIR / "prod"

    if not model_dir.exists():
        print(f"[error] model dir not found: {model_dir}")
        return

    print(f"\nLoading LGBM models from {model_dir} …")
    importances = load_importances(model_dir)

    if not importances:
        print("[error] No LGBM models found in", model_dir)
        return

    recommendations: dict[str, list[str]] = {}
    print(f"\n=== Per-target feature selection (top {args.top_k:.0f}% cumulative gain, "
          f"min {args.min_features} features) ===\n")

    for target, imp in sorted(importances.items()):
        selected = top_k_features(imp, args.top_k, args.min_features)
        recommendations[target] = selected

        total   = sum(imp.values())
        covered = sum(imp.get(f, 0) for f in selected)
        pct     = covered / total * 100 if total > 0 else 0

        print(f"{target}: {len(selected)} features  ({pct:.1f}% of gain)")
        ranked_all = sorted(imp.items(), key=lambda x: x[1], reverse=True)
        for rank, (feat, gain) in enumerate(ranked_all[:30], 1):
            pct_feat = gain / total * 100 if total > 0 else 0
            marker = " *" if feat in selected else ""
            print(f"  {rank:2d}. {feat:<45s}  gain={gain:10.1f}  ({pct_feat:.2f}%){marker}")
        if len(ranked_all) > 30:
            print(f"  ... {len(ranked_all)-30} more features (not shown)")
        print()

    print("=== Recommended TARGET_FEATURE_COLS (paste into config.py) ===\n")
    print("TARGET_FEATURE_COLS: dict[str, list[str] | None] = {")
    for target, feats in recommendations.items():
        feat_str = ", ".join(f'"{f}"' for f in feats)
        print(f'    "{target}": [{feat_str}],')
    print("}")

    if args.update_config:
        update_config(recommendations)


if __name__ == "__main__":
    main()
