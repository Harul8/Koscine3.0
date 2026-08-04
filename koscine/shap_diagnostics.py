from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from koscine.config import MODEL_DIR, REPORTS_DIR


def gain_based_feature_importance(
    model_dir: Path = MODEL_DIR,
    output_dir: Path = REPORTS_DIR,
    glob_pattern: str = "*.json",
) -> pd.DataFrame:
    rows = []
    for meta_path in sorted(model_dir.glob(glob_pattern)):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if "features" not in meta or "model_file" not in meta:
            continue
        path = model_dir / meta["model_file"]
        if not path.exists():
            continue
        if path.suffix == ".txt":
            try:
                model = lgb.Booster(model_file=str(path))
            except Exception:
                continue
            gain = model.feature_importance(importance_type="gain")
            split = model.feature_importance(importance_type="split")
        else:
            continue
        total_gain = float(gain.sum()) or 1.0
        for feature, g, s in zip(meta["features"], gain, split):
            rows.append({
                "model": meta.get("name", meta_path.stem),
                "label": meta.get("label", ""),
                "feature": feature,
                "gain": float(g),
                "gain_share": float(g) / total_gain,
                "split": int(s),
            })
    df = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "feature_importance_detailed.csv", index=False)
    if not df.empty:
        agg = (
            df.groupby("feature")
            .agg(
                mean_gain_share=("gain_share", "mean"),
                appearances=("model", "nunique"),
                total_splits=("split", "sum"),
            )
            .reset_index()
            .sort_values("mean_gain_share", ascending=False)
        )
        agg.to_csv(output_dir / "feature_importance_aggregated.csv", index=False)
    return df


def suggest_prunable_features(
    importance_df: pd.DataFrame,
    min_gain_share: float = 0.001,
    min_appearances: int = 2,
) -> pd.DataFrame:
    if importance_df.empty:
        return importance_df
    agg = (
        importance_df.groupby("feature")
        .agg(
            mean_gain_share=("gain_share", "mean"),
            appearances=("model", "nunique"),
            max_gain_share=("gain_share", "max"),
        )
        .reset_index()
    )
    weak = agg[
        (agg["mean_gain_share"].lt(min_gain_share))
        & (agg["max_gain_share"].lt(min_gain_share * 3))
    ].sort_values("mean_gain_share")
    return weak


def write_shap_style_report(
    model_dir: Path = MODEL_DIR,
    output_dir: Path = REPORTS_DIR,
    glob_pattern: str = "*.json",
    min_gain_share: float = 0.001,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    detailed = gain_based_feature_importance(
        model_dir=model_dir, output_dir=output_dir, glob_pattern=glob_pattern
    )
    prunable = suggest_prunable_features(detailed, min_gain_share=min_gain_share)
    prunable_path = output_dir / "feature_prune_candidates.csv"
    prunable.to_csv(prunable_path, index=False)
    return {
        "detailed": output_dir / "feature_importance_detailed.csv",
        "aggregated": output_dir / "feature_importance_aggregated.csv",
        "prunable": prunable_path,
    }
