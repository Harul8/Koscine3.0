from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import pandas as pd
from catboost import CatBoostClassifier

from koscine.config import MODEL_DIR, REPORTS_DIR


def write_feature_importance(model_dir: Path = MODEL_DIR, output_dir: Path = REPORTS_DIR) -> pd.DataFrame:
    rows = []
    for metadata_path in sorted(model_dir.glob("*.json")):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        features = metadata["features"]
        model_file = metadata["model_file"]
        if str(model_file).endswith(".cbm"):
            model = CatBoostClassifier()
            model.load_model(str(model_dir / model_file))
            gain = model.get_feature_importance()
            split = [None] * len(features)
            family = "catboost"
        else:
            model = lgb.Booster(model_file=str(model_dir / model_file))
            gain = model.feature_importance(importance_type="gain")
            split = model.feature_importance(importance_type="split")
            family = "lightgbm"
        for feature, gain_value, split_value in zip(features, gain, split, strict=False):
            rows.append(
                {
                    "model": metadata["name"],
                    "family": family,
                    "label": metadata["label"],
                    "feature": feature,
                    "gain": gain_value,
                    "split": split_value,
                }
            )
    importance = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    importance.to_csv(output_dir / "feature_importance.csv", index=False)
    return importance


def write_prediction_diagnostics(
    predictions: pd.DataFrame,
    output_dir: Path = REPORTS_DIR,
    top_k: int = 100,
) -> dict[str, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    scored_frames = []
    for _, group in predictions.groupby(["side", "threshold"], sort=True):
        pred_col = group["pred_col"].iloc[0]
        label_col = group["label_col"].iloc[0]
        part = group.copy()
        part["score"] = part[pred_col]
        if "actual" not in part:
            part["actual"] = part[label_col]
        scored_frames.append(part)
    scored = pd.concat(scored_frames, ignore_index=True)

    bucket_rows = []
    for (side, threshold), group in scored.groupby(["side", "threshold"], sort=True):
        ranked = group.sort_values("score", ascending=False)
        top = ranked.head(top_k)
        missed = ranked[ranked["actual"].eq(1)].sort_values("score", ascending=True).head(top_k)
        false_positive = ranked[ranked["actual"].eq(0)].head(top_k)
        bucket_rows.append(
            {
                "side": side,
                "threshold": threshold,
                "rows": len(group),
                "actual_events": int(group["actual"].sum()),
                "base_rate": group["actual"].mean(),
                f"precision_at_{top_k}": top["actual"].mean() if len(top) else None,
                "avg_score_top": top["score"].mean() if len(top) else None,
                "avg_return_top": top["fwd_return_5d"].mean() if len(top) else None,
            }
        )
        false_positive.to_csv(
            output_dir / f"false_positives_{side}_{int(threshold * 100)}pct.csv",
            index=False,
        )
        missed.to_csv(
            output_dir / f"missed_events_{side}_{int(threshold * 100)}pct.csv",
            index=False,
        )

    bucket_summary = pd.DataFrame(bucket_rows)
    symbol_summary = (
        scored.groupby(["symbol", "side", "threshold"])
        .agg(
            rows=("actual", "size"),
            events=("actual", "sum"),
            avg_score=("score", "mean"),
            avg_fwd_return=("fwd_return_5d", "mean"),
        )
        .reset_index()
    )
    symbol_summary["event_rate"] = symbol_summary["events"] / symbol_summary["rows"]

    bucket_summary.to_csv(output_dir / "prediction_bucket_diagnostics.csv", index=False)
    symbol_summary.to_csv(output_dir / "prediction_symbol_diagnostics.csv", index=False)
    return {"bucket": bucket_summary, "symbol": symbol_summary}
