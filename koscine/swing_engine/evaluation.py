from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .contract import SWING_ENGINE_CONTRACT_VERSION, SwingContract, attach_outcomes_to_predictions
from .selector import SIGNAL_LABELS


def _signal_mask(frame: pd.DataFrame) -> pd.Series:
    if "signal_label" not in frame:
        return pd.Series(False, index=frame.index)
    return frame["signal_label"].fillna("").astype(str).isin(SIGNAL_LABELS)


def _summary_row(selected: pd.DataFrame, universe_scored: pd.DataFrame, label: str, extra: dict | None = None) -> dict:
    extra = extra or {}
    evaluated = selected[selected["window_status"].eq("evaluated")]
    n = len(evaluated)
    hit = int(evaluated["hit"].sum()) if n else 0
    near = int(evaluated["near"].sum()) if n else 0
    opposite = int(evaluated["opposite"].sum()) if n else 0
    small = int(evaluated["small"].sum()) if n else 0
    top5_pool = universe_scored[universe_scored["date"].isin(selected["date"].dropna().unique())]
    if "side" in selected and selected["side"].notna().any():
        top5_pool = top5_pool[top5_pool["side"].isin(selected["side"].dropna().unique())]
    top5_available = int(
        top5_pool[top5_pool["top5_mover"].fillna(False)].drop_duplicates(["date", "side", "symbol"]).shape[0]
    )
    top5_hits = int(evaluated["top5_mover"].fillna(False).sum()) if n else 0
    hit_near = hit + near
    hit_near_pct = hit_near / n if n else np.nan
    opposite_pct = opposite / n if n else np.nan
    return {
        **extra,
        "slice": label,
        "signals": int(len(selected)),
        "evaluated": int(n),
        "pending": int(len(selected) - n),
        "dates": int(selected["date"].nunique()) if len(selected) else 0,
        "symbols": int(selected["symbol"].nunique()) if len(selected) else 0,
        "long": int(selected["side"].eq("up").sum()) if "side" in selected else 0,
        "short": int(selected["side"].eq("down").sum()) if "side" in selected else 0,
        "hit_n": hit,
        "hit_pct": hit / n if n else np.nan,
        "near_n": near,
        "near_pct": near / n if n else np.nan,
        "hit_near_n": hit_near,
        "hit_near_pct": hit_near_pct,
        "opposite_n": opposite,
        "opposite_pct": opposite_pct,
        "small_n": small,
        "small_pct": small / n if n else np.nan,
        "avg_favorable_move": float(evaluated["favorable_move"].mean()) if n else np.nan,
        "median_favorable_move": float(evaluated["favorable_move"].median()) if n else np.nan,
        "avg_signed_close_return": float(evaluated["signed_close_return"].mean()) if n else np.nan,
        "avg_net_close_return": float(evaluated["net_close_return"].mean()) if n else np.nan,
        "top5_hits": top5_hits,
        "top5_available": top5_available,
        "top5_capture_pct": top5_hits / top5_available if top5_available else np.nan,
        "daily_signal_avg": len(selected) / max(int(selected["date"].nunique()), 1) if len(selected) else 0.0,
        "max_symbol_concentration_pct": (
            float(selected["symbol"].value_counts(normalize=True).max()) if len(selected) else np.nan
        ),
    }


def summarize_gold(frame: pd.DataFrame, min_hit_near: float = 0.60, hard_opposite_cap: float = 0.25, preferred_opposite_cap: float = 0.20) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    scored = frame.copy()
    scored["date"] = pd.to_datetime(scored["date"], errors="coerce").dt.normalize()
    scored["year"] = scored["date"].dt.year.astype("Int64").astype(str)
    scored["quarter"] = scored["date"].dt.to_period("Q").astype(str)
    scored["month"] = scored["date"].dt.to_period("M").astype(str)
    scored["ALL"] = "ALL"
    signals = scored[_signal_mask(scored)].copy()
    rows = []
    groups = [
        ("aggregate", ["ALL"]),
        ("year", ["year"]),
        ("quarter", ["quarter"]),
        ("month", ["month"]),
        ("side", ["side"]),
        ("tier", ["tier"]),
        ("label", ["signal_label"]),
        ("profile", ["selection_profile"]),
        ("bucket", ["signal_bucket"]),
        ("window", ["retrain_window"]),
        ("model_window", ["train_end"]),
        ("symbol", ["symbol"]),
    ]
    for slice_name, cols in groups:
        if any(col not in signals.columns for col in cols):
            continue
        for key, group in signals.groupby(cols, dropna=False, sort=True):
            key_tuple = key if isinstance(key, tuple) else (key,)
            row = _summary_row(group, scored, slice_name, {"key": "|".join(str(part) for part in key_tuple)})
            row["passes_gold"] = bool(
                row["evaluated"] > 0
                and row["hit_near_pct"] >= min_hit_near
                and row["opposite_pct"] <= hard_opposite_cap
            )
            row["passes_preferred"] = bool(
                row["evaluated"] > 0
                and row["hit_near_pct"] >= min_hit_near
                and row["opposite_pct"] <= preferred_opposite_cap
            )
            rows.append(row)
    return pd.DataFrame(rows)


def write_gold_evaluation(
    predictions: pd.DataFrame,
    output_dir: Path,
    source: str,
    dataset_path: Path | None = None,
    universe_path: Path | None = None,
    contract: SwingContract | None = None,
    extra_manifest: dict | None = None,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    contract = contract or SwingContract()
    enriched = predictions.copy()
    if dataset_path is not None and universe_path is not None and "window_status" not in enriched:
        enriched = attach_outcomes_to_predictions(enriched, dataset_path, universe_path, contract)
    summary = summarize_gold(enriched)
    predictions_path = output_dir / "swing_predictions.parquet"
    summary_path = output_dir / "swing_summary.csv"
    manifest_path = output_dir / "manifest.json"
    enriched.to_parquet(predictions_path, index=False)
    enriched.to_csv(predictions_path.with_suffix(".csv"), index=False)
    summary.to_csv(summary_path, index=False)
    manifest = {
        "contract_version": SWING_ENGINE_CONTRACT_VERSION,
        "source": source,
        "rows": int(len(enriched)),
        "signals": int(_signal_mask(enriched).sum()) if len(enriched) else 0,
        "summary_rows": int(len(summary)),
        "promotion_bar": {
            "min_hit_near": 0.60,
            "preferred_opposite_cap": 0.20,
            "hard_opposite_cap": 0.25,
        },
    }
    if extra_manifest:
        manifest.update(extra_manifest)
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return {"predictions": predictions_path, "summary": summary_path, "manifest": manifest_path}
