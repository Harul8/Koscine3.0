from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from koscine.config import HORIZON_DAYS, MODEL_DIR, PREDICTIONS_DIR, PROCESSED_DIR, REPORTS_DIR
from koscine.tiered_clean_direction import (
    TieredCleanConfig,
    label_col,
    opposite_label_col,
    prepare_tiered_frame,
)


LONG_RESCUE_ROOT = MODEL_DIR / "long_rescue"
RESCUE_LABEL = "RGO+"
RESCUE_BUCKET = "PROD_LONG_RESCUE_GO_PLUS"
SIGNAL_LABELS = {"GO+", "GO", RESCUE_LABEL}

CURATED_FEATURES = [
    "ret_1d", "ret_3d", "ret_5d", "ret_10d", "ret_20d", "rel_ret_5d_vs_nifty",
    "ret_5d_cs_rank", "ret_20d_cs_rank",
    "close_sma20_dist", "close_sma50_dist", "close_sma100_dist", "close_sma200_dist",
    "ema_20_slope_5d", "ema_50_slope_5d", "adx_14", "above_50ma", "above_200ma",
    "atr_pct_14", "atr_pct_14_cs_rank", "atr_pct_14_rank_60d",
    "bb_width_20", "bb_width_20_cs_rank", "bb_width_20_rank_60d", "bb_width_20_rank_252d",
    "tight_range_10d", "tight_range_10d_rank_252d",
    "path_range_5d", "path_range_10d", "path_range_20d",
    "volume_dryup_score", "vol_sma20_ratio", "vol_sma20_ratio_cs_rank",
    "delivery_pct", "delivery_pct_chg_5", "delivery_pct_rank_252d",
    "fut_oi_chg_5", "fut_chg_oi_chg_5", "fut_oi_z_60d", "oi_buildup_ratio",
    "oi_acceleration", "price_oi_divergence",
    "atm_iv", "atm_iv_rank_252d", "iv_vs_hv", "iv_vs_hv_rank_252d",
    "pcr_oi", "pcr_vol", "pcr_oi_rank_252d", "pcr_vol_rank_252d",
    "gap_pct", "gap_up_count_10d", "gap_down_count_10d", "gap_both_count_20d",
    "mkt_pct_above_sma20", "mkt_pct_above_sma50", "nifty_ret_5d", "nifty_idx_ret_20d",
    "sector_rel_ret_5d", "sector_rel_ret_20d",
    "low_breakdown_3d", "low_breakdown_5d", "low_breakdown_10d",
    "high_breakout_3d", "high_breakout_5d", "high_breakout_10d",
    "close_position_10", "close_position_20", "upper_wick_pct", "lower_wick_pct",
    "dist_high_20", "dist_low_20", "past_range_10d", "past_range_20d",
]


@dataclass(frozen=True)
class RescueSpec:
    model_id: str
    side: str
    expert: str
    archetype_col: str
    model_file: str
    seed: int
    threshold_quantile: float = 0.99


RESCUE_SPECS = [
    RescueSpec("liquid30_up_4pct_5d", "up", "unified_any", "arch_any", "liquid30_up_unified_any_lgbm.txt", 1902),
    RescueSpec("rest35_up_7pct_5d", "up", "reversal", "arch_reversal", "rest35_up_reversal_lgbm.txt", 1907),
]


def has_rescue_label(label: object) -> bool:
    return RESCUE_LABEL in str(label or "")


def is_signal_label(label: object) -> bool:
    text = str(label or "")
    return text in SIGNAL_LABELS or has_rescue_label(text)


def combine_with_rescue_label(label: object) -> str:
    text = str(label or "").strip()
    if not text or text == "WATCH":
        return RESCUE_LABEL
    if has_rescue_label(text):
        return text
    if text in {"GO+", "GO"}:
        return f"{text}/{RESCUE_LABEL}"
    return RESCUE_LABEL


def _num(frame: pd.DataFrame, col: str, default: float = np.nan) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(default, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[col], errors="coerce")


def add_long_rescue_archetype_flags(frame: pd.DataFrame, side: str = "up") -> pd.DataFrame:
    out = frame.copy()
    ret_20 = _num(out, "ret_20d")
    ret_10 = _num(out, "ret_10d")
    ret_1 = _num(out, "ret_1d")
    sma50 = _num(out, "close_sma50_dist")
    sma200 = _num(out, "close_sma200_dist")
    bb_cs = _num(out, "bb_width_20_cs_rank")
    bb_60 = _num(out, "bb_width_20_rank_60d")
    bb_252 = _num(out, "bb_width_20_rank_252d")
    tight_252 = _num(out, "tight_range_10d_rank_252d")
    path_10 = _num(out, "path_range_10d")
    volume_dryup = _num(out, "volume_dryup_score")
    vol_ratio = _num(out, "vol_sma20_ratio")
    close_pos20 = _num(out, "close_position_20")

    compression = (
        bb_cs.le(0.35)
        | bb_60.le(0.40)
        | bb_252.le(0.40)
        | tight_252.le(0.40)
        | (path_10.le(0.09) & volume_dryup.le(1.05))
    )
    compression &= vol_ratio.le(1.35) | volume_dryup.le(1.15)

    if side == "up":
        reversal = (
            (ret_20.le(-0.04) | ret_10.le(-0.025) | sma50.le(-0.03) | sma200.le(-0.05))
            & ret_1.ge(-0.025)
            & close_pos20.le(0.65)
        )
    else:
        reversal = pd.Series(False, index=out.index)

    out["arch_reversal"] = reversal.fillna(False)
    out["arch_compression_breakout"] = compression.fillna(False)
    out["arch_quiet_breakdown"] = False
    out["arch_any"] = out[["arch_reversal", "arch_compression_breakout", "arch_quiet_breakdown"]].any(axis=1)
    return out


def _build_spec_frame(df: pd.DataFrame, spec) -> pd.DataFrame:
    cols = ["date", "symbol"] + [c for c in CURATED_FEATURES if c in df.columns]
    needed = [
        f"future_{HORIZON_DAYS}d_date",
        f"up_move_{HORIZON_DAYS}d",
        f"down_move_{HORIZON_DAYS}d",
        f"fwd_return_{HORIZON_DAYS}d",
        label_col(spec),
        opposite_label_col(spec),
    ]
    cols += [c for c in needed if c in df.columns and c not in cols]
    frame = df[df["symbol"].isin(spec.predict_symbols)][cols].copy()
    frame["model_id"] = spec.model_id
    frame["side"] = spec.side
    frame["threshold"] = spec.threshold
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    frame["actual_move"] = _num(frame, f"{spec.side}_move_{HORIZON_DAYS}d")
    close_return = _num(frame, f"fwd_return_{HORIZON_DAYS}d")
    frame["signed_close_return_5d"] = close_return if spec.side == "up" else -close_return
    frame["actual_hit"] = frame[label_col(spec)].fillna(False).astype(bool)
    frame["actual_opposite"] = frame[opposite_label_col(spec)].fillna(False).astype(bool)
    frame["evaluated"] = frame["actual_move"].notna() & frame[f"future_{HORIZON_DAYS}d_date"].notna()
    frame = add_long_rescue_archetype_flags(frame, spec.side)
    frame["opposite_flag"] = frame["actual_opposite"] | frame["signed_close_return_5d"].lt(0)
    frame["near_hit_flag"] = frame["actual_move"].ge(frame["threshold"] * 0.5)
    frame["move_rank"] = frame.groupby("date")["actual_move"].rank(method="first", ascending=False)
    frame["is_top5_mover"] = frame["move_rank"].le(5)
    frame["good_top5"] = (
        frame["is_top5_mover"]
        & frame["evaluated"]
        & frame["signed_close_return_5d"].gt(0)
        & ~frame["opposite_flag"]
    ).astype(int)
    return frame


def _prep_xy(train: pd.DataFrame, valid: pd.DataFrame, features: list[str]):
    med = train[features].median(numeric_only=True)
    x_train = train[features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(med).astype("float32")
    x_valid = valid[features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(med).astype("float32")
    return x_train, x_valid, med.to_dict()


def _train_binary(train: pd.DataFrame, valid: pd.DataFrame, features: list[str], seed: int) -> tuple[lgb.Booster, dict[str, float]]:
    y = train["good_top5"].astype(int)
    pos_idx = train.index[y.eq(1)]
    neg_idx = train.index[y.eq(0)]
    max_neg = min(len(neg_idx), max(len(pos_idx) * 25, 5000), 180_000)
    if max_neg < len(neg_idx):
        sampled_neg = pd.Index(neg_idx).to_series().sample(n=max_neg, random_state=seed).index
        keep = pos_idx.union(sampled_neg)
        train = train.loc[keep]
        y = y.loc[keep]
    x_train, x_valid, med = _prep_xy(train, valid, features)
    positives = int(y.sum())
    negatives = int(len(y) - positives)
    train_set = lgb.Dataset(x_train, label=y)
    valid_set = lgb.Dataset(x_valid, label=valid["good_top5"].astype(int), reference=train_set)
    params = {
        "objective": "binary",
        "metric": ["auc", "average_precision"],
        "learning_rate": 0.035,
        "num_leaves": 21,
        "min_data_in_leaf": 80,
        "feature_fraction": 0.82,
        "bagging_fraction": 0.86,
        "bagging_freq": 1,
        "lambda_l1": 1.0,
        "lambda_l2": 14.0,
        "scale_pos_weight": negatives / max(positives, 1),
        "verbosity": -1,
        "seed": seed,
    }
    model = lgb.train(
        params,
        train_set,
        valid_sets=[valid_set],
        num_boost_round=300,
        callbacks=[lgb.early_stopping(35), lgb.log_evaluation(0)],
    )
    return model, med


def train_long_rescue_models(
    dataset_path: Path = PROCESSED_DIR / "daily_features.parquet",
    output_root: Path = LONG_RESCUE_ROOT,
    run_name: str | None = None,
    train_window: tuple[str, str] = ("2010-01-01", "2024-12-31"),
    validation_window: tuple[str, str] = ("2025-01-01", "2025-12-31"),
    threshold_window: tuple[str, str] = ("2026-01-01", "2026-05-29"),
    update_current: bool = True,
) -> Path:
    run_id = run_name or f"long_rescue_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = output_root / run_id
    models_dir = run_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    config = TieredCleanConfig(train_start_year=2010, train_all_symbols=True, use_catboost=False)
    df, _base_features, specs = prepare_tiered_frame(dataset_path, config)
    feature_cols = [c for c in CURATED_FEATURES if c in df.columns] + [
        "arch_reversal", "arch_compression_breakout", "arch_quiet_breakdown", "arch_any"
    ]

    manifest_specs = []
    reports = []
    for idx, rescue_spec in enumerate(RESCUE_SPECS):
        tier_spec = next(s for s in specs if s.model_id == rescue_spec.model_id)
        frame = _build_spec_frame(df, tier_spec)
        frame = frame[frame["evaluated"]].copy()
        candidate = frame[frame[rescue_spec.archetype_col].fillna(False).astype(bool)].copy()
        train = candidate[candidate["date"].between(train_window[0], train_window[1])].copy()
        valid = candidate[candidate["date"].between(validation_window[0], validation_window[1])].copy()
        threshold_frame = candidate[candidate["date"].between(threshold_window[0], threshold_window[1])].copy()
        if train["good_top5"].nunique() < 2 or valid.empty or threshold_frame.empty:
            raise ValueError(f"Insufficient rows for {rescue_spec.model_id}:{rescue_spec.expert}")
        model, medians = _train_binary(train, valid, feature_cols, seed=rescue_spec.seed)
        model.save_model(str(models_dir / rescue_spec.model_file))

        x_threshold = threshold_frame[feature_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(pd.Series(medians)).astype("float32")
        threshold_frame["rescue_score"] = model.predict(x_threshold)
        score_threshold = float(threshold_frame["rescue_score"].quantile(rescue_spec.threshold_quantile))

        manifest_specs.append({
            "model_id": rescue_spec.model_id,
            "side": rescue_spec.side,
            "expert": rescue_spec.expert,
            "archetype_col": rescue_spec.archetype_col,
            "model_file": rescue_spec.model_file,
            "seed": rescue_spec.seed,
            "score_threshold": score_threshold,
            "threshold_quantile": rescue_spec.threshold_quantile,
            "median_values": medians,
        })
        reports.append({
            "model_id": rescue_spec.model_id,
            "expert": rescue_spec.expert,
            "train_rows": int(len(train)),
            "train_good_top5": int(train["good_top5"].sum()),
            "valid_rows": int(len(valid)),
            "valid_good_top5": int(valid["good_top5"].sum()),
            "threshold_rows": int(len(threshold_frame)),
            "score_threshold": score_threshold,
        })

    manifest = {
        "created_at": datetime.now().astimezone().isoformat(),
        "label": RESCUE_LABEL,
        "bucket": RESCUE_BUCKET,
        "daily_top_n": 3,
        "features": feature_cols,
        "train_window": list(train_window),
        "validation_window": list(validation_window),
        "threshold_window": list(threshold_window),
        "threshold_note": "Balanced RGO+ profile uses the configured threshold window score quantile.",
        "specs": manifest_specs,
        "reports": reports,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    pd.DataFrame(reports).to_csv(run_dir / "training_summary.csv", index=False)

    if update_current:
        current = output_root / "current"
        if current.exists():
            shutil.rmtree(current)
        shutil.copytree(run_dir, current)
    return run_dir


def _score_rescue_candidates(frame: pd.DataFrame, model_root: Path) -> pd.DataFrame:
    manifest = json.loads((model_root / "manifest.json").read_text(encoding="utf-8"))
    features = manifest["features"]
    out_parts = []
    for spec_meta in manifest["specs"]:
        model_id = spec_meta["model_id"]
        part = frame[frame["model_id"].astype(str).eq(model_id) & frame["side"].astype(str).eq(spec_meta["side"])].copy()
        if part.empty:
            continue
        part = add_long_rescue_archetype_flags(part, spec_meta["side"])
        part = part[part[spec_meta["archetype_col"]].fillna(False).astype(bool)].copy()
        if part.empty:
            continue
        existing_rescue = part.get("go_label", pd.Series("WATCH", index=part.index)).astype(str).map(has_rescue_label)
        part = part[~existing_rescue].copy()
        if part.empty:
            continue
        model = lgb.Booster(model_file=str(model_root / "models" / spec_meta["model_file"]))
        med = pd.Series(spec_meta.get("median_values", {}), dtype="float64")
        x = part.reindex(columns=features).apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(med).astype("float32")
        part["rescue_score"] = model.predict(x)
        part["rescue_expert"] = spec_meta["expert"]
        part["rescue_threshold_used"] = float(spec_meta["score_threshold"])
        part["rescue_selection_quantile"] = float(spec_meta.get("threshold_quantile", 0.99))
        part = part[part["rescue_score"].ge(float(spec_meta["score_threshold"]))]
        out_parts.append(part)
    if not out_parts:
        return pd.DataFrame(columns=list(frame.columns) + ["rescue_score"])
    scored = pd.concat(out_parts, ignore_index=False)
    daily_top_n = int(manifest.get("daily_top_n", 3))
    existing_rgo = pd.Series(0, dtype="int64")
    if "go_label" in frame.columns:
        existing_rgo = frame[frame["go_label"].astype(str).map(has_rescue_label)].groupby("date").size()
    picked = []
    for date, group in scored.groupby("date", sort=True):
        remaining = daily_top_n - int(existing_rgo.get(date, 0))
        if remaining <= 0:
            continue
        picked.extend(group.sort_values("rescue_score", ascending=False).head(remaining).index.tolist())
    return scored.loc[picked].copy()


def apply_long_rescue_to_frame(frame: pd.DataFrame, model_root: Path = LONG_RESCUE_ROOT / "current") -> pd.DataFrame:
    if frame.empty or not (model_root / "manifest.json").exists():
        return frame
    out = frame.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    if "go_label" not in out.columns:
        out["go_label"] = "WATCH"
    if "go_label_raw" not in out.columns:
        out["go_label_raw"] = out["go_label"]
    if "production_bucket" not in out.columns:
        out["production_bucket"] = "WATCH"
    candidates = _score_rescue_candidates(out, model_root)
    if candidates.empty:
        return out
    for idx, row in candidates.iterrows():
        current_label = out.loc[idx, "go_label"] if "go_label" in out.columns else "WATCH"
        combined_label = combine_with_rescue_label(current_label)
        out.loc[idx, "go_label"] = combined_label
        out.loc[idx, "production_signal"] = True
        out.loc[idx, "passes_min_score"] = True
        out.loc[idx, "production_bucket"] = (
            "PROD_LONG_GO_PLUS_AND_RESCUE" if combined_label != RESCUE_LABEL else RESCUE_BUCKET
        )
        out.loc[idx, "production_locked"] = True
        out.loc[idx, "final_signal_bucket"] = out.loc[idx, "production_bucket"]
        out.loc[idx, "trade_priority"] = "high" if combined_label != RESCUE_LABEL else "rescue"
        out.loc[idx, "trade_bucket"] = "high_long" if combined_label != RESCUE_LABEL else "rescue_long"
        out.loc[idx, "final_side"] = "up"
        out.loc[idx, "rescue_signal"] = True
        out.loc[idx, "rescue_label"] = RESCUE_LABEL
        out.loc[idx, "rescue_variant"] = "balanced_q99_q99"
        out.loc[idx, "rescue_expert"] = row["rescue_expert"]
        out.loc[idx, "rescue_score"] = float(row["rescue_score"])
        out.loc[idx, "rescue_rank_score"] = float(row["rescue_score"])
        out.loc[idx, "rescue_threshold_used"] = float(row["rescue_threshold_used"])
        out.loc[idx, "rescue_selection_quantile"] = float(row["rescue_selection_quantile"])
        out.loc[idx, "production_lock_note"] = "balanced_long_rescue_pipeline_RGO+"
    return out


def apply_long_rescue_to_prod_dir(
    prod_dir: Path = PREDICTIONS_DIR / "prod",
    model_root: Path = LONG_RESCUE_ROOT / "current",
    start: str | None = None,
    end: str | None = None,
) -> dict:
    files = sorted(prod_dir.glob("prod_predictions_*.parquet"))
    if start:
        start_ts = pd.Timestamp(start).normalize()
        files = [p for p in files if pd.Timestamp(p.stem.replace("prod_predictions_", "")).normalize() >= start_ts]
    if end:
        end_ts = pd.Timestamp(end).normalize()
        files = [p for p in files if pd.Timestamp(p.stem.replace("prod_predictions_", "")).normalize() <= end_ts]
    touched = 0
    rgo_rows = 0
    for path in files:
        before = pd.read_parquet(path)
        after = apply_long_rescue_to_frame(before, model_root=model_root)
        if not after.equals(before):
            touched += 1
            after.to_parquet(path, index=False)
            after.to_csv(path.with_suffix(".csv"), index=False)
        rgo_rows += int(after.get("go_label", pd.Series("", index=after.index)).astype(str).map(has_rescue_label).sum())
    ui_dir = PREDICTIONS_DIR / "ui"
    removed_ui = 0
    if ui_dir.exists():
        for cache_file in list(ui_dir.glob("ui_predictions_v*.parquet")) + list(ui_dir.glob("ui_predictions_v*.csv")):
            cache_file.unlink(missing_ok=True)
            removed_ui += 1
    return {"files_seen": len(files), "files_touched": touched, "rgo_rows": rgo_rows, "ui_cache_removed": removed_ui}
