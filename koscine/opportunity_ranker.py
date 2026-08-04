from __future__ import annotations

import json
import gc
import shutil
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from koscine.config import HORIZON_DAYS, MODEL_DIR, PREDICTIONS_DIR, REPORTS_DIR, RUNS_DIR
from koscine.tiered_clean_direction import (
    PROD_TIERED_ROOT,
    TieredCleanConfig,
    TieredProdConfig,
    _load_prod_model,
    _prediction_frame,
    _score_spec,
    _train_frame,
    apply_downside_production_lock,
    label_col,
    opposite_label_col,
    prepare_tiered_frame,
    predict_tiered_prod_many,
    train_tiered_prod_models,
)
from koscine.production_veto import apply_balanced_veto_to_production_frame


OPPORTUNITY_ROOT = MODEL_DIR / "opportunity_ranker"


@dataclass(frozen=True)
class OpportunityRankerConfig:
    train_start_year: int = 2010
    train_end_year: int = 2024
    validation_year: int = 2025
    train_cutoff_month: int = 12
    train_cutoff_day: int = 20
    adverse_limit: float = 0.0091
    max_weekly_abs_move: float = 0.50
    lgbm_weight: float = 0.60
    train_all_symbols: bool = True
    go_plus_per_side: int = 2
    go_per_side: int = 5
    min_direction_score: float = 0.50
    min_rest_direction_score: float = 0.50
    max_opposite_score: float = 0.70
    max_long_opposite_score: float = 0.60
    max_short_opposite_score: float = 0.70
    min_score_gap: float = 0.03
    min_move_power: float = 0.70
    go_plus_min_move_power: float = 0.75
    min_go_score_quantile: float = 0.0
    min_model_validation_top5_capture_rate: float = 0.0
    go_plus_annual_target: int = 180
    go_annual_target: int = 700
    go_plus_quality_threshold: float | None = None
    go_quality_threshold: float | None = None
    go_plus_quality_thresholds: dict[str, float] | None = None
    go_quality_thresholds: dict[str, float] | None = None
    target_negative_sample_ratio: float = 4.0
    target_max_negative_rows: int = 250_000
    ranker_max_train_rows: int = 350_000
    rest35_long_addon_min_score: float = 0.60
    rest35_long_addon_min_score_gap: float = 0.03
    rest35_long_addon_min_move_power: float = 0.80
    rest35_long_addon_min_bb_rank: float = 0.80
    rest35_long_addon_min_atm_iv: float = 0.40
    rest35_long_addon_min_path_range: float = 0.10
    rest35_long_addon_min_atr: float = 0.035
    liquid_long_addon_min_score: float = 0.80
    liquid_long_addon_min_score_gap: float = 0.05
    liquid_long_addon_min_move_power: float = 0.90
    liquid_long_addon_max_side_rank: int = 2
    liquid_long_addon_min_path_range: float = 0.08
    long_use_followthrough_quality: bool = True
    long_small_negative_weight: float = 6.0
    long_bad_negative_weight: float = 9.0
    long_hit_weight: float = 12.0
    long_top5_hit_weight: float = 20.0
    long_bottom_rs_penalty: float = 0.14
    long_negative_trend_penalty: float = 0.06
    long_opposite_pressure_penalty: float = 0.10
    top_mover_n: int = 5
    num_boost_round: int = 900
    early_stopping_rounds: int = 80


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _train_end(config: OpportunityRankerConfig) -> pd.Timestamp:
    return pd.Timestamp(
        year=config.train_end_year,
        month=config.train_cutoff_month,
        day=config.train_cutoff_day,
    )


def _safe_group_sizes(frame: pd.DataFrame) -> list[int]:
    return frame.groupby("date", sort=False).size().astype(int).tolist()


def _sort_rank_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values(["date", "symbol"]).reset_index(drop=True)


def _sample_ranker_train(frame: pd.DataFrame, config: OpportunityRankerConfig, seed: int) -> pd.DataFrame:
    if len(frame) <= config.ranker_max_train_rows:
        return frame
    positive = frame["opportunity_relevance"].gt(0)
    keep = frame[positive]
    background = frame[~positive]
    remaining = max(config.ranker_max_train_rows - len(keep), 0)
    if remaining <= 0:
        return keep.sample(n=config.ranker_max_train_rows, random_state=seed).copy()
    sampled_background = background.sample(n=min(remaining, len(background)), random_state=seed)
    return pd.concat([keep, sampled_background], ignore_index=True)


def _add_opportunity_labels(frame: pd.DataFrame, spec) -> pd.DataFrame:
    out = frame.copy()
    if "actual_hit" not in out.columns:
        out["actual_hit"] = out[label_col(spec)].fillna(False).astype(bool)
    if "actual_opposite" not in out.columns:
        out["actual_opposite"] = out[opposite_label_col(spec)].fillna(False).astype(bool)
    side_move = out[f"{'up' if spec.side == 'up' else 'down'}_move_{HORIZON_DAYS}d"].astype(float)
    close_return = out[f"fwd_return_{HORIZON_DAYS}d"].astype(float)
    signed_close = close_return if spec.side == "up" else -close_return
    out["side"] = spec.side
    out["threshold"] = spec.threshold
    out["opportunity_move"] = side_move
    out["opportunity_signed_close"] = signed_close
    out["opportunity_bad"] = out["actual_opposite"].fillna(False).astype(bool) | signed_close.lt(0)
    out["opportunity_rank"] = out.groupby("date")["opportunity_move"].rank(
        method="first",
        ascending=False,
    )
    out["is_top5_mover"] = out["opportunity_rank"].le(5)
    out["is_top3_mover"] = out["opportunity_rank"].le(3)
    out["is_top1_mover"] = out["opportunity_rank"].le(1)

    relevance = pd.Series(0, index=out.index, dtype="int64")
    relevance.loc[signed_close.gt(0) & ~out["opportunity_bad"]] = 1
    relevance.loc[out["actual_hit"].fillna(False).astype(bool) & ~out["opportunity_bad"]] = 2
    relevance.loc[out["is_top5_mover"] & signed_close.gt(0) & ~out["opportunity_bad"]] = 4
    relevance.loc[out["is_top3_mover"] & signed_close.gt(0) & ~out["opportunity_bad"]] = 5
    relevance.loc[out["is_top1_mover"] & signed_close.gt(0) & ~out["opportunity_bad"]] = 6
    out["opportunity_relevance"] = relevance
    return out


def _fit_ranker(train: pd.DataFrame, valid: pd.DataFrame, features: list[str], seed: int, config: OpportunityRankerConfig) -> lgb.Booster:
    train = _sample_ranker_train(train, config, seed)
    train = _sort_rank_frame(train)
    valid = _sort_rank_frame(valid)
    train_set = lgb.Dataset(
        train[features].astype("float32"),
        label=train["opportunity_relevance"].astype(int),
        group=_safe_group_sizes(train),
    )
    valid_set = lgb.Dataset(
        valid[features].astype("float32"),
        label=valid["opportunity_relevance"].astype(int),
        group=_safe_group_sizes(valid),
        reference=train_set,
    )
    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [1, 3, 5, 10],
        "label_gain": [0, 1, 3, 7, 12, 18, 25],
        "learning_rate": 0.03,
        "num_leaves": 31,
        "min_data_in_leaf": 80,
        "feature_fraction": 0.82,
        "bagging_fraction": 0.86,
        "bagging_freq": 1,
        "lambda_l1": 1.0,
        "lambda_l2": 12.0,
        "verbosity": -1,
        "seed": seed,
    }
    return lgb.train(
        params,
        train_set,
        valid_sets=[valid_set],
        num_boost_round=config.num_boost_round,
        callbacks=[
            lgb.early_stopping(config.early_stopping_rounds),
            lgb.log_evaluation(0),
        ],
    )


def _fit_target_model(train: pd.DataFrame, valid: pd.DataFrame, features: list[str], seed: int, config: OpportunityRankerConfig) -> lgb.Booster:
    train_y = (
        train["actual_hit"].fillna(False).astype(bool)
        & ~train["opportunity_bad"].fillna(True).astype(bool)
    ).astype(int)
    valid_y = (
        valid["actual_hit"].fillna(False).astype(bool)
        & ~valid["opportunity_bad"].fillna(True).astype(bool)
    ).astype(int)
    top5_train = train["is_top5_mover"].fillna(False).astype(bool)
    small_train = train["opportunity_move"].lt(train["threshold"])
    long_train = train.get("side", pd.Series("", index=train.index)).eq("up")
    weights = pd.Series(1.0, index=train.index)
    weights.loc[small_train & train_y.eq(0)] = 2.5
    weights.loc[train["opportunity_bad"].fillna(False).astype(bool)] = 5.0
    weights.loc[train_y.eq(1)] = 8.0
    weights.loc[top5_train & train_y.eq(1)] = 12.0
    weights.loc[long_train & small_train & train_y.eq(0)] = config.long_small_negative_weight
    weights.loc[long_train & train["opportunity_bad"].fillna(False).astype(bool)] = config.long_bad_negative_weight
    weights.loc[long_train & train_y.eq(1)] = config.long_hit_weight
    weights.loc[long_train & top5_train & train_y.eq(1)] = config.long_top5_hit_weight
    pos_idx = train.index[train_y.eq(1)]
    neg_idx = train.index[train_y.eq(0)]
    max_neg = min(
        len(neg_idx),
        config.target_max_negative_rows,
        max(int(len(pos_idx) * config.target_negative_sample_ratio), 1),
    )
    if max_neg < len(neg_idx):
        neg_idx = pd.Index(neg_idx).to_series().sample(n=max_neg, random_state=seed).index
        keep_idx = pos_idx.union(neg_idx)
        train = train.loc[keep_idx]
        train_y = train_y.loc[keep_idx]
        weights = weights.loc[keep_idx]

    train_set = lgb.Dataset(train[features].astype("float32"), label=train_y, weight=weights)
    valid_set = lgb.Dataset(valid[features].astype("float32"), label=valid_y, reference=train_set)
    params = {
        "objective": "binary",
        "metric": ["binary_logloss", "auc"],
        "learning_rate": 0.025,
        "num_leaves": 31,
        "min_data_in_leaf": 120,
        "feature_fraction": 0.78,
        "bagging_fraction": 0.84,
        "bagging_freq": 1,
        "lambda_l1": 1.0,
        "lambda_l2": 18.0,
        "verbosity": -1,
        "seed": seed,
    }
    return lgb.train(
        params,
        train_set,
        valid_sets=[valid_set],
        num_boost_round=config.num_boost_round,
        callbacks=[
            lgb.early_stopping(config.early_stopping_rounds),
            lgb.log_evaluation(0),
        ],
    )


def _score_opportunity(
    model: lgb.Booster,
    target_model: lgb.Booster | None,
    frame: pd.DataFrame,
    features: list[str],
) -> pd.DataFrame:
    out = frame.copy()
    out["opportunity_score"] = model.predict(out[features], num_iteration=model.best_iteration)
    score = out["opportunity_score"].astype(float)
    out["opportunity_score_pct"] = score.groupby(out["date"]).rank(pct=True)
    if target_model is not None:
        out["target_hit_score"] = target_model.predict(out[features], num_iteration=target_model.best_iteration)
    elif "target_hit_score" not in out:
        out["target_hit_score"] = np.nan
    target_score = out["target_hit_score"].astype(float)
    out["target_hit_score_pct"] = target_score.groupby(out["date"]).rank(pct=True)
    return out


def _rank_pct(frame: pd.DataFrame, col: str) -> pd.Series:
    if col not in frame:
        return pd.Series(0.5, index=frame.index)
    values = pd.to_numeric(frame[col], errors="coerce")
    ranked = values.groupby([frame["date"], frame["side"]]).rank(pct=True)
    return ranked.fillna(0.5)


def _numeric_feature(frame: pd.DataFrame, col: str, default: float = 0.5) -> pd.Series:
    if col not in frame:
        return pd.Series(default, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[col], errors="coerce").fillna(default)


def _add_selection_scores(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["global_opportunity_rank"] = out.groupby(["date", "side"])["opportunity_move"].rank(
        method="first",
        ascending=False,
    )
    out["global_top5_mover"] = out["global_opportunity_rank"].le(5)
    out["small_move"] = out["opportunity_move"].lt(out["threshold"])

    out["score_rank_pct"] = _rank_pct(out, "score")
    out["opportunity_rank_score_pct"] = _rank_pct(out, "opportunity_score")
    out["target_hit_score_rank_pct"] = _rank_pct(out, "target_hit_score")
    out["rule_strength_rank_pct"] = _rank_pct(out, "rule_gate_strength")
    vol_rank_cols = []
    for col in ("atm_iv", "atr_pct_14", "path_range_5d", "bb_width_20", "vol_sma20_ratio"):
        rank_col = f"{col}_move_rank_pct"
        out[rank_col] = _rank_pct(out, col)
        vol_rank_cols.append(rank_col)
    out["move_power_score"] = out[vol_rank_cols].mean(axis=1)
    own_score = out["score"].fillna(0.0).astype(float)
    paired_up = out.get("paired_up_score", pd.Series(np.nan, index=out.index)).astype(float)
    paired_down = out.get("paired_down_score", pd.Series(np.nan, index=out.index)).astype(float)
    score_gap = out.get("score_gap", pd.Series(np.nan, index=out.index)).astype(float)
    long_alignment = (
        out["side"].eq("up")
        & (
            paired_down.isna()
            | paired_down.le(0.70)
            | score_gap.ge(0.0)
        )
    )
    short_alignment = (
        out["side"].eq("down")
        & (
            paired_up.isna()
            | paired_up.le(0.70)
            | score_gap.le(0.0)
        )
    )
    out["direction_alignment_ok"] = long_alignment | short_alignment
    out["direction_score_ok"] = own_score.ge(0.50)
    out["direction_rule_ok"] = out["rule_gate_pass"].fillna(False).astype(bool) | own_score.ge(0.65)
    out["late_direction_gate"] = out["direction_score_ok"] & out["direction_rule_ok"] & out["direction_alignment_ok"]
    out["go_selection_score"] = (
        0.40 * out["target_hit_score_rank_pct"]
        + 0.20 * out["opportunity_rank_score_pct"]
        + 0.18 * out["score_rank_pct"]
        + 0.17 * out["move_power_score"]
        + 0.05 * out["rule_strength_rank_pct"]
    )
    penalty = pd.Series(0.0, index=out.index)
    penalty += np.where(out["direction_alignment_ok"], 0.0, 0.12)
    penalty += np.where(out["direction_score_ok"], 0.0, 0.08)
    penalty += np.where(out["direction_rule_ok"], 0.0, 0.04)
    out["trade_quality_score"] = (out["go_selection_score"] - penalty).clip(lower=0.0, upper=1.0)
    out["legacy_trade_quality_score"] = out["trade_quality_score"]

    if "side" in out and out["side"].eq("up").any():
        ret5 = _numeric_feature(out, "ret_5d_cs_rank")
        ret20 = _numeric_feature(out, "ret_20d_cs_rank")
        long_rs = (0.55 * ret5 + 0.45 * ret20).clip(lower=0.0, upper=1.0)
        trend_score = (
            0.35 * _rank_pct(out, "close_sma20_dist")
            + 0.35 * _rank_pct(out, "close_sma50_dist")
            + 0.30 * _rank_pct(out, "di_diff")
        ).clip(lower=0.0, upper=1.0)
        clean_participation = (
            0.30 * (1.0 - _rank_pct(out, "delivery_pct"))
            + 0.25 * (1.0 - _rank_pct(out, "fut_oi_chg_5"))
            + 0.25 * _rank_pct(out, "path_range_5d")
            + 0.20 * _rank_pct(out, "atr_pct_14")
        ).clip(lower=0.0, upper=1.0)
        long_opposite_pressure = (
            0.65 * _rank_pct(out, "paired_down_score")
            + 0.35 * score_gap.lt(0).astype(float)
        ).clip(lower=0.0, upper=1.0)
        bottom_rs = ret5.lt(0.15) & ret20.lt(0.15)
        negative_trend = _numeric_feature(out, "di_diff", 0.0).le(0.0)
        high_opposite = paired_down.ge(0.65) | score_gap.lt(0.0)
        long_quality = (
            0.30 * out["target_hit_score_rank_pct"]
            + 0.17 * out["opportunity_rank_score_pct"]
            + 0.14 * out["score_rank_pct"]
            + 0.13 * out["move_power_score"]
            + 0.13 * long_rs
            + 0.09 * trend_score
            + 0.04 * clean_participation
            - 0.08 * long_opposite_pressure
            - np.where(bottom_rs, 0.14, 0.0)
            - np.where(negative_trend, 0.06, 0.0)
            - np.where(high_opposite, 0.10, 0.0)
            - np.where(out["direction_rule_ok"], 0.0, 0.04)
        ).clip(lower=0.0, upper=1.0)
        out["long_relative_strength_score"] = long_rs
        out["long_trend_score"] = trend_score
        out["long_clean_participation_score"] = clean_participation
        out["long_opposite_pressure_score"] = long_opposite_pressure
        out["long_followthrough_score"] = long_quality
        out.loc[out["side"].eq("up"), "trade_quality_score"] = long_quality[out["side"].eq("up")]
    out["trade_rank"] = out.groupby("date")["trade_quality_score"].rank(method="first", ascending=False)
    out["side_trade_rank"] = out.groupby(["date", "side"])["trade_quality_score"].rank(method="first", ascending=False)
    return out


def _metrics(selected: pd.DataFrame, all_scored: pd.DataFrame, prefix: str) -> dict:
    if selected.empty:
        return {
            f"{prefix}_signals": 0,
            f"{prefix}_target_hit_rate": np.nan,
            f"{prefix}_opposite_rate": np.nan,
            f"{prefix}_top5_capture_rate": 0.0,
            f"{prefix}_avg_move": np.nan,
            f"{prefix}_small_move_rate": np.nan,
        }
    top5_col = "global_top5_mover" if "global_top5_mover" in all_scored else "is_top5_mover"
    total_top5 = int(all_scored[top5_col].sum())
    selected_top5 = int(selected[top5_col].sum())
    return {
        f"{prefix}_signals": int(len(selected)),
        f"{prefix}_target_hit_rate": float(selected["actual_hit"].mean()),
        f"{prefix}_opposite_rate": float(selected["actual_opposite"].mean()),
        f"{prefix}_top5_capture_rate": float(selected_top5 / max(total_top5, 1)),
        f"{prefix}_top5_hits": selected_top5,
        f"{prefix}_top5_available": total_top5,
        f"{prefix}_avg_move": float(selected["opportunity_move"].mean()),
        f"{prefix}_median_move": float(selected["opportunity_move"].median()),
        f"{prefix}_small_move_rate": float(selected["opportunity_move"].lt(selected["threshold"]).mean()),
        f"{prefix}_positive_close_rate": float(selected["opportunity_signed_close"].gt(0).mean()),
    }


def _select_go(scored: pd.DataFrame, config: OpportunityRankerConfig) -> pd.DataFrame:
    out = _add_selection_scores(scored)
    out["go_label"] = "WATCH"
    out["go_rank"] = np.nan
    out["opportunity_prod_signal"] = out["late_direction_gate"]
    go_threshold = pd.Series(config.go_quality_threshold, index=out.index, dtype="float64")
    go_plus_threshold = pd.Series(config.go_plus_quality_threshold, index=out.index, dtype="float64")
    if config.go_quality_thresholds:
        mapped = out["model_id"].map(config.go_quality_thresholds)
        go_threshold = mapped.fillna(go_threshold)
    if config.go_plus_quality_thresholds:
        mapped = out["model_id"].map(config.go_plus_quality_thresholds)
        go_plus_threshold = mapped.fillna(go_plus_threshold)
    out.loc[go_threshold.notna() & out["trade_quality_score"].ge(go_threshold), "go_label"] = "GO"
    out.loc[go_plus_threshold.notna() & out["trade_quality_score"].ge(go_plus_threshold), "go_label"] = "GO+"
    selected = out[out["go_label"].isin(["GO+", "GO"])].copy()
    if not selected.empty:
        selected = selected.sort_values(
            ["date", "trade_quality_score", "score"],
            ascending=[True, False, False],
        )
        selected["go_rank"] = selected.groupby("date").cumcount() + 1
        out.loc[selected.index, "go_rank"] = selected["go_rank"]
    return _add_production_and_addon_labels(out, config)


def _score_threshold_for_target(frame: pd.DataFrame, score_col: str, target_n: int) -> float | None:
    if target_n <= 0 or frame.empty or score_col not in frame:
        return None
    scores = frame[score_col].dropna().sort_values(ascending=False)
    if scores.empty:
        return None
    idx = min(max(int(target_n), 1), len(scores)) - 1
    return float(scores.iloc[idx])


def _thresholds_by_model(frame: pd.DataFrame, score_col: str, total_target: int) -> dict[str, float]:
    if frame.empty or total_target <= 0:
        return {}
    model_ids = sorted(frame["model_id"].dropna().unique().tolist())
    if not model_ids:
        return {}
    per_model = max(1, int(np.ceil(total_target / len(model_ids))))
    thresholds: dict[str, float] = {}
    for model_id, group in frame.groupby("model_id", sort=True):
        threshold = _score_threshold_for_target(group, score_col, per_model)
        if threshold is not None:
            thresholds[str(model_id)] = threshold
    return thresholds


def _num_col(frame: pd.DataFrame, col: str, default: float = np.nan) -> pd.Series:
    if col not in frame:
        return pd.Series(default, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[col], errors="coerce")


def _add_production_and_addon_labels(frame: pd.DataFrame, config: OpportunityRankerConfig) -> pd.DataFrame:
    out = frame.copy()
    model_id = out["model_id"].astype(str)
    go_label = out["go_label"].astype(str)

    liquid_down = model_id.eq("liquid30_down_4pct_5d")
    rest_down = model_id.eq("rest35_down_7pct_5d")
    out["production_bucket"] = "WATCH"
    out["production_locked"] = False
    out.loc[liquid_down & go_label.eq("GO+"), "production_bucket"] = "PROD_SHORT_GO_PLUS"
    out.loc[rest_down & go_label.eq("GO+"), "production_bucket"] = "PROD_SHORT_GO_PLUS"
    out.loc[liquid_down & go_label.eq("GO"), "production_bucket"] = "PROD_SHORT_GO"
    out.loc[out["production_bucket"].ne("WATCH"), "production_locked"] = True

    score = _num_col(out, "score", 0.0).fillna(0.0)
    score_gap = _num_col(out, "score_gap", 0.0).fillna(0.0)
    move_power = _num_col(out, "move_power_score", 0.0).fillna(0.0)
    side_rank = _num_col(out, "side_trade_rank", np.inf).fillna(np.inf)
    bb_rank = _num_col(out, "bb_width_20_rank_60d", 0.0).fillna(0.0)
    atm_iv = _num_col(out, "atm_iv", 0.0).fillna(0.0)
    path_range = _num_col(out, "path_range_5d", 0.0).fillna(0.0)
    atr = _num_col(out, "atr_pct_14", 0.0).fillna(0.0)
    aligned = out.get("direction_alignment_ok", pd.Series(False, index=out.index)).fillna(False).astype(bool)
    rule_ok = out.get("direction_rule_ok", pd.Series(False, index=out.index)).fillna(False).astype(bool)

    rest_long = model_id.eq("rest35_up_7pct_5d")
    liquid_long = model_id.eq("liquid30_up_4pct_5d")
    rest_long_addon = (
        rest_long
        & aligned
        & rule_ok
        & score.ge(config.rest35_long_addon_min_score)
        & score_gap.ge(config.rest35_long_addon_min_score_gap)
        & move_power.ge(config.rest35_long_addon_min_move_power)
        & bb_rank.ge(config.rest35_long_addon_min_bb_rank)
        & atm_iv.ge(config.rest35_long_addon_min_atm_iv)
        & path_range.ge(config.rest35_long_addon_min_path_range)
        & atr.ge(config.rest35_long_addon_min_atr)
    )
    liquid_long_addon = (
        liquid_long
        & aligned
        & rule_ok
        & score.ge(config.liquid_long_addon_min_score)
        & score_gap.ge(config.liquid_long_addon_min_score_gap)
        & move_power.ge(config.liquid_long_addon_min_move_power)
        & side_rank.le(config.liquid_long_addon_max_side_rank)
        & path_range.ge(config.liquid_long_addon_min_path_range)
    )

    out["long_addon_label"] = "WATCH"
    out["long_addon_reason"] = ""
    out.loc[rest_long_addon, "long_addon_label"] = "LONG_GO_PLUS_ADDON"
    out.loc[rest_long_addon, "long_addon_reason"] = "rest35_up_expansion_quality"
    out.loc[liquid_long_addon, "long_addon_label"] = "LONG_GO_PLUS_ADDON"
    out.loc[liquid_long_addon, "long_addon_reason"] = "liquid30_up_strict_quality"

    out["final_signal_bucket"] = out["production_bucket"]
    addon_mask = out["final_signal_bucket"].eq("WATCH") & out["long_addon_label"].eq("LONG_GO_PLUS_ADDON")
    out.loc[addon_mask, "final_signal_bucket"] = "LONG_GO_PLUS_ADDON"
    return apply_balanced_veto_to_production_frame(out)


def _selection_summary(selected: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if selected.empty:
        return pd.DataFrame(rows)
    for (model_id, side), group in selected.groupby(["model_id", "side"], sort=True):
        go_plus = group[group["go_label"].eq("GO+")]
        go = group[group["go_label"].isin(["GO+", "GO"])]
        rows.append({"model_id": model_id, "side": side, **_metrics(go_plus, group, "go_plus"), **_metrics(go, group, "go_all")})
    return pd.DataFrame(rows)


def _spec_contexts(dataset_path: Path, prod_dir: Path, config: OpportunityRankerConfig):
    prod_manifest = json.loads((prod_dir / "manifest.json").read_text(encoding="utf-8"))
    prod_config = TieredProdConfig(**{k: v for k, v in prod_manifest.get("config", {}).items() if k in TieredProdConfig.__dataclass_fields__})
    df, features, specs = prepare_tiered_frame(dataset_path, prod_config)
    contexts = []
    for meta in prod_manifest["models"]:
        spec = next(s for s in specs if s.model_id == meta["model_id"])
        models = _load_prod_model(meta, prod_dir)
        contexts.append((meta, spec, models, float(meta.get("lgbm_weight", prod_config.lgbm_weight)), prod_config, pd.Timestamp(prod_manifest["train_end"])))
    return df, features, contexts


def train_opportunity_ranker(
    dataset_path: Path,
    config: OpportunityRankerConfig | None = None,
    run_name: str | None = None,
    prod_dir: Path = PROD_TIERED_ROOT / "current",
) -> Path:
    config = config or OpportunityRankerConfig()
    run_id = run_name or f"opportunity_ranker_{timestamp()}"
    run_dir = OPPORTUNITY_ROOT / run_id
    (run_dir / "models").mkdir(parents=True, exist_ok=True)
    (run_dir / "reports").mkdir(parents=True, exist_ok=True)
    (run_dir / "predictions").mkdir(parents=True, exist_ok=True)

    df, features, contexts = _spec_contexts(dataset_path, prod_dir, config)
    train_end = _train_end(config)
    validation_start = pd.Timestamp(year=config.validation_year, month=1, day=1)
    validation_end = pd.Timestamp(year=config.validation_year, month=12, day=31)

    all_valid = []
    model_meta = []
    for meta, spec, models, lgbm_weight, prod_config, prod_train_end in contexts:
        train_base = _train_frame(
            df,
            spec,
            config.train_start_year,
            train_end,
            config.train_all_symbols,
        )
        direction_train = _train_frame(
            df,
            spec,
            prod_config.train_start_year,
            prod_train_end,
            prod_config.train_all_symbols,
        )
        train = _add_opportunity_labels(train_base, spec)
        valid_base = _prediction_frame(df, spec, validation_start, validation_end)
        valid_base = valid_base.dropna(subset=["entry_1d_open"])
        if valid_base.empty or train.empty:
            continue
        valid_direction = _score_spec(valid_base, direction_train, features, spec, models, lgbm_weight)
        valid = _add_opportunity_labels(valid_direction, spec)
        model = _fit_ranker(train, valid, features, 2100 + len(model_meta), config)
        model_file = f"{spec.model_id}_opportunity_lambdarank.txt"
        model.save_model(run_dir / "models" / model_file)
        target_model = _fit_target_model(train, valid, features, 3100 + len(model_meta), config)
        target_model_file = f"{spec.model_id}_target_hit_lgbm.txt"
        target_model.save_model(run_dir / "models" / target_model_file)
        valid_scored = _score_opportunity(model, target_model, valid, features)
        valid_scored["model_id"] = spec.model_id
        valid_scored["tier"] = spec.name
        valid_scored["side"] = spec.side
        valid_scored["threshold"] = spec.threshold
        valid_scored["prediction_year"] = config.validation_year
        all_valid.append(valid_scored)
        model_meta.append(
            {
                "model_id": spec.model_id,
                "tier": spec.name,
                "side": spec.side,
                "threshold": spec.threshold,
                "model_file": model_file,
                "target_model_file": target_model_file,
                "best_iteration": model.best_iteration,
                "target_best_iteration": target_model.best_iteration,
                "train_rows": len(train),
                "valid_rows": len(valid),
            }
        )
        del train_base, direction_train, train, valid_base, valid_direction, valid
        del model, target_model, valid_scored
        gc.collect()

    valid_all = pd.concat(all_valid, ignore_index=True) if all_valid else pd.DataFrame()
    if not valid_all.empty:
        valid_all = apply_downside_production_lock(valid_all, prod_dir=prod_dir)
    threshold_frame = _add_selection_scores(valid_all) if not valid_all.empty else valid_all
    go_plus_threshold = config.go_plus_quality_threshold
    go_threshold = config.go_quality_threshold
    go_plus_thresholds = config.go_plus_quality_thresholds
    go_thresholds = config.go_quality_thresholds
    if go_plus_threshold is None and not threshold_frame.empty:
        go_plus_threshold = _score_threshold_for_target(
            threshold_frame,
            "trade_quality_score",
            config.go_plus_annual_target,
        )
    if go_plus_thresholds is None and not threshold_frame.empty:
        go_plus_thresholds = _thresholds_by_model(
            threshold_frame,
            "trade_quality_score",
            config.go_plus_annual_target,
        )
    if go_threshold is None and not threshold_frame.empty:
        go_threshold = _score_threshold_for_target(
            threshold_frame,
            "trade_quality_score",
            config.go_annual_target,
        )
    if go_thresholds is None and not threshold_frame.empty:
        go_thresholds = _thresholds_by_model(
            threshold_frame,
            "trade_quality_score",
            config.go_annual_target,
        )
    selection_config = replace(
        config,
        go_plus_quality_threshold=go_plus_threshold,
        go_quality_threshold=go_threshold,
        go_plus_quality_thresholds=go_plus_thresholds,
        go_quality_thresholds=go_thresholds,
    )
    selected = _select_go(valid_all, selection_config) if not valid_all.empty else valid_all
    summary = _selection_summary(selected)
    eligible_models = set(summary["model_id"].tolist()) if not summary.empty else set()
    if selection_config.min_model_validation_top5_capture_rate > 0 and not summary.empty:
        eligible_models = set(
            summary.loc[
                summary["go_all_top5_capture_rate"].ge(selection_config.min_model_validation_top5_capture_rate),
                "model_id",
            ]
        )
        for side, side_summary in summary.groupby("side", sort=True):
            if side_summary.loc[side_summary["model_id"].isin(eligible_models)].empty:
                best = side_summary.sort_values("go_all_top5_capture_rate", ascending=False).iloc[0]
                eligible_models.add(best["model_id"])
        valid_all["model_go_eligible"] = valid_all["model_id"].isin(eligible_models)
        selected = _select_go(valid_all, selection_config)
        summary = _selection_summary(selected)
    for meta in model_meta:
        meta["go_eligible"] = meta["model_id"] in eligible_models
    valid_out = run_dir / "predictions" / f"{run_id}_validation_2025_scored.parquet"
    selected.to_parquet(valid_out, index=False)
    selected.to_csv(run_dir / "predictions" / f"{run_id}_validation_2025_scored.csv", index=False)
    summary.to_csv(run_dir / "reports" / "validation_2025_summary.csv", index=False)

    manifest = {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_path": str(dataset_path),
        "direction_prod_dir": str(prod_dir),
        "config": selection_config.__dict__,
        "features": features,
        "models": model_meta,
        "selection": {
            "go_plus_per_side": config.go_plus_per_side,
            "go_per_side": config.go_per_side,
            "min_direction_score": config.min_direction_score,
            "min_go_score_quantile": selection_config.min_go_score_quantile,
            "min_model_validation_top5_capture_rate": selection_config.min_model_validation_top5_capture_rate,
            "go_plus_quality_threshold": selection_config.go_plus_quality_threshold,
            "go_quality_threshold": selection_config.go_quality_threshold,
            "go_plus_quality_thresholds": selection_config.go_plus_quality_thresholds,
            "go_quality_thresholds": selection_config.go_quality_thresholds,
            "go_plus_annual_target": selection_config.go_plus_annual_target,
            "go_annual_target": selection_config.go_annual_target,
            "locked_production_buckets": {
                "PROD_SHORT_GO_PLUS": ["liquid30_down_4pct_5d:GO+", "rest35_down_7pct_5d:GO+"],
                "PROD_SHORT_GO": ["liquid30_down_4pct_5d:GO"],
            },
            "long_addon": {
                "label": "LONG_GO_PLUS_ADDON",
                "rest35_up_model_id": "rest35_up_7pct_5d",
                "liquid30_up_model_id": "liquid30_up_4pct_5d",
                "rest35_min_score": selection_config.rest35_long_addon_min_score,
                "rest35_min_score_gap": selection_config.rest35_long_addon_min_score_gap,
                "rest35_min_move_power": selection_config.rest35_long_addon_min_move_power,
                "rest35_min_bb_rank": selection_config.rest35_long_addon_min_bb_rank,
                "rest35_min_atm_iv": selection_config.rest35_long_addon_min_atm_iv,
                "rest35_min_path_range": selection_config.rest35_long_addon_min_path_range,
                "rest35_min_atr": selection_config.rest35_long_addon_min_atr,
                "liquid_min_score": selection_config.liquid_long_addon_min_score,
                "liquid_min_score_gap": selection_config.liquid_long_addon_min_score_gap,
                "liquid_min_move_power": selection_config.liquid_long_addon_min_move_power,
                "liquid_max_side_rank": selection_config.liquid_long_addon_max_side_rank,
                "liquid_min_path_range": selection_config.liquid_long_addon_min_path_range,
            },
            "go_eligible_model_ids": sorted(eligible_models),
        },
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    current = OPPORTUNITY_ROOT / "current"
    if current.exists():
        shutil.rmtree(current)
    shutil.copytree(run_dir, current)
    summary.to_csv(REPORTS_DIR / "opportunity_ranker_validation_2025_summary.csv", index=False)
    return run_dir


def apply_opportunity_ranker(
    dataset_path: Path,
    start_date: str,
    end_date: str,
    rank_dir: Path = OPPORTUNITY_ROOT / "current",
    prod_dir: Path = PROD_TIERED_ROOT / "current",
    output_dir: Path = PREDICTIONS_DIR / "opportunity",
    write_latest: bool = True,
) -> pd.DataFrame:
    manifest = json.loads((rank_dir / "manifest.json").read_text(encoding="utf-8"))
    config = OpportunityRankerConfig(**{k: v for k, v in manifest["config"].items() if k in OpportunityRankerConfig.__dataclass_fields__})
    df, features, contexts = _spec_contexts(dataset_path, prod_dir, config)
    dates = pd.bdate_range(start_date, end_date)
    base_frames = []
    keep_prediction_cols = [
        "date",
        "symbol",
        "side",
        "threshold",
        "score",
        "model_id",
        "tier",
        "rule_name",
        "rule_gate_pass",
        "rule_gate_strength",
        "model_score",
        "lgbm_score",
        "catboost_score",
        "entry_1d_date",
        "entry_1d_open",
        f"future_{HORIZON_DAYS}d_date",
        f"future_{HORIZON_DAYS}d_high",
        f"future_{HORIZON_DAYS}d_low",
        f"future_{HORIZON_DAYS}d_close",
        f"up_move_{HORIZON_DAYS}d",
        f"down_move_{HORIZON_DAYS}d",
        f"fwd_return_{HORIZON_DAYS}d",
        "actual_move",
        "actual_hit",
        "actual_opposite",
        "signed_close_return_5d",
        "atr_pct_14",
        "atm_iv",
    ]
    for _meta, spec, models, lgbm_weight, prod_config, prod_train_end in contexts:
        train = _train_frame(
            df,
            spec,
            prod_config.train_start_year,
            prod_train_end,
            prod_config.train_all_symbols,
        )
        date_rows = []
        for date in dates:
            rows = df[df["date"].eq(date) & df["symbol"].isin(spec.predict_symbols)].copy()
            if rows.empty:
                continue
            scored = _score_spec(rows, train, features, spec, models, lgbm_weight)
            keep = [c for c in keep_prediction_cols if c in scored.columns]
            date_rows.append(scored[keep])
        if not date_rows:
            continue
        base_frames.append(pd.concat(date_rows, ignore_index=True))

    if not base_frames:
        raise ValueError(f"No opportunity predictions found from {start_date} to {end_date}")

    base_all = apply_downside_production_lock(
        pd.concat(base_frames, ignore_index=True),
        prod_dir=prod_dir,
    )
    feature_rows = df[["date", "symbol"] + features].copy()
    frames = []
    for _meta, spec, _models, _lgbm_weight, _prod_config, _prod_train_end in contexts:
        base = base_all[base_all["model_id"].eq(spec.model_id)].copy()
        if base.empty:
            continue
        model_meta = next(m for m in manifest["models"] if m["model_id"] == spec.model_id)
        ranker = lgb.Booster(model_file=str(rank_dir / "models" / model_meta["model_file"]))
        target_ranker = None
        if model_meta.get("target_model_file"):
            target_ranker = lgb.Booster(model_file=str(rank_dir / "models" / model_meta["target_model_file"]))
        base = base.drop(columns=[col for col in features if col in base.columns])
        joined = base.merge(feature_rows, on=["date", "symbol"], how="left")
        joined = _score_opportunity(ranker, target_ranker, joined, features)
        joined = _add_opportunity_labels(joined, spec)
        frames.append(joined)

    combined = pd.concat(frames, ignore_index=True)
    eligible_model_ids = {
        m["model_id"]
        for m in manifest.get("models", [])
        if m.get("go_eligible", True)
    }
    combined["model_go_eligible"] = combined["model_id"].isin(eligible_model_ids)
    combined = _select_go(combined, config)
    combined["passes_min_score"] = combined["go_label"].isin(["GO+", "GO"])
    combined = combined.sort_values(["date", "side", "go_label", "opportunity_score"], ascending=[True, True, True, False])

    output_dir.mkdir(parents=True, exist_ok=True)
    safe_start = pd.Timestamp(start_date).strftime("%Y%m%d")
    safe_end = pd.Timestamp(end_date).strftime("%Y%m%d")
    out_path = output_dir / f"opportunity_predictions_{safe_start}_{safe_end}.parquet"
    combined.to_parquet(out_path, index=False)
    combined.to_csv(output_dir / f"opportunity_predictions_{safe_start}_{safe_end}.csv", index=False)
    if write_latest:
        combined.to_parquet(PREDICTIONS_DIR / "opportunity_predictions_latest.parquet", index=False)
        combined.to_csv(PREDICTIONS_DIR / "opportunity_predictions_latest.csv", index=False)
    return combined


def evaluate_opportunity(predictions: pd.DataFrame, output_name: str = "opportunity_jan_may_2026") -> pd.DataFrame:
    rows = []
    for (side, model_id), group in predictions.groupby(["side", "model_id"], sort=True):
        go_plus = group[group["go_label"].eq("GO+")]
        go_all = group[group["go_label"].isin(["GO+", "GO"])]
        rows.append({"side": side, "model_id": model_id, **_metrics(go_plus, group, "go_plus"), **_metrics(go_all, group, "go_all")})
    for side, group in predictions.groupby("side", sort=True):
        go_plus = group[group["go_label"].eq("GO+")]
        go_all = group[group["go_label"].isin(["GO+", "GO"])]
        rows.append({"side": side, "model_id": "ALL", **_metrics(go_plus, group, "go_plus"), **_metrics(go_all, group, "go_all")})
    summary = pd.DataFrame(rows)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(REPORTS_DIR / f"{output_name}_summary.csv", index=False)
    return summary


def run_opportunity_pipeline(
    dataset_path: Path,
    config: OpportunityRankerConfig | None = None,
    start_date: str = "2026-01-01",
    end_date: str = "2026-05-31",
    run_name: str | None = None,
) -> tuple[Path, pd.DataFrame, pd.DataFrame]:
    config = config or OpportunityRankerConfig()
    prod_config = TieredProdConfig(
        train_start_year=config.train_start_year,
        train_cutoff_day=config.train_cutoff_day,
        adverse_limit=config.adverse_limit,
        max_weekly_abs_move=config.max_weekly_abs_move,
        lgbm_weight=config.lgbm_weight,
        use_catboost=False,
        min_score=0.0,
        train_all_symbols=config.train_all_symbols,
    )
    train_tiered_prod_models(
        dataset_path=dataset_path,
        prediction_month=f"{config.validation_year}-01",
        config=prod_config,
        run_name=f"{run_name or 'opportunity'}_direction_2010_2024",
    )
    rank_dir = train_opportunity_ranker(dataset_path, config=config, run_name=run_name)
    dates = pd.bdate_range(start_date, end_date).strftime("%Y-%m-%d").tolist()
    predict_tiered_prod_many(dataset_path, dates, prod_dir=PROD_TIERED_ROOT / "current")
    predictions = apply_opportunity_ranker(dataset_path, start_date, end_date)
    summary = evaluate_opportunity(predictions)
    return rank_dir, predictions, summary
