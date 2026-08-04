from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from koscine.config import HORIZON_DAYS, PREDICTIONS_DIR, REPORTS_DIR, RUNS_DIR
from koscine.tiered_clean_direction import (
    TierSpec,
    add_tiered_clean_labels,
    label_col,
    opposite_label_col,
    prepare_tiered_frame,
    rule_gate,
    tier_specs,
)


@dataclass(frozen=True)
class RawRankConfig:
    train_start_year: int = 2012
    start_test_year: int = 2022
    end_test_year: int = 2025
    validation_days: int = 365
    purge_days: int = 7
    adverse_limit: float = 0.0091
    max_weekly_abs_move: float = 0.50
    bad_rate_cap: float = 0.15
    min_validation_calls: int = 20
    train_all_symbols: bool = True
    feature_fraction: float = 0.78
    max_features: int = 180


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _rank_features(features: list[str], max_features: int) -> list[str]:
    blocked_fragments = ("future_", "label_", "actual_", "target_", "entry_1d", "fwd_return")
    out = [f for f in features if not any(fragment in f for fragment in blocked_fragments)]
    priority = [
        "atm_iv",
        "atm_iv_rank_252d",
        "atm_ce_iv",
        "atm_pe_iv",
        "hv_20",
        "realized_vol_20",
        "vix_level",
        "vix_rank_252d",
        "bb_width_20",
        "bb_width_20_rank_252d",
        "tight_range_10d_rank_252d",
        "delivery_pct",
        "pcr_vol",
        "pcr_vol_rank_252d",
        "ret_20d",
        "ret_5d",
        "close_sma50_dist",
        "close_sma200_dist",
        "fut_oi_z_60d",
        "fut_oi_z_60d_cs_rank",
        "iv_vs_hv_rank_252d",
        "dist_call_wall",
        "dist_put_wall",
    ]
    selected = [f for f in priority if f in out]
    selected.extend([f for f in out if f not in selected])
    return selected[:max_features]


def _future_known_mask(df: pd.DataFrame, train_end: pd.Timestamp) -> pd.Series:
    future_col = f"future_{HORIZON_DAYS}d_date"
    if future_col not in df:
        return pd.Series(True, index=df.index)
    return pd.to_datetime(df[future_col]).le(train_end)


def _labels(frame: pd.DataFrame, spec: TierSpec) -> pd.DataFrame:
    out = frame.copy()
    out["target_label"] = out[label_col(spec)].fillna(False).astype(bool)
    out["risk_label"] = out[opposite_label_col(spec)].fillna(False).astype(bool)
    sign = 1.0 if spec.side == "up" else -1.0
    out["signed_close_return_5d"] = sign * out[f"fwd_return_{HORIZON_DAYS}d"].astype(float)
    out["safe_label"] = out["signed_close_return_5d"].gt(0) & ~out["risk_label"]
    out["rank_relevance"] = 0
    out.loc[out["safe_label"], "rank_relevance"] = 1
    out.loc[out["target_label"], "rank_relevance"] = 3
    out.loc[out["risk_label"], "rank_relevance"] = 0
    out["actual_hit"] = out["target_label"]
    out["actual_opposite"] = out["risk_label"]
    out["actual_move"] = out[f"{'up' if spec.side == 'up' else 'down'}_move_{HORIZON_DAYS}d"].astype(float)
    return out


def _groups(frame: pd.DataFrame) -> list[int]:
    return frame.groupby("date", sort=False).size().astype(int).tolist()


def _sort(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values(["date", "symbol"]).reset_index(drop=True)


def _fit_ranker(train: pd.DataFrame, valid: pd.DataFrame, features: list[str], seed: int, config: RawRankConfig) -> lgb.Booster:
    train = _sort(train)
    valid = _sort(valid)
    train_set = lgb.Dataset(
        train[features],
        label=train["rank_relevance"].astype(int),
        group=_groups(train),
    )
    valid_set = lgb.Dataset(
        valid[features],
        label=valid["rank_relevance"].astype(int),
        group=_groups(valid),
        reference=train_set,
    )
    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [1, 3, 5, 10],
        "label_gain": [0, 1, 3, 7],
        "learning_rate": 0.03,
        "num_leaves": 31 if "liquid30" in train.get("tier", pd.Series([""])).astype(str).iloc[0] else 47,
        "min_data_in_leaf": 120,
        "feature_fraction": config.feature_fraction,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        "lambda_l1": 2.0,
        "lambda_l2": 16.0,
        "verbosity": -1,
        "seed": seed,
    }
    return lgb.train(
        params,
        train_set,
        valid_sets=[valid_set],
        num_boost_round=700,
        callbacks=[lgb.early_stopping(70), lgb.log_evaluation(0)],
    )


def _metrics(frame: pd.DataFrame) -> dict:
    if frame.empty:
        return {
            "signals": 0,
            "hit_rate": np.nan,
            "bad_rate": np.nan,
            "positive_close_rate": np.nan,
            "avg_move": np.nan,
            "avg_rank_score": np.nan,
        }
    return {
        "signals": int(len(frame)),
        "hit_rate": float(frame["actual_hit"].mean()),
        "bad_rate": float(frame["actual_opposite"].mean()),
        "positive_close_rate": float(frame["safe_label"].mean()),
        "avg_move": float(frame["actual_move"].mean()),
        "avg_rank_score": float(frame["raw_rank_score"].mean()),
    }


def choose_rule(valid: pd.DataFrame, config: RawRankConfig) -> tuple[dict, pd.DataFrame]:
    rows = []
    if valid.empty:
        return {"min_score": np.inf, "daily_top": 1, "require_gate": False}, pd.DataFrame()
    scores = valid["raw_rank_score"].dropna()
    for q in (0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.925, 0.95, 0.975):
        min_score = float(scores.quantile(q))
        for daily_top in (1, 2, 3, 5, 10, 20):
            for require_gate in (False, True):
                selected = valid[valid["raw_rank_score"].ge(min_score)].copy()
                if require_gate:
                    selected = selected[selected["rule_gate_pass"]]
                selected = selected.sort_values(["date", "raw_rank_score"], ascending=[True, False]).groupby("date").head(daily_top)
                rows.append(
                    {
                        "min_score": min_score,
                        "score_quantile": q,
                        "daily_top": daily_top,
                        "require_gate": require_gate,
                        **_metrics(selected),
                    }
                )
    grid = pd.DataFrame(rows)
    eligible = grid[
        (grid["signals"] >= config.min_validation_calls)
        & (grid["bad_rate"].fillna(1.0) <= config.bad_rate_cap)
    ].copy()
    if eligible.empty:
        eligible = grid[grid["signals"] >= max(5, config.min_validation_calls // 2)].copy()
    if eligible.empty:
        eligible = grid.copy()
    chosen = eligible.sort_values(
        ["hit_rate", "positive_close_rate", "avg_move", "signals", "bad_rate"],
        ascending=[False, False, False, False, True],
    ).iloc[0]
    return {
        "min_score": float(chosen["min_score"]),
        "daily_top": int(chosen["daily_top"]),
        "require_gate": bool(chosen["require_gate"]),
    }, grid


def apply_rule(scored: pd.DataFrame, rule: dict) -> pd.DataFrame:
    selected = scored[scored["raw_rank_score"].ge(float(rule["min_score"]))].copy()
    if rule.get("require_gate", False):
        selected = selected[selected["rule_gate_pass"]]
    if selected.empty:
        return selected
    return selected.sort_values(["date", "raw_rank_score"], ascending=[True, False]).groupby("date").head(int(rule["daily_top"]))


def run_raw_rank_walkforward(dataset_path: Path, config: RawRankConfig | None = None, run_name: str | None = None) -> Path:
    config = config or RawRankConfig()
    run_id = run_name or f"raw_rank_{timestamp()}"
    run_dir = RUNS_DIR / run_id
    (run_dir / "reports").mkdir(parents=True, exist_ok=True)
    (run_dir / "predictions").mkdir(parents=True, exist_ok=True)

    df, features, specs = prepare_tiered_frame(dataset_path, config)
    features = _rank_features(features, config.max_features)
    selected_frames = []
    summary_rows = []
    grids = []

    for year in range(config.start_test_year, config.end_test_year + 1):
        train_end = pd.Timestamp(year=year - 1, month=12, day=20)
        valid_start = train_end - pd.Timedelta(days=config.validation_days)
        train_cut = valid_start - pd.Timedelta(days=config.purge_days)
        for spec in specs:
            base_mask = df["date"].dt.year.ge(config.train_start_year) & _future_known_mask(df, train_end)
            if not config.train_all_symbols:
                base_mask &= df["symbol"].isin(spec.predict_symbols)
            train = df[base_mask & df["date"].lt(train_cut)].copy()
            valid = df[base_mask & df["date"].between(valid_start, train_end)].copy()
            test = df[
                df["date"].between(pd.Timestamp(year=year, month=1, day=1), pd.Timestamp(year=year, month=12, day=31))
                & df["symbol"].isin(spec.predict_symbols)
            ].copy()
            train = _labels(train.dropna(subset=[label_col(spec), opposite_label_col(spec)]), spec)
            valid = _labels(valid.dropna(subset=[label_col(spec), opposite_label_col(spec)]), spec)
            test = _labels(test.dropna(subset=[label_col(spec), opposite_label_col(spec)]), spec)
            if train.empty or valid.empty or test.empty:
                continue
            ranker = _fit_ranker(train, valid, features, 15000 + year, config)
            valid_scored = rule_gate(valid, train[train["symbol"].isin(spec.predict_symbols)], spec)
            test_scored = rule_gate(test, train[train["symbol"].isin(spec.predict_symbols)], spec)
            valid_scored["raw_rank_score"] = ranker.predict(valid_scored[features], num_iteration=ranker.best_iteration)
            test_scored["raw_rank_score"] = ranker.predict(test_scored[features], num_iteration=ranker.best_iteration)
            for frame in (valid_scored, test_scored):
                frame["model_id"] = spec.model_id
                frame["tier"] = spec.name
                frame["side"] = spec.side
                frame["threshold"] = spec.threshold
            rule, grid = choose_rule(valid_scored, config)
            grid["year"] = year
            grid["model_id"] = spec.model_id
            grids.append(grid)
            selected = apply_rule(test_scored, rule).copy()
            selected["prediction_year"] = year
            selected["rank_rule_min_score"] = rule["min_score"]
            selected["rank_rule_daily_top"] = rule["daily_top"]
            selected["rank_rule_require_gate"] = rule["require_gate"]
            selected_frames.append(selected)
            metric = _metrics(selected)
            metric.update(
                {
                    "year": year,
                    "model_id": spec.model_id,
                    "tier": spec.name,
                    "side": spec.side,
                    "threshold": spec.threshold,
                    "rule_min_score": rule["min_score"],
                    "rule_daily_top": rule["daily_top"],
                    "rule_require_gate": rule["require_gate"],
                    "train_end": train_end,
                }
            )
            summary_rows.append(metric)

    selected_df = pd.concat(selected_frames, ignore_index=True) if selected_frames else pd.DataFrame()
    summary = pd.DataFrame(summary_rows)
    grid_df = pd.concat(grids, ignore_index=True) if grids else pd.DataFrame()
    selected_df.to_csv(run_dir / "predictions" / f"{run_id}_selected.csv", index=False)
    selected_df.to_parquet(run_dir / "predictions" / f"{run_id}_selected.parquet", index=False)
    summary.to_csv(run_dir / "reports" / "walkforward_summary.csv", index=False)
    grid_df.to_csv(run_dir / "reports" / "walkforward_rule_grid.csv", index=False)
    summary.to_csv(REPORTS_DIR / "raw_rank_walkforward_summary.csv", index=False)
    selected_df.to_csv(REPORTS_DIR / "raw_rank_walkforward_selected.csv", index=False)
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "dataset_path": str(dataset_path),
                "config": config.__dict__,
                "features": features,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    return run_dir
