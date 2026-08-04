from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from koscine.config import MODEL_DIR, PREDICTIONS_DIR, REPORTS_DIR
from koscine.meta_signal_layer import META_FEATURES, add_meta_features


RANK_ROOT = MODEL_DIR / "rank_signal"


@dataclass(frozen=True)
class RankLayerConfig:
    min_train_years: int = 2
    purge_days: int = 7
    bad_rate_cap: float = 0.15
    min_validation_calls: int = 20
    num_boost_round: int = 600
    early_stopping_rounds: int = 60


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _labels(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["target_label"] = out["actual_hit"].fillna(False).astype(bool)
    out["risk_label"] = out["actual_opposite"].fillna(False).astype(bool)
    out["safe_label"] = out["signed_close_return_5d"].astype(float).gt(0) & ~out["risk_label"]
    out["rank_relevance"] = 0
    out.loc[out["safe_label"], "rank_relevance"] = 1
    out.loc[out["target_label"], "rank_relevance"] = 3
    out.loc[out["risk_label"], "rank_relevance"] = 0
    return out


def purged_train_valid_split(frame: pd.DataFrame, valid_year: int, purge_days: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    valid = frame[frame["prediction_year"].eq(valid_year)].copy()
    if valid.empty:
        return frame.iloc[0:0].copy(), valid
    valid_start = valid["date"].min()
    train = frame[frame["date"].lt(valid_start - pd.Timedelta(days=purge_days))].copy()
    return train, valid


def _sort_for_rank(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values(["date", "symbol"]).reset_index(drop=True)


def _groups(frame: pd.DataFrame) -> list[int]:
    return frame.groupby("date", sort=False).size().astype(int).tolist()


def _fit_ranker(train: pd.DataFrame, valid: pd.DataFrame, seed: int, config: RankLayerConfig) -> lgb.Booster:
    train = _sort_for_rank(train)
    valid = _sort_for_rank(valid)
    train_set = lgb.Dataset(
        train[META_FEATURES],
        label=train["rank_relevance"].astype(int),
        group=_groups(train),
    )
    valid_set = lgb.Dataset(
        valid[META_FEATURES],
        label=valid["rank_relevance"].astype(int),
        group=_groups(valid),
        reference=train_set,
    )
    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [1, 3, 5, 10],
        "label_gain": [0, 1, 3, 7],
        "learning_rate": 0.035,
        "num_leaves": 15,
        "min_data_in_leaf": 50,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        "lambda_l1": 1.0,
        "lambda_l2": 10.0,
        "verbosity": -1,
        "seed": seed,
    }
    return lgb.train(
        params,
        train_set,
        valid_sets=[valid_set],
        num_boost_round=config.num_boost_round,
        callbacks=[lgb.early_stopping(config.early_stopping_rounds), lgb.log_evaluation(0)],
    )


def _fit_calibrators(valid: pd.DataFrame, raw_col: str = "rank_raw_score") -> dict[str, IsotonicRegression]:
    calibrators = {}
    x = valid[raw_col].astype(float).to_numpy()
    for name, label in {
        "target": "target_label",
        "safe": "safe_label",
        "risk": "risk_label",
    }.items():
        y = valid[label].astype(int).to_numpy()
        if len(np.unique(y)) < 2:
            # Constant labels cannot be isotonic-fitted; use a degenerate
            # isotonic mapping around the base rate.
            base = float(y.mean()) if len(y) else 0.0
            calibrator = IsotonicRegression(out_of_bounds="clip")
            calibrator.fit(np.array([x.min() - 1, x.max() + 1]), np.array([base, base]))
        else:
            calibrator = IsotonicRegression(out_of_bounds="clip")
            calibrator.fit(x, y)
        calibrators[name] = calibrator
    return calibrators


def _score_ranker(model: lgb.Booster, calibrators: dict[str, IsotonicRegression], frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["rank_raw_score"] = model.predict(out[META_FEATURES], num_iteration=model.best_iteration)
    out["rank_target_prob"] = calibrators["target"].predict(out["rank_raw_score"])
    out["rank_safe_prob"] = calibrators["safe"].predict(out["rank_raw_score"])
    out["rank_risk_prob"] = calibrators["risk"].predict(out["rank_raw_score"])
    out["rank_final_score"] = (
        out["rank_target_prob"].clip(0, 1)
        * out["rank_safe_prob"].clip(0, 1)
        * (1.0 - out["rank_risk_prob"].clip(0, 1))
    )
    return out


def _metrics(frame: pd.DataFrame) -> dict:
    if frame.empty:
        return {
            "signals": 0,
            "hit_rate": np.nan,
            "bad_rate": np.nan,
            "positive_close_rate": np.nan,
            "avg_move": np.nan,
            "avg_rank_final_score": np.nan,
        }
    return {
        "signals": int(len(frame)),
        "hit_rate": float(frame["target_label"].mean()),
        "bad_rate": float(frame["risk_label"].mean()),
        "positive_close_rate": float(frame["safe_label"].mean()),
        "avg_move": float(frame["actual_move"].mean()),
        "avg_rank_final_score": float(frame["rank_final_score"].mean()),
    }


def choose_rank_rule(valid: pd.DataFrame, config: RankLayerConfig) -> tuple[dict, pd.DataFrame]:
    rows = []
    if valid.empty:
        return {"min_score": 1.0, "daily_top": 1, "require_gate": False}, pd.DataFrame()
    scores = valid["rank_final_score"].dropna()
    for q in (0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.925, 0.95, 0.975):
        min_score = float(scores.quantile(q))
        for daily_top in (1, 2, 3, 5, 10, 20):
            for require_gate in (False, True):
                selected = valid[valid["rank_final_score"].ge(min_score)].copy()
                if require_gate:
                    selected = selected[selected["rule_gate_pass"]]
                selected = (
                    selected.sort_values(["date", "rank_final_score"], ascending=[True, False])
                    .groupby("date")
                    .head(daily_top)
                )
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


def apply_rank_rule(scored: pd.DataFrame, rule: dict) -> pd.DataFrame:
    selected = scored[scored["rank_final_score"].ge(float(rule["min_score"]))].copy()
    if rule.get("require_gate", False):
        selected = selected[selected["rule_gate_pass"]]
    if selected.empty:
        return selected
    return (
        selected.sort_values(["date", "rank_final_score"], ascending=[True, False])
        .groupby("date")
        .head(int(rule["daily_top"]))
        .sort_values(["date", "rank_final_score"], ascending=[True, False])
        .reset_index(drop=True)
    )


def _prepare_scored(scored_path: Path) -> pd.DataFrame:
    base = pd.read_parquet(scored_path)
    frame = _labels(add_meta_features(base))
    frame = frame[frame["split"].eq("test")].copy()
    frame["date"] = pd.to_datetime(frame["date"])
    return frame


def run_rank_walkforward(
    scored_path: Path = PREDICTIONS_DIR / "tiered_clean_direction_scored_latest.parquet",
    config: RankLayerConfig | None = None,
    run_name: str | None = None,
) -> Path:
    config = config or RankLayerConfig()
    run_id = run_name or f"rank_signal_{timestamp()}"
    run_dir = RANK_ROOT / run_id
    (run_dir / "reports").mkdir(parents=True, exist_ok=True)
    (run_dir / "predictions").mkdir(parents=True, exist_ok=True)

    frame = _prepare_scored(scored_path)
    years = sorted(frame["prediction_year"].dropna().astype(int).unique().tolist())
    selected_frames = []
    summary_rows = []
    grids = []
    for year in years:
        train_years = [value for value in years if value < year]
        if len(train_years) < config.min_train_years:
            continue
        valid_year = max(train_years)
        for model_id, model_frame in frame.groupby("model_id"):
            train_pool = model_frame[model_frame["prediction_year"].isin(train_years)].copy()
            inner, valid = purged_train_valid_split(train_pool, valid_year, config.purge_days)
            test = model_frame[model_frame["prediction_year"].eq(year)].copy()
            if inner.empty or valid.empty or test.empty:
                continue
            ranker = _fit_ranker(inner, valid, 9000 + year, config)
            valid_raw = valid.copy()
            valid_raw["rank_raw_score"] = ranker.predict(valid_raw[META_FEATURES], num_iteration=ranker.best_iteration)
            calibrators = _fit_calibrators(valid_raw)
            valid_scored = _score_ranker(ranker, calibrators, valid)
            test_scored = _score_ranker(ranker, calibrators, test)
            rule, grid = choose_rank_rule(valid_scored, config)
            grid["year"] = year
            grid["model_id"] = model_id
            grids.append(grid)
            selected = apply_rank_rule(test_scored, rule)
            selected["rank_rule_min_score"] = rule["min_score"]
            selected["rank_rule_daily_top"] = rule["daily_top"]
            selected["rank_rule_require_gate"] = rule["require_gate"]
            selected_frames.append(selected)
            metric = _metrics(selected)
            metric.update(
                {
                    "year": year,
                    "model_id": model_id,
                    "tier": test["tier"].iloc[0],
                    "side": test["side"].iloc[0],
                    "threshold": float(test["threshold"].iloc[0]),
                    "rule_min_score": rule["min_score"],
                    "rule_daily_top": rule["daily_top"],
                    "rule_require_gate": rule["require_gate"],
                    "train_years": ",".join(map(str, train_years)),
                    "valid_year": valid_year,
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
    summary.to_csv(REPORTS_DIR / "rank_signal_walkforward_summary.csv", index=False)
    selected_df.to_csv(REPORTS_DIR / "rank_signal_walkforward_selected.csv", index=False)
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "scored_path": str(scored_path),
                "config": config.__dict__,
                "features": META_FEATURES,
                "years": years,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    return run_dir


def train_current_rank_layer(
    scored_path: Path = PREDICTIONS_DIR / "tiered_clean_direction_scored_latest.parquet",
    config: RankLayerConfig | None = None,
    run_name: str | None = None,
) -> Path:
    config = config or RankLayerConfig()
    run_id = run_name or f"rank_current_{timestamp()}"
    run_dir = RANK_ROOT / run_id
    (run_dir / "models").mkdir(parents=True, exist_ok=True)

    frame = _prepare_scored(scored_path)
    years = sorted(frame["prediction_year"].dropna().astype(int).unique().tolist())
    valid_year = max(years)
    metadata = []
    for model_id, model_frame in frame.groupby("model_id"):
        inner, valid = purged_train_valid_split(model_frame, valid_year, config.purge_days)
        ranker = _fit_ranker(inner, valid, 12000, config)
        valid_raw = valid.copy()
        valid_raw["rank_raw_score"] = ranker.predict(valid_raw[META_FEATURES], num_iteration=ranker.best_iteration)
        calibrators = _fit_calibrators(valid_raw)
        valid_scored = _score_ranker(ranker, calibrators, valid)
        rule, grid = choose_rank_rule(valid_scored, config)
        model_file = f"{model_id}_ranker.joblib"
        cal_file = f"{model_id}_calibrators.joblib"
        joblib.dump(ranker, run_dir / "models" / model_file)
        joblib.dump(calibrators, run_dir / "models" / cal_file)
        grid.to_csv(run_dir / f"{model_id}_rule_grid.csv", index=False)
        metadata.append(
            {
                "model_id": model_id,
                "tier": model_frame["tier"].iloc[0],
                "side": model_frame["side"].iloc[0],
                "threshold": float(model_frame["threshold"].iloc[0]),
                "model_file": model_file,
                "calibrator_file": cal_file,
                "rule": rule,
                "validation_year": valid_year,
                "validation_metrics": _metrics(apply_rank_rule(valid_scored, rule)),
            }
        )

    manifest = {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "scored_path": str(scored_path),
        "config": config.__dict__,
        "features": META_FEATURES,
        "models": metadata,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    current = RANK_ROOT / "current"
    if current.exists():
        shutil.rmtree(current)
    shutil.copytree(run_dir, current)
    return run_dir


def apply_current_rank_layer(predictions: pd.DataFrame, rank_dir: Path = RANK_ROOT / "current") -> pd.DataFrame:
    manifest_path = rank_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"No rank layer manifest found at {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    out = add_meta_features(predictions)
    out["rank_signal"] = False
    out["rank_signal_profile"] = "rank_v1_filtered_out"
    for meta in manifest["models"]:
        mask = out["model_id"].eq(meta["model_id"])
        if not mask.any():
            continue
        ranker = joblib.load(rank_dir / "models" / meta["model_file"])
        calibrators = joblib.load(rank_dir / "models" / meta["calibrator_file"])
        scored = _score_ranker(ranker, calibrators, out.loc[mask].copy())
        selected = apply_rank_rule(scored, meta["rule"])
        cols = ["rank_raw_score", "rank_target_prob", "rank_safe_prob", "rank_risk_prob", "rank_final_score"]
        out.loc[scored.index, cols] = scored[cols]
        if not selected.empty:
            out.loc[selected.index, "rank_signal"] = True
            out.loc[selected.index, "rank_signal_profile"] = "rank_signal_v1"
    return out
