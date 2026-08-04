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
from koscine.tiered_clean_direction import apply_downside_production_lock


META_ROOT = MODEL_DIR / "meta_signal"


@dataclass(frozen=True)
class MetaLayerConfig:
    min_train_years: int = 2
    bad_rate_cap: float = 0.15
    min_train_calls: int = 25
    lgb_rounds: int = 500
    early_stopping_rounds: int = 50
    focal_gamma: float = 2.0


META_FEATURES = [
    "score",
    "model_score",
    "lgbm_score",
    "catboost_score",
    "rule_gate_strength",
    "rule_component_count",
    "rule_gate_pass_num",
    "same_side_rank_pct",
    "same_side_score_z",
    "opposite_score",
    "opposite_rule_gate_pass_num",
    "opposite_score_rank_pct",
    "score_gap_vs_opposite",
    "daily_downside_lock_count",
    "daily_model_mean_score",
    "daily_model_max_score",
    "daily_side_mean_score",
    "daily_side_max_score",
    "is_liquid30_model",
    "is_rest35_model",
    "is_up_model",
    "is_down_model",
    "threshold",
]


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _pair_keys(frame: pd.DataFrame) -> list[str]:
    keys = ["date", "symbol", "tier"]
    keys.extend([col for col in ("prediction_year", "train_end") if col in frame.columns])
    return keys


def add_meta_features(predictions: pd.DataFrame) -> pd.DataFrame:
    derived_cols = [
        "paired_up_score",
        "paired_up_rule_gate_pass",
        "opposite_score",
        "opposite_rule_gate_pass",
        "opposite_rule_gate_pass_num",
        "score_gap_vs_opposite",
        "rule_gate_pass_num",
        "same_side_rank_pct",
        "same_side_score_z",
        "daily_model_mean_score",
        "daily_model_max_score",
        "daily_side_mean_score",
        "daily_side_max_score",
        "opposite_score_rank_pct",
        "is_liquid30_model",
        "is_rest35_model",
        "is_up_model",
        "is_down_model",
        "meta_target_prob",
        "meta_safe_prob",
        "meta_risk_prob",
        "meta_final_score",
        "meta_signal",
        "meta_signal_profile",
    ]
    out = predictions.drop(columns=[col for col in derived_cols if col in predictions.columns]).copy()
    out = apply_downside_production_lock(out)
    out["date"] = pd.to_datetime(out["date"])
    keys = _pair_keys(out)

    up_pairs = out[out["side"].eq("up")][
        keys + ["score", "rule_gate_pass"]
    ].drop_duplicates(keys, keep="last").rename(
        columns={"score": "paired_up_score", "rule_gate_pass": "paired_up_rule_gate_pass"}
    )
    out = out.merge(up_pairs, on=keys, how="left")
    out["paired_up_rule_gate_pass"] = out["paired_up_rule_gate_pass"].fillna(False).astype(bool)
    out["opposite_score"] = np.where(out["side"].eq("up"), out["paired_down_score"], out["paired_up_score"])
    out["opposite_rule_gate_pass"] = np.where(
        out["side"].eq("up"),
        out["paired_down_rule_gate_pass"],
        out["paired_up_rule_gate_pass"],
    )
    out["opposite_score"] = out["opposite_score"].fillna(0.0)
    out["opposite_rule_gate_pass_num"] = out["opposite_rule_gate_pass"].fillna(False).astype(float)
    out["score_gap_vs_opposite"] = out["score"].astype(float) - out["opposite_score"].astype(float)
    out["rule_gate_pass_num"] = out["rule_gate_pass"].fillna(False).astype(float)

    group_model = out.groupby(["date", "model_id"])["score"]
    out["same_side_rank_pct"] = group_model.rank(method="average", ascending=False, pct=True)
    model_mean = group_model.transform("mean")
    model_std = group_model.transform("std").replace(0, np.nan)
    out["same_side_score_z"] = ((out["score"] - model_mean) / model_std).fillna(0.0)
    out["daily_model_mean_score"] = model_mean
    out["daily_model_max_score"] = group_model.transform("max")

    group_side = out.groupby(["date", "side"])["score"]
    out["daily_side_mean_score"] = group_side.transform("mean")
    out["daily_side_max_score"] = group_side.transform("max")
    out["opposite_score_rank_pct"] = (
        out.groupby(["date", "tier"])["opposite_score"].rank(method="average", ascending=False, pct=True)
    )

    out["is_liquid30_model"] = out["model_id"].str.startswith("liquid30").astype(float)
    out["is_rest35_model"] = out["model_id"].str.startswith("rest35").astype(float)
    out["is_up_model"] = out["side"].eq("up").astype(float)
    out["is_down_model"] = out["side"].eq("down").astype(float)
    for col in META_FEATURES:
        if col not in out.columns:
            out[col] = 0.0
    out[META_FEATURES] = out[META_FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


def _meta_labels(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["target_label"] = out["actual_hit"].fillna(False).astype(bool)
    out["risk_label"] = out["actual_opposite"].fillna(False).astype(bool)
    out["safe_label"] = out["signed_close_return_5d"].astype(float).gt(0) & ~out["risk_label"]
    return out


def _params(seed: int, positive_rate: float) -> dict:
    return {
        "objective": "binary",
        "metric": "average_precision",
        "boosting_type": "gbdt",
        "learning_rate": 0.035,
        "num_leaves": 15,
        "min_data_in_leaf": 80,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        "lambda_l1": 1.0,
        "lambda_l2": 8.0,
        "scale_pos_weight": (1.0 - positive_rate) / max(positive_rate, 1e-6),
        "verbosity": -1,
        "seed": seed,
    }


def _fit_binary(train: pd.DataFrame, valid: pd.DataFrame, label: str, seed: int, config: MetaLayerConfig | None = None) -> lgb.Booster:
    config = config or MetaLayerConfig()
    y = train[label].astype(int)
    params = _params(seed, float(y.mean()))
    weights = pd.Series(1.0, index=train.index)
    if label == "target_label":
        weights.loc[train["risk_label"].astype(bool)] *= 1.8
    elif label == "risk_label":
        weights.loc[train["risk_label"].astype(bool)] *= 2.5
    # Focal-style hard-example weighting: fit a small probe, then upweight rows
    # the probe finds difficult. This preserves LightGBM's stable binary
    # objective while applying the focal-loss idea to tabular boosting.
    probe_set = lgb.Dataset(train[META_FEATURES], label=y, weight=weights)
    probe = lgb.train(
        {**params, "learning_rate": 0.06, "num_leaves": 11},
        probe_set,
        num_boost_round=80,
        callbacks=[lgb.log_evaluation(0)],
    )
    p = pd.Series(probe.predict(train[META_FEATURES]), index=train.index).clip(1e-5, 1 - 1e-5)
    p_t = pd.Series(np.where(y.astype(bool), p, 1.0 - p), index=train.index)
    focal = (1.0 - p_t).pow(config.focal_gamma)
    if float(focal.mean()) > 0:
        focal = focal / float(focal.mean())
    weights = weights * focal.clip(lower=0.25, upper=5.0)
    train_set = lgb.Dataset(train[META_FEATURES], label=y, weight=weights)
    valid_set = lgb.Dataset(valid[META_FEATURES], label=valid[label].astype(int), reference=train_set)
    return lgb.train(
        params,
        train_set,
        valid_sets=[valid_set],
        num_boost_round=500,
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
    )


def _score_models(models: dict[str, lgb.Booster], frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["meta_target_prob"] = models["target"].predict(out[META_FEATURES], num_iteration=models["target"].best_iteration)
    out["meta_safe_prob"] = models["safe"].predict(out[META_FEATURES], num_iteration=models["safe"].best_iteration)
    out["meta_risk_prob"] = models["risk"].predict(out[META_FEATURES], num_iteration=models["risk"].best_iteration)
    out["meta_final_score"] = (
        out["meta_target_prob"].clip(0, 1)
        * out["meta_safe_prob"].clip(0, 1)
        * (1.0 - out["meta_risk_prob"].clip(0, 1))
    )
    return out


def _selected_metrics(frame: pd.DataFrame) -> dict:
    if frame.empty:
        return {
            "signals": 0,
            "hit_rate": np.nan,
            "bad_rate": np.nan,
            "positive_close_rate": np.nan,
            "avg_move": np.nan,
            "avg_final_score": np.nan,
        }
    return {
        "signals": int(len(frame)),
        "hit_rate": float(frame["target_label"].mean()),
        "bad_rate": float(frame["risk_label"].mean()),
        "positive_close_rate": float(frame["safe_label"].mean()),
        "avg_move": float(frame["actual_move"].mean()),
        "avg_final_score": float(frame["meta_final_score"].mean()),
    }


def choose_meta_rule(scored: pd.DataFrame, config: MetaLayerConfig) -> tuple[dict, pd.DataFrame]:
    rows = []
    if scored.empty:
        return {"min_score": 1.0, "daily_top": 1, "require_gate": False}, pd.DataFrame()
    scores = scored["meta_final_score"].dropna()
    for q in (0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.925, 0.95, 0.975):
        min_score = float(scores.quantile(q))
        for daily_top in (1, 2, 3, 5, 10, 20):
            for require_gate in (False, True):
                selected = scored[scored["meta_final_score"].ge(min_score)].copy()
                if require_gate:
                    selected = selected[selected["rule_gate_pass"]]
                selected = selected.sort_values(["date", "meta_final_score"], ascending=[True, False]).groupby("date").head(daily_top)
                rows.append(
                    {
                        "min_score": min_score,
                        "score_quantile": q,
                        "daily_top": daily_top,
                        "require_gate": require_gate,
                        **_selected_metrics(selected),
                    }
                )
    grid = pd.DataFrame(rows)
    eligible = grid[
        (grid["signals"] >= config.min_train_calls)
        & (grid["bad_rate"].fillna(1.0) <= config.bad_rate_cap)
    ].copy()
    if eligible.empty:
        eligible = grid[grid["signals"] >= max(5, config.min_train_calls // 2)].copy()
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


def apply_meta_rule(scored: pd.DataFrame, rule: dict) -> pd.DataFrame:
    selected = scored[scored["meta_final_score"].ge(float(rule["min_score"]))].copy()
    if rule.get("require_gate", False):
        selected = selected[selected["rule_gate_pass"]]
    if selected.empty:
        return selected
    return (
        selected.sort_values(["date", "meta_final_score"], ascending=[True, False])
        .groupby("date")
        .head(int(rule["daily_top"]))
        .sort_values(["date", "meta_final_score"], ascending=[True, False])
        .reset_index(drop=True)
    )


def run_meta_walkforward(
    scored_path: Path = PREDICTIONS_DIR / "tiered_clean_direction_scored_latest.parquet",
    config: MetaLayerConfig | None = None,
    run_name: str | None = None,
) -> Path:
    config = config or MetaLayerConfig()
    run_id = run_name or f"meta_signal_{timestamp()}"
    run_dir = META_ROOT / run_id
    (run_dir / "models").mkdir(parents=True, exist_ok=True)
    (run_dir / "reports").mkdir(parents=True, exist_ok=True)
    (run_dir / "predictions").mkdir(parents=True, exist_ok=True)

    base = pd.read_parquet(scored_path)
    frame = _meta_labels(add_meta_features(base))
    frame = frame[frame["split"].eq("test")].copy()
    years = sorted(frame["prediction_year"].dropna().astype(int).unique().tolist())

    selected_frames = []
    summary_rows = []
    rule_rows = []
    grids = []
    for year in years:
        train_years = [value for value in years if value < year]
        if len(train_years) < config.min_train_years:
            continue
        for model_id, model_frame in frame.groupby("model_id"):
            train = model_frame[model_frame["prediction_year"].isin(train_years)].copy()
            test = model_frame[model_frame["prediction_year"].eq(year)].copy()
            valid_year = max(train_years)
            inner = train[train["prediction_year"].lt(valid_year)].copy()
            valid = train[train["prediction_year"].eq(valid_year)].copy()
            if inner.empty or valid.empty or test.empty:
                continue
            models = {
                "target": _fit_binary(inner, valid, "target_label", 100 + year, config),
                "safe": _fit_binary(inner, valid, "safe_label", 200 + year, config),
                "risk": _fit_binary(inner, valid, "risk_label", 300 + year, config),
            }
            valid_scored = _score_models(models, valid)
            test_scored = _score_models(models, test)
            rule, grid = choose_meta_rule(valid_scored, config)
            grid["year"] = year
            grid["model_id"] = model_id
            grids.append(grid)
            selected = apply_meta_rule(test_scored, rule)
            selected["meta_rule_min_score"] = rule["min_score"]
            selected["meta_rule_daily_top"] = rule["daily_top"]
            selected["meta_rule_require_gate"] = rule["require_gate"]
            selected_frames.append(selected)
            metric = _selected_metrics(selected)
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
                }
            )
            summary_rows.append(metric)
            rule_rows.append({"year": year, "model_id": model_id, **rule})

    selected_df = pd.concat(selected_frames, ignore_index=True) if selected_frames else pd.DataFrame()
    summary = pd.DataFrame(summary_rows)
    rules = pd.DataFrame(rule_rows)
    grid_df = pd.concat(grids, ignore_index=True) if grids else pd.DataFrame()

    selected_df.to_parquet(run_dir / "predictions" / f"{run_id}_selected.parquet", index=False)
    selected_df.to_csv(run_dir / "predictions" / f"{run_id}_selected.csv", index=False)
    summary.to_csv(run_dir / "reports" / "walkforward_summary.csv", index=False)
    rules.to_csv(run_dir / "reports" / "walkforward_rules.csv", index=False)
    grid_df.to_csv(run_dir / "reports" / "walkforward_rule_grid.csv", index=False)
    summary.to_csv(REPORTS_DIR / "meta_signal_walkforward_summary.csv", index=False)
    selected_df.to_csv(REPORTS_DIR / "meta_signal_walkforward_selected.csv", index=False)

    manifest = {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "scored_path": str(scored_path),
        "config": config.__dict__,
        "features": META_FEATURES,
        "years": years,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return run_dir


def train_current_meta_layer(
    scored_path: Path = PREDICTIONS_DIR / "tiered_clean_direction_scored_latest.parquet",
    config: MetaLayerConfig | None = None,
    run_name: str | None = None,
) -> Path:
    config = config or MetaLayerConfig()
    run_id = run_name or f"meta_current_{timestamp()}"
    run_dir = META_ROOT / run_id
    (run_dir / "models").mkdir(parents=True, exist_ok=True)

    base = pd.read_parquet(scored_path)
    frame = _meta_labels(add_meta_features(base))
    frame = frame[frame["split"].eq("test")].copy()
    years = sorted(frame["prediction_year"].dropna().astype(int).unique().tolist())
    valid_year = max(years)
    metadata = []
    for model_id, model_frame in frame.groupby("model_id"):
        inner = model_frame[model_frame["prediction_year"].lt(valid_year)].copy()
        valid = model_frame[model_frame["prediction_year"].eq(valid_year)].copy()
        models = {
            "target": _fit_binary(inner, valid, "target_label", 501, config),
            "safe": _fit_binary(inner, valid, "safe_label", 502, config),
            "risk": _fit_binary(inner, valid, "risk_label", 503, config),
        }
        valid_scored = _score_models(models, valid)
        rule, grid = choose_meta_rule(valid_scored, config)
        model_files = {}
        for label, model in models.items():
            name = f"{model_id}_{label}.joblib"
            joblib.dump(model, run_dir / "models" / name)
            model_files[label] = name
        grid.to_csv(run_dir / f"{model_id}_rule_grid.csv", index=False)
        metadata.append(
            {
                "model_id": model_id,
                "tier": model_frame["tier"].iloc[0],
                "side": model_frame["side"].iloc[0],
                "threshold": float(model_frame["threshold"].iloc[0]),
                "models": model_files,
                "rule": rule,
                "train_years": [year for year in years if year < valid_year],
                "validation_year": valid_year,
                "validation_metrics": _selected_metrics(apply_meta_rule(valid_scored, rule)),
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
    current = META_ROOT / "current"
    if current.exists():
        shutil.rmtree(current)
    shutil.copytree(run_dir, current)
    return run_dir


def apply_current_meta_layer(predictions: pd.DataFrame, meta_dir: Path = META_ROOT / "current") -> pd.DataFrame:
    manifest_path = meta_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"No meta layer manifest found at {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    out = add_meta_features(predictions)
    out["meta_signal"] = False
    out["meta_signal_profile"] = "meta_v1_filtered_out"
    for meta in manifest["models"]:
        mask = out["model_id"].eq(meta["model_id"])
        if not mask.any():
            continue
        models = {
            label: joblib.load(meta_dir / "models" / filename)
            for label, filename in meta["models"].items()
        }
        scored = _score_models(models, out.loc[mask].copy())
        selected = apply_meta_rule(scored, meta["rule"])
        out.loc[scored.index, ["meta_target_prob", "meta_safe_prob", "meta_risk_prob", "meta_final_score"]] = scored[
            ["meta_target_prob", "meta_safe_prob", "meta_risk_prob", "meta_final_score"]
        ]
        if not selected.empty:
            out.loc[selected.index, "meta_signal"] = True
            out.loc[selected.index, "meta_signal_profile"] = "meta_signal_v1"
    return out
