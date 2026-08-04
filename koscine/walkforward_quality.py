from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from koscine.config import HORIZON_DAYS, PREDICTIONS_DIR, REPORTS_DIR, RUNS_DIR, TARGET_UNIVERSE
from koscine.experiments import select_features
from koscine.rolling import known_training_rows, lgbm_params


COMBO_ALPHAS = (0.5, 0.65, 0.8, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0)
BAD_FILTERS = (0.35, 0.4, 0.45, 0.5, 0.55)
TARGET_FILTERS = (0.6, 0.65, 0.7, 0.72, 0.75, 0.78, 0.8)


@dataclass(frozen=True)
class WalkForwardQualityConfig:
    train_start_year: int = 2012
    start_test_year: int = 2018
    end_test_year: int = 2025
    feature_profile: str = "side_compact"
    thresholds: tuple[float, ...] = (0.05, 0.07)
    validation_days: int = 365
    train_cutoff_month: int = 12
    train_cutoff_day: int = 20
    bad_rate_cap: float = 0.15
    topn_step: int = 5
    topn_max: int = 500
    calibration_lookback_years: int | None = None


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def badclose_label_col(threshold: float) -> str:
    pct = int(round(threshold * 100))
    return f"label_badclose_up_{pct}pct_{HORIZON_DAYS}d"


def ensure_badclose_labels(df: pd.DataFrame, thresholds: tuple[float, ...]) -> pd.DataFrame:
    out = df.copy()
    valid = (
        out[f"up_move_{HORIZON_DAYS}d"].notna()
        & out[f"fwd_return_{HORIZON_DAYS}d"].notna()
    )
    for threshold in thresholds:
        col = badclose_label_col(threshold)
        target_hit = out[f"up_move_{HORIZON_DAYS}d"].ge(threshold)
        close_negative = out[f"fwd_return_{HORIZON_DAYS}d"].lt(0)
        out[col] = np.where(valid, (~target_hit) & close_negative, np.nan)
    return out


def _fit_model(
    train: pd.DataFrame,
    features: list[str],
    label_col: str,
    validation_days: int,
) -> lgb.Booster:
    valid_cutoff = train["date"].max() - pd.Timedelta(days=validation_days)
    inner = train[train["date"] < valid_cutoff].copy()
    valid = train[train["date"] >= valid_cutoff].copy()
    train_set = lgb.Dataset(inner[features], label=inner[label_col])
    valid_set = lgb.Dataset(valid[features], label=valid[label_col], reference=train_set)
    return lgb.train(
        lgbm_params(inner[label_col]),
        train_set,
        valid_sets=[valid_set],
        num_boost_round=2500,
        callbacks=[lgb.early_stopping(150), lgb.log_evaluation(0)],
    )


def _year_train_end(year: int, config: WalkForwardQualityConfig) -> pd.Timestamp:
    return pd.Timestamp(
        year=year - 1,
        month=config.train_cutoff_month,
        day=config.train_cutoff_day,
    )


def _score_year(
    df: pd.DataFrame,
    features: list[str],
    year: int,
    threshold: float,
    config: WalkForwardQualityConfig,
    run_dir: Path,
    run_stamp: str,
) -> pd.DataFrame:
    pct = int(round(threshold * 100))
    train_end = _year_train_end(year, config)
    year_tag = str(year)

    test = df[
        df["date"].dt.year.eq(year) & df["symbol"].isin(TARGET_UNIVERSE)
    ].copy()
    target_label = f"label_up_{pct}pct_{HORIZON_DAYS}d"
    bad_label = badclose_label_col(threshold)

    scored = test.dropna(subset=[target_label, bad_label]).copy()
    scored["threshold"] = threshold
    scored["target_hit"] = scored[f"up_move_{HORIZON_DAYS}d"].ge(threshold)
    scored["positive_close"] = scored[f"fwd_return_{HORIZON_DAYS}d"].ge(0)
    scored["bad"] = (~scored["target_hit"]) & (~scored["positive_close"])
    scored["acceptable"] = scored["target_hit"] | scored["positive_close"]

    for objective, label_col, score_col in (
        ("target", target_label, "target_score"),
        ("badclose", bad_label, "badclose_score"),
    ):
        train = known_training_rows(df, config.train_start_year, train_end, label_col)
        model = _fit_model(train, features, label_col, config.validation_days)
        model_id = f"{run_stamp}_{year_tag}_up_{pct}pct_{objective}"
        model_path = run_dir / "models" / f"{model_id}.txt"
        model.save_model(model_path)
        metadata = {
            "model_id": model_id,
            "prediction_year": year,
            "train_end": str(train_end.date()),
            "side": "up",
            "threshold": threshold,
            "objective": objective,
            "label_col": label_col,
            "feature_profile": config.feature_profile,
            "features": features,
            "best_iteration": model.best_iteration,
            "train_rows": len(train),
            "model_file": model_path.name,
        }
        (run_dir / "models" / f"{model_id}.json").write_text(
            json.dumps(metadata, indent=2, default=str),
            encoding="utf-8",
        )
        scored[score_col] = model.predict(scored[features], num_iteration=model.best_iteration)

    scored["safe_score"] = 1.0 - scored["badclose_score"]
    for alpha in COMBO_ALPHAS:
        scored[f"combo_a{alpha}"] = scored["target_score"] * (scored["safe_score"] ** alpha)
    scored["prediction_year"] = year
    scored["train_end"] = train_end
    keep_cols = [
        "date",
        "symbol",
        "prediction_year",
        "train_end",
        "threshold",
        "close",
        "entry_1d_date",
        "entry_1d_open",
        f"future_{HORIZON_DAYS}d_high",
        f"future_{HORIZON_DAYS}d_close",
        f"future_{HORIZON_DAYS}d_date",
        f"up_move_{HORIZON_DAYS}d",
        f"fwd_return_{HORIZON_DAYS}d",
        "target_hit",
        "positive_close",
        "acceptable",
        "bad",
        "target_score",
        "badclose_score",
        "safe_score",
    ]
    keep_cols += [f"combo_a{alpha}" for alpha in COMBO_ALPHAS]
    return scored[keep_cols].copy()


def _selection_metrics(frame: pd.DataFrame) -> dict:
    if frame.empty:
        return {
            "calls": 0,
            "target_hits": 0,
            "target_hit_rate": np.nan,
            "acceptable_count": 0,
            "acceptable_rate": np.nan,
            "bad_count": 0,
            "bad_rate": np.nan,
            "avg_high_move": np.nan,
            "median_high_move": np.nan,
            "p10_high_move": np.nan,
            "avg_close_return": np.nan,
        }
    return {
        "calls": len(frame),
        "target_hits": int(frame["target_hit"].sum()),
        "target_hit_rate": float(frame["target_hit"].mean()),
        "acceptable_count": int(frame["acceptable"].sum()),
        "acceptable_rate": float(frame["acceptable"].mean()),
        "bad_count": int(frame["bad"].sum()),
        "bad_rate": float(frame["bad"].mean()),
        "avg_high_move": float(frame[f"up_move_{HORIZON_DAYS}d"].mean()),
        "median_high_move": float(frame[f"up_move_{HORIZON_DAYS}d"].median()),
        "p10_high_move": float(frame[f"up_move_{HORIZON_DAYS}d"].quantile(0.1)),
        "avg_close_return": float(frame[f"fwd_return_{HORIZON_DAYS}d"].mean()),
    }


def _apply_rule(frame: pd.DataFrame, rule: dict) -> pd.DataFrame:
    sub = frame[frame["threshold"].eq(rule["threshold"])].copy()
    filters = rule.get("filters", {})
    if "badclose_score_lte" in filters:
        sub = sub[sub["badclose_score"].le(filters["badclose_score_lte"])]
    if "target_score_gte" in filters:
        sub = sub[sub["target_score"].ge(filters["target_score_gte"])]
    score_col = rule["score_col"]
    sub = sub.sort_values(
        [score_col, "target_score", "date", "symbol"],
        ascending=[False, False, True, True],
    ).head(rule["top_n"])
    sub = sub.copy()
    sub["selection_rule"] = rule["name"]
    sub["selection_score_col"] = score_col
    sub["selection_score"] = sub[score_col]
    sub["selection_rank"] = range(1, len(sub) + 1)
    return sub


def _candidate_rules(threshold: float, topn_step: int, topn_max: int) -> list[dict]:
    rules = []
    score_cols = ["target_score"] + [f"combo_a{alpha}" for alpha in COMBO_ALPHAS]
    for score_col in score_cols:
        for top_n in range(topn_step, topn_max + topn_step, topn_step):
            rules.append(
                {
                    "name": f"top{top_n}_{score_col}",
                    "threshold": threshold,
                    "selector": f"year_top_{top_n}",
                    "score_col": score_col,
                    "top_n": top_n,
                    "filters": {},
                }
            )
    for bad_limit in BAD_FILTERS:
        for target_floor in TARGET_FILTERS:
            for score_col in ("target_score", "combo_a0.5", "combo_a1.0", "combo_a2.0"):
                rules.append(
                    {
                        "name": (
                            f"bad<={bad_limit:.2f}_target>={target_floor:.2f}_{score_col}"
                        ),
                        "threshold": threshold,
                        "selector": f"bad<={bad_limit:.2f}_target>={target_floor:.2f}",
                        "score_col": score_col,
                        "top_n": topn_max,
                        "filters": {
                            "badclose_score_lte": bad_limit,
                            "target_score_gte": target_floor,
                        },
                    }
                )
    return rules


def calibrate_rule(
    calibration: pd.DataFrame,
    threshold: float,
    config: WalkForwardQualityConfig,
) -> tuple[dict, pd.DataFrame]:
    rows = []
    calibration = calibration[calibration["threshold"].eq(threshold)].copy()
    score_cols = ["target_score"] + [f"combo_a{alpha}" for alpha in COMBO_ALPHAS]
    max_rows = min(config.topn_max, len(calibration))

    for score_col in score_cols:
        ranked = calibration.sort_values(
            [score_col, "target_score", "date", "symbol"],
            ascending=[False, False, True, True],
        ).head(config.topn_max)
        for top_n in range(config.topn_step, max_rows + 1, config.topn_step):
            selected = ranked.head(top_n)
            metrics = _selection_metrics(selected)
            rows.append(
                {
                    "name": f"top{top_n}_{score_col}",
                    "threshold": threshold,
                    "selector": f"year_top_{top_n}",
                    "score_col": score_col,
                    "top_n": top_n,
                    "filters": {},
                    **metrics,
                }
            )

    for bad_limit in BAD_FILTERS:
        for target_floor in TARGET_FILTERS:
            filtered = calibration[
                calibration["badclose_score"].le(bad_limit)
                & calibration["target_score"].ge(target_floor)
            ]
            if filtered.empty:
                continue
            for score_col in ("target_score", "combo_a0.5", "combo_a1.0", "combo_a2.0"):
                selected = filtered.sort_values(
                    [score_col, "target_score", "date", "symbol"],
                    ascending=[False, False, True, True],
                ).head(config.topn_max)
                metrics = _selection_metrics(selected)
                rows.append(
                    {
                        "name": f"bad<={bad_limit:.2f}_target>={target_floor:.2f}_{score_col}",
                        "threshold": threshold,
                        "selector": f"bad<={bad_limit:.2f}_target>={target_floor:.2f}",
                        "score_col": score_col,
                        "top_n": len(selected),
                        "filters": {
                            "badclose_score_lte": bad_limit,
                            "target_score_gte": target_floor,
                        },
                        **metrics,
                    }
                )

    curve = pd.DataFrame(rows)
    eligible = curve[curve["bad_rate"].le(config.bad_rate_cap)].copy()
    if not eligible.empty:
        row = eligible.sort_values(
            ["calls", "target_hit_rate", "avg_high_move", "acceptable_rate"],
            ascending=[False, False, False, False],
        ).iloc[0]
        best_rule = {
            "name": row["name"],
            "threshold": threshold,
            "selector": row["selector"],
            "score_col": row["score_col"],
            "top_n": int(row["top_n"]),
            "filters": row["filters"],
            "calibration_metrics": {
                key: row[key]
                for key in (
                    "calls",
                    "target_hits",
                    "target_hit_rate",
                    "acceptable_count",
                    "acceptable_rate",
                    "bad_count",
                    "bad_rate",
                    "avg_high_move",
                    "median_high_move",
                    "p10_high_move",
                    "avg_close_return",
                )
            },
        }
    else:
        eligible = curve.sort_values(
            ["bad_rate", "target_hit_rate", "calls"],
            ascending=[True, False, False],
        ).head(1)
        if eligible.empty:
            best_rule = {
                "name": f"none_{int(threshold * 100)}pct",
                "threshold": threshold,
                "selector": "none",
                "score_col": "target_score",
                "top_n": 0,
                "filters": {},
                "calibration_metrics": _selection_metrics(calibration.iloc[0:0]),
            }
        else:
            idx = int(eligible.index[0])
            row = curve.loc[idx].to_dict()
            best_rule = {
                "name": row["name"],
                "threshold": threshold,
                "selector": row["selector"],
                "score_col": row["score_col"],
                "top_n": int(row["top_n"]),
                "filters": row["filters"],
                "calibration_metrics": {
                    key: row[key]
                    for key in (
                        "calls",
                        "target_hits",
                        "target_hit_rate",
                        "acceptable_count",
                        "acceptable_rate",
                        "bad_count",
                        "bad_rate",
                        "avg_high_move",
                        "median_high_move",
                        "p10_high_move",
                        "avg_close_return",
                    )
                },
            }
    return best_rule, curve


def run_walkforward_quality(
    dataset_path: Path,
    config: WalkForwardQualityConfig,
    run_name: str | None = None,
) -> Path:
    run_id = run_name or f"walkforward_quality_{timestamp()}"
    run_dir = RUNS_DIR / run_id
    (run_dir / "models").mkdir(parents=True, exist_ok=True)
    (run_dir / "predictions").mkdir(parents=True, exist_ok=True)
    (run_dir / "reports").mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(dataset_path)
    df["date"] = pd.to_datetime(df["date"])
    df[f"future_{HORIZON_DAYS}d_date"] = pd.to_datetime(df[f"future_{HORIZON_DAYS}d_date"])
    df = ensure_badclose_labels(df, config.thresholds)
    features = select_features(df, "up", config.feature_profile)

    run_stamp = run_dir.name
    manifest = {
        "run_dir": str(run_dir),
        "dataset_path": str(dataset_path),
        "config": config.__dict__,
        "features": features,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "selection_rule": (
            "For each test year, calibrate max-call rule on prior out-of-sample "
            "years only, then apply unchanged to the test year."
        ),
        "bad_definition": "target not hit and 5-day close below t+1 open",
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))

    years_to_score = range(config.start_test_year - 1, config.end_test_year + 1)
    predictions = []
    for year in years_to_score:
        for threshold in config.thresholds:
            predictions.append(_score_year(df, features, year, threshold, config, run_dir, run_stamp))
    predictions_df = pd.concat(predictions, ignore_index=True)
    predictions_path = run_dir / "predictions" / f"{run_stamp}_all_year_predictions.parquet"
    predictions_df.to_parquet(predictions_path, index=False)

    rule_rows = []
    selected_frames = []
    curves = []
    for test_year in range(config.start_test_year, config.end_test_year + 1):
        calibration_end_year = test_year - 1
        if config.calibration_lookback_years is None:
            calibration_start_year = config.start_test_year - 1
        else:
            calibration_start_year = max(
                config.start_test_year - 1,
                test_year - config.calibration_lookback_years,
            )
        calibration = predictions_df[
            predictions_df["prediction_year"].between(calibration_start_year, calibration_end_year)
        ]
        test = predictions_df[predictions_df["prediction_year"].eq(test_year)]
        for threshold in config.thresholds:
            rule, curve = calibrate_rule(
                calibration[calibration["threshold"].eq(threshold)],
                threshold,
                config,
            )
            curve["calibration_start_year"] = calibration_start_year
            curve["calibration_end_year"] = calibration_end_year
            curve["test_year"] = test_year
            curve["threshold"] = threshold
            curves.append(curve)

            selected = _apply_rule(test, rule)
            selected["test_year"] = test_year
            selected["calibration_start_year"] = calibration_start_year
            selected["calibration_end_year"] = calibration_end_year
            selected["calibrated_bad_cap"] = config.bad_rate_cap
            selected_frames.append(selected)

            test_metrics = _selection_metrics(selected)
            cal_metrics = rule.get("calibration_metrics", {})
            rule_rows.append(
                {
                    "test_year": test_year,
                    "calibration_start_year": calibration_start_year,
                    "calibration_end_year": calibration_end_year,
                    "threshold": threshold,
                    "rule": rule["name"],
                    "selector": rule["selector"],
                    "score_col": rule["score_col"],
                    "top_n": rule["top_n"],
                    "filters": json.dumps(rule.get("filters", {}), sort_keys=True),
                    **{f"cal_{key}": value for key, value in cal_metrics.items()},
                    **{f"test_{key}": value for key, value in test_metrics.items()},
                }
            )

    selected_df = pd.concat(selected_frames, ignore_index=True) if selected_frames else pd.DataFrame()
    rules_df = pd.DataFrame(rule_rows)
    curves_df = pd.concat(curves, ignore_index=True) if curves else pd.DataFrame()

    selected_path = run_dir / "predictions" / f"{run_stamp}_selected_signals.csv"
    rules_path = run_dir / "reports" / f"{run_stamp}_yearly_rules_and_results.csv"
    curves_path = run_dir / "reports" / f"{run_stamp}_calibration_curves.csv"
    stability_path = run_dir / "reports" / f"{run_stamp}_stability_summary.csv"
    selected_df.to_csv(selected_path, index=False)
    rules_df.to_csv(rules_path, index=False)
    curves_df.to_csv(curves_path, index=False)

    stability = (
        rules_df.groupby("threshold")
        .agg(
            years=("test_year", "nunique"),
            total_calls=("test_calls", "sum"),
            avg_calls=("test_calls", "mean"),
            total_target_hits=("test_target_hits", "sum"),
            total_acceptable=("test_acceptable_count", "sum"),
            total_bad=("test_bad_count", "sum"),
            avg_year_bad_rate=("test_bad_rate", "mean"),
            max_year_bad_rate=("test_bad_rate", "max"),
            avg_year_target_hit_rate=("test_target_hit_rate", "mean"),
            avg_high_move=("test_avg_high_move", "mean"),
        )
        .reset_index()
    )
    stability["overall_target_hit_rate"] = (
        stability["total_target_hits"] / stability["total_calls"].replace(0, np.nan)
    )
    stability["overall_acceptable_rate"] = (
        stability["total_acceptable"] / stability["total_calls"].replace(0, np.nan)
    )
    stability["overall_bad_rate"] = stability["total_bad"] / stability["total_calls"].replace(0, np.nan)
    stability.to_csv(stability_path, index=False)

    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    selected_df.to_csv(PREDICTIONS_DIR / "walkforward_quality_selected_latest.csv", index=False)
    rules_df.to_csv(REPORTS_DIR / "walkforward_quality_yearly_rules_latest.csv", index=False)
    stability.to_csv(REPORTS_DIR / "walkforward_quality_stability_latest.csv", index=False)

    return run_dir
