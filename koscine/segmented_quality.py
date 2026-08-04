from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from koscine.config import (
    HORIZON_DAYS,
    PREDICTIONS_DIR,
    REPORTS_DIR,
    REST_UNIVERSE,
    RUNS_DIR,
    TARGET_UNIVERSE,
    TOP30_LIQUID_UNIVERSE,
)
from koscine.experiments import select_features
from koscine.rolling import known_training_rows, lgbm_params


SEGMENTS = {
    "top30_liquid": {
        "threshold": 0.04,
        "train_symbols": set(TOP30_LIQUID_UNIVERSE),
        "predict_symbols": set(TOP30_LIQUID_UNIVERSE),
    },
    "rest": {
        "threshold": 0.08,
        "train_symbols": None,
        "exclude_train_symbols": set(TOP30_LIQUID_UNIVERSE),
        "predict_symbols": set(REST_UNIVERSE),
    },
}


@dataclass(frozen=True)
class SegmentedQualityConfig:
    train_start_year: int = 2012
    start_test_year: int = 2018
    end_test_year: int = 2025
    feature_profile: str = "side_compact"
    validation_days: int = 365
    train_cutoff_month: int = 12
    train_cutoff_day: int = 20
    bad_rate_cap: float = 0.15
    topn_step: int = 5
    topn_max: int = 500
    calibration_start_year: int | None = None
    bad_negative_weight: float = 4.0
    soft_negative_weight: float = 0.35
    objective: str = "good_call"


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def segment_label_col(segment: str) -> str:
    pct = int(round(float(SEGMENTS[segment]["threshold"]) * 100))
    return f"label_{segment}_up_{pct}pct_{HORIZON_DAYS}d"


def segment_good_label_col(segment: str) -> str:
    pct = int(round(float(SEGMENTS[segment]["threshold"]) * 100))
    return f"label_{segment}_good_up_{pct}pct_{HORIZON_DAYS}d"


def add_segment_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    valid = (
        out[f"up_move_{HORIZON_DAYS}d"].notna()
        & out[f"fwd_return_{HORIZON_DAYS}d"].notna()
    )
    for segment, spec in SEGMENTS.items():
        threshold = float(spec["threshold"])
        label_col = segment_label_col(segment)
        good_col = segment_good_label_col(segment)
        target_hit = out[f"up_move_{HORIZON_DAYS}d"].ge(threshold)
        positive_close = out[f"fwd_return_{HORIZON_DAYS}d"].ge(0)
        out[label_col] = np.where(valid, out[f"up_move_{HORIZON_DAYS}d"].ge(threshold), np.nan)
        out[good_col] = np.where(valid, target_hit | positive_close, np.nan)
    return out


def segment_frame(df: pd.DataFrame, segment: str, for_prediction: bool = False) -> pd.DataFrame:
    spec = SEGMENTS[segment]
    if for_prediction:
        symbols = spec["predict_symbols"]
        return df[df["symbol"].isin(symbols)].copy()
    if spec.get("train_symbols") is not None:
        return df[df["symbol"].isin(spec["train_symbols"])].copy()
    exclude = spec.get("exclude_train_symbols", set())
    return df[~df["symbol"].isin(exclude)].copy()


def _train_end_for_year(year: int, config: SegmentedQualityConfig) -> pd.Timestamp:
    return pd.Timestamp(
        year=year - 1,
        month=config.train_cutoff_month,
        day=config.train_cutoff_day,
    )


def _sample_weight(train: pd.DataFrame, config: SegmentedQualityConfig) -> pd.Series:
    target_hit = train["target_hit_for_training"].astype(bool)
    bad = train["bad_for_training"].astype(bool)
    weights = pd.Series(1.0, index=train.index)
    if config.objective == "target_hit":
        weights.loc[bad & ~target_hit] = config.bad_negative_weight
        weights.loc[~bad & ~target_hit] = config.soft_negative_weight
    elif config.objective == "good_call":
        positive_close = train["positive_close_for_training"].astype(bool)
        weights.loc[target_hit] = 3.0
        weights.loc[positive_close & ~target_hit] = 1.0
        weights.loc[bad & ~target_hit] = config.bad_negative_weight
    else:
        raise ValueError(f"Unknown segmented objective: {config.objective}")
    return weights


def _fit_segment_model(
    train: pd.DataFrame,
    features: list[str],
    label_col: str,
    config: SegmentedQualityConfig,
) -> lgb.Booster:
    valid_cutoff = train["date"].max() - pd.Timedelta(days=config.validation_days)
    inner = train[train["date"] < valid_cutoff].copy()
    valid = train[train["date"] >= valid_cutoff].copy()
    train_weights = _sample_weight(inner, config)
    valid_weights = _sample_weight(valid, config)
    train_set = lgb.Dataset(inner[features], label=inner[label_col], weight=train_weights)
    valid_set = lgb.Dataset(valid[features], label=valid[label_col], weight=valid_weights, reference=train_set)
    params = {
        **lgbm_params(inner[label_col]),
        "learning_rate": 0.02,
        "min_data_in_leaf": 700,
        "lambda_l1": 3.0,
        "lambda_l2": 25.0,
    }
    return lgb.train(
        params,
        train_set,
        valid_sets=[valid_set],
        num_boost_round=2500,
        callbacks=[lgb.early_stopping(150), lgb.log_evaluation(0)],
    )


def _prepare_train(df: pd.DataFrame, segment: str, year: int, config: SegmentedQualityConfig) -> pd.DataFrame:
    label_col = (
        segment_good_label_col(segment)
        if config.objective == "good_call"
        else segment_label_col(segment)
    )
    threshold = float(SEGMENTS[segment]["threshold"])
    train_end = _train_end_for_year(year, config)
    frame = segment_frame(df, segment, for_prediction=False)
    train = known_training_rows(frame, config.train_start_year, train_end, label_col)
    train["target_hit_for_training"] = train[f"up_move_{HORIZON_DAYS}d"].ge(threshold)
    train["positive_close_for_training"] = train[f"fwd_return_{HORIZON_DAYS}d"].ge(0)
    train["bad_for_training"] = (
        ~train["target_hit_for_training"] & ~train["positive_close_for_training"]
    )
    return train


def score_year(
    df: pd.DataFrame,
    features: list[str],
    year: int,
    segment: str,
    config: SegmentedQualityConfig,
    run_dir: Path,
    run_stamp: str,
) -> pd.DataFrame:
    label_col = (
        segment_good_label_col(segment)
        if config.objective == "good_call"
        else segment_label_col(segment)
    )
    threshold = float(SEGMENTS[segment]["threshold"])
    train_end = _train_end_for_year(year, config)
    train = _prepare_train(df, segment, year, config)
    model = _fit_segment_model(train, features, label_col, config)

    model_id = f"{run_stamp}_{year}_{segment}_up_{int(round(threshold * 100))}pct"
    model_path = run_dir / "models" / f"{model_id}.txt"
    model.save_model(model_path)
    metadata = {
        "model_id": model_id,
        "prediction_year": year,
        "train_end": str(train_end.date()),
        "segment": segment,
        "side": "up",
        "threshold": threshold,
        "label_col": label_col,
        "objective": config.objective,
        "feature_profile": config.feature_profile,
        "features": features,
        "best_iteration": model.best_iteration,
        "train_rows": len(train),
        "bad_negative_weight": config.bad_negative_weight,
        "soft_negative_weight": config.soft_negative_weight,
        "model_file": model_path.name,
    }
    (run_dir / "models" / f"{model_id}.json").write_text(
        json.dumps(metadata, indent=2, default=str),
        encoding="utf-8",
    )

    test = df[df["date"].dt.year.eq(year)]
    test = segment_frame(test, segment, for_prediction=True)
    scored = test.dropna(subset=[label_col]).copy()
    scored["score"] = model.predict(scored[features], num_iteration=model.best_iteration)
    scored["prediction_year"] = year
    scored["train_end"] = train_end
    scored["segment"] = segment
    scored["threshold"] = threshold
    scored["target_hit"] = scored[f"up_move_{HORIZON_DAYS}d"].ge(threshold)
    scored["positive_close"] = scored[f"fwd_return_{HORIZON_DAYS}d"].ge(0)
    scored["bad"] = (~scored["target_hit"]) & (~scored["positive_close"])
    scored["acceptable"] = scored["target_hit"] | scored["positive_close"]
    keep = [
        "date",
        "symbol",
        "prediction_year",
        "train_end",
        "segment",
        "threshold",
        "close",
        "entry_1d_date",
        "entry_1d_open",
        f"future_{HORIZON_DAYS}d_high",
        f"future_{HORIZON_DAYS}d_close",
        f"future_{HORIZON_DAYS}d_date",
        f"up_move_{HORIZON_DAYS}d",
        f"fwd_return_{HORIZON_DAYS}d",
        "score",
        "target_hit",
        "positive_close",
        "acceptable",
        "bad",
    ]
    return scored[keep].copy()


def metrics(frame: pd.DataFrame) -> dict:
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


def apply_rule(frame: pd.DataFrame, segment: str, top_n: int) -> pd.DataFrame:
    selected = (
        frame[frame["segment"].eq(segment)]
        .sort_values(["score", "date", "symbol"], ascending=[False, True, True])
        .head(top_n)
        .copy()
    )
    selected["selection_rank"] = range(1, len(selected) + 1)
    selected["selection_rule"] = f"{segment}_top{top_n}_score"
    return selected


def calibrate_top_n(
    calibration: pd.DataFrame,
    segment: str,
    config: SegmentedQualityConfig,
) -> tuple[int, pd.DataFrame]:
    rows = []
    frame = calibration[calibration["segment"].eq(segment)].sort_values(
        ["score", "date", "symbol"], ascending=[False, True, True]
    )
    for top_n in range(config.topn_step, min(config.topn_max, len(frame)) + 1, config.topn_step):
        selected = frame.head(top_n)
        rows.append({"segment": segment, "top_n": top_n, **metrics(selected)})
    curve = pd.DataFrame(rows)
    eligible = curve[curve["bad_rate"].le(config.bad_rate_cap)]
    if eligible.empty:
        best = curve.sort_values(["bad_rate", "target_hit_rate", "calls"], ascending=[True, False, False]).iloc[0]
    else:
        best = eligible.sort_values(
            ["calls", "target_hit_rate", "avg_high_move"],
            ascending=[False, False, False],
        ).iloc[0]
    return int(best["top_n"]), curve


def run_segmented_quality(
    dataset_path: Path,
    config: SegmentedQualityConfig,
    run_name: str | None = None,
) -> Path:
    run_id = run_name or f"segmented_quality_{timestamp()}"
    run_dir = RUNS_DIR / run_id
    (run_dir / "models").mkdir(parents=True, exist_ok=True)
    (run_dir / "predictions").mkdir(parents=True, exist_ok=True)
    (run_dir / "reports").mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(dataset_path)
    df["date"] = pd.to_datetime(df["date"])
    df[f"future_{HORIZON_DAYS}d_date"] = pd.to_datetime(df[f"future_{HORIZON_DAYS}d_date"])
    df = add_segment_labels(df)
    features = select_features(df, "up", config.feature_profile)

    run_stamp = run_dir.name
    manifest = {
        "run_dir": str(run_dir),
        "dataset_path": str(dataset_path),
        "config": config.__dict__,
        "segments": {
            name: {
                key: sorted(value) if isinstance(value, set) else value
                for key, value in spec.items()
            }
            for name, spec in SEGMENTS.items()
        },
        "features": features,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "objective": (
            "Two long-only models: top30_liquid uses +4%; rest uses +8%. "
            "Default objective is good_call = target hit or positive 5-day close; "
            "real target hits and red-close misses receive higher sample weights."
        ),
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))

    scored = []
    for year in range(config.start_test_year - 1, config.end_test_year + 1):
        for segment in SEGMENTS:
            scored.append(score_year(df, features, year, segment, config, run_dir, run_stamp))
    predictions = pd.concat(scored, ignore_index=True)
    pred_path = run_dir / "predictions" / f"{run_stamp}_all_year_predictions.parquet"
    predictions.to_parquet(pred_path, index=False)

    rule_rows = []
    curve_frames = []
    selected_frames = []
    for test_year in range(config.start_test_year, config.end_test_year + 1):
        cal_start = config.calibration_start_year or (config.start_test_year - 1)
        cal_end = test_year - 1
        calibration = predictions[predictions["prediction_year"].between(cal_start, cal_end)]
        test = predictions[predictions["prediction_year"].eq(test_year)]
        for segment in SEGMENTS:
            top_n, curve = calibrate_top_n(calibration, segment, config)
            curve["calibration_start_year"] = cal_start
            curve["calibration_end_year"] = cal_end
            curve["test_year"] = test_year
            curve_frames.append(curve)
            selected = apply_rule(test, segment, top_n)
            selected["test_year"] = test_year
            selected["calibration_start_year"] = cal_start
            selected["calibration_end_year"] = cal_end
            selected["calibrated_bad_cap"] = config.bad_rate_cap
            selected_frames.append(selected)
            cal_metrics = metrics(apply_rule(calibration, segment, top_n))
            test_metrics = metrics(selected)
            rule_rows.append(
                {
                    "test_year": test_year,
                    "calibration_start_year": cal_start,
                    "calibration_end_year": cal_end,
                    "segment": segment,
                    "threshold": SEGMENTS[segment]["threshold"],
                    "top_n": top_n,
                    **{f"cal_{key}": value for key, value in cal_metrics.items()},
                    **{f"test_{key}": value for key, value in test_metrics.items()},
                }
            )

    selected = pd.concat(selected_frames, ignore_index=True)
    rules = pd.DataFrame(rule_rows)
    curves = pd.concat(curve_frames, ignore_index=True)
    stability = (
        rules.groupby(["segment", "threshold"])
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
    stability["overall_target_hit_rate"] = stability["total_target_hits"] / stability[
        "total_calls"
    ].replace(0, np.nan)
    stability["overall_acceptable_rate"] = stability["total_acceptable"] / stability[
        "total_calls"
    ].replace(0, np.nan)
    stability["overall_bad_rate"] = stability["total_bad"] / stability["total_calls"].replace(0, np.nan)

    selected.to_csv(run_dir / "predictions" / f"{run_stamp}_selected_signals.csv", index=False)
    rules.to_csv(run_dir / "reports" / f"{run_stamp}_yearly_rules.csv", index=False)
    curves.to_csv(run_dir / "reports" / f"{run_stamp}_calibration_curves.csv", index=False)
    stability.to_csv(run_dir / "reports" / f"{run_stamp}_stability.csv", index=False)

    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    selected.to_csv(PREDICTIONS_DIR / "segmented_quality_selected_latest.csv", index=False)
    rules.to_csv(REPORTS_DIR / "segmented_quality_yearly_rules_latest.csv", index=False)
    stability.to_csv(REPORTS_DIR / "segmented_quality_stability_latest.csv", index=False)
    curves.to_csv(REPORTS_DIR / "segmented_quality_calibration_curves_latest.csv", index=False)
    return run_dir
