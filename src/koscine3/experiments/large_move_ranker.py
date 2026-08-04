from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from koscine3.datasets.splits import DEFAULT_SPLITS, WalkForwardSplit, between_dates
from koscine3.evaluation.gold_metrics import build_gold_report
from koscine3.evaluation.reports import write_manifest, write_report_tables
from koscine3.paths import RUNS_DIR


try:
    from lightgbm import LGBMClassifier, LGBMRanker, LGBMRegressor

    HAS_LIGHTGBM = True
except Exception:
    LGBMClassifier = None
    LGBMRanker = None
    LGBMRegressor = None
    HAS_LIGHTGBM = False


LOCKED_SOURCE_RUN_ID = "koscine3_crossregime_v19_full_v8_n80"
LOCKED_SOURCE_RUN_DIR = RUNS_DIR / LOCKED_SOURCE_RUN_ID


@dataclass(frozen=True)
class LargeMoveRankerConfig:
    run_id: str = "koscine3_large_move_ranker_v1"
    source_run_id: str = LOCKED_SOURCE_RUN_ID
    train_start: str = "2018-01-01"
    n_estimators: int = 180
    learning_rate: float = 0.035
    num_leaves: int = 63
    min_child_samples: int = 80
    weekly_target: int = 6
    max_signals_per_day: int = 2
    max_signals_per_week_side: int = 4
    max_symbol_per_week: int = 1
    max_symbol_per_month: int = 3
    daily_pool_rank: int = 18
    selector_mode: str = "weekly_pool"
    setup_portfolio: str = ""
    setup_weekly_quota: int = 1
    rank_score_weight: float = 1.15
    clean_large_weight: float = 0.90
    hit_near_weight: float = 0.45
    floor_weight: float = 0.25
    opposite_penalty: float = 0.95
    expected_move_weight: float = 0.18
    historical_risk_penalty: float = 0.15
    year_symbol_penalty: float = 0.0
    global_symbol_penalty: float = 0.0
    random_state: int = 41


@dataclass
class SideRankerBundle:
    side: str
    ranker: Any
    floor_model: Any
    hit_near_model: Any
    clean_large_model: Any
    opposite_model: Any
    favorable_model: Any
    feature_columns: list[str]
    train_rows: int


@dataclass
class SplitRankerModel:
    model_id: str
    split: WalkForwardSplit
    config: LargeMoveRankerConfig
    long_bundle: SideRankerBundle
    short_bundle: SideRankerBundle


def _source_run_dir(config: LargeMoveRankerConfig) -> Path:
    return RUNS_DIR / config.source_run_id


def load_source_dataset(config: LargeMoveRankerConfig) -> tuple[pd.DataFrame, list[str]]:
    run_dir = _source_run_dir(config)
    dataset_path = run_dir / "dataset.parquet"
    manifest_path = run_dir / "dataset_manifest.json"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Missing source dataset: {dataset_path}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing source manifest: {manifest_path}")

    dataset = pd.read_parquet(dataset_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    feature_columns = [c for c in manifest["feature_columns"] if c in dataset.columns]
    return add_large_move_labels(dataset), feature_columns


def add_large_move_labels(dataset: pd.DataFrame) -> pd.DataFrame:
    df = dataset.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["floor_threshold"] = np.where(df["band"].eq("liquid"), 0.03, 0.06)
    df["floor_success"] = df["favorable_move"].ge(df["floor_threshold"])
    df["tradable_large"] = df["hit_or_near"].fillna(False) | df["floor_success"].fillna(False)
    df["clean_large"] = df["tradable_large"] & ~df["opposite"].fillna(False)

    rank_label = np.ones(len(df), dtype=np.int8)
    rank_label[df["opposite"].fillna(False).to_numpy()] = 0
    rank_label[df["floor_success"].fillna(False).to_numpy()] = 2
    near_clean = df["near"].fillna(False) & ~df["opposite"].fillna(False)
    rank_label[near_clean.to_numpy()] = 3
    rank_label[df["hit"].fillna(False).to_numpy()] = 4
    df["large_rank_label"] = rank_label

    quality_rank_label = np.zeros(len(df), dtype=np.int8)
    signed_close = pd.to_numeric(
        df.get("signed_close_return", pd.Series(0.0, index=df.index)),
        errors="coerce",
    ).fillna(0.0)
    small = df.get("small", pd.Series(False, index=df.index)).fillna(False)
    positive_small = small & signed_close.ge(0)
    quality_rank_label[positive_small.to_numpy()] = 1
    quality_rank_label[(df["floor_success"].fillna(False) & ~df["opposite"].fillna(False)).to_numpy()] = 2
    quality_rank_label[(near_clean).to_numpy()] = 3
    quality_rank_label[df["hit"].fillna(False).to_numpy()] = 5
    df["large_quality_rank_label"] = quality_rank_label

    df["large_move_grade"] = np.select(
        [
            df["hit"].fillna(False),
            near_clean,
            df["floor_success"].fillna(False) & ~df["opposite"].fillna(False),
            df["floor_success"].fillna(False) & df["opposite"].fillna(False),
            df["opposite"].fillna(False),
        ],
        ["contract_hit", "clean_near", "floor_clean", "floor_opposite", "opposite"],
        default="small",
    )
    return df


def add_atlas_prior_features(dataset: pd.DataFrame) -> pd.DataFrame:
    df = dataset.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "symbol", "side"]).copy()
    lag = 6

    def add_daily_priors(group_cols: list[str], prefix: str) -> None:
        nonlocal df
        daily = (
            df[df["status"].eq("evaluated")]
            .groupby(["date", *group_cols], dropna=False)
            .agg(
                clean_large=("clean_large", "mean"),
                hit_near=("hit_or_near", "mean"),
                floor_success=("floor_success", "mean"),
                opposite=("opposite", "mean"),
                favorable_move=("favorable_move", "mean"),
            )
            .reset_index()
            .sort_values([*group_cols, "date"])
        )
        grouped = daily.groupby(group_cols, dropna=False, sort=False)
        for window in (21, 63, 252):
            daily[f"{prefix}_clean_large_rate_{window}"] = grouped["clean_large"].transform(
                lambda s: s.shift(lag).rolling(window, min_periods=max(5, window // 4)).mean()
            )
            daily[f"{prefix}_hit_near_rate_{window}"] = grouped["hit_near"].transform(
                lambda s: s.shift(lag).rolling(window, min_periods=max(5, window // 4)).mean()
            )
            daily[f"{prefix}_floor_success_rate_{window}"] = grouped[
                "floor_success"
            ].transform(lambda s: s.shift(lag).rolling(window, min_periods=max(5, window // 4)).mean())
            daily[f"{prefix}_opposite_rate_{window}"] = grouped["opposite"].transform(
                lambda s: s.shift(lag).rolling(window, min_periods=max(5, window // 4)).mean()
            )
            daily[f"{prefix}_favorable_move_mean_{window}"] = grouped[
                "favorable_move"
            ].transform(lambda s: s.shift(lag).rolling(window, min_periods=max(5, window // 4)).mean())
        keep_cols = [
            "date",
            *group_cols,
            *[c for c in daily.columns if c.startswith(f"{prefix}_")],
        ]
        df = df.merge(daily[keep_cols], on=["date", *group_cols], how="left")

    add_daily_priors(["side"], "atlas_side")
    add_daily_priors(["side", "band"], "atlas_band_side")
    add_daily_priors(["side", "setup_id"], "atlas_setup_side")

    for window in (21, 63, 252):
        clean = f"atlas_setup_side_clean_large_rate_{window}"
        opp = f"atlas_setup_side_opposite_rate_{window}"
        if clean in df.columns and opp in df.columns:
            df[f"atlas_setup_side_edge_{window}"] = df[clean] - df[opp]
    return df


def add_setup_features(dataset: pd.DataFrame) -> pd.DataFrame:
    df = dataset.copy()

    def col(name: str, default: float = 0.0) -> pd.Series:
        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce").fillna(default)
        return pd.Series(default, index=df.index, dtype=float)

    ret20 = col("symbol_return_20d_rank_pct", 0.5)
    ret5 = col("symbol_return_5d_rank_pct", 0.5)
    atr_rank = col("atr_pct_14_cs_rank", 0.5)
    bb_rank = col("bb_width_20_cs_rank", 0.5)
    vol_ratio = col("vol_sma20_ratio_cs_rank", 0.5)
    hist_success = col("hist_hit_near_rate_252", 0.5)
    hist_opp = col("hist_opposite_rate_252", 0.35)
    breadth5 = col("xsec_positive_share_5d", 0.5)
    iv_rank = col("atm_iv", 0.0).groupby(df["date"]).rank(pct=True)
    oi_rank = col("fut_oi_z_60d", 0.0).groupby(df["date"]).rank(pct=True)
    compression = col("compression_composite", 0.0).groupby(df["date"]).rank(pct=True)
    dryup = col("volume_dryup_score", 0.0).groupby(df["date"]).rank(pct=True)

    side = df["side"].astype(str)
    side_aligned_momentum = np.where(side.eq("long"), ret20, 1.0 - ret20)
    side_aligned_momentum_5d = np.where(side.eq("long"), ret5, 1.0 - ret5)
    contra_stretch = np.where(side.eq("long"), 1.0 - ret20, ret20)
    side_breadth = np.where(side.eq("long"), breadth5, 1.0 - breadth5)

    df["setup_momentum_score"] = (
        0.65 * pd.Series(side_aligned_momentum, index=df.index)
        + 0.35 * pd.Series(side_aligned_momentum_5d, index=df.index)
    )
    df["setup_volatility_score"] = 0.45 * atr_rank + 0.35 * bb_rank + 0.20 * vol_ratio
    df["setup_iv_oi_score"] = 0.55 * iv_rank.fillna(0.0) + 0.45 * oi_rank.fillna(0.0)
    df["setup_compression_score"] = 0.60 * compression.fillna(0.0) + 0.40 * dryup.fillna(0.0)
    df["setup_reversal_score"] = 0.65 * pd.Series(contra_stretch, index=df.index) + 0.35 * atr_rank
    df["setup_quality_history_score"] = hist_success - 0.7 * hist_opp
    df["setup_breadth_score"] = pd.Series(side_breadth, index=df.index)

    setup_cols = [
        "setup_momentum_score",
        "setup_volatility_score",
        "setup_iv_oi_score",
        "setup_compression_score",
        "setup_reversal_score",
        "setup_quality_history_score",
        "setup_breadth_score",
    ]
    setup_names = [
        "momentum_breakout",
        "volatility_expansion",
        "iv_oi_impulse",
        "compression_release",
        "reversal_stretch",
        "quality_history",
        "market_breadth",
    ]
    setup_values = df[setup_cols].fillna(0.0).to_numpy()
    best_idx = np.argmax(setup_values, axis=1)
    df["setup_id"] = [setup_names[i] for i in best_idx]
    for name, col_name in zip(setup_names, setup_cols, strict=True):
        df[f"setup_is_{name}"] = df["setup_id"].eq(name).astype(int)
        df[f"{col_name}_x_large_target"] = df[col_name] * df["target_threshold"].astype(float)
    return df


def _feature_columns(base_features: list[str], dataset: pd.DataFrame) -> list[str]:
    setup_features = [
        c
        for c in dataset.columns
        if (c.startswith("setup_") or c.startswith("atlas_")) and c != "setup_id"
    ]
    passthrough_blocklist = {
        "threshold",
        "entry_open",
        "window_high",
        "window_low",
        "window_close",
        "window_observations",
    }
    columns = [c for c in base_features if c in dataset.columns and c not in passthrough_blocklist]
    columns.extend([c for c in setup_features if c not in columns])
    return columns


def write_large_move_atlas(
    dataset: pd.DataFrame,
    feature_columns: list[str],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    df = dataset[dataset["status"].eq("evaluated")].copy()
    df["year"] = df["date"].dt.year.astype(str)
    df["quarter"] = df["date"].dt.to_period("Q").astype(str)
    df["month"] = df["date"].dt.to_period("M").astype(str)

    def summarize(group_cols: list[str]) -> pd.DataFrame:
        grouped = df.groupby(group_cols, dropna=False)
        rows = []
        for key, group in grouped:
            if not isinstance(key, tuple):
                key = (key,)
            rows.append(
                {
                    **dict(zip(group_cols, key, strict=True)),
                    "rows": int(len(group)),
                    "contract_hits": int(group["hit"].sum()),
                    "hit_or_near": int(group["hit_or_near"].sum()),
                    "floor_success": int(group["floor_success"].sum()),
                    "clean_large": int(group["clean_large"].sum()),
                    "opposite": int(group["opposite"].sum()),
                    "contract_hit_rate": float(group["hit"].mean()),
                    "hit_near_rate": float(group["hit_or_near"].mean()),
                    "floor_success_rate": float(group["floor_success"].mean()),
                    "clean_large_rate": float(group["clean_large"].mean()),
                    "opposite_rate": float(group["opposite"].mean()),
                    "avg_favorable_move": float(group["favorable_move"].mean()),
                    "median_favorable_move": float(group["favorable_move"].median()),
                }
            )
        return pd.DataFrame(rows)

    for name, cols in {
        "year_side": ["year", "side"],
        "quarter_side": ["quarter", "side"],
        "month_side": ["month", "side"],
        "symbol_side": ["symbol", "side"],
        "band_side": ["band", "side"],
        "setup_side": ["setup_id", "side"],
        "setup_quarter": ["quarter", "setup_id"],
    }.items():
        summarize(cols).to_csv(output_dir / f"{name}.csv", index=False)

    commonality = []
    valid_features = [
        c for c in feature_columns if c in df.columns and pd.api.types.is_numeric_dtype(df[c])
    ]
    for side_name in ["both", "long", "short"]:
        side_df = df if side_name == "both" else df[df["side"].eq(side_name)]
        if side_df.empty:
            continue
        large = side_df[side_df["clean_large"]]
        rest = side_df[~side_df["clean_large"]]
        if large.empty or rest.empty:
            continue
        for col_name in valid_features:
            large_values = pd.to_numeric(large[col_name], errors="coerce")
            rest_values = pd.to_numeric(rest[col_name], errors="coerce")
            if large_values.notna().sum() < 50 or rest_values.notna().sum() < 50:
                continue
            pooled = float(side_df[col_name].std(skipna=True) or 0.0)
            if not np.isfinite(pooled) or pooled == 0:
                continue
            large_mean = float(large_values.mean())
            rest_mean = float(rest_values.mean())
            commonality.append(
                {
                    "side": side_name,
                    "feature": col_name,
                    "large_mean": large_mean,
                    "rest_mean": rest_mean,
                    "std_mean_diff": (large_mean - rest_mean) / pooled,
                    "large_median": float(large_values.median()),
                    "rest_median": float(rest_values.median()),
                    "large_coverage": float(large_values.notna().mean()),
                    "rest_coverage": float(rest_values.notna().mean()),
                    "abs_std_mean_diff": abs((large_mean - rest_mean) / pooled),
                }
            )
    commonality_df = pd.DataFrame(commonality)
    if not commonality_df.empty:
        commonality_df.sort_values(
            ["side", "abs_std_mean_diff"],
            ascending=[True, False],
        ).to_csv(output_dir / "feature_commonality_clean_large.csv", index=False)


def _clean_x(frame: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    return frame[feature_columns].replace([np.inf, -np.inf], np.nan)


def _ranker(config: LargeMoveRankerConfig, random_state: int) -> Any:
    if not HAS_LIGHTGBM:
        raise RuntimeError("large-move ranker requires lightgbm")
    return LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        n_estimators=config.n_estimators,
        learning_rate=config.learning_rate,
        num_leaves=config.num_leaves,
        min_child_samples=config.min_child_samples,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=random_state,
        verbosity=-1,
    )


def _classifier(config: LargeMoveRankerConfig, random_state: int) -> Any:
    if not HAS_LIGHTGBM:
        raise RuntimeError("large-move classifier requires lightgbm")
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                LGBMClassifier(
                    n_estimators=config.n_estimators,
                    learning_rate=config.learning_rate,
                    num_leaves=config.num_leaves,
                    min_child_samples=config.min_child_samples,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    class_weight="balanced",
                    random_state=random_state,
                    verbosity=-1,
                ),
            ),
        ]
    )


def _regressor(config: LargeMoveRankerConfig, random_state: int) -> Any:
    if not HAS_LIGHTGBM:
        raise RuntimeError("large-move regressor requires lightgbm")
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                LGBMRegressor(
                    n_estimators=config.n_estimators,
                    learning_rate=config.learning_rate,
                    num_leaves=config.num_leaves,
                    min_child_samples=config.min_child_samples,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    random_state=random_state,
                    verbosity=-1,
                ),
            ),
        ]
    )


def _predict_positive(model: Any, x: pd.DataFrame) -> np.ndarray:
    return np.asarray(model.predict_proba(x)[:, 1], dtype=float)


def _fit_classifier(
    train: pd.DataFrame,
    feature_columns: list[str],
    target: str,
    config: LargeMoveRankerConfig,
    random_state: int,
) -> Any:
    y = train[target].astype(int)
    if y.nunique() < 2:
        constant = float(y.mean())

        class ConstantClassifier:
            def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
                p = np.full(len(x), constant)
                return np.column_stack([1.0 - p, p])

        return ConstantClassifier()
    model = _classifier(config, random_state)
    model.fit(_clean_x(train, feature_columns), y)
    return model


def _fit_side_bundle(
    dataset: pd.DataFrame,
    side: str,
    split: WalkForwardSplit,
    feature_columns: list[str],
    config: LargeMoveRankerConfig,
    random_state: int,
) -> SideRankerBundle:
    side_df = dataset[
        dataset["side"].eq(side)
        & dataset["status"].eq("evaluated")
        & between_dates(dataset, start=config.train_start, end=split.base_train_end)
    ].copy()
    if side_df.empty:
        raise ValueError(f"No training rows for {side} {split.name}")
    side_df = side_df.sort_values(["date", "symbol"]).reset_index(drop=True)
    group_sizes = side_df.groupby("date", sort=False).size().to_numpy()
    x_train = _clean_x(side_df, feature_columns)

    ranker = _ranker(config, random_state)
    ranker.fit(x_train, side_df["large_quality_rank_label"].astype(int), group=group_sizes)
    floor_model = _fit_classifier(
        side_df,
        feature_columns,
        "floor_success",
        config,
        random_state + 7,
    )
    hit_near_model = _fit_classifier(
        side_df,
        feature_columns,
        "hit_or_near",
        config,
        random_state + 13,
    )
    clean_large_model = _fit_classifier(
        side_df,
        feature_columns,
        "clean_large",
        config,
        random_state + 19,
    )
    opposite_model = _fit_classifier(
        side_df,
        feature_columns,
        "opposite",
        config,
        random_state + 29,
    )
    favorable_model = _regressor(config, random_state + 37)
    favorable_model.fit(x_train, side_df["favorable_move"].astype(float))
    return SideRankerBundle(
        side=side,
        ranker=ranker,
        floor_model=floor_model,
        hit_near_model=hit_near_model,
        clean_large_model=clean_large_model,
        opposite_model=opposite_model,
        favorable_model=favorable_model,
        feature_columns=feature_columns,
        train_rows=int(len(side_df)),
    )


def train_split_ranker(
    dataset: pd.DataFrame,
    feature_columns: list[str],
    split: WalkForwardSplit,
    config: LargeMoveRankerConfig,
) -> SplitRankerModel:
    model_id = f"{config.run_id}_{split.name}_large_move_ranker"
    long_bundle = _fit_side_bundle(
        dataset,
        "long",
        split,
        feature_columns,
        config,
        config.random_state,
    )
    short_bundle = _fit_side_bundle(
        dataset,
        "short",
        split,
        feature_columns,
        config,
        config.random_state + 101,
    )
    return SplitRankerModel(
        model_id=model_id,
        split=split,
        config=config,
        long_bundle=long_bundle,
        short_bundle=short_bundle,
    )


def predict_split_ranker(model: SplitRankerModel, dataset: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    passthrough = [
        "date",
        "symbol",
        "side",
        "band",
        "threshold",
        "target_threshold",
        "floor_threshold",
        "entry_date",
        "entry_open",
        "window_end_date",
        "status",
        "verdict",
        "hit",
        "near",
        "hit_or_near",
        "floor_success",
        "clean_large",
        "opposite",
        "small",
        "favorable_move",
        "signed_close_return",
        "large_rank_label",
        "large_move_grade",
        "setup_id",
        "hist_hit_near_rate_63",
        "hist_hit_near_rate_252",
        "hist_opposite_rate_63",
        "hist_opposite_rate_252",
        "symbol_return_20d_rank_pct",
        "relative_return_20d",
        "xsec_positive_share_20d",
        "setup_momentum_score",
        "setup_volatility_score",
        "setup_iv_oi_score",
        "setup_compression_score",
        "setup_reversal_score",
        "setup_quality_history_score",
        "setup_breadth_score",
    ]
    for side, bundle in [("long", model.long_bundle), ("short", model.short_bundle)]:
        part = dataset[dataset["side"].eq(side)].copy()
        if part.empty:
            continue
        x = _clean_x(part, bundle.feature_columns)
        out = part[[c for c in passthrough if c in part.columns]].copy()
        out["rank_score_raw"] = np.asarray(bundle.ranker.predict(x), dtype=float)
        out["p_floor_success"] = np.clip(_predict_positive(bundle.floor_model, x), 0, 1)
        out["p_hit_near"] = np.clip(_predict_positive(bundle.hit_near_model, x), 0, 1)
        out["p_clean_large"] = np.clip(_predict_positive(bundle.clean_large_model, x), 0, 1)
        out["p_opposite"] = np.clip(_predict_positive(bundle.opposite_model, x), 0, 1)
        out["expected_favorable_move"] = np.maximum(
            np.asarray(bundle.favorable_model.predict(x), dtype=float),
            0.0,
        )
        out["model_id"] = model.model_id
        frames.append(out)
    predictions = pd.concat(frames, ignore_index=True)
    return add_ranker_scores(predictions, model.config)


def add_ranker_scores(
    predictions: pd.DataFrame,
    config: LargeMoveRankerConfig,
) -> pd.DataFrame:
    df = predictions.copy()
    df["date"] = pd.to_datetime(df["date"])
    rank_group = df.groupby(["date", "side"])["rank_score_raw"]
    df["rank_score_pct"] = rank_group.rank(pct=True).fillna(0.5)
    expected_over_target = (
        df["expected_favorable_move"].astype(float)
        / df["target_threshold"].replace(0, np.nan).astype(float)
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    historical_risk = pd.to_numeric(
        df.get("hist_opposite_rate_252", pd.Series(0.0, index=df.index)),
        errors="coerce",
    ).fillna(0.0)
    df["ranker_utility_score"] = (
        config.rank_score_weight * df["rank_score_pct"]
        + config.clean_large_weight * df["p_clean_large"]
        + config.hit_near_weight * df["p_hit_near"]
        + config.floor_weight * df["p_floor_success"]
        - config.opposite_penalty * df["p_opposite"]
        + config.expected_move_weight * expected_over_target.clip(0.0, 3.0)
        - config.historical_risk_penalty * historical_risk
    )
    df["p_large_margin"] = df["p_clean_large"] - df["p_opposite"]
    return df.sort_values(["date", "ranker_utility_score"], ascending=[True, False])


def select_diverse_weekly_signals(
    predictions: pd.DataFrame,
    config: LargeMoveRankerConfig,
) -> pd.DataFrame:
    df = predictions.copy().sort_values(["date", "ranker_utility_score"], ascending=[True, False])
    df["date"] = pd.to_datetime(df["date"])
    df["selected"] = False
    df["selector_config_id"] = f"{config.run_id}_weekly_large_move_selector"
    df["selection_reason"] = "candidate_not_selected"
    df["year"] = df["date"].dt.year.astype(str)
    df["month"] = df["date"].dt.to_period("M").astype(str)
    df["week"] = df["date"].dt.strftime("%G-W%V")

    # Keep only the stronger side for the same symbol and day.
    keep = (
        df.sort_values(["date", "symbol", "ranker_utility_score"], ascending=[True, True, False])
        .drop_duplicates(["date", "symbol"], keep="first")
        .index
    )
    conflict_losers = ~df.index.isin(keep)
    df.loc[conflict_losers, "selection_reason"] = "symbol_side_conflict"

    df["date_pool_rank"] = (
        df[~conflict_losers]
        .groupby("date")["ranker_utility_score"]
        .rank(method="first", ascending=False)
    )
    below_pool = ~conflict_losers & df["date_pool_rank"].gt(config.daily_pool_rank)
    df.loc[below_pool, "selection_reason"] = "below_daily_rank_pool"

    eligible = ~conflict_losers & ~below_pool
    month_symbol_counts: dict[tuple[str, str], int] = defaultdict(int)
    year_symbol_counts: dict[tuple[str, str], int] = defaultdict(int)
    global_symbol_counts: dict[str, int] = defaultdict(int)

    for week, week_rows in df[eligible].groupby("week", sort=True):
        selected_this_week = 0
        week_symbols: dict[str, int] = defaultdict(int)
        week_side_counts: dict[str, int] = defaultdict(int)
        date_counts: dict[pd.Timestamp, int] = defaultdict(int)
        rejected: set[int] = set()
        candidates = week_rows.sort_values("ranker_utility_score", ascending=False)

        while selected_this_week < config.weekly_target:
            best_idx: int | None = None
            best_score = -np.inf
            best_row: pd.Series | None = None
            for idx, row in candidates.iterrows():
                if idx in rejected or bool(df.at[idx, "selected"]):
                    continue
                date = pd.Timestamp(row["date"])
                symbol = str(row["symbol"])
                side = str(row["side"])
                year_key = (str(row["year"]), symbol)
                month_key = (str(row["month"]), symbol)

                if date_counts[date] >= config.max_signals_per_day:
                    df.at[idx, "selection_reason"] = "daily_cap"
                    rejected.add(int(idx))
                    continue
                if week_side_counts[side] >= config.max_signals_per_week_side:
                    df.at[idx, "selection_reason"] = "weekly_side_cap"
                    rejected.add(int(idx))
                    continue
                if week_symbols[symbol] >= config.max_symbol_per_week:
                    df.at[idx, "selection_reason"] = "weekly_symbol_cap"
                    rejected.add(int(idx))
                    continue
                if month_symbol_counts[month_key] >= config.max_symbol_per_month:
                    df.at[idx, "selection_reason"] = "monthly_symbol_cap"
                    rejected.add(int(idx))
                    continue

                adjusted_score = (
                    float(row["ranker_utility_score"])
                    - config.year_symbol_penalty * year_symbol_counts[year_key]
                    - config.global_symbol_penalty * global_symbol_counts[symbol]
                )
                if adjusted_score > best_score:
                    best_idx = int(idx)
                    best_score = adjusted_score
                    best_row = row

            if best_idx is None or best_row is None:
                break

            date = pd.Timestamp(best_row["date"])
            symbol = str(best_row["symbol"])
            side = str(best_row["side"])
            year_key = (str(best_row["year"]), symbol)
            month_key = (str(best_row["month"]), symbol)
            df.at[best_idx, "selected"] = True
            df.at[best_idx, "selection_reason"] = "selected"
            df.at[best_idx, "fair_adjusted_score"] = best_score
            selected_this_week += 1
            week_symbols[symbol] += 1
            week_side_counts[side] += 1
            date_counts[date] += 1
            month_symbol_counts[month_key] += 1
            year_symbol_counts[year_key] += 1
            global_symbol_counts[symbol] += 1

        if selected_this_week >= config.weekly_target:
            remaining = candidates.index[
                ~candidates.index.isin(df.index[df["selected"]])
                & df.loc[candidates.index, "selection_reason"].eq("candidate_not_selected")
            ]
            df.loc[remaining, "selection_reason"] = "weekly_target_met"

    return df.sort_values(["date", "selected", "ranker_utility_score"], ascending=[True, False, False])


def select_setup_portfolio_signals(
    predictions: pd.DataFrame,
    config: LargeMoveRankerConfig,
) -> pd.DataFrame:
    setup_order = [s.strip() for s in config.setup_portfolio.split(",") if s.strip()]
    if not setup_order:
        raise ValueError("setup_round_robin selector requires setup_portfolio")

    df = predictions.copy().sort_values(["date", "ranker_utility_score"], ascending=[True, False])
    df["date"] = pd.to_datetime(df["date"])
    df["selected"] = False
    df["selector_config_id"] = f"{config.run_id}_setup_portfolio_selector"
    df["selection_reason"] = "candidate_not_selected"
    df["year"] = df["date"].dt.year.astype(str)
    df["month"] = df["date"].dt.to_period("M").astype(str)
    df["week"] = df["date"].dt.strftime("%G-W%V")

    keep = (
        df.sort_values(["date", "symbol", "ranker_utility_score"], ascending=[True, True, False])
        .drop_duplicates(["date", "symbol"], keep="first")
        .index
    )
    conflict_losers = ~df.index.isin(keep)
    df.loc[conflict_losers, "selection_reason"] = "symbol_side_conflict"
    outside_setup = ~df["setup_id"].isin(setup_order)
    df.loc[outside_setup, "selection_reason"] = "outside_setup_portfolio"
    eligible = ~conflict_losers & ~outside_setup

    month_symbol_counts: dict[tuple[str, str], int] = defaultdict(int)
    year_symbol_counts: dict[tuple[str, str], int] = defaultdict(int)
    global_symbol_counts: dict[str, int] = defaultdict(int)

    for week, week_rows in df[eligible].groupby("week", sort=True):
        week_symbols: dict[str, int] = defaultdict(int)
        week_side_counts: dict[str, int] = defaultdict(int)
        date_counts: dict[pd.Timestamp, int] = defaultdict(int)
        selected_this_week = 0

        for setup_id in setup_order:
            setup_rows = week_rows[week_rows["setup_id"].eq(setup_id)].copy()
            setup_selected = 0
            rejected: set[int] = set()
            while setup_selected < config.setup_weekly_quota:
                best_idx: int | None = None
                best_score = -np.inf
                best_row: pd.Series | None = None
                for idx, row in setup_rows.iterrows():
                    if idx in rejected or bool(df.at[idx, "selected"]):
                        continue
                    date = pd.Timestamp(row["date"])
                    symbol = str(row["symbol"])
                    side = str(row["side"])
                    year_key = (str(row["year"]), symbol)
                    month_key = (str(row["month"]), symbol)
                    if selected_this_week >= config.weekly_target:
                        df.at[idx, "selection_reason"] = "weekly_target_met"
                        rejected.add(int(idx))
                        continue
                    if date_counts[date] >= config.max_signals_per_day:
                        df.at[idx, "selection_reason"] = "daily_cap"
                        rejected.add(int(idx))
                        continue
                    if week_side_counts[side] >= config.max_signals_per_week_side:
                        df.at[idx, "selection_reason"] = "weekly_side_cap"
                        rejected.add(int(idx))
                        continue
                    if week_symbols[symbol] >= config.max_symbol_per_week:
                        df.at[idx, "selection_reason"] = "weekly_symbol_cap"
                        rejected.add(int(idx))
                        continue
                    if month_symbol_counts[month_key] >= config.max_symbol_per_month:
                        df.at[idx, "selection_reason"] = "monthly_symbol_cap"
                        rejected.add(int(idx))
                        continue
                    adjusted_score = (
                        float(row["ranker_utility_score"])
                        - config.year_symbol_penalty * year_symbol_counts[year_key]
                        - config.global_symbol_penalty * global_symbol_counts[symbol]
                    )
                    if adjusted_score > best_score:
                        best_idx = int(idx)
                        best_score = adjusted_score
                        best_row = row

                if best_idx is None or best_row is None:
                    break

                date = pd.Timestamp(best_row["date"])
                symbol = str(best_row["symbol"])
                side = str(best_row["side"])
                year_key = (str(best_row["year"]), symbol)
                month_key = (str(best_row["month"]), symbol)
                df.at[best_idx, "selected"] = True
                df.at[best_idx, "selection_reason"] = "selected"
                df.at[best_idx, "fair_adjusted_score"] = best_score
                setup_selected += 1
                selected_this_week += 1
                week_symbols[symbol] += 1
                week_side_counts[side] += 1
                date_counts[date] += 1
                month_symbol_counts[month_key] += 1
                year_symbol_counts[year_key] += 1
                global_symbol_counts[symbol] += 1

    return df.sort_values(["date", "selected", "ranker_utility_score"], ascending=[True, False, False])


def build_capture_report(selected: pd.DataFrame, period: pd.DataFrame) -> dict[str, pd.DataFrame]:
    def add_time_slices(frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        out["date"] = pd.to_datetime(out["date"])
        out["year"] = out["date"].dt.year.astype(str)
        out["quarter"] = out["date"].dt.to_period("Q").astype(str)
        out["month"] = out["date"].dt.to_period("M").astype(str)
        return out

    chosen = add_time_slices(selected[selected["selected"]])
    evaluated = chosen[chosen["status"].eq("evaluated")].copy()
    period_eval = add_time_slices(period[period["status"].eq("evaluated")])

    def capture_by(cols: list[str]) -> pd.DataFrame:
        rows = []
        for key, group in period_eval.groupby(cols, dropna=False):
            if not isinstance(key, tuple):
                key = (key,)
            mask = pd.Series(True, index=evaluated.index)
            for col_name, value in zip(cols, key, strict=True):
                mask &= evaluated[col_name].eq(value)
            sub = evaluated[mask]
            contract_available = int(group["hit"].sum())
            floor_available = int(group["floor_success"].sum())
            rows.append(
                {
                    **dict(zip(cols, key, strict=True)),
                    "available_rows": int(len(group)),
                    "contract_hits_available": contract_available,
                    "floor_success_available": floor_available,
                    "selected": int(len(sub)),
                    "selected_contract_hits": int(sub["hit"].sum()) if not sub.empty else 0,
                    "selected_hit_or_near": int(sub["hit_or_near"].sum()) if not sub.empty else 0,
                    "selected_floor_success": int(sub["floor_success"].sum()) if not sub.empty else 0,
                    "selected_opposite": int(sub["opposite"].sum()) if not sub.empty else 0,
                    "contract_capture_rate": float(sub["hit"].sum() / contract_available)
                    if contract_available
                    else 0.0,
                    "floor_capture_rate": float(sub["floor_success"].sum() / floor_available)
                    if floor_available
                    else 0.0,
                    "hit_near_precision": float(sub["hit_or_near"].mean()) if not sub.empty else 0.0,
                    "floor_precision": float(sub["floor_success"].mean()) if not sub.empty else 0.0,
                    "opposite_rate": float(sub["opposite"].mean()) if not sub.empty else 0.0,
                    "unique_symbols": int(sub["symbol"].nunique()) if not sub.empty else 0,
                }
            )
        return pd.DataFrame(rows)

    all_row = pd.DataFrame(
        [
            {
                "slice": "ALL",
                "available_rows": int(len(period_eval)),
                "contract_hits_available": int(period_eval["hit"].sum()),
                "floor_success_available": int(period_eval["floor_success"].sum()),
                "selected": int(len(evaluated)),
                "selected_contract_hits": int(evaluated["hit"].sum()) if not evaluated.empty else 0,
                "selected_hit_or_near": int(evaluated["hit_or_near"].sum())
                if not evaluated.empty
                else 0,
                "selected_floor_success": int(evaluated["floor_success"].sum())
                if not evaluated.empty
                else 0,
                "selected_opposite": int(evaluated["opposite"].sum()) if not evaluated.empty else 0,
                "contract_capture_rate": float(evaluated["hit"].sum() / period_eval["hit"].sum())
                if int(period_eval["hit"].sum())
                else 0.0,
                "floor_capture_rate": float(
                    evaluated["floor_success"].sum() / period_eval["floor_success"].sum()
                )
                if int(period_eval["floor_success"].sum())
                else 0.0,
                "hit_near_precision": float(evaluated["hit_or_near"].mean())
                if not evaluated.empty
                else 0.0,
                "floor_precision": float(evaluated["floor_success"].mean())
                if not evaluated.empty
                else 0.0,
                "opposite_rate": float(evaluated["opposite"].mean()) if not evaluated.empty else 0.0,
                "unique_symbols": int(evaluated["symbol"].nunique()) if not evaluated.empty else 0,
            }
        ]
    )
    return {
        "aggregate_capture": all_row,
        "year_capture": capture_by(["year"]),
        "quarter_capture": capture_by(["quarter"]),
        "side_capture": capture_by(["side"]),
        "band_capture": capture_by(["band"]),
        "setup_capture": capture_by(["setup_id"]),
        "symbol_capture": capture_by(["symbol"]),
    }


def save_split_model(model: SplitRankerModel, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_dir / "model.joblib")
    write_manifest(
        {
            "model_id": model.model_id,
            "split": asdict(model.split),
            "config": asdict(model.config),
            "algorithm": "lightgbm_lambdarank_plus_quality_models",
            "feature_count": len(model.long_bundle.feature_columns),
            "feature_columns": model.long_bundle.feature_columns,
            "long_train_rows": model.long_bundle.train_rows,
            "short_train_rows": model.short_bundle.train_rows,
        },
        output_dir / "manifest.json",
    )


def _write_tables(tables: dict[str, pd.DataFrame], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(output_dir / f"{name}.csv", index=False)


def run_large_move_ranker(config: LargeMoveRankerConfig) -> Path:
    if not HAS_LIGHTGBM:
        raise RuntimeError("LightGBM is required for run-large-move-ranker")

    run_dir = RUNS_DIR / config.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    dataset, base_features = load_source_dataset(config)
    dataset = add_setup_features(dataset)
    dataset = add_atlas_prior_features(dataset)
    feature_columns = _feature_columns(base_features, dataset)

    write_large_move_atlas(dataset, feature_columns, run_dir / "large_move_atlas")
    selected_frames: list[pd.DataFrame] = []

    for split in DEFAULT_SPLITS:
        model = train_split_ranker(dataset, feature_columns, split, config)
        save_split_model(model, run_dir / "models" / split.name)

        period = dataset[between_dates(dataset, split.prediction_start, split.prediction_end)].copy()
        predictions = predict_split_ranker(model, period)
        if config.selector_mode == "setup_round_robin":
            selected = select_setup_portfolio_signals(predictions, config)
        else:
            selected = select_diverse_weekly_signals(predictions, config)
        selected["split"] = split.name

        split_dir = run_dir / "model_predictions" / split.name
        split_dir.mkdir(parents=True, exist_ok=True)
        predictions.to_parquet(split_dir / "ranker_predictions.parquet", index=False)
        selected.to_parquet(split_dir / "signals.parquet", index=False)
        selected[selected["selected"]].to_csv(split_dir / "selected_signals.csv", index=False)
        write_report_tables(build_gold_report(selected), split_dir / "gold_report")
        _write_tables(build_capture_report(selected, period), split_dir / "capture_report")
        selected_frames.append(selected)

    combined = pd.concat(selected_frames, ignore_index=True)
    combined.to_parquet(run_dir / "all_signals.parquet", index=False)
    combined[combined["selected"]].to_csv(run_dir / "selected_signals.csv", index=False)
    write_report_tables(build_gold_report(combined), run_dir / "combined_gold_report")

    combined_period = dataset[
        between_dates(dataset, DEFAULT_SPLITS[0].prediction_start, DEFAULT_SPLITS[-1].prediction_end)
    ].copy()
    _write_tables(build_capture_report(combined, combined_period), run_dir / "combined_capture_report")
    write_manifest(
        {
            "run_id": config.run_id,
            "source_run_id": config.source_run_id,
            "source_run_dir": str(_source_run_dir(config)),
            "config": asdict(config),
            "feature_count": len(feature_columns),
            "feature_columns": feature_columns,
            "splits": [asdict(split) for split in DEFAULT_SPLITS],
            "notes": [
                "Large-move atlas plus setup/ranking model.",
                "No Koscine 2.0 models, predictions, overlays, or GO labels are used.",
                "Locked high-quality v19 artifacts are read only as the dataset contract.",
            ],
        },
        run_dir / "manifest.json",
    )
    return run_dir
