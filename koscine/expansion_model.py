"""Stage-1 expansion model: predicts P(clean directional move within 5d) per
tier. Direction is handled separately by the existing per-side `volclean_*`
models (or computed downstream).

This is a single binary classifier per tier, trained on
`label_expansion_clean_{pct}pct_{horizon}d` where the threshold is 4% for
liquid30, 7% for rest35.
"""
from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import average_precision_score, roc_auc_score

from koscine.calibration import CalibratorBundle, fit_isotonic, save_calibrator, load_calibrator
from koscine.config import (
    HORIZON_DAYS,
    MODEL_DIR,
    PREDICTIONS_DIR,
    REPORTS_DIR,
    RUNS_DIR,
    TARGET_UNIVERSE,
)
from koscine.clean_direction import (
    add_label_validity_flags,
    add_liquid30_features,
    add_sector_features,
    liquid30_symbols,
    normalize_symbol,
)
from koscine.tiered_clean_direction import (
    add_tiered_research_features,
    rest35_symbols,
    stable_seed,
)
from koscine.training import feature_columns


EXPANSION_PROD_ROOT = MODEL_DIR / "expansion" / "prod"


@dataclass(frozen=True)
class ExpansionTierSpec:
    name: str
    threshold: float
    predict_symbols: tuple[str, ...]
    model_id: str


def expansion_tier_specs() -> list[ExpansionTierSpec]:
    return [
        ExpansionTierSpec("liquid30", 0.04, tuple(liquid30_symbols()), "expansion_liquid30_4pct_5d"),
        ExpansionTierSpec("rest35", 0.07, tuple(rest35_symbols()), "expansion_rest35_7pct_5d"),
    ]


def expansion_label_col(spec: ExpansionTierSpec) -> str:
    pct = int(round(spec.threshold * 100))
    return f"label_expansion_clean_{pct}pct_{HORIZON_DAYS}d"


@dataclass(frozen=True)
class ExpansionTrainConfig:
    train_start_year: int = 2012
    train_cutoff_day: int = 20
    validation_days: int = 365
    lgbm_weight: float = 0.60
    use_catboost: bool = True
    n_seeds: int = 2
    temporal_decay_per_year: float = 0.05
    use_calibration: bool = True
    train_all_symbols: bool = True


@dataclass(frozen=True)
class ExpansionWalkforwardConfig:
    train_start_year: int = 2012
    start_test_year: int = 2022
    end_test_year: int = 2025
    validation_days: int = 365
    train_cutoff_month: int = 12
    train_cutoff_day: int = 20
    lgbm_weight: float = 0.60
    use_catboost: bool = True
    n_seeds: int = 2
    temporal_decay_per_year: float = 0.05
    use_calibration: bool = True
    train_all_symbols: bool = True


# --------- shared helpers ---------


def _cutoff_for_prediction_month(prediction_month: str, cutoff_day: int) -> pd.Timestamp:
    month_start = pd.Timestamp(prediction_month + "-01")
    prev_month = month_start - pd.offsets.MonthBegin(1)
    return pd.Timestamp(year=prev_month.year, month=prev_month.month, day=cutoff_day)


def _prepare_frame(dataset_path: Path) -> tuple[pd.DataFrame, list[str], list[ExpansionTierSpec]]:
    df = pd.read_parquet(dataset_path)
    df["date"] = pd.to_datetime(df["date"])
    df["symbol"] = df["symbol"].map(normalize_symbol)
    if f"future_{HORIZON_DAYS}d_date" in df:
        df[f"future_{HORIZON_DAYS}d_date"] = pd.to_datetime(df[f"future_{HORIZON_DAYS}d_date"])
    liquid = liquid30_symbols()
    df = add_liquid30_features(df, liquid)
    df = add_sector_features(df, liquid)
    df = add_tiered_research_features(df)
    df = add_label_validity_flags(df, 0.50)
    df["is_liquid30"] = df["symbol"].isin(liquid)
    df["is_rest35"] = df["symbol"].isin(rest35_symbols())
    df["tier_name"] = np.where(df["is_liquid30"], "liquid30", np.where(df["is_rest35"], "rest35", "other"))

    features = feature_columns(df)
    blocked = {
        "price_discontinuity",
        "corp_price_action",
        "invalid_price_event",
        "future_invalid_price_event_5d",
    }
    features = [c for c in features if c not in blocked]
    for col in features:
        df[col] = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).astype("float32")
    return df, features, expansion_tier_specs()


def _future_known(df: pd.DataFrame, train_end: pd.Timestamp) -> pd.Series:
    fc = f"future_{HORIZON_DAYS}d_date"
    if fc not in df:
        return pd.Series(True, index=df.index)
    return pd.to_datetime(df[fc]).le(train_end)


def _sample_weights(train: pd.DataFrame, spec: ExpansionTierSpec, decay_per_year: float) -> pd.Series:
    weights = pd.Series(1.0, index=train.index)
    in_tier = train["symbol"].isin(spec.predict_symbols)
    weights.loc[in_tier] *= 2.5 if spec.name == "liquid30" else 2.0
    weights.loc[~in_tier] *= 0.6
    pos = train[expansion_label_col(spec)].astype(bool)
    weights.loc[pos] *= 2.5  # upweight positives
    if decay_per_year and decay_per_year > 0:
        dates = pd.to_datetime(train["date"])
        years_ago = (dates.max() - dates).dt.days / 365.25
        decay = np.power(1.0 - float(decay_per_year), years_ago.clip(lower=0).values)
        weights = weights * pd.Series(decay, index=train.index).clip(lower=0.20)
    return weights


def _train_frame_for_spec(
    df: pd.DataFrame,
    spec: ExpansionTierSpec,
    train_start_year: int,
    train_end: pd.Timestamp,
    train_all_symbols: bool,
    decay_per_year: float,
    features: list[str] | None = None,
) -> pd.DataFrame:
    label = expansion_label_col(spec)
    columns = ["date", "symbol", label]
    if features is not None:
        columns += [col for col in features if col in df.columns]
    columns = list(dict.fromkeys(columns))
    frame = df.loc[
        df["date"].dt.year.ge(train_start_year)
        & df["date"].le(train_end)
        & _future_known(df, train_end),
        columns,
    ].dropna(subset=[label]).copy()
    if not train_all_symbols:
        frame = frame[frame["symbol"].isin(spec.predict_symbols)].copy()
    frame["sample_weight"] = _sample_weights(frame, spec, decay_per_year)
    return frame


def _lgbm_params(y: pd.Series, seed: int, spec: ExpansionTierSpec) -> dict:
    pos = int(y.sum())
    neg = int(len(y) - pos)
    return {
        "objective": "binary",
        "metric": "average_precision",
        "boosting_type": "gbdt",
        "learning_rate": 0.025,
        "num_leaves": 31 if spec.name == "liquid30" else 47,
        "min_data_in_leaf": 400 if spec.name == "liquid30" else 250,
        "feature_fraction": 0.78,
        "bagging_fraction": 0.82,
        "bagging_freq": 1,
        "lambda_l1": 2.0,
        "lambda_l2": 12.0,
        "min_gain_to_split": 0.01,
        "scale_pos_weight": neg / max(pos, 1),
        "verbosity": -1,
        "seed": seed,
    }


def _fit_lgbm(inner: pd.DataFrame, valid: pd.DataFrame, features: list[str], spec: ExpansionTierSpec, seed: int) -> lgb.Booster:
    label = expansion_label_col(spec)
    train_set = lgb.Dataset(inner[features], label=inner[label].astype(int), weight=inner["sample_weight"])
    valid_set = lgb.Dataset(valid[features], label=valid[label].astype(int), weight=valid["sample_weight"], reference=train_set)
    return lgb.train(
        _lgbm_params(inner[label].astype(int), seed, spec),
        train_set,
        valid_sets=[valid_set],
        num_boost_round=2500,
        callbacks=[lgb.early_stopping(150), lgb.log_evaluation(0)],
    )


def _fit_catboost(inner: pd.DataFrame, valid: pd.DataFrame, features: list[str], spec: ExpansionTierSpec, seed: int) -> CatBoostClassifier:
    label = expansion_label_col(spec)
    y = inner[label].astype(int)
    pos = int(y.sum())
    neg = int(len(y) - pos)
    model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="PRAUC",
        iterations=500,
        learning_rate=0.04,
        depth=6,
        l2_leaf_reg=12 if spec.name == "liquid30" else 10,
        random_strength=1.5,
        bagging_temperature=1.0,
        class_weights=[1.0, neg / max(pos, 1)],
        od_type="Iter",
        od_wait=80,
        allow_writing_files=False,
        random_seed=seed,
        verbose=False,
    )
    model.fit(
        Pool(inner[features], label=inner[label].astype(int), weight=inner["sample_weight"]),
        eval_set=Pool(valid[features], label=valid[label].astype(int), weight=valid["sample_weight"]),
        use_best_model=True,
    )
    return model


def _fit_models(train: pd.DataFrame, features: list[str], spec: ExpansionTierSpec, validation_days: int, use_catboost: bool, seed: int, n_seeds: int) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    valid_cutoff = train["date"].max() - pd.Timedelta(days=validation_days)
    inner = train[train["date"] < valid_cutoff].copy()
    valid = train[train["date"] >= valid_cutoff].copy()
    if inner.empty or valid.empty:
        raise ValueError(f"Empty inner/valid split for {spec.model_id}")
    n_seeds = max(1, int(n_seeds))
    if n_seeds == 1:
        models: dict = {"lgbm": _fit_lgbm(inner, valid, features, spec, seed)}
        if use_catboost:
            models["catboost"] = _fit_catboost(inner, valid, features, spec, seed + 1000)
        return models, inner, valid
    models = {"lgbm_ensemble": [_fit_lgbm(inner, valid, features, spec, seed + 7 * i) for i in range(n_seeds)]}
    if use_catboost:
        models["catboost_ensemble"] = [_fit_catboost(inner, valid, features, spec, seed + 1000 + 11 * i) for i in range(n_seeds)]
    return models, inner, valid


def _predict_lgbm(model: lgb.Booster, frame: pd.DataFrame, features: list[str]) -> np.ndarray:
    return model.predict(frame[features], num_iteration=model.best_iteration)


def _predict_catboost(model: CatBoostClassifier, frame: pd.DataFrame, features: list[str]) -> np.ndarray:
    return model.predict_proba(frame[features])[:, 1]


def _score_frame(frame: pd.DataFrame, features: list[str], spec: ExpansionTierSpec, models: dict, lgbm_weight: float, calibrator: CalibratorBundle | None = None) -> pd.DataFrame:
    out = frame.copy()
    if "lgbm_ensemble" in models:
        lgbm = np.mean([_predict_lgbm(m, out, features) for m in models["lgbm_ensemble"]], axis=0)
    else:
        lgbm = _predict_lgbm(models["lgbm"], out, features)
    out["expansion_lgbm"] = lgbm
    if "catboost_ensemble" in models:
        cb = np.mean([_predict_catboost(m, out, features) for m in models["catboost_ensemble"]], axis=0)
        out["expansion_catboost"] = cb
        raw = lgbm_weight * lgbm + (1.0 - lgbm_weight) * cb
    elif "catboost" in models:
        cb = _predict_catboost(models["catboost"], out, features)
        out["expansion_catboost"] = cb
        raw = lgbm_weight * lgbm + (1.0 - lgbm_weight) * cb
    else:
        out["expansion_catboost"] = np.nan
        raw = lgbm
    out["expansion_raw"] = raw
    if calibrator is not None:
        out["expansion_calibrated"] = calibrator.predict(raw)
    else:
        out["expansion_calibrated"] = np.nan
    out["tier"] = spec.name
    out["expansion_threshold"] = spec.threshold
    out["expansion_model_id"] = spec.model_id
    return out


def _save_models_bundle(run_dir: Path, prefix: str, models: dict, features: list[str], spec: ExpansionTierSpec, train_end: pd.Timestamp) -> dict:
    (run_dir / "models").mkdir(parents=True, exist_ok=True)
    saved = []
    for name, model in models.items():
        if name.endswith("_ensemble") and isinstance(model, list):
            files = []
            for i, m in enumerate(model):
                fname = f"{prefix}_{name}_seed{i}"
                if isinstance(m, lgb.Booster):
                    path = run_dir / "models" / f"{fname}.txt"
                    m.save_model(path)
                    bi = m.best_iteration
                else:
                    path = run_dir / "models" / f"{fname}.cbm"
                    m.save_model(str(path))
                    bi = m.get_best_iteration()
                files.append({"model_file": path.name, "best_iteration": bi})
            saved.append({"family": name, "files": files, "n_seeds": len(model)})
            continue
        fname = f"{prefix}_{name}"
        if isinstance(model, lgb.Booster):
            path = run_dir / "models" / f"{fname}.txt"
            model.save_model(path)
            bi = model.best_iteration
        else:
            path = run_dir / "models" / f"{fname}.cbm"
            model.save_model(str(path))
            bi = model.get_best_iteration()
        saved.append({"family": name, "model_file": path.name, "best_iteration": bi})
    meta = {
        "model_id": spec.model_id,
        "tier": spec.name,
        "threshold": spec.threshold,
        "label_col": expansion_label_col(spec),
        "train_end": str(train_end.date()),
        "features": features,
        "models": saved,
    }
    (run_dir / "models" / f"{prefix}.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    return meta


def _load_bundle(meta: dict, root: Path) -> dict:
    out: dict = {}
    for fam in meta["models"]:
        if fam["family"].endswith("_ensemble") and "files" in fam:
            loaded = []
            for f in fam["files"]:
                path = root / f["model_file"]
                if not path.exists():
                    path = root / "models" / f["model_file"]
                if fam["family"].startswith("lgbm"):
                    loaded.append(lgb.Booster(model_file=str(path)))
                else:
                    m = CatBoostClassifier()
                    m.load_model(str(path))
                    loaded.append(m)
            out[fam["family"]] = loaded
            continue
        path = root / fam["model_file"]
        if not path.exists():
            path = root / "models" / fam["model_file"]
        if fam["family"] == "lgbm":
            out["lgbm"] = lgb.Booster(model_file=str(path))
        elif fam["family"] == "catboost":
            m = CatBoostClassifier()
            m.load_model(str(path))
            out["catboost"] = m
    return out


def _metrics(frame: pd.DataFrame, spec: ExpansionTierSpec, score_col: str = "expansion_raw") -> dict:
    if frame.empty:
        return {"rows": 0}
    label = expansion_label_col(spec)
    y = frame[label].astype(int)
    p = frame[score_col].astype(float)
    out: dict = {"rows": int(len(frame)), "positives": int(y.sum()), "positive_rate": float(y.mean())}
    if y.nunique() > 1:
        out["average_precision"] = float(average_precision_score(y, p))
        try:
            out["roc_auc"] = float(roc_auc_score(y, p))
        except ValueError:
            out["roc_auc"] = np.nan
    else:
        out["average_precision"] = np.nan
        out["roc_auc"] = np.nan
    ranked = frame.sort_values(score_col, ascending=False)
    for k in (10, 25, 50, 100, 200):
        top = ranked.head(k)
        if len(top):
            out[f"precision_at_{k}"] = float(top[label].mean())
    return out


# --------- training entry points ---------


def train_expansion_prod(
    dataset_path: Path,
    prediction_month: str,
    config: ExpansionTrainConfig | None = None,
    run_name: str | None = None,
    update_current: bool = True,
    output_root: Path | None = None,
) -> Path:
    """Single training for production - cutoff = prev month 20th."""
    config = config or ExpansionTrainConfig()
    run_id = run_name or f"expansion_prod_{prediction_month.replace('-', '')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    root = output_root or EXPANSION_PROD_ROOT
    run_dir = root / run_id
    (run_dir / "models").mkdir(parents=True, exist_ok=True)

    df, features, specs = _prepare_frame(dataset_path)
    train_end = _cutoff_for_prediction_month(prediction_month, config.train_cutoff_day)
    models_meta = []
    for spec in specs:
        train = _train_frame_for_spec(df, spec, config.train_start_year, train_end, config.train_all_symbols, config.temporal_decay_per_year, features)
        models, inner, valid = _fit_models(train, features, spec, config.validation_days, config.use_catboost, stable_seed(spec.model_id, 9000), config.n_seeds)
        prefix = spec.model_id
        meta = _save_models_bundle(run_dir, prefix, models, features, spec, train_end)
        # Score validation rows for calibration
        valid_scored = _score_frame(valid, features, spec, models, config.lgbm_weight, calibrator=None)
        label = expansion_label_col(spec)
        if config.use_calibration:
            cal = fit_isotonic(valid_scored["expansion_raw"].values, valid_scored[label].astype(float).values)
            save_calibrator(cal, run_dir / "models" / f"{prefix}_calibrator.json")
            meta["calibrator"] = f"{prefix}_calibrator.json"
        meta["validation_metrics"] = _metrics(valid_scored, spec)
        meta["prediction_month"] = prediction_month
        meta["lgbm_weight"] = config.lgbm_weight
        meta["predict_symbols"] = list(spec.predict_symbols)
        models_meta.append(meta)

    manifest = {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_path": str(dataset_path),
        "prediction_month": prediction_month,
        "train_end": str(train_end.date()),
        "model_stack": "expansion_stage1_lgbm_catboost",
        "config": config.__dict__,
        "models": models_meta,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    if update_current:
        current = EXPANSION_PROD_ROOT / "current"
        if current.exists():
            shutil.rmtree(current)
        shutil.copytree(run_dir, current)
    return run_dir


def predict_expansion_prod(dataset_path: Path, as_of_dates: list[str], prod_dir: Path = EXPANSION_PROD_ROOT / "current", output_dir: Path = PREDICTIONS_DIR / "expansion_prod", progress=None) -> pd.DataFrame:
    manifest_path = prod_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"No expansion prod manifest at {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config = ExpansionTrainConfig(**{k: v for k, v in manifest.get("config", {}).items() if k in ExpansionTrainConfig.__dataclass_fields__})
    df, features, specs = _prepare_frame(dataset_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    requested = [pd.Timestamp(d).normalize() for d in as_of_dates]

    spec_ctx = []
    for meta in manifest["models"]:
        spec = next(s for s in specs if s.model_id == meta["model_id"])
        models = _load_bundle(meta, prod_dir)
        cal = None
        if meta.get("calibrator"):
            cal = load_calibrator(prod_dir / "models" / meta["calibrator"])
        spec_ctx.append((meta, spec, models, cal, float(meta.get("lgbm_weight", config.lgbm_weight))))

    summary_rows = []
    keep = [
        "date", "symbol", "tier", "expansion_threshold", "expansion_model_id",
        "expansion_raw", "expansion_calibrated", "expansion_lgbm", "expansion_catboost",
        "close", "entry_1d_date", "entry_1d_open",
        f"future_{HORIZON_DAYS}d_date", "atr_pct_14", "atm_iv",
    ]
    for idx, date in enumerate(requested, start=1):
        date_label = date.strftime("%Y-%m-%d")
        if progress:
            progress(f"expansion predict {date_label} ({idx}/{len(requested)})")
        frames = []
        for _meta, spec, models, cal, lgbm_w in spec_ctx:
            rows = df[df["date"].eq(date) & df["symbol"].isin(spec.predict_symbols)].copy()
            if rows.empty:
                continue
            scored = _score_frame(rows, features, spec, models, lgbm_w, calibrator=cal)
            cols = [c for c in keep if c in scored.columns]
            frames.append(scored[cols])
        if not frames:
            summary_rows.append({"date": date_label, "rows": 0})
            continue
        combined = pd.concat(frames, ignore_index=True).sort_values("expansion_raw", ascending=False)
        safe = date.strftime("%Y%m%d")
        combined.to_parquet(output_dir / f"expansion_predictions_{safe}.parquet", index=False)
        combined.to_csv(output_dir / f"expansion_predictions_{safe}.csv", index=False)
        summary_rows.append({"date": date_label, "rows": int(len(combined)), "max_raw": float(combined["expansion_raw"].max())})
    return pd.DataFrame(summary_rows)


def run_expansion_walkforward(dataset_path: Path, config: ExpansionWalkforwardConfig | None = None, run_name: str | None = None) -> Path:
    """Walk-forward training for evaluating expansion-only model."""
    config = config or ExpansionWalkforwardConfig()
    run_id = run_name or f"expansion_wf_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = RUNS_DIR / run_id
    (run_dir / "models").mkdir(parents=True, exist_ok=True)
    (run_dir / "predictions").mkdir(parents=True, exist_ok=True)
    (run_dir / "reports").mkdir(parents=True, exist_ok=True)

    df, features, specs = _prepare_frame(dataset_path)
    all_scored = []
    metrics_rows = []
    for year in range(config.start_test_year, config.end_test_year + 1):
        train_end = pd.Timestamp(year=year - 1, month=config.train_cutoff_month, day=config.train_cutoff_day)
        for spec in specs:
            train = _train_frame_for_spec(df, spec, config.train_start_year, train_end, config.train_all_symbols, config.temporal_decay_per_year, features)
            models, inner, valid = _fit_models(train, features, spec, config.validation_days, config.use_catboost, stable_seed(spec.model_id, 42 + year), config.n_seeds)
            label = expansion_label_col(spec)
            # Fit calibrator on validation
            valid_scored = _score_frame(valid, features, spec, models, config.lgbm_weight, calibrator=None)
            cal = None
            if config.use_calibration:
                cal = fit_isotonic(valid_scored["expansion_raw"].values, valid_scored[label].astype(float).values)
            # Score test year
            test_mask = df["date"].dt.year.eq(year) & df["symbol"].isin(spec.predict_symbols)
            test_rows = df[test_mask].dropna(subset=[label]).copy()
            test_scored = _score_frame(test_rows, features, spec, models, config.lgbm_weight, calibrator=cal)
            test_scored["prediction_year"] = year
            test_scored["split"] = "test"
            all_scored.append(test_scored)
            m = _metrics(test_scored, spec, score_col="expansion_raw")
            m.update({"year": year, "tier": spec.name, "threshold": spec.threshold})
            metrics_rows.append(m)
            prefix = f"{run_dir.name}_{year}_{spec.model_id}"
            _save_models_bundle(run_dir, prefix, models, features, spec, train_end)
            if cal is not None:
                save_calibrator(cal, run_dir / "models" / f"{prefix}_calibrator.json")

    scored = pd.concat(all_scored, ignore_index=True)
    metrics = pd.DataFrame(metrics_rows)
    scored.to_parquet(run_dir / "predictions" / f"{run_dir.name}_all_scored.parquet", index=False)
    metrics.to_csv(run_dir / "reports" / f"{run_dir.name}_metrics.csv", index=False)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(REPORTS_DIR / "expansion_walkforward_metrics_latest.csv", index=False)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    scored.to_parquet(PREDICTIONS_DIR / "expansion_walkforward_scored_latest.parquet", index=False)
    manifest = {
        "run_id": run_dir.name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_path": str(dataset_path),
        "config": config.__dict__,
        "feature_count": len(features),
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return run_dir
