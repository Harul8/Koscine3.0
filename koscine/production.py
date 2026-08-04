from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import pandas as pd

from koscine.config import HORIZON_DAYS, MODEL_DIR, TARGET_UNIVERSE
from koscine.experiments import select_features
from koscine.rolling import fit_model, known_training_rows


PROD_ROOT = MODEL_DIR / "prod"


@dataclass(frozen=True)
class ProdModelConfig:
    train_start_year: int = 2012
    train_cutoff_day: int = 20
    feature_profile: str = "side_compact"
    sides: tuple[str, ...] = ("up",)
    thresholds: tuple[float, ...] = (0.05, 0.07)
    min_score: float = 0.65
    max_adverse_move: float | None = None
    cost_bps: float = 20.0


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def cutoff_for_prediction_month(prediction_month: str, cutoff_day: int) -> pd.Timestamp:
    month_start = pd.Timestamp(prediction_month + "-01")
    previous_month_start = month_start - pd.offsets.MonthBegin(1)
    return pd.Timestamp(
        year=previous_month_start.year,
        month=previous_month_start.month,
        day=cutoff_day,
    )


def train_prod_models(
    dataset_path: Path,
    prediction_month: str,
    config: ProdModelConfig,
    run_name: str | None = None,
) -> Path:
    run_id = run_name or f"prod_{prediction_month.replace('-', '')}_{timestamp()}"
    run_dir = PROD_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(dataset_path)
    df["date"] = pd.to_datetime(df["date"])
    df[f"future_{HORIZON_DAYS}d_date"] = pd.to_datetime(df[f"future_{HORIZON_DAYS}d_date"])
    train_end = cutoff_for_prediction_month(prediction_month, config.train_cutoff_day)

    models = []
    for side in config.sides:
        features = select_features(df, side, config.feature_profile)
        for threshold in config.thresholds:
            pct = int(round(threshold * 100))
            label_col = f"label_{side}_{pct}pct_{HORIZON_DAYS}d"
            train = known_training_rows(df, config.train_start_year, train_end, label_col)
            model = fit_model(train, features, label_col, validation_days=365)
            model_id = f"{side}_{pct}pct_{HORIZON_DAYS}d"
            model_file = f"{model_id}.txt"
            model.save_model(run_dir / model_file)
            metadata = {
                "model_id": model_id,
                "side": side,
                "threshold": threshold,
                "label_col": label_col,
                "feature_profile": config.feature_profile,
                "features": features,
                "best_iteration": model.best_iteration,
                "train_rows": len(train),
                "train_end": str(train_end.date()),
                "prediction_month": prediction_month,
                "model_file": model_file,
            }
            (run_dir / f"{model_id}.json").write_text(
                json.dumps(metadata, indent=2, default=str),
                encoding="utf-8",
            )
            models.append(metadata)

    manifest = {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_path": str(dataset_path),
        "prediction_month": prediction_month,
        "train_end": str(train_end.date()),
        "config": config.__dict__,
        "models": models,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))

    current_dir = PROD_ROOT / "current"
    if current_dir.exists():
        shutil.rmtree(current_dir)
    shutil.copytree(run_dir, current_dir)
    return run_dir


def predict_prod(
    dataset_path: Path,
    as_of_date: str,
    prod_dir: Path = PROD_ROOT / "current",
    output_dir: Path = Path("predictions") / "prod",
) -> pd.DataFrame:
    df = pd.read_parquet(dataset_path)
    df["date"] = pd.to_datetime(df["date"])
    as_of = pd.Timestamp(as_of_date)
    rows = df[df["date"].eq(as_of) & df["symbol"].isin(TARGET_UNIVERSE)].copy()
    if rows.empty:
        raise ValueError(f"No target-universe rows found for {as_of_date}")

    frames = []
    for metadata_path in sorted(prod_dir.glob("*.json")):
        if metadata_path.name == "manifest.json":
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        model = lgb.Booster(model_file=str(prod_dir / metadata["model_file"]))
        features = metadata["features"]
        score_col = f"score_{metadata['model_id']}"
        scored = rows[["date", "symbol", "close"] + features].copy()
        scored[score_col] = model.predict(scored[features], num_iteration=model.best_iteration)
        scored["side"] = metadata["side"]
        scored["threshold"] = metadata["threshold"]
        scored["score"] = scored[score_col]
        scored["model_id"] = metadata["model_id"]
        scored["prediction_for_entry"] = "next_trading_day_open"
        frames.append(
            scored[
                [
                    "date",
                    "symbol",
                    "close",
                    "side",
                    "threshold",
                    "score",
                    "model_id",
                    "prediction_for_entry",
                ]
            ]
        )
    predictions = pd.concat(frames, ignore_index=True)
    manifest = json.loads((prod_dir / "manifest.json").read_text(encoding="utf-8"))
    min_score = manifest["config"].get("min_score", 0.0)
    predictions["prod_min_score"] = min_score
    predictions["passes_min_score"] = predictions["score"].ge(min_score)
    predictions = predictions.sort_values(["score", "threshold"], ascending=[False, False])

    output_dir.mkdir(parents=True, exist_ok=True)
    safe_date = as_of.strftime("%Y%m%d")
    predictions.to_csv(output_dir / f"prod_predictions_{safe_date}.csv", index=False)
    predictions.to_parquet(output_dir / f"prod_predictions_{safe_date}.parquet", index=False)
    return predictions


def prod_auto(
    dataset_path: Path,
    as_of_date: str,
    config: ProdModelConfig | None = None,
) -> pd.DataFrame:
    config = config or ProdModelConfig()
    current_manifest = PROD_ROOT / "current" / "manifest.json"
    if not current_manifest.exists():
        raise FileNotFoundError("No current production model found. Run prod-train manually first.")
    return predict_prod(dataset_path=dataset_path, as_of_date=as_of_date)
