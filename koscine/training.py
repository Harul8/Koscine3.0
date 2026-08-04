from __future__ import annotations

import gc
import json
import os
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from koscine.config import (
    HORIZON_DAYS,
    MODEL_DIR,
    PROCESSED_DIR,
    PURGE_DAYS,
    REPORTS_DIR,
    TARGET_UNIVERSE,
)


EXCLUDE_PREFIXES = (
    "label_",
    "future_",
    "up_move_",
    "down_move_",
    "fwd_return_",
    "entry_",
    "long_adverse_",
    "short_adverse_",
)
EXCLUDE_COLUMNS = {"date", "symbol"}


@dataclass(frozen=True)
class SplitSpec:
    train_start_year: int | None
    train_end_year: int
    test_year: int
    purge_days: int = PURGE_DAYS


def feature_columns(df: pd.DataFrame) -> list[str]:
    cols = []
    for col in df.columns:
        if col in EXCLUDE_COLUMNS:
            continue
        if any(col.startswith(prefix) for prefix in EXCLUDE_PREFIXES):
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            cols.append(col)
    return cols


def split_frame(df: pd.DataFrame, split: SplitSpec) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.to_datetime(df["date"])
    train_mask = dates.dt.year <= split.train_end_year
    if split.train_start_year is not None:
        train_mask &= dates.dt.year >= split.train_start_year

    test_start = pd.Timestamp(year=split.test_year, month=1, day=1)
    purge_start = test_start - pd.Timedelta(days=split.purge_days * 2)
    train_mask &= dates < purge_start
    test_mask = dates.dt.year == split.test_year
    return df[train_mask].copy(), df[test_mask].copy()


def _make_params(y: pd.Series, seed: int = 42) -> dict:
    positives = int(y.sum())
    negatives = int(len(y) - positives)
    scale_pos_weight = negatives / max(positives, 1)
    return {
        "objective": "binary",
        "metric": "average_precision",
        "boosting_type": "gbdt",
        "learning_rate": 0.03,
        "num_leaves": 31,
        "min_data_in_leaf": 200,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l1": 1.0,
        "lambda_l2": 5.0,
        "min_gain_to_split": 0.01,
        "scale_pos_weight": scale_pos_weight,
        "verbosity": -1,
        "seed": seed,
    }


def _fit_lgbm(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    features: list[str],
    label_col: str,
    seed: int = 42,
) -> lgb.Booster:
    train_set = lgb.Dataset(train[features], label=train[label_col])
    valid_set = lgb.Dataset(valid[features], label=valid[label_col], reference=train_set)
    model = lgb.train(
        _make_params(train[label_col], seed=seed),
        train_set,
        valid_sets=[valid_set],
        num_boost_round=2000,
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)],
    )
    return model


def _fit_lgbm_final(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    features: list[str],
    label_col: str,
    seed: int = 42,
) -> lgb.Booster:
    train = train.dropna(subset=[label_col])
    valid = valid.dropna(subset=[label_col])
    if train.empty or valid.empty:
        raise ValueError(f"Empty train/valid split for {label_col}")
    return _fit_lgbm(train, valid, features, label_col, seed=seed)


def _score_predictions(df: pd.DataFrame, pred_col: str, label_col: str) -> dict[str, float]:
    y = df[label_col].astype(int)
    pred = df[pred_col].astype(float)
    side = -1.0 if "label_down_" in label_col else 1.0
    metrics: dict[str, float] = {
        "rows": float(len(df)),
        "positives": float(y.sum()),
        "positive_rate": float(y.mean()) if len(y) else np.nan,
        "average_precision": float(average_precision_score(y, pred)) if y.nunique() > 1 else np.nan,
    }
    try:
        metrics["roc_auc"] = float(roc_auc_score(y, pred)) if y.nunique() > 1 else np.nan
    except ValueError:
        metrics["roc_auc"] = np.nan

    ranked = df.sort_values(pred_col, ascending=False)
    for k in (10, 25, 50, 100, 200):
        top = ranked.head(k)
        if len(top):
            metrics[f"precision_at_{k}"] = float(top[label_col].mean())
            metrics[f"avg_fwd_return_at_{k}"] = float(top[f"fwd_return_{HORIZON_DAYS}d"].mean())
            metrics[f"avg_strategy_return_at_{k}"] = float(
                side * top[f"fwd_return_{HORIZON_DAYS}d"].mean()
            )
            metrics[f"avg_up_move_at_{k}"] = float(top[f"up_move_{HORIZON_DAYS}d"].mean())
            metrics[f"avg_down_move_at_{k}"] = float(top[f"down_move_{HORIZON_DAYS}d"].mean())
    for pct in (0.01, 0.02, 0.05, 0.10):
        n = max(1, int(len(ranked) * pct))
        top = ranked.head(n)
        metrics[f"precision_top_{int(pct * 100)}pct"] = float(top[label_col].mean())
    return metrics


def train_binary_for_year(
    dataset_path: Path,
    label_col: str,
    test_year: int,
    train_start_year: int | None = None,
    train_end_year: int | None = None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    df = pd.read_parquet(dataset_path)
    if label_col not in df:
        raise KeyError(f"Missing label column {label_col}")

    split = SplitSpec(
        train_start_year=train_start_year,
        train_end_year=train_end_year if train_end_year is not None else test_year - 1,
        test_year=test_year,
    )
    train, test = split_frame(df, split)
    test = test[test["symbol"].isin(TARGET_UNIVERSE)].copy()
    valid_cutoff = train["date"].max() - pd.Timedelta(days=365)
    inner_train = train[train["date"] < valid_cutoff].copy()
    valid = train[train["date"] >= valid_cutoff].copy()
    features = feature_columns(df)

    inner_train = inner_train.dropna(subset=[label_col])
    valid = valid.dropna(subset=[label_col])
    test = test.dropna(subset=[label_col])

    if inner_train.empty or valid.empty or test.empty:
        raise ValueError(
            "Empty train/valid/test split after label filtering. "
            f"label={label_col}, test_year={test_year}"
        )

    model = _fit_lgbm(inner_train, valid, features, label_col)
    pred_col = f"pred_{label_col.removeprefix('label_')}"
    test[pred_col] = model.predict(test[features], num_iteration=model.best_iteration)
    metrics = _score_predictions(test, pred_col, label_col)
    metrics.update(
        {
            "label": label_col,
            "test_year": float(test_year),
            "train_rows": float(len(train)),
            "feature_count": float(len(features)),
            "best_iteration": float(model.best_iteration or 0),
        }
    )
    return test.sort_values(["date", pred_col], ascending=[True, False]), metrics


def save_model_bundle(
    model: lgb.Booster,
    features: list[str],
    metadata: dict,
    model_dir: Path = MODEL_DIR,
) -> Path:
    model_dir.mkdir(parents=True, exist_ok=True)
    name = str(metadata["name"])
    model_path = model_dir / f"{name}.txt"
    metadata_path = model_dir / f"{name}.json"
    model.save_model(model_path)
    payload = {**metadata, "features": features, "model_file": model_path.name}
    metadata_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return metadata_path


def load_model_bundle(name: str, model_dir: Path = MODEL_DIR) -> tuple[lgb.Booster, dict]:
    metadata_path = model_dir / f"{name}.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    model = lgb.Booster(model_file=str(model_dir / metadata["model_file"]))
    return model, metadata


def train_final_binary_model(
    dataset_path: Path,
    label_col: str,
    train_end_year: int = 2024,
    train_start_year: int | None = 2012,
    validation_days: int = 365,
) -> tuple[Path, dict[str, float]]:
    df = pd.read_parquet(dataset_path)
    if label_col not in df:
        raise KeyError(f"Missing label column {label_col}")

    dates = pd.to_datetime(df["date"])
    train = df[dates.dt.year <= train_end_year].copy()
    if train_start_year is not None:
        train = train[train["date"].dt.year >= train_start_year].copy()

    valid_cutoff = train["date"].max() - pd.Timedelta(days=validation_days)
    inner_train = train[train["date"] < valid_cutoff].copy()
    valid = train[train["date"] >= valid_cutoff].copy()
    features = feature_columns(df)
    model = _fit_lgbm_final(inner_train, valid, features, label_col)

    pred_col = f"pred_{label_col.removeprefix('label_')}"
    valid_scored = valid.dropna(subset=[label_col]).copy()
    valid_scored[pred_col] = model.predict(
        valid_scored[features],
        num_iteration=model.best_iteration,
    )
    metrics = _score_predictions(valid_scored, pred_col, label_col)
    name = f"lgbm_{label_col.removeprefix('label_')}_train_{train_end_year}"
    metrics.update(
        {
            "name": name,
            "label": label_col,
            "train_start_year": float(train_start_year) if train_start_year is not None else np.nan,
            "train_end_year": float(train_end_year),
            "train_rows": float(len(train)),
            "feature_count": float(len(features)),
            "best_iteration": float(model.best_iteration or 0),
        }
    )
    metadata_path = save_model_bundle(model, features, metrics | {"name": name})
    return metadata_path, metrics


def train_final_bucket_models(
    dataset_path: Path,
    thresholds: tuple[float, ...] = (0.05, 0.07),
    train_end_year: int = 2024,
    train_start_year: int | None = 2012,
) -> pd.DataFrame:
    rows = []
    for threshold in thresholds:
        pct = int(round(threshold * 100))
        for side in ("up", "down"):
            label_col = f"label_{side}_{pct}pct_{HORIZON_DAYS}d"
            _, metrics = train_final_binary_model(
                dataset_path=dataset_path,
                label_col=label_col,
                train_end_year=train_end_year,
                train_start_year=train_start_year,
            )
            rows.append(metrics)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(rows)
    summary.to_csv(REPORTS_DIR / f"final_model_training_summary_{train_end_year}.csv", index=False)
    return summary


def train_expansion_for_year(
    dataset_path: Path,
    threshold: float,
    test_year: int,
    train_start_year: int | None = None,
    train_end_year: int | None = None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    pct = int(round(threshold * 100))
    label_col = f"label_expansion_{pct}pct_{HORIZON_DAYS}d"
    predictions, metrics = train_binary_for_year(
        dataset_path=dataset_path,
        label_col=label_col,
        test_year=test_year,
        train_start_year=train_start_year,
        train_end_year=train_end_year,
    )
    metrics["threshold"] = threshold
    return predictions, metrics


def write_report(predictions: pd.DataFrame, metrics: dict[str, float], name: str) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(REPORTS_DIR / f"{name}_predictions.parquet", index=False)
    pd.DataFrame([metrics]).to_csv(REPORTS_DIR / f"{name}_metrics.csv", index=False)


def build_dataset(
    output_path: Path = PROCESSED_DIR / "daily_features.parquet",
    source: str = "silver",
) -> Path:
    from koscine.data_io import load_cash_daily, load_index_daily, load_silver_market_daily
    from koscine.features import add_features, add_forward_labels

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    if source == "silver":
        cash, indices = load_silver_market_daily()
    elif source == "raw":
        cash = load_cash_daily()
        indices = load_index_daily()
    else:
        raise ValueError(f"Unsupported source: {source}")
    features = add_features(cash, indices)
    labelled = add_forward_labels(features)
    labelled.to_parquet(output_path, index=False)
    return output_path


def _load_market_source(source: str):
    from koscine.data_io import load_cash_daily, load_index_daily, load_silver_market_daily

    if source == "silver":
        return load_silver_market_daily()
    if source == "raw":
        return load_cash_daily(), load_index_daily()
    raise ValueError(f"Unsupported source: {source}")


def _trading_tail_bounds(
    dates: pd.Series,
    end_date: str,
    lookback_trading_days: int,
    context_trading_days: int,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    end_ts = pd.Timestamp(end_date).normalize()
    unique_dates = pd.Series(pd.to_datetime(dates).dt.normalize().dropna().unique()).sort_values()
    unique_dates = unique_dates[unique_dates.le(end_ts)].reset_index(drop=True)
    if unique_dates.empty:
        raise ValueError(f"No source trading dates found up to {end_date}")
    replace_idx = max(0, len(unique_dates) - lookback_trading_days)
    context_idx = max(0, replace_idx - context_trading_days)
    return pd.Timestamp(unique_dates.iloc[context_idx]), pd.Timestamp(unique_dates.iloc[replace_idx])


def _write_replaced_dataset(
    output_path: Path,
    replacement: pd.DataFrame,
    replace_start: pd.Timestamp,
    end_ts: pd.Timestamp,
) -> None:
    """Stream an ordered date-range replacement into a new Parquet file.

    Loading the old dataset and concatenating it with the replacement causes
    pandas to consolidate all numeric columns into a second full-size array.
    At the current production size that allocation is about 1.8 GiB.  Stream
    old row groups around the replacement instead, then atomically swap files.
    """
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    keys = ["date", "symbol"]
    replacement = (
        replacement.drop_duplicates(keys, keep="last")
        .sort_values(keys)
        .reset_index(drop=True)
    )

    parquet = pq.ParquetFile(output_path)
    schema = parquet.schema_arrow
    expected = set(schema.names)
    actual = set(replacement.columns)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(
            f"Feature schema changed during tail refresh; missing={missing}, extra={extra}"
        )
    replacement_table = pa.Table.from_pandas(
        replacement.reindex(columns=schema.names),
        schema=schema,
        preserve_index=False,
        safe=False,
    )

    date_index = schema.get_field_index("date")
    date_type = schema.field(date_index).type
    start_scalar = pa.scalar(replace_start.to_pydatetime(), type=date_type)
    end_scalar = pa.scalar(end_ts.to_pydatetime(), type=date_type)
    temp_path = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    inserted = False
    try:
        try:
            with pq.ParquetWriter(temp_path, schema, compression="snappy") as writer:
                for batch in parquet.iter_batches(batch_size=65_536):
                    dates = batch.column(date_index)
                    before = batch.filter(pc.less(dates, start_scalar))
                    after = batch.filter(pc.greater(dates, end_scalar))
                    if before.num_rows:
                        writer.write_batch(before)
                    if after.num_rows:
                        if not inserted:
                            writer.write_table(replacement_table)
                            inserted = True
                        writer.write_batch(after)
                if not inserted:
                    writer.write_table(replacement_table)
        finally:
            parquet.close()
        temp_path.replace(output_path)
    finally:
        temp_path.unlink(missing_ok=True)


def refresh_dataset_tail(
    output_path: Path = PROCESSED_DIR / "daily_features.parquet",
    source: str = "silver",
    end_date: str | None = None,
    lookback_trading_days: int = 400,
    context_trading_days: int = 320,
) -> Path:
    """Refresh recent feature rows without recomputing the full history.

    The context window is intentionally larger than the longest base rolling
    feature window, so refreshed rows are computed with proper past context.
    Existing rows after end_date are left untouched.
    """
    if not output_path.exists():
        return build_dataset(output_path, source=source)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    cash, indices = _load_market_source(source)
    cash = cash.copy()
    cash["date"] = pd.to_datetime(cash["date"]).dt.normalize()
    if end_date is None:
        end_date = pd.Timestamp(cash["date"].max()).strftime("%Y-%m-%d")
    end_ts = pd.Timestamp(end_date).normalize()
    context_start, replace_start = _trading_tail_bounds(
        cash["date"],
        end_date,
        lookback_trading_days,
        context_trading_days,
    )

    from koscine.features import add_features, add_forward_labels

    cash_tail = cash[cash["date"].between(context_start, end_ts)].copy()
    index_tail = None
    if indices is not None and not indices.empty:
        index_tail = indices.copy()
        index_tail["date"] = pd.to_datetime(index_tail["date"]).dt.normalize()
        index_tail = index_tail[index_tail["date"].between(context_start, end_ts)]
    features = add_features(cash_tail, index_tail)
    labelled = add_forward_labels(features)
    labelled["date"] = pd.to_datetime(labelled["date"]).dt.normalize()
    replacement = labelled[labelled["date"].between(replace_start, end_ts)].copy()

    # Release the large source/feature intermediates before loading the
    # existing 193-column production dataset.
    del cash, indices, cash_tail, index_tail, features, labelled
    gc.collect()

    _write_replaced_dataset(output_path, replacement, replace_start, end_ts)
    return output_path
