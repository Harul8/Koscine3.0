from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from koscine3.data.feature_registry import build_feature_registry, write_feature_audit
from koscine3.data.sources import load_market_data, read_data_source
from koscine3.data.universe import UniverseConfig, build_universe, write_universe_manifest
from koscine3.datasets.splits import DEFAULT_SPLITS, WalkForwardSplit, between_dates
from koscine3.datasets.supervised_builder import build_supervised_dataset, model_feature_columns
from koscine3.evaluation.gold_metrics import build_gold_report
from koscine3.evaluation.reports import write_manifest, write_report_tables
from koscine3.models.baselines import make_baseline_predictions
from koscine3.models.train import predict_swing_model, save_model, train_swing_model
from koscine3.paths import RUNS_DIR
from koscine3.selection.daily_selector import SelectorConfig, select_daily_signals
from koscine3.selection.tune import tune_selector_config


@dataclass(frozen=True)
class ExperimentConfig:
    run_id: str = "koscine3_v1"
    universe_cutoff: str = "2025-12-31"
    prediction_start: str | None = None
    prediction_end: str | None = None
    base_train_end: str | None = None
    calibration_start: str | None = None
    calibration_end: str | None = None
    n_estimators: int = 120
    include_baselines: bool = True
    smoke: bool = False
    tune_selector: bool = False


def _load_required_data() -> pd.DataFrame:
    return load_market_data()


def _split_prediction_dataset(dataset: pd.DataFrame, split: WalkForwardSplit) -> pd.DataFrame:
    return dataset[between_dates(dataset, split.prediction_start, split.prediction_end)].copy()


def _run_prediction_table(
    predictions: pd.DataFrame,
    split_dir: Path,
    selector_config: SelectorConfig,
) -> pd.DataFrame:
    selected = select_daily_signals(predictions, selector_config)
    split_dir.mkdir(parents=True, exist_ok=True)
    selected.to_parquet(split_dir / "signals.parquet", index=False)
    selected[selected["selected"]].to_csv(split_dir / "selected_signals.csv", index=False)
    report = build_gold_report(selected)
    write_report_tables(report, split_dir / "gold_report")
    return selected


def run_experiment(config: ExperimentConfig) -> Path:
    run_dir = RUNS_DIR / config.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    source = read_data_source()
    market = _load_required_data()
    if config.smoke:
        # Keep the full feature schema, but shorten dates for fast local verification.
        market = market[market["date"].ge(pd.Timestamp("2021-01-01"))].copy()

    registry = build_feature_registry(market)
    registry.assert_safe()
    write_feature_audit(registry, run_dir / "feature_audit.csv")

    universe_config = UniverseConfig(cutoff_date=config.universe_cutoff)
    universe = build_universe(market, universe_config)
    universe.to_csv(run_dir / "universe.csv", index=False)
    write_universe_manifest(universe, run_dir / "universe_manifest.json", universe_config)

    dataset = build_supervised_dataset(
        market,
        universe,
        registry,
        output_manifest_path=run_dir / "dataset_manifest.json",
    )
    dataset.to_parquet(run_dir / "dataset.parquet", index=False)
    feature_columns = model_feature_columns(registry, dataset)

    splits = DEFAULT_SPLITS
    if config.prediction_start and config.prediction_end:
        splits = [
            WalkForwardSplit(
                name="custom",
                base_train_end=config.base_train_end or "2024-12-31",
                calibration_start=config.calibration_start or "2025-01-01",
                calibration_end=config.calibration_end or "2025-12-31",
                prediction_start=config.prediction_start,
                prediction_end=config.prediction_end,
            )
        ]
    default_selector_config = SelectorConfig()
    all_selected: list[pd.DataFrame] = []

    if config.include_baselines:
        for baseline_id in ["liquidity", "momentum", "compression", "random"]:
            for split in splits:
                period = _split_prediction_dataset(dataset, split)
                predictions = make_baseline_predictions(period, baseline_id)
                split_dir = run_dir / "baselines" / baseline_id / split.name
                all_selected.append(_run_prediction_table(predictions, split_dir, default_selector_config))

    for split in splits:
        model_id = f"{config.run_id}_{split.name}_gbdt"
        model = train_swing_model(
            dataset,
            feature_columns,
            split,
            model_id=model_id,
            n_estimators=config.n_estimators,
        )
        model_dir = run_dir / "models" / split.name
        save_model(model, model_dir)
        selector_config = default_selector_config
        if config.tune_selector:
            calibration_period = dataset[
                between_dates(dataset, split.calibration_start, split.calibration_end)
            ].copy()
            calibration_predictions = predict_swing_model(model, calibration_period)
            selector_config, selector_sweep = tune_selector_config(
                calibration_predictions,
                selector_id=f"selector_tuned_{split.name}",
            )
        else:
            selector_sweep = pd.DataFrame()
        period = _split_prediction_dataset(dataset, split)
        predictions = predict_swing_model(model, period)
        split_dir = run_dir / "model_predictions" / split.name
        if not selector_sweep.empty:
            split_dir.mkdir(parents=True, exist_ok=True)
            selector_sweep.to_csv(split_dir / "selector_tuning.csv", index=False)
        all_selected.append(_run_prediction_table(predictions, split_dir, selector_config))

    combined = pd.concat(all_selected, ignore_index=True) if all_selected else pd.DataFrame()
    if not combined.empty:
        combined.to_parquet(run_dir / "all_signals.parquet", index=False)
        write_report_tables(build_gold_report(combined), run_dir / "combined_gold_report")

    write_manifest(
        {
            "run_id": config.run_id,
            "source": source.__dict__,
            "config": config.__dict__,
            "default_selector_config": default_selector_config.__dict__,
            "tune_selector": config.tune_selector,
            "splits": [s.__dict__ for s in splits],
        },
        run_dir / "manifest.json",
    )
    return run_dir


def read_run_manifest(run_dir: Path) -> dict[str, object]:
    return json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
