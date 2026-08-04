from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd

from koscine.backtest import run_backtest
from koscine.catboost_models import (
    ensemble_predictions,
    predict_catboost_buckets_year,
    train_catboost_bucket_models,
)
from koscine.clean_direction import CleanDirectionConfig, run_clean_direction
from koscine.config import HORIZON_DAYS, MODEL_DIR, PROCESSED_DIR, REPORTS_DIR
from koscine.diagnostics import write_feature_importance, write_prediction_diagnostics
from koscine.experiments import evaluate_predictions_by_quarter, run_quarter_experiment
from koscine.opportunity_ranker import (
    OpportunityRankerConfig,
    apply_opportunity_ranker,
    evaluate_opportunity,
    run_opportunity_pipeline,
    train_opportunity_ranker,
)
from koscine.portfolio import PortfolioConfig, run_portfolio_backtest, run_portfolio_sweep
from koscine.predict import predict_buckets_year
from koscine.production import ProdModelConfig, predict_prod, prod_auto, train_prod_models
from koscine.quality import QualityLongConfig, run_quality_long_monthly
from koscine.rolling import RollingConfig, run_monthly_retrain
from koscine.segmented_quality import SegmentedQualityConfig, run_segmented_quality
from koscine.swing_eval import (
    SwingEvalConfig,
    default_eval_output_dir,
    read_prediction_inputs,
    write_swing_evaluation,
)
from koscine.tiered_clean_direction import (
    TieredCleanConfig,
    TieredProdConfig,
    predict_tiered_prod,
    predict_tiered_prod_many,
    run_tiered_clean_direction,
    tiered_prod_auto,
    train_tiered_prod_models,
)
from koscine.top30_trade_quality import Top30TradeQualityConfig, run_top30_trade_quality
from koscine.training import (
    build_dataset,
    refresh_dataset_tail,
    train_binary_for_year,
    train_expansion_for_year,
    train_final_bucket_models,
    write_report,
)
from koscine.walkforward_quality import WalkForwardQualityConfig, run_walkforward_quality


def _dataset_path(args: argparse.Namespace) -> Path:
    return Path(args.dataset) if args.dataset else PROCESSED_DIR / "daily_features.parquet"


def cmd_build_dataset(args: argparse.Namespace) -> None:
    path = build_dataset(_dataset_path(args), source=args.source)
    print(f"wrote {path}")


def cmd_train(args: argparse.Namespace) -> None:
    predictions, metrics = train_expansion_for_year(
        dataset_path=_dataset_path(args),
        threshold=args.threshold,
        test_year=args.test_year,
        train_start_year=args.train_start_year,
        train_end_year=args.train_end_year,
    )
    name = f"lgbm_expansion_{int(args.threshold * 100)}pct_test_{args.test_year}"
    write_report(predictions, metrics, name)
    print(pd.DataFrame([metrics]).to_string(index=False))
    print(f"wrote {REPORTS_DIR / (name + '_metrics.csv')}")


def cmd_walk_forward(args: argparse.Namespace) -> None:
    rows = []
    for year in range(args.start_test_year, args.end_test_year + 1):
        predictions, metrics = train_expansion_for_year(
            dataset_path=_dataset_path(args),
            threshold=args.threshold,
            test_year=year,
            train_start_year=args.train_start_year,
            train_end_year=year - 1,
        )
        name = f"lgbm_expansion_{int(args.threshold * 100)}pct_test_{year}"
        write_report(predictions, metrics, name)
        rows.append(metrics)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(rows)
    summary_path = REPORTS_DIR / f"walk_forward_{int(args.threshold * 100)}pct_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(summary.to_string(index=False))
    print(f"wrote {summary_path}")


def cmd_train_buckets(args: argparse.Namespace) -> None:
    rows = []
    for threshold in args.thresholds:
        pct = int(round(threshold * 100))
        for side in ("up", "down"):
            label_col = f"label_{side}_{pct}pct_{HORIZON_DAYS}d"
            predictions, metrics = train_binary_for_year(
                dataset_path=_dataset_path(args),
                label_col=label_col,
                test_year=args.test_year,
                train_start_year=args.train_start_year,
                train_end_year=args.train_end_year,
            )
            name = f"lgbm_{side}_{pct}pct_test_{args.test_year}"
            write_report(predictions, metrics, name)
            rows.append(metrics)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(rows)
    summary_path = REPORTS_DIR / f"bucket_models_test_{args.test_year}_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(summary.to_string(index=False))
    print(f"wrote {summary_path}")


def cmd_walk_forward_buckets(args: argparse.Namespace) -> None:
    rows = []
    for year in range(args.start_test_year, args.end_test_year + 1):
        for threshold in args.thresholds:
            pct = int(round(threshold * 100))
            for side in ("up", "down"):
                label_col = f"label_{side}_{pct}pct_{HORIZON_DAYS}d"
                predictions, metrics = train_binary_for_year(
                    dataset_path=_dataset_path(args),
                    label_col=label_col,
                    test_year=year,
                    train_start_year=args.train_start_year,
                    train_end_year=year - 1,
                )
                name = f"wf_lgbm_{side}_{pct}pct_test_{year}"
                write_report(predictions, metrics, name)
                rows.append(metrics)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(rows)
    summary_path = REPORTS_DIR / (
        f"walk_forward_buckets_{args.start_test_year}_{args.end_test_year}_summary.csv"
    )
    summary.to_csv(summary_path, index=False)
    print(summary.to_string(index=False))
    print(f"wrote {summary_path}")


def cmd_train_final(args: argparse.Namespace) -> None:
    summary = train_final_bucket_models(
        dataset_path=_dataset_path(args),
        thresholds=tuple(args.thresholds),
        train_end_year=args.train_end_year,
        train_start_year=args.train_start_year,
    )
    print(summary.to_string(index=False))


def cmd_predict_year(args: argparse.Namespace) -> None:
    predictions = predict_buckets_year(
        dataset_path=_dataset_path(args),
        test_year=args.test_year,
        train_end_year=args.train_end_year,
        thresholds=tuple(args.thresholds),
    )
    print(f"wrote predictions for {len(predictions)} bucket rows")


def cmd_backtest(args: argparse.Namespace) -> None:
    predictions = pd.read_parquet(args.predictions)
    trades, summary = run_backtest(
        predictions,
        top_n=args.top_n,
        cost_bps=args.cost_bps,
        name=args.name,
    )
    print(summary.to_string(index=False))
    print(f"wrote {len(trades)} trades")


def cmd_portfolio(args: argparse.Namespace) -> None:
    predictions = pd.read_parquet(args.predictions)
    if "pred_col" not in predictions.columns and "score" in predictions.columns:
        predictions["pred_col"] = "score"
    if "label_col" not in predictions.columns and "actual_hit" in predictions.columns:
        predictions["label_col"] = "actual_hit"
        predictions["actual"] = predictions["actual_hit"]
    config = PortfolioConfig(
        daily_long_slots=args.daily_long_slots,
        daily_short_slots=args.daily_short_slots,
        max_open_long=args.max_open_long,
        max_open_short=args.max_open_short,
        allocation_pct=args.allocation_pct,
        cost_bps=args.cost_bps,
        min_score=args.min_score,
        max_adverse_move=args.max_adverse_move,
        score_proportional_sizing=args.score_proportional_sizing,
        score_size_floor=args.score_size_floor,
        max_allocation_pct=args.max_allocation_pct,
        use_atr_stop=args.use_atr_stop,
        atr_stop_mult=args.atr_stop_mult,
        sector_concentration_max=args.sector_concentration_max,
        use_regime_gate=args.use_regime_gate,
        min_rr_ratio=args.min_rr_ratio,
        rolling_derisk=args.rolling_derisk,
        derisk_window=args.derisk_window,
        derisk_warn_avg_pnl=args.derisk_warn_avg_pnl,
        derisk_crit_avg_pnl=args.derisk_crit_avg_pnl,
    )
    trades, equity, summary = run_portfolio_backtest(predictions, config=config, name=args.name)
    print(summary.to_string(index=False))
    print(f"wrote {len(trades)} closed trades and {len(equity)} equity rows")


def cmd_sweep(args: argparse.Namespace) -> None:
    predictions = pd.read_parquet(args.predictions)
    sweep = run_portfolio_sweep(
        predictions,
        daily_slots=tuple(args.daily_slots),
        min_scores=tuple(args.min_scores),
        max_adverse_moves=tuple(args.max_adverse_moves),
        cost_bps=args.cost_bps,
        name=args.name,
    )
    print(sweep.to_string(index=False))


def cmd_diagnostics(args: argparse.Namespace) -> None:
    predictions = pd.read_parquet(args.predictions)
    importance = write_feature_importance()
    diagnostics = write_prediction_diagnostics(predictions, top_k=args.top_k)
    print(f"wrote feature importance rows: {len(importance)}")
    print(diagnostics["bucket"].to_string(index=False))


def cmd_eval_swing(args: argparse.Namespace) -> None:
    predictions_path = Path(args.predictions)
    predictions = read_prediction_inputs(predictions_path)
    output_dir = Path(args.output_dir) if args.output_dir else default_eval_output_dir(args.name)
    config = SwingEvalConfig(
        dataset_path=_dataset_path(args),
        near_fraction=args.near_fraction,
        cost_bps=args.cost_bps,
        min_hit_near=args.min_hit_near,
        preferred_opposite_cap=args.preferred_opposite_cap,
        hard_opposite_cap=args.hard_opposite_cap,
    )
    paths = write_swing_evaluation(
        predictions,
        output_dir=output_dir,
        source=str(predictions_path),
        config=config,
        signal_set=args.signal_set,
    )
    summary = pd.read_csv(paths["summary"]) if paths["summary"].exists() else pd.DataFrame()
    if not summary.empty:
        aggregate = summary[summary["slice"].eq("aggregate")]
        print(aggregate.to_string(index=False))
    print(f"wrote swing evaluation: {output_dir}")
    print(f"summary: {paths['summary']}")
    print(f"manifest: {paths['manifest']}")


def cmd_train_catboost(args: argparse.Namespace) -> None:
    summary = train_catboost_bucket_models(
        dataset_path=_dataset_path(args),
        thresholds=tuple(args.thresholds),
        train_end_year=args.train_end_year,
        train_start_year=args.train_start_year,
    )
    print(summary.to_string(index=False))


def cmd_predict_catboost(args: argparse.Namespace) -> None:
    predictions = predict_catboost_buckets_year(
        dataset_path=_dataset_path(args),
        test_year=args.test_year,
        train_end_year=args.train_end_year,
        thresholds=tuple(args.thresholds),
    )
    print(f"wrote CatBoost predictions for {len(predictions)} bucket rows")


def cmd_ensemble(args: argparse.Namespace) -> None:
    lgbm = pd.read_parquet(args.lgbm_predictions)
    catboost = pd.read_parquet(args.catboost_predictions)
    ensemble = ensemble_predictions(
        lgbm,
        catboost,
        lgbm_weight=args.lgbm_weight,
        test_year=args.test_year,
    )
    print(f"wrote ensemble predictions for {len(ensemble)} bucket rows")


def cmd_experiment_quarters(args: argparse.Namespace) -> None:
    run_dir = run_quarter_experiment(
        dataset_path=_dataset_path(args),
        train_start_year=args.train_start_year,
        train_end_year=args.train_end_year,
        run_name=args.run_name,
        thresholds=tuple(args.thresholds),
    )
    print(f"wrote quarter experiment run: {run_dir}")


def cmd_evaluate_quarters(args: argparse.Namespace) -> None:
    run_dir = evaluate_predictions_by_quarter(
        predictions_path=Path(args.predictions),
        model_family=args.model_family,
        run_name=args.run_name,
        year=args.year,
    )
    print(f"wrote quarter evaluation run: {run_dir}")


def cmd_monthly_retrain(args: argparse.Namespace) -> None:
    config = RollingConfig(
        train_start_year=args.train_start_year,
        first_prediction_month=args.first_prediction_month,
        last_prediction_month=args.last_prediction_month,
        feature_profile=args.feature_profile,
        sides=tuple(args.sides),
        thresholds=tuple(args.thresholds),
        validation_days=args.validation_days,
        train_cutoff_day=args.train_cutoff_day,
    )
    run_dir = run_monthly_retrain(_dataset_path(args), config=config, run_name=args.run_name)
    print(f"wrote monthly retrain run: {run_dir}")


def cmd_prod_train(args: argparse.Namespace) -> None:
    config = ProdModelConfig(
        train_start_year=args.train_start_year,
        train_cutoff_day=args.train_cutoff_day,
        feature_profile=args.feature_profile,
        sides=tuple(args.sides),
        thresholds=tuple(args.thresholds),
        min_score=args.min_score,
        max_adverse_move=args.max_adverse_move,
        cost_bps=args.cost_bps,
    )
    run_dir = train_prod_models(
        dataset_path=_dataset_path(args),
        prediction_month=args.prediction_month,
        config=config,
        run_name=args.run_name,
    )
    print(f"wrote production models: {run_dir}")
    print("updated models/prod/current")


def cmd_prod_predict(args: argparse.Namespace) -> None:
    predictions = predict_prod(
        dataset_path=_dataset_path(args),
        as_of_date=args.as_of_date,
    )
    print(predictions.to_string(index=False))
    print(f"wrote {len(predictions)} production predictions")


def cmd_prod_auto(args: argparse.Namespace) -> None:
    config = ProdModelConfig(
        train_start_year=args.train_start_year,
        train_cutoff_day=args.train_cutoff_day,
        feature_profile=args.feature_profile,
        sides=tuple(args.sides),
        thresholds=tuple(args.thresholds),
        min_score=args.min_score,
        cost_bps=args.cost_bps,
    )
    predictions = prod_auto(
        dataset_path=_dataset_path(args),
        as_of_date=args.as_of_date,
        config=config,
    )
    print(predictions.to_string(index=False))
    print(f"wrote {len(predictions)} production predictions")


def cmd_quality_long(args: argparse.Namespace) -> None:
    config = QualityLongConfig(
        train_start_year=args.train_start_year,
        first_prediction_month=args.first_prediction_month,
        last_prediction_month=args.last_prediction_month,
        feature_profile=args.feature_profile,
        thresholds=tuple(args.thresholds),
        train_cutoff_day=args.train_cutoff_day,
        validation_days=args.validation_days,
        adverse_limit=args.adverse_limit,
        annual_signal_target=args.annual_signal_target,
    )
    run_dir = run_quality_long_monthly(_dataset_path(args), config=config, run_name=args.run_name)
    print(f"wrote quality long run: {run_dir}")


def cmd_walkforward_quality(args: argparse.Namespace) -> None:
    config = WalkForwardQualityConfig(
        train_start_year=args.train_start_year,
        start_test_year=args.start_test_year,
        end_test_year=args.end_test_year,
        feature_profile=args.feature_profile,
        thresholds=tuple(args.thresholds),
        validation_days=args.validation_days,
        train_cutoff_month=args.train_cutoff_month,
        train_cutoff_day=args.train_cutoff_day,
        bad_rate_cap=args.bad_rate_cap,
        topn_step=args.topn_step,
        topn_max=args.topn_max,
        calibration_lookback_years=args.calibration_lookback_years,
    )
    run_dir = run_walkforward_quality(_dataset_path(args), config=config, run_name=args.run_name)
    print(f"wrote walk-forward quality run: {run_dir}")


def cmd_segmented_quality(args: argparse.Namespace) -> None:
    config = SegmentedQualityConfig(
        train_start_year=args.train_start_year,
        start_test_year=args.start_test_year,
        end_test_year=args.end_test_year,
        feature_profile=args.feature_profile,
        validation_days=args.validation_days,
        train_cutoff_month=args.train_cutoff_month,
        train_cutoff_day=args.train_cutoff_day,
        bad_rate_cap=args.bad_rate_cap,
        topn_step=args.topn_step,
        topn_max=args.topn_max,
        calibration_start_year=args.calibration_start_year,
        bad_negative_weight=args.bad_negative_weight,
        soft_negative_weight=args.soft_negative_weight,
        objective=args.objective,
    )
    run_dir = run_segmented_quality(_dataset_path(args), config=config, run_name=args.run_name)
    print(f"wrote segmented quality run: {run_dir}")


def cmd_top30_trade_quality(args: argparse.Namespace) -> None:
    config = Top30TradeQualityConfig(
        train_start_year=args.train_start_year,
        start_test_year=args.start_test_year,
        end_test_year=args.end_test_year,
        target_pct=args.target_pct,
        stop_pct=args.stop_pct,
        max_hold_days=args.max_hold_days,
        validation_days=args.validation_days,
        train_cutoff_month=args.train_cutoff_month,
        train_cutoff_day=args.train_cutoff_day,
        min_trades=args.min_trades,
    )
    run_dir = run_top30_trade_quality(_dataset_path(args), config=config, run_name=args.run_name)
    print(f"wrote top30 trade-quality run: {run_dir}")


def cmd_clean_direction(args: argparse.Namespace) -> None:
    config = CleanDirectionConfig(
        train_start_year=args.train_start_year,
        start_test_year=args.start_test_year,
        end_test_year=args.end_test_year,
        target_pct=args.target_pct,
        adverse_limit=args.adverse_limit,
        validation_days=args.validation_days,
        train_cutoff_month=args.train_cutoff_month,
        train_cutoff_day=args.train_cutoff_day,
        precision_floor=args.precision_floor,
        min_validation_calls=args.min_validation_calls,
        train_universe=args.train_universe,
        liquid_weight=args.liquid_weight,
        max_weekly_abs_move=args.max_weekly_abs_move,
        episode_decay_weight=args.episode_decay_weight,
        ensemble_lgbm_weight=args.ensemble_lgbm_weight,
        use_catboost=not args.no_catboost,
        direct_side_weight=args.direct_side_weight,
    )
    run_dir = run_clean_direction(_dataset_path(args), config=config, run_name=args.run_name)
    print(f"wrote clean-direction run: {run_dir}")


def cmd_tiered_clean_direction(args: argparse.Namespace) -> None:
    config = TieredCleanConfig(
        train_start_year=args.train_start_year,
        start_test_year=args.start_test_year,
        end_test_year=args.end_test_year,
        validation_days=args.validation_days,
        train_cutoff_month=args.train_cutoff_month,
        train_cutoff_day=args.train_cutoff_day,
        adverse_limit=args.adverse_limit,
        max_weekly_abs_move=args.max_weekly_abs_move,
        min_validation_calls=args.min_validation_calls,
        bad_rate_cap=args.bad_rate_cap,
        topn_step=args.topn_step,
        topn_max=args.topn_max,
        lgbm_weight=args.lgbm_weight,
        use_catboost=not args.no_catboost,
        train_all_symbols=not args.tier_only_train,
        use_vol_adjusted_labels=args.use_vol_adjusted_labels,
        temporal_decay_per_year=args.temporal_decay_per_year,
        n_seeds=args.n_seeds,
        use_calibration=args.use_calibration,
    )
    run_dir = run_tiered_clean_direction(_dataset_path(args), config=config, run_name=args.run_name)
    print(f"wrote tiered clean-direction run: {run_dir}")


def cmd_tiered_prod_train(args: argparse.Namespace) -> None:
    config = TieredProdConfig(
        train_start_year=args.train_start_year,
        train_cutoff_day=args.train_cutoff_day,
        adverse_limit=args.adverse_limit,
        max_weekly_abs_move=args.max_weekly_abs_move,
        lgbm_weight=args.lgbm_weight,
        use_catboost=not args.no_catboost,
        min_score=args.min_score,
        train_all_symbols=not args.tier_only_train,
        use_vol_adjusted_labels=args.use_vol_adjusted_labels,
        temporal_decay_per_year=args.temporal_decay_per_year,
        n_seeds=args.n_seeds,
        use_calibration=args.use_calibration,
    )
    run_dir = train_tiered_prod_models(
        dataset_path=_dataset_path(args),
        prediction_month=args.prediction_month,
        config=config,
        run_name=args.run_name,
    )
    print(f"wrote tiered production models: {run_dir}")
    print("updated models/prod/current")


def cmd_tiered_prod_predict(args: argparse.Namespace) -> None:
    predictions = predict_tiered_prod(
        dataset_path=_dataset_path(args),
        as_of_date=args.as_of_date,
    )
    print(predictions.to_string(index=False))
    print(f"wrote {len(predictions)} tiered production predictions")


def cmd_tiered_prod_auto(args: argparse.Namespace) -> None:
    config = TieredProdConfig(
        train_start_year=args.train_start_year,
        train_cutoff_day=args.train_cutoff_day,
        adverse_limit=args.adverse_limit,
        max_weekly_abs_move=args.max_weekly_abs_move,
        lgbm_weight=args.lgbm_weight,
        use_catboost=not args.no_catboost,
        min_score=args.min_score,
        train_all_symbols=not args.tier_only_train,
    )
    predictions = tiered_prod_auto(
        dataset_path=_dataset_path(args),
        as_of_date=args.as_of_date,
        config=config,
    )
    print(predictions.to_string(index=False))
    print(f"wrote {len(predictions)} tiered production predictions")


def _tiered_prod_config_from_args(args: argparse.Namespace) -> TieredProdConfig:
    return TieredProdConfig(
        train_start_year=args.train_start_year,
        train_cutoff_day=args.train_cutoff_day,
        adverse_limit=args.adverse_limit,
        max_weekly_abs_move=args.max_weekly_abs_move,
        lgbm_weight=args.lgbm_weight,
        use_catboost=not args.no_catboost,
        min_score=args.min_score,
        train_all_symbols=not args.tier_only_train,
    )


def _run_esn_ingestion(start_date: str, end_date: str, esn_root: str) -> None:
    root = Path(esn_root)
    if not root.exists():
        raise FileNotFoundError(f"ESN ingestion project not found: {root}")
    print(f"fetching missing/raw data with ESN1.0: {start_date} -> {end_date}")
    subprocess.run(
        [sys.executable, "-u", "-m", "pipeline.fetch", start_date, end_date],
        cwd=root,
        check=True,
    )
    print("fetching/parsing NSDL FII cash archives")
    subprocess.run(
        [sys.executable, "-u", "-m", "pipeline.fetch_fiidii", "--fetch", start_date, end_date],
        cwd=root,
        check=True,
    )
    append_start = _silver_append_start_from_silver(start_date)
    dates = [d.strftime("%Y-%m-%d") for d in pd.bdate_range(append_start, end_date)]
    print(f"appending silver tables with ESN1.0 from {append_start} to {end_date} for {len(dates)} business dates")
    for i in range(0, len(dates), 20):
        command = [sys.executable, "-u", "-m", "pipeline.silver", "append-day", *dates[i:i + 20]]
        if "append-day" not in command:
            raise RuntimeError("Refusing full silver rebuild from CLI pipeline; use append-day only.")
        subprocess.run(command, cwd=root, check=True)


def _run_esn_fii_cash(start_date: str, end_date: str, esn_root: str) -> None:
    root = Path(esn_root)
    if not root.exists():
        raise FileNotFoundError(f"ESN ingestion project not found: {root}")
    print("fetching/parsing NSDL FII cash archives")
    subprocess.run(
        [sys.executable, "-u", "-m", "pipeline.fetch_fiidii", "--fetch", start_date, end_date],
        cwd=root,
        check=True,
    )


def _silver_latest_complete_date() -> pd.Timestamp | None:
    from koscine.config import SILVER_DATA_ROOT

    required = (
        SILVER_DATA_ROOT / "eod_stock.parquet",
        SILVER_DATA_ROOT / "eod_deriv_daily.parquet",
        SILVER_DATA_ROOT / "indices.parquet",
        SILVER_DATA_ROOT / "participant_oi.parquet",
    )
    latest_dates = []
    for path in required:
        if not path.exists():
            return None
        try:
            dates = pd.read_parquet(path, columns=["date"])["date"]
        except Exception:
            return None
        if dates.empty:
            return None
        latest_dates.append(pd.to_datetime(dates).max().normalize())
    return min(latest_dates) if latest_dates else None


def _silver_covers(end_date: str) -> bool:
    latest = _silver_latest_complete_date()
    return latest is not None and latest >= pd.Timestamp(end_date).normalize()


def _silver_append_start_from_silver(requested_start: str) -> str:
    latest = _silver_latest_complete_date()
    if latest is None:
        return requested_start
    return latest.strftime("%Y-%m-%d")


def _prediction_dates(dataset_path: Path, start_date: str, end_date: str) -> list[str]:
    df = pd.read_parquet(dataset_path, columns=["date"])
    dates = pd.to_datetime(df["date"]).dt.normalize()
    mask = dates.between(pd.Timestamp(start_date), pd.Timestamp(end_date))
    return [d.strftime("%Y-%m-%d") for d in sorted(dates[mask].drop_duplicates())]


def _requested_prediction_months(start_date: str, end_date: str) -> set[str]:
    dates = pd.bdate_range(start_date, end_date)
    return {d.strftime("%Y-%m") for d in dates}


def _current_prod_prediction_month() -> str | None:
    manifest_path = MODEL_DIR / "prod" / "current" / "manifest.json"
    if not manifest_path.exists():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    month = manifest.get("prediction_month")
    return str(month) if month else None


def _needs_monthly_feature_rebuild(dataset_path: Path, start_date: str, end_date: str) -> bool:
    if not dataset_path.exists():
        return True
    return False


def cmd_prod_full_pipeline(args: argparse.Namespace) -> None:
    fetch_mode = args.fetch_mode
    if fetch_mode == "always" or (fetch_mode == "missing" and not _silver_covers(args.end_date)):
        fetch_start = _silver_append_start_from_silver(args.start_date) if fetch_mode == "missing" else args.start_date
        print(f"resume fetch from latest silver append date: {fetch_start} -> {args.end_date}")
        _run_esn_ingestion(fetch_start, args.end_date, args.esn_root)
    elif fetch_mode == "missing":
        print("silver already covers requested end date; skipping fetch")
        _run_esn_fii_cash(args.start_date, args.end_date, args.esn_root)
    else:
        print("fetch skipped by request")

    dataset_path = _dataset_path(args)
    if _needs_monthly_feature_rebuild(dataset_path, args.start_date, args.end_date):
        print("full Koscine feature rebuild from silver")
        dataset = build_dataset(dataset_path, source=args.source)
    else:
        print("daily mode: refreshing recent 400 trading days of features")
        dataset = refresh_dataset_tail(dataset_path, source=args.source, end_date=args.end_date)
    dates = _prediction_dates(dataset, args.start_date, args.end_date)
    if not dates:
        raise ValueError(f"No trading dates found in dataset for {args.start_date} to {args.end_date}")

    summaries = []
    current_month = _current_prod_prediction_month()
    if current_month is None:
        raise RuntimeError("No current production model found. Run tiered-prod-train manually before predicting.")
    for month, month_dates in pd.Series(dates).groupby(pd.Series(dates).str.slice(0, 7)):
        print(f"using current production model ({current_month}) for {month}")
        print(f"predicting {len(month_dates)} trading dates for {month}")
        summary = predict_tiered_prod_many(
            dataset_path=dataset,
            as_of_dates=list(month_dates),
            progress=lambda msg: print(msg, flush=True),
        )
        summary["prediction_month"] = month
        summary["model_run"] = str(MODEL_DIR / "prod" / "current")
        summaries.append(summary)

    out = pd.concat(summaries, ignore_index=True)
    reports_path = REPORTS_DIR / f"prod_full_pipeline_{args.start_date.replace('-', '')}_{args.end_date.replace('-', '')}.csv"
    reports_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(reports_path, index=False)
    print(out.to_string(index=False))
    print(f"wrote batch summary: {reports_path}")


def cmd_expansion_train(args: argparse.Namespace) -> None:
    from koscine.expansion_model import ExpansionTrainConfig, train_expansion_prod

    cfg = ExpansionTrainConfig(
        train_start_year=args.train_start_year,
        train_cutoff_day=args.train_cutoff_day,
        validation_days=args.validation_days,
        lgbm_weight=args.lgbm_weight,
        use_catboost=not args.no_catboost,
        n_seeds=args.n_seeds,
        temporal_decay_per_year=args.temporal_decay_per_year,
        use_calibration=not args.no_calibration,
        train_all_symbols=not args.tier_only_train,
    )
    run_dir = train_expansion_prod(
        dataset_path=_dataset_path(args),
        prediction_month=args.prediction_month,
        config=cfg,
        run_name=args.run_name,
    )
    print(f"wrote expansion prod models: {run_dir}")
    print("updated models/expansion/prod/current")


def cmd_expansion_predict(args: argparse.Namespace) -> None:
    from koscine.expansion_model import predict_expansion_prod

    dates = pd.bdate_range(args.start, args.end).strftime("%Y-%m-%d").tolist()
    summary = predict_expansion_prod(
        dataset_path=_dataset_path(args),
        as_of_dates=dates,
        progress=lambda msg: print(msg, flush=True),
    )
    print(summary.to_string(index=False))
    print(f"wrote {len(summary)} date summary rows")


def cmd_expansion_walkforward(args: argparse.Namespace) -> None:
    from koscine.expansion_model import ExpansionWalkforwardConfig, run_expansion_walkforward

    cfg = ExpansionWalkforwardConfig(
        train_start_year=args.train_start_year,
        start_test_year=args.start_test_year,
        end_test_year=args.end_test_year,
        validation_days=args.validation_days,
        lgbm_weight=args.lgbm_weight,
        use_catboost=not args.no_catboost,
        n_seeds=args.n_seeds,
        temporal_decay_per_year=args.temporal_decay_per_year,
        use_calibration=not args.no_calibration,
        train_all_symbols=not args.tier_only_train,
    )
    run_dir = run_expansion_walkforward(_dataset_path(args), config=cfg, run_name=args.run_name)
    print(f"wrote expansion walk-forward run: {run_dir}")


def cmd_signal_cards(args: argparse.Namespace) -> None:
    from koscine.signal_card import SignalCardConfig, write_signal_cards
    from koscine.regime import apply_regime_gate

    predictions = pd.read_parquet(args.predictions)
    if args.apply_regime_gate:
        predictions = apply_regime_gate(predictions)
        if not args.keep_all:
            predictions = predictions[predictions.get("passes_regime_gate", True)].copy()
    config = SignalCardConfig(
        atr_stop_mult=args.atr_stop_mult,
        min_rr_ratio=args.min_rr_ratio,
        base_allocation_pct=args.base_allocation_pct,
        max_allocation_pct=args.max_allocation_pct,
        score_threshold_for_scaling=args.score_threshold_for_scaling,
        sector_concentration_max=args.sector_concentration_max,
    )
    output_dir = Path(args.output_dir) if args.output_dir else None
    if output_dir is not None:
        out = write_signal_cards(predictions, output_dir=output_dir, config=config, only_actionable=not args.keep_all)
    else:
        out = write_signal_cards(predictions, config=config, only_actionable=not args.keep_all)
    print(f"wrote signal cards to {out}")


def cmd_model_health(args: argparse.Namespace) -> None:
    from koscine.model_health import HealthConfig, write_health_report

    trades = pd.read_parquet(args.trades)
    config = HealthConfig(
        rolling_trades=args.rolling_trades,
        warning_hit_rate=args.warning_hit_rate,
        critical_hit_rate=args.critical_hit_rate,
    )
    out_dir = Path(args.output_dir) if args.output_dir else None
    if out_dir is not None:
        out = write_health_report(trades, output_dir=out_dir, config=config)
    else:
        out = write_health_report(trades, config=config)
    print(f"wrote model health report to {out}")


def cmd_feature_audit(args: argparse.Namespace) -> None:
    from koscine.shap_diagnostics import write_shap_style_report

    paths = write_shap_style_report(min_gain_share=args.min_gain_share)
    for name, p in paths.items():
        print(f"{name}: {p}")


def cmd_calibration_report(args: argparse.Namespace) -> None:
    from koscine.calibration import calibration_report, fit_isotonic

    predictions = pd.read_parquet(args.predictions)
    score_col = args.score_col
    label_col = args.label_col
    if label_col not in predictions or score_col not in predictions:
        raise ValueError(f"Missing columns: {score_col} or {label_col}")
    rep = calibration_report(predictions[score_col].values, predictions[label_col].astype(float).values)
    out = REPORTS_DIR / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    rep.to_csv(out, index=False)
    print(rep.to_string(index=False))
    print(f"wrote {out}")


def _opportunity_config(args: argparse.Namespace) -> OpportunityRankerConfig:
    return OpportunityRankerConfig(
        train_start_year=args.train_start_year,
        train_end_year=args.train_end_year,
        validation_year=args.validation_year,
        train_cutoff_day=args.train_cutoff_day,
        adverse_limit=args.adverse_limit,
        max_weekly_abs_move=args.max_weekly_abs_move,
        lgbm_weight=args.lgbm_weight,
        train_all_symbols=not args.tier_only_train,
        go_plus_per_side=args.go_plus_per_side,
        go_per_side=args.go_per_side,
        min_direction_score=args.min_direction_score,
        min_rest_direction_score=args.min_rest_direction_score,
        max_opposite_score=args.max_opposite_score,
        max_long_opposite_score=args.max_long_opposite_score,
        max_short_opposite_score=args.max_short_opposite_score,
        min_score_gap=args.min_score_gap,
        min_move_power=args.min_move_power,
        go_plus_min_move_power=args.go_plus_min_move_power,
        min_go_score_quantile=args.min_go_score_quantile,
        min_model_validation_top5_capture_rate=args.min_model_validation_top5_capture_rate,
        go_plus_annual_target=args.go_plus_annual_target,
        go_annual_target=args.go_annual_target,
        go_plus_quality_threshold=args.go_plus_quality_threshold,
        go_quality_threshold=args.go_quality_threshold,
        target_negative_sample_ratio=args.target_negative_sample_ratio,
        target_max_negative_rows=args.target_max_negative_rows,
        ranker_max_train_rows=args.ranker_max_train_rows,
        rest35_long_addon_min_score=args.rest35_long_addon_min_score,
        rest35_long_addon_min_score_gap=args.rest35_long_addon_min_score_gap,
        rest35_long_addon_min_move_power=args.rest35_long_addon_min_move_power,
        rest35_long_addon_min_bb_rank=args.rest35_long_addon_min_bb_rank,
        rest35_long_addon_min_atm_iv=args.rest35_long_addon_min_atm_iv,
        rest35_long_addon_min_path_range=args.rest35_long_addon_min_path_range,
        rest35_long_addon_min_atr=args.rest35_long_addon_min_atr,
        liquid_long_addon_min_score=args.liquid_long_addon_min_score,
        liquid_long_addon_min_score_gap=args.liquid_long_addon_min_score_gap,
        liquid_long_addon_min_move_power=args.liquid_long_addon_min_move_power,
        liquid_long_addon_max_side_rank=args.liquid_long_addon_max_side_rank,
        liquid_long_addon_min_path_range=args.liquid_long_addon_min_path_range,
    )


def cmd_opportunity_train(args: argparse.Namespace) -> None:
    run_dir = train_opportunity_ranker(
        dataset_path=_dataset_path(args),
        config=_opportunity_config(args),
        run_name=args.run_name,
    )
    print(f"wrote opportunity ranker: {run_dir}")


def cmd_opportunity_apply(args: argparse.Namespace) -> None:
    predictions = apply_opportunity_ranker(
        dataset_path=_dataset_path(args),
        start_date=args.start_date,
        end_date=args.end_date,
    )
    summary = evaluate_opportunity(predictions, output_name=args.output_name)
    print(summary.to_string(index=False))
    print(f"wrote opportunity predictions: {len(predictions)} rows")


def cmd_opportunity_pipeline(args: argparse.Namespace) -> None:
    run_dir, predictions, summary = run_opportunity_pipeline(
        dataset_path=_dataset_path(args),
        config=_opportunity_config(args),
        start_date=args.start_date,
        end_date=args.end_date,
        run_name=args.run_name,
    )
    print(f"wrote opportunity ranker: {run_dir}")
    print(summary.to_string(index=False))
    print(f"wrote opportunity predictions: {len(predictions)} rows")


def cmd_run_all(args: argparse.Namespace) -> None:
    dataset = build_dataset(_dataset_path(args), source=args.source)
    print(f"dataset: {dataset}")

    training_summary = train_final_bucket_models(
        dataset_path=dataset,
        thresholds=tuple(args.thresholds),
        train_end_year=args.train_end_year,
        train_start_year=args.train_start_year,
    )
    print(training_summary.to_string(index=False))

    predictions = predict_buckets_year(
        dataset_path=dataset,
        test_year=args.test_year,
        train_end_year=args.train_end_year,
        thresholds=tuple(args.thresholds),
    )
    trades, backtest_summary = run_backtest(
        predictions,
        top_n=args.top_n,
        cost_bps=args.cost_bps,
        name=f"top{args.top_n}_{args.test_year}_cost{int(args.cost_bps)}bps",
    )
    print(backtest_summary.to_string(index=False))
    print(f"predictions: {len(predictions)} rows")
    print(f"trades: {len(trades)} rows")

    portfolio_config = PortfolioConfig(
        daily_long_slots=args.daily_long_slots,
        daily_short_slots=args.daily_short_slots,
        max_open_long=args.max_open_long,
        max_open_short=args.max_open_short,
        allocation_pct=args.allocation_pct,
        cost_bps=args.cost_bps,
        min_score=args.min_score,
        max_adverse_move=args.max_adverse_move,
    )
    portfolio_trades, equity, portfolio_summary = run_portfolio_backtest(
        predictions,
        config=portfolio_config,
        name=f"portfolio_{args.test_year}",
    )
    sweep = run_portfolio_sweep(
        predictions,
        daily_slots=tuple(args.sweep_daily_slots),
        min_scores=tuple(args.sweep_min_scores),
        cost_bps=args.cost_bps,
        name=f"portfolio_sweep_{args.test_year}",
    )
    importance = write_feature_importance()
    write_prediction_diagnostics(predictions)
    print(portfolio_summary.to_string(index=False))
    print(sweep.head(10).to_string(index=False))
    print(f"portfolio trades: {len(portfolio_trades)} rows")
    print(f"equity rows: {len(equity)} rows")
    print(f"feature importance rows: {len(importance)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Koscine rare-event trading system")
    parser.add_argument("--dataset", default=None, help="Processed dataset parquet path")
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build-dataset")
    build.add_argument("--source", choices=["silver", "raw"], default="silver")
    build.set_defaults(func=cmd_build_dataset)

    train = sub.add_parser("train")
    train.add_argument("--threshold", type=float, default=0.07)
    train.add_argument("--test-year", type=int, default=2025)
    train.add_argument("--train-start-year", type=int, default=None)
    train.add_argument("--train-end-year", type=int, default=2024)
    train.set_defaults(func=cmd_train)

    walk = sub.add_parser("walk-forward")
    walk.add_argument("--threshold", type=float, default=0.07)
    walk.add_argument("--start-test-year", type=int, default=2017)
    walk.add_argument("--end-test-year", type=int, default=2025)
    walk.add_argument("--train-start-year", type=int, default=2012)
    walk.set_defaults(func=cmd_walk_forward)

    buckets = sub.add_parser("train-buckets")
    buckets.add_argument("--thresholds", type=float, nargs="+", default=[0.05, 0.07])
    buckets.add_argument("--test-year", type=int, default=2025)
    buckets.add_argument("--train-start-year", type=int, default=None)
    buckets.add_argument("--train-end-year", type=int, default=2024)
    buckets.set_defaults(func=cmd_train_buckets)

    wf_buckets = sub.add_parser("walk-forward-buckets")
    wf_buckets.add_argument("--thresholds", type=float, nargs="+", default=[0.05, 0.07])
    wf_buckets.add_argument("--start-test-year", type=int, default=2017)
    wf_buckets.add_argument("--end-test-year", type=int, default=2025)
    wf_buckets.add_argument("--train-start-year", type=int, default=2012)
    wf_buckets.set_defaults(func=cmd_walk_forward_buckets)

    train_final = sub.add_parser("train-final")
    train_final.add_argument("--thresholds", type=float, nargs="+", default=[0.05, 0.07])
    train_final.add_argument("--train-start-year", type=int, default=2012)
    train_final.add_argument("--train-end-year", type=int, default=2024)
    train_final.set_defaults(func=cmd_train_final)

    predict = sub.add_parser("predict-year")
    predict.add_argument("--thresholds", type=float, nargs="+", default=[0.05, 0.07])
    predict.add_argument("--test-year", type=int, default=2025)
    predict.add_argument("--train-end-year", type=int, default=2024)
    predict.set_defaults(func=cmd_predict_year)

    backtest = sub.add_parser("backtest")
    backtest.add_argument("--predictions", required=True)
    backtest.add_argument("--top-n", type=int, default=5)
    backtest.add_argument("--cost-bps", type=float, default=20.0)
    backtest.add_argument("--name", default="backtest")
    backtest.set_defaults(func=cmd_backtest)

    portfolio = sub.add_parser("portfolio")
    portfolio.add_argument("--predictions", required=True)
    portfolio.add_argument("--daily-long-slots", type=int, default=5)
    portfolio.add_argument("--daily-short-slots", type=int, default=5)
    portfolio.add_argument("--max-open-long", type=int, default=25)
    portfolio.add_argument("--max-open-short", type=int, default=25)
    portfolio.add_argument("--allocation-pct", type=float, default=0.02)
    portfolio.add_argument("--cost-bps", type=float, default=20.0)
    portfolio.add_argument("--min-score", type=float, default=0.0)
    portfolio.add_argument("--max-adverse-move", type=float, default=None)
    portfolio.add_argument("--score-proportional-sizing", action="store_true")
    portfolio.add_argument("--score-size-floor", type=float, default=0.65)
    portfolio.add_argument("--max-allocation-pct", type=float, default=0.20)
    portfolio.add_argument("--use-atr-stop", action="store_true")
    portfolio.add_argument("--atr-stop-mult", type=float, default=1.5)
    portfolio.add_argument("--sector-concentration-max", type=int, default=99)
    portfolio.add_argument("--use-regime-gate", action="store_true")
    portfolio.add_argument("--min-rr-ratio", type=float, default=0.0)
    portfolio.add_argument("--rolling-derisk", action="store_true")
    portfolio.add_argument("--derisk-window", type=int, default=20)
    portfolio.add_argument("--derisk-warn-avg-pnl", type=float, default=0.0)
    portfolio.add_argument("--derisk-crit-avg-pnl", type=float, default=-0.01)
    portfolio.add_argument("--name", default="portfolio")
    portfolio.set_defaults(func=cmd_portfolio)

    sweep = sub.add_parser("sweep")
    sweep.add_argument("--predictions", required=True)
    sweep.add_argument("--daily-slots", type=int, nargs="+", default=[1, 3, 5, 10])
    sweep.add_argument("--min-scores", type=float, nargs="+", default=[0.0, 0.5, 0.55, 0.6])
    sweep.add_argument("--max-adverse-moves", type=float, nargs="+", default=[None])
    sweep.add_argument("--cost-bps", type=float, default=20.0)
    sweep.add_argument("--name", default="portfolio_sweep")
    sweep.set_defaults(func=cmd_sweep)

    diagnostics = sub.add_parser("diagnostics")
    diagnostics.add_argument("--predictions", required=True)
    diagnostics.add_argument("--top-k", type=int, default=100)
    diagnostics.set_defaults(func=cmd_diagnostics)

    eval_swing = sub.add_parser("eval-swing")
    eval_swing.add_argument("--predictions", required=True, help="Prediction file or directory")
    eval_swing.add_argument("--dataset", default=None, help="Feature dataset with OHLC history")
    eval_swing.add_argument(
        "--signal-set",
        choices=["visible", "baseline", "strict", "production", "all"],
        default="visible",
        help="Rows to include in summary metrics",
    )
    eval_swing.add_argument("--near-fraction", type=float, default=0.80)
    eval_swing.add_argument("--cost-bps", type=float, default=20.0)
    eval_swing.add_argument("--min-hit-near", type=float, default=0.60)
    eval_swing.add_argument("--preferred-opposite-cap", type=float, default=0.20)
    eval_swing.add_argument("--hard-opposite-cap", type=float, default=0.25)
    eval_swing.add_argument("--name", default=None)
    eval_swing.add_argument("--output-dir", default=None)
    eval_swing.set_defaults(func=cmd_eval_swing)

    train_catboost = sub.add_parser("train-catboost")
    train_catboost.add_argument("--thresholds", type=float, nargs="+", default=[0.05, 0.07])
    train_catboost.add_argument("--train-start-year", type=int, default=2012)
    train_catboost.add_argument("--train-end-year", type=int, default=2024)
    train_catboost.set_defaults(func=cmd_train_catboost)

    predict_catboost = sub.add_parser("predict-catboost")
    predict_catboost.add_argument("--thresholds", type=float, nargs="+", default=[0.05, 0.07])
    predict_catboost.add_argument("--test-year", type=int, default=2025)
    predict_catboost.add_argument("--train-end-year", type=int, default=2024)
    predict_catboost.set_defaults(func=cmd_predict_catboost)

    ensemble = sub.add_parser("ensemble")
    ensemble.add_argument("--lgbm-predictions", required=True)
    ensemble.add_argument("--catboost-predictions", required=True)
    ensemble.add_argument("--lgbm-weight", type=float, default=0.55)
    ensemble.add_argument("--test-year", type=int, default=2025)
    ensemble.set_defaults(func=cmd_ensemble)

    experiment_quarters = sub.add_parser("experiment-quarters")
    experiment_quarters.add_argument("--thresholds", type=float, nargs="+", default=[0.05, 0.07])
    experiment_quarters.add_argument("--train-start-year", type=int, default=2012)
    experiment_quarters.add_argument("--train-end-year", type=int, default=2024)
    experiment_quarters.add_argument("--run-name", default=None)
    experiment_quarters.set_defaults(func=cmd_experiment_quarters)

    evaluate_quarters = sub.add_parser("evaluate-quarters")
    evaluate_quarters.add_argument("--predictions", required=True)
    evaluate_quarters.add_argument("--model-family", required=True, choices=["lgbm", "catboost", "ensemble"])
    evaluate_quarters.add_argument("--year", type=int, default=2025)
    evaluate_quarters.add_argument("--run-name", default=None)
    evaluate_quarters.set_defaults(func=cmd_evaluate_quarters)

    monthly = sub.add_parser("monthly-retrain")
    monthly.add_argument("--thresholds", type=float, nargs="+", default=[0.05, 0.07])
    monthly.add_argument("--train-start-year", type=int, default=2012)
    monthly.add_argument("--first-prediction-month", default="2025-01")
    monthly.add_argument("--last-prediction-month", default="2025-12")
    monthly.add_argument("--feature-profile", choices=["all", "side_curated", "side_compact"], default="all")
    monthly.add_argument("--sides", nargs="+", choices=["up", "down"], default=["up", "down"])
    monthly.add_argument("--validation-days", type=int, default=365)
    monthly.add_argument("--train-cutoff-day", type=int, default=20)
    monthly.add_argument("--run-name", default=None)
    monthly.set_defaults(func=cmd_monthly_retrain)

    prod_train = sub.add_parser("prod-train")
    prod_train.add_argument("--prediction-month", required=True, help="YYYY-MM month to predict")
    prod_train.add_argument("--thresholds", type=float, nargs="+", default=[0.05, 0.07])
    prod_train.add_argument("--sides", nargs="+", choices=["up", "down"], default=["up"])
    prod_train.add_argument("--train-start-year", type=int, default=2012)
    prod_train.add_argument("--train-cutoff-day", type=int, default=20)
    prod_train.add_argument("--feature-profile", choices=["all", "side_curated", "side_compact"], default="side_compact")
    prod_train.add_argument("--min-score", type=float, default=0.65)
    prod_train.add_argument("--max-adverse-move", type=float, default=None)
    prod_train.add_argument("--cost-bps", type=float, default=20.0)
    prod_train.add_argument("--run-name", default=None)
    prod_train.set_defaults(func=cmd_prod_train)

    prod_predict = sub.add_parser("prod-predict")
    prod_predict.add_argument("--as-of-date", required=True, help="YYYY-MM-DD signal date")
    prod_predict.set_defaults(func=cmd_prod_predict)

    prod_auto_parser = sub.add_parser("prod-auto")
    prod_auto_parser.add_argument("--as-of-date", required=True, help="YYYY-MM-DD signal date")
    prod_auto_parser.add_argument("--thresholds", type=float, nargs="+", default=[0.05, 0.07])
    prod_auto_parser.add_argument("--sides", nargs="+", choices=["up", "down"], default=["up"])
    prod_auto_parser.add_argument("--train-start-year", type=int, default=2012)
    prod_auto_parser.add_argument("--train-cutoff-day", type=int, default=20)
    prod_auto_parser.add_argument("--feature-profile", choices=["all", "side_curated", "side_compact"], default="side_compact")
    prod_auto_parser.add_argument("--min-score", type=float, default=0.65)
    prod_auto_parser.add_argument("--cost-bps", type=float, default=20.0)
    prod_auto_parser.set_defaults(func=cmd_prod_auto)

    quality = sub.add_parser("quality-long")
    quality.add_argument("--thresholds", type=float, nargs="+", default=[0.05, 0.07])
    quality.add_argument("--train-start-year", type=int, default=2012)
    quality.add_argument("--first-prediction-month", default="2025-01")
    quality.add_argument("--last-prediction-month", default="2025-12")
    quality.add_argument("--feature-profile", choices=["all", "side_curated", "side_compact"], default="side_compact")
    quality.add_argument("--train-cutoff-day", type=int, default=20)
    quality.add_argument("--validation-days", type=int, default=365)
    quality.add_argument("--adverse-limit", type=float, default=0.02)
    quality.add_argument("--annual-signal-target", type=int, default=1000)
    quality.add_argument("--run-name", default=None)
    quality.set_defaults(func=cmd_quality_long)

    wf_quality = sub.add_parser("walkforward-quality")
    wf_quality.add_argument("--thresholds", type=float, nargs="+", default=[0.05, 0.07])
    wf_quality.add_argument("--train-start-year", type=int, default=2012)
    wf_quality.add_argument("--start-test-year", type=int, default=2018)
    wf_quality.add_argument("--end-test-year", type=int, default=2025)
    wf_quality.add_argument("--feature-profile", choices=["all", "side_curated", "side_compact"], default="side_compact")
    wf_quality.add_argument("--validation-days", type=int, default=365)
    wf_quality.add_argument("--train-cutoff-month", type=int, default=12)
    wf_quality.add_argument("--train-cutoff-day", type=int, default=20)
    wf_quality.add_argument("--bad-rate-cap", type=float, default=0.15)
    wf_quality.add_argument("--topn-step", type=int, default=5)
    wf_quality.add_argument("--topn-max", type=int, default=500)
    wf_quality.add_argument("--calibration-lookback-years", type=int, default=None)
    wf_quality.add_argument("--run-name", default=None)
    wf_quality.set_defaults(func=cmd_walkforward_quality)

    segmented_quality = sub.add_parser("segmented-quality")
    segmented_quality.add_argument("--train-start-year", type=int, default=2012)
    segmented_quality.add_argument("--start-test-year", type=int, default=2018)
    segmented_quality.add_argument("--end-test-year", type=int, default=2025)
    segmented_quality.add_argument("--feature-profile", choices=["all", "side_curated", "side_compact"], default="side_compact")
    segmented_quality.add_argument("--validation-days", type=int, default=365)
    segmented_quality.add_argument("--train-cutoff-month", type=int, default=12)
    segmented_quality.add_argument("--train-cutoff-day", type=int, default=20)
    segmented_quality.add_argument("--bad-rate-cap", type=float, default=0.15)
    segmented_quality.add_argument("--topn-step", type=int, default=5)
    segmented_quality.add_argument("--topn-max", type=int, default=500)
    segmented_quality.add_argument("--calibration-start-year", type=int, default=None)
    segmented_quality.add_argument("--bad-negative-weight", type=float, default=4.0)
    segmented_quality.add_argument("--soft-negative-weight", type=float, default=0.35)
    segmented_quality.add_argument("--objective", choices=["good_call", "target_hit"], default="good_call")
    segmented_quality.add_argument("--run-name", default=None)
    segmented_quality.set_defaults(func=cmd_segmented_quality)

    top30_quality = sub.add_parser("top30-trade-quality")
    top30_quality.add_argument("--train-start-year", type=int, default=2012)
    top30_quality.add_argument("--start-test-year", type=int, default=2018)
    top30_quality.add_argument("--end-test-year", type=int, default=2025)
    top30_quality.add_argument("--target-pct", type=float, default=0.04)
    top30_quality.add_argument("--stop-pct", type=float, default=0.03)
    top30_quality.add_argument("--max-hold-days", type=int, default=5)
    top30_quality.add_argument("--validation-days", type=int, default=365)
    top30_quality.add_argument("--train-cutoff-month", type=int, default=12)
    top30_quality.add_argument("--train-cutoff-day", type=int, default=20)
    top30_quality.add_argument("--min-trades", type=int, default=100)
    top30_quality.add_argument("--run-name", default=None)
    top30_quality.set_defaults(func=cmd_top30_trade_quality)

    clean_direction = sub.add_parser("clean-direction")
    clean_direction.add_argument("--train-start-year", type=int, default=2012)
    clean_direction.add_argument("--start-test-year", type=int, default=2018)
    clean_direction.add_argument("--end-test-year", type=int, default=2025)
    clean_direction.add_argument("--target-pct", type=float, default=0.04)
    clean_direction.add_argument("--adverse-limit", type=float, default=0.0091)
    clean_direction.add_argument("--validation-days", type=int, default=365)
    clean_direction.add_argument("--train-cutoff-month", type=int, default=12)
    clean_direction.add_argument("--train-cutoff-day", type=int, default=20)
    clean_direction.add_argument("--precision-floor", type=float, default=0.90)
    clean_direction.add_argument("--min-validation-calls", type=int, default=20)
    clean_direction.add_argument("--train-universe", choices=["all", "liquid30"], default="all")
    clean_direction.add_argument("--liquid-weight", type=float, default=2.0)
    clean_direction.add_argument("--max-weekly-abs-move", type=float, default=0.50)
    clean_direction.add_argument("--episode-decay-weight", type=float, default=0.35)
    clean_direction.add_argument("--ensemble-lgbm-weight", type=float, default=0.55)
    clean_direction.add_argument("--no-catboost", action="store_true")
    clean_direction.add_argument("--direct-side-weight", type=float, default=0.50)
    clean_direction.add_argument("--run-name", default=None)
    clean_direction.set_defaults(func=cmd_clean_direction)

    tiered_clean = sub.add_parser("tiered-clean-direction")
    tiered_clean.add_argument("--train-start-year", type=int, default=2012)
    tiered_clean.add_argument("--start-test-year", type=int, default=2018)
    tiered_clean.add_argument("--end-test-year", type=int, default=2025)
    tiered_clean.add_argument("--validation-days", type=int, default=365)
    tiered_clean.add_argument("--train-cutoff-month", type=int, default=12)
    tiered_clean.add_argument("--train-cutoff-day", type=int, default=20)
    tiered_clean.add_argument("--adverse-limit", type=float, default=0.0091)
    tiered_clean.add_argument("--max-weekly-abs-move", type=float, default=0.50)
    tiered_clean.add_argument("--min-validation-calls", type=int, default=20)
    tiered_clean.add_argument("--bad-rate-cap", type=float, default=0.15)
    tiered_clean.add_argument("--topn-step", type=int, default=5)
    tiered_clean.add_argument("--topn-max", type=int, default=300)
    tiered_clean.add_argument("--lgbm-weight", type=float, default=0.60)
    tiered_clean.add_argument("--no-catboost", action="store_true")
    tiered_clean.add_argument("--tier-only-train", action="store_true")
    tiered_clean.add_argument("--use-vol-adjusted-labels", action="store_true")
    tiered_clean.add_argument("--temporal-decay-per-year", type=float, default=0.0)
    tiered_clean.add_argument("--n-seeds", type=int, default=1)
    tiered_clean.add_argument("--use-calibration", action="store_true")
    tiered_clean.add_argument("--run-name", default=None)
    tiered_clean.set_defaults(func=cmd_tiered_clean_direction)

    tiered_prod_train = sub.add_parser("tiered-prod-train")
    tiered_prod_train.add_argument("--prediction-month", required=True, help="YYYY-MM month to predict")
    tiered_prod_train.add_argument("--train-start-year", type=int, default=2012)
    tiered_prod_train.add_argument("--train-cutoff-day", type=int, default=20)
    tiered_prod_train.add_argument("--adverse-limit", type=float, default=0.0091)
    tiered_prod_train.add_argument("--max-weekly-abs-move", type=float, default=0.50)
    tiered_prod_train.add_argument("--lgbm-weight", type=float, default=0.60)
    tiered_prod_train.add_argument("--no-catboost", action="store_true")
    tiered_prod_train.add_argument("--tier-only-train", action="store_true")
    tiered_prod_train.add_argument("--min-score", type=float, default=0.0)
    tiered_prod_train.add_argument("--use-vol-adjusted-labels", action="store_true")
    tiered_prod_train.add_argument("--temporal-decay-per-year", type=float, default=0.0)
    tiered_prod_train.add_argument("--n-seeds", type=int, default=1)
    tiered_prod_train.add_argument("--use-calibration", action="store_true")
    tiered_prod_train.add_argument("--run-name", default=None)
    tiered_prod_train.set_defaults(func=cmd_tiered_prod_train)

    tiered_prod_predict = sub.add_parser("tiered-prod-predict")
    tiered_prod_predict.add_argument("--as-of-date", required=True, help="YYYY-MM-DD signal date")
    tiered_prod_predict.set_defaults(func=cmd_tiered_prod_predict)

    tiered_prod_auto_parser = sub.add_parser("tiered-prod-auto")
    tiered_prod_auto_parser.add_argument("--as-of-date", required=True, help="YYYY-MM-DD signal date")
    tiered_prod_auto_parser.add_argument("--train-start-year", type=int, default=2012)
    tiered_prod_auto_parser.add_argument("--train-cutoff-day", type=int, default=20)
    tiered_prod_auto_parser.add_argument("--adverse-limit", type=float, default=0.0091)
    tiered_prod_auto_parser.add_argument("--max-weekly-abs-move", type=float, default=0.50)
    tiered_prod_auto_parser.add_argument("--lgbm-weight", type=float, default=0.60)
    tiered_prod_auto_parser.add_argument("--no-catboost", action="store_true")
    tiered_prod_auto_parser.add_argument("--tier-only-train", action="store_true")
    tiered_prod_auto_parser.add_argument("--min-score", type=float, default=0.0)
    tiered_prod_auto_parser.set_defaults(func=cmd_tiered_prod_auto)

    prod_full = sub.add_parser("prod-full-pipeline")
    prod_full.add_argument("--start-date", required=True, help="YYYY-MM-DD first signal date")
    prod_full.add_argument("--end-date", required=True, help="YYYY-MM-DD last signal date")
    prod_full.add_argument("--source", choices=["silver", "raw"], default="silver")
    prod_full.add_argument(
        "--fetch-mode",
        choices=["missing", "always", "skip"],
        default="missing",
        help="missing fetches only when silver does not cover end date",
    )
    prod_full.add_argument("--esn-root", default=r"C:\Users\rahul\Koscine 3.0")
    prod_full.add_argument("--train-start-year", type=int, default=2012)
    prod_full.add_argument("--train-cutoff-day", type=int, default=20)
    prod_full.add_argument("--adverse-limit", type=float, default=0.0091)
    prod_full.add_argument("--max-weekly-abs-move", type=float, default=0.50)
    prod_full.add_argument("--lgbm-weight", type=float, default=0.60)
    prod_full.add_argument("--no-catboost", action="store_true")
    prod_full.add_argument("--tier-only-train", action="store_true")
    prod_full.add_argument("--min-score", type=float, default=0.0)
    prod_full.set_defaults(func=cmd_prod_full_pipeline)

    expansion_train = sub.add_parser("expansion-train")
    expansion_train.add_argument("--prediction-month", required=True, help="YYYY-MM month to predict")
    expansion_train.add_argument("--train-start-year", type=int, default=2012)
    expansion_train.add_argument("--train-cutoff-day", type=int, default=20)
    expansion_train.add_argument("--validation-days", type=int, default=365)
    expansion_train.add_argument("--lgbm-weight", type=float, default=0.60)
    expansion_train.add_argument("--no-catboost", action="store_true")
    expansion_train.add_argument("--no-calibration", action="store_true")
    expansion_train.add_argument("--n-seeds", type=int, default=2)
    expansion_train.add_argument("--temporal-decay-per-year", type=float, default=0.05)
    expansion_train.add_argument("--tier-only-train", action="store_true")
    expansion_train.add_argument("--run-name", default=None)
    expansion_train.set_defaults(func=cmd_expansion_train)

    expansion_predict = sub.add_parser("expansion-predict")
    expansion_predict.add_argument("--start", required=True, help="YYYY-MM-DD first signal date")
    expansion_predict.add_argument("--end", required=True, help="YYYY-MM-DD last signal date")
    expansion_predict.set_defaults(func=cmd_expansion_predict)

    expansion_wf = sub.add_parser("expansion-walkforward")
    expansion_wf.add_argument("--train-start-year", type=int, default=2012)
    expansion_wf.add_argument("--start-test-year", type=int, default=2022)
    expansion_wf.add_argument("--end-test-year", type=int, default=2025)
    expansion_wf.add_argument("--validation-days", type=int, default=365)
    expansion_wf.add_argument("--lgbm-weight", type=float, default=0.60)
    expansion_wf.add_argument("--no-catboost", action="store_true")
    expansion_wf.add_argument("--no-calibration", action="store_true")
    expansion_wf.add_argument("--n-seeds", type=int, default=2)
    expansion_wf.add_argument("--temporal-decay-per-year", type=float, default=0.05)
    expansion_wf.add_argument("--tier-only-train", action="store_true")
    expansion_wf.add_argument("--run-name", default=None)
    expansion_wf.set_defaults(func=cmd_expansion_walkforward)

    signal_cards = sub.add_parser("signal-cards")
    signal_cards.add_argument("--predictions", required=True)
    signal_cards.add_argument("--output-dir", default=None)
    signal_cards.add_argument("--atr-stop-mult", type=float, default=1.5)
    signal_cards.add_argument("--min-rr-ratio", type=float, default=1.4)
    signal_cards.add_argument("--base-allocation-pct", type=float, default=0.10)
    signal_cards.add_argument("--max-allocation-pct", type=float, default=0.20)
    signal_cards.add_argument("--score-threshold-for-scaling", type=float, default=0.65)
    signal_cards.add_argument("--sector-concentration-max", type=int, default=2)
    signal_cards.add_argument("--apply-regime-gate", action="store_true")
    signal_cards.add_argument("--keep-all", action="store_true",
                              help="Include non-actionable rows in output instead of only actionable")
    signal_cards.set_defaults(func=cmd_signal_cards)

    model_health = sub.add_parser("model-health")
    model_health.add_argument("--trades", required=True, help="Parquet of closed trades")
    model_health.add_argument("--output-dir", default=None)
    model_health.add_argument("--rolling-trades", type=int, default=20)
    model_health.add_argument("--warning-hit-rate", type=float, default=0.45)
    model_health.add_argument("--critical-hit-rate", type=float, default=0.35)
    model_health.set_defaults(func=cmd_model_health)

    feature_audit = sub.add_parser("feature-audit")
    feature_audit.add_argument("--min-gain-share", type=float, default=0.001)
    feature_audit.set_defaults(func=cmd_feature_audit)

    calibration_report_p = sub.add_parser("calibration-report")
    calibration_report_p.add_argument("--predictions", required=True)
    calibration_report_p.add_argument("--score-col", default="score")
    calibration_report_p.add_argument("--label-col", default="actual_hit")
    calibration_report_p.add_argument("--output", default="calibration_report.csv")
    calibration_report_p.set_defaults(func=cmd_calibration_report)

    opportunity_train = sub.add_parser("opportunity-train")
    opportunity_train.add_argument("--train-start-year", type=int, default=2010)
    opportunity_train.add_argument("--train-end-year", type=int, default=2024)
    opportunity_train.add_argument("--validation-year", type=int, default=2025)
    opportunity_train.add_argument("--train-cutoff-day", type=int, default=20)
    opportunity_train.add_argument("--adverse-limit", type=float, default=0.0091)
    opportunity_train.add_argument("--max-weekly-abs-move", type=float, default=0.50)
    opportunity_train.add_argument("--lgbm-weight", type=float, default=0.60)
    opportunity_train.add_argument("--tier-only-train", action="store_true")
    opportunity_train.add_argument("--go-plus-per-side", type=int, default=2)
    opportunity_train.add_argument("--go-per-side", type=int, default=5)
    opportunity_train.add_argument("--min-direction-score", type=float, default=0.50)
    opportunity_train.add_argument("--min-rest-direction-score", type=float, default=0.50)
    opportunity_train.add_argument("--max-opposite-score", type=float, default=0.70)
    opportunity_train.add_argument("--max-long-opposite-score", type=float, default=0.60)
    opportunity_train.add_argument("--max-short-opposite-score", type=float, default=0.70)
    opportunity_train.add_argument("--min-score-gap", type=float, default=0.03)
    opportunity_train.add_argument("--min-move-power", type=float, default=0.70)
    opportunity_train.add_argument("--go-plus-min-move-power", type=float, default=0.75)
    opportunity_train.add_argument("--min-go-score-quantile", type=float, default=0.0)
    opportunity_train.add_argument("--min-model-validation-top5-capture-rate", type=float, default=0.0)
    opportunity_train.add_argument("--go-plus-annual-target", type=int, default=180)
    opportunity_train.add_argument("--go-annual-target", type=int, default=700)
    opportunity_train.add_argument("--go-plus-quality-threshold", type=float, default=None)
    opportunity_train.add_argument("--go-quality-threshold", type=float, default=None)
    opportunity_train.add_argument("--target-negative-sample-ratio", type=float, default=4.0)
    opportunity_train.add_argument("--target-max-negative-rows", type=int, default=250000)
    opportunity_train.add_argument("--ranker-max-train-rows", type=int, default=350000)
    opportunity_train.add_argument("--rest35-long-addon-min-score", type=float, default=0.60)
    opportunity_train.add_argument("--rest35-long-addon-min-score-gap", type=float, default=0.03)
    opportunity_train.add_argument("--rest35-long-addon-min-move-power", type=float, default=0.80)
    opportunity_train.add_argument("--rest35-long-addon-min-bb-rank", type=float, default=0.80)
    opportunity_train.add_argument("--rest35-long-addon-min-atm-iv", type=float, default=0.40)
    opportunity_train.add_argument("--rest35-long-addon-min-path-range", type=float, default=0.10)
    opportunity_train.add_argument("--rest35-long-addon-min-atr", type=float, default=0.035)
    opportunity_train.add_argument("--liquid-long-addon-min-score", type=float, default=0.80)
    opportunity_train.add_argument("--liquid-long-addon-min-score-gap", type=float, default=0.05)
    opportunity_train.add_argument("--liquid-long-addon-min-move-power", type=float, default=0.90)
    opportunity_train.add_argument("--liquid-long-addon-max-side-rank", type=int, default=2)
    opportunity_train.add_argument("--liquid-long-addon-min-path-range", type=float, default=0.08)
    opportunity_train.add_argument("--run-name", default=None)
    opportunity_train.set_defaults(func=cmd_opportunity_train)

    opportunity_apply = sub.add_parser("opportunity-apply")
    opportunity_apply.add_argument("--start-date", default="2026-01-01")
    opportunity_apply.add_argument("--end-date", default="2026-05-31")
    opportunity_apply.add_argument("--output-name", default="opportunity_jan_may_2026")
    opportunity_apply.set_defaults(func=cmd_opportunity_apply)

    opportunity_pipeline = sub.add_parser("opportunity-pipeline")
    opportunity_pipeline.add_argument("--train-start-year", type=int, default=2010)
    opportunity_pipeline.add_argument("--train-end-year", type=int, default=2024)
    opportunity_pipeline.add_argument("--validation-year", type=int, default=2025)
    opportunity_pipeline.add_argument("--train-cutoff-day", type=int, default=20)
    opportunity_pipeline.add_argument("--adverse-limit", type=float, default=0.0091)
    opportunity_pipeline.add_argument("--max-weekly-abs-move", type=float, default=0.50)
    opportunity_pipeline.add_argument("--lgbm-weight", type=float, default=0.60)
    opportunity_pipeline.add_argument("--tier-only-train", action="store_true")
    opportunity_pipeline.add_argument("--go-plus-per-side", type=int, default=2)
    opportunity_pipeline.add_argument("--go-per-side", type=int, default=5)
    opportunity_pipeline.add_argument("--min-direction-score", type=float, default=0.50)
    opportunity_pipeline.add_argument("--min-rest-direction-score", type=float, default=0.50)
    opportunity_pipeline.add_argument("--max-opposite-score", type=float, default=0.70)
    opportunity_pipeline.add_argument("--max-long-opposite-score", type=float, default=0.60)
    opportunity_pipeline.add_argument("--max-short-opposite-score", type=float, default=0.70)
    opportunity_pipeline.add_argument("--min-score-gap", type=float, default=0.03)
    opportunity_pipeline.add_argument("--min-move-power", type=float, default=0.70)
    opportunity_pipeline.add_argument("--go-plus-min-move-power", type=float, default=0.75)
    opportunity_pipeline.add_argument("--min-go-score-quantile", type=float, default=0.0)
    opportunity_pipeline.add_argument("--min-model-validation-top5-capture-rate", type=float, default=0.0)
    opportunity_pipeline.add_argument("--go-plus-annual-target", type=int, default=180)
    opportunity_pipeline.add_argument("--go-annual-target", type=int, default=700)
    opportunity_pipeline.add_argument("--go-plus-quality-threshold", type=float, default=None)
    opportunity_pipeline.add_argument("--go-quality-threshold", type=float, default=None)
    opportunity_pipeline.add_argument("--target-negative-sample-ratio", type=float, default=4.0)
    opportunity_pipeline.add_argument("--target-max-negative-rows", type=int, default=250000)
    opportunity_pipeline.add_argument("--ranker-max-train-rows", type=int, default=350000)
    opportunity_pipeline.add_argument("--rest35-long-addon-min-score", type=float, default=0.60)
    opportunity_pipeline.add_argument("--rest35-long-addon-min-score-gap", type=float, default=0.03)
    opportunity_pipeline.add_argument("--rest35-long-addon-min-move-power", type=float, default=0.80)
    opportunity_pipeline.add_argument("--rest35-long-addon-min-bb-rank", type=float, default=0.80)
    opportunity_pipeline.add_argument("--rest35-long-addon-min-atm-iv", type=float, default=0.40)
    opportunity_pipeline.add_argument("--rest35-long-addon-min-path-range", type=float, default=0.10)
    opportunity_pipeline.add_argument("--rest35-long-addon-min-atr", type=float, default=0.035)
    opportunity_pipeline.add_argument("--liquid-long-addon-min-score", type=float, default=0.80)
    opportunity_pipeline.add_argument("--liquid-long-addon-min-score-gap", type=float, default=0.05)
    opportunity_pipeline.add_argument("--liquid-long-addon-min-move-power", type=float, default=0.90)
    opportunity_pipeline.add_argument("--liquid-long-addon-max-side-rank", type=int, default=2)
    opportunity_pipeline.add_argument("--liquid-long-addon-min-path-range", type=float, default=0.08)
    opportunity_pipeline.add_argument("--start-date", default="2026-01-01")
    opportunity_pipeline.add_argument("--end-date", default="2026-05-31")
    opportunity_pipeline.add_argument("--run-name", default=None)
    opportunity_pipeline.set_defaults(func=cmd_opportunity_pipeline)

    run_all = sub.add_parser("run-all")
    run_all.add_argument("--source", choices=["silver", "raw"], default="silver")
    run_all.add_argument("--thresholds", type=float, nargs="+", default=[0.05, 0.07])
    run_all.add_argument("--train-start-year", type=int, default=2012)
    run_all.add_argument("--train-end-year", type=int, default=2024)
    run_all.add_argument("--test-year", type=int, default=2025)
    run_all.add_argument("--top-n", type=int, default=5)
    run_all.add_argument("--cost-bps", type=float, default=20.0)
    run_all.add_argument("--daily-long-slots", type=int, default=5)
    run_all.add_argument("--daily-short-slots", type=int, default=5)
    run_all.add_argument("--max-open-long", type=int, default=25)
    run_all.add_argument("--max-open-short", type=int, default=25)
    run_all.add_argument("--allocation-pct", type=float, default=0.02)
    run_all.add_argument("--min-score", type=float, default=0.0)
    run_all.add_argument("--max-adverse-move", type=float, default=None)
    run_all.add_argument("--sweep-daily-slots", type=int, nargs="+", default=[1, 3, 5, 10])
    run_all.add_argument("--sweep-min-scores", type=float, nargs="+", default=[0.0, 0.5, 0.55, 0.6])
    run_all.set_defaults(func=cmd_run_all)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
