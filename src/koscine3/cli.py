from __future__ import annotations

import argparse
import json
from pathlib import Path

from koscine3.data.feature_registry import build_feature_registry, write_feature_audit
from koscine3.data.sources import inspect_source, load_market_data
from koscine3.data.universe import UniverseConfig, build_universe, write_universe_manifest
from koscine3.experiments.run_experiment import ExperimentConfig, run_experiment
from koscine3.experiments.large_move_ranker import (
    LargeMoveRankerConfig,
    run_large_move_ranker,
)
from koscine3.experiments.strict_hit_model import (
    StrictHitModelConfig,
    run_strict_hit_model,
)
from koscine3.experiments.routed_specialist_model import (
    RoutedSpecialistConfig,
    run_routed_specialist_model,
)
from koscine3.experiments.trajectory_strict_model import (
    TrajectoryStrictConfig,
    run_trajectory_strict_model,
)


def _inspect_data(args: argparse.Namespace) -> None:
    print(json.dumps(inspect_source(), indent=2))


def _audit_features(args: argparse.Namespace) -> None:
    df = load_market_data()
    registry = build_feature_registry(df)
    output = Path(args.output)
    write_feature_audit(registry, output)
    print(f"Wrote feature audit: {output}")
    print(f"safe_features={len(registry.feature_columns)} blocked={len(registry.blocked_columns)}")


def _build_universe(args: argparse.Namespace) -> None:
    columns = [
        "date",
        "symbol",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "turnover_lacs",
        "fut_close",
    ]
    df = load_market_data(columns=columns)
    config = UniverseConfig(cutoff_date=args.cutoff)
    universe = build_universe(df, config)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    universe.to_csv(output, index=False)
    write_universe_manifest(universe, output.with_suffix(".manifest.json"), config)
    print(f"Wrote universe: {output}")


def _run_experiment(args: argparse.Namespace) -> None:
    config = ExperimentConfig(
        run_id=args.run_id,
        universe_cutoff=args.universe_cutoff,
        prediction_start=args.prediction_start,
        prediction_end=args.prediction_end,
        base_train_end=args.base_train_end,
        calibration_start=args.calibration_start,
        calibration_end=args.calibration_end,
        n_estimators=args.n_estimators,
        include_baselines=not args.no_baselines,
        smoke=args.smoke,
        tune_selector=args.tune_selector and not args.no_tune_selector,
    )
    run_dir = run_experiment(config)
    print(f"Wrote run: {run_dir}")


def _run_large_move_ranker(args: argparse.Namespace) -> None:
    config = LargeMoveRankerConfig(
        run_id=args.run_id,
        source_run_id=args.source_run_id,
        train_start=args.train_start,
        n_estimators=args.n_estimators,
        weekly_target=args.weekly_target,
        max_signals_per_day=args.max_signals_per_day,
        max_signals_per_week_side=args.max_signals_per_week_side,
        max_symbol_per_month=args.max_symbol_per_month,
        daily_pool_rank=args.daily_pool_rank,
        selector_mode=args.selector_mode,
        setup_portfolio=args.setup_portfolio,
        setup_weekly_quota=args.setup_weekly_quota,
        rank_score_weight=args.rank_score_weight,
        clean_large_weight=args.clean_large_weight,
        hit_near_weight=args.hit_near_weight,
        floor_weight=args.floor_weight,
        opposite_penalty=args.opposite_penalty,
        expected_move_weight=args.expected_move_weight,
        historical_risk_penalty=args.historical_risk_penalty,
        year_symbol_penalty=args.year_symbol_penalty,
        global_symbol_penalty=args.global_symbol_penalty,
    )
    run_dir = run_large_move_ranker(config)
    print(f"Wrote large-move ranker run: {run_dir}")


def _run_strict_hit_model(args: argparse.Namespace) -> None:
    config = StrictHitModelConfig(
        run_id=args.run_id,
        source_run_id=args.source_run_id,
        train_start=args.train_start,
        n_estimators=args.n_estimators,
        weekly_target=args.weekly_target,
        max_signals_per_day=args.max_signals_per_day,
        max_signals_per_week_side=args.max_signals_per_week_side,
        max_symbol_per_month=args.max_symbol_per_month,
        daily_pool_rank=args.daily_pool_rank,
        min_pair_hit_probability=args.min_pair_hit_probability,
        min_full_hit_probability=args.min_full_hit_probability,
        max_opposite_probability=args.max_opposite_probability,
        max_range_probability=args.max_range_probability,
        min_strict_edge=args.min_strict_edge,
        pair_weight=args.pair_weight,
        full_hit_weight=args.full_hit_weight,
        opposite_penalty=args.opposite_penalty,
        range_penalty=args.range_penalty,
        favorable_move_weight=args.favorable_move_weight,
        signed_close_weight=args.signed_close_weight,
    )
    run_dir = run_strict_hit_model(config)
    print(f"Wrote strict-hit model run: {run_dir}")


def _run_routed_specialist_model(args: argparse.Namespace) -> None:
    config = RoutedSpecialistConfig(
        run_id=args.run_id,
        train_start=args.train_start,
        universe_cutoff=args.universe_cutoff,
        prediction_top_n=args.prediction_top_n,
        training_top_n=args.training_top_n,
        n_estimators=args.n_estimators,
        weekly_target=args.weekly_target,
        max_signals_per_day=args.max_signals_per_day,
        daily_pool_rank=args.daily_pool_rank,
        primary_side_only=not args.allow_non_primary_sides,
        min_pair_hit_probability=args.min_pair_hit_probability,
        min_full_hit_probability=args.min_full_hit_probability,
        max_opposite_probability=args.max_opposite_probability,
        max_range_probability=args.max_range_probability,
        min_route_utility=args.min_route_utility,
        pair_weight=args.pair_weight,
        full_hit_weight=args.full_hit_weight,
        opposite_penalty=args.opposite_penalty,
        range_penalty=args.range_penalty,
    )
    run_dir = run_routed_specialist_model(config)
    print(f"Wrote routed-specialist run: {run_dir}")


def _run_trajectory_strict_model(args: argparse.Namespace) -> None:
    config = TrajectoryStrictConfig(
        run_id=args.run_id,
        train_start=args.train_start,
        universe_cutoff=args.universe_cutoff,
        prediction_top_n=args.prediction_top_n,
        training_top_n=args.training_top_n,
        n_estimators=args.n_estimators,
        matched_opposites_per_hit=args.matched_opposites_per_hit,
        weekly_target=args.weekly_target,
        max_signals_per_day=args.max_signals_per_day,
        daily_pool_rank=args.daily_pool_rank,
        primary_side_only=not args.allow_non_primary_sides,
        min_pair_hit_probability=args.min_pair_hit_probability,
        min_full_hit_probability=args.min_full_hit_probability,
        max_opposite_probability=args.max_opposite_probability,
        max_range_probability=args.max_range_probability,
        min_trajectory_edge=args.min_trajectory_edge,
        min_trajectory_utility=args.min_trajectory_utility,
        pair_weight=args.pair_weight,
        full_hit_weight=args.full_hit_weight,
        opposite_penalty=args.opposite_penalty,
        range_penalty=args.range_penalty,
        setup_score_weight=args.setup_score_weight,
        resume_completed_splits=not args.no_resume,
    )
    run_dir = run_trajectory_strict_model(config)
    print(f"Wrote trajectory strict-hit run: {run_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="koscine3")
    sub = parser.add_subparsers(dest="command", required=True)

    inspect_data = sub.add_parser("inspect-data")
    inspect_data.set_defaults(func=_inspect_data)

    audit = sub.add_parser("audit-features")
    audit.add_argument("--output", default="reports/feature_audit.csv")
    audit.set_defaults(func=_audit_features)

    universe = sub.add_parser("build-universe")
    universe.add_argument("--cutoff", default="2025-12-31")
    universe.add_argument("--output", default="reports/universe.csv")
    universe.set_defaults(func=_build_universe)

    experiment = sub.add_parser("run-experiment")
    experiment.add_argument("--run-id", default="koscine3_v1")
    experiment.add_argument("--universe-cutoff", default="2025-12-31")
    experiment.add_argument("--prediction-start")
    experiment.add_argument("--prediction-end")
    experiment.add_argument("--base-train-end")
    experiment.add_argument("--calibration-start")
    experiment.add_argument("--calibration-end")
    experiment.add_argument("--n-estimators", type=int, default=120)
    experiment.add_argument("--no-baselines", action="store_true")
    experiment.add_argument("--tune-selector", action="store_true")
    experiment.add_argument("--no-tune-selector", action="store_true")
    experiment.add_argument("--smoke", action="store_true")
    experiment.set_defaults(func=_run_experiment)

    ranker = sub.add_parser("run-large-move-ranker")
    ranker.add_argument("--run-id", default="koscine3_large_move_ranker_v1")
    ranker.add_argument("--source-run-id", default="koscine3_crossregime_v19_full_v8_n80")
    ranker.add_argument("--train-start", default="2018-01-01")
    ranker.add_argument("--n-estimators", type=int, default=180)
    ranker.add_argument("--weekly-target", type=int, default=6)
    ranker.add_argument("--max-signals-per-day", type=int, default=2)
    ranker.add_argument("--max-signals-per-week-side", type=int, default=4)
    ranker.add_argument("--max-symbol-per-month", type=int, default=3)
    ranker.add_argument("--daily-pool-rank", type=int, default=18)
    ranker.add_argument("--selector-mode", default="weekly_pool")
    ranker.add_argument("--setup-portfolio", default="")
    ranker.add_argument("--setup-weekly-quota", type=int, default=1)
    ranker.add_argument("--rank-score-weight", type=float, default=1.15)
    ranker.add_argument("--clean-large-weight", type=float, default=0.90)
    ranker.add_argument("--hit-near-weight", type=float, default=0.45)
    ranker.add_argument("--floor-weight", type=float, default=0.25)
    ranker.add_argument("--opposite-penalty", type=float, default=0.95)
    ranker.add_argument("--expected-move-weight", type=float, default=0.18)
    ranker.add_argument("--historical-risk-penalty", type=float, default=0.15)
    ranker.add_argument("--year-symbol-penalty", type=float, default=0.0)
    ranker.add_argument("--global-symbol-penalty", type=float, default=0.0)
    ranker.set_defaults(func=_run_large_move_ranker)

    strict = sub.add_parser("run-strict-hit-model")
    strict.add_argument("--run-id", default="koscine3_strict_hit_v1")
    strict.add_argument("--source-run-id", default="koscine3_crossregime_v19_full_v8_n80")
    strict.add_argument("--train-start", default="2018-01-01")
    strict.add_argument("--n-estimators", type=int, default=180)
    strict.add_argument("--weekly-target", type=int, default=5)
    strict.add_argument("--max-signals-per-day", type=int, default=2)
    strict.add_argument("--max-signals-per-week-side", type=int, default=4)
    strict.add_argument("--max-symbol-per-month", type=int, default=3)
    strict.add_argument("--daily-pool-rank", type=int, default=24)
    strict.add_argument("--min-pair-hit-probability", type=float, default=0.56)
    strict.add_argument("--min-full-hit-probability", type=float, default=0.06)
    strict.add_argument("--max-opposite-probability", type=float, default=0.62)
    strict.add_argument("--max-range-probability", type=float, default=0.90)
    strict.add_argument("--min-strict-edge", type=float, default=-0.05)
    strict.add_argument("--pair-weight", type=float, default=1.40)
    strict.add_argument("--full-hit-weight", type=float, default=1.00)
    strict.add_argument("--opposite-penalty", type=float, default=1.20)
    strict.add_argument("--range-penalty", type=float, default=0.20)
    strict.add_argument("--favorable-move-weight", type=float, default=0.25)
    strict.add_argument("--signed-close-weight", type=float, default=0.35)
    strict.set_defaults(func=_run_strict_hit_model)

    routed = sub.add_parser("run-routed-specialist-model")
    routed.add_argument("--run-id", default="koscine3_routed_specialist_v1")
    routed.add_argument("--train-start", default="2018-01-01")
    routed.add_argument("--universe-cutoff", default="2025-12-31")
    routed.add_argument("--prediction-top-n", type=int, default=100)
    routed.add_argument("--training-top-n", type=int, default=None)
    routed.add_argument("--n-estimators", type=int, default=120)
    routed.add_argument("--weekly-target", type=int, default=6)
    routed.add_argument("--max-signals-per-day", type=int, default=2)
    routed.add_argument("--daily-pool-rank", type=int, default=36)
    routed.add_argument("--allow-non-primary-sides", action="store_true")
    routed.add_argument("--min-pair-hit-probability", type=float, default=0.58)
    routed.add_argument("--min-full-hit-probability", type=float, default=0.18)
    routed.add_argument("--max-opposite-probability", type=float, default=0.70)
    routed.add_argument("--max-range-probability", type=float, default=0.85)
    routed.add_argument("--min-route-utility", type=float, default=0.35)
    routed.add_argument("--pair-weight", type=float, default=1.35)
    routed.add_argument("--full-hit-weight", type=float, default=1.10)
    routed.add_argument("--opposite-penalty", type=float, default=1.10)
    routed.add_argument("--range-penalty", type=float, default=0.25)
    routed.set_defaults(func=_run_routed_specialist_model)

    trajectory = sub.add_parser("run-trajectory-strict-model")
    trajectory.add_argument("--run-id", default="koscine3_trajectory_strict_v1")
    trajectory.add_argument("--train-start", default="2018-01-01")
    trajectory.add_argument("--universe-cutoff", default="2025-12-31")
    trajectory.add_argument("--prediction-top-n", type=int, default=100)
    trajectory.add_argument("--training-top-n", type=int, default=None)
    trajectory.add_argument("--n-estimators", type=int, default=140)
    trajectory.add_argument("--matched-opposites-per-hit", type=int, default=3)
    trajectory.add_argument("--weekly-target", type=int, default=6)
    trajectory.add_argument("--max-signals-per-day", type=int, default=2)
    trajectory.add_argument("--daily-pool-rank", type=int, default=40)
    trajectory.add_argument("--allow-non-primary-sides", action="store_true")
    trajectory.add_argument("--min-pair-hit-probability", type=float, default=0.58)
    trajectory.add_argument("--min-full-hit-probability", type=float, default=0.16)
    trajectory.add_argument("--max-opposite-probability", type=float, default=0.68)
    trajectory.add_argument("--max-range-probability", type=float, default=0.86)
    trajectory.add_argument("--min-trajectory-edge", type=float, default=-0.02)
    trajectory.add_argument("--min-trajectory-utility", type=float, default=0.32)
    trajectory.add_argument("--pair-weight", type=float, default=1.55)
    trajectory.add_argument("--full-hit-weight", type=float, default=1.00)
    trajectory.add_argument("--opposite-penalty", type=float, default=1.30)
    trajectory.add_argument("--range-penalty", type=float, default=0.20)
    trajectory.add_argument("--setup-score-weight", type=float, default=0.22)
    trajectory.add_argument("--no-resume", action="store_true")
    trajectory.set_defaults(func=_run_trajectory_strict_model)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
