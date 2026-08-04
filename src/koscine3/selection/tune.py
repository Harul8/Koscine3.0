from __future__ import annotations

from dataclasses import asdict, replace

import pandas as pd

from koscine3.evaluation.gold_metrics import build_gold_report
from koscine3.selection.daily_selector import SelectorConfig, select_daily_signals


def _candidate_configs(selector_id: str) -> list[SelectorConfig]:
    base = SelectorConfig(selector_id=selector_id)
    candidates = [
        base,
        replace(base, max_signals_per_day=3),
        replace(base, cooldown_trading_days=20),
        replace(base, cooldown_trading_days=30),
        replace(base, long_min_p_hit_near=0.64),
        replace(base, long_min_p_hit_near=0.68),
        replace(base, short_min_p_hit_near=0.60),
        replace(base, short_min_p_hit_near=0.64),
        replace(base, long_max_p_opposite=0.52),
        replace(base, short_max_p_opposite=0.50),
        replace(base, max_symbol_return_20d_rank_pct=0.85),
        replace(base, max_symbol_return_20d_rank_pct=0.95),
        replace(base, long_max_hist_opposite_rate_63=0.40),
        replace(base, long_max_hist_opposite_rate_63=0.50),
        replace(base, max_signals_per_day=3, cooldown_trading_days=30),
        replace(base, max_signals_per_day=5, cooldown_trading_days=30),
        replace(base, long_min_p_hit_near=0.64, short_min_p_hit_near=0.60),
        replace(base, long_min_p_hit_near=0.68, short_max_p_opposite=0.50),
    ]
    unique: dict[tuple[tuple[str, object], ...], SelectorConfig] = {}
    for config in candidates:
        unique[tuple(sorted(asdict(config).items()))] = config
    return list(unique.values())


def _count_failures(report: dict[str, pd.DataFrame]) -> int:
    failures = 0
    for name in ["aggregate", "year", "quarter", "side"]:
        table = report.get(name, pd.DataFrame())
        if table.empty or "passes_gold" not in table.columns:
            failures += 1
        else:
            failures += int((~table["passes_gold"]).sum())
    return failures


def _score_candidate(report: dict[str, pd.DataFrame], min_calls: int) -> dict[str, object]:
    aggregate = report["aggregate"].iloc[0]
    quarter = report.get("quarter", pd.DataFrame())
    failures = _count_failures(report)
    calls = int(aggregate["calls"])
    hit_near = float(aggregate["hit_near_rate"])
    opposite = float(aggregate["opposite_rate"])
    enough_calls = calls >= min_calls
    max_quarter_opposite = (
        float(quarter["opposite_rate"].max()) if not quarter.empty else 1.0
    )
    min_quarter_hit_near = (
        float(quarter["hit_near_rate"].min()) if not quarter.empty else 0.0
    )
    score = (
        (4.0 if failures == 0 else 0.0)
        + hit_near
        - 2.0 * max(0.0, opposite - 0.20)
        - 2.5 * max(0.0, max_quarter_opposite - 0.25)
        + min(calls, 100) / 600
        - (0.25 if not enough_calls else 0.0)
        - failures
    )
    return {
        "calls": calls,
        "evaluated": int(aggregate["evaluated"]),
        "hit_near_rate": hit_near,
        "opposite_rate": opposite,
        "max_quarter_opposite_rate": max_quarter_opposite,
        "min_quarter_hit_near_rate": min_quarter_hit_near,
        "failures": failures,
        "enough_calls": enough_calls,
        "score": score,
    }


def tune_selector_config(
    calibration_predictions: pd.DataFrame,
    selector_id: str,
    min_calls: int = 20,
) -> tuple[SelectorConfig, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    configs = _candidate_configs(selector_id)
    for index, config in enumerate(configs):
        selected = select_daily_signals(calibration_predictions, config)
        report = build_gold_report(selected)
        metrics = _score_candidate(report, min_calls=min_calls)
        rows.append(
            {
                "candidate_index": index,
                "selector_id": selector_id,
                **metrics,
                **asdict(config),
            }
        )

    sweep = pd.DataFrame(rows).sort_values(
        ["failures", "enough_calls", "score", "hit_near_rate", "calls"],
        ascending=[True, False, False, False, False],
    )
    best_index = int(sweep.iloc[0]["candidate_index"])
    config = configs[best_index]
    sweep["chosen"] = False
    sweep.loc[sweep.index[0], "chosen"] = True
    sweep["chosen_config"] = str(asdict(config))
    return config, sweep
