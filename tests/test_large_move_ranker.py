import pandas as pd

from koscine3.experiments.large_move_ranker import (
    LargeMoveRankerConfig,
    add_large_move_labels,
    select_diverse_weekly_signals,
    select_setup_portfolio_signals,
)


def test_large_move_labels_prioritize_clean_large_moves() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-01"] * 5),
            "symbol": ["A", "B", "C", "D", "E"],
            "band": ["liquid", "liquid", "wide", "wide", "liquid"],
            "favorable_move": [0.045, 0.033, 0.061, 0.050, 0.005],
            "hit": [True, False, False, False, False],
            "near": [False, True, False, False, False],
            "hit_or_near": [True, True, False, False, False],
            "opposite": [False, False, False, True, True],
        }
    )
    labelled = add_large_move_labels(frame).set_index("symbol")

    assert labelled.loc["A", "large_rank_label"] == 4
    assert labelled.loc["B", "large_rank_label"] == 3
    assert labelled.loc["C", "large_rank_label"] == 2
    assert labelled.loc["D", "large_rank_label"] == 0
    assert labelled.loc["E", "large_rank_label"] == 0
    assert bool(labelled.loc["C", "floor_success"])
    assert bool(labelled.loc["C", "clean_large"])


def test_large_move_selector_enforces_weekly_and_monthly_diversity() -> None:
    rows = []
    dates = pd.to_datetime(["2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08"])
    symbols = ["A", "A", "B", "C", "D", "E", "F", "G"]
    for i, symbol in enumerate(symbols):
        rows.append(
            {
                "date": dates[i % len(dates)],
                "symbol": symbol,
                "side": "long" if i % 2 == 0 else "short",
                "ranker_utility_score": 10.0 - i,
            }
        )
    selected = select_diverse_weekly_signals(
        pd.DataFrame(rows),
        LargeMoveRankerConfig(
            weekly_target=4,
            max_signals_per_day=2,
            max_signals_per_week_side=3,
            max_symbol_per_week=1,
            max_symbol_per_month=1,
            daily_pool_rank=8,
        ),
    )
    chosen = selected[selected["selected"]]

    assert len(chosen) == 4
    assert chosen.groupby(["week", "symbol"]).size().max() == 1
    assert chosen.groupby(["month", "symbol"]).size().max() == 1
    assert chosen.groupby("date").size().max() <= 2


def test_setup_portfolio_selector_uses_configured_setups() -> None:
    rows = []
    dates = pd.to_datetime(["2026-01-05", "2026-01-06", "2026-01-07"])
    setups = ["iv_oi_impulse", "volatility_expansion", "market_breadth", "outside"]
    for i, setup_id in enumerate(setups * 2):
        rows.append(
            {
                "date": dates[i % len(dates)],
                "symbol": f"S{i}",
                "side": "long" if i % 2 == 0 else "short",
                "setup_id": setup_id,
                "ranker_utility_score": 10.0 - i,
            }
        )
    selected = select_setup_portfolio_signals(
        pd.DataFrame(rows),
        LargeMoveRankerConfig(
            selector_mode="setup_round_robin",
            setup_portfolio="iv_oi_impulse,volatility_expansion",
            weekly_target=2,
            setup_weekly_quota=1,
            max_signals_per_day=2,
        ),
    )
    chosen = selected[selected["selected"]]

    assert set(chosen["setup_id"]) == {"iv_oi_impulse", "volatility_expansion"}
    assert len(chosen) == 2
    assert chosen.groupby(["week", "symbol"]).size().max() == 1
