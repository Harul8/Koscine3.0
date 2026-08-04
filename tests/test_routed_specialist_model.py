import pandas as pd

from koscine3.datasets.splits import WalkForwardSplit
from koscine3.experiments.routed_specialist_model import (
    RoutedSpecialistConfig,
    assign_split_archetypes,
    select_routed_weekly_signals,
)


def test_assign_split_archetypes_uses_training_period_behavior() -> None:
    rows = []
    for i, date in enumerate(pd.date_range("2023-01-02", periods=80, freq="B")):
        rows.append(
            {
                "date": date,
                "symbol": "LONGY",
                "side": "long",
                "status": "evaluated",
                "strict_hit": i % 3 == 0,
                "strict_opposite": i % 3 != 0,
                "atr_pct_14": 0.04,
            }
        )
        rows.append(
            {
                "date": date,
                "symbol": "LONGY",
                "side": "short",
                "status": "evaluated",
                "strict_hit": i % 12 == 0,
                "strict_opposite": i % 12 != 0,
                "atr_pct_14": 0.04,
            }
        )
    dataset = pd.DataFrame(rows)
    universe = pd.DataFrame(
        {
            "symbol": ["LONGY"],
            "rank": [10],
            "median_turnover_lacs": [1000.0],
        }
    )
    split = WalkForwardSplit(
        name="toy",
        base_train_end="2023-12-31",
        calibration_start="2023-01-01",
        calibration_end="2023-12-31",
        prediction_start="2024-01-01",
        prediction_end="2024-12-31",
    )

    archetypes = assign_split_archetypes(dataset, universe, split, RoutedSpecialistConfig())
    row = archetypes.iloc[0]

    assert row["archetype"] == "long_trend_participation"
    assert row["primary_side"] == "long"


def test_routed_selector_blocks_non_primary_side_without_calibration() -> None:
    rows = []
    for side in ["long", "short"]:
        rows.append(
            {
                "date": pd.Timestamp("2026-01-05"),
                "symbol": f"S_{side}",
                "side": side,
                "archetype": "long_trend_participation",
                "primary_side": "long",
                "route_allowed": side == "long",
                "p_route_pair_hit": 0.90,
                "p_route_strict_hit": 0.50,
                "p_route_opposite": 0.10,
                "p_route_range": 0.20,
                "route_utility": 2.0 if side == "short" else 1.0,
                "status": "evaluated",
                "strict_hit": True,
                "strict_opposite": False,
                "strict_range_bound": False,
                "hit": True,
                "near": False,
                "hit_or_near": True,
                "opposite": False,
                "small": False,
                "favorable_move": 0.05,
                "signed_close_return": 0.04,
                "model_id": "m",
            }
        )

    selected = select_routed_weekly_signals(
        pd.DataFrame(rows),
        RoutedSpecialistConfig(
            weekly_target=2,
            min_pair_hit_probability=0.50,
            min_full_hit_probability=0.10,
            max_opposite_probability=0.50,
            max_range_probability=0.90,
            min_route_utility=0.0,
        ),
    )
    by_side = selected.set_index("side")

    assert bool(by_side.loc["long", "selected"])
    assert not bool(by_side.loc["short", "selected"])
    assert by_side.loc["short", "selection_reason"] == "route_side_disabled"
