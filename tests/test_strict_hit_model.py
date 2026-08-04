import pandas as pd

from koscine3.experiments.strict_hit_model import (
    StrictHitModelConfig,
    add_strict_hit_labels,
    select_strict_weekly_signals,
    strict_feature_columns,
)


def test_strict_hit_labels_are_exclusive_and_close_confirmed() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-01"] * 4),
            "symbol": ["A", "B", "C", "D"],
            "target_threshold": [0.04, 0.04, 0.07, 0.07],
            "hit": [True, True, False, False],
            "signed_close_return": [0.033, 0.020, 0.005, 0.020],
        }
    )
    labelled = add_strict_hit_labels(frame, StrictHitModelConfig()).set_index("symbol")

    assert bool(labelled.loc["A", "strict_hit"])
    assert not bool(labelled.loc["B", "strict_hit"])
    assert bool(labelled.loc["C", "strict_opposite"])
    assert bool(labelled.loc["D", "strict_range_bound"])
    assert (
        labelled[["strict_hit", "strict_opposite", "strict_range_bound"]]
        .sum(axis=1)
        .eq(1)
        .all()
    )


def test_strict_feature_columns_block_outcomes_and_allow_engineered_features() -> None:
    frame = pd.DataFrame(
        {
            "open": [100.0],
            "future_return": [0.10],
            "entry_open": [101.0],
            "favorable_move": [0.05],
            "strict_hit": [True],
            "setup_quality_history_score": [0.20],
            "strictatlas_side_edge_63": [0.10],
            "micro_close_location_value": [0.80],
        }
    )
    features = strict_feature_columns(
        ["open", "future_return", "entry_open", "favorable_move", "strict_hit"],
        frame,
    )

    assert "open" in features
    assert "setup_quality_history_score" in features
    assert "strictatlas_side_edge_63" in features
    assert "micro_close_location_value" in features
    assert "future_return" not in features
    assert "entry_open" not in features
    assert "favorable_move" not in features
    assert "strict_hit" not in features


def test_strict_selector_uses_probability_gates_and_weekly_diversity() -> None:
    rows = []
    for i, date in enumerate(pd.date_range("2026-01-05", periods=5, freq="B")):
        rows.append(
            {
                "date": date,
                "symbol": f"S{i}",
                "side": "long" if i % 2 == 0 else "short",
                "band": "liquid",
                "target_threshold": 0.04,
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
                "p_strict_hit_pair": 0.70 - i * 0.01,
                "p_strict_hit_full": 0.15,
                "p_strict_opposite": 0.20,
                "p_range_bound": 0.30,
                "strict_edge": 0.50,
                "strict_hit_utility": 10.0 - i,
                "model_id": "m",
            }
        )
    rows.append({**rows[0], "symbol": "LOW_PAIR", "p_strict_hit_pair": 0.10, "strict_hit_utility": 99.0})
    rows.append({**rows[1], "symbol": "HIGH_OPP", "p_strict_opposite": 0.90, "strict_hit_utility": 98.0})

    selected = select_strict_weekly_signals(
        pd.DataFrame(rows),
        StrictHitModelConfig(
            weekly_target=3,
            max_signals_per_day=1,
            max_symbol_per_week=1,
            min_pair_hit_probability=0.50,
            min_full_hit_probability=0.05,
            max_opposite_probability=0.40,
            max_range_probability=0.70,
        ),
    )
    chosen = selected[selected["selected"]]
    by_symbol = selected.set_index("symbol")

    assert len(chosen) == 3
    assert chosen.groupby(["week", "symbol"]).size().max() == 1
    assert chosen.groupby("date").size().max() <= 1
    assert by_symbol.loc["LOW_PAIR", "selection_reason"] == "below_pair_hit_probability"
    assert by_symbol.loc["HIGH_OPP", "selection_reason"] == "above_opposite_probability"
