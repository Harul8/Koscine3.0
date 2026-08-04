import pandas as pd

from koscine3.evaluation.gold_metrics import summarize_signals
from koscine3.selection.daily_selector import SelectorConfig, select_daily_signals


def test_selector_enforces_daily_side_and_cooldown_constraints() -> None:
    rows = []
    for day in pd.date_range("2026-01-01", periods=2, freq="B"):
        for i in range(8):
            rows.append(
                {
                    "date": day,
                    "symbol": f"S{i % 4}",
                    "side": "long" if i < 6 else "short",
                    "band": "liquid",
                    "threshold": 0.04,
                    "p_hit_near": 0.80 - i * 0.01,
                    "p_opposite": 0.10,
                    "symbol_return_20d_rank_pct": 0.50,
                    "expected_favorable_move": 0.06,
                    "expected_signed_close_return": 0.02,
                    "status": "evaluated",
                    "verdict": "hit",
                    "hit": True,
                    "near": False,
                    "hit_or_near": True,
                    "opposite": False,
                    "small": False,
                    "favorable_move": 0.05,
                    "signed_close_return": 0.02,
                    "model_id": "m",
                }
            )
    selected = select_daily_signals(
        pd.DataFrame(rows),
        SelectorConfig(max_signals_per_day=5, max_signals_per_side=3, cooldown_trading_days=5),
    )
    day_counts = selected[selected["selected"]].groupby("date").size()
    assert day_counts.max() <= 5
    side_counts = selected[selected["selected"]].groupby(["date", "side"]).size()
    assert side_counts.max() <= 3
    assert selected[selected["selected"]]["symbol"].is_unique


def test_selector_applies_side_specific_probability_gates() -> None:
    rows = [
        {
            "date": pd.Timestamp("2026-01-01"),
            "symbol": "LONG_LOW_HN",
            "side": "long",
            "band": "liquid",
            "threshold": 0.04,
            "p_hit_near": 0.64,
            "p_opposite": 0.20,
            "symbol_return_20d_rank_pct": 0.50,
            "expected_favorable_move": 0.06,
            "expected_signed_close_return": 0.02,
            "status": "evaluated",
            "verdict": "small",
            "hit": False,
            "near": False,
            "hit_or_near": False,
            "opposite": False,
            "small": True,
            "favorable_move": 0.01,
            "signed_close_return": 0.01,
            "model_id": "m",
        },
        {
            "date": pd.Timestamp("2026-01-01"),
            "symbol": "LONG_HIGH_OPP",
            "side": "long",
            "band": "liquid",
            "threshold": 0.04,
            "p_hit_near": 0.80,
            "p_opposite": 0.56,
            "symbol_return_20d_rank_pct": 0.50,
            "expected_favorable_move": 0.06,
            "expected_signed_close_return": 0.02,
            "status": "evaluated",
            "verdict": "small",
            "hit": False,
            "near": False,
            "hit_or_near": False,
            "opposite": False,
            "small": True,
            "favorable_move": 0.01,
            "signed_close_return": 0.01,
            "model_id": "m",
        },
        {
            "date": pd.Timestamp("2026-01-01"),
            "symbol": "SHORT_ALLOWED",
            "side": "short",
            "band": "liquid",
            "threshold": 0.04,
            "p_hit_near": 0.63,
            "p_opposite": 0.48,
            "symbol_return_20d_rank_pct": 0.50,
            "expected_favorable_move": 0.06,
            "expected_signed_close_return": 0.02,
            "status": "evaluated",
            "verdict": "hit",
            "hit": True,
            "near": False,
            "hit_or_near": True,
            "opposite": False,
            "small": False,
            "favorable_move": 0.05,
            "signed_close_return": 0.02,
            "model_id": "m",
        },
    ]
    selected = select_daily_signals(pd.DataFrame(rows), SelectorConfig(cooldown_trading_days=0))
    by_symbol = selected.set_index("symbol")
    assert not bool(by_symbol.loc["LONG_LOW_HN", "selected"])
    assert by_symbol.loc["LONG_LOW_HN", "selection_reason"] == "below_min_p_hit_near_long"
    assert not bool(by_symbol.loc["LONG_HIGH_OPP", "selected"])
    assert by_symbol.loc["LONG_HIGH_OPP", "selection_reason"] == "above_max_p_opposite_long"
    assert bool(by_symbol.loc["SHORT_ALLOWED", "selected"])


def test_selector_rejects_overextended_momentum_rank() -> None:
    rows = [
        {
            "date": pd.Timestamp("2026-01-01"),
            "symbol": "OVEREXTENDED",
            "side": "long",
            "band": "liquid",
            "threshold": 0.04,
            "p_hit_near": 0.80,
            "p_opposite": 0.20,
            "symbol_return_20d_rank_pct": 0.95,
            "expected_favorable_move": 0.06,
            "expected_signed_close_return": 0.02,
            "status": "evaluated",
            "verdict": "hit",
            "hit": True,
            "near": False,
            "hit_or_near": True,
            "opposite": False,
            "small": False,
            "favorable_move": 0.05,
            "signed_close_return": 0.02,
            "model_id": "m",
        },
        {
            "date": pd.Timestamp("2026-01-01"),
            "symbol": "NORMAL",
            "side": "long",
            "band": "liquid",
            "threshold": 0.04,
            "p_hit_near": 0.79,
            "p_opposite": 0.20,
            "symbol_return_20d_rank_pct": 0.50,
            "expected_favorable_move": 0.06,
            "expected_signed_close_return": 0.02,
            "status": "evaluated",
            "verdict": "hit",
            "hit": True,
            "near": False,
            "hit_or_near": True,
            "opposite": False,
            "small": False,
            "favorable_move": 0.05,
            "signed_close_return": 0.02,
            "model_id": "m",
        },
    ]
    selected = select_daily_signals(pd.DataFrame(rows), SelectorConfig(cooldown_trading_days=0))
    by_symbol = selected.set_index("symbol")
    assert not bool(by_symbol.loc["OVEREXTENDED", "selected"])
    assert by_symbol.loc["OVEREXTENDED", "selection_reason"] == "above_max_symbol_return_20d_rank"
    assert bool(by_symbol.loc["NORMAL", "selected"])


def test_gold_summary_counts_selected_evaluated_signals() -> None:
    signals = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"]),
            "symbol": ["A", "B", "C"],
            "side": ["long", "long", "short"],
            "selected": [True, True, False],
            "status": ["evaluated", "evaluated", "evaluated"],
            "hit": [True, False, False],
            "near": [False, True, False],
            "hit_or_near": [True, True, False],
            "opposite": [False, False, True],
            "small": [False, False, False],
            "favorable_move": [0.05, 0.035, 0.01],
            "signed_close_return": [0.02, 0.01, -0.02],
        }
    )
    summary = summarize_signals(signals)
    row = summary.iloc[0]
    assert row["calls"] == 2
    assert row["evaluated"] == 2
    assert row["hit_near_rate"] == 1.0
    assert row["opposite_rate"] == 0.0
