import pandas as pd

from koscine3.experiments.trajectory_strict_model import (
    TrajectoryStrictConfig,
    add_trajectory_features,
    build_matched_pair_training_frame,
    trajectory_feature_columns,
)


def _market_rows() -> list[dict[str, object]]:
    rows = []
    for symbol, offset in [("AAA", 0.0), ("BBB", 0.02)]:
        for i, date in enumerate(pd.date_range("2024-01-01", periods=24, freq="B")):
            close = 100.0 + offset * 100.0 + i
            rows.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "side": "long",
                    "open": close - 0.5,
                    "high": close + 1.0,
                    "low": close - 1.0,
                    "close": close,
                    "prev_close": close - 1.0,
                    "volume": 100000 + i * 1000,
                    "turnover_lacs": 1000 + i * 10,
                    "delivery_pct": 45 + i * 0.1,
                    "fut_oi": 10000 + i * 20,
                    "fut_vol": 2000 + i * 30,
                    "pcr_oi": 1.0 + i * 0.001,
                    "pcr_vol": 0.9 + i * 0.001,
                    "atm_iv": 18 + i * 0.02,
                    "put_call_iv_skew": 0.1,
                    "atr_pct_14": 0.03,
                    "range_pct": 0.02,
                    "bb_width_20": 0.08,
                    "realized_vol_20": 0.2,
                    "compression_composite": 0.5,
                    "ema_20_dist": 0.03,
                    "di_diff": 5.0,
                    "rel_ret_5d_vs_nifty": 0.01,
                    "stock_rel_sector_ret_5d": 0.01,
                    "stock_rel_sector_ret_20d": 0.02,
                    "nifty_ret_5d": 0.005,
                    "sector_ret_5d": 0.004,
                    "mkt_pct_above_sma20": 0.55,
                    "nifty_realized_vol_20": 0.2,
                }
            )
    return rows


def test_add_trajectory_features_adds_side_aligned_safe_columns() -> None:
    long_rows = _market_rows()
    short_rows = [{**row, "side": "short"} for row in long_rows]
    dataset = pd.DataFrame([*long_rows, *short_rows])

    enriched = add_trajectory_features(dataset)
    features = trajectory_feature_columns(["close", "volume"], enriched)

    assert "traj_side_ret_5" in enriched.columns
    assert "traj_setup_quality_score" in features
    assert "future_5d_high" not in features
    sample = enriched[(enriched["symbol"].eq("AAA")) & (enriched["date"].eq(pd.Timestamp("2024-01-15")))]
    by_side = sample.set_index("side")
    assert by_side.loc["long", "traj_side_ret_5"] > 0
    assert by_side.loc["short", "traj_side_ret_5"] < 0


def test_matched_pair_training_prefers_nearest_opposites() -> None:
    rows = []
    dates = pd.date_range("2024-01-01", periods=12, freq="B")
    for i, date in enumerate(dates):
        rows.append(
            {
                "date": date,
                "symbol": "AAA",
                "band": "liquid",
                "setup_id": "momentum_breakout",
                "traj_regime_breadth_bin": 1,
                "traj_regime_vol_bin": 1,
                "strict_hit": i in {5, 6},
                "strict_opposite": i not in {5, 6},
            }
        )
    train = pd.DataFrame(rows)

    matched = build_matched_pair_training_frame(
        train,
        TrajectoryStrictConfig(matched_opposites_per_hit=2, min_matched_pair_rows=1),
    )

    assert int(matched["strict_hit"].sum()) == 2
    assert int(matched["strict_opposite"].sum()) == 4
    assert set(matched["strict_pair_is_hit"].unique()) == {0, 1}
    chosen_opposite_dates = set(matched[matched["strict_opposite"]]["date"])
    assert dates[4] in chosen_opposite_dates
    assert dates[7] in chosen_opposite_dates

