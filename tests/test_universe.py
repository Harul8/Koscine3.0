import pandas as pd

from koscine3.data.universe import UniverseConfig, build_universe


def test_build_universe_ranks_by_turnover_and_assigns_bands() -> None:
    rows = []
    dates = pd.date_range("2025-01-01", periods=10, freq="B")
    for symbol_idx in range(5):
        for date in dates:
            rows.append(
                {
                    "date": date,
                    "symbol": f"S{symbol_idx}",
                    "open": 100.0,
                    "high": 101.0,
                    "low": 99.0,
                    "close": 100.0,
                    "turnover_lacs": 1000.0 - symbol_idx * 100.0,
                    "volume": 10000.0 - symbol_idx,
                    "fut_close": 100.0,
                }
            )
    df = pd.DataFrame(rows)
    universe = build_universe(
        df,
        UniverseConfig(
            cutoff_date="2025-01-31",
            lookback_days=10,
            top_n=3,
            liquid_n=1,
            min_coverage=1.0,
            min_total_observed_days=10,
        ),
    )
    assert list(universe["symbol"]) == ["S0", "S1", "S2"]
    assert universe.iloc[0]["band"] == "liquid"
    assert universe.iloc[0]["threshold"] == 0.04
    assert universe.iloc[1]["band"] == "wide"
    assert universe.iloc[1]["threshold"] == 0.07
