import pandas as pd

from koscine3.outcomes.swing_contract import compute_swing_outcomes


def _symbol_frame(symbol: str, highs: list[float], lows: list[float], closes: list[float]) -> pd.DataFrame:
    dates = pd.date_range("2026-01-01", periods=len(highs), freq="B")
    return pd.DataFrame(
        {
            "date": dates,
            "symbol": symbol,
            "open": [100.0] * len(highs),
            "high": highs,
            "low": lows,
            "close": closes,
        }
    )


def test_long_hit_near_opposite_small_and_pending() -> None:
    market = pd.concat(
        [
            _symbol_frame("HIT", [100, 104.2, 101, 101, 101, 101, 101], [99] * 7, [100] * 7),
            _symbol_frame("NEAR", [100, 103.3, 101, 101, 101, 101, 101], [99] * 7, [101] * 7),
            _symbol_frame("OPP", [100, 102, 101, 101, 101, 101, 101], [99] * 7, [100, 99, 99, 99, 99, 98, 98]),
            _symbol_frame("SMALL", [100, 102, 101, 101, 101, 101, 101], [99] * 7, [101] * 7),
        ],
        ignore_index=True,
    )
    universe = pd.DataFrame(
        {
            "symbol": ["HIT", "NEAR", "OPP", "SMALL"],
            "band": ["liquid"] * 4,
            "threshold": [0.04] * 4,
        }
    )
    outcomes = compute_swing_outcomes(market, universe)

    first_day = pd.Timestamp("2026-01-01")
    verdicts = outcomes[(outcomes["date"].eq(first_day)) & outcomes["side"].eq("long")].set_index(
        "symbol"
    )["verdict"]
    assert verdicts["HIT"] == "hit"
    assert verdicts["NEAR"] == "near"
    assert verdicts["OPP"] == "opposite"
    assert verdicts["SMALL"] == "small"

    last_day = market["date"].max()
    assert (
        outcomes[
            outcomes["date"].eq(last_day)
            & outcomes["symbol"].eq("HIT")
            & outcomes["side"].eq("long")
        ]["verdict"].iloc[0]
        == "pending_entry"
    )
    second_last = sorted(market["date"].unique())[-2]
    assert (
        outcomes[
            outcomes["date"].eq(second_last)
            & outcomes["symbol"].eq("HIT")
            & outcomes["side"].eq("long")
        ]["verdict"].iloc[0]
        == "pending_window"
    )


def test_short_hit_uses_window_low() -> None:
    market = _symbol_frame("SHORT", [101] * 7, [100, 96, 99, 99, 99, 99, 99], [100] * 7)
    universe = pd.DataFrame({"symbol": ["SHORT"], "band": ["liquid"], "threshold": [0.04]})
    outcomes = compute_swing_outcomes(market, universe)
    row = outcomes[
        outcomes["date"].eq(pd.Timestamp("2026-01-01"))
        & outcomes["symbol"].eq("SHORT")
        & outcomes["side"].eq("short")
    ].iloc[0]
    assert row["verdict"] == "hit"
    assert row["favorable_move"] > 0.04


def test_near_is_not_counted_as_opposite_bucket() -> None:
    market = _symbol_frame(
        "NEAR_OPP",
        [100, 103.3, 101, 101, 101, 101, 101],
        [99] * 7,
        [100, 99, 99, 99, 99, 98, 98],
    )
    universe = pd.DataFrame({"symbol": ["NEAR_OPP"], "band": ["liquid"], "threshold": [0.04]})
    outcomes = compute_swing_outcomes(market, universe)
    row = outcomes[
        outcomes["date"].eq(pd.Timestamp("2026-01-01"))
        & outcomes["symbol"].eq("NEAR_OPP")
        & outcomes["side"].eq("long")
    ].iloc[0]
    assert row["verdict"] == "near"
    assert bool(row["near"])
    assert not bool(row["opposite"])
