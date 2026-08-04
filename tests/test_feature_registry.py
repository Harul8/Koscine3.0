import pandas as pd
import pytest

from koscine3.data.feature_registry import build_feature_registry


def test_feature_registry_blocks_future_and_label_columns() -> None:
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-01"]),
            "symbol": ["AAA"],
            "close": [100.0],
            "ret_5d": [0.01],
            "future_5d_high": [105.0],
            "entry_1d_open": [101.0],
            "label_up_4pct_5d": [1],
        }
    )
    registry = build_feature_registry(df)
    assert "ret_5d" in registry.feature_columns
    assert "future_5d_high" in registry.blocked_columns
    assert "entry_1d_open" in registry.blocked_columns
    assert "label_up_4pct_5d" in registry.blocked_columns
    assert "future_5d_high" not in registry.feature_columns
    with pytest.raises(ValueError):
        registry.assert_safe(["ret_5d", "future_5d_high"])

