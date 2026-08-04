from __future__ import annotations

import pandas as pd

from koscine.training import _write_replaced_dataset


def test_write_replaced_dataset_replaces_only_requested_dates(tmp_path) -> None:
    dataset_path = tmp_path / "features.parquet"
    existing = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2026-07-01", "2026-07-02", "2026-07-03", "2026-07-04"]
            ),
            "symbol": ["AAA"] * 4,
            "value": [1.0, 2.0, 3.0, 4.0],
        }
    )
    existing.to_parquet(dataset_path, index=False)
    replacement = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2026-07-03", "2026-07-02", "2026-07-02"]
            ),
            "symbol": ["AAA"] * 3,
            "value": [30.0, 20.0, 21.0],
        }
    )

    _write_replaced_dataset(
        dataset_path,
        replacement,
        pd.Timestamp("2026-07-02"),
        pd.Timestamp("2026-07-03"),
    )
    result = pd.read_parquet(dataset_path)

    assert result[["date", "symbol"]].duplicated().sum() == 0
    assert result["date"].tolist() == list(pd.date_range("2026-07-01", periods=4))
    assert result["value"].tolist() == [1.0, 21.0, 30.0, 4.0]
