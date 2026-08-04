"""Prepare synchronized 1-minute paths and 5-minute training data."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from bars import build_straddle_1m, resample_to_5m
from features import build_exit_training_frame


def _read(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--legs",
        type=Path,
        default=Path("data/intraday/options_contract_legs_1m.parquet"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/intraday"),
    )
    parser.add_argument("--label-horizon-bars", type=int, default=6)
    args = parser.parse_args()

    legs = _read(args.legs)
    straddle_1m = build_straddle_1m(legs)
    decision_5m = resample_to_5m(straddle_1m)
    training = build_exit_training_frame(
        decision_5m,
        label_horizon_bars=args.label_horizon_bars,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    straddle_1m.to_parquet(args.output_dir / "options_straddle_1m.parquet", index=False)
    decision_5m.to_parquet(args.output_dir / "options_straddle_5m.parquet", index=False)
    training.to_parquet(args.output_dir / "exit_training_5m.parquet", index=False)
    print(
        f"prepared {straddle_1m.trade_id.nunique()} trades: "
        f"{len(straddle_1m):,} execution minutes, "
        f"{len(decision_5m):,} completed 5-minute decisions"
    )


if __name__ == "__main__":
    main()
