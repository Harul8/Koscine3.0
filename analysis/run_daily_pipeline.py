"""Drive the REAL daily pipeline end-to-end for a date range, then K3 predictions.

  ingestion: pipeline.fetch -> pipeline.fetch_fiidii -> pipeline.silver append-day  (raw -> silver, computes atm_iv)
  features:  refresh_dataset_tail                                                    (silver -> daily_features.parquet)
  books:     mover_v2 -> premium_ohlc -> direction_stage2 -> next_day -> mover_v3 -> direction_v1 -> expected_move_v1  (books, pick premiums, v3, B lean, tomorrow expected-move)

Usage:
    python analysis/run_daily_pipeline.py 2026-06-08 2026-06-12   # date range (inclusive, business days, in one go)
    python analysis/run_daily_pipeline.py 2026-06-12              # single day
    python analysis/run_daily_pipeline.py                         # default range below
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

# consolidated: pipeline (ingestion) and koscine (features) now live inside K3
K3 = r"C:\Users\rahul\Koscine 3.0"
DEFAULT_START, DEFAULT_END = "2026-06-08", "2026-06-12"
PY = sys.executable


def run(stage: str, cmd: list[str], cwd: str, env: dict | None = None) -> None:
    print(f"\n========== {stage} ==========\n>>> ({cwd}) {' '.join(cmd[-4:])}", flush=True)
    r = subprocess.run(cmd, cwd=cwd, env=env)
    if r.returncode != 0:
        raise SystemExit(f"!! STAGE FAILED ({r.returncode}): {stage}")
    print(f"-- {stage} OK", flush=True)


def main(start: str, end: str) -> None:
    dates = [d.strftime("%Y-%m-%d") for d in pd.bdate_range(start, end)]
    if not dates:
        raise SystemExit(f"!! no business days in range {start}..{end}")
    print(f"==== DAILY PIPELINE {start}..{end}  ({len(dates)} business day(s): {dates[0]}..{dates[-1]}) ====", flush=True)

    # 1) ingestion (raw fetch + FII + silver append with atm_iv) — whole range in one go
    run("fetch raw bhavcopy", [PY, "-u", "-m", "pipeline.fetch", start, end], K3)
    run("fetch FII/DII", [PY, "-u", "-m", "pipeline.fetch_fiidii", "--fetch", start, end], K3)
    run("silver append-day", [PY, "-u", "-m", "pipeline.silver", "append-day", *dates], K3)

    # 2) feature refresh (silver -> daily_features.parquet tail through `end`)
    run("refresh features",
        [PY, "-c",
         "from koscine.training import refresh_dataset_tail;"
         "from koscine.config import PROCESSED_DIR;"
         f"print(refresh_dataset_tail(PROCESSED_DIR/'daily_features.parquet', source='silver', end_date='{end}'))"],
        K3)

    # 3) K3 predictions (5-day book + direction overlay + 1-day book)
    env3 = {**os.environ, "PYTHONPATH": str(Path(K3) / "src")}
    run("K3 5-day mover book", [PY, "-m", "koscine3.largemove.mover_v2"], K3, env3)
    run("K3 pick premium OHLC", [PY, "-u", "analysis/premium_ohlc.py"], K3, env3)  # book_premiums.csv (near-ATM CE/PE OHLC per v2 pick)
    run("K3 direction overlay", [PY, "-m", "koscine3.largemove.direction_stage2"], K3, env3)
    run("K3 1-day book", [PY, "-m", "koscine3.largemove.next_day"], K3, env3)
    run("K3 v3 mover-precision book", [PY, "-m", "koscine3.largemove.mover_v3"], K3, env3)
    run("K3 B-direction lean (v1)", [PY, "-m", "koscine3.largemove.direction_v1"], K3, env3)
    run("K3 expected-move (tomorrow)", [PY, "-m", "koscine3.largemove.expected_move_v1"], K3, env3)
    print("\n================ PIPELINE COMPLETE ================", flush=True)


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) >= 2:
        s, e = args[0], args[1]
    elif len(args) == 1:
        s = e = args[0]
    else:
        s, e = DEFAULT_START, DEFAULT_END
    main(s, e)
