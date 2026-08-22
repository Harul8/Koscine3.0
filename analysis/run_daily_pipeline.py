"""Drive the REAL daily pipeline end-to-end for a date range: ingestion, features, and all three
signal families (Buy Signals, Cash Signals, Sell Signals) -- this is what the UI's "Refresh"
button (/prod2/refresh) runs.

  ingestion: pipeline.fetch -> pipeline.fetch_fiidii -> pipeline.silver append-day + indices  (raw -> silver, computes atm_iv)
  features:  refresh_dataset_tail                                                              (silver -> daily_features.parquet)
  Buy Signals:   mover_v2 -> premium_ohlc -> direction_stage2 -> next_day -> mover_v3 -> direction_v1 -> expected_move_v1
  Cash Signals:  pipeline.zones (incremental) -> gen_cash_signals.py
  Sell Signals:  build_sell_panel.py/build_broken_wing_panel.py --append -> gen_sell_signal_history.py /
                 gen_broken_wing_signal_history.py / gen_skew_signal_history.py (partial-range,
                 looking back 10 business days from `end` so still-open trades get remarked with
                 fresh data, not just brand-new entries) -> gen_signal_dedup_and_fallback.py

Previously only ingestion/features/Buy-Signals were wired in here -- Cash Signals and Sell
Signals were separate, manually-run pipelines with no connection to the UI's Refresh button at
all (verified by grepping the whole codebase for their entrypoint scripts -- nothing called them
except a human at a terminal). Panel builds use --append (incremental, seconds) rather than the
full multi-year rebuild the underlying scripts also support, to keep a daily Refresh fast.

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
CONDOR_PANEL, BW_PANEL = "panel.parquet", "bw_panel.parquet"


def run(stage: str, cmd: list[str], cwd: str, env: dict | None = None) -> None:
    print(f"\n========== {stage} ==========\n>>> ({cwd}) {' '.join(cmd[-4:])}", flush=True)
    # PYTHONIOENCODING=utf-8 forced on every subprocess: several downstream scripts print
    # non-ASCII characters (e.g. pipeline/zones.py's "->" as an actual arrow glyph), which
    # crashes with UnicodeEncodeError the moment stdout isn't a UTF-8-capable console -- exactly
    # what happens when this whole pipeline runs as a background subprocess from the API/UI
    # Refresh button on a default (cp1252) Windows console. Layered on top of the caller's own
    # env (or os.environ if none given) so it always wins without dropping anything else.
    full_env = {**os.environ, **(env or {}), "PYTHONIOENCODING": "utf-8"}
    r = subprocess.run(cmd, cwd=cwd, env=full_env)
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
    run("silver indices", [PY, "-u", "-m", "pipeline.silver", "indices"], K3)  # append-day doesn't
    # touch the indices table (separate NSE feed, ind_close_all_*.csv) -- feeds the Indices tab's
    # daily/weekly view and S/R zones; cheap enough (~1s) to just always rebuild.

    # 2) feature refresh (silver -> daily_features.parquet tail through `end`)
    run("refresh features",
        [PY, "-c",
         "from koscine.training import refresh_dataset_tail;"
         "from koscine.config import PROCESSED_DIR;"
         f"print(refresh_dataset_tail(PROCESSED_DIR/'daily_features.parquet', source='silver', end_date='{end}'))"],
        K3)

    # 3) Buy Signals: K3 predictions (5-day book + direction overlay + 1-day book)
    env3 = {**os.environ, "PYTHONPATH": str(Path(K3) / "src")}
    run("K3 5-day mover book", [PY, "-m", "koscine3.largemove.mover_v2"], K3, env3)
    run("K3 pick premium OHLC", [PY, "-u", "analysis/premium_ohlc.py", "--append"], K3, env3)  # book_premiums.csv (near-ATM CE/PE OHLC per v2 pick) -- incremental (2026-08): only the trailing LOOKBACK_DAYS of picks are recomputed, not the full ~2yr book every run
    run("K3 direction overlay", [PY, "-m", "koscine3.largemove.direction_stage2"], K3, env3)
    run("K3 1-day book", [PY, "-m", "koscine3.largemove.next_day"], K3, env3)
    run("K3 v3 mover-precision book", [PY, "-m", "koscine3.largemove.mover_v3"], K3, env3)
    run("K3 B-direction lean (v1)", [PY, "-m", "koscine3.largemove.direction_v1"], K3, env3)
    run("K3 expected-move (tomorrow)", [PY, "-m", "koscine3.largemove.expected_move_v1"], K3, env3)

    # 4) Cash Signals: incremental S/R zone update (only computes new dates, see pipeline/zones.py)
    # then the live picks snapshot off it.
    run("zones incremental update", [PY, "-u", "-m", "pipeline.zones"], K3)
    run("cash signals", [PY, "-u", "analysis/gen_cash_signals.py"], K3)

    # 5) Sell Signals: append the option panels (incremental, seconds -- not the ~15min full
    # rebuild; panel.parquet is kept fresh here for gen_daily_marks.py's benefit even though the
    # unified signal generators below all read bw_panel.parquet now), then
    # append_daily_sell_signals.py -- fires ONLY `end`'s signal (same universe/threshold/
    # selection the live panel uses) and marks already-logged trades from the trailing 5 trading
    # days to market. Does NOT re-scan/re-select any older day -- see that script's docstring for
    # why history generation is a one-time activity now, not part of the daily flow.
    run("sell panel append", [PY, "-u", "analysis/build_sell_panel.py", "--append", end, CONDOR_PANEL], K3)
    run("broken-wing panel append", [PY, "-u", "analysis/build_broken_wing_panel.py", "--append", end, BW_PANEL], K3)
    run("append daily sell signals", [PY, "-u", "analysis/append_daily_sell_signals.py", BW_PANEL, end], K3)

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
