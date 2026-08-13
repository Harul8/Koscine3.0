"""Daily operational script for Sell Signals -- NOT a backtest/history regen. This is what the
daily Refresh pipeline (analysis/run_daily_pipeline.py) runs; the full gen_*_signal_history.py
scripts (unrestricted date range, scanning/selecting across the whole history) are a ONE-TIME
activity now, run by hand when needed, not part of this daily flow. See the 2026-08 "unify Daily
Signals and History" change (explicit user decision) for why this split exists.

Two separate concerns, kept separate on purpose:

1. FIRE today (single day only): invokes the 3 unified gen_*_signal_history.py generators with
   ENTRY_DATE_MIN=ENTRY_DATE_MAX=today, then gen_signal_dedup_and_fallback.py scoped to the same
   single day. Since the window is exactly one day, this can only ever ADD today's pick, never
   touch or re-derive any past day's selection (see those scripts' _merge_partial: rows outside
   the window are always preserved as-is). Reuses the SAME selection logic the live
   /prod2/*_strategy endpoints use (same universe, same ROR gate, same entry-time-only tie-break)
   -- no separate/duplicated selection logic to drift out of sync with the live panel.

2. MARK still-open trades to market (trailing 5 trading days): for every row across all 3
   history CSVs whose entry_date falls in the trailing 5 trading days and whose outcome is still
   "pending", recompute ONLY its price walk for the ALREADY-LOGGED symbol/strike/credit (never
   re-picked, never re-scanned against other candidates) using now-available forward bars, and
   update exit_value/pnl/pnl_per_lot/ror_pct/outcome/max_dd_pct in place. This is what keeps a
   recent pick's actual PnL current without ever letting its identity change -- a trade fired 6+
   trading days ago is left untouched here (it has already resolved: either it hit its 5-day
   window or the SAFE_DTE forced-exit backstop).

Usage:
    python analysis/append_daily_sell_signals.py bw_panel.parquet               # fires for today
    python analysis/append_daily_sell_signals.py bw_panel.parquet 2026-08-12    # fires for a
        specific date instead -- e.g. when analysis/run_daily_pipeline.py is run for a past `end`
        date rather than today (a routine Refresh always passes today, so this is the common case).
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"C:\Users\rahul\Koscine 3.0")
OUTDIR = ROOT / "locks" / "prod_sell_strategies"
FWD, SAFE_DTE, MIN_VOL = 5, 4, 50

# (strategy label, history CSV, is this a 2-leg vertical (skew) rather than a 4-leg condor shape)
STRATEGIES = [
    ("condor", "signal_history.csv", False),
    ("broken_wing", "broken_wing_signal_history.csv", False),
    ("skew", "skew_signal_history.csv", True),
]


def _today_str() -> str:
    return pd.Timestamp.now(tz="Asia/Kolkata").strftime("%Y-%m-%d")


def _run(args: list[str]) -> None:
    print(f"\n>>> {' '.join(args)}", flush=True)
    r = subprocess.run([sys.executable, "-u", *args], cwd=str(ROOT))
    if r.returncode != 0:
        raise SystemExit(f"!! stage failed: {args}")


def _fire_today(panel_path: str, today: str) -> None:
    """Single-day-window run of the 3 unified generators + the coordinator, scoped to `today`
    only -- structurally cannot touch any other day (see module docstring)."""
    _run(["analysis/gen_sell_signal_history.py", panel_path, today, today])
    _run(["analysis/gen_broken_wing_signal_history.py", panel_path, today, today])
    _run(["analysis/gen_skew_signal_history.py", panel_path, today, today])
    _run(["analysis/gen_signal_dedup_and_fallback.py", today, today])


def _mark_pending(panel_path: str, today: str) -> None:
    """Refresh exit_value/pnl/outcome for already-logged, still-pending trades entered in the
    trailing 5 trading days -- reusing their EXISTING symbol/strike/credit, never re-selecting."""
    panel = pd.read_parquet(panel_path)
    panel["date"] = pd.to_datetime(panel["date"]); panel["expiry"] = pd.to_datetime(panel["expiry"])
    tdays = np.array(sorted(panel["date"].unique()))
    tpos = {d: k for k, d in enumerate(tdays)}
    cbar = {k: dict(zip(pd.to_datetime(g["date"]).values, zip(g["open"].values, g["close"].values, g["vol"].values)))
            for k, g in panel.groupby(["symbol", "opt_type", "expiry", "strike"], sort=False)}

    def bar(sym, ot, exp, strike, d):
        b = cbar.get((sym, ot, exp, strike))
        return b.get(pd.Timestamp(d)) if b else None

    cutoff = pd.bdate_range(end=today, periods=FWD + 1)[0].strftime("%Y-%m-%d")

    for name, fname, is_skew in STRATEGIES:
        fp = OUTDIR / fname
        if not fp.exists():
            continue
        df = pd.read_csv(fp)
        n_marked = 0
        for idx, row in df.iterrows():
            if row["outcome"] != "pending" or row["entry_date"] < cutoff:
                continue
            e_date = pd.Timestamp(row["entry_date"])
            p = tpos.get(e_date)
            if p is None:
                continue
            win = [pd.Timestamp(x) for x in tdays[p: p + FWD]]
            if not win:
                continue
            exp = pd.Timestamp(row["expiry"])
            credit = float(row["credit"])
            width = credit + float(row["max_risk"])   # width == credit + max_risk, always (max_risk = width - credit)
            vals, dd, exited_early = [], 0.0, False
            for d in win:
                if is_skew:
                    side = row["side"]
                    sb = bar(row["symbol"], side, exp, float(row["short_strike"]), d)
                    lb = bar(row["symbol"], side, exp, float(row["long_strike"]), d)
                    ok = sb is not None and lb is not None and sb[2] >= MIN_VOL and lb[2] >= MIN_VOL
                    value = max(0.0, min(width, sb[1] - lb[1])) if ok else None
                else:
                    legbars = [bar(row["symbol"], "CE", exp, float(row["short_ce"]), d),
                               bar(row["symbol"], "CE", exp, float(row["long_ce"]), d),
                               bar(row["symbol"], "PE", exp, float(row["short_pe"]), d),
                               bar(row["symbol"], "PE", exp, float(row["long_pe"]), d)]
                    ok = all(b is not None and b[2] >= MIN_VOL for b in legbars)
                    value = max(0.0, min(width, (legbars[0][1] - legbars[1][1]) + (legbars[2][1] - legbars[3][1]))) if ok else None
                if value is not None:
                    vals.append(value)
                    dd = min(dd, credit - value)
                if (exp - d).days <= SAFE_DTE:
                    exited_early = True
                    break
            if not vals:
                continue
            complete = exited_early or (len(win) == FWD)
            exit_value = vals[-1]
            pnl = round(credit - exit_value, 2)
            risk = float(row["max_risk"])
            lot = row.get("lot_size")
            df.at[idx, "exit_value"] = round(exit_value, 2)
            df.at[idx, "pnl"] = pnl
            df.at[idx, "pnl_per_lot"] = round(pnl * lot, 1) if pd.notna(lot) else None
            df.at[idx, "ror_pct"] = round(pnl / risk * 100, 1)
            df.at[idx, "max_dd_pct"] = round(dd / risk * 100, 1)
            if not complete:
                df.at[idx, "outcome"] = "pending"
            else:
                df.at[idx, "outcome"] = "win" if pnl > 0 else "loss"
            n_marked += 1
        if n_marked:
            df.to_csv(fp, index=False)
        print(f"[mark_pending] {name}: refreshed {n_marked} still-pending trade(s) from the trailing {FWD} trading days", flush=True)


def main() -> None:
    panel_path = sys.argv[1] if len(sys.argv) > 1 else "bw_panel.parquet"
    today = sys.argv[2] if len(sys.argv) > 2 else _today_str()
    _fire_today(panel_path, today)
    _mark_pending(panel_path, today)


if __name__ == "__main__":
    main()
