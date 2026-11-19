"""Daily operational script for Sell Signals -- NOT a backtest/history regen. This is what the
daily Refresh pipeline (analysis/run_daily_pipeline.py) runs; the full gen_*_signal_history.py
scripts (unrestricted date range, scanning/selecting across the whole history) are a ONE-TIME
activity now, run by hand when needed, not part of this daily flow. See the 2026-08 "unify Daily
Signals and History" change (explicit user decision) for why this split exists.

Two separate concerns, kept separate on purpose:

1. FIRE today (single day only): invokes the 3 unified gen_*_signal_history.py generators with
   SIGNAL_DATE_MIN=SIGNAL_DATE_MAX=today (2026-09, explicit user decision -- no-look-ahead fix:
   this used to be ENTRY_DATE_MIN/MAX; a signal now fires the moment ITS OWN signal-day close
   lands, priced entirely off that close -- no more waiting on the next day's data to exist), then
   gen_signal_dedup_and_fallback.py scoped to the DERIVED entry_date (today's next business day --
   that script's own range args are still entry_date-scoped). Since the window is exactly one day,
   this can only ever ADD today's pick, never touch or re-derive any past day's selection (see
   those scripts' _merge_partial: rows outside the window are always preserved as-is). Reuses the
   SAME selection logic the live /prod2/*_strategy endpoints use (same universe, same ROR gate,
   same signal-time-only tie-break) -- no separate/duplicated selection logic to drift out of sync
   with the live panel.

2. MARK still-open trades to market (trailing 5 trading days): for every row across all 3
   history CSVs whose entry_date falls in the trailing 5 trading days and whose outcome is still
   "pending", recompute ONLY its price walk for the ALREADY-LOGGED symbol/strike/credit (never
   re-picked, never re-scanned against other candidates) using now-available forward bars, and
   update exit_value/pnl/pnl_per_lot/ror_pct/outcome/max_dd_pct in place. Once fired, a trade is
   entered at entry_date's own OPEN and carried/marked at OPEN prices every day of the walk (never
   close, 2026-09 explicit user decision) -- a row with an empty walk (entry_date's data hasn't
   landed yet) is simply left untouched here until it does. This is what keeps a recent pick's
   actual PnL current without ever letting its identity change -- a trade fired 6+ trading days
   ago is left untouched here (it has already resolved: either it hit its 5-day window or the
   SAFE_DTE forced-exit backstop).

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
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))
from koscine3.data.sources import load_market_data  # noqa: E402

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


def _next_bday(d: str) -> str:
    """Next trading-day LABEL for the coordinator's entry_date-scoped range args -- naive
    Sat/Sun-only skip (no NSE holiday calendar here), same limitation as the generators' own
    _entry_date_for() and this file's own `cutoff` calc below. Self-corrects on the next
    gen_signal_dedup_and_fallback.py run once the real entry_date's data exists."""
    return pd.bdate_range(start=pd.Timestamp(d) + pd.Timedelta(days=1), periods=1)[0].strftime("%Y-%m-%d")


def _fire_today(panel_path: str, today: str) -> None:
    """Single-day-window run of the 3 unified generators + the coordinator, scoped to `today`
    only -- structurally cannot touch any other day (see module docstring). The 3 generators are
    scoped by SIGNAL_DATE (today IS the signal date whose close just got ingested); the
    coordinator's own range args are entry_date-scoped, so it gets the derived next-business-day
    label instead."""
    _run(["analysis/gen_sell_signal_history.py", panel_path, today, today])
    _run(["analysis/gen_broken_wing_signal_history.py", panel_path, today, today])
    _run(["analysis/gen_skew_signal_history.py", panel_path, today, today])
    entry_label = _next_bday(today)
    _run(["analysis/gen_signal_dedup_and_fallback.py", entry_label, entry_label])


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

    def legseq(sym, ot, exp, strike, win):
        # entry-open (win[0] == entry_date, since `win` here already starts at entry_date -- see
        # below) + per-day bars -- used for the "Buy PnL" hedge comparison, kept in sync with the
        # gen_*_signal_history.py generators' identical entry-at-entry_date's-own-open convention
        # (2026-09, explicit user decision) so a re-marked pending trade's buy_pnl_per_lot doesn't
        # go stale relative to its own exit_value.
        b = cbar.get((sym, ot, exp, strike))
        if not b:
            return None
        e = b.get(win[0])
        if not e or e[0] <= 0:
            return None
        return e[0], [b.get(d) for d in win]

    # underlying OPEN, for exit_underlying (Exit LTP) -- 2026-09, explicit user decision: once
    # entered, a trade is carried/marked at OPEN prices every day of the walk (never close). The
    # SIGNAL date's underlying close is already logged at fire time; the EXIT day's open isn't
    # known until now.
    mk = load_market_data(columns=["date", "symbol", "open"])
    mk["date"] = pd.to_datetime(mk["date"]); mk["symbol"] = mk["symbol"].astype(str)
    OPEN = mk.set_index(["symbol", "date"])["open"].to_dict()

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
            leg_opens = []    # (sell_open, buy_open) per kept day, parallel to `vals` -- same
                              # exit_sell_premium/exit_buy_premium breakdown the generators track,
                              # kept in sync here so a re-marked pending trade's Exit-value
                              # breakdown doesn't go stale relative to its own exit_value. 2026-09,
                              # explicit user decision: carried/marked at OPEN every day, never close.
            exit_days = []    # win[i] per kept day, parallel to `vals` -- for exit_underlying.
            for d in win:
                if is_skew:
                    side = row["side"]
                    sb = bar(row["symbol"], side, exp, float(row["short_strike"]), d)
                    lb = bar(row["symbol"], side, exp, float(row["long_strike"]), d)
                    ok = sb is not None and lb is not None and sb[2] >= MIN_VOL and lb[2] >= MIN_VOL
                    value = max(0.0, min(width, sb[0] - lb[0])) if ok else None
                    leg_pair = (sb[0], lb[0]) if ok else None
                else:
                    legbars = [bar(row["symbol"], "CE", exp, float(row["short_ce"]), d),
                               bar(row["symbol"], "CE", exp, float(row["long_ce"]), d),
                               bar(row["symbol"], "PE", exp, float(row["short_pe"]), d),
                               bar(row["symbol"], "PE", exp, float(row["long_pe"]), d)]
                    ok = all(b is not None and b[2] >= MIN_VOL for b in legbars)
                    value = max(0.0, min(width, (legbars[0][0] - legbars[1][0]) + (legbars[2][0] - legbars[3][0]))) if ok else None
                    leg_pair = (legbars[0][0] + legbars[2][0], legbars[1][0] + legbars[3][0]) if ok else None
                if value is not None:
                    vals.append(value)
                    leg_opens.append(leg_pair)
                    exit_days.append(d)
                    dd = min(dd, credit - value)
                if (exp - d).days <= SAFE_DTE:
                    exited_early = True
                    break
            complete = exited_early or (len(win) == FWD)
            if not vals:
                # win still growing (genuinely still pending, more days may land later) -- nothing
                # to write, leave the row exactly as-is. But if the window is already fully played
                # out (or force-exited) and STILL never found a single valid day (e.g. a leg's
                # volume never cleared MIN_VOL), that's RESOLVED, just unpriceable -- flip it to a
                # terminal "no_fill" instead of leaving it stuck showing "pending" forever once it
                # ages out of this function's trailing-5-day reprocessing window (2026-09,
                # discovered while verifying the no-look-ahead fix).
                if complete:
                    df.at[idx, "outcome"] = "no_fill"
                    n_marked += 1
                continue
            exit_value = vals[-1]
            exit_sell_premium, exit_buy_premium = leg_opens[-1]
            exit_underlying_raw = OPEN.get((row["symbol"], exit_days[-1]))
            exit_underlying = round(float(exit_underlying_raw), 1) if exit_underlying_raw is not None and pd.notna(exit_underlying_raw) else None
            pnl = round(credit - exit_value, 2)
            risk = float(row["max_risk"])
            lot = row.get("lot_size")

            # "Buy PnL" hedge comparison -- same OPPOSITE-type, SAME-strike-as-sold logic the
            # generators use (see gen_broken_wing_signal_history.py for the full rationale): entered
            # at entry_date's own open (win[0]), peak-searched across opens, over the full FWD-day
            # window regardless of this trade's own early-exit timing.
            if is_skew:
                side = row["side"]
                buy_ot = "PE" if side == "CE" else "CE"
                buy_strike = float(row["short_strike"])
            else:
                richer_side = row.get("richer_side")
                buy_ot = "PE" if richer_side == "CE" else "CE"
                buy_strike = float(row["short_ce"]) if richer_side == "CE" else float(row["short_pe"])
            buy_seq = legseq(row["symbol"], buy_ot, exp, buy_strike, win)
            buy_pnl_per_lot = None
            if buy_seq is not None and buy_seq[0] > 0:
                buy_valid_opens = [b[0] for b in buy_seq[1] if b is not None and b[2] >= MIN_VOL]
                if buy_valid_opens and pd.notna(lot):
                    buy_pnl_per_lot = round((max(buy_valid_opens) - buy_seq[0]) * lot, 1)

            df.at[idx, "exit_value"] = round(exit_value, 2)
            df.at[idx, "exit_sell_premium"] = round(exit_sell_premium, 2)
            df.at[idx, "exit_buy_premium"] = round(exit_buy_premium, 2)
            df.at[idx, "exit_underlying"] = exit_underlying
            df.at[idx, "buy_pnl_per_lot"] = buy_pnl_per_lot
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
