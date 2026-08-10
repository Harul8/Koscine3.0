"""Multi-day NIFTY Broken-Wing credit-spread backtest, mirroring analysis/gen_sell_signal_history.py's
EOD mark-to-market convention (the strategy that actually works for stocks: median ~Rs 2,475/lot
condor, ~Rs 6,630/lot broken-wing, ~Rs 7,660/lot skew) applied to the index instead of a tight
2-hour intraday scalp.

Why the pivot from credit_spread_path.py's 2-hour hold: that version's median PnL was ~Rs 114/lot
(lot=65) -- two orders of magnitude below the stock strategies. A 2-hour window on strikes just
0.3-0.6% OTM has almost no theta to collect; the fix isn't a better ML filter on that structure,
it's holding a position long enough (multiple trading days, through a real chunk of a weekly
option's decay curve) to collect a comparable premium, same as the stock book already proven to
work. NIFTY options are cash-settled (unlike stock options), so the physical-delivery-margin
DTE/SAFE_DTE logic in gen_sell_signal_history.py doesn't apply here -- SAFE_DTE below is just a
gamma-risk cutoff (exit before expiry-day whipsaw), not a delivery-avoidance rule.

NIFTY runs WEEKLY expiries (~7 calendar days apart, confirmed from the chain data: median gap
7 days, range 5-8), a much shorter cycle than the monthly stock-option chain gen_sell_signal_history
rolls on -- DTE_MIN/FWD below are recalibrated for that, not copied from the stock constants.

EOD mark = the LAST available 5-min bar of each trading day per (expiry, strike) -- the closest
proxy to a daily close this 5-min-granularity chain data offers.
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"C:\Users\rahul\Koscine 3.0")
OPTION_CHAIN_DIR = ROOT / "data" / "intraday" / "nifty_option_chain_5m"
EOD_CACHE = ROOT / "data" / "intraday" / "nifty_option_chain_eod_cache.parquet"
NIFTY_LOT_SIZE = 65   # current lot as of Aug 2026 (silver/lot_size.parquet); PnL-per-lot uses this

_EOD_COLUMNS = ["timestamp", "expiry", "strike", "underlying_close",
                "ce_close", "ce_volume", "ce_oi", "pe_close", "pe_volume", "pe_oi"]


def load_eod_table(option_chain_dir: Path = OPTION_CHAIN_DIR, cache_path: Path = EOD_CACHE) -> pd.DataFrame:
    """One row per (date, expiry, strike): the LAST 5-min bar of that trading day. Mtime-invalidated
    cache against the newest source file, same pattern as option_features.load_chain()."""
    files = sorted(glob.glob(str(option_chain_dir / "*.parquet")))
    if not files:
        raise FileNotFoundError(f"no parquet files under {option_chain_dir}")
    newest_source = max(Path(f).stat().st_mtime for f in files)
    if cache_path.exists() and cache_path.stat().st_mtime >= newest_source:
        return pd.read_parquet(cache_path)

    frames = [pd.read_parquet(f, columns=_EOD_COLUMNS) for f in files]
    chain = pd.concat(frames, ignore_index=True)
    chain["timestamp"] = pd.to_datetime(chain["timestamp"])
    chain["expiry"] = pd.to_datetime(chain["expiry"])
    chain["date"] = chain["timestamp"].dt.date
    # last bar per (date, expiry, strike) = EOD mark
    idx = chain.groupby(["date", "expiry", "strike"])["timestamp"].idxmax()
    eod = chain.loc[idx].sort_values(["date", "expiry", "strike"]).reset_index(drop=True)
    eod.to_parquet(cache_path, index=False)
    return eod


def build_eod_spreads(eod: pd.DataFrame, short_otm: float, wing_ce: float, wing_pe: float,
                       fwd_days: int, dte_min: int, safe_dte: int, min_vol: int = 20,
                       min_risk_frac: float = 0.05, stop_loss_frac: float | None = None) -> pd.DataFrame:
    """One entry per trading day (subject to dte_min/safe_dte gating), Broken-Wing structure
    (narrow call wing / wide put wing -- same direction that works for stocks), held up to
    fwd_days trading days or until DTE drops to safe_dte, whichever first. Mirrors
    gen_sell_signal_history.py's clamp-to-[0,width] mark-to-market and pnl/ror/dd conventions."""
    trading_days = sorted(eod["date"].unique())
    tpos = {d: i for i, d in enumerate(trading_days)}
    by_day = {d: g for d, g in eod.groupby("date", sort=False)}

    out = []
    for e_date in trading_days:
        day = by_day[e_date]
        cand_exps = sorted(day["expiry"].unique())
        exp = next((e for e in cand_exps if (pd.Timestamp(e).date() - e_date).days >= dte_min), None)
        if exp is None:
            continue
        dte = (pd.Timestamp(exp).date() - e_date).days
        chain_today = day[day["expiry"] == exp]
        u = float(chain_today["underlying_close"].iloc[0])

        def pick(is_call: bool, target: float):
            c = chain_today[chain_today["strike"].sub(0).notna()]
            return c.iloc[(c["strike"] - target).abs().argmin()]

        sce = pick(True, u * (1 + short_otm))
        lce = pick(True, u * (1 + short_otm + wing_ce))
        spe = pick(False, u * (1 - short_otm))
        lpe = pick(False, u * (1 - short_otm - wing_pe))
        if sce["strike"] == lce["strike"] or spe["strike"] == lpe["strike"]:
            continue   # degenerate: wing distance smaller than strike spacing
        if pd.isna(sce["ce_close"]) or pd.isna(spe["pe_close"]) or sce["ce_close"] <= 0 or spe["pe_close"] <= 0:
            continue
        if sce["ce_volume"] < min_vol or spe["pe_volume"] < min_vol:
            continue

        credit = (sce["ce_close"] + spe["pe_close"]) - (lce["ce_close"] + lpe["pe_close"])
        width = max(lce["strike"] - sce["strike"], spe["strike"] - lpe["strike"])
        risk = width - credit
        if credit <= 0 or risk <= 0 or risk < min_risk_frac * width:
            continue
        entry_ror = credit / risk * 100

        p0 = tpos[e_date]
        window_dates = trading_days[p0 + 1: p0 + 1 + fwd_days]
        strikes = {"sc": (True, sce["strike"]), "lc": (True, lce["strike"]),
                   "sp": (False, spe["strike"]), "lp": (False, lpe["strike"])}

        vals, dd, exit_date, exit_dte = [], 0.0, e_date, dte
        for d in window_dates:
            d_dte = (pd.Timestamp(exp).date() - d).days
            if d_dte < safe_dte:
                break
            g = by_day.get(d)
            if g is None:
                continue
            g_exp = g[g["expiry"] == exp]
            if g_exp.empty:
                continue
            rows = {}
            ok = True
            for k, (is_call, strike) in strikes.items():
                r = g_exp[g_exp["strike"] == strike]
                col = "ce_close" if is_call else "pe_close"
                volcol = "ce_volume" if is_call else "pe_volume"
                if r.empty or pd.isna(r[col].iloc[0]) or r[volcol].iloc[0] < min_vol:
                    ok = False
                    break
                rows[k] = float(r[col].iloc[0])
            if not ok:
                continue   # thin/missing day, skip the mark (matches stock version's guard)
            value = (rows["sc"] - rows["lc"]) + (rows["sp"] - rows["lp"])
            value = max(0.0, min(width, value))
            upnl = credit - value
            dd = min(dd, upnl)
            vals.append(value)
            exit_date, exit_dte = d, d_dte
            if stop_loss_frac is not None and upnl <= stop_loss_frac * risk:
                break   # risk control: cap the loss rather than ride it to the DTE-gated exit
            if d_dte <= safe_dte:
                break
        if not vals:
            continue

        exit_value = vals[-1]
        pnl = credit - exit_value
        out.append({
            "entry_date": e_date.isoformat(), "exit_date": exit_date.isoformat(),
            "expiry": pd.Timestamp(exp).date().isoformat(), "dte_entry": dte, "dte_exit": exit_dte,
            "underlying": round(u, 1),
            "short_ce": float(sce["strike"]), "long_ce": float(lce["strike"]),
            "short_pe": float(spe["strike"]), "long_pe": float(lpe["strike"]),
            "credit": round(credit, 2), "width": round(width, 2), "max_risk": round(risk, 2),
            "entry_ror_pct": round(entry_ror, 1),
            "exit_value": round(exit_value, 2), "pnl": round(pnl, 2),
            "pnl_per_lot": round(pnl * NIFTY_LOT_SIZE, 1),
            "ror_pct": round(pnl / risk * 100, 1),
            "max_dd_pct": round(dd / risk * 100, 1),
            "outcome": "win" if pnl > 0 else "loss",
            "n_hold_days": len(vals),
        })
    return pd.DataFrame(out)


if __name__ == "__main__":
    eod = load_eod_table()
    print(f"[eod_credit_spread] EOD table: {len(eod):,} rows, "
          f"{eod['date'].nunique()} trading days, {eod['expiry'].nunique()} expiries")
    df = build_eod_spreads(eod, short_otm=0.01, wing_ce=0.015, wing_pe=0.03, fwd_days=3, dte_min=2, safe_dte=0)
    print(f"n={len(df)}")
    if len(df):
        print(df[["ror_pct", "pnl_per_lot", "max_dd_pct", "n_hold_days"]].describe(percentiles=[.1, .5, .9]).round(1))
        print("win_rate:", (df["outcome"] == "win").mean().round(3))
