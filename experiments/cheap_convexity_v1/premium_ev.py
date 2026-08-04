"""Validate cheap_convexity_v1 picks on REAL option premiums (the profit gold-standard).

For each pick in results/picks.csv, simulate an ATM straddle on the actual F&O bhavcopy:
  entry = close of day t   (ATM strike, nearest expiry >= t+WINDOW, premium = CE+PE)
  exit  = close of day t+WINDOW (same strike+expiry)
  straddle_return = (exit_premium - entry_premium) / entry_premium
Aggregate mean return per selector (cheap_convexity vs atm_iv_baseline). This is the premium-adjusted EV
that decides whether out-moving `atm_iv` actually converts to profit (mover-precision did not).

    set PYTHONPATH=src && python experiments/cheap_convexity_v1/premium_ev.py

Heavier: loads one FO bhavcopy per entry/exit date (cached). Needs data/raw/derivatives_bhavcopy.
Read-only; PROD untouched. (Held-to-horizon close exit; exit-at-peak is a future refinement.)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "analysis"))
from koscine3.data.sources import load_market_data       # noqa: E402
from options_bhavcopy import load_bhavcopy                # noqa: E402

WINDOW = 5
_BC: dict = {}


def bc(date: pd.Timestamp) -> pd.DataFrame:
    k = date.normalize()
    if k not in _BC:
        try:
            _BC[k] = load_bhavcopy(k, kinds=("STKOPT",))
        except Exception:  # noqa: BLE001
            _BC[k] = pd.DataFrame()
    return _BC[k]


def atm_straddle(date: pd.Timestamp, sym: str):
    """Return (strike, expiry, premium=CE+PE) for the ATM straddle on `date`, nearest expiry >= date+WINDOW."""
    df = bc(date)
    if df.empty:
        return None
    ch = df[df.symbol.eq(sym)]
    if ch.empty:
        return None
    exps = sorted(pd.to_datetime(ch.expiry.dropna().unique()))
    nxt = [e for e in exps if e >= date + pd.Timedelta(days=WINDOW)]
    exp = nxt[0] if nxt else (exps[-1] if exps else None)
    if exp is None:
        return None
    ce = ch[(ch.expiry.eq(exp)) & ch.opt_type.eq("CE")]
    pe = ch[(ch.expiry.eq(exp)) & ch.opt_type.eq("PE")]
    common = sorted(set(ce.strike) & set(pe.strike))
    if not common:
        return None
    und = ch.underlying.dropna()
    if len(und):
        u = float(und.iloc[0])
    else:                                                  # old format: ATM via put-call parity (CE~PE)
        mrg = ce.merge(pe, on="strike", suffixes=("_c", "_p"))
        u = float(mrg.iloc[(mrg.close_c - mrg.close_p).abs().to_numpy().argmin()].strike)
    atm = min(common, key=lambda s: abs(s - u))
    cp = float(ce[ce.strike.eq(atm)].close.iloc[0]) + float(pe[pe.strike.eq(atm)].close.iloc[0])
    return atm, exp, cp


def exit_premium(date: pd.Timestamp, sym: str, strike: float, expiry) -> "float | None":
    df = bc(date)
    if df.empty:
        return None
    ch = df[df.symbol.eq(sym) & df.expiry.eq(expiry) & df.strike.eq(strike)]
    ce = ch[ch.opt_type.eq("CE")]; pe = ch[ch.opt_type.eq("PE")]
    if ce.empty or pe.empty:
        return None
    return float(ce.close.iloc[0]) + float(pe.close.iloc[0])


def main():
    picks = pd.read_csv(HERE / "results" / "picks.csv", parse_dates=["date"])
    cal = np.sort(pd.to_datetime(load_market_data(columns=["date"]).date.unique()))
    pos = {d: i for i, d in enumerate(cal)}

    def exit_date(d):
        i = pos.get(pd.Timestamp(d).normalize())
        return cal[i + WINDOW] if i is not None and i + WINDOW < len(cal) else None

    rows = []
    for r in picks.itertuples(index=False):
        ed = exit_date(r.date)
        if ed is None:
            continue
        ent = atm_straddle(pd.Timestamp(r.date), r.symbol)
        if not ent or ent[2] <= 0:
            continue
        strike, expiry, entry_prem = ent
        xp = exit_premium(pd.Timestamp(ed), r.symbol, strike, expiry)
        if xp is None:
            continue
        rows.append({"selector": r.selector, "date": r.date, "symbol": r.symbol,
                     "entry_prem": entry_prem, "exit_prem": xp, "ret": xp / entry_prem - 1.0})
    res = pd.DataFrame(rows)
    if res.empty:
        print("no priced picks — check data/raw/derivatives_bhavcopy coverage for the book dates")
        return
    print(f"priced picks: {len(res)} / {len(picks)}\n")
    summ = {}
    for name, d in res.groupby("selector"):
        win = float((d.ret > 0).mean())
        summ[name] = {"n": int(len(d)), "mean_straddle_return_pct": round(float(d.ret.mean()) * 100, 2),
                      "median_return_pct": round(float(d.ret.median()) * 100, 2), "win_rate": round(win, 3)}
        print(f"{name:18s} n={len(d):5d}  mean straddle EV = {d.ret.mean()*100:+6.2f}%  median {d.ret.median()*100:+6.2f}%  win {win:.3f}")
    res.to_csv(HERE / "results" / "premium_ev_trades.csv", index=False)
    (HERE / "results" / "premium_ev.json").write_text(pd.Series(summ).to_json(indent=2))
    print("\nsaved -> results/premium_ev.json , premium_ev_trades.csv")
    print("If cheap_convexity mean EV > atm_iv_baseline (and > 0 net of ~1-2% costs) => real cheap-convexity edge.")


if __name__ == "__main__":
    main()
