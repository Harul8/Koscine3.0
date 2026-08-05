"""Generate the historical Sell-Strategy signal log: only the dates a condor signal actually
fired (entry-time credit/max_risk > MIN_ENTRY_ROR, DTE >= DTE_MIN), with entry / exit / PnL /
max-drawdown per trade. Writes locks/prod_sell_strategies/signal_history.csv for the API
(/prod2/sell_signal_history) to serve.

The gating criterion is MIN_ENTRY_ROR: the entry-time credit/max_risk ratio (the theoretical
max-profit/max-risk of the SETUP, knowable before the trade, unlike the realized ror_pct column
which is the ex-post outcome and can't be gated on in advance). IV-richness is recorded for
context but is no longer a hard filter.

One safety rule is still non-negotiable (NSE physical-settlement / delivery-margin avoidance --
ITM margin ramps 10%/25%/45%/70%/100%+ of contract value starting E-4, i.e. 4 trading days
before expiry):
  1. Entry requires DTE >= DTE_MIN (never enter close enough to expiry to risk being caught
     in the E-4 ramp before the position can be closed).
  2. Forced exit if the position is still open when DTE drops to <= SAFE_DTE, regardless of
     the normal FWD-day hold or the 50%-max-profit target.
Liquidity guard applies on EVERY day of the hold (not just entry): a day where either leg's
traded volume is too thin is treated as unusable (skipped) rather than marked at a possibly
stale/illiquid EOD print -- this is what a KAYNES trade (Jul 2026) exposed: an interim mark of
104.45 against a wing width of exactly 100, which is mathematically impossible at real
settlement (a capped condor can never owe more than its width) and was traced to a thin-volume
print, not a real loss. A [0, width] clamp on the daily mark backstops this.

Pipeline:
  1. python analysis/build_sell_panel.py 2024-08-01 2026-08-05 panel.parquet   # cache the A/B option panel
  2. python analysis/gen_sell_signal_history.py panel.parquet                   # -> locks/prod_sell_strategies/signal_history.csv
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(r"C:\Users\rahul\Koscine 3.0")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))
from koscine3.data.sources import load_market_data  # noqa: E402
from koscine3.largemove.mover_v2 import LOCK_V2  # noqa: E402

SHORT_OTM, WING, FWD, SAFE_DTE = 0.02, 0.03, 5, 4
DTE_MIN, MIN_ENTRY_ROR, MIN_VOL = 9, 150.0, 50
panel = pd.read_parquet(sys.argv[1])
panel["date"] = pd.to_datetime(panel["date"]); panel["expiry"] = pd.to_datetime(panel["expiry"])
g2 = {s: g for g, syms in json.loads((LOCK_V2 / "universe_groups.json").read_text()).items() for s in syms}

mk = load_market_data(columns=["date", "symbol", "atm_iv"])
mk["date"] = pd.to_datetime(mk["date"]); mk["symbol"] = mk["symbol"].astype(str); mk = mk.sort_values(["symbol", "date"])
mk["iv_ratio"] = mk.groupby("symbol")["atm_iv"].transform(lambda s: s / s.rolling(252, min_periods=60).median())
IV = mk.set_index(["symbol", "date"])["iv_ratio"].to_dict()
tdays = np.array(sorted(mk["date"].unique())); tpos = {d: k for k, d in enumerate(tdays)}
cbar = {k: dict(zip(pd.to_datetime(g["date"]).values, zip(g["open"].values, g["close"].values, g["vol"].values)))
        for k, g in panel.groupby(["symbol", "opt_type", "expiry", "strike"], sort=False)}

def legseq(sym, ot, exp, strike, win):
    b = cbar.get((sym, ot, exp, strike))
    if not b:
        return None
    e = b.get(win[0])
    if not e or e[0] <= 0:
        return None
    bars = [b.get(d) for d in win]     # (open, close, vol) or None per day
    return e[0], bars                  # entry open, list of daily (open, close, vol)

out = []
for (sym, e_date), day in panel.groupby(["symbol", "date"], sort=True):
    p = tpos.get(pd.Timestamp(e_date))
    if p is None or p == 0 or p + FWD - 1 >= len(tdays):
        continue
    t = pd.Timestamp(tdays[p - 1])
    ivr = IV.get((sym, t))                          # recorded for context, no longer a hard gate
    u = day["underlying"].iloc[0]
    exp = day["expiry"].min(); chain = day[day["expiry"] == exp]   # always the nearest expiry
    dte = int((exp - pd.Timestamp(e_date)).days)
    if dte < DTE_MIN:                              # delivery-margin safety floor
        continue
    win = [pd.Timestamp(x) for x in tdays[p: p + FWD]]
    def pick(ot, tgt):
        c = chain[chain["opt_type"] == ot]
        if c.empty:
            return None
        return c.iloc[(c["strike"] - tgt).abs().argmin()]
    sce, lce = pick("CE", u * (1 + SHORT_OTM)), pick("CE", u * (1 + SHORT_OTM + WING))
    spe, lpe = pick("PE", u * (1 - SHORT_OTM)), pick("PE", u * (1 - SHORT_OTM - WING))
    if any(x is None for x in (sce, lce, spe, lpe)):
        continue
    if sce["open"] < 3 or spe["open"] < 3 or sce["vol"] < MIN_VOL or spe["vol"] < MIN_VOL:
        continue
    seqs = {k: legseq(sym, ot, exp, float(r["strike"]), win)
            for k, (r, ot) in {"sc": (sce, "CE"), "lc": (lce, "CE"), "sp": (spe, "PE"), "lp": (lpe, "PE")}.items()}
    if any(v is None for v in seqs.values()):
        continue
    credit = (seqs["sc"][0] + seqs["sp"][0]) - (seqs["lc"][0] + seqs["lp"][0])
    width = max(float(lce["strike"] - sce["strike"]), float(spe["strike"] - lpe["strike"]))
    risk = width - credit
    if credit <= 0 or risk <= 0:
        continue
    entry_ror = credit / risk * 100
    if entry_ror <= MIN_ENTRY_ROR:                  # THE gating criterion: entry-time credit/max_risk
        continue
    # daily mark-to-close: cost to close the condor each day; unrealized pnl = credit - value.
    # Any day where a leg's volume is too thin is skipped (not marked at a stale/illiquid print);
    # forced exit once DTE drops to <=SAFE_DTE (ahead of the E-4 delivery-margin ramp).
    vals, dd = [], 0.0
    for i in range(FWD):
        legbars = [seqs[k][1][i] for k in ("sc", "lc", "sp", "lp")]
        if not any(b is None or b[2] < MIN_VOL for b in legbars):
            value = (legbars[0][1] - legbars[1][1]) + (legbars[2][1] - legbars[3][1])
            value = max(0.0, min(width, value))   # a condor's combined value can never legitimately
                                                    # exceed the wing width (both sides can't be maxed
                                                    # at once) -- clamp out stale-print artifacts
            upnl = credit - value
            vals.append((upnl, value))
            dd = min(dd, upnl)
        if (exp - win[i]).days <= SAFE_DTE:   # forced exit before the E-4 delivery-margin ramp
            break
    if not vals:
        continue
    exit_value = vals[-1][1]; pnl = credit - exit_value
    out.append({
        "symbol": sym, "group": g2[sym], "signal_date": t.date().isoformat(), "entry_date": pd.Timestamp(e_date).date().isoformat(),
        "expiry": exp.date().isoformat(), "dte": dte, "underlying": round(u, 1),
        "iv_ratio": round(float(ivr), 2) if ivr is not None and pd.notna(ivr) else None,
        "short_ce": float(sce["strike"]), "long_ce": float(lce["strike"]), "short_pe": float(spe["strike"]), "long_pe": float(lpe["strike"]),
        "sell_premium": round(seqs["sc"][0] + seqs["sp"][0], 2), "buy_premium": round(seqs["lc"][0] + seqs["lp"][0], 2),
        "credit": round(credit, 2), "max_risk": round(risk, 2), "entry_ror_pct": round(entry_ror, 1),
        "exit_value": round(exit_value, 2),
        "pnl": round(pnl, 2), "ror_pct": round(pnl / risk * 100, 1), "max_dd_pct": round(dd / risk * 100, 1),
        "outcome": "win" if pnl > 0 else "loss",
    })

df = pd.DataFrame(out).sort_values("entry_date")
outdir = ROOT / "locks" / "prod_sell_strategies"; outdir.mkdir(parents=True, exist_ok=True)
df.to_csv(outdir / "signal_history.csv", index=False)
print(f"wrote {len(df)} signals -> {outdir / 'signal_history.csv'}")
print(df.groupby("group").agg(n=("pnl", "size"), win=("outcome", lambda s: (s == "win").mean()),
                              ev_ror=("ror_pct", "mean"), median_ror=("ror_pct", "median"),
                              worst_ror=("ror_pct", "min"), worst_dd=("max_dd_pct", "min")).round(1).to_string())
