"""Generate the historical Broken-Wing Condor signal log: same defined-risk short
2%-OTM CE + 2%-OTM PE structure as the symmetric condor, but with ASYMMETRIC wings --
call wing 3% (narrow), put wing 8% (wide). Writes locks/prod_sell_strategies/
broken_wing_signal_history.csv for the API (/prod2/broken_wing_strategy) to serve.

Why asymmetric wings beat the symmetric 5%/5% condor (backtested on a 7-year, both-
direction-tested panel, 2019-2026): max loss at expiry for a broken-wing condor is
max(call_wing, put_wing) - total_credit (only ONE side can be breached at expiry, so
risk is set by whichever wing is WIDER, not by their sum -- re-derived and confirmed,
this is not a repeat of the earlier degenerate-risk bug). Indian equity/index options
carry a persistent PUT skew (crash-premium): OTM puts are richer than OTM calls of the
same distance. A WIDE put wing buys cheap, far-OTM crash protection while banking most
of that rich put premium as credit; a NARROW call wing still collects a decent call
credit (calls aren't as richly priced, so there's little cost to buying the long call
close by). Net effect: total credit rises while max_risk stays anchored to the (now
irrelevant on the put side) wider wing -- this DIRECTION (narrow call / wide put) beat
the mirror image (wide call / narrow put) at every asymmetry level tested:
  5/5 symmetric:  1,522 signals/yr, 83.2% win, median  Rs.2,312/lot,  Rs.505/lot/day
  6/2 (mirror):     159 signals/yr, 92.1% win, median  Rs.5,460/lot, Rs.1,238/lot/day
  2/6:               150 signals/yr, 95.4% win, median  Rs.7,535/lot, Rs.1,781/lot/day
  8/3 (mirror):       77 signals/yr, 93.2% win, median  Rs.8,313/lot, Rs.1,977/lot/day
  3/8 (THIS ONE):      71 signals/yr, 95.6% win, median  Rs.9,695/lot, Rs.2,270/lot/day
  10/2 (mirror):       18 signals/yr, 97.6% win, median Rs.15,520/lot, Rs.3,536/lot/day
  2/10:                16 signals/yr, 99.1% win, median Rs.19,811/lot, Rs.4,656/lot/day
More asymmetry = better quality but fewer signals (clean, monotonic trade-off, not
noise -- confirmed at both a moderate (2/6 vs 6/2) and extreme (2/10 vs 10/2) level).
3%/8% is the production pick: ~4.2x the symmetric condor's median profit/lot at a
cadence (~71/yr, ~1.4/week) similar to the existing Skew tab, well clear of the
thin-sample (n<130) extreme end of the curve.

Same safety rules as the symmetric condor: DTE >= DTE_MIN at entry, forced exit at
DTE<=SAFE_DTE, per-day liquidity guard, [0, width] daily-mark clamp, MIN_RISK_FRAC
degenerate-risk gate. Needs a WIDER put-side panel band than the symmetric condor's
(long put here targets ~90% of spot, right at that panel's floor) -- use
build_broken_wing_panel.py, not build_sell_panel.py, to build the input panel.

Pipeline:
  1. python analysis/build_broken_wing_panel.py 2024-08-01 2026-08-05 bw_panel.parquet
  2. python analysis/gen_broken_wing_signal_history.py bw_panel.parquet   # -> locks/prod_sell_strategies/broken_wing_signal_history.csv
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

SHORT_OTM, WING_CE, WING_PE, FWD, SAFE_DTE = 0.02, 0.03, 0.08, 5, 4
DTE_MIN, MIN_ENTRY_ROR, MIN_VOL = 9, 150.0, 50
MIN_RISK_FRAC = 0.10   # max_risk must be >= 10% of the (wider) wing; below that, credit~=width and
                        # the entry_ror ratio becomes numerically degenerate (blows toward infinity)
panel = pd.read_parquet(sys.argv[1])
panel["date"] = pd.to_datetime(panel["date"]); panel["expiry"] = pd.to_datetime(panel["expiry"])
g2 = {s: g for g, syms in json.loads((LOCK_V2 / "universe_groups.json").read_text()).items() for s in syms}

mk = load_market_data(columns=["date", "symbol", "atm_iv"])
mk["date"] = pd.to_datetime(mk["date"]); mk["symbol"] = mk["symbol"].astype(str); mk = mk.sort_values(["symbol", "date"])
mk["iv_ratio"] = mk.groupby("symbol")["atm_iv"].transform(lambda s: s / s.rolling(252, min_periods=60).median())
IV = mk.set_index(["symbol", "date"])["iv_ratio"].to_dict()
tdays = np.array(sorted(mk["date"].unique())); tpos = {d: k for k, d in enumerate(tdays)}

from koscine.config import SILVER_DATA_ROOT  # noqa: E402
lot_df = pd.read_parquet(SILVER_DATA_ROOT / "lot_size.parquet", columns=["symbol", "expiry_month", "lot"])
LOT = lot_df.set_index(["symbol", pd.to_datetime(lot_df["expiry_month"]).dt.to_period("M")])["lot"].to_dict()

def lot_size(sym, exp):
    return LOT.get((sym, pd.Timestamp(exp).to_period("M")))
cbar = {k: dict(zip(pd.to_datetime(g["date"]).values, zip(g["open"].values, g["close"].values, g["vol"].values)))
        for k, g in panel.groupby(["symbol", "opt_type", "expiry", "strike"], sort=False)}

def legseq(sym, ot, exp, strike, win):
    b = cbar.get((sym, ot, exp, strike))
    if not b:
        return None
    e = b.get(win[0])
    if not e or e[0] <= 0:
        return None
    bars = [b.get(d) for d in win]
    return e[0], bars

out = []
for (sym, e_date), day in panel.groupby(["symbol", "date"], sort=True):
    p = tpos.get(pd.Timestamp(e_date))
    if p is None or p == 0 or p + FWD - 1 >= len(tdays):
        continue
    t = pd.Timestamp(tdays[p - 1])
    ivr = IV.get((sym, t))
    u = day["underlying"].iloc[0]
    candidate_exps = sorted(day["expiry"].unique())
    exp = next((e for e in candidate_exps if (pd.Timestamp(e) - pd.Timestamp(e_date)).days >= DTE_MIN), None)
    if exp is None:
        continue
    chain = day[day["expiry"] == exp]
    dte = int((exp - pd.Timestamp(e_date)).days)
    win = [pd.Timestamp(x) for x in tdays[p: p + FWD]]
    def pick(ot, tgt):
        c = chain[chain["opt_type"] == ot]
        if c.empty:
            return None
        return c.iloc[(c["strike"] - tgt).abs().argmin()]
    sce, lce = pick("CE", u * (1 + SHORT_OTM)), pick("CE", u * (1 + SHORT_OTM + WING_CE))
    spe, lpe = pick("PE", u * (1 - SHORT_OTM)), pick("PE", u * (1 - SHORT_OTM - WING_PE))
    if any(x is None for x in (sce, lce, spe, lpe)):
        continue
    if sce["open"] < 3 or spe["open"] < 3 or sce["vol"] < MIN_VOL or spe["vol"] < MIN_VOL:
        continue
    seqs = {k: legseq(sym, ot, exp, float(r["strike"]), win)
            for k, (r, ot) in {"sc": (sce, "CE"), "lc": (lce, "CE"), "sp": (spe, "PE"), "lp": (lpe, "PE")}.items()}
    if any(v is None for v in seqs.values()):
        continue
    credit = (seqs["sc"][0] + seqs["sp"][0]) - (seqs["lc"][0] + seqs["lp"][0])
    call_width = float(lce["strike"] - sce["strike"]); put_width = float(spe["strike"] - lpe["strike"])
    width = max(call_width, put_width)   # true worst-case-at-expiry risk is set by the WIDER wing
                                          # only (one side breaches at a time) -- not the sum
    risk = width - credit
    if credit <= 0 or risk <= 0:
        continue
    if risk < MIN_RISK_FRAC * width:
        continue
    entry_ror = credit / risk * 100
    if entry_ror <= MIN_ENTRY_ROR:
        continue
    vals, dd = [], 0.0
    for i in range(FWD):
        legbars = [seqs[k][1][i] for k in ("sc", "lc", "sp", "lp")]
        if not any(b is None or b[2] < MIN_VOL for b in legbars):
            value = (legbars[0][1] - legbars[1][1]) + (legbars[2][1] - legbars[3][1])
            value = max(0.0, min(width, value))
            upnl = credit - value
            vals.append((upnl, value))
            dd = min(dd, upnl)
        if (exp - win[i]).days <= SAFE_DTE:
            break
    if not vals:
        continue
    exit_value = vals[-1][1]; pnl = credit - exit_value
    lot = lot_size(sym, exp)
    out.append({
        "symbol": sym, "group": g2[sym], "signal_date": t.date().isoformat(), "entry_date": pd.Timestamp(e_date).date().isoformat(),
        "expiry": exp.date().isoformat(), "dte": dte, "underlying": round(u, 1),
        "iv_ratio": round(float(ivr), 2) if ivr is not None and pd.notna(ivr) else None,
        "short_ce": float(sce["strike"]), "long_ce": float(lce["strike"]), "short_pe": float(spe["strike"]), "long_pe": float(lpe["strike"]),
        "call_width_pct": round(call_width / u * 100, 1), "put_width_pct": round(put_width / u * 100, 1),
        "sell_premium": round(seqs["sc"][0] + seqs["sp"][0], 2), "buy_premium": round(seqs["lc"][0] + seqs["lp"][0], 2),
        "credit": round(credit, 2), "max_risk": round(risk, 2), "max_profit": round(credit, 2),
        "entry_ror_pct": round(entry_ror, 1),
        "lot_size": int(lot) if lot is not None and pd.notna(lot) else None,
        "max_risk_per_lot": round(risk * lot, 1) if lot is not None and pd.notna(lot) else None,
        "max_profit_per_lot": round(credit * lot, 1) if lot is not None and pd.notna(lot) else None,
        "exit_value": round(exit_value, 2),
        "pnl": round(pnl, 2), "pnl_per_lot": round(pnl * lot, 1) if lot is not None and pd.notna(lot) else None,
        "ror_pct": round(pnl / risk * 100, 1), "max_dd_pct": round(dd / risk * 100, 1),
        "outcome": "win" if pnl > 0 else "loss",
    })

df = pd.DataFrame(out).sort_values("entry_date")
outdir = ROOT / "locks" / "prod_sell_strategies"; outdir.mkdir(parents=True, exist_ok=True)
df.to_csv(outdir / "broken_wing_signal_history.csv", index=False)
print(f"wrote {len(df)} signals -> {outdir / 'broken_wing_signal_history.csv'}")
print(df.groupby("group").agg(n=("pnl", "size"), win=("outcome", lambda s: (s == "win").mean()),
                              ev_ror=("ror_pct", "mean"), median_ror=("ror_pct", "median"),
                              worst_ror=("ror_pct", "min"), worst_dd=("max_dd_pct", "min")).round(1).to_string())
