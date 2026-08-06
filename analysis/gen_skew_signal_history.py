"""Generate the historical IV-skew signal log: only the dates a signal actually fired
(entry-time credit/max_risk > MIN_ENTRY_ROR, DTE >= DTE_MIN), selling whichever side (CE/PE)
has the richer Black-Scholes implied vol at entry, held to day-5/expiry close (no interim
stop-loss -- every tested EOD stop level made results worse). Writes
locks/prod_sell_strategies/skew_signal_history.csv for the API (/prod2/skew_signal_history)
to serve.

The gating criterion is MIN_ENTRY_ROR: the entry-time credit/max_risk ratio (knowable before
the trade), not the realized ror_pct column (an ex-post outcome). IV-richness is recorded for
context but is no longer a hard filter.

One safety rule is still non-negotiable (NSE physical-settlement / delivery-margin avoidance --
ITM margin ramps 10%/25%/45%/70%/100%+ of contract value starting E-4, i.e. 4 trading days
before expiry):
  1. Entry requires DTE >= DTE_MIN.
  2. Forced exit if the position is still open when DTE drops to <= SAFE_DTE.
Liquidity guard applies on EVERY day of the hold (not just entry): a day where either leg's
traded volume is too thin is skipped rather than marked at a possibly stale EOD print. A
[0, width] clamp on the daily mark backstops this.

Pipeline:
  1. python analysis/build_sell_panel.py 2024-08-01 2026-08-05 panel.parquet   # cache the A/B option panel
  2. python analysis/gen_skew_signal_history.py panel.parquet                   # -> locks/prod_sell_strategies/skew_signal_history.csv
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq

ROOT = Path(r"C:\Users\rahul\Koscine 3.0")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))
from koscine3.data.sources import load_market_data  # noqa: E402
from koscine3.largemove.mover_v2 import LOCK_V2  # noqa: E402

SHORT_OTM, WING, FWD, SAFE_DTE = 0.02, 0.03, 5, 4
DTE_MIN, MIN_ENTRY_ROR, MIN_VOL = 15, 150.0, 50
MIN_RISK_FRAC = 0.10   # max_risk must be >= 10% of wing width; below that, credit~=width and the
                        # entry_ror ratio becomes numerically degenerate (blows toward infinity)
R = 0.065


def bs_price(S, K, T, sigma, r, is_call):
    if sigma <= 0 or T <= 0:
        return max(0.0, (S - K) if is_call else (K - S))
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if is_call:
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def implied_vol(price, S, K, T, r, is_call):
    intrinsic = max(0.0, (S - K) if is_call else (K - S))
    if price <= intrinsic + 1e-6 or T <= 0:
        return None
    try:
        return brentq(lambda s: bs_price(S, K, T, s, r, is_call) - price, 1e-4, 6.0, xtol=1e-4)
    except ValueError:
        return None


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
    bars = [b.get(d) for d in win]     # (open, close, vol) or None per day
    return e[0], bars


out = []
for (sym, e_date), day in panel.groupby(["symbol", "date"], sort=True):
    p = tpos.get(pd.Timestamp(e_date))
    if p is None or p == 0 or p + FWD - 1 >= len(tdays):
        continue
    t = pd.Timestamp(tdays[p - 1])
    ivr = IV.get((sym, t))                          # recorded for context, no longer a hard gate
    u = day["underlying"].iloc[0]
    # roll forward: use the EARLIEST expiry that already clears the DTE safety floor, rather than
    # the nearest calendar expiry -- avoids the dead zone in the days right before each expiry.
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

    sce, lce = pick("CE", u * (1 + SHORT_OTM)), pick("CE", u * (1 + SHORT_OTM + WING))
    spe, lpe = pick("PE", u * (1 - SHORT_OTM)), pick("PE", u * (1 - SHORT_OTM - WING))
    if any(x is None for x in (sce, lce, spe, lpe)):
        continue
    if sce["open"] < 3 or spe["open"] < 3 or sce["vol"] < MIN_VOL or spe["vol"] < MIN_VOL:
        continue

    T = dte / 365.0
    ce_iv = implied_vol(float(sce["open"]), u, float(sce["strike"]), T, R, True)
    pe_iv = implied_vol(float(spe["open"]), u, float(spe["strike"]), T, R, False)
    if ce_iv is None or pe_iv is None:
        continue
    skew = ce_iv - pe_iv
    side = "CE" if skew > 0 else "PE"
    short_row, long_row, ot = (sce, lce, "CE") if side == "CE" else (spe, lpe, "PE")

    seq_s = legseq(sym, ot, exp, float(short_row["strike"]), win)
    seq_l = legseq(sym, ot, exp, float(long_row["strike"]), win)
    if seq_s is None or seq_l is None:
        continue
    credit = seq_s[0] - seq_l[0]
    width = abs(float(long_row["strike"]) - float(short_row["strike"]))
    risk = width - credit
    if credit <= 0 or risk <= 0:
        continue
    if risk < MIN_RISK_FRAC * width:                 # exclude numerically degenerate near-zero-risk
        continue                                      # spreads (credit~=width -> ROR blows toward infinity)
    entry_ror = credit / risk * 100
    if entry_ror <= MIN_ENTRY_ROR:                  # THE gating criterion: entry-time credit/max_risk
        continue

    # daily mark-to-close, skipping thin-volume days; forced exit before the E-4 delivery-margin ramp.
    vals, dd = [], 0.0
    for i in range(FWD):
        sb, lb = seq_s[1][i], seq_l[1][i]
        if not (sb is None or lb is None or sb[2] < MIN_VOL or lb[2] < MIN_VOL):
            value = sb[1] - lb[1]
            value = max(0.0, min(width, value))   # a credit spread's value can never legitimately
                                                    # exceed its width -- clamp out stale-print artifacts
            upnl = credit - value
            vals.append(value)
            dd = min(dd, upnl)
        if (exp - win[i]).days <= SAFE_DTE:
            break
    if not vals:
        continue
    exit_value = vals[-1]; pnl = credit - exit_value
    lot = lot_size(sym, exp)
    out.append({
        "symbol": sym, "group": g2[sym], "signal_date": t.date().isoformat(), "entry_date": pd.Timestamp(e_date).date().isoformat(),
        "expiry": exp.date().isoformat(), "dte": dte, "underlying": round(u, 1),
        "iv_ratio": round(float(ivr), 2) if ivr is not None and pd.notna(ivr) else None,
        "side": side, "ce_iv": round(ce_iv, 3), "pe_iv": round(pe_iv, 3), "skew": round(skew, 3),
        "short_strike": float(short_row["strike"]), "long_strike": float(long_row["strike"]),
        "sell_premium": round(seq_s[0], 2), "buy_premium": round(seq_l[0], 2),
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
df.to_csv(outdir / "skew_signal_history.csv", index=False)
print(f"wrote {len(df)} signals -> {outdir / 'skew_signal_history.csv'}")
print(df.groupby("group").agg(n=("pnl", "size"), win=("outcome", lambda s: (s == "win").mean()),
                              ev_ror=("ror_pct", "mean"), median_ror=("ror_pct", "median"),
                              worst_ror=("ror_pct", "min"), worst_dd=("max_dd_pct", "min")).round(1).to_string())
