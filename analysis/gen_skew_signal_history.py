"""Generate the historical IV-skew signal log: only the dates a signal actually fired
(IV-rich, no DTE window -- always the nearest available expiry), selling whichever side
(CE/PE) has the richer Black-Scholes implied vol at entry, held to day-5/expiry close
(no interim stop-loss -- every tested EOD stop level made results worse). Writes
locks/prod_sell_strategies/skew_signal_history.csv for the API (/prod2/skew_signal_history)
to serve.

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

SHORT_OTM, WING, FWD = 0.02, 0.03, 5
IV_RICH = 1.1   # no DTE window -- signal fires on any day IV is rich, using whatever expiry is nearest
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
cbar = {k: dict(zip(pd.to_datetime(g["date"]).values, zip(g["open"].values, g["close"].values)))
        for k, g in panel.groupby(["symbol", "opt_type", "expiry", "strike"], sort=False)}


def legseq(sym, ot, exp, strike, win):
    b = cbar.get((sym, ot, exp, strike))
    if not b:
        return None
    e = b.get(win[0])
    if not e or e[0] <= 0:
        return None
    closes = [b[d][1] if d in b else None for d in win]
    return e[0], closes


out = []
for (sym, e_date), day in panel.groupby(["symbol", "date"], sort=True):
    p = tpos.get(pd.Timestamp(e_date))
    if p is None or p == 0 or p + FWD - 1 >= len(tdays):
        continue
    t = pd.Timestamp(tdays[p - 1])
    ivr = IV.get((sym, t))
    if ivr is None or not (ivr >= IV_RICH):
        continue
    u = day["underlying"].iloc[0]
    exp = day["expiry"].min(); chain = day[day["expiry"] == exp]   # always the nearest expiry, no DTE filter
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
    if sce["open"] < 3 or spe["open"] < 3 or sce["vol"] < 50 or spe["vol"] < 50:
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

    vals, dd = [], 0.0
    for i in range(FWD):
        sc, lc = seq_s[1][i], seq_l[1][i]
        if sc is None or lc is None:
            continue
        value = sc - lc
        upnl = credit - value
        vals.append(value)
        dd = min(dd, upnl)
    if not vals:
        continue
    exit_value = vals[-1]; pnl = credit - exit_value
    out.append({
        "symbol": sym, "group": g2[sym], "signal_date": t.date().isoformat(), "entry_date": pd.Timestamp(e_date).date().isoformat(),
        "expiry": exp.date().isoformat(), "dte": dte, "underlying": round(u, 1), "iv_ratio": round(float(ivr), 2),
        "side": side, "ce_iv": round(ce_iv, 3), "pe_iv": round(pe_iv, 3), "skew": round(skew, 3),
        "short_strike": float(short_row["strike"]), "long_strike": float(long_row["strike"]),
        "sell_premium": round(seq_s[0], 2), "buy_premium": round(seq_l[0], 2),
        "credit": round(credit, 2), "max_risk": round(risk, 2), "exit_value": round(exit_value, 2),
        "pnl": round(pnl, 2), "ror_pct": round(pnl / risk * 100, 1), "max_dd_pct": round(dd / risk * 100, 1),
        "outcome": "win" if pnl > 0 else "loss",
    })

df = pd.DataFrame(out).sort_values("entry_date")
outdir = ROOT / "locks" / "prod_sell_strategies"; outdir.mkdir(parents=True, exist_ok=True)
df.to_csv(outdir / "skew_signal_history.csv", index=False)
print(f"wrote {len(df)} signals -> {outdir / 'skew_signal_history.csv'}")
print(df.groupby("group").agg(n=("pnl", "size"), win=("outcome", lambda s: (s == "win").mean()),
                              ev_ror=("ror_pct", "mean"), median_ror=("ror_pct", "median"),
                              worst_ror=("ror_pct", "min"), worst_dd=("max_dd_pct", "min")).round(1).to_string())
