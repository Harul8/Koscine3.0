"""Generate the historical Sell-Strategy signal log: only the dates a condor signal actually
fired (IV-rich, no DTE window -- always the nearest available expiry), with entry / exit / PnL /
max-drawdown per trade. Writes locks/prod_sell_strategies/signal_history.csv for the API
(/prod2/sell_signal_history) to serve.

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

SHORT_OTM, WING, FWD = 0.02, 0.03, 5
IV_RICH = 1.1   # no DTE window -- signal fires on any day IV is rich, using whatever expiry is nearest
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
    return e[0], closes            # entry open, list of daily closes

out = []
for (sym, e_date), day in panel.groupby(["symbol", "date"], sort=True):
    p = tpos.get(pd.Timestamp(e_date))
    if p is None or p == 0 or p + FWD - 1 >= len(tdays):
        continue
    t = pd.Timestamp(tdays[p - 1])
    ivr = IV.get((sym, t))
    if ivr is None or not (ivr >= IV_RICH):        # signal = IV rich at t
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
    seqs = {k: legseq(sym, ot, exp, float(r["strike"]), win)
            for k, (r, ot) in {"sc": (sce, "CE"), "lc": (lce, "CE"), "sp": (spe, "PE"), "lp": (lpe, "PE")}.items()}
    if any(v is None for v in seqs.values()):
        continue
    credit = (seqs["sc"][0] + seqs["sp"][0]) - (seqs["lc"][0] + seqs["lp"][0])
    width = max(float(lce["strike"] - sce["strike"]), float(spe["strike"] - lpe["strike"]))
    risk = width - credit
    if credit <= 0 or risk <= 0:
        continue
    # daily mark-to-close: cost to close the condor each day; unrealized pnl = credit - value
    vals, dd = [], 0.0
    for i in range(FWD):
        legc = [seqs[k][1][i] for k in ("sc", "lc", "sp", "lp")]
        if any(c is None for c in legc):
            continue
        value = (legc[0] - legc[1]) + (legc[2] - legc[3])
        upnl = credit - value
        vals.append((upnl, value))
        dd = min(dd, upnl)
    if not vals:
        continue
    exit_value = vals[-1][1]; pnl = credit - exit_value
    out.append({
        "symbol": sym, "group": g2[sym], "signal_date": t.date().isoformat(), "entry_date": pd.Timestamp(e_date).date().isoformat(),
        "expiry": exp.date().isoformat(), "dte": dte, "underlying": round(u, 1), "iv_ratio": round(float(ivr), 2),
        "short_ce": float(sce["strike"]), "long_ce": float(lce["strike"]), "short_pe": float(spe["strike"]), "long_pe": float(lpe["strike"]),
        "sell_premium": round(seqs["sc"][0] + seqs["sp"][0], 2), "buy_premium": round(seqs["lc"][0] + seqs["lp"][0], 2),
        "credit": round(credit, 2), "max_risk": round(risk, 2), "exit_value": round(exit_value, 2),
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
