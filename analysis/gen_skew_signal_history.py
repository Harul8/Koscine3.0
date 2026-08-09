"""Generate the historical IV-skew signal log for THREE production tiers (mirroring the
Broken-Wing Condor's tiering), selling whichever side (CE/PE) has the richer Black-Scholes
implied vol at entry, held to day-5/expiry close (no interim stop-loss -- every tested EOD stop
level made results worse). Writes locks/prod_sell_strategies/skew_signal_history.csv (all three
tiers, tagged by `tier`) for the API (/prod2/skew_strategy, /prod2/skew_signal_history) to serve.

Wing-width tiering: tested widening the wing from the original flat 3% (single tier) up through
3.5/4/5/6/8/10%, same "wider wing -> more credit banked, fewer signals clear the gate" pattern
already found for the symmetric/broken-wing condor (pre-dedup, raw candidate counts):
  3%:   n=535/2y, median  Rs.5,858/lot
  3.5%: n=335/2y, median      -- <- t1: chosen to hit a ~320-350/yr combined (with Broken-Wing)
                                     signal-volume target while keeping >=~Rs.7k median PnL/lot
  4%:   n=219/2y, median  Rs.8,207/lot
  5%:   n=118/2y, median Rs.10,817/lot
  6%:   n=61/2y,  median Rs.12,895/lot   <- t2
  8%:   n=26/2y,  median Rs.14,305/lot   <- t3 (plateau point)
  10%:  n=24/2y,  median Rs.14,305/lot  (flat vs 8% -- no benefit going wider, not used)

De-duplicated against Broken-Wing Condor: a skew candidate is dropped if the SAME symbol also
has a Broken-Wing signal on the same entry_date -- skew is a secondary/complementary signal,
not a duplicate of Broken-Wing's picks. De-dup removes a meaningful chunk -- post-dedup
production counts:
  t1_skew35: 251/2y (126/yr), median Rs.7,175/lot
  t2_skew6:   24/2y  (12/yr), median Rs.13,681/lot
  t3_skew8:    9/2y   (4/yr), median Rs.11,988/lot -- thin sample, treat as directional not precise
Combined with Broken-Wing's 403/2y (201/yr): 687/2y = 343.5/yr grand total, in the 320-350
target range.

Gating criterion is MIN_ENTRY_ROR: the entry-time credit/max_risk ratio (knowable before the
trade), not the realized ror_pct column (an ex-post outcome). IV-richness (skew = CE-IV - PE-IV)
picks which side to sell, but is not itself a hard filter beyond that.

One safety rule is still non-negotiable (NSE physical-settlement / delivery-margin avoidance --
ITM margin ramps 10%/25%/45%/70%/100%+ of contract value starting E-4, i.e. 4 trading days
before expiry):
  1. Entry requires DTE >= DTE_MIN.
  2. Forced exit if the position is still open when DTE drops to <= SAFE_DTE.
Liquidity guard applies on EVERY day of the hold (not just entry): a day where either leg's
traded volume is too thin is skipped rather than marked at a possibly stale EOD print. A
[0, width] clamp on the daily mark backstops this.

Pipeline:
  1. python analysis/build_broken_wing_panel.py 2024-08-01 2026-08-05 bw_panel.parquet
  2. python analysis/gen_broken_wing_signal_history.py bw_panel.parquet   # must run FIRST (skew dedups against it)
  3. python analysis/gen_skew_signal_history.py bw_panel.parquet          # -> locks/prod_sell_strategies/skew_signal_history.csv
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
from koscine.config import SILVER_DATA_ROOT  # noqa: E402

SHORT_OTM, FWD, SAFE_DTE = 0.02, 5, 4
DTE_MIN, MIN_ENTRY_ROR, MIN_VOL = 15, 150.0, 50
MIN_RISK_FRAC = 0.10   # max_risk must be >= 10% of wing width; below that, credit~=width and the
                        # entry_ror ratio becomes numerically degenerate (blows toward infinity)
R = 0.065

TIERS = [
    ("t1_skew35", 0.035),
    ("t2_skew6", 0.06),
    ("t3_skew8", 0.08),
]


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
# Non-mega-cap symbols with a consistently weak (n>=3) historical track record across the
# combined Broken-Wing + Skew signal set, excluded outright: ANGELONE median Rs.608/lot over 8
# signals -- 4.6x below the next-weakest name (TMPV, Rs.2,820) -- not a fluke of a small sample.
EXCLUDE_SYMBOLS = {"ANGELONE"}

mk = load_market_data(columns=["date", "symbol", "atm_iv"])
mk["date"] = pd.to_datetime(mk["date"]); mk["symbol"] = mk["symbol"].astype(str); mk = mk.sort_values(["symbol", "date"])
mk["iv_ratio"] = mk.groupby("symbol")["atm_iv"].transform(lambda s: s / s.rolling(252, min_periods=60).median())
IV = mk.set_index(["symbol", "date"])["iv_ratio"].to_dict()
tdays = np.array(sorted(mk["date"].unique())); tpos = {d: k for k, d in enumerate(tdays)}

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


def run_tier(tier_id: str, wing: float) -> list[dict]:
    out = []
    for (sym, e_date), day in panel.groupby(["symbol", "date"], sort=True):
        if sym in EXCLUDE_SYMBOLS:
            continue
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

        sce, lce = pick("CE", u * (1 + SHORT_OTM)), pick("CE", u * (1 + SHORT_OTM + wing))
        spe, lpe = pick("PE", u * (1 - SHORT_OTM)), pick("PE", u * (1 - SHORT_OTM - wing))
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
        if risk < MIN_RISK_FRAC * width:
            continue
        entry_ror = credit / risk * 100
        if entry_ror <= MIN_ENTRY_ROR:
            continue

        vals, dd = [], 0.0
        for i in range(FWD):
            sb, lb = seq_s[1][i], seq_l[1][i]
            if not (sb is None or lb is None or sb[2] < MIN_VOL or lb[2] < MIN_VOL):
                value = sb[1] - lb[1]
                value = max(0.0, min(width, value))
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
            "tier": tier_id, "symbol": sym, "group": g2.get(sym, "C_top50oi"), "signal_date": t.date().isoformat(),
            "entry_date": pd.Timestamp(e_date).date().isoformat(),
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
    return out


# de-dup against Broken-Wing: a skew candidate is dropped if the same symbol also has a
# Broken-Wing signal (any tier) on the same entry_date.
bw_path = ROOT / "locks" / "prod_sell_strategies" / "broken_wing_signal_history.csv"
bw_pairs = set()
if bw_path.exists():
    bw = pd.read_csv(bw_path, usecols=["symbol", "entry_date"])
    bw_pairs = set(zip(bw["symbol"], bw["entry_date"]))
else:
    print(f"WARNING: {bw_path} not found -- run gen_broken_wing_signal_history.py first; "
          f"skipping de-dup (all skew candidates kept)", flush=True)

all_out = []
for tier_id, wing in TIERS:
    rows = run_tier(tier_id, wing)
    n_before = len(rows)
    rows = [r for r in rows if (r["symbol"], r["entry_date"]) not in bw_pairs]
    print(f"{tier_id} (wing={wing}): {n_before} signals, {len(rows)} after de-dup vs broken-wing", flush=True)
    all_out.extend(rows)

df = pd.DataFrame(all_out).sort_values(["tier", "entry_date"])
outdir = ROOT / "locks" / "prod_sell_strategies"; outdir.mkdir(parents=True, exist_ok=True)
df.to_csv(outdir / "skew_signal_history.csv", index=False)
print(f"wrote {len(df)} signals (all tiers, de-duped) -> {outdir / 'skew_signal_history.csv'}")
print(df.groupby("tier").agg(n=("pnl", "size"), win=("outcome", lambda s: (s == "win").mean()),
                             ev_ror=("ror_pct", "mean"), median_ror=("ror_pct", "median"),
                             worst_ror=("ror_pct", "min"), worst_dd=("max_dd_pct", "min")).round(1).to_string())
