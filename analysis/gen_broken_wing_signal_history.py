"""Generate the historical Broken-Wing Condor signal log for THREE production tiers, all
narrow-call/wide-put asymmetric condors at increasing asymmetry: sell ~2%-OTM CE+PE like the
symmetric condor, buy asymmetric wings. Writes locks/prod_sell_strategies/
broken_wing_signal_history.csv (all three tiers, tagged by `tier`) for the API
(/prod2/broken_wing_strategy, /prod2/broken_wing_signal_history) to serve.

Why asymmetric wings beat the symmetric 5%/5% condor (backtested on a 7-year, both-direction-
tested panel, 2019-2026): max loss at expiry for a broken-wing condor is max(call_wing,
put_wing) - total_credit (only ONE side can be breached at expiry, so risk is set by whichever
wing is WIDER, not their sum -- re-derived and confirmed, not a repeat of the earlier
degenerate-risk bug). Indian equity/index options carry a persistent PUT skew (crash premium):
OTM puts are richer than OTM calls of the same distance. A WIDE put wing buys cheap, far-OTM
crash protection while banking most of that rich put premium as credit; a NARROW call wing
still collects a decent call credit (calls aren't as richly priced, so there's little cost to
buying the long call close by). This direction (narrow call / wide put) beat the mirror image
at every asymmetry level tested.

Three tiers, all live in production simultaneously -- more asymmetry = better quality but
fewer signals (clean, monotonic trade-off, confirmed at both moderate and extreme levels).

Universe: UNION of the static A/B universe_groups.json (65 symbols) and each day's dynamic
top-50-by-OI-in-lots stocks (see build_broken_wing_panel.py for why -- top-50-only actually
REDUCED both frequency and mega-cap share vs the static universe).

Entry gate is STRATIFIED, not a single threshold: mega-caps (A_mcap30) need entry_ror >
MCAP_MIN_ROR (120%), everyone else needs > OTHER_MIN_ROR (150%, unchanged). Why: mega-caps are
structurally lower-IV (steadier, less wing-breach risk -- itself a quality trait for a defined-
risk seller) so they rarely clear a uniform 150% bar; relaxing it specifically for them lifted
mega-cap share from ~13% to 13-28%/tier and pushed unique-symbol diversity from ~28-40 to
17-68/tier, at the cost of a modest 150->~200/yr frequency increase and no win-rate loss
(98-100% in backtest). A uniform lower threshold for everyone was tried first and overshot to
~314 signals/yr -- too far from the 150-200/yr, 3-4/week target.

KNOWN GAP: does not exclude F&O-ban-listed stocks (no ban-list data source in this codebase).
Cross-check live picks against NSE's published ban list before trading.

Same safety rules as the symmetric condor: DTE >= DTE_MIN at entry, forced exit at
DTE<=SAFE_DTE, per-day liquidity guard, [0, width] daily-mark clamp, MIN_RISK_FRAC
degenerate-risk gate. Needs a WIDER put-side panel band than the symmetric condor's (t3's long
put targets ~88% of spot) -- use build_broken_wing_panel.py, not build_sell_panel.py.

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
from koscine.config import SILVER_DATA_ROOT  # noqa: E402
from koscine import liquid_universe  # noqa: E402

# Backtest universe = fixed Nifty50 (see koscine/liquid_universe.py's module docstring for why
# it's fixed rather than the live 50+15-daily-tail the API endpoints use).
BACKTEST_UNIVERSE = liquid_universe.backtest_universe()

SHORT_OTM, FWD, SAFE_DTE = 0.02, 5, 4
DTE_MIN, MIN_VOL = 9, 50
MCAP_MIN_ROR, OTHER_MIN_ROR = 108.0, 138.0   # stratified entry gate -- see module docstring.
# Lowered from 120/150 -> 108/138 (tuned UP in frequency toward ~80-100/yr; was firing ~65/yr at
# 120/150) so Broken-Wing, Skew, and Condor land at comparable ~80-100/yr each (~240-300/yr
# combined) in the merged Sell Signals view, rather than Condor's much higher natural firing
# rate drowning the other two out whenever they weren't also firing that same day. Calibrated
# empirically against the 2024-08..2026-08 panel (entry-ROR distribution of every structurally-
# valid candidate per tier, with the candidate->final shrinkage ratio measured from the real
# 120/150 run) -- same data-driven-threshold approach as pipeline/labels.py's _calibrate_k.
MIN_RISK_FRAC = 0.10   # max_risk must be >= 10% of the (wider) wing; below that, credit~=width
                        # and the entry_ror ratio becomes numerically degenerate

TIERS = [
    ("t1_2x6", 0.02, 0.06),
    ("t2_3x8", 0.03, 0.08),
    ("t3_2x10", 0.02, 0.10),
]

# Optional partial-range regen: `python gen_broken_wing_signal_history.py bw_panel.parquet
# 2026-07-01 2026-08-31` only recomputes entries in [ENTRY_DATE_MIN, ENTRY_DATE_MAX] and merges
# the result into the existing CSV (rows outside that range are preserved untouched).
ENTRY_DATE_MIN = sys.argv[2] if len(sys.argv) > 2 else None
ENTRY_DATE_MAX = sys.argv[3] if len(sys.argv) > 3 else None
panel = pd.read_parquet(sys.argv[1])
panel["date"] = pd.to_datetime(panel["date"]); panel["expiry"] = pd.to_datetime(panel["expiry"])
g2 = {s: g for g, syms in json.loads((LOCK_V2 / "universe_groups.json").read_text()).items() for s in syms}
A_MCAP = set(json.loads((LOCK_V2 / "universe_groups.json").read_text()).get("A_mcap30", []))
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
    bars = [b.get(d) for d in win]
    return e[0], bars


def run_tier(tier_id: str, wing_ce: float, wing_pe: float) -> list[dict]:
    out = []
    for (sym, e_date), day in panel.groupby(["symbol", "date"], sort=True):
        e_date_str = pd.Timestamp(e_date).date().isoformat()
        if (ENTRY_DATE_MIN and e_date_str < ENTRY_DATE_MIN) or (ENTRY_DATE_MAX and e_date_str > ENTRY_DATE_MAX):
            continue
        if sym in EXCLUDE_SYMBOLS:
            continue
        if sym not in BACKTEST_UNIVERSE:
            continue                  # outside the backtest's fixed Nifty50 universe
        p = tpos.get(pd.Timestamp(e_date))
        if p is None or p == 0:
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
        # win may be shorter than FWD near the end of available history -- the trade is still
        # shown, marked "pending" below, rather than withheld until the full window exists.
        win = [pd.Timestamp(x) for x in tdays[p: p + FWD]]
        if not win:
            continue   # entry is the very last day in history -- no forward data at all yet
        def pick(ot, tgt):
            c = chain[chain["opt_type"] == ot]
            if c.empty:
                return None
            return c.iloc[(c["strike"] - tgt).abs().argmin()]
        sce, lce = pick("CE", u * (1 + SHORT_OTM)), pick("CE", u * (1 + SHORT_OTM + wing_ce))
        spe, lpe = pick("PE", u * (1 - SHORT_OTM)), pick("PE", u * (1 - SHORT_OTM - wing_pe))
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
        width = max(call_width, put_width)   # true worst-case-at-expiry risk is set by the WIDER
                                              # wing only (one side breaches at a time) -- not the sum
        risk = width - credit
        if credit <= 0 or risk <= 0:
            continue
        if risk < MIN_RISK_FRAC * width:
            continue
        entry_ror = credit / risk * 100
        min_ror_here = MCAP_MIN_ROR if sym in A_MCAP else OTHER_MIN_ROR
        # No early `continue` on the primary gate here -- every structurally-valid candidate is
        # carried through to the cross-tier fallback pass below (FALLBACK_MIN_ROR).
        vals, dd, exited_early = [], 0.0, False
        for i in range(len(win)):
            legbars = [seqs[k][1][i] for k in ("sc", "lc", "sp", "lp")]
            if not any(b is None or b[2] < MIN_VOL for b in legbars):
                value = (legbars[0][1] - legbars[1][1]) + (legbars[2][1] - legbars[3][1])
                value = max(0.0, min(width, value))
                upnl = credit - value
                vals.append((upnl, value))
                dd = min(dd, upnl)
            if (exp - win[i]).days <= SAFE_DTE:
                exited_early = True
                break
        if not vals:
            continue
        # complete = outcome is actually FINAL; otherwise the trade is still genuinely open (win
        # just hasn't reached FWD days yet) -- but its PnL is still shown, mark-to-market as of
        # the latest available day, same as a real broker P&L screen for an open position.
        # `complete` gates ONLY the `outcome` label (win/loss vs pending), never pnl/ror_pct.
        complete = exited_early or (len(win) == FWD)
        exit_value = vals[-1][1]
        pnl = round(credit - exit_value, 2)
        lot = lot_size(sym, exp)
        out.append({
            "tier": tier_id, "symbol": sym, "group": g2.get(sym, "C_top50oi"), "signal_date": t.date().isoformat(),
            "entry_date": pd.Timestamp(e_date).date().isoformat(),
            "expiry": exp.date().isoformat(), "dte": dte, "underlying": round(u, 1),
            "iv_ratio": round(float(ivr), 2) if ivr is not None and pd.notna(ivr) else None,
            "short_ce": float(sce["strike"]), "long_ce": float(lce["strike"]), "short_pe": float(spe["strike"]), "long_pe": float(lpe["strike"]),
            "call_width_pct": round(call_width / u * 100, 1), "put_width_pct": round(put_width / u * 100, 1),
            "sell_premium": round(seqs["sc"][0] + seqs["sp"][0], 2), "buy_premium": round(seqs["lc"][0] + seqs["lp"][0], 2),
            "credit": round(credit, 2), "max_risk": round(risk, 2), "max_profit": round(credit, 2),
            "entry_ror_pct": round(entry_ror, 1), "clears_primary_bar": entry_ror > min_ror_here,
            "lot_size": int(lot) if lot is not None and pd.notna(lot) else None,
            "max_risk_per_lot": round(risk * lot, 1) if lot is not None and pd.notna(lot) else None,
            "max_profit_per_lot": round(credit * lot, 1) if lot is not None and pd.notna(lot) else None,
            "exit_value": round(exit_value, 2),
            "pnl": pnl, "pnl_per_lot": (round(pnl * lot, 1) if lot is not None and pd.notna(lot) else None),
            "ror_pct": round(pnl / risk * 100, 1), "max_dd_pct": round(dd / risk * 100, 1),
            "outcome": ("win" if pnl > 0 else "loss") if complete else "pending",
        })
    return out


all_out = []
for tier_id, wing_ce, wing_pe in TIERS:
    rows = run_tier(tier_id, wing_ce, wing_pe)
    n_passing = sum(1 for r in rows if r["clears_primary_bar"])
    print(f"{tier_id} (ce={wing_ce} pe={wing_pe}): {len(rows)} candidates, {n_passing} clear the primary bar", flush=True)
    all_out.extend(rows)

outdir = ROOT / "locks" / "prod_sell_strategies"; outdir.mkdir(parents=True, exist_ok=True)


def _merge_partial(new_rows: list[dict], out_path, sort_cols):
    """Partial-range run: merge into the existing file instead of overwriting it -- keep every
    existing row OUTSIDE [ENTRY_DATE_MIN, ENTRY_DATE_MAX] untouched, replace only rows inside it.
    Shared by both the real-signal CSV and the raw-candidates cache below."""
    new_df = pd.DataFrame(new_rows).sort_values(sort_cols) if new_rows else pd.DataFrame(new_rows)
    if not (ENTRY_DATE_MIN or ENTRY_DATE_MAX) or not out_path.exists():
        return new_df
    old = pd.read_parquet(out_path) if out_path.suffix == ".parquet" else pd.read_csv(out_path)
    keep_old = old[(old["entry_date"] < (ENTRY_DATE_MIN or "0000-00-00")) | (old["entry_date"] > (ENTRY_DATE_MAX or "9999-99-99"))]
    return pd.concat([keep_old, new_df], ignore_index=True).sort_values(sort_cols)


# Raw-candidates cache: EVERY structurally-valid candidate across all 3 tiers (not just ones
# clearing their tier's primary ROR bar), for gen_no_signal_fallback.py to pool across all 3
# STRATEGIES and pick a single cross-strategy best pick on a day where none of them produced a
# real signal. Must run BEFORE gen_no_signal_fallback.py, which reads this file.
cand_path = outdir / "broken_wing_candidates_raw.parquet"
cand_df = _merge_partial(all_out, cand_path, ["tier", "entry_date"])
cand_df.to_parquet(cand_path, index=False)

# The real signal-history CSV only ever contains candidates that clear their tier's primary bar
# -- the fallback "NS" row for a no-signal day is added later, by gen_no_signal_fallback.py, once
# it can see ALL 3 strategies' candidate pools at once (not just this one).
all_out = [r for r in all_out if r["clears_primary_bar"]]
for r in all_out:
    del r["clears_primary_bar"]

out_path = outdir / "broken_wing_signal_history.csv"
df = _merge_partial(all_out, out_path, ["tier", "entry_date"])
df.to_csv(out_path, index=False)
n_pending = int((df["outcome"] == "pending").sum())
print(f"wrote {len(df)} signals (all tiers, {n_pending} pending -- pnl/ror_pct still shown, "
      f"mark-to-market as of the latest available day) -> {outdir / 'broken_wing_signal_history.csv'}")
completed = df[df["outcome"] != "pending"]
print(completed.groupby("tier").agg(n=("pnl", "size"),
                             win=("outcome", lambda s: (s == "win").mean()),
                             ev_ror=("ror_pct", "mean"), median_ror=("ror_pct", "median"),
                             worst_ror=("ror_pct", "min"), worst_dd=("max_dd_pct", "min")).round(1).to_string())
