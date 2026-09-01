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

UNIFIED with the live /prod2/broken_wing_strategy endpoint (2026-08, explicit user decision):
universe and ROR gate now match exactly (koscine.liquid_universe.live_universe_for_date ==
NIFTY50 + that day's top-15 liquid non-Nifty50 tail, replacing the old static-A/B-UNION-top-50
panel universe; TIER_MCAP_MIN_ROR/TIER_OTHER_MIN_ROR == the identically-named dicts in
api/main.py, per-tier since 2026-09). Only the SINGLE best (highest entry_ror_pct) candidate PER TIER PER DAY is kept as a
real signal -- matching the live panel's per-tier "top_picks[0] is THE 1-pick/day rule" -- not
every candidate that happened to clear that tier's bar that day. Selection is entry-time-only
(deterministic, never revisited); see analysis/append_daily_sell_signals.py for the separate,
ongoing "mark still-open trades to market" concern this script does NOT handle day to day
anymore.

Entry gate is STRATIFIED, not a single threshold: mega-caps (A_mcap30) need entry_ror >
TIER_MCAP_MIN_ROR[tier], everyone else needs > TIER_OTHER_MIN_ROR[tier]. Why: mega-caps are
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

SHORT_OTM, FWD, SAFE_DTE = 0.02, 5, 4
DTE_MIN, MIN_VOL = 7, 50   # DTE_MIN lowered 9 -> 7 (2026-08, explicit user decision): rolling
# forward to the NEXT expiry any time the current one drops below the floor pushes DTE (and, for
# Skew, all the way to the following month) further out than needed -- 7 still clears E-4 forced
# exit (SAFE_DTE=4) with a 3-day margin, same safety logic as before, just less conservative.
# TIER_MCAP_MIN_ROR / TIER_OTHER_MIN_ROR (per-tier, 2026-09) -- == the identically-named dicts
# in api/main.py. Previously ONE shared MCAP_MIN_ROR/OTHER_MIN_ROR pair applied to all 3 tiers;
# split per-tier so t2/t3 can be relaxed independently of t1, without re-tuning t1's own gate.
# t1_2x6 unchanged at 150/155 (history: 150/155 -> 145/150 -> 120/125 -> 190/195 -> 150/155,
# 2026-08 explicit user decision -- the 190/195 step, data-verified against
# broken_wing_candidates_raw.parquet, fixed the DTE=7/OI-filter volume blowup but only got mean
# realized pnl/lot to ~Rs.6,270; ROR% alone doesn't select for absolute rupee payout, so the
# actual quality lifting is done by MIN_PROFIT_PER_LOT below instead -- loosening t1's ROR bar
# further costs more mean pnl than it's worth in added volume, per that sweep).
# t2_3x8/t3_2x10 = 110/115 (2026-09, explicit user decision, data-verified against the same raw
# pool): the "Option B" of a 2-option sweep for each tier -- t2 110/115 -> n/yr 20->57, mean
# pnl/lot -13.8%, worst DD unchanged at -15.2% (no cliff, unlike Skew's equivalent sweep); t3
# 110/115 -> n/yr 5->14, mean pnl/lot -15.7%, worst DD -7.5%->-8.9% (a small tick, not a cliff).
TIER_MCAP_MIN_ROR = {"t1_2x6": 150.0, "t2_3x8": 110.0, "t3_2x10": 110.0}
TIER_OTHER_MIN_ROR = {"t1_2x6": 155.0, "t2_3x8": 115.0, "t3_2x10": 115.0}
MIN_RISK_FRAC = 0.10   # max_risk must be >= 10% of the (wider) wing; below that, credit~=width
                        # and the entry_ror ratio becomes numerically degenerate
MIN_OI_LOTS = 100   # 2026-08, explicit user decision: a strike with <=100 lots of open interest
# is too thin to actually fill/exit at a sane price (wide bid/ask, real slippage not visible in
# an EOD close print) -- excluded from strike selection entirely (not just flagged), same logic
# as the existing MIN_VOL day-of-hold liquidity guard but applied at ENTRY strike-picking time.
MIN_OI_LOTS_SELL = 500   # 2026-09, explicit user decision: the SOLD (short) leg specifically
# needs a much higher liquidity floor than the bought (long) leg -- it's the leg actually
# carrying the premium/risk and the one you need to reliably exit early if the trade needs to be
# closed before expiry, whereas the long leg is cheap, far-OTM protection rarely touched before
# expiry. The long leg keeps the general MIN_OI_LOTS=100 floor.
MIN_PROFIT_PER_LOT = 19_000.0   # BW-specific floor (Condor/Skew use Rs.10,000, see
# gen_sell_signal_history.py's identical constant). Raised 10k -> 19k (2026-08, explicit user
# decision, data-verified against broken_wing_candidates_raw.parquet): this is the ACTUAL quality
# lever for BW's mean realized pnl/lot target (see TIER_MCAP_MIN_ROR/TIER_OTHER_MIN_ROR above for why ROR%
# alone couldn't get there) -- 19k is the highest floor that still clears a Rs.8k mean while
# keeping ~70/yr volume at the 150/155 ROR gate.
_universe_cache: dict = {}


def _universe_for(e_date) -> set:
    """live_universe_for_date(), memoized per distinct date -- see gen_sell_signal_history.py's
    identical helper for why (redundant per-symbol recomputation for a full-history rebuild)."""
    key = pd.Timestamp(e_date)
    if key not in _universe_cache:
        _universe_cache[key] = liquid_universe.live_universe_for_date(key)
    return _universe_cache[key]

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

mk = load_market_data(columns=["date", "symbol", "atm_iv", "close"])
mk["date"] = pd.to_datetime(mk["date"]); mk["symbol"] = mk["symbol"].astype(str); mk = mk.sort_values(["symbol", "date"])
mk["iv_ratio"] = mk.groupby("symbol")["atm_iv"].transform(lambda s: s / s.rolling(252, min_periods=60).median())
IV = mk.set_index(["symbol", "date"])["iv_ratio"].to_dict()
# 2026-09, explicit user decision: the existing `underlying` field is the ENTRY date's close (the
# option bhavcopy's own reference price, from the panel row at entry_date) -- LTP should instead
# be the SIGNAL date's close (the day the signal actually fired, one trading day earlier), so it's
# a separate lookup keyed off `t` (signal_date) below, not the panel's own per-row underlying.
CLOSE = mk.set_index(["symbol", "date"])["close"].to_dict()
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
        if sym not in _universe_for(e_date):
            continue                  # same universe rule the live panel uses for this date
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
        lot = lot_size(sym, exp)   # needed up-front now: the OI-lots liquidity filter below
                                    # divides raw share-OI by lot size before comparing to MIN_OI_LOTS
        def pick(ot, tgt, min_oi):
            c = chain[chain["opt_type"] == ot]
            if lot is not None and pd.notna(lot) and lot > 0:
                c = c[c["oi"] / lot >= min_oi]   # exclude thin strikes from consideration
                                                  # entirely, not just flag them
            if c.empty:
                return None
            return c.iloc[(c["strike"] - tgt).abs().argmin()]
        sce, lce = pick("CE", u * (1 + SHORT_OTM), MIN_OI_LOTS_SELL), pick("CE", u * (1 + SHORT_OTM + wing_ce), MIN_OI_LOTS)
        spe, lpe = pick("PE", u * (1 - SHORT_OTM), MIN_OI_LOTS_SELL), pick("PE", u * (1 - SHORT_OTM - wing_pe), MIN_OI_LOTS)
        if any(x is None for x in (sce, lce, spe, lpe)):
            continue
        if sce["open"] < 3 or spe["open"] < 3 or sce["vol"] < MIN_VOL or spe["vol"] < MIN_VOL:
            continue
        seqs = {k: legseq(sym, ot, exp, float(r["strike"]), win)
                for k, (r, ot) in {"sc": (sce, "CE"), "lc": (lce, "CE"), "sp": (spe, "PE"), "lp": (lpe, "PE")}.items()}
        if any(v is None for v in seqs.values()):
            continue
        credit = (seqs["sc"][0] + seqs["sp"][0]) - (seqs["lc"][0] + seqs["lp"][0])
        ce_credit, pe_credit = seqs["sc"][0] - seqs["lc"][0], seqs["sp"][0] - seqs["lp"][0]
        richer_side = "CE" if ce_credit >= pe_credit else "PE"   # 2026-09, explicit user decision:
        # the Position column shows only this side's strikes now (both legs are still genuinely
        # part of the trade -- credit/risk/pnl below are unchanged, still the full 4-leg number --
        # this only picks which pair to DISPLAY when both are shown, matching how Skew already
        # picks a single richer side).
        call_width = float(lce["strike"] - sce["strike"]); put_width = float(spe["strike"] - lpe["strike"])
        width = max(call_width, put_width)   # true worst-case-at-expiry risk is set by the WIDER
                                              # wing only (one side breaches at a time) -- not the sum
        risk = width - credit
        if credit <= 0 or risk <= 0:
            continue
        if risk < MIN_RISK_FRAC * width:
            continue
        entry_ror = credit / risk * 100
        min_ror_here = TIER_MCAP_MIN_ROR[tier_id] if sym in A_MCAP else TIER_OTHER_MIN_ROR[tier_id]
        # No early `continue` on the primary gate here -- every structurally-valid candidate is
        # carried through to the cross-tier fallback pass below (FALLBACK_MIN_ROR).
        vals, dd, exited_early = [], 0.0, False
        leg_closes = []   # (sc_close, lc_close, sp_close, lp_close) per kept day, parallel to `vals`
                          # -- lets the exit row show the ACTUAL per-leg closing premiums the
                          # aggregate exit_value was computed from, not just the aggregate itself.
        exit_days = []    # win[i] per kept day, parallel to `vals` -- which calendar day the
                          # exit actually landed on, for the exit-day underlying (LTP) lookup below.
        for i in range(len(win)):
            legbars = [seqs[k][1][i] for k in ("sc", "lc", "sp", "lp")]
            if not any(b is None or b[2] < MIN_VOL for b in legbars):
                value = (legbars[0][1] - legbars[1][1]) + (legbars[2][1] - legbars[3][1])
                value = max(0.0, min(width, value))
                upnl = credit - value
                vals.append((upnl, value))
                leg_closes.append((legbars[0][1], legbars[1][1], legbars[2][1], legbars[3][1]))
                exit_days.append(win[i])
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
        exit_sc, exit_lc, exit_sp, exit_lp = leg_closes[-1]
        # raw (unclamped) per-leg exit premiums -- their sum can differ slightly from the
        # [0,width]-clamped exit_value above on a rare stale/illiquid print, same as entry's
        # sell_premium/buy_premium vs credit already can.
        exit_sell_premium = round(exit_sc + exit_sp, 2)
        exit_buy_premium = round(exit_lc + exit_lp, 2)
        exit_underlying_raw = CLOSE.get((sym, exit_days[-1]))
        exit_underlying = round(float(exit_underlying_raw), 1) if exit_underlying_raw is not None and pd.notna(exit_underlying_raw) else None
        max_profit_per_lot = credit * lot if lot is not None and pd.notna(lot) else None

        # "Buy PnL" hedge analysis (2026-09, explicit user decision): the OPPOSITE side of
        # whichever strike was actually SOLD (richer_side) -- e.g. sold short_ce(CE) -> hypothetically
        # BUY a PE at that SAME strike at entry, hold it, and take the best (peak) close reached
        # across the full FWD-day window (independent of this trade's own exit timing -- "next 5
        # trading sessions" is its own window, not truncated by SAFE_DTE forcing the spread out
        # early). Purely a comparison figure -- does not affect credit/risk/pnl above at all.
        buy_ot = "PE" if richer_side == "CE" else "CE"
        buy_strike = float(sce["strike"]) if richer_side == "CE" else float(spe["strike"])
        buy_seq = legseq(sym, buy_ot, exp, buy_strike, win)
        buy_pnl_per_lot = None
        if buy_seq is not None and buy_seq[0] > 0:
            buy_valid_closes = [b[1] for b in buy_seq[1] if b is not None and b[2] >= MIN_VOL]
            if buy_valid_closes and lot is not None and pd.notna(lot):
                buy_pnl_per_lot = round((max(buy_valid_closes) - buy_seq[0]) * lot, 1)

        def oi_lots(row):
            v = row.get("oi")
            if v is None or pd.isna(v) or lot is None or pd.isna(lot) or lot == 0:
                return None
            return round(float(v) / float(lot))

        signal_underlying = CLOSE.get((sym, t))
        out.append({
            "tier": tier_id, "symbol": sym, "group": g2.get(sym, "C_top50oi"), "signal_date": t.date().isoformat(),
            "entry_date": pd.Timestamp(e_date).date().isoformat(),
            "expiry": exp.date().isoformat(), "dte": dte, "underlying": round(u, 1),
            "signal_underlying": round(float(signal_underlying), 1) if signal_underlying is not None and pd.notna(signal_underlying) else None,
            "iv_ratio": round(float(ivr), 2) if ivr is not None and pd.notna(ivr) else None,
            "short_ce": float(sce["strike"]), "long_ce": float(lce["strike"]), "short_pe": float(spe["strike"]), "long_pe": float(lpe["strike"]),
            "oi_short_ce": oi_lots(sce), "oi_long_ce": oi_lots(lce), "oi_short_pe": oi_lots(spe), "oi_long_pe": oi_lots(lpe),
            "call_width_pct": round(call_width / u * 100, 1), "put_width_pct": round(put_width / u * 100, 1),
            "richer_side": richer_side,
            "sell_premium": round(seqs["sc"][0] + seqs["sp"][0], 2), "buy_premium": round(seqs["lc"][0] + seqs["lp"][0], 2),
            "credit": round(credit, 2), "max_risk": round(risk, 2), "max_profit": round(credit, 2),
            "entry_ror_pct": round(entry_ror, 1),
            "clears_primary_bar": entry_ror > min_ror_here and max_profit_per_lot is not None and max_profit_per_lot >= MIN_PROFIT_PER_LOT,
            "lot_size": int(lot) if lot is not None and pd.notna(lot) else None,
            "max_risk_per_lot": round(risk * lot, 1) if lot is not None and pd.notna(lot) else None,
            "max_profit_per_lot": round(credit * lot, 1) if lot is not None and pd.notna(lot) else None,
            "exit_underlying": exit_underlying,
            "exit_value": round(exit_value, 2),
            "exit_sell_premium": exit_sell_premium, "exit_buy_premium": exit_buy_premium,
            "pnl": pnl, "pnl_per_lot": (round(pnl * lot, 1) if lot is not None and pd.notna(lot) else None),
            "buy_pnl_per_lot": buy_pnl_per_lot,
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

# The real signal-history CSV contains at most ONE row per (tier, day) -- the single highest-
# entry_ror_pct candidate that clears that tier's primary bar, matching the live panel's per-tier
# "top_picks[0] is THE 1-pick/day rule" (not every candidate that happened to clear the bar that
# day). The cross-strategy fallback "NS" row for a day nothing anywhere clears is added later, by
# gen_signal_dedup_and_fallback.py, once it can see all 3 strategies' candidate pools at once.
by_tier_date: dict[tuple, list[dict]] = {}
for r in all_out:
    by_tier_date.setdefault((r["tier"], r["entry_date"]), []).append(r)
all_out = []
for rows in by_tier_date.values():
    passing = [r for r in rows if r["clears_primary_bar"]]
    if passing:
        all_out.append(max(passing, key=lambda r: r["entry_ror_pct"]))
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
