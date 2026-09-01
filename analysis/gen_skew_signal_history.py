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

UNIFIED with the live /prod2/skew_strategy endpoint (2026-08, explicit user decision): universe
and ROR gate now match exactly (koscine.liquid_universe.live_universe_for_date == NIFTY50 +
that day's top-15 liquid non-Nifty50 tail; TIER_MIN_ROR == SKEW_TIER_MIN_ROR in api/main.py).
Only the SINGLE best (highest entry_ror_pct) candidate PER TIER PER DAY is kept as a real signal
-- matching the live panel's per-tier "top_picks[0] is THE 1-pick/day rule". Selection is
entry-time-only (deterministic, never revisited); see analysis/append_daily_sell_signals.py for
the separate, ongoing "mark still-open trades to market" concern this script does NOT handle day
to day anymore.

De-duplicated against Broken-Wing Condor (profitability-based, see
gen_signal_dedup_and_fallback.py): a skew candidate is dropped if the SAME symbol also has a
MORE PROFITABLE Broken-Wing signal on the same entry_date -- skew is a secondary/complementary
signal, not a duplicate of Broken-Wing's picks. De-dup removes a meaningful chunk -- post-dedup
production counts (pre-unification numbers, kept for context):
  t1_skew35: 251/2y (126/yr), median Rs.7,175/lot
  t2_skew6:   24/2y  (12/yr), median Rs.13,681/lot
  t3_skew8:    9/2y   (4/yr), median Rs.11,988/lot -- thin sample, treat as directional not precise
Combined with Broken-Wing's 403/2y (201/yr): 687/2y = 343.5/yr grand total, in the 320-350
target range.

Gating criterion is TIER_MIN_ROR (per-tier): the entry-time credit/max_risk ratio (knowable before the
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
from koscine import liquid_universe  # noqa: E402

SHORT_OTM, FWD, SAFE_DTE = 0.02, 5, 4
DTE_MIN, MIN_VOL = 7, 50   # DTE_MIN lowered 15 -> 7 (2026-08, explicit user
# decision): 15 meant rolling past the WHOLE current expiry (not just its last few days) any time
# fewer than 15 days remained, jumping a full month out (e.g. ITC fired Aug-25 expiry's 13 DTE
# wasn't enough, so it rolled to Sep-29/48 DTE) -- 7 still clears E-4 forced exit (SAFE_DTE=4)
# with a 3-day margin, same safety logic, just less conservative about how early it rolls.
# TIER_MIN_ROR (per-tier, 2026-09): previously ONE shared MIN_ENTRY_ROR across all 3 tiers --
# split per-tier (== SKEW_TIER_MIN_ROR in api/main.py) so t2/t3 can be relaxed independently of
# t1, without re-tuning t1's own gate. t1=115 unchanged (data-verified 2026-08: Skew is the
# strongest performer of the three, mean realized ROR ~119-142% -- looser bands still
# meaningfully outperform Condor's tightest tail on realized pnl/lot, and Condor has since been
# killed entirely, see gen_sell_signal_history.py). t2=100, t3=90 (2026-09, explicit user
# decision, data-verified against skew_candidates_raw.parquet): t2 was the "safe" relax option
# of a 2-option sweep (n/yr 40->59, mean pnl/lot -2.6%, worst DD unchanged at -5.3% -- the more
# aggressive t2 option crossed a real DD cliff to -18.5% and was rejected); t3 was the
# "aggressive" option of its own 3-option sweep (n/yr 20->45, mean pnl/lot -10.7%, worst DD
# unchanged at -2.7% -- t3's candidate pool tolerates loosening much better than t2's does).
TIER_MIN_ROR = {"t1_skew35": 115.0, "t2_skew6": 100.0, "t3_skew8": 90.0}
MIN_RISK_FRAC = 0.10   # max_risk must be >= 10% of wing width; below that, credit~=width and the
                        # entry_ror ratio becomes numerically degenerate (blows toward infinity)
MIN_PROFIT_PER_LOT = 10_000.0   # 2026-08, explicit user decision -- see
                                 # gen_sell_signal_history.py's identical constant for rationale.
MIN_OI_LOTS = 100   # 2026-08, explicit user decision -- see gen_broken_wing_signal_history.py's
                     # identical constant for rationale (excludes thin strikes from selection).
MIN_OI_LOTS_SELL = 500   # 2026-09, explicit user decision -- see gen_broken_wing_signal_history.py's
                          # identical constant for rationale (the sold/short leg needs a much
                          # higher liquidity floor than the bought/long leg).
_universe_cache: dict = {}


def _universe_for(e_date) -> set:
    """live_universe_for_date(), memoized per distinct date -- see gen_sell_signal_history.py's
    identical helper for why (redundant per-symbol recomputation for a full-history rebuild)."""
    key = pd.Timestamp(e_date)
    if key not in _universe_cache:
        _universe_cache[key] = liquid_universe.live_universe_for_date(key)
    return _universe_cache[key]
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


# Optional partial-range regen: `python gen_skew_signal_history.py bw_panel.parquet 2026-07-01
# 2026-08-31` only recomputes entries in [ENTRY_DATE_MIN, ENTRY_DATE_MAX] and merges the result
# into the existing CSV (rows outside that range are preserved untouched).
ENTRY_DATE_MIN = sys.argv[2] if len(sys.argv) > 2 else None
ENTRY_DATE_MAX = sys.argv[3] if len(sys.argv) > 3 else None
panel = pd.read_parquet(sys.argv[1])
panel["date"] = pd.to_datetime(panel["date"]); panel["expiry"] = pd.to_datetime(panel["expiry"])
g2 = {s: g for g, syms in json.loads((LOCK_V2 / "universe_groups.json").read_text()).items() for s in syms}
# Non-mega-cap symbols with a consistently weak (n>=3) historical track record across the
# combined Broken-Wing + Skew signal set, excluded outright: ANGELONE median Rs.608/lot over 8
# signals -- 4.6x below the next-weakest name (TMPV, Rs.2,820) -- not a fluke of a small sample.
EXCLUDE_SYMBOLS = {"ANGELONE"}

mk = load_market_data(columns=["date", "symbol", "atm_iv", "close"])
mk["date"] = pd.to_datetime(mk["date"]); mk["symbol"] = mk["symbol"].astype(str); mk = mk.sort_values(["symbol", "date"])
mk["iv_ratio"] = mk.groupby("symbol")["atm_iv"].transform(lambda s: s / s.rolling(252, min_periods=60).median())
IV = mk.set_index(["symbol", "date"])["iv_ratio"].to_dict()
# 2026-09, explicit user decision -- see gen_broken_wing_signal_history.py's identical constant
# for rationale: `underlying` is the ENTRY date's close, LTP needs the SIGNAL date's close.
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
    bars = [b.get(d) for d in win]     # (open, close, vol) or None per day
    return e[0], bars


def run_tier(tier_id: str, wing: float) -> list[dict]:
    min_ror = TIER_MIN_ROR[tier_id]
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

        # sce/spe are always the near-strike (SOLD, whichever side ends up richer below) picks;
        # lce/lpe are always the far-strike (bought) picks -- so the sell/buy OI floors apply by
        # position here, before the CE-vs-PE richer-side decision, same as BW.
        sce, lce = pick("CE", u * (1 + SHORT_OTM), MIN_OI_LOTS_SELL), pick("CE", u * (1 + SHORT_OTM + wing), MIN_OI_LOTS)
        spe, lpe = pick("PE", u * (1 - SHORT_OTM), MIN_OI_LOTS_SELL), pick("PE", u * (1 - SHORT_OTM - wing), MIN_OI_LOTS)
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
        # No early `continue` on the primary gate here -- every structurally-valid candidate is
        # carried through to the cross-tier fallback pass below (FALLBACK_MIN_ROR).

        vals, dd, exited_early = [], 0.0, False
        leg_closes = []   # (short_close, long_close) per kept day, parallel to `vals` -- lets
                          # the exit row show the ACTUAL per-leg closing premiums exit_value was
                          # computed from, not just the aggregate.
        exit_days = []    # win[i] per kept day, parallel to `vals` -- for the exit-day underlying (LTP) lookup below.
        for i in range(len(win)):
            sb, lb = seq_s[1][i], seq_l[1][i]
            if not (sb is None or lb is None or sb[2] < MIN_VOL or lb[2] < MIN_VOL):
                value = sb[1] - lb[1]
                value = max(0.0, min(width, value))
                upnl = credit - value
                vals.append(value)
                leg_closes.append((sb[1], lb[1]))
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
        exit_value = vals[-1]
        pnl = round(credit - exit_value, 2)
        # raw (unclamped) per-leg exit premiums -- can differ slightly from the [0,width]-clamped
        # exit_value above on a rare stale/illiquid print, same as entry's sell/buy_premium vs
        # credit already can.
        exit_sell_premium, exit_buy_premium = round(leg_closes[-1][0], 2), round(leg_closes[-1][1], 2)
        exit_underlying_raw = CLOSE.get((sym, exit_days[-1]))
        exit_underlying = round(float(exit_underlying_raw), 1) if exit_underlying_raw is not None and pd.notna(exit_underlying_raw) else None
        max_profit_per_lot = credit * lot if lot is not None and pd.notna(lot) else None

        # "Buy PnL" hedge analysis -- see gen_broken_wing_signal_history.py's identical block for
        # the full rationale. Skew only sells one side already (`side`/`short_row`), so the hedge
        # is simply the opposite type at that SAME short strike.
        buy_ot = "PE" if side == "CE" else "CE"
        buy_strike = float(short_row["strike"])
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
            "side": side, "ce_iv": round(ce_iv, 3), "pe_iv": round(pe_iv, 3), "skew": round(skew, 3),
            "short_strike": float(short_row["strike"]), "long_strike": float(long_row["strike"]),
            "oi_short": oi_lots(short_row), "oi_long": oi_lots(long_row),
            "sell_premium": round(seq_s[0], 2), "buy_premium": round(seq_l[0], 2),
            "credit": round(credit, 2), "max_risk": round(risk, 2), "max_profit": round(credit, 2),
            "entry_ror_pct": round(entry_ror, 1),
            "clears_primary_bar": entry_ror > min_ror and max_profit_per_lot is not None and max_profit_per_lot >= MIN_PROFIT_PER_LOT,
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


# NOTE: the old BW-vs-skew de-dup ("drop skew if BW has the same symbol/day") used to live here,
# but is now handled by gen_signal_dedup_and_fallback.py -- profitability-based (whichever of
# BW/skew is actually more profitable that symbol/day wins), not "BW always wins" -- since it
# needs both strategies' candidate pools at once. Run that script AFTER this one.

all_out = []
for tier_id, wing in TIERS:
    rows = run_tier(tier_id, wing)
    n_passing = sum(1 for r in rows if r["clears_primary_bar"])
    print(f"{tier_id} (wing={wing}): {len(rows)} candidates, {n_passing} clear the primary bar", flush=True)
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


# Raw-candidates cache: EVERY structurally-valid candidate across all 3 tiers, BEFORE the
# Broken-Wing de-dup too (gen_signal_dedup_and_fallback.py needs the undeduped pool to make its
# own profitability-based dedup + cross-strategy fallback decisions).
cand_path = outdir / "skew_candidates_raw.parquet"
cand_df = _merge_partial(all_out, cand_path, ["tier", "entry_date"])
cand_df.to_parquet(cand_path, index=False)

# The real signal-history CSV contains at most ONE row per (tier, day) -- the single highest-
# entry_ror_pct candidate that clears that tier's primary bar, matching the live panel's per-tier
# "top_picks[0] is THE 1-pick/day rule". BW de-dup and the cross-strategy fallback "NS" row are
# both added later, by gen_signal_dedup_and_fallback.py.
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

out_path = outdir / "skew_signal_history.csv"
df = _merge_partial(all_out, out_path, ["tier", "entry_date"])
df.to_csv(out_path, index=False)
n_pending = int((df["outcome"] == "pending").sum())
print(f"wrote {len(df)} signals (all tiers, de-duped, {n_pending} pending -- pnl/ror_pct still "
      f"shown, mark-to-market as of the latest available day) -> {outdir / 'skew_signal_history.csv'}")
completed = df[df["outcome"] != "pending"]
print(completed.groupby("tier").agg(n=("pnl", "size"),
                             win=("outcome", lambda s: (s == "win").mean()),
                             ev_ror=("ror_pct", "mean"), median_ror=("ror_pct", "median"),
                             worst_ror=("ror_pct", "min"), worst_dd=("max_dd_pct", "min")).round(1).to_string())
