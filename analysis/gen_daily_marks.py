"""Persists the full day-by-day per-leg mark path for every trade already in the calibrated
signal histories (gen_sell_signal_history.py / gen_broken_wing_signal_history.py /
gen_skew_signal_history.py) -- same entries, same thresholds, no change to which signals fire.

Why: those scripts only ever wrote each trade's ENTRY and FINAL EXIT summary; the daily path in
between was computed transiently and discarded. koscine/portfolio_sim.py needs to value a still-
OPEN position on any day before its natural exit (to decide whether to rotate out of it early
for a better opportunity) -- that requires the actual per-leg premiums each day, not just the
final outcome. Reuses the already-cached panel.parquet / bw_panel.parquet, no new bhavcopy
fetching.

trade_id = f"{symbol}|{entry_date}|{tier}" (tier="condor" for the symmetric strategy), joinable
back to each strategy's signal_history CSV on (symbol, entry_date) (+ tier for BW/skew).

Usage:
    python analysis/gen_daily_marks.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm

ROOT = Path(r"C:\Users\rahul\Koscine 3.0")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))
from koscine3.largemove.mover_v2 import LOCK_V2  # noqa: E402
from koscine import liquid_universe  # noqa: E402

OUT_DIR = ROOT / "locks" / "prod_sell_strategies"
BT_UNIVERSE = liquid_universe.backtest_universe()
EXCLUDE_SYMBOLS = {"ANGELONE"}


def _lot_map():
    lot_df = pd.read_parquet(ROOT / "data" / "silver" / "lot_size.parquet")
    return lot_df.set_index(["symbol", pd.to_datetime(lot_df["expiry_month"]).dt.to_period("M")])["lot"].to_dict()


def _cbar(panel: pd.DataFrame):
    """symbol/opt_type/expiry/strike -> {date: (open, close, vol)}, same shape every gen_*
    script already builds -- open needed for entry fills, close for daily marks."""
    return {k: dict(zip(pd.to_datetime(g["date"]).values, zip(g["open"].values, g["close"].values, g["vol"].values)))
            for k, g in panel.groupby(["symbol", "opt_type", "expiry", "strike"], sort=False)}


def _legseq(cbar, sym, ot, exp, strike, win):
    b = cbar.get((sym, ot, exp, strike))
    if not b:
        return None
    e = b.get(win[0])
    if not e or e[0] <= 0:
        return None
    return e[0], [b.get(d) for d in win]


# ----------------------------------------------------------------- Condor (4-leg, symmetric)
def condor_marks(fwd: int = 5, safe_dte: int = 4, dte_min: int = 9, min_vol: int = 50,
                  min_risk_frac: float = 0.10, min_entry_ror: float = 260.0,
                  short_otm: float = 0.02, wing: float = 0.05) -> pd.DataFrame:
    panel = pd.read_parquet(ROOT / "panel.parquet")
    panel["date"] = pd.to_datetime(panel["date"]); panel["expiry"] = pd.to_datetime(panel["expiry"])
    panel = panel[panel["symbol"].isin(BT_UNIVERSE)]
    tdays = np.array(sorted(panel["date"].unique())); tpos = {d: k for k, d in enumerate(tdays)}
    lot_map = _lot_map()
    cbar = _cbar(panel)

    rows = []
    for (sym, e_date), day in panel.groupby(["symbol", "date"], sort=True):
        p = tpos.get(pd.Timestamp(e_date))
        if p is None or p == 0:
            continue
        u = day["underlying"].iloc[0]
        exps = sorted(day["expiry"].unique())
        exp = next((e for e in exps if (pd.Timestamp(e) - pd.Timestamp(e_date)).days >= dte_min), None)
        if exp is None:
            continue
        chain = day[day["expiry"] == exp]

        def pick(ot, tgt):
            c = chain[chain["opt_type"] == ot]
            return None if c.empty else c.iloc[(c["strike"] - tgt).abs().argmin()]
        sce, lce = pick("CE", u * (1 + short_otm)), pick("CE", u * (1 + short_otm + wing))
        spe, lpe = pick("PE", u * (1 - short_otm)), pick("PE", u * (1 - short_otm - wing))
        if any(x is None for x in (sce, lce, spe, lpe)):
            continue
        if sce["open"] < 3 or spe["open"] < 3 or sce["vol"] < min_vol or spe["vol"] < min_vol:
            continue
        win = [pd.Timestamp(x) for x in tdays[p:p + fwd]]
        if not win:
            continue
        seqs = {k: _legseq(cbar, sym, ot, exp, float(r["strike"]), win)
                for k, (r, ot) in {"sc": (sce, "CE"), "lc": (lce, "CE"), "sp": (spe, "PE"), "lp": (lpe, "PE")}.items()}
        if any(v is None for v in seqs.values()):
            continue
        credit = (seqs["sc"][0] + seqs["sp"][0]) - (seqs["lc"][0] + seqs["lp"][0])
        width = max(float(lce["strike"] - sce["strike"]), float(spe["strike"] - lpe["strike"]))
        risk = width - credit
        if credit <= 0 or risk <= 0 or risk < min_risk_frac * width:
            continue
        if (credit / risk * 100) <= min_entry_ror:
            continue
        lot = lot_map.get((sym, pd.Timestamp(exp).to_period("M")))
        if lot is None or pd.isna(lot):
            continue
        trade_id = f"{sym}|{pd.Timestamp(e_date).date().isoformat()}|condor"
        # entry fill = the OPEN price on the entry bar (seqs[k][0], same as the credit calc
        # above uses) -- NOT day-0's close, which the per-day loop below records instead.
        # Stored once per trade (repeated across its rows) so entry brokerage can be computed
        # from the actual fill, not approximated from the entry day's close.
        entry_prices = dict(sc_entry=seqs["sc"][0], lc_entry=seqs["lc"][0],
                            sp_entry=seqs["sp"][0], lp_entry=seqs["lp"][0])
        for i in range(len(win)):
            legbars = [seqs[k][1][i] for k in ("sc", "lc", "sp", "lp")]
            if any(b is None or b[2] < min_vol for b in legbars):
                exited = (exp - win[i]).days <= safe_dte
                if exited:
                    break
                continue
            rows.append({"trade_id": trade_id, "symbol": sym, "date": win[i], "dte": (exp - win[i]).days,
                         "sc_close": legbars[0][1], "lc_close": legbars[1][1],
                         "sp_close": legbars[2][1], "lp_close": legbars[3][1],
                         "credit": credit, "max_risk": risk, "width": width, "lot_size": int(lot),
                         **entry_prices})
            if (exp - win[i]).days <= safe_dte:
                break
    return pd.DataFrame(rows)


# ----------------------------------------------------------------- Broken-Wing (4-leg, tiered)
def broken_wing_marks(fwd: int = 5, safe_dte: int = 4, dte_min: int = 9, min_vol: int = 50,
                       min_risk_frac: float = 0.10, short_otm: float = 0.02,
                       mcap_min_ror: float = 108.0, other_min_ror: float = 138.0) -> pd.DataFrame:
    TIERS = [("t1_2x6", 0.02, 0.06), ("t2_3x8", 0.03, 0.08), ("t3_2x10", 0.02, 0.10)]
    A_MCAP = set(json.loads((LOCK_V2 / "universe_groups.json").read_text()).get("A_mcap30", []))

    panel = pd.read_parquet(ROOT / "bw_panel.parquet")
    panel["date"] = pd.to_datetime(panel["date"]); panel["expiry"] = pd.to_datetime(panel["expiry"])
    panel = panel[panel["symbol"].isin(BT_UNIVERSE)]
    tdays = np.array(sorted(panel["date"].unique())); tpos = {d: k for k, d in enumerate(tdays)}
    lot_map = _lot_map()
    cbar = _cbar(panel)
    grouped = list(panel.groupby(["symbol", "date"], sort=True))

    rows = []
    for tid, wing_ce, wing_pe in TIERS:
        for (sym, e_date), day in grouped:
            if sym in EXCLUDE_SYMBOLS:
                continue
            p = tpos.get(pd.Timestamp(e_date))
            if p is None or p == 0:
                continue
            u = day["underlying"].iloc[0]
            exps = sorted(day["expiry"].unique())
            exp = next((e for e in exps if (pd.Timestamp(e) - pd.Timestamp(e_date)).days >= dte_min), None)
            if exp is None:
                continue
            chain = day[day["expiry"] == exp]

            def pick(ot, tgt):
                c = chain[chain["opt_type"] == ot]
                return None if c.empty else c.iloc[(c["strike"] - tgt).abs().argmin()]
            sce, lce = pick("CE", u * (1 + short_otm)), pick("CE", u * (1 + short_otm + wing_ce))
            spe, lpe = pick("PE", u * (1 - short_otm)), pick("PE", u * (1 - short_otm - wing_pe))
            if any(x is None for x in (sce, lce, spe, lpe)):
                continue
            if sce["open"] < 3 or spe["open"] < 3 or sce["vol"] < min_vol or spe["vol"] < min_vol:
                continue
            win = [pd.Timestamp(x) for x in tdays[p:p + fwd]]
            if not win:
                continue
            seqs = {k: _legseq(cbar, sym, ot, exp, float(r["strike"]), win)
                    for k, (r, ot) in {"sc": (sce, "CE"), "lc": (lce, "CE"), "sp": (spe, "PE"), "lp": (lpe, "PE")}.items()}
            if any(v is None for v in seqs.values()):
                continue
            credit = (seqs["sc"][0] + seqs["sp"][0]) - (seqs["lc"][0] + seqs["lp"][0])
            call_width = float(lce["strike"] - sce["strike"]); put_width = float(spe["strike"] - lpe["strike"])
            width = max(call_width, put_width)
            risk = width - credit
            if credit <= 0 or risk <= 0 or risk < min_risk_frac * width:
                continue
            min_ror_here = mcap_min_ror if sym in A_MCAP else other_min_ror
            if (credit / risk * 100) <= min_ror_here:
                continue
            lot = lot_map.get((sym, pd.Timestamp(exp).to_period("M")))
            if lot is None or pd.isna(lot):
                continue
            trade_id = f"{sym}|{pd.Timestamp(e_date).date().isoformat()}|{tid}"
            entry_prices = dict(sc_entry=seqs["sc"][0], lc_entry=seqs["lc"][0],
                                sp_entry=seqs["sp"][0], lp_entry=seqs["lp"][0])
            for i in range(len(win)):
                legbars = [seqs[k][1][i] for k in ("sc", "lc", "sp", "lp")]
                if any(b is None or b[2] < min_vol for b in legbars):
                    if (exp - win[i]).days <= safe_dte:
                        break
                    continue
                rows.append({"trade_id": trade_id, "symbol": sym, "date": win[i], "dte": (exp - win[i]).days,
                             "sc_close": legbars[0][1], "lc_close": legbars[1][1],
                             "sp_close": legbars[2][1], "lp_close": legbars[3][1],
                             "credit": credit, "max_risk": risk, "width": width, "lot_size": int(lot),
                             **entry_prices})
                if (exp - win[i]).days <= safe_dte:
                    break
    return pd.DataFrame(rows)


# ----------------------------------------------------------------- Skew (2-leg, single side)
def _bs_price(S, K, T, sigma, r, is_call):
    if sigma <= 0 or T <= 0:
        return max(0.0, (S - K) if is_call else (K - S))
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if is_call:
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def _implied_vol(price, S, K, T, r, is_call):
    intrinsic = max(0.0, (S - K) if is_call else (K - S))
    if price <= intrinsic + 1e-6 or T <= 0:
        return None
    try:
        return brentq(lambda s: _bs_price(S, K, T, s, r, is_call) - price, 1e-4, 6.0, xtol=1e-4)
    except ValueError:
        return None


def skew_marks(fwd: int = 5, safe_dte: int = 4, dte_min: int = 15, min_vol: int = 50,
               min_risk_frac: float = 0.10, min_entry_ror: float = 130.0,
               short_otm: float = 0.02, r: float = 0.065) -> pd.DataFrame:
    TIERS = [("t1_skew35", 0.035), ("t2_skew6", 0.06), ("t3_skew8", 0.08)]

    panel = pd.read_parquet(ROOT / "bw_panel.parquet")
    panel["date"] = pd.to_datetime(panel["date"]); panel["expiry"] = pd.to_datetime(panel["expiry"])
    panel = panel[panel["symbol"].isin(BT_UNIVERSE)]
    tdays = np.array(sorted(panel["date"].unique())); tpos = {d: k for k, d in enumerate(tdays)}
    lot_map = _lot_map()
    cbar = _cbar(panel)

    # De-dup against Broken-Wing, mirroring gen_skew_signal_history.py exactly: a skew candidate
    # is dropped if the same symbol also has a Broken-Wing signal (any tier) on the same
    # entry_date -- skew is a secondary/complementary signal there, not counted twice. Without
    # this, skew_marks() overcounts every tier vs. the production skew_signal_history.csv (was
    # 284 unique trade_id vs. 201 signals -- the exact ~83-trade gap this closes).
    bw_path = OUT_DIR / "broken_wing_signal_history.csv"
    bw_pairs = set()
    if bw_path.exists():
        bw = pd.read_csv(bw_path, usecols=["symbol", "entry_date"])
        bw_pairs = set(zip(bw["symbol"], bw["entry_date"]))
    else:
        print(f"WARNING: {bw_path} not found -- run gen_broken_wing_signal_history.py first; "
              f"skipping de-dup (all skew candidates kept)", flush=True)

    rows = []
    for (sym, e_date), day in panel.groupby(["symbol", "date"], sort=True):
        if sym in EXCLUDE_SYMBOLS:
            continue
        if (sym, pd.Timestamp(e_date).date().isoformat()) in bw_pairs:
            continue
        p = tpos.get(pd.Timestamp(e_date))
        if p is None or p == 0:
            continue
        u = day["underlying"].iloc[0]
        exps = sorted(day["expiry"].unique())
        exp = next((e for e in exps if (pd.Timestamp(e) - pd.Timestamp(e_date)).days >= dte_min), None)
        if exp is None:
            continue
        chain = day[day["expiry"] == exp]

        def pick(ot, tgt):
            c = chain[chain["opt_type"] == ot]
            return None if c.empty else c.iloc[(c["strike"] - tgt).abs().argmin()]
        sce_s, spe_s = pick("CE", u * (1 + short_otm)), pick("PE", u * (1 - short_otm))
        if sce_s is None or spe_s is None:
            continue
        if sce_s["open"] < 3 or spe_s["open"] < 3 or sce_s["vol"] < min_vol or spe_s["vol"] < min_vol:
            continue
        dte0 = (exp - pd.Timestamp(e_date)).days
        T = dte0 / 365.0
        ce_iv = _implied_vol(float(sce_s["open"]), u, float(sce_s["strike"]), T, r, True)
        pe_iv = _implied_vol(float(spe_s["open"]), u, float(spe_s["strike"]), T, r, False)
        if ce_iv is None or pe_iv is None:
            continue
        side = "CE" if (ce_iv - pe_iv) > 0 else "PE"
        short_row, ot = (sce_s, "CE") if side == "CE" else (spe_s, "PE")

        win = [pd.Timestamp(x) for x in tdays[p:p + fwd]]
        if not win:
            continue
        for tid, wing in TIERS:
            tgt = u * (1 + short_otm + wing) if side == "CE" else u * (1 - short_otm - wing)
            long_row = pick(ot, tgt)
            if long_row is None:
                continue
            seq_s = _legseq(cbar, sym, ot, exp, float(short_row["strike"]), win)
            seq_l = _legseq(cbar, sym, ot, exp, float(long_row["strike"]), win)
            if seq_s is None or seq_l is None:
                continue
            credit = seq_s[0] - seq_l[0]
            width = abs(float(long_row["strike"]) - float(short_row["strike"]))
            risk = width - credit
            if credit <= 0 or risk <= 0 or risk < min_risk_frac * width:
                continue
            if (credit / risk * 100) <= min_entry_ror:
                continue
            lot = lot_map.get((sym, pd.Timestamp(exp).to_period("M")))
            if lot is None or pd.isna(lot):
                continue
            trade_id = f"{sym}|{pd.Timestamp(e_date).date().isoformat()}|{tid}"
            entry_prices = dict(short_entry=seq_s[0], long_entry=seq_l[0])
            for i in range(len(win)):
                sb, lb = seq_s[1][i], seq_l[1][i]
                if sb is None or lb is None or sb[2] < min_vol or lb[2] < min_vol:
                    if (exp - win[i]).days <= safe_dte:
                        break
                    continue
                rows.append({"trade_id": trade_id, "symbol": sym, "date": win[i], "dte": (exp - win[i]).days,
                             "short_close": sb[1], "long_close": lb[1], "side": side,
                             "credit": credit, "max_risk": risk, "width": width, "lot_size": int(lot),
                             **entry_prices})
                if (exp - win[i]).days <= safe_dte:
                    break
    return pd.DataFrame(rows)


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, fn in [("condor", condor_marks), ("broken_wing", broken_wing_marks), ("skew", skew_marks)]:
        print(f"[gen_daily_marks] building {name} ...", flush=True)
        df = fn()
        n_trades = df["trade_id"].nunique() if len(df) else 0
        print(f"[gen_daily_marks] {name}: {len(df)} rows, {n_trades} trades", flush=True)
        df.to_parquet(OUT_DIR / f"{name}_daily_marks.parquet", index=False)
