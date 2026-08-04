"""direction_overlay_v1 / group-B MONTHLY-retrain sweep. (contained; no PROD touch)

Focus the direction edge where it lives (group B = 35 movers). MONTHLY walk-forward: train <= Dec-2025 -> predict
Jan-2026, then retrain every month and predict through the latest labelable date (5d forward => ~end-May 2026).
Grid: train window {3m, 6m, 9m, expanding} x feature set {ALL, NO_CALENDAR, MARKET+MOMENTUM, OI+FLOW}, plus
ALL-features trained on B-only, plus the regime-rule baseline. Eval = group-B universe AUC/hit/IC AND real
ATM+/-2% CALL/PUT premium EV (held 5d, net 3%) on the group-B v3 book picks, bootstrap 95% CI.

    set PYTHONPATH=src && python experiments/direction_overlay_v1/run_groupB_monthly.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
from koscine3.data.sources import load_market_data  # noqa: E402

LOCK_V2 = ROOT / "locks" / "prod_largemove_v2"
BOOK5 = ROOT / "locks" / "prod_largemove_v3" / "mover_v3_book_5d.csv"
OPTC = ROOT / "experiments" / "option_move_v1" / "results" / "option_contracts.csv"
OUT = HERE / "results"; OUT.mkdir(exist_ok=True)

H, EMBARGO, COST, RET_CAP = 5, 6, 0.03, 5.0
EVAL_MONTHS = pd.period_range("2026-01", "2026-06", freq="M")
GB = "B_turn35"
CB = dict(iterations=500, depth=5, learning_rate=0.03, l2_leaf_reg=6.0, random_seed=7,
          allow_writing_files=False, verbose=False, task_type="GPU", devices="0")

LEAK = ("future", "fwd", "next", "ahead", "label", "adverse", "up_move", "down_move", "expansion",
        "volclean", "outcome", "entry_1d", "_date", "tomorrow")
ID = {"date", "symbol", "open", "high", "low", "close", "last", "prev_close", "volume", "group",
      "in_univ", "eligible", "y", "fwd_ret", "per", "reg5", "volbk"}
CAL = {"month", "day_of_week", "days_to_month_end", "is_expiry_week"}
MKT_MOM = ["nifty_ret_1d", "nifty_ret_5d", "nifty_realized_vol_20", "mkt_pct_above_sma20", "mkt_pct_above_sma50",
           "mkt_advance_ratio", "ret_1d", "ret_5d", "ret_20d", "rel_ret_5d_vs_nifty", "ret_5d_cs_rank",
           "ret_20d_cs_rank", "sector_ret_5d", "sector_ret_20d", "stock_rel_sector_ret_5d",
           "stock_rel_sector_ret_20d", "ema_20_slope_5d", "ema_50_slope_5d", "adx_14", "di_diff",
           "consec_up_days", "consec_down_days", "pos_day_share_20d", "close_sma50_dist", "ema_50_dist"]
OI_FLOW = ["fut_chg_oi", "fut_oi_chg_5", "fut_oi_z_60d", "fut_oi_ratio_20", "oi_buildup_ratio", "oi_long_buildup",
           "oi_short_buildup", "oi_long_unwind", "oi_short_unwind", "oi_long_buildup_5d", "oi_short_buildup_5d",
           "oi_long_unwind_5d", "oi_short_unwind_5d", "price_oi_divergence", "oi_acceleration", "pcr_oi",
           "pcr_oi_chg_5", "pcr_vol", "pcr_vol_chg_5", "delivery_pct", "delivery_pct_chg_5", "delivery_qty_ratio_20",
           "iv_skew_norm", "iv_skew_chg_5d", "put_call_iv_skew", "max_pain_dist", "call_wall_1_dist", "put_wall_1_dist"]
WINDOWS = {"3m": 90, "6m": 180, "9m": 270, "expand": None}


def regime(dp, doi):
    r = np.full(len(dp), "flat", dtype=object)
    r[(dp > 0) & (doi > 0)] = "LongBuildup"; r[(dp < 0) & (doi > 0)] = "ShortBuildup"
    r[(dp > 0) & (doi < 0)] = "ShortCovering"; r[(dp < 0) & (doi < 0)] = "LongUnwinding"
    return r


def load_panel():
    g2 = {s: g for g, syms in json.loads((LOCK_V2 / "universe_groups.json").read_text()).items() for s in syms}
    m = load_market_data()
    m["symbol"] = m.symbol.astype(str); m["date"] = pd.to_datetime(m["date"])
    m = m.sort_values(["symbol", "date"]).reset_index(drop=True)
    g = m.groupby("symbol", sort=False)
    m["fwd_ret"] = g["close"].shift(-H) / m.close - 1.0
    m["y"] = (m.fwd_ret > 0).astype("float"); m.loc[m.fwd_ret.isna(), "y"] = np.nan
    m["group"] = m.symbol.map(g2)
    m["eligible"] = m.close.ge(100.0) & m.atm_iv.notna()
    m["reg5"] = regime(np.sign(m.ret_5d.values), np.sign(m.fut_oi_chg_5.values))
    m["volbk"] = np.where(m.realized_vol_20 > m.groupby("date").realized_vol_20.transform("median"), "hi", "lo")
    full = [c for c in m.columns if c not in ID and pd.api.types.is_numeric_dtype(m[c])
            and not any(h in c.lower() for h in LEAK)]
    fsets = {"ALL": full, "NO_CAL": [c for c in full if c not in CAL],
             "MKT_MOM": [c for c in MKT_MOM if c in m.columns], "OI_FLOW": [c for c in OI_FLOW if c in m.columns]}
    return m, fsets, g2


def wf_monthly(m, feats, win, train_univ):
    from catboost import CatBoostClassifier
    parts = []
    for mo in EVAL_MONTHS:
        ms, me = mo.start_time, mo.end_time
        cut = ms - pd.Timedelta(days=EMBARGO)
        lo = pd.Timestamp("2010-01-01") if win is None else cut - pd.Timedelta(days=win)
        tr = m[(m.date >= lo) & (m.date < cut) & m.eligible & m.y.notna()]
        if train_univ == "B":
            tr = tr[tr.group == GB]
        ev = m[(m.date >= ms) & (m.date <= me) & m.eligible & (m.group == GB) & m.y.notna()]
        if len(tr) < 1500 or ev.empty:
            continue
        mdl = CatBoostClassifier(**CB, loss_function="Logloss").fit(tr[feats], tr.y.astype(int))
        e = ev[["date", "symbol", "group", "y", "fwd_ret"]].copy()
        e["p"] = mdl.predict_proba(ev[feats])[:, 1]; e["mo"] = str(mo)
        parts.append(e)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def wf_rule_monthly(m):
    parts = []
    for mo in EVAL_MONTHS:
        ms, me = mo.start_time, mo.end_time
        cut = ms - pd.Timedelta(days=EMBARGO)
        tr = m[(m.date >= cut - pd.Timedelta(days=270)) & (m.date < cut) & m.eligible & m.y.notna()]
        ev = m[(m.date >= ms) & (m.date <= me) & m.eligible & (m.group == GB) & m.y.notna()]
        if tr.empty or ev.empty:
            continue
        tab = tr.groupby(["reg5", "volbk"]).y.mean()
        e = ev[["date", "symbol", "group", "y", "fwd_ret"]].copy()
        e["p"] = ev.set_index(["reg5", "volbk"]).index.map(tab).astype(float)
        e["p"] = e["p"].fillna(0.5); e["mo"] = str(mo)
        parts.append(e)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def option_legs():
    oc = pd.read_csv(OPTC, parse_dates=["date"]); oc["symbol"] = oc.symbol.astype(str)
    oc = oc[oc.dte >= EMBARGO]
    near = oc[oc.groupby(["date", "symbol"]).dte.transform("min") == oc.dte].copy()
    near["dC"] = (near.moneyness - 2).abs(); near["dP"] = (near.moneyness + 2).abs()
    c = (near[near.ot == "CE"].sort_values(["date", "symbol", "dC"]).groupby(["date", "symbol"]).first()
         [["held_ratio"]].rename(columns={"held_ratio": "c_held"}))
    p = (near[near.ot == "PE"].sort_values(["date", "symbol", "dP"]).groupby(["date", "symbol"]).first()
         [["held_ratio"]].rename(columns={"held_ratio": "p_held"}))
    return c.join(p, how="inner").reset_index()


def boot(x, n=2000, seed=7):
    rng = np.random.default_rng(seed); x = np.asarray(x)
    if len(x) < 10:
        return np.nan, np.nan, np.nan
    b = np.array([x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)])
    return x.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5)


def evaluate(preds, bookB, legs):
    from sklearn.metrics import roc_auc_score
    d = preds.dropna(subset=["p", "y", "fwd_ret"])
    auc = roc_auc_score(d.y, d.p) if d.y.nunique() > 1 else np.nan
    call = d.p > 0.5
    hit = float(((call & (d.fwd_ret > 0)) | (~call & (d.fwd_ret < 0))).mean())
    ic = float(d.p.corr(d.fwd_ret, "spearman"))
    t = bookB.merge(preds[["date", "symbol", "p"]], on=["date", "symbol"]).merge(legs, on=["date", "symbol"])
    for col in ("c_held", "p_held"):
        t[col] = t[col].clip(0, 1 + RET_CAP)
    t["held"] = np.where(t.p > 0.5, t.c_held, t.p_held) - 1 - COST
    ev, lo, hi = boot(t.held.values)
    return dict(n_uni=int(len(d)), auc=round(float(auc), 4), hit=round(hit, 4), ic=round(ic, 4),
                n_bk=int(len(t)), ev=round(float(ev), 4), lo=round(float(lo), 4), hi=round(float(hi), 4),
                win=round(float((t.held > 0).mean()), 3)), t


def main():
    m, fsets, g2 = load_panel()
    book = pd.read_csv(BOOK5, parse_dates=["date"]); book["symbol"] = book.symbol.astype(str)
    bookB = book[(~book.live) & (book.group == GB)][["date", "symbol"]]
    legs = option_legs()
    print(f"feat sizes: " + ", ".join(f"{k}={len(v)}" for k, v in fsets.items()))
    print(f"group-B book picks {len(bookB)} (all yrs) | eval months {EVAL_MONTHS[0]}..{EVAL_MONTHS[-1]}\n")

    grid = [(f"{fs}/{wn}/trAll", fs, wv, "ALL") for fs in ["ALL", "NO_CAL", "MKT_MOM", "OI_FLOW"] for wn, wv in WINDOWS.items()]
    grid += [(f"ALL/{wn}/trB", "ALL", wv, "B") for wn, wv in [("6m", 180), ("9m", 270), ("expand", None)]]
    res, mtab, top = {}, [], None
    for name, fs, wv, tu in grid:
        preds = wf_monthly(m, fsets[fs], wv, tu)
        if preds.empty:
            continue
        r, t = evaluate(preds, bookB, legs); res[name] = r
        mtab.append((name, r)); preds["cfg"] = name
        if top is None or r["ev"] > top[1]["ev"]:
            top = (name, r, preds)
    # regime rule
    pr = wf_rule_monthly(m)
    if not pr.empty:
        r, _ = evaluate(pr, bookB, legs); res["REGIME_rule/9m"] = r; mtab.append(("REGIME_rule/9m", r))

    print("=" * 104)
    print("GROUP-B 2026 MONTHLY-RETRAIN SWEEP  (eval Jan->late-May 2026; univ AUC/hit/IC ; book-B option EV held, 95% CI)")
    print(f"{'config':22s} | {'uni n':>6s} {'AUC':>6s} {'hit':>6s} {'IC':>7s} | {'bk n':>5s} {'held EV':>8s} {'95% CI':>17s} {'win':>5s}")
    for name, r in sorted(mtab, key=lambda x: -x[1]["ev"]):
        print(f"{name:22s} | {r['n_uni']:>6d} {r['auc']:>6.3f} {r['hit']:>6.3f} {r['ic']:>+7.3f} | "
              f"{r['n_bk']:>5d} {r['ev']:>+8.3f} [{r['lo']:>+6.3f},{r['hi']:>+6.3f}] {r['win']:>5.2f}")

    if top is not None:
        name, r, preds = top
        print(f"\nbest by EV: {name}  -> per-month progression (does monthly retrain adapt into 2026?):")
        tb = bookB.merge(preds, on=["date", "symbol"]).merge(legs, on=["date", "symbol"])
        for col in ("c_held", "p_held"):
            tb[col] = tb[col].clip(0, 1 + RET_CAP)
        tb["held"] = np.where(tb.p > 0.5, tb.c_held, tb.p_held) - 1 - COST
        tb["correct"] = ((tb.p > 0.5) & (tb.fwd_ret > 0)) | ((tb.p <= 0.5) & (tb.fwd_ret < 0))
        for mo in sorted(tb.mo.unique()):
            s = tb[tb.mo == mo]
            print(f"   {mo}: n={len(s):3d}  dir-hit {s.correct.mean():.3f}  held EV {s.held.mean():+.3f}  call-share {(s.p>0.5).mean():.2f}")
    json.dump(res, open(OUT / "groupB_monthly.json", "w"), indent=2)
    print(f"\nsaved -> {OUT/'groupB_monthly.json'}")


if __name__ == "__main__":
    main()
