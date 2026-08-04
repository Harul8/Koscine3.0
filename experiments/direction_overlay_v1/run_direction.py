"""direction_overlay_v1 — OVERLAY vs FULL-RETRAIN for a 5-day PUT/CALL directional lean. (contained; no PROD touch)

Question (user): for adding a direction (CALL/PUT) call to the direction-agnostic v3 book, is a light
recency-trained OVERLAY on the focused directional features better, or a COMPLETE retrain on ALL features?

Both predict 5-day forward direction y=1[close[t+5]>close[t]], purged+embargoed quarterly walk-forward
2024Q1-2026Q2, eval on the v3 65-name A/B universe. Configs (2x2 feature-set x train-window, + regime rule):
  FULL_expand   : all non-leak feats, long (~1760d) window     <- "complete retrain with new features"
  OVERLAY_recent: focused directional feats, recent (365d) win  <- "adaptive overlay" (regime-flip aware)
  FULL_recent   : all feats, recent window     (ablation: window axis)
  OVERLAY_expand: focused feats, long window    (ablation: feature axis)
  REGIME_rule   : transparent P(up | OI-regime x vol-bucket), recency-estimated (interpretable baseline)
Metrics: OOS AUC / directional hit-rate / rank-IC, overall and by year (the regime flip lives in 2026).
Then REAL option premium EV: apply each lean to the v3 5d book, buy ATM+2% CALL or ATM-2% PUT (real bhavcopy
forward held/peak ratios from option_move_v1), net 3% cost, vs coin-flip / always-call / always-put / anti.

    set PYTHONPATH=src && python experiments/direction_overlay_v1/run_direction.py
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

QUARTERS = pd.period_range("2024Q1", "2026Q2", freq="Q")
H = 5                       # forward horizon (days)
EMBARGO = 6                 # purge gap (>= H+1)
RECENT, LONG = 365, 1760    # train-window lengths (calendar days)
COST = 0.03                 # round-trip premium cost (fraction of premium)
RET_CAP = 5.0              # cap single-leg return at +500% (penny-option guard)

CB = dict(iterations=500, depth=5, learning_rate=0.03, l2_leaf_reg=6.0, random_seed=7,
          allow_writing_files=False, verbose=False, task_type="GPU", devices="0")

FOCUSED = [
    # momentum / trend
    "ret_1d", "ret_5d", "ret_20d", "nifty_ret_1d", "nifty_ret_5d", "rel_ret_5d_vs_nifty", "ret_5d_cs_rank",
    "ret_20d_cs_rank", "sector_ret_5d", "stock_rel_sector_ret_5d", "consec_up_days", "consec_down_days",
    "pos_day_share_20d", "ema_20_slope_5d", "ema_50_slope_5d", "adx_14", "di_diff", "close_sma50_dist",
    # OI positioning
    "fut_chg_oi", "fut_oi_chg_5", "fut_oi_z_60d", "oi_buildup_ratio", "oi_long_buildup", "oi_short_buildup",
    "oi_long_unwind", "oi_short_unwind", "price_oi_divergence", "oi_acceleration",
    # sentiment / positioning
    "pcr_oi", "pcr_oi_chg_5", "pcr_vol", "pcr_vol_chg_5", "max_pain_dist", "call_wall_1_dist", "put_wall_1_dist",
    "iv_skew_norm", "iv_skew_chg_5d", "put_call_iv_skew",
    # regime / breadth
    "realized_vol_20", "nifty_realized_vol_20", "atr_pct_14", "mkt_pct_above_sma50", "mkt_advance_ratio",
    "gap_pct", "delivery_pct", "delivery_pct_chg_5",
]
LEAK = ("future", "fwd", "next", "ahead", "label", "adverse", "up_move", "down_move", "expansion",
        "volclean", "outcome", "entry_1d", "_date", "tomorrow")
ID = {"date", "symbol", "open", "high", "low", "close", "last", "prev_close", "volume", "group",
      "in_univ", "eligible", "y", "fwd_ret", "per", "reg5", "volbk"}


def regime(dp, doi):
    r = np.full(len(dp), "flat", dtype=object)
    r[(dp > 0) & (doi > 0)] = "LongBuildup"
    r[(dp < 0) & (doi > 0)] = "ShortBuildup"
    r[(dp > 0) & (doi < 0)] = "ShortCovering"
    r[(dp < 0) & (doi < 0)] = "LongUnwinding"
    return r


def load_panel():
    g2 = {s: g for g, syms in json.loads((LOCK_V2 / "universe_groups.json").read_text()).items() for s in syms}
    m = load_market_data()
    m["symbol"] = m["symbol"].astype(str); m["date"] = pd.to_datetime(m["date"])
    m = m.sort_values(["symbol", "date"]).reset_index(drop=True)
    g = m.groupby("symbol", sort=False)
    m["fwd_ret"] = g["close"].shift(-H) / m.close - 1.0
    m["y"] = (m.fwd_ret > 0).astype("float")          # NaN where no forward
    m.loc[m.fwd_ret.isna(), "y"] = np.nan
    m["in_univ"] = m.symbol.isin(g2)
    m["group"] = m.symbol.map(g2)
    m["eligible"] = m.close.ge(100.0) & m.atm_iv.notna()
    m["per"] = m.date.dt.year.astype(str)
    m["reg5"] = regime(np.sign(m.ret_5d.values), np.sign(m.fut_oi_chg_5.values))
    m["volbk"] = np.where(m.realized_vol_20 > m.groupby("date").realized_vol_20.transform("median"), "hi", "lo")
    full = [c for c in m.columns if c not in ID and pd.api.types.is_numeric_dtype(m[c])
            and not any(h in c.lower() for h in LEAK)]
    focused = [c for c in FOCUSED if c in m.columns]
    return m, full, focused, g2


def wf_cb(m, feats, window):
    from catboost import CatBoostClassifier
    parts = []
    for q in QUARTERS:
        cut = q.start_time - pd.Timedelta(days=EMBARGO)
        tr = m[(m.date < cut) & (m.date >= cut - pd.Timedelta(days=window)) & m.eligible & m.y.notna()]
        ev = m[(m.date >= q.start_time) & (m.date <= q.end_time) & m.eligible & m.in_univ & m.y.notna()].copy()
        if len(tr) < 3000 or ev.empty:
            continue
        mdl = CatBoostClassifier(**CB, loss_function="Logloss").fit(tr[feats], tr.y.astype(int))
        ev["p"] = mdl.predict_proba(ev[feats])[:, 1]
        parts.append(ev[["date", "symbol", "group", "per", "y", "fwd_ret", "p"]])
    return pd.concat(parts, ignore_index=True)


def wf_rule(m, window):
    parts = []
    for q in QUARTERS:
        cut = q.start_time - pd.Timedelta(days=EMBARGO)
        tr = m[(m.date < cut) & (m.date >= cut - pd.Timedelta(days=window)) & m.eligible & m.y.notna()]
        ev = m[(m.date >= q.start_time) & (m.date <= q.end_time) & m.eligible & m.in_univ & m.y.notna()].copy()
        if len(tr) < 3000 or ev.empty:
            continue
        tab = tr.groupby(["reg5", "volbk"]).y.mean()
        ev["p"] = ev.set_index(["reg5", "volbk"]).index.map(tab).astype(float)
        ev["p"] = ev["p"].fillna(0.5)
        parts.append(ev[["date", "symbol", "group", "per", "y", "fwd_ret", "p"]])
    return pd.concat(parts, ignore_index=True)


def metrics(preds):
    from sklearn.metrics import roc_auc_score
    d = preds.dropna(subset=["p", "y", "fwd_ret"])
    rows = {}
    for per, sub in [("ALL", d)] + [(p, d[d.per == p]) for p in ("2024", "2025", "2026")]:
        if len(sub) < 100 or sub.y.nunique() < 2:
            continue
        call = sub.p > 0.5
        hit = ((call & (sub.fwd_ret > 0)) | (~call & (sub.fwd_ret < 0))).mean()
        rows[per] = dict(auc=round(roc_auc_score(sub.y, sub.p), 4), hit=round(float(hit), 4),
                         ic=round(float(sub.p.corr(sub.fwd_ret, "spearman")), 4), n=int(len(sub)))
    return rows


def option_legs():
    oc = pd.read_csv(OPTC, parse_dates=["date"])
    oc["symbol"] = oc.symbol.astype(str)
    oc = oc[oc.dte >= EMBARGO].copy()
    near = oc[oc.groupby(["date", "symbol"]).dte.transform("min") == oc.dte].copy()
    near["dC"] = (near.moneyness - 2).abs(); near["dP"] = (near.moneyness + 2).abs()
    c = (near[near.ot == "CE"].sort_values(["date", "symbol", "dC"]).groupby(["date", "symbol"]).first()
         [["held_ratio", "peak_ratio"]].rename(columns={"held_ratio": "c_held", "peak_ratio": "c_peak"}))
    p = (near[near.ot == "PE"].sort_values(["date", "symbol", "dP"]).groupby(["date", "symbol"]).first()
         [["held_ratio", "peak_ratio"]].rename(columns={"held_ratio": "p_held", "peak_ratio": "p_peak"}))
    return c.join(p, how="inner").reset_index()


def ev_eval(preds, book, legs):
    d = (book.merge(preds[["date", "symbol", "p"]], on=["date", "symbol"], how="inner")
             .merge(legs, on=["date", "symbol"], how="inner"))
    for col in ("c_held", "p_held", "c_peak", "p_peak"):
        d[col] = d[col].clip(0, 1 + RET_CAP)
    d["lean_call"] = d.p > 0.5
    held = np.where(d.lean_call, d.c_held, d.p_held) - 1 - COST
    peak = np.where(d.lean_call, d.c_peak, d.p_peak) - 1 - COST
    anti = np.where(~d.lean_call, d.c_held, d.p_held) - 1 - COST
    coin = 0.5 * ((d.c_held - 1) + (d.p_held - 1)) - COST
    allc = (d.c_held - 1) - COST; allp = (d.p_held - 1) - COST
    d = d.assign(held=held, peak=peak, anti=anti, coin=coin, allc=allc, allp=allp)
    out = {}
    for per, sub in [("ALL", d)] + [(p, d[d.per == p]) for p in ("2024", "2025", "2026")]:
        if len(sub) < 30:
            continue
        out[per] = dict(n=int(len(sub)), pct_call=round(float(sub.lean_call.mean()), 3),
                        held_ev=round(float(sub.held.mean()), 4), held_win=round(float((sub.held > 0).mean()), 3),
                        peak_ev=round(float(sub.peak.mean()), 4),
                        coin_ev=round(float(sub.coin.mean()), 4), anti_ev=round(float(sub.anti.mean()), 4),
                        allcall_ev=round(float(sub.allc.mean()), 4), allput_ev=round(float(sub.allp.mean()), 4))
    return out


def main():
    m, full, focused, g2 = load_panel()
    print(f"panel rows {len(m):,} | full feats {len(full)} | focused feats {len(focused)} | "
          f"univ {len(g2)} | base P(up5) eligible {(m[m.eligible].y.mean()):.3f}")
    book = pd.read_csv(BOOK5, parse_dates=["date"]); book["symbol"] = book.symbol.astype(str)
    book = book[~book.live][["date", "symbol", "group", "per"]] if "per" in book else book[~book.live][["date", "symbol", "group"]]
    book["per"] = book.date.dt.year.astype(str)
    legs = option_legs()
    print(f"book(done) {len(book):,} | option legs {len(legs):,} ({legs.date.min().date()}..{legs.date.max().date()})")

    configs = [("FULL_expand", "cb", full, LONG), ("OVERLAY_recent", "cb", focused, RECENT),
               ("FULL_recent", "cb", full, RECENT), ("OVERLAY_expand", "cb", focused, LONG),
               ("REGIME_rule", "rule", None, RECENT)]
    allm, allev, store = {}, {}, {}
    for name, kind, feats, win in configs:
        preds = wf_cb(m, feats, win) if kind == "cb" else wf_rule(m, win)
        store[name] = preds
        allm[name] = metrics(preds)
        allev[name] = ev_eval(preds, book, legs)
        a = allm[name]
        print(f"\n[{name}]  feats={'-' if feats is None else len(feats)} win={win}d")
        for per in ("ALL", "2024", "2025", "2026"):
            if per in a:
                r = a[per]; print(f"   {per:4s} AUC {r['auc']:.3f}  hit {r['hit']:.3f}  IC {r['ic']:+.3f}  n={r['n']:,}")

    # save + summary tables
    json.dump({"metrics": allm, "ev": allev}, open(OUT / "direction_results.json", "w"), indent=2)
    pd.concat([s.assign(cfg=n) for n, s in store.items()], ignore_index=True).to_csv(OUT / "direction_preds.csv", index=False)

    print("\n" + "=" * 92)
    print("DIRECTIONAL ACCURACY (OOS) — AUC / hit-rate / IC")
    print(f"{'config':16s} | {'ALL AUC':>8s} {'hit':>6s} | {'2024 AUC':>8s} | {'2025 AUC':>8s} | {'2026 AUC':>8s} {'hit':>6s} {'IC':>7s}")
    for n in allm:
        a = allm[n]
        def g(p, k, f="{:.3f}"): return (f.format(a[p][k]) if p in a and k in a[p] else "  -  ")
        print(f"{n:16s} | {g('ALL','auc'):>8s} {g('ALL','hit'):>6s} | {g('2024','auc'):>8s} | {g('2025','auc'):>8s} | "
              f"{g('2026','auc'):>8s} {g('2026','hit'):>6s} {g('2026','ic','{:+.3f}'):>7s}")

    print("\n" + "=" * 92)
    print("OPTION PREMIUM EV on v3 5d book — buy leaned ATM+/-2% leg, held 5d, net 3% cost")
    print(f"{'config':16s} | {'ALL held_ev':>11s} {'win':>5s} | {'coin':>7s} {'anti':>7s} | {'2026 held':>9s} {'2026 coin':>9s} {'peak':>7s} n")
    for n in allev:
        e = allev[n]
        def gg(p, k): return ("{:+.3f}".format(e[p][k]) if p in e and k in e[p] else "  -  ")
        n26 = e.get("2026", {}).get("n", "-")
        print(f"{n:16s} | {gg('ALL','held_ev'):>11s} {gg('ALL','held_win'):>5s} | {gg('ALL','coin_ev'):>7s} {gg('ALL','anti_ev'):>7s} | "
              f"{gg('2026','held_ev'):>9s} {gg('2026','coin_ev'):>9s} {gg('2026','peak_ev'):>7s} {n26}")
    print(f"\nsaved -> {OUT/'direction_results.json'} , direction_preds.csv")


if __name__ == "__main__":
    main()
