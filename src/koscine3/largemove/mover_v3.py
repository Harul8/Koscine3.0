"""PRODUCTION v3 — mover-precision books (direction-agnostic LARGE-MOVE signals for option buying), TWO horizons:
  - 5-day  (mover_v3_book_5d.csv): peak excursion over t+1..t+5
  - 1-day  (mover_v3_book_1d.csv): next-day peak excursion (t+1)

Each: ONE ranked list, top-3/day across all IV (the big movers are high-IV), ATM+2% liquidity-gated (>=1000
contracts both legs), every signal cost-tagged (LOW/HIGH-IV + ATM+2% premium + 'expensive: needs big move').
The user takes DIRECTION offline (coin flip) and decides whether each expensive premium is worth it.

Engine: ensemble = rank-avg( CatBoost clf P(move_mag>=big_thr) + CatBoost reg move_mag + atm_iv ), quarterly
walk-forward retrain, leak-safe. Reads market data + F&O bhavcopy READ-ONLY; writes locks/prod_largemove_v3/.
Does NOT touch v1/v2.

    python -m koscine3.largemove.mover_v3
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from koscine3.data.sources import load_market_data
from koscine3.largemove.mover_v2 import LOCK_V2          # read-only: shared universe_groups.json

ROOT = LOCK_V2.parents[1]
LOCK_V3 = ROOT / "locks" / "prod_largemove_v3"
sys.path.insert(0, str(ROOT / "analysis"))

VERSION = "prod_largemove_v3"
K = 3                       # signals per day
MIN_CONTRACTS = 1000        # ATM+2% liquidity gate (both legs)
MIN_UNDERLYING = 100.0
MIN_DTE = 6
TRAIN_DAYS = 1100
QUARTERS = pd.period_range("2024Q1", pd.Timestamp.today().to_period("Q"), freq="Q")
# (horizon name, window days, big-move clf threshold, reg clip)
HORIZONS = [("5d", 5, 0.08, 0.40), ("1d", 1, 0.04, 0.20)]
ID = {"date", "symbol", "open", "high", "low", "close", "volume", "move_mag_5", "move_mag_1", "in_univ", "eligible", "group"}
LEAK = ("future", "fwd", "next", "label", "tomorrow", "ahead", "adverse", "move_5d", "move_1d", "move_3d",
        "move_10d", "expansion", "volclean", "outcome")
CB = dict(iterations=600, depth=6, learning_rate=0.03, l2_leaf_reg=6.0, random_seed=7,
          allow_writing_files=False, verbose=False, task_type="GPU", devices="0")


def load_panel():
    g2 = {s: g for g, syms in json.loads((LOCK_V2 / "universe_groups.json").read_text()).items() for s in syms}
    m = load_market_data()
    m["symbol"] = m["symbol"].astype(str)
    m["date"] = pd.to_datetime(m["date"])
    m = m.sort_values(["symbol", "date"]).reset_index(drop=True)
    g = m.groupby("symbol", sort=False)

    def fwd_move(W):
        hi = pd.concat([g["high"].shift(-i) for i in range(1, W + 1)], axis=1).max(axis=1)
        lo = pd.concat([g["low"].shift(-i) for i in range(1, W + 1)], axis=1).min(axis=1)
        return np.maximum(hi / m.close - 1.0, m.close / lo - 1.0)

    m["move_mag_5"] = fwd_move(5)
    m["move_mag_1"] = fwd_move(1)
    m["in_univ"] = m.symbol.isin(g2)
    m["group"] = m.symbol.map(g2)
    m["eligible"] = m.close.ge(MIN_UNDERLYING) & m.atm_iv.notna()
    feats = [c for c in m.columns if c not in ID and pd.api.types.is_numeric_dtype(m[c])
             and not any(h in c.lower() for h in LEAK)]
    for c in ["atm_iv", "realized_vol_20", "atr_pct_14", "donchian_width_20"]:
        if c in m.columns:
            m[c + "_xrank"] = m.groupby("date")[c].rank(pct=True); feats.append(c + "_xrank")
    return m, feats, g2


def _wf(m, feats, mag_col, clf, big_thr, reg_clip):
    from catboost import CatBoostClassifier, CatBoostRegressor
    parts = []
    for q in QUARTERS:
        cut = q.start_time - pd.Timedelta(days=6)
        tr = m[(m.date < cut) & (m.date >= cut - pd.Timedelta(days=TRAIN_DAYS * 1.6)) & m.eligible & m[mag_col].notna()].dropna(subset=["atm_iv"])
        ev = m[(m.date >= q.start_time) & (m.date <= q.end_time) & m.eligible & m.in_univ].copy()  # incl. live tail
        if len(tr) < 5000 or ev.empty:
            continue
        if clf:
            mdl = CatBoostClassifier(**CB, loss_function="Logloss").fit(tr[feats], (tr[mag_col] >= big_thr).astype(int))
            ev["s"] = mdl.predict_proba(ev[feats])[:, 1]
        else:
            mdl = CatBoostRegressor(**CB, loss_function="RMSE").fit(tr[feats], tr[mag_col].clip(0, reg_clip))
            ev["s"] = mdl.predict(ev[feats])
        parts.append(ev[["date", "symbol", "s"]])
    return pd.concat(parts, ignore_index=True)


def score(m, feats, mag_col, big_thr, reg_clip):
    clf = _wf(m, feats, mag_col, True, big_thr, reg_clip).rename(columns={"s": "s_clf"})
    reg = _wf(m, feats, mag_col, False, big_thr, reg_clip).rename(columns={"s": "s_reg"})
    e = m.loc[m.eligible & m.in_univ & (m.date >= QUARTERS[0].start_time),
              ["date", "symbol", "group", "atm_iv", mag_col]].rename(columns={mag_col: "move_mag"}).copy()
    e = e.merge(clf, on=["date", "symbol"]).merge(reg, on=["date", "symbol"])
    e["s_iv"] = e["atm_iv"]
    for c in ("s_clf", "s_reg", "s_iv"):
        e[c + "r"] = e.groupby("date")[c].rank(pct=True)
    e["ens"] = e[["s_clfr", "s_regr", "s_ivr"]].mean(axis=1)
    e["conv_pctile"] = e.groupby("date")["ens"].rank(pct=True)
    return e


def liquidity(dates, g2):
    from options_bhavcopy import load_bhavcopy
    px = load_market_data(columns=["date", "symbol", "close"])
    px["symbol"] = px["symbol"].astype(str); px["date"] = pd.to_datetime(px["date"])
    pxd = {(r.date, r.symbol): r.close for r in px.itertuples(index=False)}
    rows = []
    for d in pd.DatetimeIndex(sorted(pd.to_datetime(pd.Index(dates)).unique())):
        bc = load_bhavcopy(d)
        if bc is None or bc.empty:
            continue
        bcx = bc.dropna(subset=["strike", "expiry"]); bcx = bcx.assign(expiry=pd.to_datetime(bcx.expiry))
        for sym, sub in bcx.groupby("symbol"):
            if sym not in g2:
                continue
            U = pxd.get((d, sym))
            if U is None or U < MIN_UNDERLYING:
                continue
            exps = sorted(e for e in sub.expiry.dropna().unique() if (pd.Timestamp(e) - d).days >= MIN_DTE)
            if not exps:
                continue
            ch = sub[sub.expiry.eq(pd.Timestamp(exps[0]))]
            ce, pe = ch[ch.opt_type.eq("CE")], ch[ch.opt_type.eq("PE")]
            if ce.empty or pe.empty:
                continue
            ck = min(ce.strike.unique(), key=lambda s: abs(s - U * 1.02))
            pk = min(pe.strike.unique(), key=lambda s: abs(s - U * 0.98))
            cr, pr = ce[ce.strike.eq(ck)], pe[pe.strike.eq(pk)]
            if cr.empty or pr.empty:
                continue
            rows.append({"date": d, "symbol": sym, "c_vol": float(cr.vol.iloc[0]), "p_vol": float(pr.vol.iloc[0]),
                         "c_prem": float(cr.close.iloc[0]), "p_prem": float(pr.close.iloc[0])})
    return pd.DataFrame(rows)


def build_book(e, liq, horizon):
    e = e.merge(liq, on=["date", "symbol"], how="left")
    e["atm2_contracts"] = e[["c_vol", "p_vol"]].min(axis=1)
    e["liquid"] = e.atm2_contracts >= MIN_CONTRACTS
    e["iv_group"] = np.where(e.atm_iv > e.groupby("date").atm_iv.transform("median"), "HIGH", "LOW")
    e["expensive"] = e.iv_group == "HIGH"
    el = e[e.liquid].copy()
    el["rank"] = el.groupby(["date", "group"])["ens"].rank(ascending=False, method="first")  # per group: A mega-cap / B movers (kept separate, as v2)
    book = el[el["rank"] <= K].copy()
    book["horizon"] = horizon
    book["live"] = book.move_mag.isna()
    book["needs_big_move"] = book.expensive
    # Persist the raw regression forecast as well as the rank ensemble.  The
    # selector deliberately remains `ens`; `pred_move_pct` is an explainable
    # horizon-matched sizing estimate for the UI and post-trade calibration.
    book["pred_move_pct"] = book["s_reg"] * 100
    book["prob_big_move"] = book["s_clf"]
    cols = ["date", "horizon", "group", "symbol", "rank", "iv_group", "expensive", "needs_big_move", "atm_iv",
            "atm2_contracts", "c_prem", "p_prem", "ens", "conv_pctile", "pred_move_pct", "prob_big_move",
            "move_mag", "live"]
    return book.sort_values(["date", "rank"])[cols]


def main():
    LOCK_V3.mkdir(parents=True, exist_ok=True)
    m, feats, g2 = load_panel()
    dates = m.loc[m.eligible & m.in_univ & (m.date >= QUARTERS[0].start_time), "date"].unique()
    liq = liquidity(dates, g2)
    manifest = {"version": VERSION, "selector": "ensemble(clf P(move>=thr)+reg move_mag+atm_iv), top-3/GROUP/day (A mega-cap + B movers), all-IV cost-tagged",
                "rules": {"signals_per_group_per_day": K, "groups": "A_mcap30 + B_turn35 kept separate (as v2)",
                          "atm2_min_contracts": MIN_CONTRACTS, "iv_split": "within-day median cost tag",
                          "direction": "agnostic (user offline)"}, "horizons": {}}
    for hname, W, big_thr, reg_clip in HORIZONS:
        e = score(m, feats, f"move_mag_{W}", big_thr, reg_clip)
        book = build_book(e, liq, hname)
        book.to_csv(LOCK_V3 / f"mover_v3_book_{hname}.csv", index=False)
        done = book[~book.live]
        manifest["horizons"][hname] = {"window_days": W, "big_thr": big_thr, "rows": int(len(book)),
                                       "dates": [str(book.date.min().date()), str(book.date.max().date())],
                                       "hit_ge_6pct": round(float((done.move_mag >= 0.06).mean()), 3),
                                       "hit_ge_4pct": round(float((done.move_mag >= 0.04).mean()), 3),
                                       "pct_expensive": round(float(done.expensive.mean()), 3)}
        print(f"v3 {hname}: {len(book)} rows | hit>=6% {manifest['horizons'][hname]['hit_ge_6pct']} "
              f"hit>=4% {manifest['horizons'][hname]['hit_ge_4pct']} | -> mover_v3_book_{hname}.csv")
    (LOCK_V3 / "universe_groups.json").write_text((LOCK_V2 / "universe_groups.json").read_text())
    (LOCK_V3 / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"saved lock -> {LOCK_V3}")


if __name__ == "__main__":
    main()
