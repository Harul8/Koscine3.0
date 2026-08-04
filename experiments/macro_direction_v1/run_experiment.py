"""macro_direction_v1 — next-day DIRECTION with cross-asset macro features, purged/embargoed walk-forward.

Reads PROD features READ-ONLY (load_market_data) for the A/B universe, merges leak-safe macro features
(macro.py), builds a next-day direction label (labels.py), and runs a PURGED + EMBARGOED quarterly
walk-forward XGBoost classifier. Compares price-only baseline vs +macro and reports acc / AUC / Brier,
high-conviction precision, per-quarter stability and feature importances. Writes results/ in this folder.

    python experiments/macro_direction_v1/fetch_macro.py        # 1) external EOD fetch (internet)
    set PYTHONPATH=src
    python experiments/macro_direction_v1/run_experiment.py     # 2) experiment (GPU XGBoost)

PROD is never imported or written. Universe + market data are read-only.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))           # read-only: koscine3.data.sources (the allowed dependency)
sys.path.insert(0, str(HERE))

from koscine3.data.sources import load_market_data            # noqa: E402
from labels import realized_fwd, signed_target                # noqa: E402
from macro import macro_features, MACRO_COLS                  # noqa: E402

RESULTS = HERE / "results"
LOCK_V2 = ROOT / "locks" / "prod_largemove_v2"               # read-only (universe list only)

HORIZON = 1
ENTRY = "open"             # execution: ~7AM IST t+1 pre-open refresh -> trade the t+1 OPEN. open->close is the
                          # capturable return (the overnight gap is already in the open). "close" overstates via the gap.
DEAD_BAND = 0.010          # drop |next-day move| <= 1.0% from TRAINING (threshold labeling)
EMBARGO_DAYS = 5
TOP_FRAC = 0.10            # high-conviction subset = top decile by |proba - 0.5|
QUARTERS = pd.period_range("2024Q1", "2026Q2", freq="Q")

ID_COLS = {"date", "symbol", "open", "high", "low", "close", "volume", "group", "eligible",
           "target", "in_band", "fwd_ret", "y"}
LEAK_HINTS = ("future", "fwd", "next", "target", "label", "tomorrow", "_t1", "ahead",
              "adverse", "move_5d", "move_1d", "move_3d", "move_10d", "expansion", "volclean", "outcome")
LEAK_CORR = 0.05            # |corr(feature, forward target)| above this = a forward-looking leak (direction is a coin flip)

XGB = dict(n_estimators=400, max_depth=4, learning_rate=0.03, subsample=0.8, colsample_bytree=0.7,
           reg_lambda=5.0, min_child_weight=30, tree_method="hist", device="cuda",
           objective="binary:logistic", eval_metric="logloss", verbosity=0)


def universe() -> dict[str, str]:
    groups = json.loads((LOCK_V2 / "universe_groups.json").read_text())
    return {s: g for g, syms in groups.items() for s in syms}


def load_panel():
    g2 = universe()
    m = load_market_data()                                   # all features, read-only
    m["symbol"] = m["symbol"].astype(str)
    m = m[m.symbol.isin(g2)].copy()
    m["date"] = pd.to_datetime(m["date"])
    m["group"] = m["symbol"].map(g2)
    m["eligible"] = m["close"].ge(100) & m["atm_iv"].notna()
    m = m.sort_values(["symbol", "date"]).reset_index(drop=True)

    m["fwd_ret"] = realized_fwd(m, HORIZON, entry=ENTRY)
    m["target"] = signed_target(m["fwd_ret"], DEAD_BAND)     # training label (dead-band middle dropped)
    m["y"] = (m["fwd_ret"] > 0).astype("Int64")              # eval truth (full set)

    native = [c for c in m.columns
              if c not in ID_COLS and pd.api.types.is_numeric_dtype(m[c])
              and not any(h in c.lower() for h in LEAK_HINTS)
              and m[c].notna().any()]

    # leak-guard (name-independent): direction is ~coin flip, so a legit backward feature has |corr| with
    # the forward target < LEAK_CORR. Anything above is a forward-looking K2 label/target column -> drop it.
    elig = m[m.eligible & m.fwd_ret.notna()]
    fr = elig["fwd_ret"]
    dropped = []
    for c in native:
        cc = abs(elig[c].replace([np.inf, -np.inf], np.nan).corr(fr))
        if cc > LEAK_CORR:
            dropped.append((c, round(float(cc), 3)))
    if dropped:
        bad = {c for c, _ in dropped}
        native = [c for c in native if c not in bad]
        print(f"  [leak-guard] dropped {len(dropped)} forward-looking feature(s): {dropped}")

    mac = macro_features(m["date"].unique())
    m = m.merge(mac, on="date", how="left")
    macro = [c for c in MACRO_COLS if c in m.columns and m[c].notna().any()]
    return m, native, macro


def evaluate(m, feats):
    from xgboost import XGBClassifier
    from sklearn.metrics import accuracy_score, roc_auc_score

    per_q, imp_sum, evs, n_folds = [], {}, [], 0
    for q in QUARTERS:
        cut = q.start_time - pd.Timedelta(days=HORIZON + EMBARGO_DAYS)   # purge horizon + embargo
        # No feature-wise dropna: XGBoost(hist) handles NaN natively, so baseline and +macro evaluate on the
        # IDENTICAL full eligible row set. A dropna(subset=feats) would bias to fully-populated rows (sparse
        # sector_* feats are only ~39% covered -> would discard ~80% of eval) and could give the two arms
        # different rows. Only target (train) / fwd_ret (eval) must be present.
        tr = m[(m.date < cut) & m.eligible & m.target.notna()]
        ev = m[(m.date >= q.start_time) & (m.date <= q.end_time) & m.eligible & m.fwd_ret.notna()].copy()
        if len(tr) < 2000 or ev.empty:
            continue
        clf = XGBClassifier(**XGB).fit(tr[feats], tr.target.astype(int))
        ev["proba"] = clf.predict_proba(ev[feats])[:, 1]
        ev["yt"] = (ev.fwd_ret > 0).astype(int)
        evs.append(ev[["date", "symbol", "group", "proba", "yt", "fwd_ret"]])
        n_folds += 1
        for f, g in zip(feats, clf.feature_importances_):
            imp_sum[f] = imp_sum.get(f, 0.0) + float(g)
        per_q.append({"quarter": str(q), "n_train": int(len(tr)), "n_eval": int(len(ev)),
                      "acc": round(float(accuracy_score(ev.yt, (ev.proba > 0.5).astype(int))), 4),
                      "auc": round(float(roc_auc_score(ev.yt, ev.proba)), 4) if ev.yt.nunique() > 1 else None})
    ev_all = pd.concat(evs, ignore_index=True) if evs else pd.DataFrame()
    imp = {k: v / max(n_folds, 1) for k, v in imp_sum.items()}
    return per_q, imp, ev_all


def conviction(ev) -> dict:
    if ev.empty:
        return {}
    e = ev.assign(conv=(ev.proba - 0.5).abs())
    k = max(1, int(len(e) * TOP_FRAC))
    top = e.nlargest(k, "conv")
    pred = (top.proba > 0.5).astype(int)
    return {"n_confident": int(k),
            "precision_confident": round(float((pred == top.yt).mean()), 4),
            "base_rate_up": round(float(ev.yt.mean()), 4)}


def summarize(name, per_q, ev) -> dict:
    from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
    if ev.empty:
        print(f"[{name}] no eval rows")
        return {}
    acc = float(accuracy_score(ev.yt, (ev.proba > 0.5).astype(int)))
    auc = float(roc_auc_score(ev.yt, ev.proba)) if ev.yt.nunique() > 1 else float("nan")
    brier = float(brier_score_loss(ev.yt, ev.proba))
    cv = conviction(ev)
    print(f"\n[{name}]  pooled acc={acc:.4f}  AUC={auc:.4f}  Brier={brier:.4f}  "
          f"conf-precision(top{int(TOP_FRAC*100)}%)={cv.get('precision_confident', float('nan')):.4f} "
          f"(base up-rate {cv.get('base_rate_up', float('nan')):.3f}, n_eval={len(ev):,})")
    return {"acc": round(acc, 4), "auc": round(auc, 4), "brier": round(brier, 4), **cv, "by_quarter": per_q}


def main():
    RESULTS.mkdir(exist_ok=True)
    m, native, macro = load_panel()
    print(f"panel rows={len(m):,}  symbols={m.symbol.nunique()}  "
          f"native feats={len(native)}  macro feats={len(macro)}")
    print("macro features:", macro)

    out = {"config": {"horizon": HORIZON, "dead_band": DEAD_BAND, "embargo_days": EMBARGO_DAYS,
                      "quarters": [str(q) for q in QUARTERS], "n_native": len(native), "macro": macro}}
    importances = {}
    for name, feats in [("baseline_price_only", native), ("price+macro", native + macro)]:
        per_q, imp, ev = evaluate(m, feats)
        out[name] = summarize(name, per_q, ev)
        if name == "price+macro":
            importances = imp

    if out.get("baseline_price_only") and out.get("price+macro"):
        d_auc = out["price+macro"]["auc"] - out["baseline_price_only"]["auc"]
        d_acc = out["price+macro"]["acc"] - out["baseline_price_only"]["acc"]
        out["macro_lift"] = {"d_auc": round(d_auc, 4), "d_acc": round(d_acc, 4)}
        print(f"\n==> macro lift:  dAUC={d_auc:+.4f}   dACC={d_acc:+.4f}")

    if importances:
        impser = pd.Series(importances).sort_values(ascending=False)
        impser.to_csv(RESULTS / "feature_importance.csv", header=["gain_share"])
        macimp = impser[impser.index.isin(macro)]
        print("\n  macro-feature importance (gain share, incl. the DJIA vs S&P vs Nasdaq comparison):")
        for f, v in macimp.items():
            print(f"    {f:18s} {v:.4f}")

    (RESULTS / "metrics.json").write_text(json.dumps(out, indent=2, default=float))
    print(f"\nsaved -> {RESULTS / 'metrics.json'} , {RESULTS / 'feature_importance.csv'}")
    print("VERDICT GUIDE: dAUC < ~0.01 and conf-precision ~= base-rate  =>  macro adds no tradeable direction edge.")


if __name__ == "__main__":
    main()
