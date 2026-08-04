"""What features (option data + underlying stock) are common among STRONG option gainers? -> feeds an ML model.

Joins the 5-day option-gain tape (option_gain_trades.csv) to the full equity feature table, defines a strong
gainer (peak high_ratio >= 3x), and measures:
  1) univariate separation: single-feature AUC for predicting strong-gainer (which features separate gainers),
  2) multivariate: CatBoost (purged walk-forward) OOS AUC + feature importances, pooled and per side.
Leak-safe: forward-looking K2 label columns and the trade-outcome columns are excluded; features are as-of entry.

    set PYTHONPATH=src && python experiments/option_gain_study_v1/feature_study.py
Read-only; PROD untouched.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
from koscine3.data.sources import load_market_data  # noqa: E402

GAIN = 3.0                       # strong gainer = option peak >= 3x
EMBARGO_DAYS = 5
QUARTERS = pd.period_range("2024Q1", "2026Q2", freq="Q")
TRADE_COLS = {"symbol", "group", "side", "strike_label", "otm_pct", "strike", "expiry", "entry_date", "U",
              "entry_open", "max_high", "last_close", "peak_day", "exit_close_u", "high_ratio", "close_ratio",
              "stock_move", "date", "open", "high", "low", "close", "volume"}
LEAK = ("future", "fwd", "next", "label", "tomorrow", "ahead", "adverse", "move_5d", "move_1d", "move_3d",
        "move_10d", "expansion", "volclean", "outcome", "_ratio")
OPT_FEATS = ["otm_pct", "dte", "entry_open"]
CB = dict(iterations=400, depth=6, learning_rate=0.03, l2_leaf_reg=6.0, loss_function="Logloss",
          random_seed=7, allow_writing_files=False, verbose=False, task_type="GPU", devices="0")


def load():
    t = pd.read_csv(HERE / "results" / "option_gain_trades.csv", parse_dates=["entry_date"])
    t["dte"] = t.dte.astype(float)
    t = t.drop(columns=[c for c in ["atm_iv"] if c in t.columns])   # avoid merge collision; use lagged feature-table atm_iv
    f = load_market_data()
    f["symbol"] = f["symbol"].astype(str)
    f["date"] = pd.to_datetime(f["date"])
    f = f.sort_values(["symbol", "date"])
    stock_feats = [c for c in f.columns if c not in TRADE_COLS and pd.api.types.is_numeric_dtype(f[c])
                   and not any(h in c.lower() for h in LEAK)]
    # LEAK-SAFE: features as-of the PRIOR trading day (known by the ~7AM refresh) apply to a day-d OPEN entry.
    f2 = f[["symbol", "date"] + stock_feats].copy()
    f2["apply_date"] = f2.groupby("symbol")["date"].shift(-1)
    f2 = f2.dropna(subset=["apply_date"]).drop(columns=["date"])
    m = t.merge(f2, left_on=["entry_date", "symbol"], right_on=["apply_date", "symbol"], how="left")
    m["win"] = (m.high_ratio >= GAIN).astype(int)
    feats = OPT_FEATS + stock_feats
    feats = [c for c in feats if c in m.columns and m[c].notna().any()]
    return m, feats, stock_feats


def main():
    from catboost import CatBoostClassifier
    from sklearn.metrics import roc_auc_score
    m, feats, stock_feats = load()
    base = m.win.mean()
    print(f"trades {len(m):,} | strong-gainer base rate (peak>={GAIN}x) = {base:.3f} | "
          f"calls {m[m.side=='CALL'].win.mean():.3f} puts {m[m.side=='PUT'].win.mean():.3f}")

    # ---- 1) univariate separation ----
    y = m.win.to_numpy()
    rows = []
    for c in feats:
        x = m[c].replace([np.inf, -np.inf], np.nan)
        if x.notna().sum() < 20000:
            continue
        xf = x.fillna(x.median()).to_numpy()
        auc = roc_auc_score(y, xf)
        sep = abs(auc - 0.5)
        rows.append((c, round(auc, 3), round(sep, 3),
                     round(float(m.loc[m.win == 1, c].mean()), 4), round(float(m.loc[m.win == 0, c].mean()), 4)))
    uni = pd.DataFrame(rows, columns=["feature", "auc", "sep", "mean_gainer", "mean_rest"]).sort_values("sep", ascending=False)
    print("\n=== univariate separation (single-feature AUC for strong-gainer), top 25 ===")
    print(uni.head(25).to_string(index=False))
    leaks = uni[uni.sep > 0.40]
    if len(leaks):
        print("\n!! possible residual leaks (sep>0.40):", leaks.feature.tolist())
    uni.to_csv(HERE / "results" / "feature_univariate.csv", index=False)

    # ---- 2) multivariate CatBoost, purged walk-forward ----
    cats = [c for c in ["side", "strike_label", "group"] if c in m.columns]
    for c in cats:
        m[c] = m[c].astype(str).fillna("NA")
    mfeats = feats + cats

    def wf(df, label):
        ev, imp, n = [], {}, 0
        for q in QUARTERS:
            cut = q.start_time - pd.Timedelta(days=5 + EMBARGO_DAYS)
            tr = df[df.entry_date < cut]
            te = df[(df.entry_date >= q.start_time) & (df.entry_date <= q.end_time)]
            if len(tr) < 5000 or te.empty or tr.win.nunique() < 2:
                continue
            clf = CatBoostClassifier(**CB).fit(tr[mfeats], tr.win, cat_features=cats)
            p = clf.predict_proba(te[mfeats])[:, 1]
            ev.append(pd.DataFrame({"y": te.win.to_numpy(), "p": p}))
            for f, g in zip(mfeats, clf.get_feature_importance()):
                imp[f] = imp.get(f, 0.0) + float(g)
            n += 1
        e = pd.concat(ev, ignore_index=True)
        auc = roc_auc_score(e.y, e.p)
        # precision in the top decile of predicted score
        k = max(1, int(len(e) * 0.10)); top = e.nlargest(k, "p")
        print(f"\n[{label}] OOS AUC={auc:.4f}  base={e.y.mean():.3f}  top-decile precision={top.y.mean():.3f} "
              f"(lift {top.y.mean()/e.y.mean():.2f}x)  n={len(e):,}")
        return {f: v / n for f, v in imp.items()}

    print("\nunivariate done; running multivariate (GPU CatBoost, purged walk-forward)...", flush=True)
    imp = wf(m, "pooled (calls+puts)")
    wf(m[m.side == "CALL"], "calls only")
    wf(m[m.side == "PUT"], "puts only")
    impser = pd.Series(imp).sort_values(ascending=False)
    impser.to_csv(HERE / "results" / "feature_importance_gainer.csv", header=["importance"])
    print("\n=== multivariate top-20 features (pooled) ===")
    for f, v in impser.head(20).items():
        tag = "OPT" if f in OPT_FEATS + cats else "stk"
        print(f"   {tag} {f:26s} {v:6.2f}")
    print("\nsaved -> results/feature_univariate.csv , feature_importance_gainer.csv")


if __name__ == "__main__":
    main()
