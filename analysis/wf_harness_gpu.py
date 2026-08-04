"""GPU walk-forward harness (XGBoost CUDA). >=4% move, top-30, point-in-time, train broad.
Re-baselines on XGB, tests tuned config, and maps the CONVICTION CURVE (precision vs how
selective we are) — the real lever for precision since features don't move it.
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from koscine3.data.sources import load_market_data
from koscine3.data.universe import UniverseConfig, build_universe
from koscine3.outcomes.clean_move_contract import CleanMoveContract, compute_clean_move_outcomes
from xgboost import XGBClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score

THR = 0.04
TEST_YEARS = [2024, 2025, 2026]
LEAN = ["atm_iv", "atr_pct_14", "atm_ce_iv", "atm_pe_iv", "nifty_realized_vol_20", "mkt_pct_above_sma50",
        "days_to_earnings", "atr_pct_14_cs_rank", "realized_vol_20", "atr_pct_14_rank_60d", "sector_vol_20",
        "ret_20d_cs_rank", "pcr_oi", "fut_oi_ratio_20", "close_sma50_dist", "vol_5v20_ratio",
        "atm_iv_ratio_20", "donchian_width_20", "mkt_pct_above_sma20", "month"]
def _clean(f, c): return f[c].replace([np.inf, -np.inf], np.nan).astype(np.float32)


def make_xgb(params, spw):
    p = dict(n_estimators=400, max_depth=6, learning_rate=0.04, subsample=0.85, colsample_bytree=0.85,
             tree_method="hist", device="cuda", scale_pos_weight=spw, verbosity=0)
    p.update(params); return XGBClassifier(**p)


def evaluate(df, feats, params, collect=False):
    rows, picks = [], []
    for ty in TEST_YEARS:
        tr = df[df.year < ty]; te = df[(df.year == ty) & df.eligible & (df["rank"] <= 30)].copy()
        te["p"] = np.nan
        for side in ("long", "short"):
            b = tr[tr.side.eq(side)]
            imp = SimpleImputer(strategy="median").fit(_clean(b, feats)); Xb = imp.transform(_clean(b, feats))
            yb = (b["ceiling"] >= THR).astype(int); spw = (len(yb) - yb.sum()) / max(1, yb.sum())
            m = te.side.eq(side)
            clf = make_xgb(params, spw).fit(Xb, yb)
            te.loc[m, "p"] = clf.predict_proba(imp.transform(_clean(te[m], feats)))[:, 1]
        te["y"] = (te["ceiling"] >= THR).astype(int)
        t1 = te.sort_values("p", ascending=False).groupby("date").head(1)
        t3 = te.sort_values("p", ascending=False).groupby("date").head(3)
        rows.append({"auc": roc_auc_score(te["y"], te["p"]), "p1": t1["y"].mean()*100, "p3": t3["y"].mean()*100})
        if collect: picks.append(t1[["date", "p", "y"]])
    r = pd.DataFrame(rows)
    out = {"AUC": round(r["auc"].mean(), 4), "prec@1": round(r["p1"].mean(), 1), "prec@3": round(r["p3"].mean(), 1),
           "by_yr": "/".join(f"{v:.0f}" for v in r["p1"])}
    return (out, pd.concat(picks) if collect else None)


def main():
    t0 = time.time()
    needed = sorted(set(["date", "symbol", "open", "high", "low", "close", "turnover_lacs", "volume", *LEAN]))
    market = load_market_data(columns=needed)
    oc = compute_clean_move_outcomes(market, universe=None, contract=CleanMoveContract())
    oc = oc[oc.status.eq("evaluated")][["date", "symbol", "side", "ceiling"]].copy()
    oc["symbol"] = oc["symbol"].astype(str)
    mk = market.copy(); mk["symbol"] = mk["symbol"].astype(str); mk = mk.drop_duplicates(["date", "symbol"])
    cols = list(dict.fromkeys([*LEAN, "close"]))
    df = oc.merge(mk[["date", "symbol", *cols]], on=["date", "symbol"], how="left")
    rk = build_universe(market, UniverseConfig(cutoff_date="2025-12-31", top_n=30))
    rk = rk.set_index(rk["symbol"].astype(str))["rank"]
    df["rank"] = df["symbol"].map(rk)
    df["eligible"] = df["atm_iv"].notna() & df["close"].ge(100)
    df["year"] = df["date"].dt.year
    df = df.reset_index(drop=True)
    print(f"data ready {time.time()-t0:.0f}s | rows {len(df):,}", flush=True)

    configs = [
        ("X1 XGB lean", LEAN, {}),
        ("X2 XGB tuned", LEAN, dict(n_estimators=800, max_depth=8, learning_rate=0.025, min_child_weight=50, reg_lambda=5.0)),
        ("X3 XGB shallow+more", LEAN, dict(n_estimators=1200, max_depth=4, learning_rate=0.02)),
    ]
    res, best_picks = [], None
    for name, feats, params in configs:
        t = time.time(); out, picks = evaluate(df, feats, params, collect=(name == "X1 XGB lean"))
        out["config"] = name; res.append(out)
        if picks is not None: best_picks = picks
        print(f"{name:22s} {out} ({time.time()-t:.0f}s)", flush=True)
    pd.set_option("display.width", 200)
    print("\n===== GPU ROUND: walk-forward >=4% precision =====")
    print(pd.DataFrame(res)[["config", "AUC", "prec@1", "prec@3", "by_yr"]].to_string(index=False))

    # CONVICTION CURVE (the precision lever): precision@1 when firing only top X% of days by confidence
    print("\n===== CONVICTION CURVE (X1): precision vs selectivity =====")
    bp = best_picks.sort_values("p", ascending=False)
    n = len(bp)
    rows = [{"fire_top_%": int(f*100), "days_fired": int(n*f), "per_yr": round(n*f/3),
             "precision@1": round(bp.head(int(n*f))["y"].mean()*100, 1)} for f in [1.0, 0.7, 0.5, 0.3, 0.2, 0.1]]
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
