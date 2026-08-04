"""Round 3 (GPU): ensemble (XGB+CatBoost) and magnitude-weighted training — confirm precision ceiling.
Reuses lean features, walk-forward, top-30 >=4%."""
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
from catboost import CatBoostClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score

THR = 0.04; TEST_YEARS = [2024, 2025, 2026]
LEAN = ["atm_iv", "atr_pct_14", "atm_ce_iv", "atm_pe_iv", "nifty_realized_vol_20", "mkt_pct_above_sma50",
        "days_to_earnings", "atr_pct_14_cs_rank", "realized_vol_20", "atr_pct_14_rank_60d", "sector_vol_20",
        "ret_20d_cs_rank", "pcr_oi", "fut_oi_ratio_20", "close_sma50_dist", "vol_5v20_ratio",
        "atm_iv_ratio_20", "donchian_width_20", "mkt_pct_above_sma20", "month"]
def _clean(f): return f[LEAN].replace([np.inf, -np.inf], np.nan).astype(np.float32)


def evaluate(df, mode):
    rows = []
    for ty in TEST_YEARS:
        tr = df[df.year < ty]; te = df[(df.year == ty) & df.eligible & (df["rank"] <= 30)].copy(); te["p"] = np.nan
        for side in ("long", "short"):
            b = tr[tr.side.eq(side)]; imp = SimpleImputer(strategy="median").fit(_clean(b))
            Xb = imp.transform(_clean(b)); yb = (b["ceiling"] >= THR).astype(int)
            spw = (len(yb) - yb.sum()) / max(1, yb.sum()); m = te.side.eq(side); Xt = imp.transform(_clean(te[m]))
            if mode == "ensemble":
                xgb = XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.04, subsample=0.85,
                                    colsample_bytree=0.85, tree_method="hist", device="cuda", scale_pos_weight=spw, verbosity=0).fit(Xb, yb)
                cb = CatBoostClassifier(iterations=500, depth=6, learning_rate=0.04, task_type="GPU", devices="0",
                                        auto_class_weights="Balanced", verbose=0).fit(Xb, yb)
                p = (xgb.predict_proba(Xt)[:, 1] + cb.predict_proba(Xt)[:, 1]) / 2
            elif mode == "magweight":
                w = 1.0 + 8.0 * b["ceiling"].clip(0, 0.3).values  # weight bigger moves more
                xgb = XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.04, subsample=0.85,
                                    colsample_bytree=0.85, tree_method="hist", device="cuda", scale_pos_weight=spw, verbosity=0).fit(Xb, yb, sample_weight=w)
                p = xgb.predict_proba(Xt)[:, 1]
            te.loc[m, "p"] = p
        te["y"] = (te["ceiling"] >= THR).astype(int)
        t1 = te.sort_values("p", ascending=False).groupby("date").head(1)
        t3 = te.sort_values("p", ascending=False).groupby("date").head(3)
        rows.append({"auc": roc_auc_score(te["y"], te["p"]), "p1": t1["y"].mean()*100, "p3": t3["y"].mean()*100})
    r = pd.DataFrame(rows)
    return {"AUC": round(r["auc"].mean(), 4), "prec@1": round(r["p1"].mean(), 1), "prec@3": round(r["p3"].mean(), 1),
            "by_yr": "/".join(f"{v:.0f}" for v in r["p1"])}


def main():
    t0 = time.time()
    needed = sorted(set(["date", "symbol", "open", "high", "low", "close", "turnover_lacs", "volume", *LEAN]))
    market = load_market_data(columns=needed)
    oc = compute_clean_move_outcomes(market, universe=None, contract=CleanMoveContract())
    oc = oc[oc.status.eq("evaluated")][["date", "symbol", "side", "ceiling"]].copy(); oc["symbol"] = oc["symbol"].astype(str)
    mk = market.copy(); mk["symbol"] = mk["symbol"].astype(str); mk = mk.drop_duplicates(["date", "symbol"])
    df = oc.merge(mk[["date", "symbol", *list(dict.fromkeys([*LEAN, "close"]))]], on=["date", "symbol"], how="left")
    rk = build_universe(market, UniverseConfig(cutoff_date="2025-12-31", top_n=30)); rk = rk.set_index(rk["symbol"].astype(str))["rank"]
    df["rank"] = df["symbol"].map(rk); df["eligible"] = df["atm_iv"].notna() & df["close"].ge(100)
    df["year"] = df["date"].dt.year; df = df.reset_index(drop=True)
    print(f"data ready {time.time()-t0:.0f}s", flush=True)
    res = []
    for mode in ["ensemble", "magweight"]:
        t = time.time(); out = evaluate(df, mode); out["config"] = mode
        res.append(out); print(f"{mode:12s} {out} ({time.time()-t:.0f}s)", flush=True)
    print("\n===== ROUND 3 vs baseline (X1 was AUC 0.656 / prec@1 49.4) =====")
    pd.set_option("display.width", 200); print(pd.DataFrame(res)[["config", "AUC", "prec@1", "prec@3", "by_yr"]].to_string(index=False))


if __name__ == "__main__":
    main()
