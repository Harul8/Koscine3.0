"""Final-attempt round A: test under-used existing data — Vol/OI (unusual options activity)
ratios + dismissed momentum/OI features. Walk-forward, XGB-CUDA, top-30 >=4%."""
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

THR = 0.04; TEST_YEARS = [2024, 2025, 2026]
LEAN = ["atm_iv", "atr_pct_14", "atm_ce_iv", "atm_pe_iv", "nifty_realized_vol_20", "mkt_pct_above_sma50",
        "days_to_earnings", "atr_pct_14_cs_rank", "realized_vol_20", "atr_pct_14_rank_60d", "sector_vol_20",
        "ret_20d_cs_rank", "pcr_oi", "fut_oi_ratio_20", "close_sma50_dist", "vol_5v20_ratio",
        "atm_iv_ratio_20", "donchian_width_20", "mkt_pct_above_sma20", "month"]
RAW = ["opt_call_vol", "opt_call_oi", "opt_put_vol", "opt_put_oi", "di_diff", "consec_up_days",
       "consec_down_days", "intraday_body_pct", "pos_day_share_20d", "oi_long_unwind", "oi_short_unwind"]
NEW = ["call_vol_oi", "put_vol_oi", "tot_vol_oi", "voi_imbalance", "di_diff", "consec_up_days",
       "consec_down_days", "intraday_body_pct", "pos_day_share_20d", "oi_long_unwind", "oi_short_unwind"]
UOA = LEAN + NEW
def _cl(f, c): return f[c].replace([np.inf, -np.inf], np.nan).astype(np.float32)


def evaluate(df, feats):
    rows = []
    for ty in TEST_YEARS:
        tr = df[df.year < ty]; te = df[(df.year == ty) & df.eligible & (df["rank"] <= 30)].copy(); te["p"] = np.nan
        for side in ("long", "short"):
            b = tr[tr.side.eq(side)]; imp = SimpleImputer(strategy="median").fit(_cl(b, feats))
            Xb = imp.transform(_cl(b, feats)); yb = (b["ceiling"] >= THR).astype(int)
            spw = (len(yb) - yb.sum()) / max(1, yb.sum()); m = te.side.eq(side)
            clf = XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.04, subsample=0.85,
                                colsample_bytree=0.85, tree_method="hist", device="cuda", scale_pos_weight=spw, verbosity=0).fit(Xb, yb)
            te.loc[m, "p"] = clf.predict_proba(imp.transform(_cl(te[m], feats)))[:, 1]
        te["y"] = (te["ceiling"] >= THR).astype(int)
        t1 = te.sort_values("p", ascending=False).groupby("date").head(1)
        t3 = te.sort_values("p", ascending=False).groupby("date").head(3)
        rows.append({"auc": roc_auc_score(te["y"], te["p"]), "p1": t1["y"].mean()*100, "p3": t3["y"].mean()*100})
    r = pd.DataFrame(rows)
    return {"AUC": round(r["auc"].mean(), 4), "prec@1": round(r["p1"].mean(), 1), "prec@3": round(r["p3"].mean(), 1),
            "by_yr": "/".join(f"{v:.0f}" for v in r["p1"])}


def main():
    t0 = time.time()
    needed = sorted(set(["date", "symbol", "open", "high", "low", "close", "turnover_lacs", "volume", *LEAN, *RAW]))
    market = load_market_data(columns=needed)
    oc = compute_clean_move_outcomes(market, universe=None, contract=CleanMoveContract())
    oc = oc[oc.status.eq("evaluated")][["date", "symbol", "side", "ceiling"]].copy(); oc["symbol"] = oc["symbol"].astype(str)
    mk = market.copy(); mk["symbol"] = mk["symbol"].astype(str); mk = mk.drop_duplicates(["date", "symbol"])
    mk["call_vol_oi"] = mk["opt_call_vol"] / (mk["opt_call_oi"] + 1)
    mk["put_vol_oi"] = mk["opt_put_vol"] / (mk["opt_put_oi"] + 1)
    mk["tot_vol_oi"] = (mk["opt_call_vol"] + mk["opt_put_vol"]) / (mk["opt_call_oi"] + mk["opt_put_oi"] + 1)
    mk["voi_imbalance"] = mk["call_vol_oi"] - mk["put_vol_oi"]
    keep = list(dict.fromkeys([*LEAN, *NEW, "close"]))
    df = oc.merge(mk[["date", "symbol", *keep]], on=["date", "symbol"], how="left")
    rk = build_universe(market, UniverseConfig(cutoff_date="2025-12-31", top_n=30)); rk = rk.set_index(rk["symbol"].astype(str))["rank"]
    df["rank"] = df["symbol"].map(rk); df["eligible"] = df["atm_iv"].notna() & df["close"].ge(100)
    df["year"] = df["date"].dt.year; df = df.reset_index(drop=True)
    print(f"data ready {time.time()-t0:.0f}s", flush=True)
    res = []
    for name, feats in [("LEAN (baseline)", LEAN), ("LEAN + Vol/OI + dismissed", UOA)]:
        t = time.time(); out = evaluate(df, feats); out["config"] = name
        res.append(out); print(f"{name:28s} {out} ({time.time()-t:.0f}s)", flush=True)
    pd.set_option("display.width", 200)
    print("\n===== ROUND A: unusual-options-activity (Vol/OI) + dismissed features =====")
    print(pd.DataFrame(res)[["config", "AUC", "prec@1", "prec@3", "by_yr"]].to_string(index=False))


if __name__ == "__main__":
    main()
