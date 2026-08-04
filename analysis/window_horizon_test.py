"""Does a 10-day window predict better than 5-day? Walk-forward precision@1 for favorable
moves >=4/6/8%, base rates, and direction accuracy of the top pick. Top-30, lean features.
CPU XGB (subsampled train) to coexist with other GPU jobs.
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

TEST_YEARS = [2024, 2025, 2026]
LEAN = ["atm_iv", "atr_pct_14", "atm_ce_iv", "atm_pe_iv", "nifty_realized_vol_20", "mkt_pct_above_sma50",
        "days_to_earnings", "atr_pct_14_cs_rank", "realized_vol_20", "atr_pct_14_rank_60d", "sector_vol_20",
        "ret_20d_cs_rank", "pcr_oi", "fut_oi_ratio_20", "close_sma50_dist", "vol_5v20_ratio",
        "atm_iv_ratio_20", "donchian_width_20", "mkt_pct_above_sma20", "month"]
def _cl(f, c): return f[c].replace([np.inf, -np.inf], np.nan).astype(np.float32)


def fwd_close(market, w):
    g = market.sort_values(["symbol", "date"]).groupby("symbol", sort=False)
    eo = g["open"].shift(-1); fc = g["close"].shift(-w)
    return pd.DataFrame({"date": market["date"], "symbol": market["symbol"].astype(str),
                         "entry_open": eo.values, "fwd_close": fc.values})


def evaluate(df, thr):
    rows = []
    for ty in TEST_YEARS:
        tr = df[df.year < ty].sample(frac=0.4, random_state=1)  # subsample for CPU speed
        te = df[(df.year == ty) & df.eligible & (df["rank"] <= 30)].copy(); te["p"] = np.nan
        for side in ("long", "short"):
            b = tr[tr.side.eq(side)]; imp = SimpleImputer(strategy="median").fit(_cl(b, LEAN))
            Xb = imp.transform(_cl(b, LEAN)); yb = (b["ceiling"] >= thr).astype(int)
            spw = (len(yb) - yb.sum()) / max(1, yb.sum()); m = te.side.eq(side)
            clf = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.85,
                                colsample_bytree=0.85, tree_method="hist", device="cpu", scale_pos_weight=spw, verbosity=0, n_jobs=-1).fit(Xb, yb)
            te.loc[m, "p"] = clf.predict_proba(imp.transform(_cl(te[m], LEAN)))[:, 1]
        te["y"] = (te["ceiling"] >= thr).astype(int)
        t1 = te.sort_values("p", ascending=False).groupby("date").head(1)
        # direction: did the pick close in its favorable direction?
        dirok = np.where(t1["side"].eq("long"), t1["fwd_close"] > t1["entry_open"], t1["fwd_close"] < t1["entry_open"])
        rows.append({"auc": roc_auc_score(te["y"], te["p"]), "p1": t1["y"].mean()*100,
                     "base": te["y"].mean()*100, "dir": np.nanmean(dirok)*100})
    r = pd.DataFrame(rows)
    return {"AUC": round(r["auc"].mean(), 3), "base_rate": round(r["base"].mean(), 1),
            "prec@1": round(r["p1"].mean(), 1), "dir_acc@1": round(r["dir"].mean(), 1)}


def main():
    t0 = time.time()
    market = load_market_data(columns=sorted(set(["date", "symbol", "open", "high", "low", "close", "turnover_lacs", "volume", *LEAN])))
    rk = build_universe(market, UniverseConfig(cutoff_date="2025-12-31", top_n=30)); rk = rk.set_index(rk["symbol"].astype(str))["rank"]
    mk = market.copy(); mk["symbol"] = mk["symbol"].astype(str); mk = mk.drop_duplicates(["date", "symbol"])
    base_feat = mk[["date", "symbol", *list(dict.fromkeys([*LEAN, "close"]))]]

    res = []
    for w in [5, 10]:
        oc = compute_clean_move_outcomes(market, universe=None, contract=CleanMoveContract(window_days=w))
        oc = oc[oc.status.eq("evaluated")][["date", "symbol", "side", "ceiling"]].copy(); oc["symbol"] = oc["symbol"].astype(str)
        fc = fwd_close(market, w)
        df = oc.merge(base_feat, on=["date", "symbol"], how="left").merge(fc, on=["date", "symbol"], how="left")
        df["rank"] = df["symbol"].map(rk); df["eligible"] = df["atm_iv"].notna() & df["close"].ge(100)
        df["year"] = df["date"].dt.year; df = df.reset_index(drop=True)
        thrs = [0.04] if w == 5 else [0.04, 0.06, 0.08]
        for thr in thrs:
            out = evaluate(df, thr); out["window"] = f"{w}d"; out["thr"] = f">={int(thr*100)}%"
            res.append(out); print(f"{w}d >={int(thr*100)}%  {out} ({time.time()-t0:.0f}s)", flush=True)
    pd.set_option("display.width", 200)
    print("\n===== 5-DAY vs 10-DAY horizon (walk-forward, top-30) =====")
    print(pd.DataFrame(res)[["window", "thr", "base_rate", "AUC", "prec@1", "dir_acc@1"]].to_string(index=False))
    print("\ndir_acc@1 = of the top pick, % that CLOSED in its favorable direction over the window.")


if __name__ == "__main__":
    main()
