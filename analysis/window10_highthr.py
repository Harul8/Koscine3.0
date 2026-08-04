"""10-day horizon, BIG targets (>=8/10/12%). Walk-forward prec@1, AUC, dir + conviction curve.
Tests whether a 10d big-move hunter (higher AUC for big moves) is viable with selectivity. GPU."""
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


def evaluate(df, thr, collect=False):
    rows, picks = [], []
    for ty in TEST_YEARS:
        tr = df[df.year < ty]; te = df[(df.year == ty) & df.eligible & (df["rank"] <= 30)].copy(); te["p"] = np.nan
        for side in ("long", "short"):
            b = tr[tr.side.eq(side)]; imp = SimpleImputer(strategy="median").fit(_cl(b, LEAN))
            Xb = imp.transform(_cl(b, LEAN)); yb = (b["ceiling"] >= thr).astype(int)
            spw = (len(yb) - yb.sum()) / max(1, yb.sum()); m = te.side.eq(side)
            clf = XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.04, subsample=0.85, colsample_bytree=0.85,
                                tree_method="hist", device="cuda", scale_pos_weight=spw, verbosity=0).fit(Xb, yb)
            te.loc[m, "p"] = clf.predict_proba(imp.transform(_cl(te[m], LEAN)))[:, 1]
        te["y"] = (te["ceiling"] >= thr).astype(int)
        t1 = te.sort_values("p", ascending=False).groupby("date").head(1)
        rows.append({"auc": roc_auc_score(te["y"], te["p"]), "p1": t1["y"].mean()*100, "base": te["y"].mean()*100})
        if collect: picks.append(t1[["date", "p", "y"]])
    r = pd.DataFrame(rows)
    out = {"AUC": round(r["auc"].mean(), 3), "base_rate": round(r["base"].mean(), 1), "prec@1": round(r["p1"].mean(), 1),
           "by_yr": "/".join(f"{v:.0f}" for v in r["p1"])}
    return out, (pd.concat(picks) if collect else None)


def main():
    t0 = time.time()
    market = load_market_data(columns=sorted(set(["date", "symbol", "open", "high", "low", "close", "turnover_lacs", "volume", *LEAN])))
    oc = compute_clean_move_outcomes(market, universe=None, contract=CleanMoveContract(window_days=10))
    oc = oc[oc.status.eq("evaluated")][["date", "symbol", "side", "ceiling"]].copy(); oc["symbol"] = oc["symbol"].astype(str)
    mk = market.copy(); mk["symbol"] = mk["symbol"].astype(str); mk = mk.drop_duplicates(["date", "symbol"])
    df = oc.merge(mk[["date", "symbol", *list(dict.fromkeys([*LEAN, "close"]))]], on=["date", "symbol"], how="left")
    rk = build_universe(market, UniverseConfig(cutoff_date="2025-12-31", top_n=30)); rk = rk.set_index(rk["symbol"].astype(str))["rank"]
    df["rank"] = df["symbol"].map(rk); df["eligible"] = df["atm_iv"].notna() & df["close"].ge(100)
    df["year"] = df["date"].dt.year; df = df.reset_index(drop=True)
    print(f"data ready {time.time()-t0:.0f}s (10-day window)", flush=True)

    res, picks10 = [], None
    for thr in [0.08, 0.10, 0.12]:
        out, pk = evaluate(df, thr, collect=(thr == 0.10)); out["thr"] = f">={int(thr*100)}%"
        res.append(out); print(f"10d >={int(thr*100)}%  {out}", flush=True)
        if pk is not None: picks10 = pk
    pd.set_option("display.width", 200)
    print("\n===== 10-DAY BIG-MOVE targets (walk-forward, top-30) =====")
    print(pd.DataFrame(res)[["thr", "base_rate", "AUC", "prec@1", "by_yr"]].to_string(index=False))

    print("\n===== CONVICTION CURVE: 10d >=10% (precision vs selectivity) =====")
    bp = picks10.sort_values("p", ascending=False); n = len(bp)
    rows = [{"fire_top_%": int(f*100), "trades/yr": round(n*f/3), "precision@1": round(bp.head(int(n*f))["y"].mean()*100, 1)}
            for f in [1.0, 0.5, 0.3, 0.2, 0.1]]
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
