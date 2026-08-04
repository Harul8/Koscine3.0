"""Quantify the market-cap universe vs turnover universe for the >=4% 5d strategy.
Base rate of >=4% moves + walk-forward precision@1, for mcap-top20 / mcap-top30 / turnover-top30."""
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
MCAP30 = ['RELIANCE','HDFCBANK','BHARTIARTL','ICICIBANK','SBIN','TCS','BAJFINANCE','LT','HINDUNILVR','LICI',
          'INFY','SUNPHARMA','ADANIPOWER','AXISBANK','MARUTI','ADANIPORTS','KOTAKBANK','M&M','ADANIENT','TITAN',
          'ITC','NTPC','ULTRACEMCO','JSWSTEEL','ONGC','HCLTECH','BEL','BAJAJ-AUTO','HAL','COALINDIA']
MCAP20 = MCAP30[:20]
def _cl(f): return f[LEAN].replace([np.inf, -np.inf], np.nan).astype(np.float32)


def eval_universe(df, uni_syms):
    rows = []
    for ty in TEST_YEARS:
        tr = df[df.year < ty]; te = df[(df.year == ty) & df.eligible & df.symbol.isin(uni_syms)].copy(); te["p"] = np.nan
        for side in ("long", "short"):
            b = tr[tr.side.eq(side)]; imp = SimpleImputer(strategy="median").fit(_cl(b)); Xb = imp.transform(_cl(b))
            yb = (b["ceiling"] >= THR).astype(int); spw = (len(yb)-yb.sum())/max(1, yb.sum()); m = te.side.eq(side)
            clf = XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.04, subsample=0.85, colsample_bytree=0.85,
                                tree_method="hist", device="cuda", scale_pos_weight=spw, verbosity=0).fit(Xb, yb)
            te.loc[m, "p"] = clf.predict_proba(imp.transform(_cl(te[m])))[:, 1]
        te["y"] = (te["ceiling"] >= THR).astype(int)
        t1 = te.sort_values("p", ascending=False).groupby("date").head(1)
        rows.append({"auc": roc_auc_score(te["y"], te["p"]), "p1": t1["y"].mean()*100, "base": te["y"].mean()*100})
    r = pd.DataFrame(rows)
    return {"base_rate>=4%": round(r["base"].mean(), 1), "AUC": round(r["auc"].mean(), 3), "prec@1": round(r["p1"].mean(), 1)}


def main():
    t0 = time.time()
    market = load_market_data(columns=sorted(set(["date","symbol","open","high","low","close","turnover_lacs","volume",*LEAN])))
    oc = compute_clean_move_outcomes(market, universe=None, contract=CleanMoveContract(window_days=5))
    oc = oc[oc.status.eq("evaluated")][["date","symbol","side","ceiling"]].copy(); oc["symbol"]=oc["symbol"].astype(str)
    mk = market.copy(); mk["symbol"]=mk["symbol"].astype(str); mk=mk.drop_duplicates(["date","symbol"])
    df = oc.merge(mk[["date","symbol",*list(dict.fromkeys([*LEAN,"close"]))]], on=["date","symbol"], how="left")
    df["eligible"]=df["atm_iv"].notna() & df["close"].ge(100); df["year"]=df["date"].dt.year; df=df.reset_index(drop=True)
    turn30 = set(build_universe(market, UniverseConfig(cutoff_date="2025-12-31", top_n=30))["symbol"].astype(str))
    print(f"data ready {time.time()-t0:.0f}s", flush=True)
    res = []
    for name, syms in [("turnover top-30 (current)", turn30), ("MCAP top-30", set(MCAP30)), ("MCAP top-20", set(MCAP20))]:
        out = eval_universe(df, syms); out["universe"] = name; res.append(out)
        print(f"{name:26s} {out}", flush=True)
    pd.set_option("display.width", 200)
    print("\n===== UNIVERSE COMPARISON (5d >=4%, walk-forward) =====")
    print(pd.DataFrame(res)[["universe", "base_rate>=4%", "AUC", "prec@1"]].to_string(index=False))


if __name__ == "__main__":
    main()
