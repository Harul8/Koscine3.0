"""Test the two-stage hurdle model: classifier P(move>=thr) + regressor E[move|moved], rank by product.
A: >=3% (mcap-30), B: >=4% (turnover-35). Compare confidence-rank vs hurdle-rank: precision@N + realized move.
"""
from __future__ import annotations
import sys, time, json
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from koscine3.data.sources import load_market_data
from koscine3.outcomes.clean_move_contract import CleanMoveContract, compute_clean_move_outcomes
from xgboost import XGBClassifier, XGBRegressor
from sklearn.impute import SimpleImputer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import roc_auc_score

TEST_YEARS = [2024, 2025, 2026]
LEAN = ["atm_iv", "atr_pct_14", "atm_ce_iv", "atm_pe_iv", "nifty_realized_vol_20", "mkt_pct_above_sma50",
        "days_to_earnings", "atr_pct_14_cs_rank", "realized_vol_20", "atr_pct_14_rank_60d", "sector_vol_20",
        "ret_20d_cs_rank", "pcr_oi", "fut_oi_ratio_20", "close_sma50_dist", "vol_5v20_ratio",
        "atm_iv_ratio_20", "donchian_width_20", "mkt_pct_above_sma20", "month"]
def _cl(f): return f[LEAN].replace([np.inf, -np.inf], np.nan).astype(np.float32)


def evaluate(df, syms, thr):
    rows = []
    for T in TEST_YEARS:
        base = df[df.year < T - 1]; calib = df[df.year == T - 1]
        te = df[(df.year == T) & df.eligible & df.symbol.isin(syms)].copy()
        if te.empty: continue
        te["conf"] = np.nan; te["mag"] = np.nan
        for side in ("long", "short"):
            b = base[base.side.eq(side)]; c = calib[calib.side.eq(side)]; m = te.side.eq(side)
            imp = SimpleImputer(strategy="median").fit(_cl(b)); Xb = imp.transform(_cl(b))
            yb = (b["ceiling"] >= thr).astype(int); spw = (len(yb)-yb.sum())/max(1, yb.sum())
            clf = XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.04, subsample=0.85, colsample_bytree=0.85,
                                tree_method="hist", device="cuda", scale_pos_weight=spw, verbosity=0).fit(Xb, yb)
            cal = CalibratedClassifierCV(FrozenEstimator(clf), method="isotonic").fit(imp.transform(_cl(c)), (c["ceiling"] >= thr).astype(int))
            pos = b[b["ceiling"] >= thr]  # Stage-2: regressor on POSITIVES only (magnitude | moved)
            reg = XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.04, subsample=0.85, colsample_bytree=0.85,
                               tree_method="hist", device="cuda", verbosity=0).fit(imp.transform(_cl(pos)), pos["ceiling"].clip(0, 0.5))
            Xe = imp.transform(_cl(te[m]))
            te.loc[m, "conf"] = cal.predict_proba(Xe)[:, 1]; te.loc[m, "mag"] = np.clip(reg.predict(Xe), 0, None)
        te = te.dropna(subset=["conf"]); te["y"] = (te["ceiling"] >= thr).astype(int)
        te["hurdle"] = te["conf"] * te["mag"]
        for rk, col in [("confidence", "conf"), ("hurdle(PxE)", "hurdle")]:
            for n in (1, 3, 5):
                top = te.sort_values(col, ascending=False).groupby("date").head(n)
                rows.append({"rank_by": rk, "topN": n, "prec%": top["y"].mean()*100, "realized_move%": top["ceiling"].mean()*100})
    r = pd.DataFrame(rows)
    return r.groupby(["rank_by", "topN"]).agg(prec=("prec%", "mean"), move=("realized_move%", "mean")).round(2).reset_index()


def main():
    t0 = time.time()
    grp = json.load(open(ROOT / "reports" / "universe_groups.json"))
    market = load_market_data(columns=sorted(set(["date", "symbol", "open", "high", "low", "close", "turnover_lacs", "volume", *LEAN])))
    oc = compute_clean_move_outcomes(market, universe=None, contract=CleanMoveContract(window_days=5))
    oc = oc[oc.status.eq("evaluated")][["date", "symbol", "side", "ceiling"]].copy(); oc["symbol"] = oc["symbol"].astype(str)
    mk = market.copy(); mk["symbol"] = mk["symbol"].astype(str); mk = mk.drop_duplicates(["date", "symbol"])
    df = oc.merge(mk[["date", "symbol", *list(dict.fromkeys([*LEAN, "close"]))]], on=["date", "symbol"], how="left")
    df["eligible"] = df["atm_iv"].notna() & df["close"].ge(100); df["year"] = df["date"].dt.year; df = df.reset_index(drop=True)
    print(f"data ready {time.time()-t0:.0f}s", flush=True)
    for label, syms, thr in [("A_mcap30 >=3%", set(grp["A_mcap30"]), 0.03), ("B_turn35 >=4%", set(grp["B_turn35"]), 0.04)]:
        res = evaluate(df, syms, thr)
        print(f"\n===== {label}: confidence-rank vs hurdle-rank =====")
        print(res.to_string(index=False)); print(f"({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
