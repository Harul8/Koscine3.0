"""Final-attempt round B: IV TERM STRUCTURE (backwardation) from bhavcopy — the research-backed
signal NOT in the equity table. Extract front vs next-month ATM straddle-implied vol slope,
join to lean features, walk-forward test whether it lifts >=4% precision. Top-50 train, top-30 eval.
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
from analysis.options_bhavcopy import load_bhavcopy
from xgboost import XGBClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score

THR = 0.04; TEST_YEARS = [2024, 2025, 2026]; START = pd.Timestamp("2021-01-01")
LEAN = ["atm_iv", "atr_pct_14", "atm_ce_iv", "atm_pe_iv", "nifty_realized_vol_20", "mkt_pct_above_sma50",
        "days_to_earnings", "atr_pct_14_cs_rank", "realized_vol_20", "atr_pct_14_rank_60d", "sector_vol_20",
        "ret_20d_cs_rank", "pcr_oi", "fut_oi_ratio_20", "close_sma50_dist", "vol_5v20_ratio",
        "atm_iv_ratio_20", "donchian_width_20", "mkt_pct_above_sma20", "month"]
TS = ["iv_front_proxy", "term_slope", "implied_move_front"]
def _cl(f, c): return f[c].replace([np.inf, -np.inf], np.nan).astype(np.float32)


def straddle_iv(ge, spot):
    ks = sorted(ge["strike"].dropna().unique())
    if not ks or not (spot and spot > 0): return np.nan
    atm = min(ks, key=lambda k: abs(k - spot))
    ce = ge[(ge.strike == atm) & ge.opt_type.eq("CE")]; pe = ge[(ge.strike == atm) & ge.opt_type.eq("PE")]
    if ce.empty or pe.empty: return np.nan
    def px(r):
        for c in ("close", "settle"):
            if pd.notna(r[c].iloc[0]) and r[c].iloc[0] > 0: return float(r[c].iloc[0])
        return np.nan
    c, p = px(ce), px(pe)
    if not (c and p): return np.nan
    return (c + p) / spot  # straddle as fraction of spot


def main():
    t0 = time.time()
    market = load_market_data(columns=sorted(set(["date", "symbol", "open", "high", "low", "close", "turnover_lacs", "volume", *LEAN])))
    rk = build_universe(market, UniverseConfig(cutoff_date="2025-12-31", top_n=50))
    rk = rk.set_index(rk["symbol"].astype(str))["rank"]; syms = set(rk.index)
    close_map = market.assign(symbol=market["symbol"].astype(str)).set_index(["date", "symbol"])["close"]

    cal = sorted(market.loc[market.date >= START, "date"].unique())
    print(f"extracting term structure over {len(cal)} days ...", flush=True)
    recs = []
    for k, d in enumerate(cal):
        bc = load_bhavcopy(d)
        if bc.empty: continue
        bc = bc[bc.symbol.isin(syms) & bc.opt_type.isin(["CE", "PE"]) & bc.expiry.gt(d)]
        for sym, g in bc.groupby("symbol"):
            exps = sorted(g["expiry"].dropna().unique())
            if len(exps) < 2: continue
            spot = close_map.get((pd.Timestamp(d), sym), np.nan)
            f_str = straddle_iv(g[g.expiry == exps[0]], spot); n_str = straddle_iv(g[g.expiry == exps[1]], spot)
            if not (f_str and n_str): continue
            tf = max((pd.Timestamp(exps[0]) - pd.Timestamp(d)).days, 1); tn = max((pd.Timestamp(exps[1]) - pd.Timestamp(d)).days, 1)
            ivf = f_str / np.sqrt(tf / 365); ivn = n_str / np.sqrt(tn / 365)
            recs.append({"date": pd.Timestamp(d), "symbol": sym, "iv_front_proxy": ivf,
                         "term_slope": ivn - ivf, "implied_move_front": f_str})
        if (k + 1) % 200 == 0: print(f"  {k+1}/{len(cal)} ({time.time()-t0:.0f}s, recs={len(recs)})", flush=True)
    ts = pd.DataFrame(recs)
    ts.to_csv(ROOT / "reports" / "term_structure_features.csv", index=False)
    print(f"term-structure rows: {len(ts)} | backwardation share: {(ts['term_slope']<0).mean()*100:.0f}%", flush=True)

    oc = compute_clean_move_outcomes(market, universe=None, contract=CleanMoveContract())
    oc = oc[oc.status.eq("evaluated")][["date", "symbol", "side", "ceiling"]].copy(); oc["symbol"] = oc["symbol"].astype(str)
    mk = market.copy(); mk["symbol"] = mk["symbol"].astype(str); mk = mk.drop_duplicates(["date", "symbol"])
    df = oc.merge(mk[["date", "symbol", *list(dict.fromkeys([*LEAN, "close"]))]], on=["date", "symbol"], how="left")
    df = df.merge(ts, on=["date", "symbol"], how="left")
    df["rank"] = df["symbol"].map(rk); df = df[df["rank"].notna()]   # focused top-50 train
    df["eligible"] = df["atm_iv"].notna() & df["close"].ge(100); df["year"] = df["date"].dt.year
    df = df[df.date >= START].reset_index(drop=True)

    def evaluate(feats):
        rows = []
        for ty in TEST_YEARS:
            tr = df[df.year < ty]; te = df[(df.year == ty) & df.eligible & (df["rank"] <= 30)].copy(); te["p"] = np.nan
            for side in ("long", "short"):
                b = tr[tr.side.eq(side)]; imp = SimpleImputer(strategy="median").fit(_cl(b, feats))
                Xb = imp.transform(_cl(b, feats)); yb = (b["ceiling"] >= THR).astype(int)
                spw = (len(yb) - yb.sum()) / max(1, yb.sum()); m = te.side.eq(side)
                clf = XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.04, subsample=0.85, colsample_bytree=0.85,
                                    tree_method="hist", device="cuda", scale_pos_weight=spw, verbosity=0).fit(Xb, yb)
                te.loc[m, "p"] = clf.predict_proba(imp.transform(_cl(te[m], feats)))[:, 1]
            te["y"] = (te["ceiling"] >= THR).astype(int)
            t1 = te.sort_values("p", ascending=False).groupby("date").head(1)
            rows.append({"auc": roc_auc_score(te["y"], te["p"]), "p1": t1["y"].mean()*100})
        r = pd.DataFrame(rows); return {"AUC": round(r["auc"].mean(), 4), "prec@1": round(r["p1"].mean(), 1),
                                        "by_yr": "/".join(f"{v:.0f}" for v in r["p1"])}
    res = []
    for name, feats in [("LEAN (2021+ focused)", LEAN), ("LEAN + IV term structure", LEAN + TS)]:
        out = evaluate(feats); out["config"] = name; res.append(out); print(f"{name:26s} {out}", flush=True)
    pd.set_option("display.width", 200)
    print("\n===== ROUND B: IV term structure (backwardation) =====")
    print(pd.DataFrame(res)[["config", "AUC", "prec@1", "by_yr"]].to_string(index=False))


if __name__ == "__main__":
    main()
