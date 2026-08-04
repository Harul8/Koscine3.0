"""Walk-forward model harness for >=4% move prediction (top-30, point-in-time). Memory-safe.
Train BROAD (all ~450) on years < test_year; eval on test_year top-30 eligible.
Reports AUC, precision@1, precision@3 per test year + mean, for a set of configs.
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
from lightgbm import LGBMClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score

THR = 0.04
TEST_YEARS = [2024, 2025, 2026]
LEAN = ["atm_iv", "atr_pct_14", "atm_ce_iv", "atm_pe_iv", "nifty_realized_vol_20", "mkt_pct_above_sma50",
        "days_to_earnings", "atr_pct_14_cs_rank", "realized_vol_20", "atr_pct_14_rank_60d", "sector_vol_20",
        "ret_20d_cs_rank", "pcr_oi", "fut_oi_ratio_20", "close_sma50_dist", "vol_5v20_ratio",
        "atm_iv_ratio_20", "donchian_width_20", "mkt_pct_above_sma20", "month"]
EXTRA = ["bb_width_20", "range_contraction_5v20", "compression_composite", "volume_dryup_score",
         "oi_buildup_ratio", "oi_acceleration", "price_oi_divergence", "fut_oi_z_60d", "iv_skew_chg_5d",
         "iv_skew_norm", "pcr_oi_chg_5", "delivery_pct", "delivery_qty_ratio_20", "ema_50_slope_5d",
         "new_high_count_20d", "adx_14", "rel_ret_5d_vs_nifty", "stock_rel_sector_ret_5d",
         "gap_up_count_20d", "earnings_within_5d", "is_expiry_week", "ret_5d_cs_rank",
         "close_sma20_dist", "fut_close_dist", "max_pain_dist"]
EXTENDED = LEAN + EXTRA
def _clean(f, c): return f[c].replace([np.inf, -np.inf], np.nan)


def make_clf(cfg, seed):
    p = dict(n_estimators=400, learning_rate=0.04, num_leaves=31, subsample=0.85, colsample_bytree=0.85,
             class_weight="balanced", random_state=seed, verbosity=-1, n_jobs=-1)
    p.update(cfg.get("params", {}))
    return LGBMClassifier(**p)


def evaluate(df, feats, cfg):
    rows = []
    for ty in TEST_YEARS:
        tr = df[df.year < ty]; te = df[(df.year == ty) & df.eligible & (df["rank"] <= 30)].copy()
        te["p"] = np.nan
        for side in ("long", "short"):
            b = tr[tr.side.eq(side)]
            imp = SimpleImputer(strategy="median").fit(_clean(b, feats)); Xb = imp.transform(_clean(b, feats))
            yb = (b["ceiling"] >= THR).astype(int)
            seeds = cfg.get("seeds", [17])
            m = te.side.eq(side); Xt = imp.transform(_clean(te[m], feats))
            preds = np.zeros(int(m.sum()))
            for s in seeds:
                preds += make_clf(cfg, s).fit(Xb, yb).predict_proba(Xt)[:, 1]
            te.loc[m, "p"] = preds / len(seeds)
        te["y"] = (te["ceiling"] >= THR).astype(int)
        top1 = te.sort_values("p", ascending=False).groupby("date").head(1)
        top3 = te.sort_values("p", ascending=False).groupby("date").head(3)
        rows.append({"auc": roc_auc_score(te["y"], te["p"]), "p1": top1["y"].mean()*100, "p3": top3["y"].mean()*100})
    r = pd.DataFrame(rows)
    return {"AUC": round(r["auc"].mean(), 4), "prec@1": round(r["p1"].mean(), 1), "prec@3": round(r["p3"].mean(), 1),
            "by_yr_p@1": "/".join(f"{v:.0f}" for v in r["p1"])}


def main():
    t0 = time.time()
    needed = sorted(set(["date", "symbol", "open", "high", "low", "close", "turnover_lacs", "volume", *EXTENDED]))
    market = load_market_data(columns=needed)
    oc = compute_clean_move_outcomes(market, universe=None, contract=CleanMoveContract())
    oc = oc[oc.status.eq("evaluated")][["date", "symbol", "side", "ceiling"]].copy()
    oc["symbol"] = oc["symbol"].astype(str)
    mk = market.copy(); mk["symbol"] = mk["symbol"].astype(str); mk = mk.drop_duplicates(["date", "symbol"])
    cols = list(dict.fromkeys([*EXTENDED, "close"]))
    df = oc.merge(mk[["date", "symbol", *cols]], on=["date", "symbol"], how="left")
    for c in EXTENDED:
        df[c] = df[c].astype(np.float32)
    rk = build_universe(market, UniverseConfig(cutoff_date="2025-12-31", top_n=30))
    rk = rk.set_index(rk["symbol"].astype(str))["rank"]
    df["rank"] = df["symbol"].map(rk)
    df["eligible"] = df["atm_iv"].notna() & df["close"].ge(100)
    df["year"] = df["date"].dt.year
    df = df.reset_index(drop=True)
    print(f"data ready {time.time()-t0:.0f}s | rows {len(df):,} | mem {df.memory_usage(deep=True).sum()/1e9:.2f}GB", flush=True)

    configs = [
        {"name": "C1 baseline lean", "feats": LEAN, "params": {}},
        {"name": "C2 extended-45", "feats": EXTENDED, "params": {}},
        {"name": "C3 tuned lean", "feats": LEAN, "params": dict(n_estimators=800, learning_rate=0.02, num_leaves=63, min_child_samples=200, reg_lambda=5.0)},
        {"name": "C4 ens-3 lean", "feats": LEAN, "params": {}, "seeds": [17, 41, 99]},
    ]
    res = []
    for cfg in configs:
        t = time.time(); out = evaluate(df, cfg["feats"], cfg); out["config"] = cfg["name"]
        res.append(out); print(f"{cfg['name']:18s} {out} ({time.time()-t:.0f}s)", flush=True)
    pd.set_option("display.width", 200)
    print("\n===== ROUND 1: walk-forward >=4% precision (train broad, eval top-30) =====")
    print(pd.DataFrame(res)[["config", "AUC", "prec@1", "prec@3", "by_yr_p@1"]].to_string(index=False))


if __name__ == "__main__":
    main()
