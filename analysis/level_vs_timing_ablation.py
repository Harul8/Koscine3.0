"""Ablation: is the big-move signal in volatility LEVEL, TIMING, or both + engineered?
Train BROAD (all ~450) per tier target; eval precision@1 (daily top-pick) + AUC on the tier.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from koscine3.data.feature_registry import build_feature_registry
from koscine3.data.sources import load_market_data
from koscine3.data.universe import UniverseConfig, build_universe
from koscine3.outcomes.clean_move_contract import CleanMoveContract, compute_clean_move_outcomes
from lightgbm import LGBMClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score

TRAIN_END = pd.Timestamp("2023-12-31")
def _clean(f, c): return f[c].replace([np.inf, -np.inf], np.nan)

LEVEL = ["atm_iv", "atm_ce_iv", "atm_pe_iv", "atr_5", "atr_14", "atr_pct_14", "realized_vol_20",
         "range_pct", "bb_width_20", "nifty_realized_vol_20", "sector_vol_20"]
TIMING = ["atr_pct_14_rank_60d", "bb_width_20_rank_60d", "range_contraction_5v20", "compression_composite",
          "vol_5v20_ratio", "volume_dryup_score", "nr7_flag", "inside_bar_count_5d", "donchian_width_20",
          "atm_iv_chg_5", "atm_iv_ratio_20", "iv_skew_chg_5d", "iv_skew_norm", "pcr_oi_chg_5", "pcr_vol_chg_5",
          "oi_buildup_ratio", "oi_long_buildup", "oi_short_buildup", "oi_acceleration", "price_oi_divergence",
          "fut_oi_z_60d", "fut_oi_chg_5", "fut_chg_oi_ratio_20", "delivery_pct_chg_5", "delivery_qty_ratio_20",
          "turnover_ratio_20", "close_sma20_dist", "close_sma50_dist", "ema_20_slope_5d", "ema_50_slope_5d",
          "new_high_10d", "new_high_count_20d", "ret_5d_cs_rank", "ret_20d_cs_rank", "stock_rel_sector_ret_5d",
          "gap_up_count_20d", "consec_up_days", "days_to_earnings", "earnings_within_5d", "earnings_within_10d",
          "is_expiry_week", "days_to_month_end", "month", "atr_pct_14_cs_rank", "bb_width_20_cs_rank",
          "vol_sma20_ratio_cs_rank"]
NEW = ["atm_iv_z252", "atr_pct_14_z252", "ivrv_spread_ann", "comp_x_earn", "oi_x_comp"]


def main():
    market = load_market_data()
    reg = build_feature_registry(market)
    market = market.sort_values(["symbol", "date"]).copy()
    g = market.groupby("symbol")
    for col in ["atm_iv", "atr_pct_14"]:
        m = g[col].transform(lambda s: s.rolling(252, min_periods=60).mean())
        sd = g[col].transform(lambda s: s.rolling(252, min_periods=60).std())
        market[col + "_z252"] = ((market[col] - m) / sd).replace([np.inf, -np.inf], np.nan)
    dte = market["days_to_earnings"].fillna(999).clip(lower=0)
    market["ivrv_spread_ann"] = market["atm_iv"] - market["realized_vol_20"] * np.sqrt(252)
    market["comp_x_earn"] = market["compression_composite"] * (1.0 / (1.0 + dte))
    market["oi_x_comp"] = market["oi_buildup_ratio"] * market["range_contraction_5v20"]
    print("engineered features built", flush=True)

    oc = compute_clean_move_outcomes(market, universe=None, contract=CleanMoveContract())
    oc = oc[(oc.status == "evaluated") & (oc.side == "long")][["date", "symbol", "ceiling"]].copy()
    oc["symbol"] = oc["symbol"].astype(str)
    mk = market.copy(); mk["symbol"] = mk["symbol"].astype(str)
    allcols = LEVEL + TIMING + NEW
    df = oc.merge(mk[["date", "symbol", *allcols]], on=["date", "symbol"], how="left")
    uni = build_universe(market, UniverseConfig(cutoff_date="2025-12-31", top_n=50))
    rk = uni.set_index(uni["symbol"].astype(str))["rank"]
    df["rank"] = df["symbol"].map(rk)
    train = df[df.date <= TRAIN_END]

    sets = {"LEVEL only": LEVEL, "TIMING only": TIMING, "ALL + engineered": allcols}
    rows = []
    for tier, emask, thr in [("A top20 >=5%", df["rank"] <= 20, 0.05), ("B 21-50 >=10%", (df["rank"] > 20) & (df["rank"] <= 50), 0.10)]:
        ytr_all = (train["ceiling"] >= thr).astype(int)
        for sname, fs in sets.items():
            imp = SimpleImputer(strategy="median").fit(_clean(train, fs))
            clf = LGBMClassifier(n_estimators=400, learning_rate=0.04, num_leaves=31, subsample=0.85,
                                 colsample_bytree=0.85, class_weight="balanced", random_state=17,
                                 verbosity=-1).fit(imp.transform(_clean(train, fs)), ytr_all)
            ev = df[(df.date > TRAIN_END) & emask].copy()
            ev["y"] = (ev["ceiling"] >= thr).astype(int)
            ev["p"] = clf.predict_proba(imp.transform(_clean(ev, fs)))[:, 1]
            top = ev.sort_values("p", ascending=False).groupby("date").head(1)
            rows.append({"tier": tier, "feature_set": sname, "n_feats": len(fs),
                         "AUC": round(roc_auc_score(ev["y"], ev["p"]), 4),
                         "precision@1": round(top["y"].mean() * 100, 1), "base_rate": round(ev["y"].mean() * 100, 1)})
    pd.set_option("display.width", 200)
    print("\n===== LEVEL vs TIMING vs ALL+ENGINEERED (train broad ~450, eval per tier) =====")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
