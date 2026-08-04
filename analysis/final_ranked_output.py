"""FINAL production-style output: top-20 @4% + 21-40 @6%.
Per day per tier: rank candidates, output rank, side(call/put), CALIBRATED confidence,
EXPECTED move (regressor), realized move. Metrics: precision@1, precision@3 (>=1 of 3 hits),
by year. Calibration reliability check. Saves daily top-3 to CSV + prints a recent sample.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from koscine3.data.sources import load_market_data
from koscine3.data.universe import UniverseConfig, build_universe
from koscine3.outcomes.clean_move_contract import CleanMoveContract, compute_clean_move_outcomes
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.impute import SimpleImputer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator

BASE_END, CAL_END = pd.Timestamp("2022-12-31"), pd.Timestamp("2023-12-31")
LEAN = ["atm_iv", "atr_pct_14", "atm_ce_iv", "atm_pe_iv", "nifty_realized_vol_20", "mkt_pct_above_sma50",
        "days_to_earnings", "atr_pct_14_cs_rank", "realized_vol_20", "atr_pct_14_rank_60d", "sector_vol_20",
        "ret_20d_cs_rank", "pcr_oi", "fut_oi_ratio_20", "close_sma50_dist", "vol_5v20_ratio",
        "atm_iv_ratio_20", "donchian_width_20", "mkt_pct_above_sma20", "month"]
def _clean(f, c): return f[c].replace([np.inf, -np.inf], np.nan)


def main():
    market = load_market_data()
    oc = compute_clean_move_outcomes(market, universe=build_universe(market, UniverseConfig(cutoff_date="2025-12-31", top_n=40)), contract=CleanMoveContract())
    oc = oc[oc.status.eq("evaluated")][["date", "symbol", "side", "ceiling"]].copy()
    oc["symbol"] = oc["symbol"].astype(str)
    mk = market.copy(); mk["symbol"] = mk["symbol"].astype(str)
    df = oc.merge(mk[["date", "symbol", "close", *LEAN]], on=["date", "symbol"], how="left")
    rk = build_universe(market, UniverseConfig(cutoff_date="2025-12-31", top_n=40))
    rk = rk.set_index(rk["symbol"].astype(str))["rank"]
    df["rank"] = df["symbol"].map(rk)
    df["tier"] = np.where(df["rank"] <= 20, "A_top20", "B_21-40")
    df["thr"] = np.where(df["rank"] <= 20, 0.04, 0.06)
    df["eligible"] = df["atm_iv"].notna() & df["close"].ge(100)
    df["year"] = df["date"].dt.year
    df["dir"] = np.where(df["side"].eq("long"), "CALL", "PUT")

    base = df[df.date <= BASE_END]; cal = df[(df.date > BASE_END) & (df.date <= CAL_END)]; evl = df[df.date > CAL_END].copy()

    # train per (tier-threshold, side): calibrated confidence + expected-move regressor
    models = {}
    for thr in (0.04, 0.06):
        for side in ("long", "short"):
            b = base[base.side.eq(side)]
            imp = SimpleImputer(strategy="median").fit(_clean(b, LEAN))
            clf = LGBMClassifier(n_estimators=400, learning_rate=0.04, num_leaves=31, subsample=0.85,
                                 colsample_bytree=0.85, class_weight="balanced", random_state=17, verbosity=-1).fit(imp.transform(_clean(b, LEAN)), (b["ceiling"] >= thr).astype(int))
            c = cal[cal.side.eq(side)]
            calib = CalibratedClassifierCV(FrozenEstimator(clf), method="isotonic").fit(imp.transform(_clean(c, LEAN)), (c["ceiling"] >= thr).astype(int))
            reg = LGBMRegressor(n_estimators=400, learning_rate=0.04, num_leaves=31, subsample=0.85,
                                colsample_bytree=0.85, random_state=17, verbosity=-1).fit(imp.transform(_clean(b, LEAN)), b["ceiling"].clip(0, 0.5))
            models[(thr, side)] = (imp, calib, reg)

    # score eval
    parts = []
    for (thr, side), (imp, calib, reg) in models.items():
        m = evl[evl.side.eq(side) & (evl.thr == thr)].copy()
        if m.empty: continue
        X = imp.transform(_clean(m, LEAN))
        m["confidence"] = calib.predict_proba(X)[:, 1]
        m["exp_move"] = np.clip(reg.predict(X), 0, None)
        parts.append(m)
    ev = pd.concat(parts, ignore_index=True)
    ev = ev[ev["eligible"]].copy()
    ev["hit"] = (ev["ceiling"] >= ev["thr"]).astype(int)
    ev["rank_in_day"] = ev.groupby(["date", "tier"])["confidence"].rank(ascending=False, method="first")
    top3 = ev[ev["rank_in_day"] <= 3].copy()

    print("===== PRECISION: top-20 @4% & 21-40 @6% (point-in-time, calibrated) =====")
    rows = []
    for tier in ("A_top20", "B_21-40"):
        t = top3[top3.tier.eq(tier)]
        p1 = t[t.rank_in_day == 1]
        for yr in ["all", 2024, 2025, 2026]:
            d1 = p1 if yr == "all" else p1[p1.year == yr]
            d3 = t if yr == "all" else t[t.year == yr]
            any3 = d3.groupby("date")["hit"].max()  # >=1 of top-3 hit that day
            rows.append({"tier": tier, "year": yr, "precision@1": round(d1["hit"].mean()*100, 1),
                         "hit_rate@top3": round(d3["hit"].mean()*100, 1), "P(>=1 of top3)": round(any3.mean()*100, 1),
                         "avg_conf@1": round(d1["confidence"].mean(), 2), "avg_exp_move@1%": round(d1["exp_move"].mean()*100, 1)})
    pd.set_option("display.width", 220)
    print(pd.DataFrame(rows).to_string(index=False))

    # calibration reliability on tier-A top picks
    pa = top3[(top3.tier == "A_top20") & (top3.rank_in_day == 1)].copy()
    pa["conf_bin"] = pd.cut(pa["confidence"], [0, .3, .4, .5, .6, .7, 1.0])
    print("\n===== calibration check (tier-A top-1): predicted confidence vs actual hit =====")
    print(pa.groupby("conf_bin", observed=True).agg(n=("hit", "size"), pred_conf=("confidence", "mean"), actual_hit=("hit", "mean")).round(3).to_string())

    out = top3.sort_values(["date", "tier", "rank_in_day"])[
        ["date", "tier", "rank_in_day", "symbol", "dir", "confidence", "exp_move", "ceiling", "hit"]].copy()
    out["confidence"] = out["confidence"].round(3); out["exp_move_%"] = (out["exp_move"]*100).round(1)
    out["actual_move_%"] = (out["ceiling"]*100).round(1)
    out = out.drop(columns=["exp_move", "ceiling"])
    out.to_csv(ROOT / "reports" / "daily_ranked_top3.csv", index=False)
    print("\n===== SAMPLE daily output (most recent 6 trading days, tier A top-3) =====")
    recent = out[(out.tier == "A_top20") & (out.date >= out.date.max() - pd.Timedelta(days=12))]
    print(recent.to_string(index=False))
    print("\nfull daily top-3 saved: reports/daily_ranked_top3.csv")


if __name__ == "__main__":
    main()
