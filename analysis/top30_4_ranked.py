"""FINAL single-tier: top-30 @4%. Daily top-3 (both directions pooled), ranked by calibrated
confidence, with expected move. Metrics: precision@1, hit@top3, P(>=1 of top3), by year + calib + sample."""
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

BASE_END, CAL_END, THR = pd.Timestamp("2022-12-31"), pd.Timestamp("2023-12-31"), 0.04
LEAN = ["atm_iv", "atr_pct_14", "atm_ce_iv", "atm_pe_iv", "nifty_realized_vol_20", "mkt_pct_above_sma50",
        "days_to_earnings", "atr_pct_14_cs_rank", "realized_vol_20", "atr_pct_14_rank_60d", "sector_vol_20",
        "ret_20d_cs_rank", "pcr_oi", "fut_oi_ratio_20", "close_sma50_dist", "vol_5v20_ratio",
        "atm_iv_ratio_20", "donchian_width_20", "mkt_pct_above_sma20", "month"]
def _clean(f, c): return f[c].replace([np.inf, -np.inf], np.nan)


def main():
    market = load_market_data()
    uni = build_universe(market, UniverseConfig(cutoff_date="2025-12-31", top_n=30))
    oc = compute_clean_move_outcomes(market, universe=uni, contract=CleanMoveContract())
    oc = oc[oc.status.eq("evaluated")][["date", "symbol", "side", "ceiling"]].copy()
    oc["symbol"] = oc["symbol"].astype(str)
    mk = market.copy(); mk["symbol"] = mk["symbol"].astype(str)
    df = oc.merge(mk[["date", "symbol", "close", *LEAN]], on=["date", "symbol"], how="left")
    df["eligible"] = df["atm_iv"].notna() & df["close"].ge(100)
    df["year"] = df["date"].dt.year
    df["dir"] = np.where(df["side"].eq("long"), "CALL", "PUT")
    base = df[df.date <= BASE_END]; cal = df[(df.date > BASE_END) & (df.date <= CAL_END)]; evl = df[df.date > CAL_END].copy()

    parts = []
    for side in ("long", "short"):
        b, c = base[base.side.eq(side)], cal[cal.side.eq(side)]
        imp = SimpleImputer(strategy="median").fit(_clean(b, LEAN))
        clf = LGBMClassifier(n_estimators=400, learning_rate=0.04, num_leaves=31, subsample=0.85,
                             colsample_bytree=0.85, class_weight="balanced", random_state=17, verbosity=-1).fit(imp.transform(_clean(b, LEAN)), (b["ceiling"] >= THR).astype(int))
        calib = CalibratedClassifierCV(FrozenEstimator(clf), method="isotonic").fit(imp.transform(_clean(c, LEAN)), (c["ceiling"] >= THR).astype(int))
        reg = LGBMRegressor(n_estimators=400, learning_rate=0.04, num_leaves=31, subsample=0.85,
                            colsample_bytree=0.85, random_state=17, verbosity=-1).fit(imp.transform(_clean(b, LEAN)), b["ceiling"].clip(0, 0.5))
        m = evl[evl.side.eq(side)].copy(); X = imp.transform(_clean(m, LEAN))
        m["confidence"] = calib.predict_proba(X)[:, 1]; m["exp_move"] = np.clip(reg.predict(X), 0, None)
        parts.append(m)
    ev = pd.concat(parts, ignore_index=True)
    ev = ev[ev["eligible"]].copy()
    ev["hit"] = (ev["ceiling"] >= THR).astype(int)
    ev["rank_in_day"] = ev.groupby("date")["confidence"].rank(ascending=False, method="first")  # pool both dirs
    top3 = ev[ev["rank_in_day"] <= 3].copy()

    print("===== top-30 @4% (point-in-time, calibrated, daily top-3 pooled both dirs) =====")
    rows = []
    for yr in ["all", 2024, 2025, 2026]:
        p1 = top3[top3.rank_in_day == 1]; p1 = p1 if yr == "all" else p1[p1.year == yr]
        t3 = top3 if yr == "all" else top3[top3.year == yr]
        any3 = t3.groupby("date")["hit"].max()
        rows.append({"year": yr, "precision@1": round(p1["hit"].mean()*100, 1), "hit@top3": round(t3["hit"].mean()*100, 1),
                     "P(>=1 of top3)": round(any3.mean()*100, 1), "avg_conf@1": round(p1["confidence"].mean(), 2),
                     "avg_exp_move@1%": round(p1["exp_move"].mean()*100, 1), "days": int(p1["date"].nunique())})
    pd.set_option("display.width", 220)
    print(pd.DataFrame(rows).to_string(index=False))

    pa = top3[top3.rank_in_day == 1].copy(); pa["b"] = pd.cut(pa["confidence"], [0, .3, .4, .5, .6, .7, 1.0])
    print("\n===== calibration (top-1): predicted vs actual =====")
    print(pa.groupby("b", observed=True).agg(n=("hit", "size"), pred=("confidence", "mean"), actual=("hit", "mean")).round(3).to_string())

    out = top3.sort_values(["date", "rank_in_day"])[["date", "rank_in_day", "symbol", "dir", "confidence", "exp_move", "ceiling", "hit"]].copy()
    out["confidence"] = out["confidence"].round(3); out["exp_move_%"] = (out["exp_move"]*100).round(1); out["actual_move_%"] = (out["ceiling"]*100).round(1)
    out.drop(columns=["exp_move", "ceiling"]).to_csv(ROOT / "reports" / "daily_top30_4_top3.csv", index=False)
    print("\n===== SAMPLE (recent days, top-3) =====")
    print(out[out.date >= out.date.max() - pd.Timedelta(days=12)].drop(columns=["exp_move", "ceiling"]).to_string(index=False))
    print("\nsaved: reports/daily_top30_4_top3.csv")


if __name__ == "__main__":
    main()
