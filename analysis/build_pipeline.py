"""COMPLETE 3-book walk-forward pipeline. Saves calibrated 2024-2026May predictions.
Book A: 5d >=4%, top-30 | Book B: 10d >=8%, top-30 | Book C: 10d >=10%, rank 31-65.
Train BROAD (~450), walk-forward (base < T-1, isotonic-calibrate on T-1, predict T), per side.
Output per pick: direction, calibrated confidence, expected move, daily rank. Metric = underlying
favorable-move hit rate (no option EV). Saves full predictions + daily shortlists + combined.
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
from xgboost import XGBClassifier, XGBRegressor
from sklearn.impute import SimpleImputer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import roc_auc_score

OUT = ROOT / "reports" / "predictions"; OUT.mkdir(parents=True, exist_ok=True)
TEST_YEARS = [2024, 2025, 2026]
EVAL_END = pd.Timestamp("2026-05-31")
LEAN = ["atm_iv", "atr_pct_14", "atm_ce_iv", "atm_pe_iv", "nifty_realized_vol_20", "mkt_pct_above_sma50",
        "days_to_earnings", "atr_pct_14_cs_rank", "realized_vol_20", "atr_pct_14_rank_60d", "sector_vol_20",
        "ret_20d_cs_rank", "pcr_oi", "fut_oi_ratio_20", "close_sma50_dist", "vol_5v20_ratio",
        "atm_iv_ratio_20", "donchian_width_20", "mkt_pct_above_sma20", "month"]
BOOKS = [
    {"name": "A_5d_4pct", "window": 5, "thr": 0.04, "rlo": 1, "rhi": 30},
    {"name": "B_10d_8pct", "window": 10, "thr": 0.08, "rlo": 1, "rhi": 30},
    {"name": "C_10d_10pct_midcap", "window": 10, "thr": 0.10, "rlo": 31, "rhi": 65},
]
def _cl(f): return f[LEAN].replace([np.inf, -np.inf], np.nan).astype(np.float32)


def xgbclf(spw):
    return XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.04, subsample=0.85,
                         colsample_bytree=0.85, tree_method="hist", device="cuda", scale_pos_weight=spw, verbosity=0)


def run_book(df_book, ceil_col, book):
    """Walk-forward predictions for one book. df_book has features + ceil_col + rank/eligible/year/side."""
    preds = []
    for T in TEST_YEARS:
        base = df_book[df_book.year < T - 1]; calib = df_book[df_book.year == T - 1]
        ev = df_book[(df_book.year == T) & df_book.eligible & df_book["rank"].between(book["rlo"], book["rhi"])
                     & (df_book.date <= EVAL_END)].copy()
        if ev.empty:
            continue
        ev["confidence"] = np.nan; ev["exp_move"] = np.nan
        for side in ("long", "short"):
            b = base[base.side.eq(side)]; c = calib[calib.side.eq(side)]; m = ev.side.eq(side)
            if b.empty or c.empty or m.sum() == 0:
                continue
            imp = SimpleImputer(strategy="median").fit(_cl(b)); Xb = imp.transform(_cl(b))
            yb = (b[ceil_col] >= book["thr"]).astype(int); spw = (len(yb) - yb.sum()) / max(1, yb.sum())
            clf = xgbclf(spw).fit(Xb, yb)
            cal = CalibratedClassifierCV(FrozenEstimator(clf), method="isotonic").fit(imp.transform(_cl(c)), (c[ceil_col] >= book["thr"]).astype(int))
            reg = XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.04, subsample=0.85,
                               colsample_bytree=0.85, tree_method="hist", device="cuda", verbosity=0).fit(Xb, b[ceil_col].clip(0, 0.5))
            Xe = imp.transform(_cl(ev[m]))
            ev.loc[m, "confidence"] = cal.predict_proba(Xe)[:, 1]
            ev.loc[m, "exp_move"] = np.clip(reg.predict(Xe), 0, None)
        ev = ev.dropna(subset=["confidence"])
        ev["hit"] = (ev[ceil_col] >= book["thr"]).astype(int)
        ev["rank_in_day"] = ev.groupby("date")["confidence"].rank(ascending=False, method="first")
        preds.append(ev)
    p = pd.concat(preds, ignore_index=True)
    p["book"] = book["name"]; p["dir"] = np.where(p.side.eq("long"), "CALL", "PUT")
    p["actual_move_%"] = (p[ceil_col] * 100).round(2); p["exp_move_%"] = (p["exp_move"] * 100).round(1)
    p["confidence"] = p["confidence"].round(3)
    return p[["date", "book", "rank_in_day", "symbol", "side", "dir", "confidence", "exp_move_%", "actual_move_%", "hit", "year"]]


def main():
    t0 = time.time()
    market = load_market_data(columns=sorted(set(["date", "symbol", "open", "high", "low", "close", "turnover_lacs", "volume", *LEAN])))
    uni = build_universe(market, UniverseConfig(cutoff_date="2025-12-31", top_n=65))
    rk = uni.set_index(uni["symbol"].astype(str))["rank"]
    mk = market.copy(); mk["symbol"] = mk["symbol"].astype(str); mk = mk.drop_duplicates(["date", "symbol"])
    feat = mk[["date", "symbol", *list(dict.fromkeys([*LEAN, "close"]))]]

    outc = {}
    for w in (5, 10):
        oc = compute_clean_move_outcomes(market, universe=None, contract=CleanMoveContract(window_days=w))
        oc = oc[oc.status.eq("evaluated")][["date", "symbol", "side", "ceiling"]].rename(columns={"ceiling": f"ceil_{w}"})
        oc["symbol"] = oc["symbol"].astype(str); outc[w] = oc
    print(f"outcomes ready {time.time()-t0:.0f}s", flush=True)

    summaries, all_preds = [], []
    for book in BOOKS:
        w = book["window"]; oc = outc[w]
        df = oc.merge(feat, on=["date", "symbol"], how="left")
        df["rank"] = df["symbol"].map(rk); df["eligible"] = df["atm_iv"].notna() & df["close"].ge(100)
        df["year"] = df["date"].dt.year
        p = run_book(df, f"ceil_{w}", book)
        p.to_csv(OUT / f"{book['name']}_predictions.csv", index=False)
        # daily top-3 shortlist for this book
        p[p.rank_in_day <= 3].to_csv(OUT / f"{book['name']}_daily_top3.csv", index=False)
        all_preds.append(p)
        # metrics (underlying favorable-move precision)
        for yr in ["all", 2024, 2025, 2026]:
            d = p if yr == "all" else p[p.year == yr]
            d1 = d[d.rank_in_day == 1]; d3 = d[d.rank_in_day <= 3]
            any3 = d3.groupby("date")["hit"].max()
            summaries.append({"book": book["name"], "year": yr, "elig_rows": len(d),
                              "prec@1": round(d1["hit"].mean()*100, 1), "P(>=1 of top3)": round(any3.mean()*100, 1),
                              "top1_days/yr": int(d1["date"].nunique() / (1 if yr != "all" else 2.4))})
        print(f"{book['name']:20s} done ({time.time()-t0:.0f}s)", flush=True)

    # combined daily shortlist: top-1 of each book per day, dedup (date,symbol) keep highest confidence
    comb = pd.concat([p[p.rank_in_day <= (3 if p['book'].iloc[0].startswith('A') else 2)] for p in all_preds], ignore_index=True)
    comb = comb.sort_values("confidence", ascending=False).drop_duplicates(["date", "symbol"], keep="first")
    comb = comb.sort_values(["date", "confidence"], ascending=[True, False])
    comb.to_csv(OUT / "combined_daily_shortlist.csv", index=False)

    summ = pd.DataFrame(summaries)
    summ.to_csv(OUT / "_summary.csv", index=False)
    pd.set_option("display.width", 220)
    print("\n===== PIPELINE SUMMARY (underlying favorable-move precision) =====")
    print(summ.to_string(index=False))
    print(f"\ncombined distinct trades 2024-26May: {len(comb)} | distinct stocks: {comb['symbol'].nunique()} | "
          f"~{int(len(comb)/2.4)}/yr")
    print(f"saved to {OUT}")
    print(f"total runtime {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
