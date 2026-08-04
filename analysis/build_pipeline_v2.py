"""2-group walk-forward pipeline. A = top-30 by market cap, B = next-35 by turnover.
Both 5d >=4%. Train broad (~450), isotonic-calibrated confidence + expected-move regressor.
Saves full predictions + t+3-cooldown daily shortlist (1/day per group) + combined. Metric = underlying move.
"""
from __future__ import annotations
import sys, time, json
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
PRED = ROOT / "reports" / "predictions"; PRED.mkdir(parents=True, exist_ok=True)
TEST_YEARS = [2024, 2025, 2026]; EVAL_END = pd.Timestamp("2026-05-31"); COOLDOWN = 3
GROUP_THR = {"A_mcap30": 0.03, "B_turn35": 0.04}   # A (mega-caps) lower bar; B (movers) 4%
LEAN = ["atm_iv", "atr_pct_14", "atm_ce_iv", "atm_pe_iv", "nifty_realized_vol_20", "mkt_pct_above_sma50",
        "days_to_earnings", "atr_pct_14_cs_rank", "realized_vol_20", "atr_pct_14_rank_60d", "sector_vol_20",
        "ret_20d_cs_rank", "pcr_oi", "fut_oi_ratio_20", "close_sma50_dist", "vol_5v20_ratio",
        "atm_iv_ratio_20", "donchian_width_20", "mkt_pct_above_sma20", "month"]
from xgboost import XGBClassifier, XGBRegressor
from sklearn.impute import SimpleImputer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import roc_auc_score
from koscine3.data.sources import load_market_data
from koscine3.outcomes.clean_move_contract import CleanMoveContract, compute_clean_move_outcomes
def _cl(f): return f[LEAN].replace([np.inf, -np.inf], np.nan).astype(np.float32)


def run_group(df, syms, thr):
    preds = []
    for T in TEST_YEARS:
        base = df[df.year < T - 1]; calib = df[df.year == T - 1]
        ev = df[(df.year == T) & df.eligible & df.symbol.isin(syms) & (df.date <= EVAL_END)].copy()
        if ev.empty: continue
        ev["confidence"] = np.nan; ev["exp_move"] = np.nan
        for side in ("long", "short"):
            b = base[base.side.eq(side)]; c = calib[calib.side.eq(side)]; m = ev.side.eq(side)
            if b.empty or c.empty or m.sum() == 0: continue
            imp = SimpleImputer(strategy="median").fit(_cl(b)); Xb = imp.transform(_cl(b))
            yb = (b["ceiling"] >= thr).astype(int); spw = (len(yb)-yb.sum())/max(1, yb.sum())
            clf = XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.04, subsample=0.85, colsample_bytree=0.85,
                                tree_method="hist", device="cuda", scale_pos_weight=spw, verbosity=0).fit(Xb, yb)
            cal = CalibratedClassifierCV(FrozenEstimator(clf), method="isotonic").fit(imp.transform(_cl(c)), (c["ceiling"] >= thr).astype(int))
            reg = XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.04, subsample=0.85, colsample_bytree=0.85,
                               tree_method="hist", device="cuda", verbosity=0).fit(Xb, b["ceiling"].clip(0, 0.5))
            Xe = imp.transform(_cl(ev[m]))
            ev.loc[m, "confidence"] = cal.predict_proba(Xe)[:, 1]; ev.loc[m, "exp_move"] = np.clip(reg.predict(Xe), 0, None)
        ev = ev.dropna(subset=["confidence"]); ev["hit"] = (ev["ceiling"] >= thr).astype(int)
        ev["rank_in_day"] = ev.groupby("date")["confidence"].rank(ascending=False, method="first")
        preds.append(ev)
    p = pd.concat(preds, ignore_index=True)
    p["dir"] = np.where(p.side.eq("long"), "CALL", "PUT")
    p["confidence"] = p["confidence"].round(3); p["exp_move_%"] = (p["exp_move"]*100).round(1); p["actual_move_%"] = (p["ceiling"]*100).round(2)
    return p


def cooldown_top1(p, cal):
    p = p.sort_values(["date", "confidence"], ascending=[True, False]); last, keep = {}, []
    for day, g in p.groupby("date", sort=True):
        i = cal[pd.Timestamp(day)]
        for idx, s in zip(g.index, g["symbol"]):
            if i - last.get(s, -10**9) < COOLDOWN: continue
            keep.append(idx); last[s] = i; break
    return p.loc[keep]


def summ(p, sl, label):
    out = []
    for yr in ["all", 2024, 2025, 2026]:
        d = p if yr == "all" else p[p.date.dt.year == yr]
        s = sl if yr == "all" else sl[sl.date.dt.year == yr]
        t1 = d[d.rank_in_day == 1]; t3 = d[d.rank_in_day <= 3]
        out.append({"group": label, "year": yr, "base>=4%": round(d["hit"].mean()*100, 1),
                    "prec@1(full)": round(t1["hit"].mean()*100, 1), "P(>=1of3)": round(t3.groupby("date")["hit"].max().mean()*100, 1),
                    "prec@1(cooled)": round(s["hit"].mean()*100, 1), "trades/yr": int(len(s)/(2.4 if yr == "all" else 1)),
                    "stocks": s["symbol"].nunique()})
    return out


def main():
    t0 = time.time()
    grp = json.load(open(ROOT / "reports" / "universe_groups.json"))
    A, B = set(grp["A_mcap30"]), set(grp["B_turn35"])
    market = load_market_data(columns=sorted(set(["date", "symbol", "open", "high", "low", "close", "turnover_lacs", "volume", *LEAN])))
    oc = compute_clean_move_outcomes(market, universe=None, contract=CleanMoveContract(window_days=5))
    oc = oc[oc.status.eq("evaluated")][["date", "symbol", "side", "ceiling"]].copy(); oc["symbol"] = oc["symbol"].astype(str)
    mk = market.copy(); mk["symbol"] = mk["symbol"].astype(str); mk = mk.drop_duplicates(["date", "symbol"])
    df = oc.merge(mk[["date", "symbol", *list(dict.fromkeys([*LEAN, "close"]))]], on=["date", "symbol"], how="left")
    df["eligible"] = df["atm_iv"].notna() & df["close"].ge(100); df["year"] = df["date"].dt.year; df = df.reset_index(drop=True)
    cal = {pd.Timestamp(d): i for i, d in enumerate(sorted(df["date"].unique()))}
    print(f"data ready {time.time()-t0:.0f}s", flush=True)

    rows, shorts = [], {}
    for label, syms in [("A_mcap30", A), ("B_turn35", B)]:
        p = run_group(df, syms, GROUP_THR[label]); p.to_csv(PRED / f"group_{label}_predictions.csv", index=False)
        sl = cooldown_top1(p, cal); sl.to_csv(PRED / f"group_{label}_shortlist_t3.csv", index=False); shorts[label] = sl
        rows += summ(p, sl, label); print(f"{label} done ({time.time()-t0:.0f}s)", flush=True)
    comb = cooldown_top1(pd.concat(shorts.values(), ignore_index=True), cal); comb.to_csv(PRED / "group_combined_shortlist_t3.csv", index=False)
    rows.append({"group": "COMBINED", "year": "all", "base>=4%": np.nan, "prec@1(full)": np.nan, "P(>=1of3)": np.nan,
                 "prec@1(cooled)": round(comb["hit"].mean()*100, 1), "trades/yr": int(len(comb)/2.4), "stocks": comb["symbol"].nunique()})
    pd.set_option("display.width", 220)
    print("\n===== 2-GROUP PIPELINE (5d >=4%, walk-forward) =====")
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"saved to {PRED}/group_*")


if __name__ == "__main__":
    main()
