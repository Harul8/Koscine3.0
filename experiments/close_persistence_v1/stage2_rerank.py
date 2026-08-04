"""Close-persistence experiment - STAGE-2 learned re-ranker (walk-forward, sandbox-only).

Stage 1 (frozen PROD): confidence -> daily top-N candidate gate (magnitude / top-3-mover).
Stage 2 (this file):    XGB classifiers trained on the CLOSE target
                          p_above = P(close_move >= thr),  p_opp = P(close_move < 0)
                        walk-forward (base<T-1, isotonic-calibrate T-1, predict T), per group/side.
Selection: gate top-N by PROD confidence (t+3 cooldown), pick 2 by  p_above - lambda*p_opp.

Reports OOS AUC (is close-persistence even learnable?) + close-outcome table vs baseline & ORACLE.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from largemove import pipeline as P
from largemove.config import PROD, XGB_CLF_PARAMS
from koscine3.data.sources import load_market_data

THR = dict(PROD.group_thresholds)
FEATS = list(PROD.features)
COOLDOWN = PROD.cooldown_trading_days
PROD_PRED = HERE.parents[1] / "locks" / "prod_largemove_v1" / "predictions"
OUT = HERE / "stage2_oos.parquet"
pd.set_option("display.width", 240)


# ------------------------------------------------------------------- close target
def add_close_move(df: pd.DataFrame):
    m = load_market_data(columns=["date", "symbol", "open", "close"])
    m["symbol"] = m["symbol"].astype(str)
    m = m.sort_values(["symbol", "date"])
    g = m.groupby("symbol", sort=False)
    cm = pd.DataFrame({
        "date": m["date"].values, "symbol": m["symbol"].values,
        "entry_open": g["open"].shift(-1).values, "win_close": g["close"].shift(-5).values,
    })
    df = df.merge(cm, on=["date", "symbol"], how="left")
    df["close_move"] = np.where(
        df.side.eq("long"), (df.win_close - df.entry_open) / df.entry_open,
        (df.entry_open - df.win_close) / df.entry_open,
    )
    cal = {pd.Timestamp(d): i for i, d in enumerate(sorted(m["date"].unique()))}
    return df, cal


def fit_clf(base, calib, feats, y_base, y_calib):
    imp = SimpleImputer(strategy="median").fit(P._clean(base, feats))
    Xb = imp.transform(P._clean(base, feats))
    spw = (len(y_base) - y_base.sum()) / max(1, y_base.sum())
    clf = XGBClassifier(scale_pos_weight=spw, **XGB_CLF_PARAMS).fit(Xb, y_base)
    cal = CalibratedClassifierCV(FrozenEstimator(clf), method="isotonic").fit(
        imp.transform(P._clean(calib, feats)), y_calib)
    return imp, cal


def proba(imp, cal, frame, feats):
    return cal.predict_proba(imp.transform(P._clean(frame, feats)))[:, 1]


def walk_forward_close(df):
    parts = []
    for T in PROD.test_years:
        base, calib = df[df.year < T - 1], df[df.year == T - 1]
        ev = df[(df.year == T) & df.eligible].copy()
        if ev.empty:
            continue
        ev["p_above"] = np.nan
        ev["p_opp"] = np.nan
        for group, thr in PROD.group_thresholds:
            for side in ("long", "short"):
                b = base[(base.group == group) & (base.side == side)]
                c = calib[(calib.group == group) & (calib.side == side)]
                m = (ev.group == group) & (ev.side == side)
                if b.empty or c.empty or not m.any():
                    continue
                ia, ca = fit_clf(b, c, FEATS, (b.close_move >= thr).astype(int), (c.close_move >= thr).astype(int))
                io, co = fit_clf(b, c, FEATS, (b.close_move < 0).astype(int), (c.close_move < 0).astype(int))
                ev.loc[m, "p_above"] = proba(ia, ca, ev[m], FEATS)
                ev.loc[m, "p_opp"] = proba(io, co, ev[m], FEATS)
        parts.append(ev)
    return pd.concat(parts, ignore_index=True).dropna(subset=["p_above"])


# ----------------------------------------------------------------------- selection
def select(dfg, N, signal, cal, n_pick=2):
    last, keep = {}, []
    for day, g in dfg.groupby("date", sort=True):
        i = cal[pd.Timestamp(day)]
        avail = g[g["symbol"].map(lambda s: (i - last.get(s, -10**9)) >= COOLDOWN)]
        poolN = avail.sort_values("confidence", ascending=False).head(N)
        ranked = poolN.sort_values(signal, ascending=False)
        seen = set()
        for idx, s in zip(ranked.index, ranked["symbol"]):
            if s in seen:
                continue
            keep.append(idx); seen.add(s)
            if len(seen) >= n_pick:
                break
        for s in seen:
            last[s] = i
    return dfg.loc[keep]


def outcomes(sel):
    s = sel.dropna(subset=["close_move"])
    if s.empty:
        return dict(trades=0, above=0, small=0, opp=0, top3=0, peak=0)
    return dict(
        trades=len(s),
        above=round((s.close_move >= s.threshold).mean() * 100, 1),
        small=round(((s.close_move >= 0) & (s.close_move < s.threshold)).mean() * 100, 1),
        opp=round((s.close_move < 0).mean() * 100, 1),
        top3=round((s.mover_rank <= 3).mean() * 100, 1),
        peak=round((s.ceiling >= s.threshold).mean() * 100, 1),
    )


def select_all(oos, N, signal, cal):
    return pd.concat([select(oos[oos.group.eq(b)], N, signal, cal) for b in THR], ignore_index=True)


def main():
    df = P.load_dataset(PROD)
    df, cal = add_close_move(df)
    df = df.dropna(subset=["close_move"])
    print(f"dataset rows={len(df)} | years={sorted(df.year.unique())}")

    oos = walk_forward_close(df)

    # attach PROD confidence (Stage-1 gate) + thresholds
    prod = pd.concat([pd.read_csv(PROD_PRED / f"group_{b}_predictions.csv", parse_dates=["date"]) for b in THR],
                     ignore_index=True)
    prod["symbol"] = prod["symbol"].astype(str)
    oos = oos.merge(prod[["date", "symbol", "side", "confidence"]], on=["date", "symbol", "side"], how="inner")
    oos["threshold"] = oos["group"].map(THR)

    mag = oos.groupby(["date", "group", "symbol"], as_index=False)["ceiling"].max().rename(columns={"ceiling": "smag"})
    mag["mover_rank"] = mag.groupby(["date", "group"])["smag"].rank(ascending=False, method="first")
    oos = oos.merge(mag[["date", "group", "symbol", "mover_rank"]], on=["date", "group", "symbol"], how="left")
    oos.to_parquet(OUT)
    print(f"OOS rows={len(oos)} saved -> {OUT.name}\n")

    # --- is it learnable? OOS AUC ---------------------------------------------------
    print("=" * 80)
    print("STAGE-2 OOS AUC  (0.5 = no signal)")
    print("=" * 80)
    for group, thr in PROD.group_thresholds:
        d = oos[oos.group == group]
        a = roc_auc_score((d.close_move >= thr).astype(int), d.p_above)
        o = roc_auc_score((d.close_move < 0).astype(int), d.p_opp)
        print(f"  {group}: AUC p_above={a:.3f}  p_opp={o:.3f}  (base rate above={ (d.close_move>=thr).mean()*100:.0f}%)")

    # --- selection comparison -------------------------------------------------------
    oos["oracle"] = oos["close_move"]
    rows = []
    rows.append({"pick_by": "confidence (PROD)", "N": 7, **outcomes(select_all(oos, 7, "confidence", cal))})
    for N in (5, 7, 10):
        for lam in (0.0, 0.5, 1.0):
            oos["score"] = oos["p_above"] - lam * oos["p_opp"]
            rows.append({"pick_by": f"stage2 lam={lam}", "N": N, **outcomes(select_all(oos, N, "score", cal))})
    rows.append({"pick_by": "ORACLE", "N": 7, **outcomes(select_all(oos, 7, "oracle", cal))})
    res = pd.DataFrame(rows)[["pick_by", "N", "trades", "above", "small", "opp", "top3", "peak"]]
    print("\n" + "=" * 80)
    print("SELECTION (combined groups): baseline vs stage2(lambda,N) vs ORACLE")
    print("=" * 80)
    print(res.to_string(index=False))

    # --- per-group / per-year for the best-looking stage2 config --------------------
    best = res[res.pick_by.str.startswith("stage2")].sort_values(["above", "opp"], ascending=[False, True]).iloc[0]
    N, lam = int(best["N"]), float(best["pick_by"].split("=")[1])
    oos["score"] = oos["p_above"] - lam * oos["p_opp"]
    print("\n" + "=" * 80)
    print(f"BEST stage2 config: N={N}, lambda={lam}  — per group/year vs baseline")
    print("=" * 80)
    rows = []
    for b in THR:
        base_b = outcomes(select(oos[oos.group.eq(b)], 2, "confidence", cal)); base_b["scope"] = f"{b} BASE"; rows.append(base_b)
        s2 = select(oos[oos.group.eq(b)], N, "score", cal)
        s2o = outcomes(s2); s2o["scope"] = f"{b} STG2"; rows.append(s2o)
        s2 = s2.assign(yr=s2.date.dt.year)
        for yr, d in s2.groupby("yr"):
            oo = outcomes(d); oo["scope"] = f"  {b} {yr}"; rows.append(oo)
    print(pd.DataFrame(rows)[["scope", "trades", "above", "small", "opp", "top3", "peak"]].to_string(index=False))


if __name__ == "__main__":
    main()
