"""Real option P&L backtest for the large-move options strategy.

For each daily pick (top-1 long call + top-1 short put), buy the nearest strike to 3% OTM
of the monthly contract, using ACTUAL bhavcopy option prices:
  entry premium = option OPEN on t+1 (fallback close/settle)
  exits         = best window HIGH (sell at peak), close on underlying-peak day, close on t+5
Reports realized option multiples + EV. Shows confidence P(big) and quantum (pred move) per pick.
Read-only research.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from koscine3.data.feature_registry import build_feature_registry  # noqa: E402
from koscine3.data.sources import load_market_data  # noqa: E402
from koscine3.data.universe import UniverseConfig, build_universe  # noqa: E402
from koscine3.datasets.supervised_builder import build_supervised_dataset, model_feature_columns  # noqa: E402
from koscine3.outcomes.clean_move_contract import CleanMoveContract, compute_clean_move_outcomes  # noqa: E402
from analysis.options_bhavcopy import load_bhavcopy  # noqa: E402

from lightgbm import LGBMClassifier, LGBMRegressor  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402

TRAIN_END = pd.Timestamp("2023-12-31")
BIG_TRAIN = 0.10
OTM = 0.03  # buy strike ~3% away from entry open


def _clean(frame, feats):
    return frame[feats].replace([np.inf, -np.inf], np.nan)


def _entry_premium(row) -> float:
    for c in ("open", "close", "settle"):
        v = row[c]
        if pd.notna(v) and v > 0:
            return float(v)
    return np.nan


def main() -> None:
    print("loading equity + features ...")
    market = load_market_data()
    registry = build_feature_registry(market)
    universe = build_universe(market, UniverseConfig(cutoff_date="2025-12-31", top_n=100))
    syms = set(universe["symbol"].astype(str))
    dataset = build_supervised_dataset(market, universe, registry)
    feats = model_feature_columns(registry, dataset)
    dataset["symbol"] = dataset["symbol"].astype(str)

    oc = compute_clean_move_outcomes(market, universe=universe, contract=CleanMoveContract())
    oc = oc[oc["status"].eq("evaluated")][
        ["date", "symbol", "side", "ceiling", "days_to_peak", "entry_date", "entry_open", "window_end_date"]
    ].copy()
    oc["symbol"] = oc["symbol"].astype(str)
    df = dataset.merge(oc, on=["date", "symbol", "side"], how="inner",
                       suffixes=("", "_oc"))
    # prefer the clean-move contract's entry fields
    for c in ("entry_date", "entry_open", "window_end_date"):
        if f"{c}_oc" in df.columns:
            df[c] = df[f"{c}_oc"]
    train, evl = df[df["date"] <= TRAIN_END], df[df["date"] > TRAIN_END]

    print("fitting large-move model + generating picks ...")
    picks = []
    for side in ("long", "short"):
        tr_s, ev_s = train[train["side"].eq(side)], evl[evl["side"].eq(side)].copy()
        imp = SimpleImputer(strategy="median").fit(_clean(tr_s, feats))
        Xtr, Xev = imp.transform(_clean(tr_s, feats)), imp.transform(_clean(ev_s, feats))
        clf = LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, subsample=0.85,
                             colsample_bytree=0.85, class_weight="balanced", random_state=17,
                             verbosity=-1).fit(Xtr, (tr_s["ceiling"] >= BIG_TRAIN).astype(int))
        reg = LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31, subsample=0.85,
                            colsample_bytree=0.85, random_state=17, verbosity=-1).fit(Xtr, tr_s["ceiling"])
        ev_s["confidence"] = clf.predict_proba(Xev)[:, 1]   # P(big)
        ev_s["pred_move"] = np.maximum(reg.predict(Xev), 0)  # quantum
        picks.append(ev_s.sort_values("confidence", ascending=False).groupby("date").head(1))
    picks = pd.concat(picks, ignore_index=True)
    picks["opt_type"] = np.where(picks["side"].eq("long"), "CE", "PE")
    print(f"picks: {len(picks)} ({picks['side'].value_counts().to_dict()})")

    # Trading calendar + window dates per pick
    cal = np.array(sorted(market["date"].unique()))
    pos = {d: i for i, d in enumerate(cal)}

    def window_dates(entry_date):
        i = pos.get(pd.Timestamp(entry_date))
        if i is None:
            return []
        return [pd.Timestamp(cal[j]) for j in range(i, min(i + 5, len(cal)))]

    # Load option data for all needed dates once (universe STKOPT, near-money strikes).
    close_map = market.set_index(["date", "symbol"])["close"]
    needed = sorted({d for _, r in picks.iterrows() for d in window_dates(r["entry_date"])})
    print(f"loading option bhavcopy for {len(needed)} trading days ...")
    opt = {}
    for k, d in enumerate(needed):
        bc = load_bhavcopy(d)
        if bc.empty:
            continue
        bc = bc[bc["symbol"].isin(syms) & bc["opt_type"].isin(["CE", "PE"])].copy()
        und = bc["underlying"]
        miss = und.isna()
        if miss.any():
            bc.loc[miss, "underlying"] = bc.loc[miss].apply(
                lambda r: close_map.get((pd.Timestamp(d), r["symbol"]), np.nan), axis=1)
        bc = bc[(bc["strike"] >= bc["underlying"] * 0.85) & (bc["strike"] <= bc["underlying"] * 1.15)]
        opt[d] = bc
        if (k + 1) % 100 == 0:
            print(f"  {k+1}/{len(needed)}")

    def chain(d, symbol, opt_type):
        b = opt.get(pd.Timestamp(d))
        if b is None:
            return None
        return b[b["symbol"].eq(symbol) & b["opt_type"].eq(opt_type)]

    rows = []
    for _, p in picks.iterrows():
        wd = window_dates(p["entry_date"])
        if len(wd) < 5:
            continue
        entry_date, wend = wd[0], pd.Timestamp(p["window_end_date"])
        ch = chain(entry_date, p["symbol"], p["opt_type"])
        if ch is None or ch.empty:
            continue
        exps = sorted(e for e in ch["expiry"].dropna().unique() if pd.Timestamp(e) >= wend)
        expiry = pd.Timestamp(exps[0]) if exps else pd.Timestamp(sorted(ch["expiry"].dropna().unique())[-1])
        ce = ch[ch["expiry"].eq(expiry)]
        if ce.empty:
            continue
        target = p["entry_open"] * (1 + OTM if p["opt_type"] == "CE" else 1 - OTM)
        krow = ce.iloc[(ce["strike"] - target).abs().argsort().iloc[0]]
        strike = krow["strike"]
        entry_prem = _entry_premium(krow)
        if not (entry_prem and entry_prem > 0):
            continue
        highs, t5_close, peak_close = [], np.nan, np.nan
        dtp = int(p["days_to_peak"]) if pd.notna(p["days_to_peak"]) else 5
        for j, d in enumerate(wd, start=1):
            cc = chain(d, p["symbol"], p["opt_type"])
            if cc is None:
                continue
            cc = cc[cc["expiry"].eq(expiry) & cc["strike"].eq(strike)]
            if cc.empty:
                continue
            r = cc.iloc[0]
            hi = r["high"] if pd.notna(r["high"]) and r["high"] > 0 else r["close"]
            if pd.notna(hi):
                highs.append(float(hi))
            if j == 5:
                t5_close = float(r["close"]) if pd.notna(r["close"]) else np.nan
            if j == dtp:
                peak_close = float(r["close"]) if pd.notna(r["close"]) else np.nan
        if not highs:
            continue
        best = max(highs)
        rows.append({
            "date": p["date"], "symbol": p["symbol"], "side": p["side"],
            "confidence": round(float(p["confidence"]), 3), "pred_move_%": round(float(p["pred_move"]) * 100, 2),
            "actual_move_%": round(float(p["ceiling"]) * 100, 2), "strike": strike,
            "entry_prem": round(entry_prem, 2), "best_prem": round(best, 2),
            "mult_best": round(best / entry_prem, 2),
            "mult_peakclose": round(peak_close / entry_prem, 2) if pd.notna(peak_close) else np.nan,
            "mult_t5close": round(t5_close / entry_prem, 2) if pd.notna(t5_close) else np.nan,
            "entry_liq_vol": int(krow["vol"]) if pd.notna(krow["vol"]) else 0,
            "year": pd.Timestamp(p["date"]).year,
        })
    res = pd.DataFrame(rows)
    res.to_csv(ROOT / "reports" / "option_payoff_trades_2024_2026.csv", index=False)
    print(f"\nmatched option trades: {len(res)} / {len(picks)} picks")

    liq = res[res["entry_liq_vol"] > 0]
    print(f"with traded entry (vol>0): {len(liq)} ({len(liq)/len(res)*100:.0f}%)")
    for label, d in [("ALL matched", res), ("LIQUID entry (vol>0)", liq)]:
        print(f"\n===== {label}  (n={len(d)}) =====")
        for col in ["mult_best", "mult_peakclose", "mult_t5close"]:
            s = d[col].dropna()
            print(f"  {col:15s} mean={s.mean():.2f}x median={s.median():.2f}x "
                  f"P>=2x={ (s>=2).mean()*100:4.1f}% P>=3x={(s>=3).mean()*100:4.1f}% "
                  f"P>=5x={(s>=5).mean()*100:4.1f}%  EV/trade(best-case sell)={ (s.clip(0)-1).mean()*100:+.0f}%")
    print("\n===== mult_best by year (sell-at-peak) =====")
    print(liq.groupby("year")["mult_best"].agg(["size", "mean", "median",
          lambda s: round((s >= 3).mean()*100, 1)]).rename(columns={"<lambda_0>": "P>=3x_%"}).round(2).to_string())
    print("\nsample trades (confidence = P(big), quantum = pred_move):")
    print(res.sort_values("confidence", ascending=False).head(12).to_string(index=False))


if __name__ == "__main__":
    main()
