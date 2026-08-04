"""cheap_convexity_v1 — predict which options out-move what `atm_iv` priced in (the VRP residual).

Target  surprise = |close[t+W]/close[t]-1|  -  atm_iv*sqrt(W/252)      ( >0 => moved MORE than priced )
Model   CatBoost regressor on IV-structure / skew / PCR / flow / catalyst features (orthogonal to atm_iv LEVEL).
Eval    purged+embargoed quarterly walk-forward; rank top-k/group by predicted surprise; compare realized
        surprise + premium-adjusted straddle-PnL proxy vs the atm_iv baseline (which buys the most expensive).

    set PYTHONPATH=src && python experiments/cheap_convexity_v1/run_experiment.py

Reads features READ-ONLY; writes only into this folder; never imports koscine3.largemove.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
from koscine3.data.sources import load_market_data  # noqa: E402

RESULTS = HERE / "results"
LOCK_V2 = ROOT / "locks" / "prod_largemove_v2"

WINDOW = 5
ANN = 252.0
STRADDLE_K = 0.8                      # ATM straddle breakeven ~ 0.8 * implied_move (held to horizon)
EMBARGO_DAYS = 5
MIN_UNDERLYING = 100.0
QUARTERS = pd.period_range("2024Q1", "2026Q2", freq="Q")
TOPK = (3, 5)

# leak-safe, as-of-close[t] candidate features (orthogonal-ish to the atm_iv LEVEL)
CAND_NUM = [
    "atm_iv", "atm_ce_iv", "atm_pe_iv", "atm_iv_chg_5", "atm_iv_ratio_20",
    "put_call_iv_skew", "iv_skew_ce_minus_pe", "iv_skew_norm", "iv_skew_chg_5d",
    "pcr_oi", "pcr_vol", "pcr_oi_chg_5", "pcr_oi_ratio_20", "pcr_vol_chg_5", "pcr_vol_ratio_20",
    "oi_buildup_ratio", "fut_chg_oi", "price_oi_divergence",
    "realized_vol_20", "atr_pct_14", "atr_pct_14_rank_60d", "vol_sma20_ratio", "donchian_width_20",
    "sector_vol_20", "nifty_realized_vol_20",
    "delivery_pct", "delivery_pct_chg_5", "delivery_qty_ratio_20", "turnover_ratio_20",
    "days_to_earnings", "earnings_within_5d", "gap_pct", "ret_1d", "ret_5d", "dist_52wh",
]
CAND_CAT = ["sector", "group"]
LEAK_HINTS = ("future", "fwd", "next", "label", "tomorrow", "ahead", "adverse",
              "move_5d", "move_1d", "move_3d", "move_10d", "expansion", "volclean", "outcome", "surprise")
CB = dict(iterations=600, depth=6, learning_rate=0.03, loss_function="RMSE", l2_leaf_reg=6.0,
          subsample=0.8, random_seed=7, allow_writing_files=False, verbose=False)


def universe() -> dict[str, str]:
    g = json.loads((LOCK_V2 / "universe_groups.json").read_text())
    return {s: grp for grp, syms in g.items() for s in syms}


def load_panel():
    g2 = universe()
    m = load_market_data()
    m["symbol"] = m["symbol"].astype(str)
    m = m[m.symbol.isin(g2)].copy()
    m["date"] = pd.to_datetime(m["date"])
    m["group"] = m["symbol"].map(g2)
    m = m.sort_values(["symbol", "date"]).reset_index(drop=True)

    if m["atm_iv"].median() > 3:                       # safety: ensure atm_iv is a decimal (0.30), not 30
        m["atm_iv"] = m["atm_iv"] / 100.0

    g = m.groupby("symbol", sort=False)
    fwd_close = g["close"].shift(-WINDOW)
    m["realized_cc"] = (fwd_close / m["close"] - 1).abs()
    m["implied_move"] = m["atm_iv"] * np.sqrt(WINDOW / ANN)
    m["surprise"] = m["realized_cc"] - m["implied_move"]                 # target
    m["straddle_pnl"] = m["realized_cc"] - STRADDLE_K * m["implied_move"]  # premium-adjusted profit proxy
    m["eligible"] = m["close"].ge(MIN_UNDERLYING) & m["atm_iv"].notna()

    if "realized_vol_20" in m and "atm_iv" in m:
        m["rv_over_iv"] = m["realized_vol_20"] / m["atm_iv"].replace(0, np.nan)   # recent realized vs implied
    num = [c for c in CAND_NUM if c in m.columns and m[c].notna().any()]
    if "rv_over_iv" in m.columns:
        num.append("rv_over_iv")
    cat = [c for c in CAND_CAT if c in m.columns]
    num = [c for c in num if not any(h in c.lower() for h in LEAK_HINTS)]   # safety net vs forward labels
    for c in num:
        m[c] = m[c].replace([np.inf, -np.inf], np.nan)
    for c in cat:
        m[c] = m[c].astype(str).fillna("NA")
    return m, num, cat


def selector_stats(ev: pd.DataFrame, score: str, k: int) -> dict:
    asc = False
    picks = ev.sort_values(score, ascending=asc).groupby(["date", "group"], sort=False).head(k)
    return {"mean_surprise_bps": round(float(picks.surprise.mean()) * 1e4, 1),
            "mean_straddle_pnl_bps": round(float(picks.straddle_pnl.mean()) * 1e4, 1),
            "hit_rate_realized_gt_implied": round(float((picks.realized_cc > picks.implied_move).mean()), 3),
            "mean_realized_move_pct": round(float(picks.realized_cc.mean()) * 100, 2),
            "n": int(len(picks))}


def main():
    from catboost import CatBoostRegressor
    RESULTS.mkdir(exist_ok=True)
    m, num, cat = load_panel()
    feats = num + cat
    print(f"panel rows={len(m):,}  symbols={m.symbol.nunique()}  features={len(feats)} ({len(cat)} categorical)")
    print(f"sanity: median atm_iv={m.atm_iv.median():.3f}  implied_move(5d)={m.implied_move.median()*100:.2f}%  "
          f"realized_cc={m.realized_cc.median()*100:.2f}%  base surprise(mean)={m.surprise.mean()*1e4:.0f}bps "
          f"(negative = options expensive on avg / positive VRP)")

    rows, imp_sum, evs, nq = [], {}, [], 0
    for q in QUARTERS:
        cut = q.start_time - pd.Timedelta(days=WINDOW + EMBARGO_DAYS)          # purge horizon + embargo
        tr = m[(m.date < cut) & m.eligible & m.surprise.notna()]
        ev = m[(m.date >= q.start_time) & (m.date <= q.end_time) & m.eligible & m.surprise.notna()].copy()
        if len(tr) < 3000 or ev.empty:
            continue
        model = CatBoostRegressor(**CB)
        model.fit(tr[feats], tr["surprise"], cat_features=cat)
        ev["pred"] = model.predict(ev[feats])
        evs.append(ev)
        nq += 1
        for f, g in zip(feats, model.get_feature_importance()):
            imp_sum[f] = imp_sum.get(f, 0.0) + float(g)

    ev = pd.concat(evs, ignore_index=True)
    # rank-IC: cross-sectional spearman per day, predicted-surprise vs realized-surprise (and atm_iv for reference)
    def daily_ic(score):
        return ev.groupby("date")[[score, "surprise"]].apply(
            lambda d: d.corr("spearman").iloc[0, 1] if len(d) > 5 else np.nan).mean()
    ic_pred = float(daily_ic("pred"))
    ic_iv = float(daily_ic("atm_iv"))
    print(f"\nrank-IC(predicted surprise, realized surprise) = {ic_pred:+.4f}")
    print(f"rank-IC(atm_iv, realized surprise)             = {ic_iv:+.4f}  (expect <=0: high IV => priced => smaller surprise)")

    out = {"config": {"window": WINDOW, "embargo": EMBARGO_DAYS, "straddle_k": STRADDLE_K,
                      "quarters": [str(q) for q in QUARTERS], "n_features": len(feats), "n_folds": nq},
           "rank_ic_pred_surprise": round(ic_pred, 4), "rank_ic_atm_iv_surprise": round(ic_iv, 4),
           "selectors": {}}
    print(f"\n{'selector':16s} {'k':>2s} {'straddle_pnl(bps)':>18s} {'surprise(bps)':>14s} {'hit P(real>impl)':>17s} {'realized move%':>15s}")
    for k in TOPK:
        for name, score in [("cheap_convexity", "pred"), ("atm_iv_baseline", "atm_iv")]:
            s = selector_stats(ev, score, k)
            out["selectors"][f"{name}_top{k}"] = s
            print(f"{name:16s} {k:>2d} {s['mean_straddle_pnl_bps']:>18.1f} {s['mean_surprise_bps']:>14.1f} "
                  f"{s['hit_rate_realized_gt_implied']:>17.3f} {s['mean_realized_move_pct']:>15.2f}")
        # universe reference (all eligible, equal weight)
        ref = {"mean_straddle_pnl_bps": round(float(ev.straddle_pnl.mean()) * 1e4, 1),
               "mean_surprise_bps": round(float(ev.surprise.mean()) * 1e4, 1)}
        out["selectors"][f"universe_all"] = ref

    # per-group + per-quarter straddle-PnL of the model book (top-3)
    bygrp = {}
    for grp, d in ev.groupby("group"):
        picks = d.sort_values("pred", ascending=False).groupby("date").head(3)
        bygrp[grp] = {"straddle_pnl_bps": round(float(picks.straddle_pnl.mean()) * 1e4, 1),
                      "surprise_bps": round(float(picks.surprise.mean()) * 1e4, 1),
                      "realized_move_pct": round(float(picks.realized_cc.mean()) * 100, 2)}
    out["by_group_model_top3"] = bygrp
    print("\nmodel top-3 by group:", json.dumps(bygrp))

    # feature importance
    imp = pd.Series({k: v / max(nq, 1) for k, v in imp_sum.items()}).sort_values(ascending=False)
    imp.to_csv(RESULTS / "feature_importance.csv", header=["importance"])
    print("\ntop predictors of the surprise:")
    for f, v in imp.head(14).items():
        print(f"    {f:22s} {v:6.2f}")

    # picks book for premium_ev.py (model + baseline, top-3/group/day)
    pk = []
    for name, score in [("cheap_convexity", "pred"), ("atm_iv_baseline", "atm_iv")]:
        p = ev.sort_values(score, ascending=False).groupby(["date", "group"], sort=False).head(3).copy()
        p["selector"] = name
        pk.append(p[["date", "symbol", "group", "selector", "atm_iv", "pred", "implied_move",
                     "realized_cc", "surprise", "straddle_pnl"]] if "pred" in p else p)
    pd.concat(pk, ignore_index=True).to_csv(RESULTS / "picks.csv", index=False)

    (RESULTS / "metrics.json").write_text(json.dumps(out, indent=2, default=float))
    print(f"\nsaved -> {RESULTS/'metrics.json'} , feature_importance.csv , picks.csv")
    print("VERDICT: model straddle_pnl > 0 AND > atm_iv (stable, + confirmed on real premiums) => cheap-convexity edge.")
    print("         model straddle_pnl ~ atm_iv (both <= 0) => surprise not predictable; VRP dominates; atm_iv rule stands.")


if __name__ == "__main__":
    main()
