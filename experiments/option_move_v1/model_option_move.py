"""option_move_v1 — model which OPTION CONTRACTS give big premium moves (vega + gamma, not just the underlying).

Contract dataset (option_contracts.csv): ATM+/-5% ladder (CE & PE), entry close[d], 5-day fwd peak/held + per-strike
OI/volume. Here we add per-strike BS implied vol, engineer IV/flow/structure features + merge stock features, then:
  1) VEGA-PRIZE diagnostic: do cheaper-IV / IV-with-room contracts gain MORE at peak?
  2) model comparison: STOCK-ONLY vs STOCK+OPTION-microstructure (does the contract view add?).
  3) grouped feature importance (stock-move vs IV/vega vs structure/flow).
  4) selection: top-k contracts/day (best-side per strike, direction picked offline) -> real peak gain vs baselines.

    set PYTHONPATH=src && python experiments/option_move_v1/model_option_move.py
Read-only; PROD untouched.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
from koscine3.data.sources import load_market_data  # noqa: E402

R = 0.065
EMBARGO = 5
QUARTERS = pd.period_range("2024Q1", "2026Q2", freq="Q")
LEAK = ("future", "fwd", "next", "label", "tomorrow", "ahead", "adverse", "move_5d", "move_1d", "move_3d",
        "move_10d", "expansion", "volclean", "outcome")
DROP = {"date", "symbol", "open", "high", "low", "close", "volume", "group", "expiry"}
CB = dict(iterations=400, depth=6, learning_rate=0.03, l2_leaf_reg=6.0, random_seed=7,
          allow_writing_files=False, verbose=False, thread_count=-1)   # CPU (avoid GPU context conflict)


def bs_iv(price, S, K, T, is_call, init):
    from scipy.stats import norm
    sig = np.clip(np.where(np.isfinite(init), init, 0.5), 0.05, 3.0).astype(float)
    T = np.maximum(T, 1 / 365)
    for _ in range(12):
        sq = sig * np.sqrt(T)
        d1 = (np.log(S / K) + (R + sig ** 2 / 2) * T) / sq
        d2 = d1 - sq
        disc = np.exp(-R * T)
        model = np.where(is_call, S * norm.cdf(d1) - K * disc * norm.cdf(d2),
                         K * disc * norm.cdf(-d2) - S * norm.cdf(-d1))
        vega = S * norm.pdf(d1) * np.sqrt(T)
        sig = np.clip(sig - (model - price) / np.maximum(vega, 1e-6), 0.01, 5.0)
    return sig


def load():
    c = pd.read_csv(HERE / "results" / "option_contracts.csv", parse_dates=["date"])
    c["is_call"] = (c.ot == "CE").astype(int)
    c["iv"] = bs_iv(c.entry.values, c.U.values, c.strike.values, c.dte.values / 365.0, c.is_call.values, c.atm_iv.values)
    c["iv_dev"] = c.iv - c.atm_iv                          # per-strike smile/skew vs ATM
    c["abs_mny"] = c.moneyness.abs()
    c["prem_pct"] = c.entry / c.U * 100                    # premium as % of spot (cheapness/leverage)
    c["log_oi"] = np.log1p(c.oi); c["log_vol"] = np.log1p(c.vol)
    c["oi_rank"] = c.groupby("date").oi.rank(pct=True)
    c["vol_rank"] = c.groupby("date").vol.rank(pct=True)
    c["inv_sqrt_dte"] = 1 / np.sqrt(c.dte.clip(1))         # gamma proxy (near-expiry)
    OPT = ["iv", "iv_dev", "moneyness", "abs_mny", "is_call", "dte", "inv_sqrt_dte", "prem_pct",
           "log_oi", "log_vol", "oi_rank", "vol_rank"]
    # stock features (as-of close[d]; exclude forward labels)
    f = load_market_data()
    f["symbol"] = f["symbol"].astype(str); f["date"] = pd.to_datetime(f["date"])
    stock = [col for col in f.columns if col not in DROP and pd.api.types.is_numeric_dtype(f[col])
             and not any(h in col.lower() for h in LEAK) and col not in c.columns]   # avoid atm_iv (etc.) merge collision
    m = c.merge(f[["date", "symbol"] + stock], on=["date", "symbol"], how="left")
    m["win2"] = (m.peak_ratio >= 2).astype(int)
    return m, OPT, stock


def wf(m, feats):
    from catboost import CatBoostClassifier
    from sklearn.metrics import roc_auc_score
    ev, imp, n = [], {}, 0
    for q in QUARTERS:
        cut = q.start_time - pd.Timedelta(days=5 + EMBARGO)
        tr = m[m.date < cut]
        te = m[(m.date >= q.start_time) & (m.date <= q.end_time)].copy()
        if len(tr) < 8000 or te.empty:
            continue
        clf = CatBoostClassifier(**CB, loss_function="Logloss").fit(tr[feats], tr.win2)
        te["score"] = clf.predict_proba(te[feats])[:, 1]
        ev.append(te); n += 1
        for f_, g in zip(feats, clf.get_feature_importance()):
            imp[f_] = imp.get(f_, 0.0) + float(g)
    return pd.concat(ev, ignore_index=True), {k: v / n for k, v in imp.items()}


def main():
    m, OPT, stock = load()
    print(f"contracts {len(m):,} | symbols {m.symbol.nunique()} | P(peak>=2x) {m.win2.mean():.3f} | "
          f"median iv {m.iv.median():.3f} (atm {m.atm_iv.median():.3f})")

    # 1) VEGA PRIZE — does cheap IV / IV-with-room gain more at peak?
    print("\n=== 1) vega-prize: contract peak gain by entry IV-percentile (within day) ===")
    m["iv_pct_day"] = m.groupby("date").iv.rank(pct=True)
    for lo, hi, lab in [(0, .2, "cheapest IV"), (.2, .4, ""), (.4, .6, "mid"), (.6, .8, ""), (.8, 1.01, "priciest IV")]:
        b = m[(m.iv_pct_day >= lo) & (m.iv_pct_day < hi)]
        print(f"   IV pctile {lo:.1f}-{hi:.1f} {lab:12s}: mean peak {b.peak_ratio.mean():.2f}x  P>=2x {(b.peak_ratio>=2).mean():.3f}  median held {b.held_ratio.median():.2f}")
    if "atm_iv_ratio_20" in m:
        print(f"   corr(peak, atm_iv_ratio_20 [IV vs own 20d]) = {m.peak_ratio.corr(m.atm_iv_ratio_20):+.3f} (neg => cheap-vs-recent gains more)")

    # 2) model comparison: stock-only vs +option microstructure
    print("\n=== 2) model: predict big contract move (peak>=2x), purged WF ===")
    from sklearn.metrics import roc_auc_score
    ev_s, imp_s = wf(m, stock)
    ev_o, imp_o = wf(m, stock + OPT)
    print(f"   STOCK-only      OOS AUC {roc_auc_score(ev_s.win2, ev_s.score):.4f}")
    print(f"   STOCK+OPTION    OOS AUC {roc_auc_score(ev_o.win2, ev_o.score):.4f}")

    # 3) grouped importance (stock+option model)
    grp = {"IV/vega": ["iv", "iv_dev", "atm_iv", "atm_iv_ratio_20", "atm_iv_chg_5", "put_call_iv_skew", "iv_skew_norm"],
           "structure": ["moneyness", "abs_mny", "is_call", "dte", "inv_sqrt_dte", "prem_pct"],
           "flow": ["log_oi", "log_vol", "oi_rank", "vol_rank", "pcr_oi", "fut_chg_oi", "oi_buildup_ratio"]}
    tot = sum(imp_o.values())
    print("\n=== 3) grouped feature importance (STOCK+OPTION model) ===")
    used = set()
    for gname, cols in grp.items():
        s = sum(imp_o.get(c, 0) for c in cols); used |= set(cols)
        print(f"   {gname:10s} {s/tot*100:5.1f}%")
    print(f"   other-stock {sum(v for k,v in imp_o.items() if k not in used)/tot*100:5.1f}%")
    print("   top-12:", {k: round(v, 1) for k, v in sorted(imp_o.items(), key=lambda x: -x[1])[:12]})

    # 4) selection eval: best-side per (stock, strike), top-3/day -> real peak gain
    print("\n=== 4) selection: top-3 contracts/day by model vs atm_iv vs random (real peak gain) ===")
    ev_o["bestside_key"] = ev_o.symbol + "|" + ev_o.strike.astype(str) + "|" + ev_o.date.astype(str)
    def topk(df, score, k=3):
        d = df.sort_values(score, ascending=False).groupby("date").head(k)
        return d.peak_ratio.mean(), (d.peak_ratio >= 2).mean(), len(d) / d.date.nunique()
    np.random.seed(1); ev_o["rnd"] = np.random.rand(len(ev_o))
    for name, sc in [("model", "score"), ("atm_iv", "atm_iv"), ("random", "rnd")]:
        mp, p2, perday = topk(ev_o, sc)
        print(f"   {name:8s}: mean peak {mp:.2f}x  P>=2x {p2:.3f}  ({perday:.1f}/day)")
    print("\n(direction agnostic: user buys the side per offline view; peak = best forward exit over 5d)")


if __name__ == "__main__":
    main()
