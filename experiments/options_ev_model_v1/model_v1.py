"""Options-gain ML model v1 — select a direction-agnostic daily options book maximizing real-premium EV.

OBJECTIVE: rank (stock, day) by predicted straddle/strangle premium return; pick top-K/group/day; the book's
realized held-5d premium EV (net ~3% cost) must beat (a) atm_iv-rank, (b) random, (c) cheap_convexity (+1.9%).
Direction is a coin flip -> structure is a STRADDLE (ATM call+put) or STRANGLE (ATM+3% call+put), bought at the
t+1 OPEN, held 5d. Target/eval = the realized premium-weighted held return (exact, from the 384k single-leg tape).
Features are leak-clean (lagged to the prior trading day = as-of the 7AM refresh). CatBoost, purged walk-forward.

    set PYTHONPATH=src && python experiments/options_ev_model_v1/model_v1.py
Read-only on data; writes only here; PROD untouched.
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

TAPE = ROOT / "experiments" / "option_gain_study_v1" / "results" / "option_gain_trades.csv"
EMBARGO = 5
COST = 0.03
QUARTERS = pd.period_range("2024Q1", "2026Q2", freq="Q")
TRADE_COLS = {"symbol", "group", "side", "strike_label", "otm_pct", "strike", "expiry", "entry_date", "U",
              "entry_open", "max_high", "last_close", "peak_day", "exit_close_u", "high_ratio", "close_ratio",
              "stock_move", "date", "open", "high", "low", "close", "volume", "atm_iv"}
LEAK = ("future", "fwd", "next", "label", "tomorrow", "ahead", "adverse", "move_5d", "move_1d", "move_3d",
        "move_10d", "expansion", "volclean", "outcome", "_ratio")
CB = dict(iterations=500, depth=6, learning_rate=0.03, l2_leaf_reg=6.0, random_seed=7,
          allow_writing_files=False, verbose=False)


def build_structures() -> pd.DataFrame:
    """From the single-leg tape build per (symbol, day) straddle (ATM) & strangle (ATM+3%) held-5d returns."""
    t = pd.read_csv(TAPE, parse_dates=["entry_date"])
    t["last_close"] = t.close_ratio * t.entry_open          # premium at day-5 close

    def leg(side, lab):
        x = t[(t.side == side) & (t.strike_label == lab)]
        return x.groupby(["symbol", "entry_date", "group"]).agg(
            entry=("entry_open", "first"), exitp=("last_close", "first"),
            peak=("high_ratio", "first"), dte=("dte", "first"), atm_iv=("atm_iv", "first")).reset_index()

    def struct(call_lab, put_lab, name):
        c = leg("CALL", call_lab).rename(columns={"entry": "ce", "exitp": "cx", "peak": "cpk"})
        p = leg("PUT", put_lab).rename(columns={"entry": "pe", "exitp": "px", "peak": "ppk"})
        m = c.merge(p[["symbol", "entry_date", "pe", "px", "ppk"]], on=["symbol", "entry_date"])
        m["entry_prem"] = m.ce + m.pe
        m["held_ret"] = (m.cx + m.px) / m.entry_prem - 1.0                    # EXACT premium-weighted held return
        m["peak_approx"] = (m.cpk * m.ce + m.ppk * m.pe) / m.entry_prem - 1.0  # optimistic (legs peak apart)
        m["structure"] = name
        return m

    return pd.concat([struct("ATM", "ATM", "straddle"), struct("ATM+3%", "ATM+3%", "strangle")], ignore_index=True)


def add_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    f = load_market_data()
    f["symbol"] = f["symbol"].astype(str)
    f["date"] = pd.to_datetime(f["date"])
    f = f.sort_values(["symbol", "date"])
    sfeats = [c for c in f.columns if c not in TRADE_COLS and pd.api.types.is_numeric_dtype(f[c])
              and not any(h in c.lower() for h in LEAK)]
    f2 = f[["symbol", "date"] + sfeats].copy()
    f2["apply_date"] = f2.groupby("symbol")["date"].shift(-1)        # prior-day features apply to today's open
    f2 = f2.dropna(subset=["apply_date"]).drop(columns=["date"])
    m = df.merge(f2, left_on=["entry_date", "symbol"], right_on=["apply_date", "symbol"], how="left")
    # engineered (cross-sectional cheapness/vol ranks within day, realized-vs-implied)
    if "realized_vol_20" in m:
        m["rv_over_iv"] = m.realized_vol_20 / m.atm_iv.replace(0, np.nan)
    m["iv_rank_day"] = m.groupby("entry_date").atm_iv.rank(pct=True)
    if "realized_vol_20" in m:
        m["rv_rank_day"] = m.groupby("entry_date").realized_vol_20.rank(pct=True)
    m["prem_pct_u"] = m.entry_prem / m.groupby(["entry_date", "symbol"]).entry_prem.transform("first")  # =1, placeholder
    opt = ["dte", "atm_iv", "entry_prem", "rv_over_iv", "iv_rank_day", "rv_rank_day"]
    feats = [c for c in opt + sfeats if c in m.columns and m[c].notna().any()]
    return m, feats


def topk_ev(d: pd.DataFrame, score: str, k: int, per_group: bool) -> dict:
    keys = ["entry_date", "group"] if per_group else ["entry_date"]
    picks = d.sort_values(score, ascending=False).groupby(keys, sort=False).head(k)
    r = picks.held_ret.to_numpy() - COST
    return {"net_ev_%": round(r.mean() * 100, 2), "win": round((r > 0).mean(), 3),
            "gross_%": round(picks.held_ret.mean() * 100, 2), "n": len(picks),
            "trades_yr": round(len(picks) / 2.45)}


def walkforward(d, feats, target, classify=False):
    from catboost import CatBoostClassifier, CatBoostRegressor
    out = []
    for q in QUARTERS:
        cut = q.start_time - pd.Timedelta(days=5 + EMBARGO)
        tr = d[d.entry_date < cut]
        te = d[(d.entry_date >= q.start_time) & (d.entry_date <= q.end_time)].copy()
        if len(tr) < 3000 or te.empty:
            continue
        if classify:
            mdl = CatBoostClassifier(**CB).fit(tr[feats], (tr[target] >= 0.5).astype(int))
            te["score"] = mdl.predict_proba(te[feats])[:, 1]
        else:
            mdl = CatBoostRegressor(**CB, loss_function="RMSE").fit(tr[feats], tr[target].clip(-1, 6))
            te["score"] = mdl.predict(te[feats])
        out.append(te)
    return pd.concat(out, ignore_index=True)


def main():
    base = build_structures()
    print(f"structures built: {len(base):,} rows  | straddle {sum(base.structure=='straddle'):,} strangle {sum(base.structure=='strangle'):,}")
    for s in ("straddle", "strangle"):
        u = base[base.structure == s]
        print(f"  {s}: universe held EV (gross) {u.held_ret.mean()*100:+.2f}%  median {u.held_ret.median()*100:+.2f}%  win {(u.held_ret>0).mean():.3f}")

    results = []
    for s in ("straddle", "strangle"):
        d0 = base[base.structure == s].copy()
        d, feats = add_features(d0)
        d = d[d.held_ret.notna()]
        print(f"\n===== {s} | rows {len(d):,} | feats {len(feats)} =====")
        # baselines
        for k in (2, 3):
            b_iv = topk_ev(d, "atm_iv", k, True)
            print(f"  [baseline atm_iv-rank top{k}/grp] net {b_iv['net_ev_%']}%  win {b_iv['win']}  ({b_iv['trades_yr']}/yr)")
        # model: regressor + classifier
        for clf, tag in ((False, "REG"), (True, "CLF")):
            ev = walkforward(d, feats, "held_ret", classify=clf)
            for k in (2, 3):
                m = topk_ev(ev, "score", k, True)
                print(f"  [{tag} top{k}/grp] net {m['net_ev_%']}%  gross {m['gross_%']}%  win {m['win']}  ({m['trades_yr']}/yr)")
                results.append((s, tag, k, m["net_ev_%"], m["win"], m["trades_yr"]))
            # conviction curve (top X% of days by score, pooled)
            ev2 = ev.sort_values("score", ascending=False)
            for frac in (0.05, 0.10, 0.20):
                kk = max(1, int(len(ev2) * frac)); top = ev2.head(kk)
                r = top.held_ret.to_numpy() - COST
                print(f"     {tag} top {int(frac*100)}% by score: net {r.mean()*100:+.2f}%  win {(r>0).mean():.3f}  n {kk}")
    print("\n=== summary (net EV %, higher=better) ===")
    for r in sorted(results, key=lambda x: -x[3]):
        print(f"  {r[0]:9s} {r[1]} top{r[2]}/grp: net {r[3]:+.2f}%  win {r[4]}  {r[5]}/yr")


if __name__ == "__main__":
    main()
