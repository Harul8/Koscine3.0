"""DIRECTION OVERLAY for the v2 mover book — informational only, does NOT change v2 selection.

Learned from Koscine 2.0 clean_direction.py: condition on a move happening, then predict which way.
We train P(up | a big move happened) on the history of big movers (max(up,down) 5d >= 4%), with the
research direction features, walk-forward (quarterly). For each v2 pick we emit:
    p_up, dir_label (UP/DOWN), confidence = |2*p_up-1|, conf_tier (low/med/high)
Direction is ~a coin flip (research: ~0.52-0.55 AUC) — this captures only the small edge; grain of salt.

    python -m koscine3.largemove.direction_overlay      # writes locks/prod_largemove_v2/direction_overlay.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from koscine3.data.sources import load_market_data
from koscine3.largemove.mover_v2 import LOCK_V2
import json

ROOT = Path(__file__).resolve().parents[3]
BIG = 0.04                                   # "a move happened" gate for the direction training set
RAW = ["date", "symbol", "open", "high", "low", "close", "atm_iv", "atm_ce_iv", "atm_pe_iv",
       "iv_skew_ce_minus_pe", "pcr_oi", "pcr_vol", "pcr_oi_chg_5", "fut_chg_oi", "oi_buildup_ratio",
       "fut_chg_oi_ratio_20", "delivery_pct_chg_5", "ret_5d", "ret_20d", "ret_20d_cs_rank",
       "rel_ret_5d_vs_nifty", "close_sma50_dist", "close_sma20_dist", "adx_14", "gap_pct",
       "mkt_pct_above_sma50", "atr_pct_14"]
FEATS = ["atm_iv", "atr_pct_14", "vol_spread", "iv_skew_ce_minus_pe", "pcr_oi", "pcr_vol",
         "pcr_oi_chg_5", "fut_chg_oi", "oi_buildup_ratio", "fut_chg_oi_ratio_20", "delivery_pct_chg_5",
         "ret_5d", "ret_20d", "ret_20d_cs_rank", "rel_ret_5d_vs_nifty", "close_sma50_dist",
         "close_sma20_dist", "adx_14", "gap_pct", "mkt_pct_above_sma50", "dist_52wh"]
CLF = dict(n_estimators=350, max_depth=4, learning_rate=0.03, subsample=0.8,
           colsample_bytree=0.8, tree_method="hist", device="cuda", verbosity=0)


def _data():
    groups = json.loads((LOCK_V2 / "universe_groups.json").read_text())
    g2 = {s: g for g, v in groups.items() for s in v}
    m = load_market_data(columns=RAW)
    m["symbol"] = m["symbol"].astype(str)
    m = m[m.symbol.isin(g2)].sort_values(["symbol", "date"]).reset_index(drop=True)
    g = m.groupby("symbol", sort=False)
    entry = g["open"].shift(-1)
    H = pd.concat([g["high"].shift(-i) for i in range(1, 6)], axis=1).max(axis=1)
    L = pd.concat([g["low"].shift(-i) for i in range(1, 6)], axis=1).min(axis=1)
    m["up_move"] = (H - entry) / entry
    m["down_move"] = (entry - L) / entry
    m["vol_spread"] = m.atm_ce_iv - m.atm_pe_iv
    hi252 = g["close"].transform(lambda s: s.rolling(252, min_periods=120).max())
    m["dist_52wh"] = m.close / hi252 - 1.0
    m["group"] = m.symbol.map(g2)
    m["eligible"] = m.close.ge(100) & m.atm_iv.notna()
    m["big"] = np.maximum(m.up_move, m.down_move) >= BIG
    m["target_up"] = (m.up_move > m.down_move).astype(float)   # which side dominated (NaN if no fwd)
    m.loc[m.up_move.isna(), "target_up"] = np.nan
    return m


def walk_forward(m):
    parts = []
    for q in pd.period_range("2024Q1", "2026Q2", freq="Q"):
        tr = m[(m.date < q.start_time) & m.big & m.target_up.notna()].dropna(subset=FEATS)
        ev = m[(m.date >= q.start_time) & (m.date <= q.end_time) & m.eligible].copy()
        if tr.empty or ev.empty:
            continue
        clf = XGBClassifier(**CLF).fit(tr[FEATS], tr.target_up.astype(int))
        ev["p_up"] = clf.predict_proba(ev[FEATS])[:, 1]
        parts.append(ev)
    return pd.concat(parts, ignore_index=True)


def main():
    m = _data()
    oos = walk_forward(m)

    # honest OOS direction quality on big-move rows (where we know the answer)
    big = oos[oos.big & oos.target_up.notna()]
    auc = roc_auc_score(big.target_up.astype(int), big.p_up) if big.target_up.nunique() > 1 else np.nan
    print(f"OOS direction AUC on big movers = {auc:.3f}  (0.5 = coin flip)  n={len(big)}")

    # overlay onto the v2 book picks
    book = pd.read_csv(LOCK_V2 / "book_2024_26.csv", parse_dates=["date"])
    book["symbol"] = book["symbol"].astype(str)
    ov = book.merge(oos[["date", "symbol", "p_up"]], on=["date", "symbol"], how="left")
    ov["dir_label"] = np.where(ov.p_up >= 0.5, "UP", "DOWN")
    ov["confidence"] = (2 * ov.p_up - 1).abs()
    ov["conf_tier"] = pd.cut(ov.confidence, [-1, 0.1, 0.25, 2], labels=["low", "med", "high"])
    ov["actual_up"] = np.where(ov.up_move.notna(), (ov.up_move > ov.down_move).astype(float), np.nan)
    ov["dir_correct"] = np.where(ov.actual_up.notna(), (ov.dir_label.eq("UP").astype(float) == ov.actual_up), np.nan)

    done = ov[ov.dir_correct.notna()]
    print(f"\nDirection overlay on v2 picks (historical, n={len(done)}):")
    print(f"  overall accuracy = {done.dir_correct.mean()*100:.1f}%")
    for tier in ("low", "med", "high"):
        d = done[done.conf_tier == tier]
        if len(d):
            print(f"  {tier:4s} conf (n={len(d):4d}): acc {d.dir_correct.mean()*100:.1f}% | "
                  f"UP {(d.dir_label=='UP').mean()*100:.0f}%")

    out = ov[["date", "group", "symbol", "p_up", "dir_label", "confidence", "conf_tier",
              "actual_up", "dir_correct"]].copy()
    out["p_up"] = out.p_up.round(3)
    out["confidence"] = out.confidence.round(3)
    out.to_csv(LOCK_V2 / "direction_overlay.csv", index=False)
    print(f"\nsaved overlay -> {LOCK_V2 / 'direction_overlay.csv'}  (rows={len(out)})")
    # latest live picks' direction call
    live = out[out.actual_up.isna()].sort_values(["date", "group"]).tail(8)
    if len(live):
        print("\nlatest live direction calls:")
        for _, r in live.iterrows():
            print(f"  {r.date.date()} {r.group} {r.symbol}: {r.dir_label} (p_up={r.p_up}, {r.conf_tier})")


if __name__ == "__main__":
    main()
