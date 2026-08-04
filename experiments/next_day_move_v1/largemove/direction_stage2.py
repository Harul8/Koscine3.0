"""STAGE-2 DIRECTION model (Koscine-2.0 style) — overlay on K3 v2 picks. Does NOT touch PROD selection.

K2.0 learnings applied:
  - CLEAN dominance labels: clean_bull = up_move_5d > THR & down_move_5d < DOM*THR (move went clearly one way);
    clean_bear symmetric. (K2.0 used a tight adverse limit / 0.8*thr dominance — the source of its directional purity.)
  - Per-side models (bull, bear), not a single P(up). dir = argmax; p_up = bull/(bull+bear).
  - Conviction = max(bull,bear) and margin |bull-bear|; HIGH-CONVICTION subset is where the edge lives.
Honest expectation (from K2.0's own reports): ~55-60% directional accuracy on the selective high-conviction
subset, ~coin-flip otherwise. Captured as confidence so low-conviction is taken with a grain of salt.

    python -m largemove.direction_stage2     # writes locks/prod_largemove_v2/direction_overlay.csv
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from koscine3.data.sources import load_market_data
from largemove.mover_v2 import LOCK_V2

THR = 0.04            # "a real move" threshold for a clean directional event
DOM = 0.6             # dominance: opposite excursion must be < DOM*THR for the move to count as clean
RAW = ["date", "symbol", "open", "high", "low", "close",
       "atm_iv", "atm_ce_iv", "atm_pe_iv", "iv_skew_ce_minus_pe", "put_call_iv_skew", "atm_iv_chg_5",
       "pcr_oi", "pcr_vol", "pcr_oi_chg_5", "pcr_vol_chg_5",
       "fut_chg_oi", "oi_buildup_ratio", "fut_chg_oi_ratio_20", "fut_oi_chg_5",
       "delivery_pct_chg_5", "delivery_qty_ratio_20",
       "ret_1d", "ret_5d", "ret_10d", "ret_20d", "ret_5d_cs_rank", "ret_20d_cs_rank",
       "rel_ret_5d_vs_nifty", "close_sma20_dist", "close_sma50_dist", "adx_14",
       "mkt_pct_above_sma50", "gap_pct", "stock_rel_sector_ret_5d", "stock_rel_sector_ret_20d", "sector_ret_5d"]
FEATS = ["atm_iv", "vol_spread", "iv_skew_ce_minus_pe", "put_call_iv_skew", "atm_iv_chg_5",
         "pcr_oi", "pcr_vol", "pcr_oi_chg_5", "pcr_vol_chg_5",
         "fut_chg_oi", "oi_buildup_ratio", "fut_chg_oi_ratio_20", "fut_oi_chg_5",
         "delivery_pct_chg_5", "delivery_qty_ratio_20",
         "ret_1d", "ret_5d", "ret_10d", "ret_20d", "ret_5d_cs_rank", "ret_20d_cs_rank",
         "rel_ret_5d_vs_nifty", "close_sma20_dist", "close_sma50_dist", "dist_52wh", "adx_14",
         "mkt_pct_above_sma50", "gap_pct", "stock_rel_sector_ret_5d", "stock_rel_sector_ret_20d", "sector_ret_5d"]
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
    c5 = g["close"].shift(-5)
    m["up_move"] = (H - entry) / entry
    m["down_move"] = (entry - L) / entry
    m["signed_close"] = (c5 - entry) / entry
    m["vol_spread"] = m.atm_ce_iv - m.atm_pe_iv
    hi252 = g["close"].transform(lambda s: s.rolling(252, min_periods=120).max())
    m["dist_52wh"] = m.close / hi252 - 1.0
    m["group"] = m.symbol.map(g2)
    m["eligible"] = m.close.ge(100) & m.atm_iv.notna()
    # CLEAN dominance labels (K2.0 style)
    m["clean_bull"] = ((m.up_move > THR) & (m.down_move < DOM * THR)).astype(float)
    m["clean_bear"] = ((m.down_move > THR) & (m.up_move < DOM * THR)).astype(float)
    m.loc[m.up_move.isna(), ["clean_bull", "clean_bear"]] = np.nan
    return m


def walk_forward(m):
    parts = []
    for q in pd.period_range("2024Q1", "2026Q2", freq="Q"):
        tr = m[(m.date < q.start_time) & m.clean_bull.notna()].dropna(subset=FEATS)
        ev = m[(m.date >= q.start_time) & (m.date <= q.end_time) & m.eligible].copy()
        if tr.empty or ev.empty:
            continue
        yb, yr = tr.clean_bull.astype(int), tr.clean_bear.astype(int)
        bull = XGBClassifier(scale_pos_weight=(len(yb) - yb.sum()) / max(1, yb.sum()), **CLF).fit(tr[FEATS], yb)
        bear = XGBClassifier(scale_pos_weight=(len(yr) - yr.sum()) / max(1, yr.sum()), **CLF).fit(tr[FEATS], yr)
        ev["bull_score"] = bull.predict_proba(ev[FEATS])[:, 1]
        ev["bear_score"] = bear.predict_proba(ev[FEATS])[:, 1]
        parts.append(ev)
    return pd.concat(parts, ignore_index=True)


def main():
    m = _data()
    oos = walk_forward(m)
    oos["p_up"] = oos.bull_score / (oos.bull_score + oos.bear_score + 1e-9)
    oos["dir_label"] = np.where(oos.p_up >= 0.5, "UP", "DOWN")
    oos["confidence"] = (2 * oos.p_up - 1).abs()
    oos["conviction"] = oos[["bull_score", "bear_score"]].max(axis=1)
    # high-conviction = top tertile of the chosen-side score (K2.0: fire only when confident)
    hi_cut = oos.conviction.quantile(0.66)
    oos["high_conviction"] = oos.conviction >= hi_cut
    oos["conf_tier"] = np.where(oos.high_conviction, "high",
                                np.where(oos.conviction >= oos.conviction.quantile(0.33), "med", "low"))

    # honest direction quality on big movers (we know the answer): closed in the called direction?
    big = oos[(np.maximum(oos.up_move, oos.down_move) >= THR) & oos.signed_close.notna()].copy()
    big["called_up"] = big.dir_label.eq("UP")
    big["closed_up"] = big.signed_close > 0
    big["dir_correct"] = big.called_up == big.closed_up
    print(f"on big movers (n={len(big)}): close-direction accuracy by conviction tier")
    for t in ("low", "med", "high"):
        d = big[big.conf_tier == t]
        if len(d):
            print(f"  {t:4s} (n={len(d):5d}): {d.dir_correct.mean()*100:.1f}% closed in called dir | "
                  f"called UP {d.called_up.mean()*100:.0f}%")

    # overlay onto the v2 book picks
    book = pd.read_csv(LOCK_V2 / "book_2024_26.csv", parse_dates=["date"])
    book["symbol"] = book["symbol"].astype(str)
    cols = ["date", "symbol", "group", "p_up", "bull_score", "bear_score", "dir_label",
            "confidence", "conf_tier", "high_conviction"]
    ov = book.merge(oos[cols], on=["date", "symbol", "group"], how="left")
    # realized check on historical picks
    ovm = ov.merge(oos[["date", "symbol", "group", "signed_close", "up_move", "down_move"]],
                   on=["date", "symbol", "group"], how="left")
    ovm["actual_up"] = np.where(ovm.signed_close.notna(), (ovm.signed_close > 0).astype(float), np.nan)
    ovm["dir_correct"] = np.where(ovm.actual_up.notna(),
                                  (ovm.dir_label.eq("UP").astype(float) == ovm.actual_up), np.nan)
    done = ovm[ovm.dir_correct.notna()]
    print(f"\non v2 picks (historical, n={len(done)}): close-direction accuracy")
    print(f"  overall {done.dir_correct.mean()*100:.1f}%")
    for t in ("low", "med", "high"):
        d = done[done.conf_tier == t]
        if len(d):
            print(f"  {t:4s} (n={len(d):4d}): {d.dir_correct.mean()*100:.1f}%")
    hc = done[done.high_conviction == True]  # noqa: E712
    if len(hc):
        print(f"  HIGH-CONVICTION subset (n={len(hc)}): {hc.dir_correct.mean()*100:.1f}% directional accuracy")

    out = ov[cols].copy()
    out["p_up"] = out.p_up.round(3)
    out["bull_score"] = out.bull_score.round(3)
    out["bear_score"] = out.bear_score.round(3)
    out["confidence"] = out.confidence.round(3)
    out.to_csv(LOCK_V2 / "direction_overlay.csv", index=False)
    print(f"\nsaved K2.0-style overlay -> {LOCK_V2 / 'direction_overlay.csv'} (rows={len(out)})")


if __name__ == "__main__":
    main()
