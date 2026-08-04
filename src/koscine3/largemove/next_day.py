"""PROD 1-day (t+1) movement model — patched ALONGSIDE the frozen v2 5-day engine (does not touch it).

Builds next_day_book.csv: per (date, symbol) in the A/B universe — predicted next-day move magnitude
(calibrated XGB, ~= atm_iv) + realized next-day move (signed close-to-close & magnitude). Live tail (latest
date, no t+1 yet) carries the prediction with null realized. Rank within group by predicted move = the daily
"biggest movers tomorrow" list. Direction-agnostic (1-day direction is ~coin flip).

    python -m koscine3.largemove.next_day      # writes locks/prod_largemove_v2/next_day_book.csv
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from koscine3.data.sources import load_market_data
from koscine3.largemove.mover_v2 import LOCK_V2

G2 = {s: g for g, syms in json.loads((LOCK_V2 / "universe_groups.json").read_text()).items() for s in syms}
RAW = ["date", "symbol", "open", "high", "low", "close", "atm_iv", "atm_iv_chg_5", "atm_iv_ratio_20",
       "atr_pct_14", "atr_pct_14_rank_60d", "realized_vol_20", "donchian_width_20", "sector_vol_20",
       "nifty_realized_vol_20", "vol_sma20_ratio", "days_to_earnings", "earnings_within_5d", "gap_pct",
       "ret_1d", "ret_5d", "fut_chg_oi", "oi_buildup_ratio", "pcr_oi_chg_5", "turnover_ratio_20"]
FEATS = ["atm_iv", "atm_iv_chg_5", "atm_iv_ratio_20", "atr_pct_14", "atr_pct_14_rank_60d", "realized_vol_20",
         "donchian_width_20", "sector_vol_20", "nifty_realized_vol_20", "vol_sma20_ratio", "days_to_earnings",
         "earnings_within_5d", "gap_pct", "abs_ret_1d", "abs_ret_5d", "abs_fut_chg_oi", "oi_buildup_ratio",
         "pcr_oi_chg_5", "turnover_ratio_20"]
REG = dict(n_estimators=500, max_depth=5, learning_rate=0.03, subsample=0.8, colsample_bytree=0.7,
           reg_lambda=5.0, min_child_weight=20, tree_method="hist", device="cuda", verbosity=0)


def build_book() -> pd.DataFrame:
    m = load_market_data(columns=RAW)
    m["symbol"] = m["symbol"].astype(str)
    m = m[m.symbol.isin(G2)].sort_values(["symbol", "date"]).reset_index(drop=True)
    m["date"] = pd.to_datetime(m["date"])
    g = m.groupby("symbol", sort=False)
    up = (g["high"].shift(-1) - m["close"]) / m["close"]
    dn = (m["close"] - g["low"].shift(-1)) / m["close"]
    m["next_move"] = np.maximum(up, dn)                                   # realized magnitude (either way)
    m["next_signed"] = (g["close"].shift(-1) - m["close"]) / m["close"]  # realized signed close-to-close
    m["abs_ret_1d"] = m["ret_1d"].abs()
    m["abs_ret_5d"] = m["ret_5d"].abs()
    m["abs_fut_chg_oi"] = m["fut_chg_oi"].abs()
    m[FEATS] = m[FEATS].replace([np.inf, -np.inf], np.nan)
    m["group"] = m["symbol"].map(G2)
    m["eligible"] = m["close"].ge(100) & m["atm_iv"].notna()

    parts = []
    for q in pd.period_range("2024Q1", pd.Timestamp.today().to_period("Q"), freq="Q"):
        tr = m[(m.date < q.start_time) & m.eligible & m.next_move.notna()].dropna(subset=FEATS)
        ev = m[(m.date >= q.start_time) & (m.date <= q.end_time) & m.eligible].copy()
        if tr.empty or ev.empty:
            continue
        reg = XGBRegressor(**REG).fit(tr[FEATS], tr.next_move.clip(0, 0.25))
        ev["pred"] = reg.predict(ev[FEATS])
        parts.append(ev)
    book = pd.concat(parts, ignore_index=True)
    out = pd.DataFrame({
        "date": book.date, "group": book.group, "symbol": book.symbol,
        "atm_iv": book.atm_iv.round(4),
        "pred_move_pct": (book.pred * 100).round(2),
        "next_move_pct": (book.next_move * 100).round(2),
        "next_signed_pct": (book.next_signed * 100).round(2),
        "live": book.next_move.isna(),
    }).sort_values(["date", "group", "pred_move_pct"], ascending=[True, True, False])
    out.to_csv(LOCK_V2 / "next_day_book.csv", index=False)
    return out


def main():
    out = build_book()
    done = out[~out.live]
    print(f"next_day_book rows={len(out)} | {out.date.min().date()}..{out.date.max().date()} "
          f"| live(latest) picks={out.live.sum()}")
    print(f"realized mean next-day move={done.next_move_pct.mean():.2f}% | "
          f"avg pred={done.pred_move_pct.mean():.2f}%")
    # IC sanity (per-day rank corr pred vs realized)
    ic = done.groupby("date").apply(lambda d: d[["pred_move_pct", "next_move_pct"]].corr("spearman").iloc[0, 1]
                                    if len(d) > 5 else np.nan).mean()
    print(f"cross-sectional rank IC = {ic:.3f}")
    print(f"saved -> {LOCK_V2 / 'next_day_book.csv'}")


if __name__ == "__main__":
    main()
