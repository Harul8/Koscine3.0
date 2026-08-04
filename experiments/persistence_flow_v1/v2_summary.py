"""v2 mover-picker — QUARTERLY-retrained walk-forward 2024-26 + per-group summary.

Ranker compared: atm_iv (rule, no retrain) vs quarterly-retrained XGB on move_mag.
Summary per group (A top-30, B next-35) x depth (top-3, top-5 picks/day):
  - % picks 5-day |move| > 6% and > 8%   (what a straddle / peak-exit monetizes)
  - % "closed opposite" = spiked one way but 5-day close ended the other way (whipsaw)
  - coverage = # distinct stocks picked / group size
  - top-5 concentration = share of picks from the 5 most-picked names
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from largemove import pipeline as P
from largemove.config import PROD
from koscine3.data.sources import load_market_data

pd.set_option("display.width", 240)
EXTRA = ["atm_iv_chg_5", "earnings_within_5d", "gap_pct", "fut_chg_oi", "oi_buildup_ratio",
         "vol_sma20_ratio", "pcr_oi_chg_5", "ret_5d", "ret_20d", "fut_chg_oi_ratio_20", "delivery_qty_ratio_20"]
FEATS = ["atm_iv", "atr_pct_14", "realized_vol_20", "donchian_width_20", "sector_vol_20",
         "nifty_realized_vol_20", "atm_iv_ratio_20", "atm_iv_chg_5", "atr_pct_14_rank_60d",
         "atr_pct_14_cs_rank", "days_to_earnings", "earnings_within_5d", "gap_pct", "vol_sma20_ratio",
         "vol_5v20_ratio", "abs_ret_5d", "abs_ret_20d", "abs_fut_chg_oi", "oi_buildup_ratio",
         "pcr_oi_chg_5", "ret_20d_cs_rank", "delivery_qty_ratio_20", "dist_52wh"]
REG = dict(n_estimators=400, max_depth=6, learning_rate=0.04, subsample=0.85,
           colsample_bytree=0.85, tree_method="hist", device="cuda", verbosity=0)
GROUP_SIZE = {"A_mcap30": 30, "B_turn35": 35}


def build():
    df = P.load_dataset(PROD)
    piv = df.pivot_table(index=["date", "symbol"], columns="side", values="ceiling").reset_index()
    piv = piv.rename(columns={"long": "up_move", "short": "down_move"})
    base = df[df.side.eq("long")].drop(columns=["side", "ceiling"]).merge(piv, on=["date", "symbol"])
    base["move_mag"] = base[["up_move", "down_move"]].max(axis=1)
    base["peak_up"] = base.up_move > base.down_move

    mk = load_market_data(columns=["date", "symbol", "open", "close", *EXTRA])
    mk["symbol"] = mk["symbol"].astype(str)
    mk = mk.sort_values(["symbol", "date"])
    g = mk.groupby("symbol", sort=False)
    sc = pd.DataFrame({"date": mk["date"].values, "symbol": mk["symbol"].values,
                       "signed_close": (g["close"].shift(-5).values - g["open"].shift(-1).values)
                       / g["open"].shift(-1).values})
    base = base.merge(mk.drop(columns=["open", "close"]).drop_duplicates(["date", "symbol"]),
                      on=["date", "symbol"], how="left").merge(sc, on=["date", "symbol"], how="left")

    base = base.sort_values(["symbol", "date"])
    hi252 = base.groupby("symbol")["close"].transform(lambda s: s.rolling(252, min_periods=120).max())
    base["dist_52wh"] = base["close"] / hi252 - 1.0
    base["abs_ret_5d"] = base["ret_5d"].abs()
    base["abs_ret_20d"] = base["ret_20d"].abs()
    base["abs_fut_chg_oi"] = base["fut_chg_oi"].abs()
    base["closed_opp"] = (base.peak_up & (base.signed_close < 0)) | (~base.peak_up & (base.signed_close > 0))
    return base


def quarterly(base):
    parts = []
    for q in pd.period_range("2024Q1", "2026Q2", freq="Q"):
        tr = base[base.date < q.start_time].dropna(subset=FEATS + ["move_mag"])
        te = base[(base.date >= q.start_time) & (base.date <= q.end_time)
                  & base.eligible & base.group.notna()].copy()
        if te.empty:
            continue
        reg = XGBRegressor(**REG).fit(tr[FEATS], tr["move_mag"].clip(0, 0.5))
        te["model_pred"] = reg.predict(te[FEATS])
        parts.append(te)
    return pd.concat(parts, ignore_index=True)


def summarize(ev, ranker):
    yrs = (ev.date.max() - ev.date.min()).days / 365.25
    ev = ev.copy()
    ev["r"] = ev.groupby(["date", "group"])[ranker].rank(ascending=False, method="first")
    ev["arank"] = ev.groupby(["date", "group"])["move_mag"].rank(ascending=False, method="first")
    rows = []
    for grp in ("A_mcap30", "B_turn35"):
        for depth in (3, 5):
            d = ev[(ev.group == grp) & (ev.r <= depth)]
            vc = d.symbol.value_counts()
            rows.append({
                "group": grp, "depth": f"top{depth}", "trades": len(d), "per_yr": round(len(d) / yrs),
                "move>6%": round((d.move_mag >= 0.06).mean() * 100, 1),
                "move>8%": round((d.move_mag >= 0.08).mean() * 100, 1),
                "in_top3%": round((d.arank <= 3).mean() * 100, 1),
                "in_top5%": round((d.arank <= 5).mean() * 100, 1),
                "closed_opp%": round(d.closed_opp.mean() * 100, 1),
                "coverage": f"{d.symbol.nunique()}/{GROUP_SIZE[grp]}",
                "top5_share%": round(vc.head(5).sum() / len(d) * 100, 1),
                "top5_names": ", ".join(f"{s}({c})" for s, c in vc.head(5).items()),
            })
    return pd.DataFrame(rows)


def main():
    base = build()
    ev = quarterly(base)
    print(f"quarterly walk-forward rows={len(ev)} | {ev.date.min().date()}..{ev.date.max().date()} "
          f"| quarters={ev.date.dt.to_period('Q').nunique()}\n")

    # ranker comparison at top-3 (combined)
    for ranker in ("atm_iv", "model_pred"):
        e = ev.copy()
        e["r"] = e.groupby(["date", "group"])[ranker].rank(ascending=False, method="first")
        e["arank"] = e.groupby(["date", "group"])["move_mag"].rank(ascending=False, method="first")
        d = e[e.r <= 3]
        print(f"[{ranker}] top-3 combined: move>6%={ (d.move_mag>=.06).mean()*100:.1f}  "
              f">8%={ (d.move_mag>=.08).mean()*100:.1f}  in_top5%={ (d.arank<=5).mean()*100:.1f}")

    cols = ["group", "depth", "trades", "per_yr", "move>6%", "move>8%", "in_top3%", "in_top5%",
            "closed_opp%", "coverage", "top5_share%"]
    for ranker in ("atm_iv", "model_pred"):
        print("\n" + "=" * 110)
        print(f"SUMMARY — ranker = {ranker}  (quarterly retrain {'used' if ranker=='model_pred' else 'n/a: rule'})")
        print("=" * 110)
        s = summarize(ev, ranker)
        print(s[cols].to_string(index=False))
        print("\ntop-5 most-picked names per block:")
        for _, row in s.iterrows():
            print(f"  {row['group']} {row['depth']}: {row['top5_names']}")


if __name__ == "__main__":
    main()
