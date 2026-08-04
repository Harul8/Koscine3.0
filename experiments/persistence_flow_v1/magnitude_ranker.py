"""Direction-agnostic LARGE-MOVER ranker — maximize precision of picking the day's top movers.

Objective (per user): forget direction; pick the stocks that will be the day's biggest movers
(by |move|) with high precision. move_mag = max(up_move, down_move) over the 5-day window.

Model: XGBoost regression on move_mag (trained broad on all optionable stocks), walk-forward
(train<T, predict T). Rich magnitude/catalyst features (vol level + IV change + earnings + gap +
OI/volume surges). Rank within (date, group); compare precision@k vs PROD confidence ranking.

Metric: of our top-N picks, what fraction are in the actual top-3 / top-5 movers of the group that day.
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

PROD_PRED = HERE.parents[1] / "locks" / "prod_largemove_v1" / "predictions"
pd.set_option("display.width", 200)

EXTRA = ["atm_iv_chg_5", "earnings_within_5d", "gap_pct", "fut_chg_oi", "oi_buildup_ratio",
         "vol_sma20_ratio", "pcr_oi_chg_5", "ret_5d", "ret_20d", "fut_chg_oi_ratio_20",
         "delivery_qty_ratio_20", "adx_14"]

FEATS = ["atm_iv", "atr_pct_14", "realized_vol_20", "donchian_width_20", "sector_vol_20",
         "nifty_realized_vol_20", "atm_iv_ratio_20", "atm_iv_chg_5", "atr_pct_14_rank_60d",
         "atr_pct_14_cs_rank", "days_to_earnings", "earnings_within_5d", "gap_pct",
         "vol_sma20_ratio", "vol_5v20_ratio", "abs_ret_5d", "abs_ret_20d", "abs_fut_chg_oi",
         "oi_buildup_ratio", "pcr_oi_chg_5", "ret_20d_cs_rank", "delivery_qty_ratio_20"]

REG = dict(n_estimators=400, max_depth=6, learning_rate=0.04, subsample=0.85,
           colsample_bytree=0.85, tree_method="hist", device="cuda", verbosity=0)


def build():
    df = P.load_dataset(PROD)
    mag = df.groupby(["date", "symbol"])["ceiling"].max().reset_index(name="move_mag")
    base = df[df.side.eq("long")].drop(columns=["side", "ceiling"]).merge(mag, on=["date", "symbol"])
    extra = load_market_data(columns=["date", "symbol", *EXTRA])
    extra["symbol"] = extra["symbol"].astype(str)
    base = base.merge(extra.drop_duplicates(["date", "symbol"]), on=["date", "symbol"], how="left")
    base["abs_ret_5d"] = base["ret_5d"].abs()
    base["abs_ret_20d"] = base["ret_20d"].abs()
    base["abs_fut_chg_oi"] = base["fut_chg_oi"].abs()
    base = base.dropna(subset=["move_mag"])

    # PROD confidence (max over sides) = baseline mover-ranker
    prod = pd.concat([pd.read_csv(PROD_PRED / f"group_{b}_predictions.csv", parse_dates=["date"])
                      for b in dict(PROD.group_thresholds)], ignore_index=True)
    prod["symbol"] = prod["symbol"].astype(str)
    conf = prod.groupby(["date", "symbol"])["confidence"].max().reset_index(name="prod_conf")
    base = base.merge(conf, on=["date", "symbol"], how="left")
    return base


def walk_forward(base):
    parts = []
    for T in PROD.test_years:
        tr = base[base.year < T].dropna(subset=FEATS + ["move_mag"])
        ev = base[(base.year == T) & base.eligible & base.group.notna()].copy()
        if ev.empty:
            continue
        reg = XGBRegressor(**REG).fit(tr[FEATS], tr["move_mag"].clip(0, 0.5))
        ev["mag_hat"] = reg.predict(ev[FEATS])
        parts.append(ev)
    return pd.concat(parts, ignore_index=True), reg


def precision(ev, score, label="picks", N=2):
    ev = ev.copy()
    ev["actual_rank"] = ev.groupby(["date", "group"])["move_mag"].rank(ascending=False, method="first")
    ev["day_max"] = ev.groupby(["date", "group"])["move_mag"].transform("max")
    ev["pred_rank"] = ev.groupby(["date", "group"])[score].rank(ascending=False, method="first")
    out = {}
    for n in (1, 2, 3):
        picks = ev[ev.pred_rank <= n]
        out[n] = {
            "in_top3_%": round((picks.actual_rank <= 3).mean() * 100, 1),
            "in_top5_%": round((picks.actual_rank <= 5).mean() * 100, 1),
            "avg_move_%": round(picks.move_mag.mean() * 100, 1),
            "capture_%": round((picks.move_mag / picks.day_max).mean() * 100, 1),
        }
    return out


def show(name, ev, score):
    print(f"\n### {name}")
    res = precision(ev, score)
    df = pd.DataFrame(res).T
    df.index.name = "pick_topN"
    print(df.to_string())


def main():
    base = build()
    print(f"rows={len(base)} | optionable A/B test rows pending walk-forward")
    ev, reg = walk_forward(base)
    print(f"eval rows (A/B, eligible, test yrs)={len(ev)} | "
          f"avg group size/day={ev.groupby(['date','group']).size().mean():.0f}")

    print("\n" + "=" * 72)
    print("PRECISION @ topN picks — are picks among the day's biggest movers (per group)?")
    print("=" * 72)
    show("BASELINE: PROD confidence ranking", ev.dropna(subset=["prod_conf"]), "prod_conf")
    show("NEW: magnitude regressor", ev, "mag_hat")

    # per group / year for the magnitude ranker, top-2
    print("\n" + "=" * 72)
    print("MAGNITUDE ranker — top-2 picks, per group/year (in_top3 / in_top5 / capture)")
    print("=" * 72)
    rows = []
    for b in dict(PROD.group_thresholds):
        for yr in (None, 2024, 2025, 2026):
            d = ev[ev.group.eq(b)] if yr is None else ev[ev.group.eq(b) & ev.date.dt.year.eq(yr)]
            if d.empty:
                continue
            r = precision(d, "mag_hat")[2]
            rows.append({"scope": b if yr is None else f"  {b} {yr}", **r})
    print(pd.DataFrame(rows).to_string(index=False))

    imp = pd.Series(reg.feature_importances_, index=FEATS).sort_values(ascending=False)
    print("\ntop magnitude features:", ", ".join(f"{k}={v:.2f}" for k, v in imp.head(10).items()))
    ev.to_parquet(HERE / "magnitude_oos.parquet")


if __name__ == "__main__":
    main()
