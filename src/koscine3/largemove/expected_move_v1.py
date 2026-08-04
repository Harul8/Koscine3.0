"""PRODUCTION expected_move_v1 — NEXT-DAY expected MOVE size (magnitude) per name + Nifty/FII context.

The validated, reliable edge (experiments/megacap_direction_v1): direction is a coin flip, but next-day MOVE SIZE
is predictable (rank-IC ~0.15-0.21; model beats atm_iv in 2026). This engine forecasts each name's next-day |move|
for the 65 universe (CatBoost regressor on vol/IV/range features), alongside the atm_iv-implied move and trailing
realized move for transparency, plus a Nifty expected-move + FII-flow context line for the daily "Tomorrow" panel.

Use: size positions / judge if the premium is justified / pick straddle-strangle vs a side. INCREMENTAL (recompute
live month only). Reads data READ-ONLY; writes locks/prod_expected_move_v1/. Does NOT touch other PROD engines.

    python -m koscine3.largemove.expected_move_v1            # incremental (fast)
    python -m koscine3.largemove.expected_move_v1 --full     # full monthly walk-forward rebuild
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from koscine3.data.sources import load_market_data
from koscine3.largemove.mover_v2 import LOCK_V2  # read-only: shared universe_groups.json

ROOT = LOCK_V2.parents[1]
LOCK = ROOT / "locks" / "prod_expected_move_v1"
FII = ROOT / "data" / "silver" / "fii_dii_cash.parquet"
BOOK_PATH = LOCK / "expected_move_book.csv"

VERSION = "prod_expected_move_v1"
START = pd.Timestamp("2024-01-01")
MIN_UNDERLYING = 100.0
CB = dict(iterations=400, depth=5, learning_rate=0.03, l2_leaf_reg=6.0, random_seed=7,
          allow_writing_files=False, verbose=False, task_type="GPU", devices="0")
MAGF = ["atm_iv", "realized_vol_20", "atr_pct_14", "bb_width_20", "donchian_width_20", "abs_ret1", "abs_ret5",
        "vol_5v20_ratio", "gap_abs", "atm_iv_chg_5", "atr_5", "range_pct", "day_of_week", "is_expiry_week"]
COLS = ["date", "symbol", "group", "atm_iv", "exp_move_pct", "iv_implied_pct", "realized20_pct", "fwd1_abs_pct", "live"]


def load_panel():
    g2 = {s: g for g, syms in json.loads((LOCK_V2 / "universe_groups.json").read_text()).items() for s in syms}
    m = load_market_data()
    m["symbol"] = m.symbol.astype(str); m["date"] = pd.to_datetime(m["date"])
    m = m.sort_values(["symbol", "date"]).reset_index(drop=True)
    g = m.groupby("symbol", sort=False)
    m["fwd1"] = g["close"].shift(-1) / m.close - 1.0
    m["mag"] = m["fwd1"].abs()
    m["abs_ret1"] = m.ret_1d.abs() if "ret_1d" in m else np.nan
    m["abs_ret5"] = m.ret_5d.abs() if "ret_5d" in m else np.nan
    m["gap_abs"] = m.gap_pct.abs() if "gap_pct" in m else np.nan
    m["realized20"] = g["ret_1d"].transform(lambda s: s.abs().rolling(20).mean())
    m["group"] = m.symbol.map(g2)
    m["eligible"] = m.close.ge(MIN_UNDERLYING) & m.atm_iv.notna()
    feats = [c for c in MAGF if c in m.columns]
    return m, feats, g2


def scores_for_months(m, feats, months):
    from catboost import CatBoostRegressor
    parts = []
    for mo in months:
        ms, me = mo.start_time, mo.end_time
        cut = ms - pd.Timedelta(days=2)
        tr = m[(m.date < cut) & m.eligible & m.mag.notna()]
        ev = m[(m.date >= ms) & (m.date <= me) & m.eligible & m.group.notna()]
        if len(tr) < 5000 or ev.empty:
            continue
        mdl = CatBoostRegressor(**CB, loss_function="RMSE").fit(tr[feats], tr.mag.clip(0, 0.2))
        e = ev[["date", "symbol"]].copy()
        e["exp_move_pct"] = mdl.predict(ev[feats]).clip(0, 0.2) * 100
        parts.append(e)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["date", "symbol", "exp_move_pct"])


def build_rows(m, scores, months):
    base = m[(m.group.notna()) & m.eligible & m.date.dt.to_period("M").isin(set(months))].copy()
    base = base.merge(scores, on=["date", "symbol"], how="left")
    base["iv_implied_pct"] = base.atm_iv / np.sqrt(252) * 100
    base["realized20_pct"] = base.realized20 * 100
    base["fwd1_abs_pct"] = base.mag * 100
    base["live"] = base.fwd1.isna()
    return base.reindex(columns=COLS)


def nifty_fii_context(m):
    last = m[m.eligible & m.group.notna()].date.max()
    nrv = m.loc[m.date == last, "nifty_realized_vol_20"].dropna()
    n5 = m.loc[m.date == last, "nifty_ret_5d"].dropna()
    ctx = {"date": str(last.date()),
           "nifty_exp_move_pct": round(float(nrv.iloc[0] * 100), 2) if len(nrv) else None,
           "nifty_ret_5d_pct": round(float(n5.iloc[0] * 100), 2) if len(n5) else None}
    if FII.exists():
        f = pd.read_parquet(FII)[["date", "fii_net"]]; f["date"] = pd.to_datetime(f.date); f = f.sort_values("date")
        f = f[f.date <= last]
        ctx["fii_net_latest"] = round(float(f.fii_net.iloc[-1]), 0) if len(f) else None
        ctx["fii_net_5d"] = round(float(f.fii_net.tail(5).sum()), 0) if len(f) else None
        ctx["fii_signal"] = ("buying" if (ctx.get("fii_net_5d") or 0) > 0 else "selling")
    return ctx


def main(full: bool = False):
    LOCK.mkdir(parents=True, exist_ok=True)
    m, feats, g2 = load_panel()
    latest = m.date.max().to_period("M")
    existing = pd.read_csv(BOOK_PATH, parse_dates=["date"]) if (BOOK_PATH.exists() and not full) else None
    if existing is None or existing.empty:
        recompute_from, keep, mode = START.to_period("M"), pd.DataFrame(columns=COLS), "FULL"
    else:
        existing["symbol"] = existing.symbol.astype(str); existing["_mo"] = existing.date.dt.to_period("M")
        live_mos = existing.loc[existing.live == True, "_mo"]                     # noqa: E712
        recompute_from = (min(live_mos) if len(live_mos) else existing["_mo"].max() + 1)
        keep = existing[existing["_mo"] < recompute_from].drop(columns=["_mo"]).reindex(columns=COLS)
        mode = "INCREMENTAL"
    months = list(pd.period_range(recompute_from, latest, freq="M"))
    print(f"[{mode}] recompute {months[0] if months else '-'}..{latest} ({len(months)} month(s)) | kept {len(keep)} rows")
    scores = scores_for_months(m, feats, months) if months else pd.DataFrame(columns=["date", "symbol", "exp_move_pct"])
    new_rows = build_rows(m, scores, months) if months else pd.DataFrame(columns=COLS)
    book = pd.concat([keep, new_rows], ignore_index=True).sort_values(["date", "symbol"]).reset_index(drop=True)
    book.to_csv(BOOK_PATH, index=False)

    done = book[(book.live != True) & book.exp_move_pct.notna()]                  # noqa: E712
    ic = round(float(done.exp_move_pct.corr(done.fwd1_abs_pct, "spearman")), 4) if len(done) else None
    ctx = nifty_fii_context(m)
    manifest = {"version": VERSION, "purpose": "next-day expected MOVE size (magnitude) per name + Nifty/FII context",
                "model": {"type": "CatBoost regressor |next-day move|", "train_window": "EXPANDING", "retrain": "MONTHLY (incremental)",
                          "features": len(feats), "note": "direction is a coin flip; this is the reliable magnitude edge"},
                "book_rows": int(len(book)), "dates": [str(book.date.min().date()), str(book.date.max().date())],
                "rank_ic_pred_vs_realized": ic, "tomorrow": ctx}
    (LOCK / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (LOCK / "universe_groups.json").write_text((LOCK_V2 / "universe_groups.json").read_text())
    print(f"expected_move_v1: {len(book)} rows | rank-IC {ic} | tomorrow {ctx.get('date')}: "
          f"Nifty ~+/-{ctx.get('nifty_exp_move_pct')}% , FII 5d {ctx.get('fii_net_5d')} ({ctx.get('fii_signal')})")
    print(f"  -> {BOOK_PATH}\nsaved lock -> {LOCK}")


if __name__ == "__main__":
    main(full="--full" in sys.argv)
