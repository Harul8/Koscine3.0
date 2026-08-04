"""PRODUCTION direction_v1 — group-B (movers) PUT/CALL directional LEAN over the frozen v3 5d book.

Design (settled by experiments/direction_overlay_v1): for group A (mega-caps) direction is a coin flip → v3 stays
DIRECTION-AGNOSTIC. For group B (35 movers) a monthly-retrained, EXPANDING-window, full-feature CatBoost classifier
predicts P(close[t+5] > close[t]); each v3 B pick is tagged CALL (P>=0.5) or PUT (P<0.5) with a conviction.

Key engine facts (do not "improve" without a fresh experiment — short windows INVERT, OI/flow-only ≈ 0):
  - EXPANDING train window (all history < cut), MONTHLY retrain, embargo 6d (purged for the 5d label).
  - Train on ALL eligible names (best OOS AUC/IC); predict group B. Full non-leak feature set.
  - It is a market-timing / beta tilt (signals cluster same-side per month) — a SMALL lean, not high conviction.
2026 group-B OOS: AUC 0.565, hit 0.541, IC +0.146, ATM±2% held EV ~+5% (train-All).

INCREMENTAL by default: a month's OOS prediction is fixed once made (its train set is all data < month_start-embargo,
which never changes), so each run recomputes only the still-live month(s) + any new months and keeps the rest — ~1-2
fits instead of ~30. Use `--full` to rebuild the whole book from scratch. Reads data + v3 lock READ-ONLY; writes
locks/prod_direction_v1/. Does NOT touch v1/v2/v3.

    python -m koscine3.largemove.direction_v1            # incremental (fast)
    python -m koscine3.largemove.direction_v1 --full     # full monthly walk-forward rebuild
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
LOCK_V3 = ROOT / "locks" / "prod_largemove_v3"
LOCK = ROOT / "locks" / "prod_direction_v1"
BOOK_PATH = LOCK / "direction_v1_book_B.csv"

VERSION = "prod_direction_v1"
GROUP = "B_turn35"
H = 5                         # forward horizon (days)
EMBARGO = 6                   # purge gap (>= H+1)
START = pd.Timestamp("2024-01-01")   # book start (aligns with v3)
MIN_UNDERLYING = 100.0
CB = dict(iterations=400, depth=5, learning_rate=0.03, l2_leaf_reg=6.0, random_seed=7,
          allow_writing_files=False, verbose=False, task_type="GPU", devices="0")
LEAK = ("future", "fwd", "next", "ahead", "label", "adverse", "up_move", "down_move", "expansion",
        "volclean", "outcome", "entry_1d", "_date", "tomorrow")
ID = {"date", "symbol", "open", "high", "low", "close", "last", "prev_close", "volume", "group",
      "in_univ", "eligible", "y", "fwd_ret"}
COLS = ["date", "horizon", "group", "symbol", "rank", "iv_group", "expensive", "atm_iv", "atm2_contracts",
        "c_prem", "p_prem", "move_mag", "p_up", "lean", "dir_conf", "dir_pctile", "fwd_ret", "lean_correct", "live"]


def load_panel():
    g2 = {s: g for g, syms in json.loads((LOCK_V2 / "universe_groups.json").read_text()).items() for s in syms}
    m = load_market_data()
    m["symbol"] = m.symbol.astype(str); m["date"] = pd.to_datetime(m["date"])
    m = m.sort_values(["symbol", "date"]).reset_index(drop=True)
    g = m.groupby("symbol", sort=False)
    m["fwd_ret"] = g["close"].shift(-H) / m.close - 1.0
    m["y"] = (m.fwd_ret > 0).astype("float"); m.loc[m.fwd_ret.isna(), "y"] = np.nan
    m["group"] = m.symbol.map(g2)
    m["eligible"] = m.close.ge(MIN_UNDERLYING) & m.atm_iv.notna()
    feats = [c for c in m.columns if c not in ID and pd.api.types.is_numeric_dtype(m[c])
             and not any(h in c.lower() for h in LEAK)]
    return m, feats, g2


def scores_for_months(m, feats, months):
    """Monthly-retrained expanding-window P(up) for group B, only for the given months."""
    from catboost import CatBoostClassifier
    parts = []
    for mo in months:
        ms, me = mo.start_time, mo.end_time
        cut = ms - pd.Timedelta(days=EMBARGO)
        tr = m[(m.date < cut) & m.eligible & m.y.notna()]
        ev = m[(m.date >= ms) & (m.date <= me) & m.eligible & (m.group == GROUP)]
        if len(tr) < 5000 or ev.empty:
            continue
        mdl = CatBoostClassifier(**CB, loss_function="Logloss").fit(tr[feats], tr.y.astype(int))
        e = ev[["date", "symbol"]].copy()
        e["p_up"] = mdl.predict_proba(ev[feats])[:, 1]
        parts.append(e)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["date", "symbol", "p_up"])


def build_rows(m, scores, months):
    """v3 group-B picks in `months`, tagged with the directional lean."""
    bk = pd.read_csv(LOCK_V3 / "mover_v3_book_5d.csv", parse_dates=["date"])
    bk = bk[bk.group == GROUP].copy(); bk["symbol"] = bk.symbol.astype(str)
    bk = bk[bk.date.dt.to_period("M").isin(set(months))]
    bk = bk.merge(scores, on=["date", "symbol"], how="left")
    bk = bk.merge(m[["date", "symbol", "fwd_ret"]], on=["date", "symbol"], how="left")
    bk["lean"] = np.where(bk.p_up >= 0.5, "CALL", np.where(bk.p_up.notna(), "PUT", "n/a"))
    bk["dir_conf"] = (bk.p_up - 0.5).abs() * 2.0
    bk["dir_pctile"] = bk.groupby("date")["p_up"].rank(pct=True)
    bk["live"] = bk.fwd_ret.isna()
    bk["lean_correct"] = np.where(bk.fwd_ret.notna() & bk.p_up.notna(),
                                  ((bk.p_up >= 0.5) & (bk.fwd_ret > 0)) | ((bk.p_up < 0.5) & (bk.fwd_ret < 0)), np.nan)
    return bk.reindex(columns=COLS)


def main(full: bool = False):
    from sklearn.metrics import roc_auc_score
    LOCK.mkdir(parents=True, exist_ok=True)
    m, feats, g2 = load_panel()
    latest = m.date.max().to_period("M")

    existing = pd.read_csv(BOOK_PATH, parse_dates=["date"]) if (BOOK_PATH.exists() and not full) else None
    if existing is None or existing.empty:
        recompute_from, keep = START.to_period("M"), pd.DataFrame(columns=COLS)
        mode = "FULL"
    else:
        existing["symbol"] = existing.symbol.astype(str)
        existing["_mo"] = existing.date.dt.to_period("M")
        live_mos = existing.loc[existing.live == True, "_mo"]            # noqa: E712
        recompute_from = (min(live_mos) if len(live_mos) else existing["_mo"].max() + 1)
        keep = existing[existing["_mo"] < recompute_from].drop(columns=["_mo"]).reindex(columns=COLS)
        mode = "INCREMENTAL"

    months = list(pd.period_range(recompute_from, latest, freq="M"))
    print(f"[{mode}] recompute {months[0] if months else '-'}..{latest} ({len(months)} month(s)) | kept {len(keep)} rows")
    scores = scores_for_months(m, feats, months) if months else pd.DataFrame(columns=["date", "symbol", "p_up"])
    new_rows = build_rows(m, scores, months) if months else pd.DataFrame(columns=COLS)
    book = pd.concat([keep, new_rows], ignore_index=True).sort_values(["date", "rank"]).reset_index(drop=True)
    book.to_csv(BOOK_PATH, index=False)

    done = book[(book.live != True) & book.p_up.notna()].copy()                  # noqa: E712
    done["yr"] = done.date.dt.year.astype(str)
    hit_by_year = {y: round(float(g.lean_correct.mean()), 4) for y, g in done.groupby("yr")}
    auc = round(float(roc_auc_score((done.fwd_ret > 0).astype(int), done.p_up)), 4) if done.fwd_ret.notna().any() else None
    ic = round(float(done.p_up.corr(done.fwd_ret, "spearman")), 4)
    manifest = {
        "version": VERSION,
        "scope": "group B (B_turn35 movers) ONLY — adds PUT/CALL lean to the frozen v3 5d B picks; group A stays v3 agnostic",
        "model": {"type": "CatBoost classifier P(close[t+5]>close[t])", "train_window": "EXPANDING (all history < cut)",
                  "retrain": "MONTHLY (incremental — recomputes live/new months only)", "embargo_days": EMBARGO,
                  "train_universe": "ALL eligible", "features": len(feats), "horizon_days": H,
                  "lean": "CALL if P_up>=0.5 else PUT", "nature": "market-timing/beta tilt — small lean"},
        "warnings": "Do NOT shorten the window (3/6/9m INVERT) or use OI/flow-only (~0 IC). Concentrated monthly "
                    "market-direction bet; EV CI includes 0; validated mainly on 2026 momentum regime.",
        "book_rows": int(len(book)), "done_rows": int(len(done)),
        "dates": [str(book.date.min().date()), str(book.date.max().date())],
        "hit_overall": round(float(done.lean_correct.mean()), 4), "hit_by_year": hit_by_year, "auc_overall": auc, "ic_overall": ic,
        "ref_2026_groupB_OOS": {"auc": 0.578, "hit": 0.550, "ic": 0.171, "atm2_held_ev": 0.049, "source": "experiments/direction_overlay_v1"},
    }
    (LOCK / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (LOCK / "universe_groups.json").write_text((LOCK_V2 / "universe_groups.json").read_text())
    print(f"direction_v1 (group B): {len(book)} rows ({len(done)} done) | hit {manifest['hit_overall']} "
          f"by year {hit_by_year} | AUC {auc} IC {ic}")
    print(f"  -> {BOOK_PATH}\nsaved lock -> {LOCK}")


if __name__ == "__main__":
    main(full="--full" in sys.argv)
