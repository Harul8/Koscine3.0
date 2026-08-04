"""LOCKED production pipeline for the Large-Move engine: dataset, walk-forward, train, predict, rank."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from koscine3.data.sources import load_market_data
from koscine3.outcomes.clean_move_contract import CleanMoveContract, compute_clean_move_outcomes
from koscine3.largemove.config import (
    LargeMoveConfig, PROD, MODELS_DIR, PREDICTIONS_DIR, LOCK_DIR, XGB_CLF_PARAMS, XGB_REG_PARAMS,
)


def _clean(frame: pd.DataFrame, feats) -> pd.DataFrame:
    return frame[list(feats)].replace([np.inf, -np.inf], np.nan).astype(np.float32)


def load_groups(config: LargeMoveConfig = PROD) -> dict[str, list[str]]:
    return json.loads((LOCK_DIR / config.universe_groups_file).read_text())


def load_dataset(config: LargeMoveConfig = PROD) -> pd.DataFrame:
    feats = list(config.features)
    cols = sorted(set(["date", "symbol", "open", "high", "low", "close", "turnover_lacs", "volume", *feats]))
    market = load_market_data(columns=cols)
    oc = compute_clean_move_outcomes(market, universe=None, contract=CleanMoveContract(window_days=config.window_days))
    oc = oc[oc.status.eq("evaluated")][["date", "symbol", "side", "ceiling"]].copy()
    oc["symbol"] = oc["symbol"].astype(str)
    mk = market.copy(); mk["symbol"] = mk["symbol"].astype(str); mk = mk.drop_duplicates(["date", "symbol"])
    keep = list(dict.fromkeys([*feats, "close"]))
    df = oc.merge(mk[["date", "symbol", *keep]], on=["date", "symbol"], how="left")
    groups = load_groups(config)
    g2 = {s: g for g, syms in groups.items() for s in syms}
    df["group"] = df["symbol"].map(g2)
    df["eligible"] = df["close"].ge(config.min_underlying)
    if config.requires_optionable:
        df["eligible"] &= df["atm_iv"].notna()
    df["year"] = df["date"].dt.year
    return df.reset_index(drop=True)


def _fit_side(base: pd.DataFrame, calib: pd.DataFrame, feats, thr: float, clf_params, reg_params):
    from xgboost import XGBClassifier, XGBRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.frozen import FrozenEstimator
    imp = SimpleImputer(strategy="median").fit(_clean(base, feats)); Xb = imp.transform(_clean(base, feats))
    yb = (base["ceiling"] >= thr).astype(int); spw = (len(yb) - yb.sum()) / max(1, yb.sum())
    clf = XGBClassifier(scale_pos_weight=spw, **clf_params).fit(Xb, yb)
    cal = CalibratedClassifierCV(FrozenEstimator(clf), method="isotonic").fit(
        imp.transform(_clean(calib, feats)), (calib["ceiling"] >= thr).astype(int))
    reg = XGBRegressor(**reg_params).fit(Xb, base["ceiling"].clip(0, 0.5))
    return {"imputer": imp, "clf": cal, "reg": reg}


def _score(models_side, frame, feats) -> tuple[np.ndarray, np.ndarray]:
    X = models_side["imputer"].transform(_clean(frame, feats))
    return models_side["clf"].predict_proba(X)[:, 1], np.clip(models_side["reg"].predict(X), 0, None)


def walk_forward(config: LargeMoveConfig = PROD, df: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
    """Out-of-sample predictions (the validation artifact). base<T-1, calibrate T-1, predict T."""
    df = load_dataset(config) if df is None else df
    feats = list(config.features); out = {}
    for group, thr in config.group_thresholds:
        preds = []
        for T in config.test_years:
            base, calib = df[df.year < T - 1], df[df.year == T - 1]
            ev = df[(df.year == T) & df.eligible & df.group.eq(group) & (df.date <= pd.Timestamp(config.eval_end))].copy()
            if ev.empty: continue
            ev["confidence"] = np.nan; ev["exp_move"] = np.nan
            for side in ("long", "short"):
                b, c, m = base[base.side.eq(side)], calib[calib.side.eq(side)], ev.side.eq(side)
                if b.empty or c.empty or not m.any(): continue
                ms = _fit_side(b, c, feats, thr, XGB_CLF_PARAMS, XGB_REG_PARAMS)
                conf, mag = _score(ms, ev[m], feats)
                ev.loc[m, "confidence"] = conf; ev.loc[m, "exp_move"] = mag
            ev = ev.dropna(subset=["confidence"]); ev["hit"] = (ev["ceiling"] >= thr).astype(int)
            ev["rank_in_day"] = ev.groupby("date")["confidence"].rank(ascending=False, method="first")
            preds.append(ev)
        p = pd.concat(preds, ignore_index=True)
        p["dir"] = np.where(p.side.eq("long"), "CALL", "PUT")
        p["confidence"] = p["confidence"].round(3); p["exp_move_%"] = (p["exp_move"] * 100).round(1)
        p["actual_move_%"] = (p["ceiling"] * 100).round(2); p["threshold"] = thr
        out[group] = p[["date", "group", "symbol", "side", "dir", "confidence", "exp_move_%",
                        "actual_move_%", "hit", "rank_in_day", "year", "threshold"]]
    return out


def train_production(config: LargeMoveConfig = PROD, df: pd.DataFrame | None = None) -> None:
    """Fit on ALL available data (base = all but most recent year for calibration), save models for going-forward scoring."""
    from koscine3.largemove.config import XGB_CLF_PARAMS, XGB_REG_PARAMS
    df = load_dataset(config) if df is None else df
    feats = list(config.features); MODELS_DIR.mkdir(parents=True, exist_ok=True)
    cal_year = int(df["year"].max())
    base, calib = df[df.year < cal_year], df[df.year == cal_year]
    for group, thr in config.group_thresholds:
        for side in ("long", "short"):
            b, c = base[base.side.eq(side)], calib[calib.side.eq(side)]
            ms = _fit_side(b, c, feats, thr, XGB_CLF_PARAMS, XGB_REG_PARAMS)
            joblib.dump(ms, MODELS_DIR / f"{group}_{side}.joblib")
    (MODELS_DIR / "trained_through.txt").write_text(f"{df['date'].max().date()} | calib_year={cal_year}\n")


def predict(df: pd.DataFrame, config: LargeMoveConfig = PROD, on_date: str | None = None) -> pd.DataFrame:
    """Score eligible rows with the saved production models. Returns ranked picks (date, group, dir, confidence, exp_move)."""
    feats = list(config.features); rows = []
    sub = df[df.eligible].copy()
    if on_date:
        sub = sub[sub.date.eq(pd.Timestamp(on_date))]
    for group, thr in config.group_thresholds:
        gsub = sub[sub.group.eq(group)].copy()
        if gsub.empty: continue
        gsub["confidence"] = np.nan; gsub["exp_move"] = np.nan
        for side in ("long", "short"):
            mpath = MODELS_DIR / f"{group}_{side}.joblib"
            if not mpath.exists(): continue
            ms = joblib.load(mpath); m = gsub.side.eq(side)
            if not m.any(): continue
            conf, mag = _score(ms, gsub[m], feats)
            gsub.loc[m, "confidence"] = conf; gsub.loc[m, "exp_move"] = mag
        gsub = gsub.dropna(subset=["confidence"]); gsub["threshold"] = thr
        gsub["rank_in_day"] = gsub.groupby("date")["confidence"].rank(ascending=False, method="first")
        rows.append(gsub)
    p = pd.concat(rows, ignore_index=True)
    p["dir"] = np.where(p.side.eq("long"), "CALL", "PUT")
    p["confidence"] = p["confidence"].round(3); p["exp_move_%"] = (p["exp_move"] * 100).round(1)
    return p[["date", "group", "symbol", "side", "dir", "confidence", "exp_move_%", "rank_in_day", "threshold"]].sort_values(
        ["date", "group", "confidence"], ascending=[True, True, False])


def rank_cooldown(preds: pd.DataFrame, config: LargeMoveConfig = PROD, n_per_day: int = 2) -> pd.DataFrame:
    """Per-group daily top-N with per-stock cooldown (picked t -> repeat allowed at t+cooldown)."""
    cal = {pd.Timestamp(d): i for i, d in enumerate(sorted(preds["date"].unique()))}
    keep = []
    for group, g in preds.groupby("group"):
        g = g.sort_values(["date", "confidence"], ascending=[True, False]); last = {}
        for day, day_g in g.groupby("date", sort=True):
            i = cal[pd.Timestamp(day)]; c = 0
            for idx, sym in zip(day_g.index, day_g["symbol"]):
                if i - last.get(sym, -10**9) < config.cooldown_trading_days: continue
                keep.append(idx); last[sym] = i; c += 1
                if c >= n_per_day: break
    return preds.loc[keep].sort_values(["date", "group", "confidence"], ascending=[True, True, False])
