from __future__ import annotations

import numpy as np
import pandas as pd


def _date_rank_score(df: pd.DataFrame, raw_score: pd.Series) -> pd.Series:
    ranks = raw_score.groupby(df["date"]).rank(pct=True, method="average")
    return ranks.fillna(0.5).clip(0, 1)


def make_baseline_predictions(dataset: pd.DataFrame, baseline_id: str) -> pd.DataFrame:
    df = dataset.copy()
    if baseline_id == "liquidity":
        raw = df.get("turnover_ratio_20", pd.Series(1.0, index=df.index)).fillna(1.0)
    elif baseline_id == "momentum":
        ret = df.get("ret_5d", pd.Series(0.0, index=df.index)).fillna(0.0)
        raw = np.where(df["side"].eq("long"), ret, -ret)
        raw = pd.Series(raw, index=df.index)
    elif baseline_id == "compression":
        raw = df.get("compression_composite", pd.Series(0.0, index=df.index)).fillna(0.0)
    elif baseline_id == "random":
        rng = np.random.default_rng(42)
        raw = pd.Series(rng.random(len(df)), index=df.index)
    else:
        raise ValueError(f"Unknown baseline id: {baseline_id}")

    score = _date_rank_score(df, raw)
    out_cols = [
        "date",
        "symbol",
        "side",
        "band",
        "threshold",
        "entry_date",
        "entry_open",
        "window_end_date",
        "status",
        "verdict",
        "hit",
        "near",
        "hit_or_near",
        "opposite",
        "small",
        "favorable_move",
        "signed_close_return",
        "symbol_return_20d_rank_pct",
        "relative_return_20d",
        "hist_opposite_rate_63",
    ]
    out = df[[c for c in out_cols if c in df.columns]].copy()
    out["p_hit_near"] = (0.30 + 0.45 * score).clip(0, 1)
    out["p_opposite"] = (0.60 - 0.35 * score).clip(0, 1)
    out["expected_favorable_move"] = df["threshold"].astype(float) * (0.50 + score)
    out["expected_signed_close_return"] = df["threshold"].astype(float) * (score - 0.50)
    out["model_id"] = f"baseline_{baseline_id}"
    return out
