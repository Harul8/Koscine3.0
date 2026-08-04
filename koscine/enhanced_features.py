from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from koscine.config import HORIZON_DAYS, SILVER_DATA_ROOT


def add_oi_buildup_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values(["symbol", "date"]).copy()
    if "fut_oi" not in out or "fut_chg_oi" not in out:
        return out
    fut_oi_lag = out.groupby("symbol", group_keys=False)["fut_oi"].shift(1)
    out["oi_buildup_ratio"] = out["fut_chg_oi"] / (fut_oi_lag.replace(0, np.nan))
    out["oi_buildup_ratio"] = out["oi_buildup_ratio"].clip(-2.0, 2.0)

    if "ret_1d" in out:
        long_buildup = (out["oi_buildup_ratio"] > 0.02) & (out["ret_1d"] > 0)
        short_buildup = (out["oi_buildup_ratio"] > 0.02) & (out["ret_1d"] < 0)
        long_unwind = (out["oi_buildup_ratio"] < -0.02) & (out["ret_1d"] > 0)
        short_unwind = (out["oi_buildup_ratio"] < -0.02) & (out["ret_1d"] < 0)
        out["oi_long_buildup"] = long_buildup.astype(float)
        out["oi_short_buildup"] = short_buildup.astype(float)
        out["oi_long_unwind"] = long_unwind.astype(float)
        out["oi_short_unwind"] = short_unwind.astype(float)
        grouped = out.groupby("symbol", group_keys=False)
        for col in ("oi_long_buildup", "oi_short_buildup", "oi_long_unwind", "oi_short_unwind"):
            out[f"{col}_5d"] = grouped[col].transform(
                lambda s: s.rolling(5, min_periods=2).sum()
            )
    return out


def add_iv_skew_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "atm_ce_iv" in out and "atm_pe_iv" in out:
        out["iv_skew_ce_minus_pe"] = out["atm_ce_iv"] - out["atm_pe_iv"]
        denom = (out["atm_ce_iv"] + out["atm_pe_iv"]).replace(0, np.nan) / 2.0
        out["iv_skew_norm"] = out["iv_skew_ce_minus_pe"] / denom
    if "put_call_iv_skew" in out:
        grouped = out.sort_values(["symbol", "date"]).groupby("symbol", group_keys=False)
        out["iv_skew_chg_5d"] = grouped["put_call_iv_skew"].diff(5)
    return out


def add_gap_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values(["symbol", "date"]).copy()
    grouped = out.groupby("symbol", group_keys=False)
    prev_close = grouped["close"].shift(1)
    gap = out["open"] / prev_close - 1.0
    out["gap_pct"] = gap
    out["gap_up_flag"] = gap.gt(0.005).astype(float)
    out["gap_down_flag"] = gap.lt(-0.005).astype(float)
    for window in (10, 20):
        out[f"gap_up_count_{window}d"] = grouped["gap_up_flag"].transform(
            lambda s, w=window: s.rolling(w, min_periods=3).sum()
        )
        out[f"gap_down_count_{window}d"] = grouped["gap_down_flag"].transform(
            lambda s, w=window: s.rolling(w, min_periods=3).sum()
        )
    out["intraday_body_pct"] = (out["close"] - out["open"]) / out["open"]
    return out


def add_sector_relative_features(
    df: pd.DataFrame,
    sector_index_map: dict[str, str] | None = None,
    silver_root: Path = SILVER_DATA_ROOT,
) -> pd.DataFrame:
    from koscine.clean_direction import LIQUID30_SECTOR_INDEX, normalize_symbol

    mapping = sector_index_map or LIQUID30_SECTOR_INDEX
    indices_path = silver_root / "indices.parquet"
    if not indices_path.exists():
        return df
    idx = pd.read_parquet(indices_path)
    idx["date"] = pd.to_datetime(idx["date"])
    idx["index_key"] = idx["index_name"].astype(str).str.upper().str.strip()
    needed = set(mapping.values())
    idx = idx[idx["index_key"].isin(needed)].copy()
    if idx.empty:
        return df

    sector_frames = []
    for key, group in idx.groupby("index_key"):
        group = group.sort_values("date").copy()
        close = pd.to_numeric(group["close"], errors="coerce")
        group["sector_close"] = close
        group["sector_ret_1d"] = close.pct_change()
        group["sector_ret_5d"] = close.pct_change(5)
        group["sector_ret_20d"] = close.pct_change(20)
        group["sector_vol_20"] = group["sector_ret_1d"].rolling(20, min_periods=10).std()
        group["sector_close_sma50_dist"] = close / close.rolling(50, min_periods=25).mean() - 1.0
        sector_frames.append(group[
            ["date", "index_key", "sector_ret_1d", "sector_ret_5d", "sector_ret_20d",
             "sector_vol_20", "sector_close_sma50_dist"]
        ])
    sector_long = pd.concat(sector_frames, ignore_index=True)

    sym_to_idx = {normalize_symbol(k): v for k, v in mapping.items()}
    out = df.copy()
    out["_sector_index"] = out["symbol"].map(sym_to_idx)
    merged = out.merge(
        sector_long.rename(columns={"index_key": "_sector_index"}),
        on=["date", "_sector_index"],
        how="left",
    )
    if "ret_5d" in merged and "sector_ret_5d" in merged:
        merged["stock_rel_sector_ret_5d"] = merged["ret_5d"] - merged["sector_ret_5d"]
    if "ret_20d" in merged and "sector_ret_20d" in merged:
        merged["stock_rel_sector_ret_20d"] = merged["ret_20d"] - merged["sector_ret_20d"]
    return merged.drop(columns=["_sector_index"])


def add_earnings_proximity_features(
    df: pd.DataFrame,
    silver_root: Path = SILVER_DATA_ROOT,
) -> pd.DataFrame:
    from koscine.clean_direction import normalize_symbol

    out = df.copy()
    earnings_path = silver_root / "earnings.parquet"
    if not earnings_path.exists():
        out["days_to_earnings"] = np.nan
        out["earnings_within_5d"] = 0.0
        out["earnings_within_10d"] = 0.0
        return out

    earn = pd.read_parquet(earnings_path)
    if earn.empty:
        out["days_to_earnings"] = np.nan
        out["earnings_within_5d"] = 0.0
        out["earnings_within_10d"] = 0.0
        return out

    earn = earn.copy()
    earn["date"] = pd.to_datetime(earn["date"]).dt.normalize()
    earn["symbol"] = earn["symbol"].map(normalize_symbol)
    earn = earn.dropna(subset=["symbol", "date"]).drop_duplicates(["symbol", "date"])

    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    out = out.sort_values(["symbol", "date"]).reset_index(drop=True)

    pieces = []
    earn_by_sym = {sym: g["date"].sort_values().values for sym, g in earn.groupby("symbol")}
    for sym, group in out.groupby("symbol", sort=False):
        dates = group["date"].values
        future = earn_by_sym.get(sym)
        if future is None or len(future) == 0:
            days = np.full(len(dates), np.nan, dtype=float)
        else:
            idx = np.searchsorted(future, dates, side="left")
            days = np.full(len(dates), np.nan, dtype=float)
            valid = idx < len(future)
            days[valid] = (future[idx[valid]] - dates[valid]).astype("timedelta64[D]").astype(float)
        piece = group.copy()
        piece["days_to_earnings"] = days
        pieces.append(piece)
    out = pd.concat(pieces, ignore_index=True)
    out["earnings_within_5d"] = (out["days_to_earnings"].le(5) & out["days_to_earnings"].ge(0)).astype(float)
    out["earnings_within_10d"] = (out["days_to_earnings"].le(10) & out["days_to_earnings"].ge(0)).astype(float)
    out["earnings_in_horizon"] = (
        out["days_to_earnings"].le(HORIZON_DAYS) & out["days_to_earnings"].ge(0)
    ).astype(float)
    return out


def _consec_streak(series: pd.Series, direction: str) -> pd.Series:
    if direction == "up":
        flag = (series > 0).astype(int)
    else:
        flag = (series < 0).astype(int)
    reset_groups = (flag == 0).cumsum()
    return flag.groupby(reset_groups).cumsum()


def add_momentum_persistence_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values(["symbol", "date"]).copy()
    if "ret_1d" not in out:
        return out
    grouped = out.groupby("symbol", group_keys=False)
    out["consec_up_days"] = grouped["ret_1d"].transform(lambda s: _consec_streak(s, "up"))
    out["consec_down_days"] = grouped["ret_1d"].transform(lambda s: _consec_streak(s, "down"))
    out["pos_day_share_10d"] = grouped["ret_1d"].transform(
        lambda s: (s > 0).rolling(10, min_periods=5).mean()
    )
    out["pos_day_share_20d"] = grouped["ret_1d"].transform(
        lambda s: (s > 0).rolling(20, min_periods=10).mean()
    )
    return out


def add_signal_age_features(df: pd.DataFrame, score_col: str = "score") -> pd.DataFrame:
    if score_col not in df:
        return df
    out = df.sort_values(["symbol", "date"]).copy()
    grouped = out.groupby(["symbol", "side"], group_keys=False) if "side" in out else out.groupby("symbol", group_keys=False)
    prev_score = grouped[score_col].shift(1)
    out["score_delta_1d"] = out[score_col] - prev_score
    out["score_max_5d"] = grouped[score_col].transform(
        lambda s: s.rolling(5, min_periods=1).max()
    )
    out["score_fresh_signal"] = (
        prev_score.lt(out[score_col]).fillna(False).astype(float)
        * out["score_delta_1d"].abs().gt(0.05).fillna(False).astype(float)
    )
    return out


def enrich_dataset(
    df: pd.DataFrame,
    silver_root: Path = SILVER_DATA_ROOT,
    add_oi: bool = True,
    add_skew: bool = True,
    add_gaps: bool = True,
    add_sector: bool = True,
    add_earnings: bool = True,
    add_persistence: bool = True,
) -> pd.DataFrame:
    out = df.copy()
    if add_oi:
        out = add_oi_buildup_features(out)
    if add_skew:
        out = add_iv_skew_features(out)
    if add_gaps:
        out = add_gap_features(out)
    if add_sector:
        out = add_sector_relative_features(out, silver_root=silver_root)
    if add_earnings:
        out = add_earnings_proximity_features(out, silver_root=silver_root)
    if add_persistence:
        out = add_momentum_persistence_features(out)
    return out.sort_values(["date", "symbol"]).reset_index(drop=True)
