from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .contract import SwingContract, add_forward_paths, expand_sides_with_outcomes, load_universe, normalize_symbol


IDENTITY_COLUMNS = {
    "date",
    "symbol",
    "tier",
    "tier_position",
    "universe_name",
    "side",
    "threshold",
    "entry_date",
    "entry_open",
    "exit_date",
    "exit_close",
    "window_high",
    "window_low",
    "window_status",
    "verdict",
    "contract_version",
}
TARGET_COLUMNS = {
    "favorable_move",
    "signed_close_return",
    "opposite_move",
    "net_close_return",
    "hit",
    "near",
    "hit_near",
    "opposite",
    "small",
    "side_rank",
    "top5_mover",
    "utility_target",
    "hit_near_target",
    "opposite_target",
    "top5_target",
}
LEAKY_PREFIXES = (
    "future_",
    "label_",
    "entry_",
    "up_move_",
    "down_move_",
    "fwd_return_",
    "long_adverse_",
    "short_adverse_",
    "actual_",
    "target_",
    "prediction_",
    "pred_",
    "swing_",
)
LEAKY_TOKENS = (
    "go_label",
    "production",
    "bucket",
    "verdict",
    "model_id",
    "future",
    "actual",
)
RAW_PRICE_COLUMNS = {"open", "high", "low", "close", "last", "prev_close"}


@dataclass(frozen=True)
class DatasetConfig:
    start_year: int = 2010
    validation_days: int = 365
    drop_raw_prices: bool = True
    max_missing_fraction: float = 0.35


def read_feature_source(dataset_path: Path, universe_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    universe = load_universe(universe_path)
    frame = pd.read_parquet(dataset_path)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame["symbol"] = frame["symbol"].map(normalize_symbol)
    frame = frame[frame["symbol"].isin(set(universe["symbol"]))].copy()
    frame = frame.sort_values(["symbol", "date"]).reset_index(drop=True)
    return frame, universe


def _is_leaky_column(column: str) -> bool:
    lower = column.lower()
    return column.startswith(LEAKY_PREFIXES) or any(token in lower for token in LEAKY_TOKENS)


def directional_columns(columns: list[str]) -> list[str]:
    out = []
    for column in columns:
        lower = column.lower()
        if lower.startswith(("ret_", "rel_ret_", "stock_rel_sector_ret_", "sector_ret_", "nifty_ret_")):
            out.append(column)
        elif lower.endswith("_dist") or lower.endswith("_dist_pct"):
            out.append(column)
        elif lower in {"gap_pct", "iv_skew_ce_minus_pe", "iv_skew_norm", "iv_skew_chg_5d"}:
            out.append(column)
        elif lower.startswith(("consec_up", "consec_down", "pos_day_share")):
            out.append(column)
    return out


def add_independent_side_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    sign = np.where(out["side"].eq("down"), -1.0, 1.0)
    out["side_sign"] = sign
    out["side_is_up"] = out["side"].eq("up").astype(float)
    out["side_is_down"] = out["side"].eq("down").astype(float)
    out["tier_liquid30"] = out["tier"].eq("liquid30").astype(float)
    out["tier_rest35"] = out["tier"].eq("rest35").astype(float)
    numeric = out.select_dtypes(include=[np.number]).columns.tolist()
    for column in directional_columns(numeric):
        out[f"side_{column}"] = pd.to_numeric(out[column], errors="coerce") * sign
    out["threshold_pct"] = pd.to_numeric(out["threshold"], errors="coerce")
    return out


def add_targets(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    threshold = pd.to_numeric(out["threshold"], errors="coerce").replace(0, np.nan)
    favorable_scaled = (pd.to_numeric(out["favorable_move"], errors="coerce") / threshold).clip(lower=-0.50, upper=2.00)
    signed_scaled = (pd.to_numeric(out["signed_close_return"], errors="coerce") / threshold).clip(lower=-1.50, upper=1.50)
    out["hit_near_target"] = out["hit_near"].fillna(False).astype(int)
    out["opposite_target"] = out["opposite"].fillna(False).astype(int)
    out["top5_target"] = out["top5_mover"].fillna(False).astype(int)
    out["utility_target"] = (
        0.70 * favorable_scaled.fillna(0.0)
        + 0.60 * out["hit"].fillna(False).astype(float)
        + 0.32 * out["near"].fillna(False).astype(float)
        + 0.22 * out["top5_mover"].fillna(False).astype(float)
        + 0.18 * signed_scaled.fillna(0.0)
        - 1.05 * out["opposite"].fillna(False).astype(float)
        - 0.08 * out["small"].fillna(False).astype(float)
    )
    return out


def feature_columns(frame: pd.DataFrame, config: DatasetConfig) -> list[str]:
    numeric = frame.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    blocked = set(IDENTITY_COLUMNS) | set(TARGET_COLUMNS)
    if config.drop_raw_prices:
        blocked |= RAW_PRICE_COLUMNS
    candidates = [col for col in numeric if col not in blocked and not _is_leaky_column(col)]
    if not candidates:
        raise ValueError("No leak-safe numeric features available")
    missing = frame[candidates].isna().mean()
    candidates = [col for col in candidates if float(missing[col]) <= config.max_missing_fraction]
    return candidates


def build_swing_table(
    dataset_path: Path,
    universe_path: Path,
    contract: SwingContract | None = None,
    config: DatasetConfig | None = None,
) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    contract = contract or SwingContract()
    config = config or DatasetConfig()
    source, universe = read_feature_source(dataset_path, universe_path)
    paths = add_forward_paths(source, contract)
    table = expand_sides_with_outcomes(paths, universe, contract, include_unevaluated=True)
    table = add_independent_side_features(table)
    table = add_targets(table)
    features = feature_columns(table, config)
    table[features] = table[features].replace([np.inf, -np.inf], np.nan)
    return table, features, universe


def train_validation_split(
    table: pd.DataFrame,
    train_end: pd.Timestamp,
    config: DatasetConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_end = pd.Timestamp(train_end).normalize()
    valid_start = train_end - pd.Timedelta(days=int(config.validation_days))
    eligible = (
        table["window_status"].eq("evaluated")
        & pd.to_datetime(table["exit_date"], errors="coerce").le(train_end)
        & pd.to_datetime(table["date"], errors="coerce").dt.year.ge(int(config.start_year))
    )
    eligible_frame = table[eligible].copy()
    train = eligible_frame[eligible_frame["date"] < valid_start].copy()
    valid = eligible_frame[eligible_frame["date"] >= valid_start].copy()
    if train.empty or valid.empty:
        raise ValueError(f"Empty train/validation split for train_end={train_end.date()}")
    return train, valid


def prediction_slice(table: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    start = pd.Timestamp(start).normalize()
    end = pd.Timestamp(end).normalize()
    return table[table["date"].between(start, end)].copy()
