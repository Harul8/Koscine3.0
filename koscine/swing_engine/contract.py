from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


SWING_ENGINE_CONTRACT_VERSION = "independent_swing_peak_5d_v1"
SIDES = ("up", "down")


@dataclass(frozen=True)
class SwingContract:
    horizon_days: int = 5
    near_fraction: float = 0.80
    cost_bps: float = 20.0


def normalize_symbol(symbol: object) -> str:
    return str(symbol).strip().upper()


def load_universe(path: Path) -> pd.DataFrame:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    rows: list[dict] = []
    for tier_name, threshold_key in (("liquid30", "liquid30_threshold"), ("rest35", "rest35_threshold")):
        threshold = float(raw[threshold_key])
        for position, symbol in enumerate(raw[tier_name], start=1):
            rows.append(
                {
                    "symbol": normalize_symbol(symbol),
                    "tier": tier_name,
                    "tier_position": position,
                    "threshold": threshold,
                    "universe_name": raw.get("name", Path(path).stem),
                }
            )
    universe = pd.DataFrame(rows)
    if universe["symbol"].duplicated().any():
        duplicated = sorted(universe.loc[universe["symbol"].duplicated(), "symbol"].unique())
        raise ValueError(f"Universe contains duplicate symbols: {duplicated}")
    return universe


def read_ohlc(dataset_path: Path, symbols: Iterable[str] | None = None) -> pd.DataFrame:
    cols = ["date", "symbol", "open", "high", "low", "close"]
    frame = pd.read_parquet(dataset_path, columns=cols)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame["symbol"] = frame["symbol"].map(normalize_symbol)
    if symbols is not None:
        wanted = {normalize_symbol(symbol) for symbol in symbols}
        frame = frame[frame["symbol"].isin(wanted)].copy()
    numeric = ["open", "high", "low", "close"]
    frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="coerce")
    return frame.sort_values(["symbol", "date"]).reset_index(drop=True)


def add_forward_paths(ohlc: pd.DataFrame, contract: SwingContract) -> pd.DataFrame:
    out = ohlc.sort_values(["symbol", "date"]).copy()
    grouped = out.groupby("symbol", group_keys=False)
    h = int(contract.horizon_days)
    out["entry_date"] = grouped["date"].shift(-1)
    out["entry_open"] = grouped["open"].shift(-1)
    out["exit_date"] = grouped["date"].shift(-h)
    out["exit_close"] = grouped["close"].shift(-h)
    out["window_high"] = grouped["high"].transform(lambda s: s.shift(-1).rolling(h, min_periods=h).max().shift(-(h - 1)))
    out["window_low"] = grouped["low"].transform(lambda s: s.shift(-1).rolling(h, min_periods=h).min().shift(-(h - 1)))
    return out


def attach_universe(base: pd.DataFrame, universe: pd.DataFrame) -> pd.DataFrame:
    out = base.copy()
    out["symbol"] = out["symbol"].map(normalize_symbol)
    return out.merge(universe, on="symbol", how="inner")


def expand_sides_with_outcomes(
    base: pd.DataFrame,
    universe: pd.DataFrame,
    contract: SwingContract,
    include_unevaluated: bool = True,
) -> pd.DataFrame:
    attached = attach_universe(base, universe)
    parts = []
    for side in SIDES:
        side_frame = attached.copy()
        side_frame["side"] = side
        parts.append(side_frame)
    out = pd.concat(parts, ignore_index=True)

    entry = pd.to_numeric(out["entry_open"], errors="coerce")
    high = pd.to_numeric(out["window_high"], errors="coerce")
    low = pd.to_numeric(out["window_low"], errors="coerce")
    close = pd.to_numeric(out["exit_close"], errors="coerce")
    threshold = pd.to_numeric(out["threshold"], errors="coerce")
    is_short = out["side"].eq("down")

    out["favorable_move"] = np.where(is_short, entry / low - 1.0, high / entry - 1.0)
    out["signed_close_return"] = np.where(is_short, entry / close - 1.0, close / entry - 1.0)
    out["opposite_move"] = np.where(is_short, high / entry - 1.0, entry / low - 1.0)
    out["net_close_return"] = out["signed_close_return"] - float(contract.cost_bps) / 10000.0

    evaluated = entry.notna() & high.notna() & low.notna() & close.notna() & threshold.notna()
    has_entry = entry.notna()
    out["window_status"] = np.select(
        [evaluated, has_entry],
        ["evaluated", "pending_full_5d_window"],
        default="pending_entry_bar",
    )
    hit = evaluated & pd.Series(out["favorable_move"], index=out.index).ge(threshold)
    near = evaluated & ~hit & pd.Series(out["favorable_move"], index=out.index).ge(float(contract.near_fraction) * threshold)
    opposite = evaluated & ~hit & pd.Series(out["signed_close_return"], index=out.index).lt(0)
    small = evaluated & ~hit & ~near & ~opposite

    out["hit"] = hit
    out["near"] = near
    out["hit_near"] = hit | near
    out["opposite"] = opposite
    out["small"] = small
    out["verdict"] = ""
    out.loc[hit, "verdict"] = "hit"
    out.loc[near, "verdict"] = "near"
    out.loc[opposite, "verdict"] = "opposite"
    out.loc[small, "verdict"] = "small"
    out.loc[out["window_status"].ne("evaluated"), "verdict"] = "pending"

    out["side_rank"] = np.nan
    evaluated_mask = out["window_status"].eq("evaluated")
    if evaluated_mask.any():
        out.loc[evaluated_mask, "side_rank"] = (
            out.loc[evaluated_mask].groupby(["date", "side"])["favorable_move"].rank(method="first", ascending=False)
        )
    out["top5_mover"] = out["side_rank"].le(5)
    out["contract_version"] = SWING_ENGINE_CONTRACT_VERSION
    if not include_unevaluated:
        out = out[out["window_status"].eq("evaluated")].copy()
    return out


def outcome_frame_from_dataset(
    dataset_path: Path,
    universe_path: Path,
    contract: SwingContract | None = None,
    include_unevaluated: bool = True,
) -> pd.DataFrame:
    contract = contract or SwingContract()
    universe = load_universe(universe_path)
    ohlc = read_ohlc(dataset_path, universe["symbol"])
    paths = add_forward_paths(ohlc, contract)
    return expand_sides_with_outcomes(paths, universe, contract, include_unevaluated=include_unevaluated)


def attach_outcomes_to_predictions(
    predictions: pd.DataFrame,
    dataset_path: Path,
    universe_path: Path,
    contract: SwingContract | None = None,
) -> pd.DataFrame:
    if predictions.empty:
        return predictions.copy()
    contract = contract or SwingContract()
    outcome = outcome_frame_from_dataset(dataset_path, universe_path, contract, include_unevaluated=True)
    cols = [
        "date",
        "symbol",
        "side",
        "tier",
        "threshold",
        "entry_date",
        "entry_open",
        "exit_date",
        "exit_close",
        "window_high",
        "window_low",
        "window_status",
        "favorable_move",
        "signed_close_return",
        "opposite_move",
        "net_close_return",
        "hit",
        "near",
        "hit_near",
        "opposite",
        "small",
        "verdict",
        "side_rank",
        "top5_mover",
        "contract_version",
    ]
    left = predictions.copy()
    left["date"] = pd.to_datetime(left["date"], errors="coerce").dt.normalize()
    left["symbol"] = left["symbol"].map(normalize_symbol)
    return left.merge(outcome[cols], on=["date", "symbol", "side"], how="left", suffixes=("", "_outcome"))
