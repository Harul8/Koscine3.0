from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from koscine.config import BACKTEST_DIR, DEFAULT_COST_BPS, HORIZON_DAYS


@dataclass(frozen=True)
class PortfolioConfig:
    initial_capital: float = 1_000_000.0
    daily_long_slots: int = 5
    daily_short_slots: int = 5
    max_open_long: int = 25
    max_open_short: int = 25
    allocation_pct: float = 0.02
    cost_bps: float = DEFAULT_COST_BPS
    min_score: float = 0.0
    max_adverse_move: float | None = None
    score_proportional_sizing: bool = False
    score_size_floor: float = 0.65
    max_allocation_pct: float = 0.20
    use_atr_stop: bool = False
    atr_stop_mult: float = 1.5
    sector_concentration_max: int = 99
    use_regime_gate: bool = False
    min_rr_ratio: float = 0.0
    rolling_derisk: bool = False
    derisk_window: int = 20
    derisk_warn_avg_pnl: float = 0.0
    derisk_crit_avg_pnl: float = -0.01
    derisk_warn_multiplier: float = 0.50
    derisk_crit_multiplier: float = 0.0


def _score_column(group: pd.DataFrame) -> str:
    values = group["pred_col"].dropna().unique()
    if len(values) != 1:
        raise ValueError("Expected one pred_col value per bucket group")
    return str(values[0])


def make_candidates(predictions: pd.DataFrame, min_score: float = 0.0) -> pd.DataFrame:
    frames = []
    for _, group in predictions.groupby(["side", "threshold"], sort=False):
        part = group.copy()
        pred_cols = part["pred_col"].dropna().unique()
        if len(pred_cols) == 1:
            part["score"] = part[pred_cols[0]]
        else:
            part["score"] = part.apply(lambda row: row[row["pred_col"]], axis=1)
        frames.append(part)
    candidates = pd.concat(frames, ignore_index=True)
    candidates = candidates[candidates["score"].ge(min_score)].copy()
    candidates = candidates.dropna(
        subset=[
            "entry_1d_open",
            f"future_{HORIZON_DAYS}d_high",
            f"future_{HORIZON_DAYS}d_low",
            f"future_{HORIZON_DAYS}d_date",
        ]
    )

    # Keep the strongest threshold signal per side for each stock/date.
    candidates = candidates.sort_values("score", ascending=False)
    candidates = candidates.drop_duplicates(["date", "symbol", "side"], keep="first")

    # If long and short fire on the same stock/date, keep the stronger side.
    candidates = candidates.sort_values("score", ascending=False)
    candidates = candidates.drop_duplicates(["date", "symbol"], keep="first")
    return candidates.sort_values(["date", "side", "score"], ascending=[True, True, False])


def _select_for_date(
    day_candidates: pd.DataFrame,
    active_symbols: set[str],
    open_long_count: int,
    open_short_count: int,
    config: PortfolioConfig,
) -> pd.DataFrame:
    selected = []
    for side, daily_slots, max_open, open_count in (
        ("up", config.daily_long_slots, config.max_open_long, open_long_count),
        ("down", config.daily_short_slots, config.max_open_short, open_short_count),
    ):
        capacity = max(0, min(daily_slots, max_open - open_count))
        if capacity == 0:
            continue
        side_rows = day_candidates[day_candidates["side"].eq(side)]
        side_rows = side_rows[~side_rows["symbol"].isin(active_symbols)]
        selected.append(side_rows.sort_values("score", ascending=False).head(capacity))
    if not selected:
        return pd.DataFrame(columns=day_candidates.columns)
    return pd.concat(selected, ignore_index=True)


def _trade_return(
    row: pd.Series,
    cost_bps: float,
    use_atr_stop: bool = False,
    atr_stop_mult: float = 1.5,
) -> tuple[float, float, float, float, bool]:
    entry = row["entry_1d_open"]
    threshold = float(row.get("threshold", 0.04))
    atr_pct = row.get("atr_pct_14") if "atr_pct_14" in row.index else None
    stop_dist = None
    if use_atr_stop and atr_pct is not None and pd.notna(atr_pct):
        stop_dist = float(np.clip(atr_stop_mult * float(atr_pct), 0.005, 0.05))
    if row["side"] == "down":
        favorable = 1.0 - row[f"future_{HORIZON_DAYS}d_low"] / entry
        adverse = row[f"future_{HORIZON_DAYS}d_high"] / entry - 1.0
        target_hit = favorable >= threshold
        stop_hit = stop_dist is not None and adverse >= stop_dist
        if target_hit and not (stop_hit and adverse > favorable * 1.0):
            gross = threshold
            exit_price = entry * (1.0 - threshold)
        elif stop_hit:
            gross = -stop_dist
            exit_price = entry * (1.0 + stop_dist)
        else:
            gross = -(row[f"future_{HORIZON_DAYS}d_close"] / entry - 1.0)
            exit_price = row[f"future_{HORIZON_DAYS}d_close"]
    else:
        favorable = row[f"future_{HORIZON_DAYS}d_high"] / entry - 1.0
        adverse = 1.0 - row[f"future_{HORIZON_DAYS}d_low"] / entry
        target_hit = favorable >= threshold
        stop_hit = stop_dist is not None and adverse >= stop_dist
        if target_hit and not (stop_hit and adverse > favorable * 1.0):
            gross = threshold
            exit_price = entry * (1.0 + threshold)
        elif stop_hit:
            gross = -stop_dist
            exit_price = entry * (1.0 - stop_dist)
        else:
            gross = row[f"future_{HORIZON_DAYS}d_close"] / entry - 1.0
            exit_price = row[f"future_{HORIZON_DAYS}d_close"]
    net = gross - cost_bps / 10000.0
    return float(gross), float(net), float(exit_price), float(adverse), bool(target_hit and not stop_hit)


def run_portfolio_backtest(
    predictions: pd.DataFrame,
    config: PortfolioConfig,
    output_dir: Path = BACKTEST_DIR,
    name: str = "portfolio",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    from koscine.signal_card import attach_sector

    candidates = make_candidates(predictions, min_score=config.min_score)
    if config.use_regime_gate:
        from koscine.regime import apply_regime_gate

        candidates = apply_regime_gate(candidates)
        candidates = candidates[candidates.get("passes_regime_gate", True)].copy()

    if config.sector_concentration_max < 99:
        candidates = attach_sector(candidates)
        candidates = candidates.sort_values(["date", "score"], ascending=[True, False])
        candidates["_sector_rank"] = (
            candidates.groupby(["date", "sector"]).cumcount() + 1
        )
        candidates = candidates[candidates["_sector_rank"].le(config.sector_concentration_max)].copy()
        candidates = candidates.drop(columns=["_sector_rank"])

    dates = sorted(pd.to_datetime(candidates["date"]).dropna().unique())
    active: list[dict] = []
    closed: list[dict] = []
    equity_rows = []
    cash = config.initial_capital

    for date in dates:
        still_active = []
        for position in active:
            if pd.Timestamp(position["exit_date"]) <= pd.Timestamp(date):
                cash += position["notional"] * (1.0 + position["net_return"])
                position["realized_date"] = pd.Timestamp(date)
                closed.append(position)
            else:
                still_active.append(position)
        active = still_active

        active_symbols = {str(pos["symbol"]) for pos in active}
        open_long_count = sum(1 for pos in active if pos["side"] == "up")
        open_short_count = sum(1 for pos in active if pos["side"] == "down")
        day_candidates = candidates[candidates["date"].eq(date)]
        selected = _select_for_date(
            day_candidates,
            active_symbols=active_symbols,
            open_long_count=open_long_count,
            open_short_count=open_short_count,
            config=config,
        )

        for _, row in selected.iterrows():
            if config.score_proportional_sizing:
                score = float(row.get("score", 0.0))
                scaling = max(0.0, min(1.0, (score - config.score_size_floor) / max(1e-6, 1.0 - config.score_size_floor)))
                pct = config.allocation_pct * (1.0 + scaling)
                pct = min(pct, config.max_allocation_pct)
            else:
                pct = config.allocation_pct

            if config.rolling_derisk and closed:
                model_id = row.get("model_id", "all")
                recent = [c for c in closed if c.get("model_id") == model_id][-config.derisk_window:]
                if len(recent) >= max(5, config.derisk_window // 2):
                    recent_pnl = sum(c.get("net_return", 0.0) for c in recent) / len(recent)
                    if recent_pnl < config.derisk_crit_avg_pnl:
                        pct *= config.derisk_crit_multiplier
                    elif recent_pnl < config.derisk_warn_avg_pnl:
                        pct *= config.derisk_warn_multiplier
                if pct <= 1e-6:
                    continue

            notional = config.initial_capital * pct
            if cash < notional:
                continue
            gross_return, net_return, exit_price, adverse_move, target_hit = _trade_return(
                row, config.cost_bps,
                use_atr_stop=config.use_atr_stop,
                atr_stop_mult=config.atr_stop_mult,
            )
            if config.max_adverse_move is not None and adverse_move > config.max_adverse_move:
                continue
            if config.min_rr_ratio > 0 and "atr_pct_14" in row.index and pd.notna(row.get("atr_pct_14")):
                stop_dist = max(0.005, min(0.05, config.atr_stop_mult * float(row["atr_pct_14"])))
                rr = float(row.get("threshold", 0.04)) / stop_dist
                if rr < config.min_rr_ratio:
                    continue
            hit_value = row["actual"] if "actual" in row.index else row[row["label_col"]]
            cash -= notional
            active.append(
                {
                    "signal_date": pd.Timestamp(row["date"]),
                    "entry_date": pd.Timestamp(row["entry_1d_date"]),
                    "exit_date": pd.Timestamp(row[f"future_{HORIZON_DAYS}d_date"]),
                    "symbol": row["symbol"],
                    "side": row["side"],
                    "tier": row.get("tier"),
                    "model_id": row.get("model_id"),
                    "threshold": row["threshold"],
                    "score": row["score"],
                    "notional": notional,
                    "position_pct": pct,
                    "signal_close": row["close"],
                    "entry_open": row["entry_1d_open"],
                    "exit_price": exit_price,
                    "future_close": row[f"future_{HORIZON_DAYS}d_close"],
                    "adverse_move": adverse_move,
                    "gross_return": gross_return,
                    "net_return": net_return,
                    "target_hit": target_hit,
                    "hit": bool(hit_value == 1),
                }
            )

        open_notional = sum(pos["notional"] for pos in active)
        equity = cash + open_notional
        equity_rows.append(
            {
                "date": pd.Timestamp(date),
                "cash": cash,
                "open_notional": open_notional,
                "equity": equity,
                "open_positions": len(active),
                "open_long": sum(1 for pos in active if pos["side"] == "up"),
                "open_short": sum(1 for pos in active if pos["side"] == "down"),
            }
        )

    for position in active:
        cash += position["notional"] * (1.0 + position["net_return"])
        position["realized_date"] = pd.Timestamp(position["exit_date"])
        closed.append(position)

    trades = pd.DataFrame(closed)
    equity_curve = pd.DataFrame(equity_rows)
    summary = summarize_portfolio(trades, equity_curve, config)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trades.to_parquet(output_dir / f"{name}_portfolio_trades.parquet", index=False)
    equity_curve.to_csv(output_dir / f"{name}_equity_curve.csv", index=False)
    summary.to_csv(output_dir / f"{name}_portfolio_summary.csv", index=False)
    return trades, equity_curve, summary


def summarize_portfolio(
    trades: pd.DataFrame,
    equity_curve: pd.DataFrame,
    config: PortfolioConfig,
) -> pd.DataFrame:
    if equity_curve.empty:
        total_return = np.nan
        max_drawdown = np.nan
    else:
        equity = equity_curve["equity"]
        total_return = equity.iloc[-1] / config.initial_capital - 1.0
        max_drawdown = (equity / equity.cummax() - 1.0).min()

    rows = [
        {
            "scope": "portfolio",
            "trades": len(trades),
            "hit_rate": trades["hit"].mean() if len(trades) else np.nan,
            "avg_net_return": trades["net_return"].mean() if len(trades) else np.nan,
            "median_net_return": trades["net_return"].median() if len(trades) else np.nan,
            "win_rate_net": trades["net_return"].gt(0).mean() if len(trades) else np.nan,
            "total_return": total_return,
            "max_drawdown": max_drawdown,
            "final_equity": equity_curve["equity"].iloc[-1] if len(equity_curve) else np.nan,
            "daily_long_slots": config.daily_long_slots,
            "daily_short_slots": config.daily_short_slots,
            "max_open_long": config.max_open_long,
            "max_open_short": config.max_open_short,
            "allocation_pct": config.allocation_pct,
            "cost_bps": config.cost_bps,
            "min_score": config.min_score,
            "max_adverse_move": config.max_adverse_move,
        }
    ]

    if len(trades):
        for side, group in trades.groupby("side"):
            rows.append(
                {
                    "scope": side,
                    "trades": len(group),
                    "hit_rate": group["hit"].mean(),
                    "avg_net_return": group["net_return"].mean(),
                    "median_net_return": group["net_return"].median(),
                    "win_rate_net": group["net_return"].gt(0).mean(),
                    "total_return": np.nan,
                    "max_drawdown": np.nan,
                    "final_equity": np.nan,
                    "daily_long_slots": config.daily_long_slots,
                    "daily_short_slots": config.daily_short_slots,
                    "max_open_long": config.max_open_long,
                    "max_open_short": config.max_open_short,
                    "allocation_pct": config.allocation_pct,
                    "cost_bps": config.cost_bps,
                    "min_score": config.min_score,
                    "max_adverse_move": config.max_adverse_move,
                }
            )
    return pd.DataFrame(rows)


def run_portfolio_sweep(
    predictions: pd.DataFrame,
    daily_slots: tuple[int, ...] = (1, 3, 5, 10),
    min_scores: tuple[float, ...] = (0.0, 0.5, 0.55, 0.6),
    max_adverse_moves: tuple[float | None, ...] = (None,),
    cost_bps: float = DEFAULT_COST_BPS,
    output_dir: Path = BACKTEST_DIR,
    name: str = "portfolio_sweep",
) -> pd.DataFrame:
    rows = []
    for slots in daily_slots:
        for min_score in min_scores:
            for max_adverse_move in max_adverse_moves:
                config = PortfolioConfig(
                    daily_long_slots=slots,
                    daily_short_slots=slots,
                    max_open_long=slots * HORIZON_DAYS,
                    max_open_short=slots * HORIZON_DAYS,
                    allocation_pct=1.0 / max(1, slots * HORIZON_DAYS * 2),
                    cost_bps=cost_bps,
                    min_score=min_score,
                    max_adverse_move=max_adverse_move,
                )
                adverse_tag = "none" if max_adverse_move is None else str(max_adverse_move).replace(".", "")
                _, _, summary = run_portfolio_backtest(
                    predictions,
                    config=config,
                    output_dir=output_dir,
                    name=(
                        f"{name}_slots{slots}_score{str(min_score).replace('.', '')}"
                        f"_adverse{adverse_tag}"
                    ),
                )
                portfolio_row = summary[summary["scope"].eq("portfolio")].iloc[0].to_dict()
                rows.append(portfolio_row)
    sweep = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    sweep.to_csv(output_dir / f"{name}_summary.csv", index=False)
    return sweep.sort_values(["total_return", "max_drawdown"], ascending=[False, False])
