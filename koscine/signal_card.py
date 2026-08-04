from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from koscine.config import HORIZON_DAYS, PREDICTIONS_DIR


@dataclass(frozen=True)
class SignalCardConfig:
    atr_stop_mult: float = 1.5
    target_threshold_default: float = 0.04
    min_rr_ratio: float = 1.4
    base_allocation_pct: float = 0.10
    max_allocation_pct: float = 0.20
    score_threshold_for_scaling: float = 0.65
    sector_concentration_max: int = 2


SECTOR_MAP_FALLBACK = {
    "HDFCBANK": "Banking",
    "ICICIBANK": "Banking",
    "SBIN": "Banking",
    "AXISBANK": "Banking",
    "KOTAKBANK": "Banking",
    "INDUSINDBK": "Banking",
    "IDFCFIRSTB": "Banking",
    "RBLBANK": "Banking",
    "YESBANK": "Banking",
    "BANKBARODA": "Banking",
    "PNB": "Banking",
    "CANBK": "Banking",
    "BAJFINANCE": "Financials",
    "BAJAJFINSV": "Financials",
    "CHOLAFIN": "Financials",
    "JIOFIN": "Financials",
    "PFC": "Financials",
    "RECLTD": "Financials",
    "MUTHOOTFIN": "Financials",
    "SHRIRAMFIN": "Financials",
    "LICHSGFIN": "Financials",
    "SBICARD": "Financials",
    "HDFCLIFE": "Financials",
    "SBILIFE": "Financials",
    "INFY": "IT",
    "TCS": "IT",
    "WIPRO": "IT",
    "HCLTECH": "IT",
    "TECHM": "IT",
    "PERSISTENT": "IT",
    "COFORGE": "IT",
    "LTM": "IT",
    "RELIANCE": "Energy",
    "ONGC": "Energy",
    "BPCL": "Energy",
    "COALINDIA": "Energy",
    "NTPC": "Power",
    "POWERGRID": "Power",
    "TATAPOWER": "Power",
    "ADANIGREEN": "Power",
    "ADANIENSOL": "Power",
    "JSWENERGY": "Power",
    "INOXWIND": "Power",
    "SUZLON": "Power",
    "WAAREEENER": "Power",
    "MARUTI": "Auto",
    "M&M": "Auto",
    "TMPV": "Auto",
    "BAJAJ-AUTO": "Auto",
    "EICHERMOT": "Auto",
    "HEROMOTOCO": "Auto",
    "TVSMOTOR": "Auto",
    "ASHOKLEY": "Auto",
    "HYUNDAI": "Auto",
    "SUNPHARMA": "Pharma",
    "CIPLA": "Pharma",
    "DIVISLAB": "Pharma",
    "AUROPHARMA": "Pharma",
    "GLENMARK": "Pharma",
    "APOLLOHOSP": "Pharma",
    "MAXHEALTH": "Pharma",
    "HINDUNILVR": "FMCG",
    "ITC": "FMCG",
    "NESTLEIND": "FMCG",
    "COLPAL": "FMCG",
    "PATANJALI": "FMCG",
    "UNITDSPR": "FMCG",
    "JUBLFOOD": "FMCG",
    "VBL": "FMCG",
    "DMART": "Retail",
    "TRENT": "Retail",
    "NAUKRI": "Retail",
    "SWIGGY": "Retail",
    "ETERNAL": "Retail",
    "TATASTEEL": "Metals",
    "JSWSTEEL": "Metals",
    "HINDALCO": "Metals",
    "VEDL": "Metals",
    "HINDZINC": "Metals",
    "SAIL": "Metals",
    "NMDC": "Metals",
    "ULTRACEMCO": "Cement",
    "AMBUJACEM": "Cement",
    "GRASIM": "Cement",
    "ASIANPAINT": "ConsumerDurables",
    "TITAN": "ConsumerDurables",
    "ASTRAL": "ConsumerDurables",
    "CROMPTON": "ConsumerDurables",
    "VOLTAS": "ConsumerDurables",
    "DIXON": "ConsumerDurables",
    "KAYNES": "ConsumerDurables",
    "LT": "Capital Goods",
    "BEL": "Capital Goods",
    "BHEL": "Capital Goods",
    "HAL": "Capital Goods",
    "RVNL": "Capital Goods",
    "BHARTIARTL": "Telecom",
    "IDEA": "Telecom",
    "INDUSTOWER": "Telecom",
    "ADANIENT": "Conglomerate",
    "ADANIPORTS": "Conglomerate",
    "INDIGO": "Aviation",
    "LODHA": "Realty",
    "GODREJPROP": "Realty",
    "OBEROIRLTY": "Realty",
    "DLF": "Realty",
}


def attach_sector(df: pd.DataFrame, sector_map: dict[str, str] | None = None) -> pd.DataFrame:
    mapping = sector_map or SECTOR_MAP_FALLBACK
    out = df.copy()
    out["sector"] = out["symbol"].map(mapping).fillna("Other")
    return out


def attach_atr_stop_target(
    df: pd.DataFrame,
    config: SignalCardConfig | None = None,
) -> pd.DataFrame:
    config = config or SignalCardConfig()
    out = df.copy()
    atr_pct = out.get("atr_pct_14")
    if atr_pct is None:
        atr_pct = pd.Series(np.nan, index=out.index)
    entry = out["entry_1d_open"] if "entry_1d_open" in out else out.get("close")
    threshold = out["threshold"] if "threshold" in out else pd.Series(config.target_threshold_default, index=out.index)
    side = out["side"] if "side" in out else pd.Series("up", index=out.index)
    stop_dist = (config.atr_stop_mult * atr_pct).clip(lower=0.005, upper=0.05)
    target_dist = threshold.astype(float)
    long_mask = side.eq("up")
    out["stop_pct"] = stop_dist
    out["target_pct"] = target_dist
    out["stop_price"] = np.where(
        long_mask,
        entry * (1.0 - stop_dist),
        entry * (1.0 + stop_dist),
    )
    out["target_price"] = np.where(
        long_mask,
        entry * (1.0 + target_dist),
        entry * (1.0 - target_dist),
    )
    out["partial_target_price"] = np.where(
        long_mask,
        entry * (1.0 + 0.6 * target_dist),
        entry * (1.0 - 0.6 * target_dist),
    )
    out["risk_reward_ratio"] = (target_dist / stop_dist.replace(0, np.nan)).round(3)
    out["passes_rr_filter"] = out["risk_reward_ratio"].ge(config.min_rr_ratio).fillna(False)
    return out


def attach_position_size(
    df: pd.DataFrame,
    config: SignalCardConfig | None = None,
) -> pd.DataFrame:
    config = config or SignalCardConfig()
    out = df.copy()
    score_col = "score" if "score" in out.columns else (
        "meta_final_score" if "meta_final_score" in out.columns else None
    )
    if score_col is None:
        out["position_size_pct"] = config.base_allocation_pct
        return out
    scaling = (
        (out[score_col] - config.score_threshold_for_scaling)
        / (1.0 - config.score_threshold_for_scaling)
    ).clip(lower=0.0, upper=1.0)
    sized = config.base_allocation_pct * (1.0 + scaling)
    out["position_size_pct"] = sized.clip(
        lower=config.base_allocation_pct * 0.5,
        upper=config.max_allocation_pct,
    )
    return out


def attach_sector_concentration(
    df: pd.DataFrame,
    config: SignalCardConfig | None = None,
) -> pd.DataFrame:
    config = config or SignalCardConfig()
    out = df.copy()
    if "sector" not in out.columns:
        out = attach_sector(out)
    score_col = "score" if "score" in out.columns else (
        "meta_final_score" if "meta_final_score" in out.columns else None
    )
    if score_col is None:
        out["sector_rank_in_day"] = 1
        out["sector_concentration_ok"] = True
        return out
    out["sector_rank_in_day"] = (
        out.sort_values(score_col, ascending=False)
        .groupby(["date", "sector"])
        .cumcount() + 1
    )
    out["sector_concentration_ok"] = out["sector_rank_in_day"].le(config.sector_concentration_max)
    return out


def build_signal_cards(
    predictions: pd.DataFrame,
    config: SignalCardConfig | None = None,
    include_filters: bool = True,
    attach_regime: bool = True,
) -> pd.DataFrame:
    config = config or SignalCardConfig()
    out = predictions.copy()
    out = attach_sector(out)
    out = attach_atr_stop_target(out, config)
    out = attach_position_size(out, config)
    out = attach_sector_concentration(out, config)

    if attach_regime and "regime" not in out.columns:
        try:
            from koscine.regime import apply_regime_gate

            out = apply_regime_gate(out)
        except Exception:
            pass

    if include_filters:
        gate = out.get("rule_gate_pass", pd.Series(True, index=out.index)).fillna(False).astype(bool)
        rr = out.get("passes_rr_filter", pd.Series(True, index=out.index)).fillna(False).astype(bool)
        conc = out.get("sector_concentration_ok", pd.Series(True, index=out.index)).fillna(False).astype(bool)
        production = out.get("production_signal", pd.Series(False, index=out.index))
        if not isinstance(production, pd.Series):
            production = pd.Series(False, index=out.index)
        production = production.fillna(False).astype(bool)
        if not production.any():
            score_col = "score" if "score" in out.columns else (
                "meta_final_score" if "meta_final_score" in out.columns else None
            )
            if score_col is not None:
                threshold_floor = out.get("regime_score_floor", pd.Series(np.nan, index=out.index))
                fallback_floor = threshold_floor.fillna(0.65)
                production = out[score_col].ge(fallback_floor)
        regime_pass = out.get("passes_regime_gate", pd.Series(True, index=out.index)).fillna(True).astype(bool)
        out["actionable"] = production & gate & rr & conc & regime_pass
    return out


def render_signal_card(row: pd.Series) -> str:
    side_label = "LONG" if str(row.get("side")) == "up" else "SHORT"
    tier = row.get("tier", "")
    threshold_pct = float(row.get("threshold", 0.0)) * 100
    score = float(row.get("score", row.get("meta_final_score", 0.0)))
    entry = float(row.get("entry_1d_open", row.get("close", 0.0)))
    stop = float(row.get("stop_price", 0.0))
    target = float(row.get("target_price", 0.0))
    partial = float(row.get("partial_target_price", 0.0))
    rr = float(row.get("risk_reward_ratio", 0.0))
    pos_size = float(row.get("position_size_pct", 0.0)) * 100
    gate_pass = bool(row.get("rule_gate_pass", False))
    sector = row.get("sector", "Other")
    regime = row.get("regime", "n/a")
    reason = row.get("lock_reason", row.get("trade_arbitration_reason", ""))

    lines = [
        f"{row.get('symbol', '?'):<12} | {side_label} | {tier} | target {threshold_pct:.1f}%",
        f"  Date: {row.get('date')}  Sector: {sector}  Regime: {regime}",
        f"  Score: {score:.3f}  Gate: {'PASS' if gate_pass else 'FAIL'}  R:R: {rr:.2f}",
        f"  Entry (next open): ~{entry:.2f}",
        f"  Target: {target:.2f} ({threshold_pct:+.1f}%)   Partial: {partial:.2f}",
        f"  Stop: {stop:.2f}",
        f"  Position size: {pos_size:.1f}% of capital",
        f"  Reason: {reason}",
    ]
    return "\n".join(lines)


def write_signal_cards(
    predictions: pd.DataFrame,
    output_dir: Path = PREDICTIONS_DIR / "signal_cards",
    config: SignalCardConfig | None = None,
    only_actionable: bool = True,
) -> Path:
    cards = build_signal_cards(predictions, config=config)
    if only_actionable and "actionable" in cards:
        cards_to_print = cards[cards["actionable"]].copy()
    else:
        cards_to_print = cards.copy()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cards.to_parquet(output_dir / "signal_cards.parquet", index=False)
    cards.to_csv(output_dir / "signal_cards.csv", index=False)

    if not cards_to_print.empty:
        report_path = output_dir / "signal_cards_readable.txt"
        rendered = "\n\n".join(render_signal_card(row) for _, row in cards_to_print.iterrows())
        report_path.write_text(rendered, encoding="utf-8")
    return output_dir
