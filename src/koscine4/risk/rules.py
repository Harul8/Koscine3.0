"""Position sizing and risk limits (requirements R-1 to R-10, decision A3).

The owner sets every limit in the `[risk]` section of the settings file. This module computes
the maximum loss of a trade, sizes it, refuses trades that break a limit, applies the drawdown
rule, and proposes settings from a backtest for the owner to approve. A proposal is written
to a file; it never changes the settings.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from koscine4.settings import Settings
from koscine4.store.surface import black_price, implied_vol
from koscine4.strategies.structures import SELL, max_loss_per_share, net_debit

STATE_FILE = Path("risk") / "state.json"
PROPOSAL_FILE = Path("risk") / "proposal.json"


@dataclass(frozen=True)
class Sized:
    lots: int
    max_loss_per_lot: float          # rupees
    max_loss: float                  # rupees, for all lots
    margin_estimate: float           # rupees; 0 for bought options
    refused: str = ""                # reason, when lots is 0


def max_loss_per_lot(legs, prices, lot: float) -> float:
    """Rupees one lot can lose, before costs (R-1)."""
    return max_loss_per_share(legs, prices) * lot


def margin_estimate(legs, prices, lot: float, margin_factor: float) -> float:
    """Fallback margin per lot for a trade with a sold leg: maximum loss x factor (R-6). Bought options need no margin."""
    if not any(l.side == SELL for l in legs):
        return 0.0
    return max_loss_per_lot(legs, prices, lot) * margin_factor


def scenario_margin(legs, prices, lot: float, spot: float, date, cfg, book: str, margin_factor: float) -> tuple[float, dict]:
    """Estimated margin per lot (R-6), in the manner of the exchange's SPAN method: what the position would owe
    at the worst of a grid of price and volatility moves, plus exposure margin on the sold legs' notional value.

    As in SPAN, the premium received is not netted off: the requirement is the worst-case loss plus what the
    position owes now (SPAN subtracts the net option value, which is negative for a credit). Each leg is
    revalued with Black's formula at its own implied volatility, worked out from its reference price; the
    index level stands in for the forward. The price scan comes from those volatilities, never less than a
    floor. The [margin] settings were fitted to Zerodha's F&O calculator on 2026-11-18 prices. When a
    volatility cannot be worked out, the fallback is the most the legs can owe at expiry (a spread's width)
    x `margin_factor`, plus exposure. Bought options only: no margin. `cfg` is the [margin] section.
    """
    if not any(l.side == SELL for l in legs):
        return 0.0, {"method": "bought options only: no margin"}
    side = np.array([l.side for l in legs], dtype=float)
    sold = float(-side[side < 0].sum())
    rate = float(cfg["exposure_pct_index" if book == "index" else "exposure_pct_stock"]) / 100.0
    exposure = rate * float(spot) * sold if spot and spot > 0 else 0.0
    owed_at_expiry = max(max_loss_per_share(legs, prices) - net_debit(legs, prices), 0.0)
    fallback = (owed_at_expiry * margin_factor + exposure) * lot
    date = pd.Timestamp(date)
    t = np.array([max((l.expiry - date).days, 0.5) / 365.0 for l in legs])
    strikes = np.array([l.strike for l in legs], dtype=float)
    calls = np.array([l.opt_type == "CE" for l in legs])
    px = np.array(prices, dtype=float)
    iv = implied_vol(px, np.full(len(legs), float(spot)), strikes, t, calls) if spot and spot > 0 else np.full(len(legs), np.nan)
    if not np.isfinite(iv).all():
        return fallback, {"method": f"fallback: the most the legs can owe at expiry x {margin_factor:g}, plus exposure "
                                    "(a leg's implied volatility could not be worked out)"}
    daily_sd = float(iv.mean()) / math.sqrt(252.0)
    floor = float(cfg["min_price_scan_pct_index" if book == "index" else "min_price_scan_pct_stock"]) / 100.0
    scan = max(float(cfg["price_scan_sigmas"]) * daily_sd * math.sqrt(float(cfg["margin_period_days"])), floor)
    vol_moves = (1.0 - float(cfg["vol_scan_fraction"]), 1.0 + float(cfg["vol_scan_fraction"]))
    value_now = float((side * black_price(spot, strikes, t, iv, calls)).sum())

    def loss(move: float, vol_mult: float) -> float:
        value = float((side * black_price(spot * (1.0 + move), strikes, t, iv * vol_mult, calls)).sum())
        return value_now - value

    worst, where = 0.0, "none"
    for frac in (-1.0, -2 / 3, -1 / 3, 0.0, 1 / 3, 2 / 3, 1.0):
        for vm in vol_moves:
            l = loss(frac * scan, vm)
            if l > worst:
                worst, where = l, f"price {frac * scan * 100:+.1f}%, volatility x{vm:g}"
    extreme, cover = float(cfg["extreme_move_multiple"]), float(cfg["extreme_move_cover"])
    for sign in (-1.0, 1.0):
        l = cover * loss(sign * extreme * scan, 1.0)
        if l > worst:
            worst, where = l, f"extreme price {sign * extreme * scan * 100:+.1f}% at {cover:g} of its loss"
    span = max(worst - value_now, 0.0)                   # worst loss plus what the position owes now
    per_lot = (span + exposure) * lot
    return per_lot, {"method": "scenario", "price_scan_pct": round(scan * 100, 2), "worst_scenario": where,
                     "span_per_share": round(span, 2), "exposure_per_share": round(exposure, 2),
                     "implied_vols": [round(float(v), 4) for v in iv], "fallback_would_be": round(fallback, 0)}


def read_state(settings: Settings) -> dict:
    path = settings.path("records") / STATE_FILE
    if not path.exists():
        return {"paused": False, "peak_equity": None, "note": ""}
    return json.loads(path.read_text(encoding="utf-8"))


def write_state(settings: Settings, state: dict) -> None:
    path = settings.path("records") / STATE_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def drawdown_rule(risk: dict, state: dict, realised_pnl: float) -> tuple[float, dict]:
    """(size multiplier, new state). Halve after the first loss level, pause after the second (R-4).

    A pause stays until the owner clears it with `resume`.
    """
    capital = float(risk["capital"])
    equity = capital + realised_pnl
    peak = max(equity, float(state.get("peak_equity") or capital))
    drawdown_pct = 100.0 * (peak - equity) / peak if peak > 0 else 0.0
    new = dict(state, peak_equity=peak, drawdown_pct=drawdown_pct)
    if state.get("paused") or drawdown_pct >= float(risk["drawdown_pause_pct"]):
        new["paused"] = True
        return 0.0, new
    return (0.5 if drawdown_pct >= float(risk["drawdown_halve_pct"]) else 1.0), new


def resume(settings: Settings, note: str) -> dict:
    """The owner's action to trade again after a pause. Resets the high-water mark."""
    state = dict(read_state(settings), paused=False, peak_equity=None, note=f"resumed by owner: {note}")
    write_state(settings, state)
    return state


def size(candidate, risk: dict, open_risk: float, size_multiplier: float, margin_factor: float, open_margin: float = 0.0,
         margin_per_lot: float | None = None) -> Sized:
    """Lots for one candidate, or a refusal with its reason (R-2, R-3, R-6).

    `margin_per_lot` is the scenario estimate when the caller has one; otherwise maximum loss x `margin_factor`.
    """
    per_lot = max_loss_per_lot(candidate.legs, candidate.reference_prices, candidate.lot)
    margin_lot = margin_estimate(candidate.legs, candidate.reference_prices, candidate.lot, margin_factor) if margin_per_lot is None else float(margin_per_lot)
    if size_multiplier <= 0:
        return Sized(0, per_lot, 0.0, 0.0, "new_trades_paused_by_drawdown_rule")
    capital = float(risk["capital"])
    budget = capital * float(risk["risk_per_trade_pct"]) / 100.0 * size_multiplier
    lots = int(math.floor(budget / per_lot)) if per_lot > 0 else 0
    if lots < 1:
        return Sized(0, per_lot, 0.0, 0.0, "one_lot_exceeds_risk_per_trade")
    room = capital * float(risk["max_open_risk_pct"]) / 100.0 - open_risk
    if lots * per_lot > room:
        lots = int(math.floor(room / per_lot))
        if lots < 1:
            return Sized(0, per_lot, 0.0, 0.0, "total_open_risk_limit")
    if margin_lot > 0:
        free = capital - open_margin
        if lots * margin_lot > free:
            lots = int(math.floor(free / margin_lot))
            if lots < 1:
                return Sized(0, per_lot, 0.0, margin_lot, "margin_exceeds_free_capital")
    return Sized(lots, per_lot, lots * per_lot, lots * margin_lot)


def expiry_ok(candidate, entry_date, risk: dict) -> bool:
    """Entry must leave the position clear of the safety window before expiry (R-5)."""
    expiry = min(l.expiry for l in candidate.legs)
    return (expiry - pd.Timestamp(entry_date)).days > int(risk["expiry_safety_days"])


def propose(priced_trades: pd.DataFrame, capital: float, largest_acceptable_loss: float, max_concurrent: int = 5) -> dict:
    """Risk settings suggested by a backtest (R-10). Nothing here takes effect until the owner saves it.

    Risk per trade is set so that the worst losing run in the backtest, repeated at that size,
    costs no more than the largest loss the owner will accept.
    """
    from koscine4.harness.stats import max_drawdown, worst_run

    r = priced_trades.sort_values("exit_date")["result_pct"].dropna()
    run = worst_run(r)
    dd = max_drawdown(r)                                # in percent of one trade's maximum loss
    worst_units = max(dd, abs(run["total_result_pct"]), 100.0) / 100.0      # at least one full loss
    risk_per_trade = largest_acceptable_loss / worst_units
    risk_pct = 100.0 * risk_per_trade / capital
    accept_pct = 100.0 * largest_acceptable_loss / capital
    return {
        "inputs": {"capital": capital, "largest_acceptable_loss": largest_acceptable_loss, "backtest_trades": int(len(r))},
        "worst_losing_run": run,
        "largest_drawdown_pct_of_one_max_loss": dd,
        "proposed": {
            "risk_per_trade_pct": round(risk_pct, 2),
            "max_open_risk_pct": round(min(risk_pct * max_concurrent, accept_pct), 2),
            "drawdown_halve_pct": round(accept_pct / 2.0, 2),
            "drawdown_pause_pct": round(accept_pct, 2),
        },
        "worst_run_in_rupees_at_proposed_size": round(abs(run["total_result_pct"]) / 100.0 * risk_per_trade, 0),
        "largest_drawdown_in_rupees_at_proposed_size": round(dd / 100.0 * risk_per_trade, 0),
        "status": "proposal only; copy the values into [risk] in configs/settings.toml to adopt them",
    }


def write_proposal(settings: Settings, proposal: dict) -> Path:
    path = settings.path("records") / PROPOSAL_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(proposal, indent=2), encoding="utf-8")
    return path
