"""Trading costs for options (requirement V-8).

Per order: brokerage. On premium turnover: exchange transaction charge and SEBI fee, both sides;
securities transaction tax on the sell side; stamp duty on the buy side. GST applies to
brokerage, the exchange charge and the SEBI fee. Rates that changed over time come from the
dated schedule in the settings file, so a 2015 trade is charged 2015 rates.
"""
from __future__ import annotations

from typing import Mapping

import pandas as pd

TICK = 0.05


def rates_on(costs: Mapping, date) -> dict:
    """The cost rates in force on `date`."""
    date = pd.Timestamp(date)
    rates = {k: v for k, v in costs.items() if k != "schedule"}
    chosen = None
    for row in sorted(costs.get("schedule", []), key=lambda r: pd.Timestamp(r["from"])):
        if pd.Timestamp(row["from"]) <= date:
            chosen = row
    if chosen is None and costs.get("schedule"):
        chosen = sorted(costs["schedule"], key=lambda r: pd.Timestamp(r["from"]))[0]
    if chosen:
        rates.update({k: v for k, v in chosen.items() if k != "from"})
    for key in ("brokerage_per_order", "stt_sell_pct", "exchange_txn_pct", "sebi_per_crore", "stamp_buy_pct", "gst_pct"):
        if key not in rates:
            raise KeyError(f"Cost setting '{key}' is missing for {date.date()}.")
    return rates


def order_cost(price: float, quantity: float, is_buy: bool, rates: Mapping) -> float:
    """Cost in rupees of one option order of `quantity` shares at `price`."""
    turnover = price * quantity
    brokerage = float(rates["brokerage_per_order"])
    exchange = turnover * float(rates["exchange_txn_pct"]) / 100.0
    sebi = turnover * float(rates["sebi_per_crore"]) / 1e7
    stt = 0.0 if is_buy else turnover * float(rates["stt_sell_pct"]) / 100.0
    stamp = turnover * float(rates["stamp_buy_pct"]) / 100.0 if is_buy else 0.0
    gst = (brokerage + exchange + sebi) * float(rates["gst_pct"]) / 100.0
    return brokerage + exchange + sebi + stt + stamp + gst


def with_slippage(price: float, is_buy: bool, slippage_pct: float) -> float:
    """The price after slippage: a buyer pays more, a seller receives less, by at least one tick when slippage is on."""
    if slippage_pct <= 0:
        return price
    move = max(price * slippage_pct / 100.0, TICK)
    return price + move if is_buy else max(price - move, 0.0)
