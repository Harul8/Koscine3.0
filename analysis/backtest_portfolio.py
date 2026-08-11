"""Runs koscine/portfolio_sim.py's ₹10L capital-allocation backtest and reports: equity curve
stats (CAGR, max drawdown), brokerage drag as % of gross P&L, rotation frequency, and a
side-by-side vs. a no-rotation baseline (same capital/candidates, fill-and-hold only) -- the
number that actually tells us whether rotation earns its brokerage cost. See the approved plan
(~/.claude/plans -- "10L capital-allocation engine") for the design.

Usage:
    python analysis/backtest_portfolio.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"C:\Users\rahul\Koscine 3.0")
sys.path.insert(0, str(ROOT))
from koscine import portfolio_sim as ps  # noqa: E402

OUT_DIR = ROOT / "experiments" / "portfolio_v1" / "results"


def _report(label: str, result: dict) -> dict:
    trades = result["trade_log"]
    eq = result["equity_curve"]
    capital = result["capital"]
    if not len(eq):
        return {"label": label, "n_trades": 0}

    final_equity = eq["equity"].iloc[-1]
    days = (eq["date"].iloc[-1] - eq["date"].iloc[0]).days
    years = max(days / 365, 1 / 365)
    cagr = (final_equity / capital) ** (1 / years) - 1

    running_max = eq["equity"].cummax()
    drawdown = (eq["equity"] - running_max) / running_max
    max_dd = drawdown.min()

    min_free_capital = eq["capital_free"].min()
    max_n_open = eq["n_open"].max()
    mean_capital_free_pct = eq["capital_free"].mean() / capital * 100

    gross_pnl = sum((t.credit - t.exit_value) * t.lot_size for t in trades if t.exit_value is not None)
    total_brokerage = sum(t.entry_brokerage + t.exit_brokerage for t in trades)
    net_pnl = sum(t.realized_pnl for t in trades if t.realized_pnl is not None)
    n_rotated = sum(1 for t in trades if t.exit_reason == "rotated_out")

    by_strategy = {}
    for strat in ("condor", "broken_wing", "skew"):
        st = [t for t in trades if t.strategy == strat]
        by_strategy[strat] = dict(n=len(st), pnl=sum(t.realized_pnl for t in st if t.realized_pnl is not None))

    return dict(label=label, n_trades=len(trades), n_rotated=n_rotated,
                final_equity=round(final_equity, 0), cagr_pct=round(cagr * 100, 2),
                max_drawdown_pct=round(max_dd * 100, 2), gross_pnl=round(gross_pnl, 0),
                total_brokerage=round(total_brokerage, 0),
                brokerage_pct_of_gross=round(total_brokerage / gross_pnl * 100, 2) if gross_pnl else None,
                net_pnl=round(net_pnl, 0), by_strategy=by_strategy,
                open_at_end=len(result["open_at_end"]),
                min_free_capital=round(min_free_capital, 0), max_n_open=int(max_n_open),
                mean_capital_free_pct=round(mean_capital_free_pct, 1))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # SHIPPED config: fill-and-hold (no rotation -- backtested worse than this, see
    # portfolio_sim.py's docstring) + a 3-trading-day same-symbol cooldown.
    print("[backtest_portfolio] running SHIPPED config (no rotation, 3-day symbol cooldown) ...", flush=True)
    shipped = ps.simulate(allow_rotation=False, symbol_cooldown_days=3)
    # kept only as a diagnostic comparison, not shipped -- see portfolio_sim.py's docstring
    print("[backtest_portfolio] running rotation diagnostic (not shipped) ...", flush=True)
    with_rotation = ps.simulate(allow_rotation=True, symbol_cooldown_days=3)

    r1 = _report("shipped_no_rotation_cooldown3", shipped)
    r2 = _report("diagnostic_with_rotation", with_rotation)

    for r in (r1, r2):
        print(f"\n=== {r['label']} ===")
        for k, v in r.items():
            if k not in ("label", "by_strategy"):
                print(f"  {k}: {v}")
        print("  by_strategy:")
        for strat, d in r.get("by_strategy", {}).items():
            print(f"    {strat}: n={d['n']} pnl={d['pnl']:.0f}")

    pd.DataFrame([r1, r2]).drop(columns=["by_strategy"]).to_csv(OUT_DIR / "summary.csv", index=False)
    shipped["equity_curve"].to_csv(OUT_DIR / "equity_curve_shipped.csv", index=False)
    with_rotation["equity_curve"].to_csv(OUT_DIR / "equity_curve_rotation_diagnostic.csv", index=False)

    trade_rows = []
    for label, res in [("shipped_no_rotation_cooldown3", shipped), ("diagnostic_with_rotation", with_rotation)]:
        for t in res["trade_log"]:
            trade_rows.append(dict(run=label, trade_id=t.trade_id, strategy=t.strategy, tier=t.tier,
                                   symbol=t.symbol, entry_date=t.entry_date, exit_date=t.exit_date,
                                   exit_reason=t.exit_reason, credit=t.credit, max_risk=t.max_risk,
                                   lot_size=t.lot_size, entry_brokerage=round(t.entry_brokerage, 2),
                                   exit_brokerage=round(t.exit_brokerage, 2),
                                   realized_pnl=round(t.realized_pnl, 2) if t.realized_pnl is not None else None))
    pd.DataFrame(trade_rows).to_csv(OUT_DIR / "trade_log.csv", index=False)
    print(f"\n[backtest_portfolio] wrote summary.csv, equity_curve_*.csv, trade_log.csv -> {OUT_DIR}")


if __name__ == "__main__":
    main()
