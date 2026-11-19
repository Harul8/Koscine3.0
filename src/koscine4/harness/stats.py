"""Run statistics (requirements V-6, V-9, V-10).

Results are stated as a share of the maximum loss of each trade, so a result does not depend on
position size. Statistics count independent bets: trades on one instrument whose holding periods
overlap count once.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from koscine4.harness.engine import price_trades


def independent_count(trades: pd.DataFrame) -> int:
    """Trades left after dropping, per instrument, any trade that starts before the previous kept one ended."""
    n = 0
    for _, g in trades.sort_values("entry_date").groupby("instrument", sort=False):
        last_exit = None
        for entry, exit_ in zip(g["entry_date"], g["exit_date"]):
            if last_exit is None or entry > last_exit:
                n += 1
                last_exit = exit_
    return n


def max_drawdown(results: pd.Series) -> float:
    """Largest peak-to-trough fall of the running sum of results (same unit as the results)."""
    if results.empty:
        return 0.0
    curve = results.cumsum()
    return float((curve.cummax().clip(lower=0.0) - curve).max())


def worst_run(results: pd.Series) -> dict:
    """The longest run of losing trades and its total."""
    longest = current = 0
    worst_sum = current_sum = 0.0
    for r in results:
        if r < 0:
            current += 1
            current_sum += r
            if current > longest or (current == longest and current_sum < worst_sum):
                longest, worst_sum = current, current_sum
        else:
            current, current_sum = 0, 0.0
    return {"losing_trades_in_a_row": int(longest), "total_result_pct": float(worst_sum)}


def break_even_slippage(trades: pd.DataFrame, costs: dict, hi: float = 20.0) -> float:
    """Slippage per leg, in percent of price, at which the mean result is zero. 0 if it loses with none."""
    if trades.empty:
        return float("nan")
    mean_at = lambda s: float(price_trades(trades, costs, s)["result_pct"].mean())
    if mean_at(0.0) <= 0:
        return 0.0
    if mean_at(hi) > 0:
        return hi
    lo = 0.0
    for _ in range(24):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if mean_at(mid) > 0 else (lo, mid)
    return round(0.5 * (lo + hi), 3)


def _block(p: pd.DataFrame) -> dict:
    r = p["result_pct"]
    return {
        "trades": int(len(p)),
        "independent_bets": independent_count(p),
        "mean_result_pct": float(r.mean()) if len(p) else float("nan"),
        "median_result_pct": float(r.median()) if len(p) else float("nan"),
        "win_rate_pct": float((r > 0).mean() * 100.0) if len(p) else float("nan"),
        "total_result_pct": float(r.sum()),
        "max_drawdown_pct": max_drawdown(p.sort_values("exit_date")["result_pct"]),
        "mean_pnl_per_lot": float(p["pnl_per_lot"].mean()) if len(p) else float("nan"),
        "mean_costs_per_lot": float(p["costs_per_lot"].mean()) if len(p) else float("nan"),
    }


def summarise(trades: pd.DataFrame, skips: list[dict], costs: dict, slippage_pct: float, periods: int = 3) -> dict:
    """Every field requirement V-9 lists, for one slippage level."""
    p = price_trades(trades, costs, slippage_pct)
    report = {"slippage_pct": slippage_pct, **_block(p)}
    if p.empty:
        report.update({"by_year": {}, "by_trust": {}, "by_expiry_kind": {}, "by_period": {}, "by_exit_reason": {},
                       "worst_run": worst_run(pd.Series(dtype=float)), "break_even_slippage_pct": float("nan"),
                       "at_twice_slippage": {}, "skipped": _skips(skips)})
        return report
    p = p.sort_values("exit_date")
    report["by_year"] = {int(y): _block(g) for y, g in p.groupby(p["entry_date"].dt.year)}
    report["by_trust"] = {str(k): _block(g) for k, g in p.groupby("entry_trust")}                 # V-6: shown separately
    kinds = p["expiry_kind"] if "expiry_kind" in p.columns else pd.Series("", index=p.index)
    report["by_expiry_kind"] = {str(k): _block(g) for k, g in p.groupby(kinds.replace("", "unknown"))}   # IX-7: never pooled silently
    edges = np.array_split(np.sort(p["entry_date"].dt.year.unique()), min(periods, p["entry_date"].dt.year.nunique()))
    report["by_period"] = {f"{int(e[0])}-{int(e[-1])}": _block(p[p["entry_date"].dt.year.isin(e)]) for e in edges if len(e)}
    report["by_exit_reason"] = {str(k): int(v) for k, v in p["exit_reason"].value_counts().items()}
    report["worst_run"] = worst_run(p["result_pct"])
    report["break_even_slippage_pct"] = break_even_slippage(trades, costs)
    report["at_twice_slippage"] = _block(price_trades(trades, costs, 2.0 * slippage_pct))
    report["skipped"] = _skips(skips)
    return report


def _skips(skips: list[dict]) -> dict:
    counts: dict[str, int] = {}
    for s in skips:
        counts[s["reason"]] = counts.get(s["reason"], 0) + 1
    entry = sum(v for k, v in counts.items() if k.startswith("entry:"))
    return {"at_entry": entry, "by_reason": dict(sorted(counts.items(), key=lambda kv: -kv[1]))}


def to_markdown(name: str, report: dict, meta: dict) -> str:
    """A short readable report."""
    f = lambda x: "n/a" if x is None or x != x else f"{x:,.2f}"
    lines = [f"# {name}", "",
             f"Period {meta.get('start')} to {meta.get('end')}. Slippage {report['slippage_pct']}% per leg. "
             f"Run {meta.get('run_id')}. Variants tried in this family: {meta.get('variants_tried')}.", ""]
    if meta.get("sealed_note"):
        lines += [meta["sealed_note"], ""]
    lines += ["| Measure | Value |", "|---|---|",
              f"| Trades | {report['trades']} |", f"| Independent bets | {report['independent_bets']} |",
              f"| Mean result, % of maximum loss | {f(report['mean_result_pct'])} |",
              f"| Win rate, % | {f(report['win_rate_pct'])} |",
              f"| Largest drawdown, % of one trade's maximum loss | {f(report['max_drawdown_pct'])} |",
              f"| Break-even slippage, % per leg | {f(report.get('break_even_slippage_pct'))} |",
              f"| Mean result at twice the slippage | {f(report.get('at_twice_slippage', {}).get('mean_result_pct'))} |",
              f"| Skipped at entry | {report['skipped']['at_entry']} |", ""]
    for title, key in (("By year", "by_year"), ("By period", "by_period"), ("By fill quality", "by_trust"), ("By expiry kind", "by_expiry_kind")):
        rows = report.get(key) or {}
        if rows:
            lines += [f"## {title}", "", "| | Trades | Independent | Mean result % | Win rate % |", "|---|---|---|---|---|"]
            lines += [f"| {k} | {v['trades']} | {v['independent_bets']} | {f(v['mean_result_pct'])} | {f(v['win_rate_pct'])} |" for k, v in rows.items()]
            lines.append("")
    if report["skipped"]["by_reason"]:
        lines += ["## Skipped and refused", "", "| Reason | Count |", "|---|---|"]
        lines += [f"| {k} | {v} |" for k, v in list(report["skipped"]["by_reason"].items())[:15]]
    return "\n".join(lines) + "\n"
