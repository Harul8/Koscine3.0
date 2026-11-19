"""Grid search over one strategy's settings (requirements IX-2 to IX-5, IX-7).

The grid is written down in `configs/searches/<name>.toml` before it runs. Every cell is run
over the whole period; the best cell is chosen on the earlier part only, and its result on the
later part is shown beside it. The whole grid is reported, so one good cell among bad
neighbours can be told from a stable region. Every cell is logged as a variant tried.

Fills (IX-5). A search runs twice, then the two are compared:

    --fills daily   every fill from the daily file: the open, or the day's close standing in for
                    15:15. The whole history, for length. The setting is chosen here.
    --fills 5m      15:15 fills from the 5-minute option-chain files, over the months they cover
                    (from October 2024), for realistic fills. Every cell is run again; nothing is
                    chosen on this part.
    --compare       the two side by side on the months both cover, cell by cell.

Weekly and monthly expiries (IX-7). They are different instruments, so a grid that varies
`expiry.kind` chooses a setting for each kind separately; results are never pooled across them.

    python -m koscine4.harness.search --grid nifty_credit_spread_grid --fills daily
    python -m koscine4.harness.search --grid nifty_credit_spread_grid --fills 5m
    python -m koscine4.harness.search --grid nifty_credit_spread_grid --compare

The first two are sets of backtests. They start only when the owner runs them (M-8).
"""
from __future__ import annotations

import argparse
import itertools
import json
import tomllib

import pandas as pd

from koscine4.harness import records
from koscine4.harness.engine import price_trades, run_engine
from koscine4.harness.market import IntradayMarks, MarketData
from koscine4.harness.run import load_predictions
from koscine4.harness.stats import _block
from koscine4.settings import Settings, load_settings
from koscine4.strategies.base import SpecStrategy
from koscine4.strategies.spec import StrategySpec, from_dict, load_spec

FILLS = ("daily", "5m")


def load_grid(settings: Settings, name: str) -> dict:
    with (settings.root / "configs" / "searches" / f"{name}.toml").open("rb") as fh:
        return tomllib.load(fh)


def cells(grid: dict) -> list[dict]:
    keys = sorted(grid)
    return [dict(zip(keys, combo)) for combo in itertools.product(*(grid[k] for k in keys))]


def apply_cell(base: dict, cell: dict, suffix: str) -> dict:
    d = json.loads(json.dumps(base))
    for key, value in cell.items():
        section, _, field = key.partition(".")
        if field:
            d[section][field] = value
        else:
            d[section] = value
    d["name"] = f"{base['name']}__{suffix}"
    d["family"] = base.get("family") or base["name"]
    return d


def intraday_start(settings: Settings, spec: StrategySpec) -> pd.Timestamp | None:
    """First day the 5-minute chain files cover every instrument of a 15:15 strategy, or None."""
    if spec.entry_time != "1515":
        return None
    firsts = []
    for instrument in spec.instruments:
        if instrument not in IntradayMarks.FOLDERS:
            return None
        marks = IntradayMarks(settings, instrument)
        if not marks.dates:
            return None
        firsts.append(min(marks.dates))
    return max(firsts) if firsts else None


def results_dir(settings: Settings):
    res = settings.path("records") / "results"
    res.mkdir(parents=True, exist_ok=True)
    return res


def _stem(name: str, fills: str) -> str:
    return f"search_{name}" if fills == "daily" else f"search_{name}_{fills}"


def search(name: str, settings: Settings | None = None, start=None, end=None, fills: str = "daily") -> dict:
    if fills not in FILLS:
        raise ValueError(f"fills must be one of {FILLS}.")
    settings = settings or load_settings()
    g = load_grid(settings, name)
    base = load_spec(settings, g["base"])
    cfg = settings.section("harness")
    slippage = float(g.get("slippage_pct", cfg["default_slippage_pct"]))
    costs = dict(settings.section("costs"))
    first_5m = intraday_start(settings, base)
    if fills == "5m":
        if base.entry_time != "1515":
            raise ValueError("5-minute fills exist only for 15:15 entries; an evening strategy is searched on daily data alone.")
        if first_5m is None:
            raise FileNotFoundError(f"No 5-minute option-chain files for {list(base.instruments)} under {settings.path('intraday')}.")
    data = MarketData(settings, base.decision_time, load_predictions(settings, base), use_intraday=(fills == "5m"))
    start = pd.Timestamp(start or g.get("start") or cfg["default_start"])
    if fills == "5m":
        start = max(start, first_5m)
    end, sealed_note = records.clip_to_seal(settings, end, data.calendar.days[-1], False)
    split = pd.Timestamp(g["select_until"])
    min_trades = int(g.get("min_trades", 30))
    limits = dict(settings.section("selection"))
    snapshot = records.data_snapshot(settings)
    rows = []
    for i, cell in enumerate(cells(g["grid"])):
        spec = from_dict(apply_cell(base.to_dict(), cell, f"g{i:03d}"))
        result = run_engine(SpecStrategy(spec), data, start, end, limits, apply_weekly_cap=bool(spec.selection.get("weekly_cap", True)),
                            slippage_pct=slippage)
        priced = price_trades(result.trades, costs, slippage)
        rid = records.run_id(spec.fingerprint(), start, end, slippage, snapshot, "grid" if fills == "daily" else f"grid_{fills}")
        records.append_log(settings, {"kind": "run", "run_id": rid, "strategy": spec.name, "family": spec.family, "grid": name, "fills": fills,
                                      "start": str(start.date()), "end": str(end.date()), "slippage_pct": slippage,
                                      "spec_fingerprint": spec.fingerprint(), "data_snapshot": snapshot, "spec": spec.to_dict(),
                                      "summary": _block(priced)})
        row = {"cell": i, **cell, "expiry_kind": str(spec.expiry.get("kind", "nearest")), "run_id": rid, "all": _block(priced)}
        if fills == "daily":
            row["early"] = _block(priced[priced["entry_date"] <= split])
            row["late"] = _block(priced[priced["entry_date"] > split])
            row["overlap_5m"] = _block(priced[priced["entry_date"] >= first_5m]) if first_5m is not None else None
        rows.append(row)

    report = {"grid": name, "base": base.name, "fills": fills, "start": str(start.date()), "end": str(end.date()),
              "select_until": str(split.date()), "slippage_pct": slippage, "first_5m_day": None if first_5m is None else str(first_5m.date()),
              "cells": rows, "sealed_note": sealed_note, "variants_tried": records.variants_tried(settings, base.family or base.name)}
    if fills == "daily":
        report["best_by_kind"] = choose(rows, min_trades)
        report["best_cell"] = next(iter(report["best_by_kind"].values())) if len(report["best_by_kind"]) == 1 else None
    res = results_dir(settings)
    stem = _stem(name, fills)
    (res / f"{stem}.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    (res / f"{stem}.md").write_text(to_markdown(report, sorted(g["grid"])), encoding="utf-8")
    return report


def choose(rows: list[dict], min_trades: int) -> dict:
    """The best cell on the early part, for each expiry kind on its own (IX-3, IX-7)."""
    best: dict[str, dict] = {}
    for r in rows:
        if r["early"]["trades"] < min_trades:
            continue
        k = r["expiry_kind"]
        if k not in best or r["early"]["mean_result_pct"] > best[k]["early"]["mean_result_pct"]:
            best[k] = r
    return {k: r["cell"] for k, r in sorted(best.items())}


def compare(name: str, settings: Settings | None = None) -> dict:
    """IX-5: daily fills against 5-minute fills, cell by cell, on the months both cover."""
    settings = settings or load_settings()
    res = results_dir(settings)
    paths = {f: res / f"{_stem(name, f)}.json" for f in FILLS}
    for f, p in paths.items():
        if not p.exists():
            raise FileNotFoundError(f"No {f} search report for '{name}'. Run: python -m koscine4.harness.search --grid {name} --fills {f}")
    daily, five = (json.loads(paths[f].read_text(encoding="utf-8")) for f in FILLS)
    five_cells = {c["cell"]: c for c in five["cells"]}
    chosen = {v: k for k, v in daily.get("best_by_kind", {}).items()}
    rows = []
    for c in daily["cells"]:
        f = five_cells.get(c["cell"])
        ov = c.get("overlap_5m") or {}
        d_mean, f_mean = ov.get("mean_result_pct", float("nan")), (f or {}).get("all", {}).get("mean_result_pct", float("nan"))
        rows.append({"cell": c["cell"], "expiry_kind": c["expiry_kind"], "chosen_for": chosen.get(c["cell"]),
                     "daily_all_trades": c["all"]["trades"], "daily_all_mean_pct": c["all"]["mean_result_pct"],
                     "daily_overlap_trades": ov.get("trades", 0), "daily_overlap_mean_pct": d_mean,
                     "five_min_trades": (f or {}).get("all", {}).get("trades", 0), "five_min_mean_pct": f_mean,
                     "difference_pct": f_mean - d_mean})
    out = {"grid": name, "first_5m_day": daily.get("first_5m_day"), "slippage_pct": daily["slippage_pct"], "cells": rows,
           "daily_run": daily["end"], "five_min_run": five["end"]}
    (res / f"search_{name}_compare.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    (res / f"search_{name}_compare.md").write_text(compare_markdown(out), encoding="utf-8")
    return out


def _f(x) -> str:
    return "n/a" if x is None or x != x else f"{x:.2f}"


def to_markdown(report: dict, keys: list[str]) -> str:
    fills = report.get("fills", "daily")
    head = (f"# Grid search: {report['grid']} (base {report['base']}), {'daily' if fills == 'daily' else '5-minute'} fills", "")
    if fills != "daily":
        lines = [*head, f"15:15 fills from the 5-minute option chain, {report['start']} to {report['end']}. Nothing is chosen on this run; "
                 f"see the comparison. Slippage {report['slippage_pct']}% per leg. {report['sealed_note']}", "",
                 "| Cell | " + " | ".join(keys) + " | Trades | Mean % | Win % |", "|---|" + "---|" * (len(keys) + 3)]
        for r in report["cells"]:
            a = r["all"]
            lines.append(f"| {r['cell']} | " + " | ".join(str(r[k]) for k in keys) + f" | {a['trades']} | {_f(a['mean_result_pct'])} | {_f(a['win_rate_pct'])} |")
        return "\n".join(lines) + "\n"
    chosen = {v: k for k, v in report.get("best_by_kind", {}).items()}
    several = len(report.get("best_by_kind", {})) > 1
    lines = [*head, f"Setting chosen on trades entered up to {report['select_until']}, then applied unchanged to later trades"
             + (", separately for each expiry kind" if several else "") + ". "
             f"Slippage {report['slippage_pct']}% per leg. Variants tried in this family: {report['variants_tried']}. {report['sealed_note']}", "",
             "| Cell | " + " | ".join(keys) + " | Early trades | Early mean % | Early win % | Later trades | Later mean % | Later win % |",
             "|---|" + "---|" * (len(keys) + 6)]
    for r in report["cells"]:
        mark = ""
        if r["cell"] in chosen:
            mark = f" **chosen ({chosen[r['cell']]})**" if several else " **chosen**"
        lines.append(f"| {r['cell']}{mark} | " + " | ".join(str(r[k]) for k in keys) +
                     f" | {r['early']['trades']} | {_f(r['early']['mean_result_pct'])} | {_f(r['early']['win_rate_pct'])}"
                     f" | {r['late']['trades']} | {_f(r['late']['mean_result_pct'])} | {_f(r['late']['win_rate_pct'])} |")
    if not chosen:
        lines += ["", "No cell had enough trades in the early part to be chosen."]
    return "\n".join(lines) + "\n"


def compare_markdown(out: dict) -> str:
    lines = [f"# Daily fills against 5-minute fills: {out['grid']}", "",
             f"On trades entered from {out['first_5m_day']}, the months both cover. A large gap means the daily close was a poor "
             f"stand-in for a 15:15 fill. Slippage {out['slippage_pct']}% per leg on both.", "",
             "| Cell | Kind | Chosen | Daily, all years: trades | mean % | Daily, same months: trades | mean % | 5-minute: trades | mean % | 5-minute minus daily |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in out["cells"]:
        lines.append(f"| {r['cell']} | {r['expiry_kind']} | {r['chosen_for'] or ''} | {r['daily_all_trades']} | {_f(r['daily_all_mean_pct'])} | "
                     f"{r['daily_overlap_trades']} | {_f(r['daily_overlap_mean_pct'])} | {r['five_min_trades']} | {_f(r['five_min_mean_pct'])} | "
                     f"{_f(r['difference_pct'])} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--grid", required=True)
    parser.add_argument("--fills", choices=FILLS, default="daily")
    parser.add_argument("--compare", action="store_true", help="compare the stored daily and 5-minute reports; runs no backtest")
    args = parser.parse_args()
    if args.compare:
        print(compare_markdown(compare(args.grid)))
        return
    report = search(args.grid, fills=args.fills)
    print(to_markdown(report, sorted(load_grid(load_settings(), args.grid)["grid"])))


if __name__ == "__main__":
    main()
