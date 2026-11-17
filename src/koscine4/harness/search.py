"""Grid search over one strategy's settings (requirements IX-2 to IX-4).

The grid is written down in `configs/searches/<name>.toml` before it runs. Every cell is run
over the whole period; the best cell is chosen on the earlier part only, and its result on the
later part is shown beside it. The whole grid is reported, so one good cell among bad
neighbours can be told from a stable region. Every cell is logged as a variant tried.

    python -m koscine4.harness.search --grid nifty_condor_grid

This is a set of backtests. It starts only when the owner runs it (M-8).
"""
from __future__ import annotations

import argparse
import itertools
import json
import tomllib

import pandas as pd

from koscine4.harness import records
from koscine4.harness.engine import price_trades, run_engine
from koscine4.harness.market import MarketData
from koscine4.harness.run import load_predictions
from koscine4.harness.stats import _block
from koscine4.settings import Settings, load_settings
from koscine4.strategies.base import SpecStrategy
from koscine4.strategies.spec import from_dict, load_spec


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


def search(name: str, settings: Settings | None = None, start=None, end=None) -> dict:
    settings = settings or load_settings()
    g = load_grid(settings, name)
    base = load_spec(settings, g["base"])
    cfg = settings.section("harness")
    slippage = float(g.get("slippage_pct", cfg["default_slippage_pct"]))
    costs = dict(settings.section("costs"))
    data = MarketData(settings, base.decision_time, load_predictions(settings, base))
    start = pd.Timestamp(start or g.get("start") or cfg["default_start"])
    end, sealed_note = records.clip_to_seal(settings, end, data.calendar.days[-1], False)
    split = pd.Timestamp(g["select_until"])
    min_trades = int(g.get("min_trades", 30))
    limits = dict(settings.section("selection"))
    snapshot = records.data_snapshot(settings)
    rows = []
    for i, cell in enumerate(cells(g["grid"])):
        spec = from_dict(apply_cell(base.to_dict(), cell, f"g{i:03d}"))
        result = run_engine(SpecStrategy(spec), data, start, end, limits, apply_weekly_cap=bool(spec.selection.get("weekly_cap", True)))
        priced = price_trades(result.trades, costs, slippage)
        early, late = priced[priced["entry_date"] <= split], priced[priced["entry_date"] > split]
        rid = records.run_id(spec.fingerprint(), start, end, slippage, snapshot, "grid")
        records.append_log(settings, {"kind": "run", "run_id": rid, "strategy": spec.name, "family": spec.family, "grid": name,
                                      "start": str(start.date()), "end": str(end.date()), "slippage_pct": slippage,
                                      "spec_fingerprint": spec.fingerprint(), "data_snapshot": snapshot, "spec": spec.to_dict(),
                                      "summary": _block(priced)})
        rows.append({"cell": i, **cell, "run_id": rid, "early": _block(early), "late": _block(late)})
    eligible = [r for r in rows if r["early"]["trades"] >= min_trades]
    best = max(eligible, key=lambda r: r["early"]["mean_result_pct"]) if eligible else None
    report = {"grid": name, "base": base.name, "start": str(start.date()), "end": str(end.date()), "select_until": str(split.date()),
              "slippage_pct": slippage, "cells": rows, "best_cell": None if best is None else best["cell"], "sealed_note": sealed_note,
              "variants_tried": records.variants_tried(settings, base.family or base.name)}
    res = settings.path("records") / "results"
    res.mkdir(parents=True, exist_ok=True)
    (res / f"search_{name}.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    (res / f"search_{name}.md").write_text(to_markdown(report, sorted(g["grid"])), encoding="utf-8")
    return report


def to_markdown(report: dict, keys: list[str]) -> str:
    f = lambda x: "n/a" if x != x else f"{x:.2f}"
    lines = [f"# Grid search: {report['grid']} (base {report['base']})", "",
             f"Setting chosen on trades entered up to {report['select_until']}, then applied unchanged to later trades. "
             f"Slippage {report['slippage_pct']}% per leg. Variants tried in this family: {report['variants_tried']}. {report['sealed_note']}", "",
             "| Cell | " + " | ".join(keys) + " | Early trades | Early mean % | Early win % | Later trades | Later mean % | Later win % |",
             "|---|" + "---|" * (len(keys) + 6)]
    for r in report["cells"]:
        mark = " **chosen**" if r["cell"] == report["best_cell"] else ""
        lines.append(f"| {r['cell']}{mark} | " + " | ".join(str(r[k]) for k in keys) +
                     f" | {r['early']['trades']} | {f(r['early']['mean_result_pct'])} | {f(r['early']['win_rate_pct'])}"
                     f" | {r['late']['trades']} | {f(r['late']['mean_result_pct'])} | {f(r['late']['win_rate_pct'])} |")
    if report["best_cell"] is None:
        lines += ["", "No cell had enough trades in the early part to be chosen."]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--grid", required=True)
    args = parser.parse_args()
    report = search(args.grid)
    print(to_markdown(report, sorted(load_grid(load_settings(), args.grid)["grid"])))


if __name__ == "__main__":
    main()
