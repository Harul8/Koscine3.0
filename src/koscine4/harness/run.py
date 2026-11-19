"""Run one strategy through the harness and record the result (requirements V-1, V-9, V-11, V-12, V-15).

    python -m koscine4.harness.run --strategy nifty_credit_spread_1515
    python -m koscine4.harness.run --strategy nifty_credit_spread_1515 --start 2015-01-01 --slippage 1.0
    python -m koscine4.harness.run --strategy X --unlock-sealed --reason "final check before gate 2"

This is a backtest. It starts only when the owner runs it (M-8).
"""
from __future__ import annotations

import argparse
import json

import pandas as pd

from koscine4.harness import records
from koscine4.harness.engine import run_engine
from koscine4.harness.market import MarketData
from koscine4.harness.stats import summarise, to_markdown
from koscine4.settings import Settings, load_settings
from koscine4.strategies.base import SpecStrategy, Strategy
from koscine4.strategies.spec import StrategySpec, load_spec

RESULTS = "results"


def load_predictions(settings: Settings, spec: StrategySpec) -> dict[str, pd.DataFrame]:
    """Walk-forward predictions for every model the specification names."""
    names = set()
    if spec.score.get("source") == "model":
        names.add(str(spec.score["model"]))
    if spec.direction_model:
        names.add(spec.direction_model)
    if spec.tilt:
        names.add(str(spec.tilt["model"]))                  # DR-7: the forecast that may tilt the strikes
    out = {}
    for name in sorted(names):
        path = settings.path("pit") / "predictions" / f"{name}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"No predictions for model '{name}'. Train it first: python -m koscine4.models.train --model {name}")
        out[name] = pd.read_parquet(path)
    return out


def run(strategy: Strategy | StrategySpec | str, start=None, end=None, settings: Settings | None = None,
        slippage_pct: float | None = None, unlock_sealed: bool = False, reason: str = "", data: MarketData | None = None,
        write: bool = True, apply_weekly_cap: bool | None = None) -> dict:
    """Run, summarise, log. Returns {"run_id", "report", "trades", "skips", "meta"}."""
    settings = settings or load_settings()
    if isinstance(strategy, str):
        strategy = load_spec(settings, strategy)
    if isinstance(strategy, StrategySpec):
        strategy = SpecStrategy(strategy)
    spec = strategy.spec
    cfg = settings.section("harness")
    slippage = float(cfg["default_slippage_pct"]) if slippage_pct is None else float(slippage_pct)
    data = data or MarketData(settings, spec.decision_time, load_predictions(settings, spec))
    last = data.calendar.days[-1]
    start = pd.Timestamp(start) if start is not None else pd.Timestamp(cfg["default_start"])
    end, sealed_note = records.clip_to_seal(settings, end, last, unlock_sealed, reason)
    limits = dict(settings.section("selection"))
    cap = bool(spec.selection.get("weekly_cap", True)) if apply_weekly_cap is None else apply_weekly_cap
    result = run_engine(strategy, data, start, end, limits, groups=dict(settings.section("selection").get("groups", {})), apply_weekly_cap=cap,
                        slippage_pct=slippage)
    report = summarise(result.trades, result.skips, dict(settings.section("costs")), slippage)

    snapshot = records.data_snapshot(settings)
    rid = records.run_id(spec.fingerprint(), start, end, slippage, snapshot, "cap" if cap else "nocap")
    family = spec.family or spec.name
    already = any(e.get("run_id") == rid for e in records.read_log(settings))
    meta = {"run_id": rid, "strategy": spec.name, "family": family, "start": str(start.date()), "end": str(end.date()),
            "sealed_note": sealed_note, "data_snapshot": snapshot, "code_version": records.code_version(settings),
            "spec": spec.to_dict(), "spec_fingerprint": spec.fingerprint(), "weekly_cap": cap}
    if write:
        if not already:
            records.append_log(settings, {"kind": "run", **{k: meta[k] for k in ("run_id", "strategy", "family", "start", "end", "data_snapshot", "code_version", "spec_fingerprint", "spec", "weekly_cap")},
                                          "slippage_pct": slippage, "unlocked_sealed": bool(unlock_sealed and sealed_note),
                                          "summary": {k: report[k] for k in ("trades", "independent_bets", "mean_result_pct", "win_rate_pct", "max_drawdown_pct", "break_even_slippage_pct")}})
        meta["variants_tried"] = records.variants_tried(settings, family)
        out_dir = settings.path("runs") / rid
        out_dir.mkdir(parents=True, exist_ok=True)
        trades = result.trades.copy()
        for col in ("legs", "entry_prices", "exit_prices"):
            trades[col] = trades[col].map(json.dumps)
        trades.to_parquet(out_dir / "trades.parquet", index=False)
        pd.DataFrame(result.skips).to_csv(out_dir / "skips.csv", index=False)
        (out_dir / "report.json").write_text(json.dumps({"meta": meta, "report": report}, indent=2, default=str), encoding="utf-8")
        if spec.selection.get("save_as_outcome"):
            from koscine4.harness.engine import price_trades
            from koscine4.outcomes.build import write_structure

            write_structure(settings, spec.name, price_trades(result.trades, dict(settings.section("costs")), slippage))
        res = settings.path("records") / RESULTS
        res.mkdir(parents=True, exist_ok=True)
        (res / f"{spec.name}__{rid}.md").write_text(to_markdown(spec.name, report, meta), encoding="utf-8")
    else:
        meta["variants_tried"] = records.variants_tried(settings, family)
    return {"run_id": rid, "report": report, "trades": result.trades, "skips": result.skips, "meta": meta}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--strategy", required=True)
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--slippage", type=float, default=None, help="percent of price per leg")
    parser.add_argument("--unlock-sealed", action="store_true")
    parser.add_argument("--reason", default="")
    args = parser.parse_args()
    out = run(args.strategy, args.start, args.end, slippage_pct=args.slippage, unlock_sealed=args.unlock_sealed, reason=args.reason)
    print(to_markdown(out["meta"]["strategy"], out["report"], out["meta"]))


if __name__ == "__main__":
    main()
