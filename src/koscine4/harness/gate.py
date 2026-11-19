"""Measure a run against the gate 2 criteria, and record the owner's gate decision (N-9, release gates).

    python -m koscine4.harness.gate check  --run <run_id> [--sealed-run <id>] [--baseline-run <id>]
    python -m koscine4.harness.gate record --strategy X --gate 2 --decision pass --runs <id> [<id> ...] --note "..."
    python -m koscine4.harness.gate propose-risk --run <run_id> --capital 1000000 --largest-loss 100000

`check` only measures. The decision is the owner's, and `record` writes it.
"""
from __future__ import annotations

import argparse
import json

import pandas as pd

from koscine4.harness import records
from koscine4.harness.engine import price_trades
from koscine4.harness.market import TrackRecord
from koscine4.risk import rules as risk_rules
from koscine4.settings import Settings, load_settings


def load_report(settings: Settings, run_id: str) -> dict:
    path = settings.path("runs") / run_id / "report.json"
    if not path.exists():
        raise FileNotFoundError(f"No stored run {run_id} under {settings.path('runs')}.")
    return json.loads(path.read_text(encoding="utf-8"))


def load_trades(settings: Settings, run_id: str) -> pd.DataFrame:
    t = pd.read_parquet(settings.path("runs") / run_id / "trades.parquet")
    for col in ("legs", "entry_prices", "exit_prices"):
        t[col] = t[col].map(json.loads)
    return t


def track_record(settings: Settings, strategy: str) -> TrackRecord | None:
    """S-5: the closed trades of the runs behind the strategy's latest gate 2 pass, priced at each run's own
    slippage. A trade that appears in more than one of those runs counts once. None without a pass on record."""
    passes = [r for r in records.read_gates(settings, strategy) if r["gate"] == 2 and r["decision"] == "pass"]
    if not passes:
        return None
    frames = []
    for rid in passes[-1]["runs"]:
        if not (settings.path("runs") / rid / "trades.parquet").exists():
            continue
        slippage = float(load_report(settings, rid)["report"]["slippage_pct"])
        frames.append(price_trades(load_trades(settings, rid), dict(settings.section("costs")), slippage))
    if not frames:
        return None
    t = pd.concat(frames, ignore_index=True).drop_duplicates(["instrument", "entry_date", "exit_date"])
    return TrackRecord.from_frame(t)


def check(settings: Settings, run_id: str, sealed_run: str | None = None, baseline_run: str | None = None) -> dict:
    report = load_report(settings, run_id)["report"]
    sealed = load_report(settings, sealed_run)["report"] if sealed_run else None
    baseline = load_report(settings, baseline_run)["report"]["mean_result_pct"] if baseline_run else None
    return records.check_gate_2(report, sealed, baseline, dict(settings.section("gate2")))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("--run", required=True)
    c.add_argument("--sealed-run")
    c.add_argument("--baseline-run")
    r = sub.add_parser("record")
    r.add_argument("--strategy", required=True)
    r.add_argument("--gate", type=int, required=True)
    r.add_argument("--decision", required=True, choices=["pass", "fail"])
    r.add_argument("--runs", nargs="*", default=[])
    r.add_argument("--note", default="")
    p = sub.add_parser("propose-risk")
    p.add_argument("--run", required=True)
    p.add_argument("--capital", type=float, required=True)
    p.add_argument("--largest-loss", type=float, required=True)
    u = sub.add_parser("resume", help="trade again after the drawdown rule paused new trades")
    u.add_argument("--note", required=True)
    args = parser.parse_args()
    settings = load_settings()
    if args.cmd == "check":
        print(json.dumps(check(settings, args.run, args.sealed_run, args.baseline_run), indent=2))
    elif args.cmd == "record":
        print(json.dumps(records.record_gate(settings, args.strategy, args.gate, args.decision, args.runs, args.note), indent=2))
    elif args.cmd == "propose-risk":
        slippage = float(settings.section("harness")["default_slippage_pct"])
        priced = price_trades(load_trades(settings, args.run), dict(settings.section("costs")), slippage)
        proposal = risk_rules.propose(priced, args.capital, args.largest_loss)
        print(json.dumps(proposal, indent=2))
        print("written to", risk_rules.write_proposal(settings, proposal))
    else:
        print(json.dumps(risk_rules.resume(settings, args.note), indent=2))


if __name__ == "__main__":
    main()
