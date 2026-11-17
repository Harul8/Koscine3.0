"""Run log, data snapshot, sealed period and gate records (requirements V-11, V-12, N-7, N-9, S-11).

The run log and the gate decisions are small text files under `records/`, which git tracks.
Full run outputs (every trade) go under `runs/`, which it does not.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from koscine4.settings import Settings

RUN_LOG = Path("run_log") / "run_log.jsonl"     # not "runs/": that name is ignored by git everywhere
GATES = Path("gates")


class SealedPeriodError(RuntimeError):
    """Raised when a run asks for the sealed months without an explicit unlock."""


class GateError(RuntimeError):
    """Raised for a malformed gate record."""


def data_snapshot(settings: Settings) -> dict:
    """Size and modification time of every store table, and one hash of them (N-7)."""
    files = {}
    pit = settings.path("pit")
    for p in sorted(pit.rglob("*.parquet")) if pit.exists() else []:
        if "option_quotes" in p.parts or "_superseded" in p.parts:
            continue
        st = p.stat()
        files[str(p.relative_to(pit)).replace("\\", "/")] = [st.st_size, int(st.st_mtime)]
    quotes = pit / "option_quotes"
    if quotes.exists():
        sizes = [f.stat().st_size for f in quotes.rglob("*.parquet")]
        files["option_quotes/*"] = [sum(sizes), len(sizes)]
    digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()[:12]
    return {"hash": digest, "tables": len(files)}


def code_version(settings: Settings) -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=settings.root, capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def seal_date(settings: Settings, last_session) -> pd.Timestamp:
    """The last date a normal run may report on: `sealed_months` before the last session (V-12)."""
    return pd.Timestamp(last_session) - pd.DateOffset(months=int(settings.section("harness")["sealed_months"]))


def clip_to_seal(settings: Settings, end, last_session, unlock: bool, reason: str = "") -> tuple[pd.Timestamp, str]:
    """(end date to use, note). A run past the seal date needs `unlock` and a reason, and the unlock is logged."""
    seal = seal_date(settings, last_session)
    end = pd.Timestamp(end) if end is not None else pd.Timestamp(last_session)
    if end <= seal:
        return end, ""
    if not unlock:
        return seal, f"Stopped at the seal date {seal.date()}; the last {settings.section('harness')['sealed_months']} months are sealed."
    if not reason.strip():
        raise SealedPeriodError("Unlocking the sealed months needs a written reason.")
    append_log(settings, {"kind": "sealed_period_unlocked", "reason": reason, "end": str(end.date())})
    return end, f"SEALED PERIOD UNLOCKED (to {end.date()}): {reason}"


def read_log(settings: Settings) -> list[dict]:
    path = settings.path("records") / RUN_LOG
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def append_log(settings: Settings, entry: dict) -> dict:
    path = settings.path("records") / RUN_LOG
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {"logged_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), **entry}
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, sort_keys=True, default=str) + "\n")
    return entry


def variants_tried(settings: Settings, family: str) -> int:
    """How many different specifications of one idea have been run so far (V-11)."""
    seen = {e.get("spec_fingerprint") for e in read_log(settings) if e.get("kind") == "run" and e.get("family") == family}
    return len(seen - {None})


def run_id(spec_fingerprint: str, start, end, slippage_pct: float, snapshot: dict, extra: str = "") -> str:
    """The same inputs always give the same id (V-15)."""
    text = "|".join([spec_fingerprint, str(pd.Timestamp(start).date()), str(pd.Timestamp(end).date()), f"{slippage_pct:.4f}", snapshot["hash"], extra])
    return hashlib.sha256(text.encode()).hexdigest()[:12]


# ---------------------------------------------------------------- gates

def gate_path(settings: Settings, strategy: str) -> Path:
    return settings.path("records") / GATES / f"{strategy}.json"


def read_gates(settings: Settings, strategy: str) -> list[dict]:
    path = gate_path(settings, strategy)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []


def record_gate(settings: Settings, strategy: str, gate: int, decision: str, run_ids: list[str], note: str = "",
                decided_by: str = "owner") -> dict:
    """Write a gate decision. Decisions are appended, never edited."""
    if gate not in (0, 1, 2, 3, 4) or decision not in ("pass", "fail"):
        raise GateError("A gate record needs gate 0 to 4 and decision 'pass' or 'fail'.")
    if gate >= 2 and not run_ids:
        raise GateError("A gate decision for a strategy must name the runs that support it.")
    known = {e.get("run_id") for e in read_log(settings)}
    missing = [r for r in run_ids if r not in known]
    if missing:
        raise GateError(f"Runs not found in the run log: {missing}")
    record = {"strategy": strategy, "gate": gate, "decision": decision, "runs": list(run_ids), "note": note,
              "decided_by": decided_by, "decided_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    path = gate_path(settings, strategy)
    path.parent.mkdir(parents=True, exist_ok=True)
    records = read_gates(settings, strategy) + [record]
    path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    return record


def has_passed(settings: Settings, strategy: str, gate: int) -> bool:
    """True if the latest decision for this gate is a pass."""
    decisions = [r for r in read_gates(settings, strategy) if r["gate"] == gate]
    return bool(decisions) and decisions[-1]["decision"] == "pass"


def check_gate_2(report: dict, sealed_report: dict | None, baseline_mean: float | None, criteria: dict) -> dict:
    """Each gate 2 criterion with whether the report meets it. The owner makes the decision; this only measures."""
    years = report.get("by_year", {})
    positive_years = sum(1 for v in years.values() if v["mean_result_pct"] > 0)
    twice = report.get("at_twice_slippage", {}).get("mean_result_pct")
    checks = {
        "positive_at_twice_slippage": bool(twice is not None and twice > 0),
        "positive_in_two_of_three_years": bool(years) and positive_years >= float(criteria["min_positive_year_share"]) * len(years),
        "positive_on_sealed_months": None if sealed_report is None else bool(sealed_report.get("mean_result_pct", float("nan")) > 0),
        "better_than_best_baseline": None if baseline_mean is None else bool(report["mean_result_pct"] > baseline_mean),
        "enough_independent_bets": report["independent_bets"] >= int(criteria["min_independent_bets"]),
        "drawdown_within_limit": None if criteria.get("max_drawdown_pct") is None else bool(report["max_drawdown_pct"] <= float(criteria["max_drawdown_pct"])),
    }
    checks["all_measured_pass"] = all(v for v in checks.values() if v is not None)
    checks["not_yet_measured"] = [k for k, v in checks.items() if v is None]
    return checks
