"""Freeze the PROD direction_v1 engine (immutable) — the group-B PUT/CALL directional lean over the v3 book.
Locks the model DEFINITION only (code + universe), not the daily book output (that keeps updating as it runs).

    python experiments/freeze_direction_v1.py            # write the engine lock
    python experiments/freeze_direction_v1.py --verify   # assert the engine is unchanged
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "locks" / "prod_direction_v1"
ENGINE_FILES = [
    ROOT / "src" / "koscine3" / "largemove" / "direction_v1.py",   # engine + all config constants
    LOCK / "universe_groups.json",                                  # locked A/B universe
]
LEDGER = LOCK / "ENGINE_LOCK.json"


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def lock() -> int:
    files = {p.relative_to(ROOT).as_posix(): _sha(p) for p in ENGINE_FILES}
    payload = {
        "version": "prod_direction_v1",
        "frozen_on": date.today().isoformat(),
        "status": "FROZEN — do not edit direction_v1.py or universe_groups.json. Experiments must clone.",
        "model": {
            "scope": "group B (movers) ONLY — PUT/CALL lean over the frozen v3 5d B picks; group A stays v3 agnostic",
            "type": "CatBoost classifier P(close[t+5]>close[t]), EXPANDING window, MONTHLY retrain, embargo 6d",
            "train_universe": "ALL eligible", "lean": "CALL if P_up>=0.5 else PUT", "horizon_days": 5,
            "nature": "market-timing/beta tilt — SMALL lean; short windows INVERT, OI/flow-only ~0 (do not change)",
        },
        "engine_files": files,
    }
    LEDGER.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"FROZEN prod_direction_v1 — engine ledger -> {LEDGER.relative_to(ROOT)}")
    for f, h in files.items():
        print(f"  {f}  {h[:16]}…")
    return 0


def verify() -> int:
    if not LEDGER.exists():
        print("No ENGINE_LOCK.json — run without --verify first.")
        return 2
    want = json.loads(LEDGER.read_text())["engine_files"]
    bad = [rel for rel, h in want.items() if not (ROOT / rel).exists() or _sha(ROOT / rel) != h]
    if bad:
        print("PROD direction_v1 ENGINE CHANGED:")
        for r in bad:
            print(f"  ALTERED {r}")
        return 1
    print(f"PROD direction_v1 engine intact — {len(want)} files match the freeze.")
    return 0


if __name__ == "__main__":
    sys.exit(verify() if "--verify" in sys.argv else lock())
