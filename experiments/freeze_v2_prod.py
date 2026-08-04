"""Freeze the PROD v2 mover engine (immutable). Locks the model DEFINITION only — code + config + universe —
not the daily book/overlay outputs (those keep updating operationally as the locked engine runs each day).

    python experiments/freeze_v2_prod.py            # write the engine lock
    python experiments/freeze_v2_prod.py --verify   # assert the v2 engine is unchanged
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "locks" / "prod_largemove_v2"
ENGINE_FILES = [
    ROOT / "src" / "koscine3" / "largemove" / "mover_v2.py",   # selection engine + all config constants
    LOCK / "universe_groups.json",                              # locked A/B universe
]
LEDGER = LOCK / "ENGINE_LOCK.json"


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def lock() -> int:
    files = {p.relative_to(ROOT).as_posix(): _sha(p) for p in ENGINE_FILES}
    payload = {
        "version": "prod_largemove_v2",
        "frozen_on": date.today().isoformat(),
        "status": "FROZEN — do not edit mover_v2.py or universe_groups.json. Experiments must clone.",
        "model": {
            "ranker": "atm_iv", "cooldown_trading_days": 3, "per_stock_trade_cap": 0.15,
            "picks_per_group_per_day": 3, "window_days": 5, "min_underlying": 100.0,
            "direction": "agnostic (direction overlay is a separate, non-PROD signal under active research)",
        },
        "engine_files": files,
    }
    LEDGER.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"FROZEN prod_largemove_v2 — engine ledger -> {LEDGER.relative_to(ROOT)}")
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
        print("PROD v2 ENGINE CHANGED:")
        for r in bad:
            print(f"  ALTERED {r}")
        return 1
    print(f"PROD v2 engine intact — {len(want)} files match the freeze.")
    return 0


if __name__ == "__main__":
    sys.exit(verify() if "--verify" in sys.argv else lock())
