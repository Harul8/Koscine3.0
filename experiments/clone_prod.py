"""Clone the LOCKED PROD large-move engine into an isolated experiment sandbox.

    python experiments/clone_prod.py <name>          # create experiments/<name>/ with a self-contained copy
    python experiments/clone_prod.py <name> --force  # overwrite an existing sandbox
    python experiments/clone_prod.py --verify        # assert PROD files are unchanged vs the locked checksum ledger

The clone copies src/koscine3/largemove/*.py into experiments/<name>/largemove/ and rewrites the
intra-package imports `koscine3.largemove` -> `largemove`, and repoints LOCK_DIR to the sandbox.
Shared infra (koscine3.data, koscine3.outcomes) is intentionally still imported from src/ — it is
stable and not the subject of experiments. PROD is never imported or mutated by the copy.
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "src" / "koscine3" / "largemove"
LOCK = ROOT / "locks" / "prod_largemove_v1"
LEDGER = LOCK / "checksums_sha256.csv"


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    h.update(p.read_bytes())
    return h.hexdigest()


def verify() -> int:
    """Re-hash exactly the files the locked ledger enumerates; fail on any change/missing file."""
    if not LEDGER.exists():
        print(f"NO LEDGER at {LEDGER} — run `python -m koscine3.largemove.lock` first.")
        return 2
    bad, missing = [], []
    n = 0
    for line in LEDGER.read_text().splitlines()[1:]:
        if not line.strip():
            continue
        rel, digest = line.rsplit(",", 1)
        rel, digest = rel.strip(), digest.strip()
        n += 1
        p = ROOT / rel
        if not p.exists():
            missing.append(rel)
        elif _sha256(p) != digest:
            bad.append(rel)
    if bad or missing:
        print("PROD INTEGRITY FAILURE — PROD differs from the lock ledger:")
        for r in bad:
            print(f"  CHANGED {r}")
        for r in missing:
            print(f"  MISSING {r}")
        return 1
    print(f"PROD intact — {n} files match the locked checksums.")
    return 0


def clone(name: str, force: bool) -> int:
    dst = ROOT / "experiments" / name
    if dst.exists():
        if not force:
            print(f"{dst} already exists. Use --force to overwrite.")
            return 1
        shutil.rmtree(dst)
    (dst / "largemove").mkdir(parents=True)

    for src in sorted(PKG.glob("*.py")):
        text = src.read_text(encoding="utf-8").replace("koscine3.largemove", "largemove")
        if src.name == "config.py":
            text = text.replace(
                'LOCK_DIR = Path(__file__).resolve().parents[3] / "locks" / VERSION',
                "LOCK_DIR = Path(__file__).resolve().parents[1]   # sandbox root (experiment-local artifacts)",
            ).replace('VERSION = "prod_largemove_v1"', f'VERSION = "exp_{name}"')
        (dst / "largemove" / src.name).write_text(text, encoding="utf-8")

    shutil.copy2(LOCK / "universe_groups.json", dst / "universe_groups.json")

    (dst / "run_experiment.py").write_text(
        '"""Entry point for this experiment. Edit largemove/ freely — PROD is untouched."""\n'
        "import sys\n"
        "from pathlib import Path\n\n"
        "HERE = Path(__file__).resolve().parent\n"
        'sys.path.insert(0, str(HERE))                       # -> the cloned `largemove` package\n'
        'sys.path.insert(0, str(HERE.parents[1] / "src"))   # -> shared koscine3.data / koscine3.outcomes\n\n'
        "import pandas as pd\n"
        "from largemove import pipeline as P\n"
        "from largemove.config import PROD, PREDICTIONS_DIR\n\n"
        "def main():\n"
        "    df = P.load_dataset(PROD)\n"
        "    out = P.walk_forward(PROD, df)\n"
        "    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)\n"
        "    allp = []\n"
        "    for group, p in out.items():\n"
        '        p.to_csv(PREDICTIONS_DIR / f"group_{group}_predictions.csv", index=False)\n'
        "        allp.append(p)\n"
        "        d1 = p[p.rank_in_day == 1]\n"
        '        by = d1.assign(y=pd.to_datetime(d1.date).dt.year).groupby("y")["hit"].mean().mul(100).round(1).to_dict()\n'
        "        print(f\"{group}: prec@1 {d1['hit'].mean()*100:.1f}% by-year {by} | {len(p)} rows\")\n"
        "    P.rank_cooldown(pd.concat(allp, ignore_index=True), PROD, n_per_day=2)\\\n"
        '        .to_csv(PREDICTIONS_DIR / "combined_shortlist.csv", index=False)\n\n'
        'if __name__ == "__main__":\n'
        "    main()\n",
        encoding="utf-8",
    )

    (dst / "README.md").write_text(
        f"# Experiment: {name}\n\n"
        f"Self-contained clone of PROD (`{LOCK.name}`). Edit `largemove/` here — PROD is frozen.\n\n"
        "```\npython run_experiment.py\n```\n\n"
        "Artifacts (predictions/, models/) are written under this folder. "
        "When done, compare against PROD; promote only via a new lock (see EXPERIMENT_POLICY.md).\n",
        encoding="utf-8",
    )
    print(f"Cloned PROD -> {dst.relative_to(ROOT)}  (edit largemove/, run run_experiment.py)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("name", nargs="?")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    if args.verify:
        return verify()
    if not args.name:
        ap.error("provide an experiment name, or --verify")
    return clone(args.name, args.force)


if __name__ == "__main__":
    sys.exit(main())
