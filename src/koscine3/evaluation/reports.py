from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def write_report_tables(report: dict[str, pd.DataFrame], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, table in report.items():
        table.to_csv(output_dir / f"{name}.csv", index=False)


def write_manifest(payload: dict[str, object], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

