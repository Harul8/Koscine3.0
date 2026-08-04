"""Probe Dhan's expired-option endpoint at 15-minute resolution.

The rolling-option API may change the actual strike behind an `ATM` label as
spot moves. Such a series is useful for rolling analytics but invalid for a
backtest that buys one fixed contract and holds it. This small authenticated
probe checks strike stability before any large download is attempted.

Required environment variable:
    DHAN_ACCESS_TOKEN

Example:
    python dhan_15m_probe.py --security-id 1333 --instrument OPTSTK \
        --from-date 2026-01-05 --to-date 2026-01-12
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pandas as pd


URL = "https://api.dhan.co/v2/charts/rollingoption"
HERE = Path(__file__).resolve().parent
FIELDS = ["open", "high", "low", "close", "iv", "volume", "strike", "oi", "spot"]


def _fetch(token: str, payload: dict[str, object]) -> dict[str, object]:
    request = Request(
        URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
            "access-token": token,
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Dhan rolling-option request failed: HTTP {exc.code}: {detail}") from exc


def _to_frame(response: dict[str, object], option_type: str) -> pd.DataFrame:
    branch = "ce" if option_type == "CALL" else "pe"
    data = response.get("data") or {}
    values = data.get(branch) if isinstance(data, dict) else None
    if not isinstance(values, dict) or not values.get("timestamp"):
        raise ValueError(f"response contains no {branch.upper()} timestamp data")
    length = len(values["timestamp"])
    frame = pd.DataFrame(
        {
            field: values.get(field, [None] * length)
            for field in [*FIELDS, "timestamp"]
        }
    )
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="s", utc=True).dt.tz_convert("Asia/Kolkata")
    frame["option_type"] = option_type
    return frame


def _strike_report(frame: pd.DataFrame) -> dict[str, object]:
    strikes = pd.to_numeric(frame["strike"], errors="coerce")
    changes = strikes.ne(strikes.shift()).fillna(False)
    return {
        "bars": int(len(frame)),
        "from": frame.timestamp.min().isoformat(),
        "to": frame.timestamp.max().isoformat(),
        "unique_strikes": sorted(float(value) for value in strikes.dropna().unique()),
        "strike_changes": int(changes.sum()),
        "fixed_contract_safe": bool(strikes.dropna().nunique() == 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--security-id", required=True, help="Dhan underlying security ID")
    parser.add_argument("--instrument", default="OPTSTK", choices=["OPTSTK", "OPTIDX"])
    parser.add_argument("--expiry-flag", default="MONTH", choices=["MONTH", "WEEK"])
    parser.add_argument("--expiry-code", type=int, default=1)
    parser.add_argument("--strike", default="ATM")
    parser.add_argument("--from-date", required=True)
    parser.add_argument("--to-date", required=True, help="Non-inclusive end date")
    parser.add_argument("--output-dir", type=Path, default=HERE / "results")
    args = parser.parse_args()

    token = os.environ.get("DHAN_ACCESS_TOKEN")
    if not token:
        raise SystemExit("DHAN_ACCESS_TOKEN is not set; no request was made")

    frames = []
    for option_type in ("CALL", "PUT"):
        payload = {
            "exchangeSegment": "NSE_FNO",
            "interval": "15",
            "securityId": args.security_id,
            "instrument": args.instrument,
            "expiryFlag": args.expiry_flag,
            "expiryCode": args.expiry_code,
            "strike": args.strike,
            "drvOptionType": option_type,
            "requiredData": FIELDS,
            "fromDate": args.from_date,
            "toDate": args.to_date,
        }
        frames.append(_to_frame(_fetch(token, payload), option_type))

    bars = pd.concat(frames, ignore_index=True).sort_values(["option_type", "timestamp"])
    report = {side: _strike_report(frame) for side, frame in bars.groupby("option_type")}
    report["usable_for_fixed_contract_backtest"] = bool(
        report["CALL"]["fixed_contract_safe"] and report["PUT"]["fixed_contract_safe"]
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    bars.to_parquet(args.output_dir / "dhan_rollingoption_15m_probe.parquet", index=False)
    (args.output_dir / "dhan_rollingoption_15m_probe.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
