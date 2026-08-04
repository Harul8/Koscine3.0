"""Download exact expired option-contract candles from Upstox.

Input manifest columns:
    trade_id, symbol, option_type, expired_instrument_key, from_date, to_date

Each trade must contain one CE and one PE row. The script enforces the approved
top-30 stock + NIFTY/BANKNIFTY/SENSEX/FINNIFTY scope before any request.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen

import pandas as pd

from universe import normalize_underlying, require_in_universe


BASE_URL = "https://api.upstox.com/v2/expired-instruments/historical-candle"
REQUIRED_MANIFEST = {
    "trade_id",
    "symbol",
    "option_type",
    "expired_instrument_key",
    "from_date",
    "to_date",
}


def _fetch(token: str, row: object) -> list[list[object]]:
    key = quote(str(row.expired_instrument_key), safe="")
    url = f"{BASE_URL}/{key}/1minute/{row.to_date}/{row.from_date}"
    request = Request(
        url,
        headers={
            "Accept": "application/json",
            "Authorization": f"Bearer {token}",
        },
    )
    try:
        with urlopen(request, timeout=60) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        if "UDAPI1149" in detail:
            raise RuntimeError(
                "Upstox rejected expired-contract history with UDAPI1149. "
                "This endpoint requires an Upstox Plus subscription."
            ) from exc
        raise RuntimeError(f"Upstox candle request failed: HTTP {exc.code}: {detail}") from exc
    candles = (payload.get("data") or {}).get("candles") or []
    return candles


def _to_frame(row: object, candles: list[list[object]]) -> pd.DataFrame:
    frame = pd.DataFrame(
        candles,
        columns=["timestamp", "open", "high", "low", "close", "volume", "oi"],
    )
    if frame.empty:
        return frame
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True).dt.tz_convert(
        "Asia/Kolkata"
    )
    frame["trade_id"] = str(row.trade_id)
    frame["symbol"] = normalize_underlying(row.symbol)
    frame["option_type"] = str(row.option_type).upper()
    frame["instrument_key"] = str(row.expired_instrument_key)
    for column in ("signal_date", "expiry", "strike", "group", "pred"):
        if hasattr(row, column):
            frame[column] = getattr(row, column)
    return frame.sort_values("timestamp")


def validate_manifest(manifest: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(REQUIRED_MANIFEST - set(manifest.columns))
    if missing:
        raise ValueError(f"missing Upstox manifest columns: {missing}")
    out = manifest.copy()
    out["symbol"] = out["symbol"].map(normalize_underlying)
    out["option_type"] = out["option_type"].astype(str).str.upper()
    require_in_universe(out["symbol"])
    side_counts = (
        out.groupby("trade_id").option_type.agg(lambda values: set(values)).to_dict()
    )
    invalid = sorted(
        str(trade_id) for trade_id, sides in side_counts.items() if sides != {"CE", "PE"}
    )
    if invalid:
        raise ValueError(f"each trade must have exactly CE and PE manifest rows: {invalid[:10]}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/intraday/options_contract_legs_1m.parquet"),
    )
    args = parser.parse_args()

    token = os.environ.get("UPSTOX_ACCESS_TOKEN")
    if not token:
        raise SystemExit(
            "UPSTOX_ACCESS_TOKEN is not set; use a read-only Analytics Token and retry"
        )
    manifest = validate_manifest(pd.read_csv(args.manifest))
    frames: list[pd.DataFrame] = []
    for row in manifest.itertuples(index=False):
        frame = _to_frame(row, _fetch(token, row))
        if frame.empty:
            print(f"warning: no candles for {row.trade_id} {row.option_type}")
        else:
            frames.append(frame)
    if not frames:
        raise SystemExit("no 1-minute option candles were returned")
    bars = pd.concat(frames, ignore_index=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    bars.to_parquet(args.output, index=False)
    print(
        f"saved {len(bars):,} exact-contract 1-minute candles for "
        f"{bars.trade_id.nunique()} trades -> {args.output}"
    )


if __name__ == "__main__":
    main()
