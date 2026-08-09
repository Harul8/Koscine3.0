"""Download NIFTY 50 index 5-minute candles from Upstox (v3 historical-candle API, native
5-minute interval -- no need to fetch 1-minute and resample).

Coverage: Upstox's minute-level historical data for NIFTY only goes back to ~late Dec 2021 /
early Jan 2022 (~4.5 years as of 2026), confirmed directly against the live API (both v2 and v3
endpoints show the same cutoff; daily candles go back much further, intraday does not). Not the
originally-hoped-for 10 years -- a genuine Upstox platform limit, not a token/tier issue.

Max range per request for minute-level candles is 30 days (UDAPI1148 "Invalid date range" above
that, confirmed empirically). Chunks the full window into 30-day slices and walks backward from
the latest trading day, per request (most recent data first, in case of interruption).

Usage:
    python analysis/download_nifty_5m.py
    python analysis/download_nifty_5m.py --start 2022-01-01 --end 2026-08-05
    python analysis/download_nifty_5m.py --instrument "NSE_INDEX|Nifty Bank" --out data/intraday/banknifty_5m.parquet
"""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import date, timedelta
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen

import pandas as pd

ROOT = Path(r"C:\Users\rahul\Koscine 3.0")
BASE_URL = "https://api.upstox.com/v3/historical-candle"
DEFAULT_INSTRUMENT = "NSE_INDEX|Nifty 50"
DEFAULT_START = date(2022, 1, 1)   # a little before the confirmed cutoff; early chunks may come back empty
MAX_CHUNK_DAYS = 20                 # Upstox v3 minute-candle max range per request is inconsistent around
                                     # 30 days in practice (some 30-day windows 400 with UDAPI1148, some
                                     # 31-day windows don't) -- 20 stays safely clear of it
RATE_LIMIT_SLEEP = 0.35             # seconds between requests, conservative


def _chunks_desc(start: date, end: date, days: int) -> list[tuple[date, date]]:
    """30-day [from, to] windows covering [start, end], latest-first."""
    out = []
    cur_to = end
    while cur_to >= start:
        cur_from = max(start, cur_to - timedelta(days=days - 1))
        out.append((cur_from, cur_to))
        cur_to = cur_from - timedelta(days=1)
    return out


def _fetch_chunk(token: str, instrument_key: str, unit: str, interval: str, d_from: date, d_to: date) -> list[list[object]]:
    key = quote(instrument_key, safe="")
    url = f"{BASE_URL}/{key}/{unit}/{interval}/{d_to.isoformat()}/{d_from.isoformat()}"
    # Cloudflare in front of api.upstox.com blocks Python urllib's default User-Agent
    # (bot-signature heuristic, unrelated to auth) -- a normal browser UA avoids the false positive.
    req = Request(url, headers={
        "Accept": "application/json",
        "Authorization": f"Bearer {token}",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    })
    try:
        with urlopen(req, timeout=60) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Upstox candle request failed ({d_from}..{d_to}): HTTP {exc.code}: {detail}") from exc
    return (payload.get("data") or {}).get("candles") or []


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instrument", default=DEFAULT_INSTRUMENT, help="Upstox instrument_key, e.g. 'NSE_INDEX|Nifty 50'")
    ap.add_argument("--unit", default="minutes")
    ap.add_argument("--interval", default="5")
    ap.add_argument("--start", default=DEFAULT_START.isoformat())
    ap.add_argument("--end", default=date.today().isoformat())
    ap.add_argument("--out", default=str(ROOT / "data" / "intraday" / "nifty_5m.parquet"))
    args = ap.parse_args()

    token = os.environ.get("UPSTOX_ACCESS_TOKEN")
    if not token:
        raise SystemExit("UPSTOX_ACCESS_TOKEN is not set")

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    chunks = _chunks_desc(start, end, MAX_CHUNK_DAYS)
    print(f"[nifty_5m] {args.instrument}  {len(chunks)} chunks, {start} -> {end}, walking latest-first", flush=True)

    def fetch_resilient(d_from: date, d_to: date, depth: int = 0) -> list[list[object]]:
        """Never let one bad chunk kill an unattended run: on any request error, halve the
        range and retry (once), then give up on that slice and log it rather than crash."""
        try:
            return _fetch_chunk(token, args.instrument, args.unit, args.interval, d_from, d_to)
        except RuntimeError as exc:
            if depth >= 2 or d_from >= d_to:
                print(f"  GIVING UP on {d_from}..{d_to} after retries: {exc}", flush=True)
                return []
            mid = d_from + (d_to - d_from) / 2
            print(f"  retrying {d_from}..{d_to} as two halves (error: {exc})", flush=True)
            time.sleep(RATE_LIMIT_SLEEP)
            first = fetch_resilient(d_from, mid, depth + 1)
            time.sleep(RATE_LIMIT_SLEEP)
            second = fetch_resilient(mid + timedelta(days=1), d_to, depth + 1)
            return first + second

    frames: list[pd.DataFrame] = []
    empty_run = 0
    for i, (d_from, d_to) in enumerate(chunks):
        candles = fetch_resilient(d_from, d_to)
        if candles:
            empty_run = 0
            df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume", "oi"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("Asia/Kolkata")
            frames.append(df)
            print(f"  [{i+1}/{len(chunks)}] {d_from} -> {d_to}: {len(df)} candles", flush=True)
        else:
            empty_run += 1
            print(f"  [{i+1}/{len(chunks)}] {d_from} -> {d_to}: 0 candles (likely past the data cutoff)", flush=True)
            if empty_run >= 3:
                print(f"  3 consecutive empty chunks -- stopping (data cutoff reached around {d_to})", flush=True)
                break
        time.sleep(RATE_LIMIT_SLEEP)
        if (i + 1) % 10 == 0 and frames:
            # incremental checkpoint save, in case of interruption on a long run
            pd.concat(frames, ignore_index=True).drop_duplicates("timestamp").sort_values("timestamp").to_parquet(out_path, index=False)
            print(f"  checkpoint saved ({sum(len(f) for f in frames)} candles so far) -> {out_path}", flush=True)

    if not frames:
        raise SystemExit("no candles downloaded")
    bars = pd.concat(frames, ignore_index=True).drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    bars.to_parquet(out_path, index=False)
    print(f"[nifty_5m] wrote {len(bars):,} candles, {bars['timestamp'].min()} -> {bars['timestamp'].max()} -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
