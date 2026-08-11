"""Download the full NIFTY OPTION CHAIN history at 5-minute resolution from Upstox
(requires Upstox Plus -- expired-contract lookup and candles are Plus-gated).

Output is a proper CHAIN table, not a long list of per-contract candles: one row per
(timestamp, expiry, strike), with the CE and PE side of that strike side by side --
ce_open/high/low/close/volume/oi and pe_open/high/low/close/volume/oi as separate columns,
same shape as looking at a strike row on a live option-chain screen. OI is real and included
(Upstox's candle payload's last field) -- CE/PE just come back as separate per-contract candle
series from the API (there's no bulk "whole chain over time" endpoint), so this script fetches
each contract individually and PIVOTS them into the chain shape before saving -- the fetch
granularity doesn't change, only what's written to disk.

Each strike's CE and PE are individually fetched as 1-minute candles (the expired-instruments
candle endpoint only supports 1minute, not native 5-minute like the regular v3 endpoint) and
resampled to 5-minute locally, so the sampling stays mathematically consistent (same approach
already used in experiments/intraday_exit_v1).

Coverage: expired-contract LOOKUP (not just candles) only goes back to ~Oct 2024 (~97 NIFTY
expiries as of Aug 2026, confirmed via /v2/expired-instruments/expiries -- no pagination
parameter changes the result, this is a hard cutoff) -- shorter than the ~4.5-year window the
plain index candles have. ~200 contracts/expiry x 97 expiries =~ 19,000+ contracts total.

Writes ONE PARQUET FILE PER EXPIRY (not one giant file) to
data/intraday/nifty_option_chain_5m/{expiry}.parquet -- keeps memory bounded and makes the run
resumable (skips expiries whose file already exists). Also joins the NIFTY spot 5-min close
(from data/intraday/nifty_5m.parquet, already downloaded) as an `underlying_close` column, for
moneyness/context, when available.

Usage:
    python analysis/download_nifty_option_chain_5m.py
    python analysis/download_nifty_option_chain_5m.py --instrument "NSE_INDEX|Nifty Bank"
"""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import date, datetime, timedelta
from pathlib import Path
import socket
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

import pandas as pd

ROOT = Path(r"C:\Users\rahul\Koscine 3.0")
V2_BASE = "https://api.upstox.com/v2"
DEFAULT_INSTRUMENT = "NSE_INDEX|Nifty 50"
MAX_CHUNK_DAYS = 20        # same safe margin as the index downloader (Upstox's ~30-day cap is unreliable)
RATE_LIMIT_SLEEP = 0.4     # ~2.5 req/s -- the 0.15s (~6.7 req/s) rate looked safe in a short burst test
                            # but over a sustained multi-hour run tripped Upstox's real quota heavily
                            # (many expiries saw 30-60% chunk failure rates); Upstox's rate limit is
                            # evidently a sustained/rolling quota, not just a burst guard -- back off hard
LOOKBACK_DAYS = 50         # how far before expiry to start looking for candles (covers monthly contracts
                            # listed weeks in advance; early empty chunks are cheap and expected)
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
NETWORK_RETRIES = 4        # transient connection resets (Cloudflare closing connections under sustained
                            # request volume) are common on long unattended runs -- retry with backoff
                            # before surfacing, separately from the HTTPError/RuntimeError (API error) path
RATE_LIMIT_RETRIES = 6     # HTTP 429 gets its own, longer retry budget -- a sub-3s backoff (as used for
                            # generic retryable errors) is nowhere near enough to clear a per-minute quota


SERVER_ERROR_CODES = {500, 502, 503, 504}  # transient server-side errors -- short backoff is fine


def _get(url: str, token: str) -> dict:
    req = Request(url, headers={"Accept": "application/json", "Authorization": f"Bearer {token}", "User-Agent": UA})
    last_exc: Exception | None = None
    rate_limit_attempt = 0
    attempt = 0
    while attempt < NETWORK_RETRIES and rate_limit_attempt < RATE_LIMIT_RETRIES:
        try:
            with urlopen(req, timeout=60) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            if exc.code == 429:
                # a per-minute-scale quota, not a burst guard -- honor Retry-After if Upstox sends
                # one, otherwise back off on a scale of seconds-to-tens-of-seconds, not sub-3s
                retry_after = exc.headers.get("Retry-After") if exc.headers else None
                wait = float(retry_after) if retry_after and retry_after.isdigit() else min(3 * (2 ** rate_limit_attempt), 60)
                rate_limit_attempt += 1
                if rate_limit_attempt < RATE_LIMIT_RETRIES:
                    last_exc = exc
                    time.sleep(wait)
                    continue
                raise RuntimeError(f"GET {url} failed: HTTP 429 after {RATE_LIMIT_RETRIES} attempts: {detail}") from exc
            if exc.code in SERVER_ERROR_CODES and attempt < NETWORK_RETRIES - 1:
                last_exc = exc
                attempt += 1
                time.sleep(0.5 * attempt)
                continue
            raise RuntimeError(f"GET {url} failed: HTTP {exc.code}: {detail}") from exc
        except (URLError, ConnectionResetError, ConnectionAbortedError, TimeoutError, socket.timeout, socket.error) as exc:
            # transient network-level failure (e.g. WinError 10054 connection reset) -- not an
            # API error response, so it's not wrapped as RuntimeError; retry with backoff instead
            last_exc = exc
            attempt += 1
            if attempt < NETWORK_RETRIES:
                time.sleep(0.5 * attempt)
                continue
            raise RuntimeError(f"GET {url} failed: network error after {NETWORK_RETRIES} attempts: {exc}") from exc
    raise RuntimeError(f"GET {url} failed: unreachable") from last_exc


def list_expiries(token: str, instrument_key: str) -> list[str]:
    url = f"{V2_BASE}/expired-instruments/expiries?instrument_key={quote(instrument_key, safe='')}"
    return _get(url, token).get("data") or []


def list_contracts(token: str, instrument_key: str, expiry: str) -> list[dict]:
    url = f"{V2_BASE}/expired-instruments/option/contract?instrument_key={quote(instrument_key, safe='')}&expiry_date={expiry}"
    return _get(url, token).get("data") or []


def _chunks_desc(start: date, end: date, days: int) -> list[tuple[date, date]]:
    out = []
    cur_to = end
    while cur_to >= start:
        cur_from = max(start, cur_to - timedelta(days=days - 1))
        out.append((cur_from, cur_to))
        cur_to = cur_from - timedelta(days=1)
    return out


def fetch_contract_candles(token: str, expired_key: str, expiry: date) -> tuple[pd.DataFrame, int, int]:
    """1-minute candles for one expired contract's whole plausible life, resampled to 5-min.
    Returns (bars5, n_chunks_failed, n_chunks_ok). A failed chunk (request error surviving
    _get's internal retries -- e.g. a sustained rate-limit window) is NOT the same thing as a
    genuinely empty chunk (successful call, contract just wasn't trading yet) -- conflating the
    two silently swallows real data loss as "past the data cutoff". Failures are logged and
    counted so the caller can decide whether the whole contract/expiry is trustworthy, but the
    backward walk still treats a failure like an empty chunk for the purposes of the
    stop-after-2-consecutive-empties heuristic (retrying indefinitely here would stall the run --
    the failure count is what protects data integrity, not this loop)."""
    start = expiry - timedelta(days=LOOKBACK_DAYS)
    frames = []
    empty_run = 0
    n_failed = 0
    n_ok = 0
    for d_from, d_to in _chunks_desc(start, expiry, MAX_CHUNK_DAYS):
        key = quote(expired_key, safe="")
        url = f"{V2_BASE}/expired-instruments/historical-candle/{key}/1minute/{d_to.isoformat()}/{d_from.isoformat()}"
        try:
            candles = (_get(url, token).get("data") or {}).get("candles") or []
            n_ok += 1
        except RuntimeError as exc:
            candles = []
            n_failed += 1
            print(f"    chunk fetch error {expired_key} {d_from}..{d_to}: {exc}", flush=True)
        time.sleep(RATE_LIMIT_SLEEP)
        if candles:
            empty_run = 0
            df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume", "oi"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("Asia/Kolkata")
            frames.append(df)
        else:
            empty_run += 1
            if empty_run >= 2:   # 2 consecutive empty/failed chunks walking backward = before contract's first trade
                break
    if not frames:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume", "oi"]), n_failed, n_ok
    bars = pd.concat(frames, ignore_index=True).drop_duplicates("timestamp").sort_values("timestamp").set_index("timestamp")
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum", "oi": "last"}
    bars5 = bars.resample("5min", label="left", closed="left").agg(agg).dropna(subset=["open"]).reset_index()
    return bars5, n_failed, n_ok


SYMBOL_SLUGS = {  # instrument_key -> filename slug, matching this project's existing naming
                  # convention (see download_nifty_5m.py's own --instrument example)
    "NSE_INDEX|Nifty 50": "nifty",
    "NSE_INDEX|Nifty Bank": "banknifty",
}


def _symbol_slug(instrument_key: str) -> str:
    if instrument_key in SYMBOL_SLUGS:
        return SYMBOL_SLUGS[instrument_key]
    name = instrument_key.split("|", 1)[-1]
    return "".join(ch.lower() for ch in name if ch.isalnum())


def _load_spot(instrument_key: str) -> pd.DataFrame | None:
    f = ROOT / "data" / "intraday" / f"{_symbol_slug(instrument_key)}_5m.parquet"
    if not f.exists():
        return None
    spot = pd.read_parquet(f, columns=["timestamp", "close"]).rename(columns={"close": "underlying_close"})
    return spot


def _pivot_to_chain(rows: list[tuple[dict, pd.DataFrame]], expiry_str: str, spot: pd.DataFrame | None) -> pd.DataFrame:
    """rows: list of (contract_meta, 5m_bars_df) for every CE/PE contract in this expiry.
    Returns one row per (timestamp, strike): ce_* and pe_* columns side by side.

    Upstox's contract list can carry more than one instrument_key for the same (strike, side)
    in a single expiry (e.g. re-listed/re-issued contracts) -- concatenating same-side legs and
    de-duplicating by (timestamp, strike) before the CE/PE merge avoids that colliding into
    pandas' "duplicate columns after suffixes" MergeError."""
    side_frames: dict[str, list[pd.DataFrame]] = {"ce": [], "pe": []}
    for c, bars5 in rows:
        if bars5.empty:
            continue
        side = c["instrument_type"].lower()  # "ce" / "pe"
        if side not in side_frames:
            continue
        leg = bars5.rename(columns={col: f"{side}_{col}" for col in ("open", "high", "low", "close", "volume", "oi")})
        leg["strike"] = c["strike_price"]
        leg[f"{side}_lot_size"] = c.get("lot_size")
        side_frames[side].append(leg)

    sides: dict[str, pd.DataFrame] = {}
    for side, legs in side_frames.items():
        if not legs:
            continue
        df = pd.concat(legs, ignore_index=True)
        # if the same (timestamp, strike) shows up from more than one instrument_key, keep the
        # row with the larger OI (the more "real"/liquid of the duplicates) rather than silently
        # picking an arbitrary one
        oi_col = f"{side}_oi"
        df = df.sort_values(oi_col, na_position="first").drop_duplicates(subset=["timestamp", "strike"], keep="last")
        sides[side] = df

    if not sides:
        return pd.DataFrame()
    if len(sides) == 2:
        merged = sides["ce"].merge(sides["pe"], on=["timestamp", "strike"], how="outer")
    else:
        merged = next(iter(sides.values()))
    merged["expiry"] = expiry_str
    if spot is not None:
        merged = merged.merge(spot, on="timestamp", how="left")
    cols = ["timestamp", "expiry", "strike", "underlying_close",
            "ce_open", "ce_high", "ce_low", "ce_close", "ce_volume", "ce_oi", "ce_lot_size",
            "pe_open", "pe_high", "pe_low", "pe_close", "pe_volume", "pe_oi", "pe_lot_size"]
    cols = [c for c in cols if c in merged.columns]
    return merged[cols].sort_values(["timestamp", "strike"]).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instrument", default=DEFAULT_INSTRUMENT)
    ap.add_argument("--out-dir", default=None, help="default: data/intraday/{symbol}_option_chain_5m, "
                                                       "derived from --instrument")
    args = ap.parse_args()

    token = os.environ.get("UPSTOX_ACCESS_TOKEN")
    if not token:
        raise SystemExit("UPSTOX_ACCESS_TOKEN is not set")

    slug = _symbol_slug(args.instrument)
    out_dir = Path(args.out_dir) if args.out_dir else ROOT / "data" / "intraday" / f"{slug}_option_chain_5m"
    out_dir.mkdir(parents=True, exist_ok=True)
    spot = _load_spot(args.instrument)
    spot_file = ROOT / "data" / "intraday" / f"{slug}_5m.parquet"
    if spot is not None:
        spot_msg = "available"
    else:
        spot_msg = f"NOT FOUND -- run download_nifty_5m.py --instrument '{args.instrument}' --out {spot_file} first"
    print(f"[option_chain_5m] underlying spot join: {spot_msg}", flush=True)

    expiries = list_expiries(token, args.instrument)
    expiries = sorted(expiries, reverse=True)  # latest expiry first, walk backward in time
    print(f"[option_chain_5m] {args.instrument}: {len(expiries)} expired expiries found, "
          f"walking latest-first ({expiries[0]} -> {expiries[-1]})", flush=True)

    t0 = time.time()
    total_contracts = 0
    total_rows = 0
    for ei, expiry_str in enumerate(expiries):
        out_path = out_dir / f"{expiry_str}.parquet"
        if out_path.exists():
            print(f"  [{ei+1}/{len(expiries)}] {expiry_str}: already downloaded, skipping", flush=True)
            continue
        expiry = date.fromisoformat(expiry_str)
        try:
            contracts = list_contracts(token, args.instrument, expiry_str)
        except RuntimeError as exc:
            print(f"  [{ei+1}/{len(expiries)}] {expiry_str}: SKIPPED, contract list failed: {exc}", flush=True)
            continue
        time.sleep(RATE_LIMIT_SLEEP)
        leg_rows = []
        expiry_failed_chunks = 0
        expiry_ok_chunks = 0
        for c in contracts:
            bars5, n_failed, n_ok = fetch_contract_candles(token, c["instrument_key"], expiry)
            leg_rows.append((c, bars5))
            expiry_failed_chunks += n_failed
            expiry_ok_chunks += n_ok
        total_contracts += len(contracts)
        chain_df = _pivot_to_chain(leg_rows, expiry_str, spot)
        fail_rate = expiry_failed_chunks / max(1, expiry_failed_chunks + expiry_ok_chunks)
        # a nonzero-but-small failure rate is normal noise (a handful of chunks may genuinely
        # exhaust retries); a high rate means a sustained bad window (rate limit, outage) likely
        # produced a chain that LOOKS complete but is silently missing real data -- refuse to
        # checkpoint it as done so the next run retries this expiry from scratch instead
        FAIL_RATE_THRESHOLD = 0.05
        if fail_rate > FAIL_RATE_THRESHOLD:
            print(f"  [{ei+1}/{len(expiries)}] {expiry_str}: SKIPPED writing output, {expiry_failed_chunks}/"
                  f"{expiry_failed_chunks + expiry_ok_chunks} chunk requests failed ({fail_rate:.0%}) -- "
                  f"likely a sustained bad window, will retry this expiry next run", flush=True)
        elif not chain_df.empty:
            chain_df.to_parquet(out_path, index=False)
            total_rows += len(chain_df)
            elapsed_min = (time.time() - t0) / 60
            n_strikes = chain_df["strike"].nunique()
            fail_note = f", {expiry_failed_chunks} chunk failures tolerated" if expiry_failed_chunks else ""
            print(f"  [{ei+1}/{len(expiries)}] {expiry_str}: {len(contracts)} contracts -> {n_strikes} strikes, "
                  f"{len(chain_df):,} chain rows -> {out_path.name}  ({elapsed_min:.1f}min elapsed, "
                  f"{total_rows:,} rows total{fail_note})", flush=True)
        else:
            print(f"  [{ei+1}/{len(expiries)}] {expiry_str}: {len(contracts)} contracts, no candle data returned "
                  f"({expiry_failed_chunks} chunk failures, {expiry_ok_chunks} ok -- likely genuinely before this "
                  f"expiry's contracts started trading, or entirely past the data cutoff)", flush=True)

    print(f"[option_chain_5m] done: {total_contracts:,} contracts scanned, {total_rows:,} chain rows written "
          f"across {len(expiries)} expiries -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
