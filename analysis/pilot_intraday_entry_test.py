"""PILOT: fetch the real intraday path of every leg of the Broken-Wing trades the backtest
actually fired, so we can answer -- with data rather than assumption -- two open questions:

  1. Is the 09:15 OPEN print the backtest sells at actually TRADEABLE, or a stale/wide opening
     auction artifact? 82% of the strategy's recorded PnL is the day-1 open->close reversion
     (see the 2026-08 analysis), so this single question decides whether the strategy's
     profitability is real or an artifact of unfillable prints.
  2. What would a ~14:30 signal / ~15:15 entry (the "trade the same day, before the overnight
     gap" idea) actually have collected, versus the next-morning open the backtest assumes?

Scope is deliberately small: only the legs of trades that actually fired, only on their own
entry date -- ~960 contract-days, not a full chain history. Upstox expired-instrument data only
reaches ~Oct 2024, so pre-Oct-2024 trades are skipped (reported, not silently dropped).

Writes data/intraday/pilot_bw_legs_1m.parquet (one row per leg per minute). Resumable: legs
already present in the output file are skipped on re-run.

Usage:
    python analysis/pilot_intraday_entry_test.py
    python analysis/pilot_intraday_entry_test.py --limit 40      # smaller smoke run
"""
from __future__ import annotations

import argparse
import gzip
import io
import json
import os
import socket
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

import pandas as pd

ROOT = Path(r"C:\Users\rahul\Koscine 3.0")
LOCK = ROOT / "locks" / "prod_sell_strategies"
OUT = ROOT / "data" / "intraday" / "pilot_bw_legs_1m.parquet"
V2 = "https://api.upstox.com/v2"
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/124.0.0.0 Safari/537.36")

# same pacing lessons as analysis/download_nifty_option_chain_5m.py: Upstox enforces a sustained
# quota, not just a burst guard -- 429 needs a seconds-scale backoff, not a sub-3s one
RATE_SLEEP = 0.4
NET_RETRIES = 4
RL_RETRIES = 6
EXPIRED_DATA_CUTOFF = "2024-10-01"

MCAP_MIN_ROR, OTHER_MIN_ROR = 150.0, 155.0
MIN_PROFIT_PER_LOT = 10_000.0


def _get(url: str, token: str) -> dict:
    req = Request(url, headers={"Accept": "application/json",
                                "Authorization": f"Bearer {token}", "User-Agent": UA})
    attempt = rl = 0
    while attempt < NET_RETRIES and rl < RL_RETRIES:
        try:
            with urlopen(req, timeout=60) as r:
                return json.loads(r.read().decode())
        except HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            if e.code == 429:
                ra = e.headers.get("Retry-After") if e.headers else None
                wait = float(ra) if ra and ra.isdigit() else min(3 * (2 ** rl), 60)
                rl += 1
                if rl < RL_RETRIES:
                    time.sleep(wait)
                    continue
            raise RuntimeError(f"HTTP {e.code}: {body[:200]}") from e
        except (URLError, ConnectionResetError, TimeoutError, socket.timeout, OSError) as e:
            attempt += 1
            if attempt < NET_RETRIES:
                time.sleep(0.5 * attempt)
                continue
            raise RuntimeError(f"network: {e}") from e
    raise RuntimeError("unreachable")


def equity_keys() -> dict[str, str]:
    req = Request("https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz",
                  headers={"User-Agent": UA})
    with urlopen(req, timeout=180) as r:
        raw = gzip.GzipFile(fileobj=io.BytesIO(r.read())).read()
    out = {}
    for i in json.loads(raw):
        if i.get("segment") == "NSE_EQ" and i.get("instrument_type") == "EQ":
            ts = i.get("trading_symbol") or i.get("tradingsymbol")
            if ts:
                out[ts] = i["instrument_key"]
    return out


def fired_trades() -> pd.DataFrame:
    """Replay the real selection rule -> the trades history actually logged."""
    bw = pd.read_parquet(LOCK / "broken_wing_candidates_raw.parquet")
    bar = bw["group"].eq("A_mcap30").map({True: MCAP_MIN_ROR, False: OTHER_MIN_ROR})
    ok = bw[(bw["entry_ror_pct"] > bar) & (bw["max_profit_per_lot"] >= MIN_PROFIT_PER_LOT)]
    sel = ok.loc[ok.groupby(["tier", "entry_date"])["entry_ror_pct"].idxmax()]
    return sel[sel["outcome"] != "pending"].reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="only the N most recent trades")
    args = ap.parse_args()

    token = os.environ.get("UPSTOX_ACCESS_TOKEN")
    if not token:
        raise SystemExit("UPSTOX_ACCESS_TOKEN is not set")

    sel = fired_trades()
    n_all = len(sel)
    sel = sel[sel["entry_date"] >= EXPIRED_DATA_CUTOFF].copy()
    print(f"[pilot] {n_all} fired trades; {len(sel)} on/after {EXPIRED_DATA_CUTOFF} "
          f"(Upstox expired-data cutoff), {n_all - len(sel)} skipped as unreachable", flush=True)
    sel = sel.sort_values("entry_date", ascending=False)
    if args.limit:
        sel = sel.head(args.limit)
        print(f"[pilot] --limit {args.limit} -> {len(sel)} trades", flush=True)

    eq = equity_keys()
    print(f"[pilot] {len(eq):,} NSE_EQ keys loaded", flush=True)

    done: set[tuple] = set()
    frames: list[pd.DataFrame] = []
    if OUT.exists():
        prev = pd.read_parquet(OUT)
        frames.append(prev)
        done = set(map(tuple, prev[["symbol", "expiry", "strike", "opt_type", "entry_date"]]
                       .drop_duplicates().astype(str).values))
        print(f"[pilot] resuming: {len(done)} leg-days already downloaded", flush=True)

    contracts_cache: dict[tuple, dict] = {}

    def contract_key(sym, expiry, strike, ot):
        ck = (sym, expiry)
        if ck not in contracts_cache:
            ikey = eq.get(sym)
            if not ikey:
                contracts_cache[ck] = {}
            else:
                try:
                    r = _get(f"{V2}/expired-instruments/option/contract"
                             f"?instrument_key={quote(ikey, safe='')}&expiry_date={expiry}", token)
                    contracts_cache[ck] = {
                        (float(c["strike_price"]), c["instrument_type"]): c["instrument_key"]
                        for c in (r.get("data") or [])}
                except RuntimeError as e:
                    print(f"    contract-list failed {sym} {expiry}: {e}", flush=True)
                    contracts_cache[ck] = {}
                time.sleep(RATE_SLEEP)
        return contracts_cache[ck].get((float(strike), ot))

    t0 = time.time()
    n_legs = n_ok = n_miss = 0
    for ti, (_, tr) in enumerate(sel.iterrows()):
        sym, exp, d = tr["symbol"], str(tr["expiry"]), str(tr["entry_date"])
        legs = [("CE", tr["short_ce"], "short_ce"), ("CE", tr["long_ce"], "long_ce"),
                ("PE", tr["short_pe"], "short_pe"), ("PE", tr["long_pe"], "long_pe")]
        for ot, strike, role in legs:
            n_legs += 1
            # NOTE: different tiers routinely share legs -- t1_2x6 and t3_2x10 both use a 2% call
            # wing, so their short_ce/long_ce strikes are identical. Without marking each leg-day
            # done as soon as it's fetched (not just the ones loaded from a previous run), the
            # same contract-day gets re-requested and appended two or three times.
            leg_id = (sym, exp, str(float(strike)), ot, d)
            if leg_id in done:
                continue
            done.add(leg_id)
            ik = contract_key(sym, exp, strike, ot)
            if not ik:
                n_miss += 1
                continue
            try:
                r = _get(f"{V2}/expired-instruments/historical-candle/"
                         f"{quote(ik, safe='')}/1minute/{d}/{d}", token)
                candles = (r.get("data") or {}).get("candles") or []
            except RuntimeError as e:
                print(f"    candles failed {sym} {strike}{ot} {d}: {e}", flush=True)
                candles = []
            time.sleep(RATE_SLEEP)
            if not candles:
                n_miss += 1
                continue
            df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low",
                                                "close", "volume", "oi"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("Asia/Kolkata")
            df["symbol"], df["expiry"], df["strike"] = sym, exp, float(strike)
            # `role` is per-trade, not per-contract (the same leg can be short_ce for one tier
            # and long_ce for another) -- it is deliberately NOT stored, since the output is a
            # per-contract-day price series that the analysis joins back onto trades by
            # (symbol, expiry, strike, opt_type, entry_date).
            df["opt_type"], df["entry_date"] = ot, d
            frames.append(df)
            n_ok += 1
        if (ti + 1) % 20 == 0:
            pd.concat(frames, ignore_index=True).to_parquet(OUT, index=False)
            el = (time.time() - t0) / 60
            print(f"  [{ti+1}/{len(sel)}] trades | legs ok={n_ok} miss={n_miss} "
                  f"| {el:.1f}min elapsed -> checkpoint", flush=True)

    if frames:
        pd.concat(frames, ignore_index=True).to_parquet(OUT, index=False)
    print(f"[pilot] done: {n_ok} leg-days downloaded, {n_miss} unavailable, "
          f"of {n_legs} legs across {len(sel)} trades -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
