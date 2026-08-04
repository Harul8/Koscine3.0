"""Rank stock-option underlyings by tradability and buy-side premium movement.

This is a universe-selection diagnostic, not a predictive backtest. It samples
recent local NSE F&O bhavcopies, chooses a near-expiry ATM CE+PE pair for every
stock/day, applies minimum continuity gates, and ranks the survivors using:

    65% option liquidity + 35% robust single-leg upside movement

The output is evidence for the locked study universe. It does not mutate the
production signal universe.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "analysis"))

from options_bhavcopy import BASE, load_bhavcopy  # noqa: E402


SYMBOL_ALIASES = {
    "ZOMATO": "ETERNAL",
    "TATAMOTORS": "TMPV",
    "LTI": "LTM",
    "LTIM": "LTM",
    "INFOEDGE": "NAUKRI",
}


def available_recent_dates(start: pd.Timestamp, end: pd.Timestamp) -> list[pd.Timestamp]:
    dates: set[pd.Timestamp] = set()
    for path in BASE.rglob("*"):
        if not path.is_file():
            continue
        old = re.fullmatch(r"fo(\d{2})([A-Z]{3})(\d{4})bhav\.csv\.zip", path.name)
        new = re.search(r"_(\d{8})_F_0000\.csv(?:\.zip)?$", path.name)
        if old:
            date = pd.to_datetime("".join(old.groups()), format="%d%b%Y", errors="coerce")
        elif new:
            date = pd.to_datetime(new.group(1), format="%Y%m%d", errors="coerce")
        else:
            continue
        if pd.notna(date) and start <= date <= end:
            dates.add(pd.Timestamp(date))
    return sorted(dates)


def _atm_row(date: pd.Timestamp, symbol: str, chain: pd.DataFrame) -> dict[str, object] | None:
    expiries = sorted(
        expiry
        for expiry in pd.to_datetime(chain.expiry.dropna().unique())
        if expiry >= date + pd.Timedelta(days=7)
    )
    if not expiries:
        return None
    expiry = pd.Timestamp(expiries[0])
    near = chain[chain.expiry.eq(expiry)]
    ce = near[near.opt_type.eq("CE")]
    pe = near[near.opt_type.eq("PE")]
    common = sorted(set(ce.strike.dropna()) & set(pe.strike.dropna()))
    if not common:
        return None

    underlying = pd.to_numeric(near.underlying, errors="coerce").dropna()
    if not underlying.empty:
        spot = float(underlying.iloc[0])
    else:
        parity = ce[["strike", "close"]].merge(
            pe[["strike", "close"]], on="strike", suffixes=("_ce", "_pe")
        )
        parity = parity.replace([np.inf, -np.inf], np.nan).dropna()
        if parity.empty:
            return None
        spot = float(
            parity.iloc[
                (parity.close_ce - parity.close_pe).abs().to_numpy().argmin()
            ].strike
        )
    strike = min(common, key=lambda value: abs(float(value) - spot))
    call = ce[ce.strike.eq(strike)].iloc[0]
    put = pe[pe.strike.eq(strike)].iloc[0]

    ce_open, pe_open = float(call.open), float(put.open)
    ce_close, pe_close = float(call.close), float(put.close)
    valid_open = ce_open > 0 and pe_open > 0
    ce_vol, pe_vol = float(call.vol), float(put.vol)
    ce_oi, pe_oi = float(call.oi), float(put.oi)
    premium_turnover = max(ce_close, 0) * max(ce_vol, 0) + max(pe_close, 0) * max(
        pe_vol, 0
    )
    return {
        "date": date,
        "symbol": symbol,
        "expiry": expiry,
        "strike": float(strike),
        "pair_volume": ce_vol + pe_vol,
        "minimum_leg_volume": min(ce_vol, pe_vol),
        "pair_oi": ce_oi + pe_oi,
        "premium_turnover": premium_turnover,
        "both_legs_traded": bool(ce_vol > 0 and pe_vol > 0 and valid_open),
        # Buy-side movement: a leg's own high/open excursion is observable from
        # its OHLC candle and does not have the cross-leg timestamp ambiguity of
        # a synthetic straddle high. Close expansion discounts isolated high prints.
        "ce_peak_return": (
            max(float(call.high) / ce_open - 1, 0) if ce_open > 0 else np.nan
        ),
        "pe_peak_return": (
            max(float(put.high) / pe_open - 1, 0) if pe_open > 0 else np.nan
        ),
        "ce_positive_close_return": (
            max(ce_close / ce_open - 1, 0) if ce_open > 0 and ce_close > 0 else np.nan
        ),
        "pe_positive_close_return": (
            max(pe_close / pe_open - 1, 0) if pe_open > 0 and pe_close > 0 else np.nan
        ),
    }


def collect_daily_metrics(dates: list[pd.Timestamp]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for position, date in enumerate(dates, start=1):
        bhavcopy = load_bhavcopy(date, kinds=("STKOPT",))
        if bhavcopy.empty:
            continue
        bhavcopy["symbol"] = bhavcopy["symbol"].replace(SYMBOL_ALIASES)
        for symbol, chain in bhavcopy.groupby("symbol", sort=False):
            row = _atm_row(date, str(symbol), chain)
            if row is not None:
                rows.append(row)
        print(f"[{position:03d}/{len(dates):03d}] {date.date()} -> {bhavcopy.symbol.nunique()} stocks")
    return pd.DataFrame(rows)


def _percentile(series: pd.Series) -> pd.Series:
    return series.rank(pct=True, method="average").fillna(0)


def rank_symbols(daily: pd.DataFrame, sampled_days: int) -> pd.DataFrame:
    if daily.empty:
        raise ValueError("no daily ATM option metrics were collected")
    summary = (
        daily.groupby("symbol", as_index=False)
        .agg(
            observed_days=("date", "nunique"),
            traded_both_legs_rate=("both_legs_traded", "mean"),
            median_pair_volume=("pair_volume", "median"),
            p25_minimum_leg_volume=("minimum_leg_volume", lambda x: x.quantile(0.25)),
            median_pair_oi=("pair_oi", "median"),
            median_premium_turnover=("premium_turnover", "median"),
            median_ce_peak_return=("ce_peak_return", "median"),
            median_pe_peak_return=("pe_peak_return", "median"),
            p75_ce_peak_return=("ce_peak_return", lambda x: x.quantile(0.75)),
            p75_pe_peak_return=("pe_peak_return", lambda x: x.quantile(0.75)),
            mean_ce_positive_close_return=("ce_positive_close_return", "mean"),
            mean_pe_positive_close_return=("pe_positive_close_return", "mean"),
        )
    )
    summary["coverage_rate"] = summary.observed_days / sampled_days
    eligible = summary[
        (summary.coverage_rate >= 0.70)
        & (summary.traded_both_legs_rate >= 0.65)
        & (summary.p25_minimum_leg_volume > 0)
    ].copy()
    eligible["liquidity_score"] = (
        0.30 * _percentile(np.log1p(eligible.median_premium_turnover))
        + 0.25 * _percentile(np.log1p(eligible.median_pair_volume))
        + 0.20 * _percentile(np.log1p(eligible.p25_minimum_leg_volume))
        + 0.15 * _percentile(np.log1p(eligible.median_pair_oi))
        + 0.10 * _percentile(eligible.traded_both_legs_rate)
    )
    peak_score = (
        0.35 * _percentile(eligible.median_ce_peak_return)
        + 0.35 * _percentile(eligible.median_pe_peak_return)
        + 0.15 * _percentile(eligible.p75_ce_peak_return)
        + 0.15 * _percentile(eligible.p75_pe_peak_return)
    )
    close_score = (
        0.50 * _percentile(eligible.mean_ce_positive_close_return)
        + 0.50 * _percentile(eligible.mean_pe_positive_close_return)
    )
    eligible["movement_score"] = 0.70 * peak_score + 0.30 * close_score
    eligible["composite_score"] = (
        0.65 * eligible.liquidity_score + 0.35 * eligible.movement_score
    )
    eligible = eligible.sort_values(
        ["composite_score", "liquidity_score"], ascending=False
    ).reset_index(drop=True)
    eligible["rank"] = np.arange(1, len(eligible) + 1)
    return eligible


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2025-01-01")
    parser.add_argument("--end", default="2026-07-22")
    parser.add_argument(
        "--max-sample-days",
        type=int,
        default=80,
        help="Evenly sample at most this many trading days from the date range",
    )
    parser.add_argument("--top", type=int, default=30)
    args = parser.parse_args()

    all_dates = available_recent_dates(pd.Timestamp(args.start), pd.Timestamp(args.end))
    if not all_dates:
        raise SystemExit("no local F&O bhavcopy dates in the requested range")
    if len(all_dates) > args.max_sample_days:
        positions = np.linspace(0, len(all_dates) - 1, args.max_sample_days).round().astype(int)
        dates = [all_dates[position] for position in sorted(set(positions))]
    else:
        dates = all_dates
    daily = collect_daily_metrics(dates)
    ranked = rank_symbols(daily, len(dates))

    output = HERE / "results"
    output.mkdir(exist_ok=True)
    daily.to_parquet(output / "universe_atm_daily_sample.parquet", index=False)
    ranked.to_csv(output / "universe_rank_latest.csv", index=False)
    top = ranked.head(args.top)
    report = {
        "date_range": [args.start, args.end],
        "sampled_days": len(dates),
        "eligible_stocks": len(ranked),
        "selection_size": len(top),
        "score": "65% liquidity + 35% balanced ATM CE/PE buy-side movement",
        "top_symbols": top.symbol.tolist(),
    }
    (output / "universe_rank_latest.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    columns = [
        "rank",
        "symbol",
        "composite_score",
        "liquidity_score",
        "movement_score",
        "coverage_rate",
        "traded_both_legs_rate",
        "median_pair_volume",
        "median_ce_peak_return",
        "median_pe_peak_return",
    ]
    print(top[columns].to_string(index=False))


if __name__ == "__main__":
    main()
