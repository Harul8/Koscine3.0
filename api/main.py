from __future__ import annotations

import subprocess
import sys
import os
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import BackgroundTasks, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from scipy.optimize import brentq
from scipy.stats import norm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from functools import lru_cache  # noqa: E402

from koscine3.paths import RUNS_DIR  # noqa: E402
from koscine3.evaluation.gold_metrics import build_gold_report  # noqa: E402
from koscine3.largemove.config import LOCK_DIR as LM_LOCK, PREDICTIONS_DIR as LM_PRED  # noqa: E402
from koscine3.data.sources import load_market_data  # noqa: E402


_OHLC_CACHE: dict = {"mtime": None, "df": None, "signs": None}


def _market_ohlc() -> pd.DataFrame:
    """Cached OHLC panel, auto-invalidated when the feature parquet is rewritten (daily refresh) — so the API
    serves fresh data without a restart."""
    from koscine3.data.sources import read_data_source
    try:
        mt = os.path.getmtime(read_data_source().path)
    except OSError:
        mt = None
    if _OHLC_CACHE["df"] is None or _OHLC_CACHE["mtime"] != mt:
        df = load_market_data(columns=["date", "symbol", "open", "high", "low", "close", "volume", "delivery_qty"])
        df["symbol"] = df["symbol"].astype(str)
        df["date"] = pd.to_datetime(df["date"])
        _OHLC_CACHE.update(mtime=mt, df=df, signs=None)
    return _OHLC_CACHE["df"]


app = FastAPI(title="Koscine 3.0 API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _run_dirs() -> list[Path]:
    if not RUNS_DIR.exists():
        return []
    return sorted(
        [p for p in RUNS_DIR.iterdir() if p.is_dir()],
        key=lambda p: (not (p / "model_predictions").exists(), -p.stat().st_mtime),
    )


def _resolve_run(run_id: str | None = None) -> Path:
    if run_id:
        run_dir = RUNS_DIR / run_id
        if not run_dir.exists():
            raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")
        return run_dir
    runs = _run_dirs()
    if not runs:
        raise HTTPException(status_code=404, detail="No Koscine 3.0 runs found")
    return runs[0]


def _read_json(path: Path) -> dict[str, object]:
    import json

    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_signals(run_dir: Path) -> pd.DataFrame:
    model_frames = [pd.read_parquet(p) for p in run_dir.glob("model_predictions/*/signals.parquet")]
    if model_frames:
        return pd.concat(model_frames, ignore_index=True)
    combined = run_dir / "all_signals.parquet"
    if combined.exists():
        return pd.read_parquet(combined)
    frames = [pd.read_parquet(p) for p in run_dir.glob("**/signals.parquet")]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _load_metrics(run_dir: Path) -> dict[str, list[dict[str, object]]]:
    signals = _load_signals(run_dir)
    if not signals.empty:
        report = build_gold_report(signals)
        return {name: _records(table) for name, table in report.items()}
    report_dir = run_dir / "combined_gold_report"
    if not report_dir.exists():
        report_dir = run_dir / "model_predictions"
    tables: dict[str, list[dict[str, object]]] = {}
    if report_dir.name == "combined_gold_report":
        for csv_path in report_dir.glob("*.csv"):
            tables[csv_path.stem] = pd.read_csv(csv_path).to_dict(orient="records")
    else:
        for csv_path in run_dir.glob("**/gold_report/*.csv"):
            key = "_".join(csv_path.parts[-4:]).replace(".csv", "")
            tables[key] = pd.read_csv(csv_path).to_dict(orient="records")
    return tables


def _records(table: pd.DataFrame) -> list[dict[str, object]]:
    out = table.copy()
    for column in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[column]):
            out[column] = out[column].dt.strftime("%Y-%m-%d")
    return out.where(pd.notna(out), None).to_dict(orient="records")


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "project_root": str(PROJECT_ROOT),
        "run_count": len(_run_dirs()),
    }


@app.get("/runs")
def runs() -> list[dict[str, object]]:
    payload = []
    for run_dir in _run_dirs():
        manifest = _read_json(run_dir / "manifest.json")
        payload.append(
            {
                "run_id": run_dir.name,
                "path": str(run_dir),
                "modified": run_dir.stat().st_mtime,
                "manifest": manifest,
            }
        )
    return payload


@app.get("/dates")
def dates(run_id: str | None = None) -> list[str]:
    run_dir = _resolve_run(run_id)
    signals = _load_signals(run_dir)
    if signals.empty:
        return []
    if "selected" in signals.columns:
        signals = signals[signals["selected"]]
    return sorted(pd.to_datetime(signals["date"]).dt.strftime("%Y-%m-%d").unique().tolist())


@app.get("/signals")
def signals(
    date: str | None = Query(default=None),
    run_id: str | None = Query(default=None),
    selected_only: bool = Query(default=True),
    limit: int = Query(default=500, ge=1, le=5000),
) -> list[dict[str, object]]:
    run_dir = _resolve_run(run_id)
    table = _load_signals(run_dir)
    if table.empty:
        return []
    table["date"] = pd.to_datetime(table["date"])
    if selected_only and "selected" in table.columns:
        table = table[table["selected"]]
    if date:
        table = table[table["date"].eq(pd.Timestamp(date))]
    table = table.sort_values(["date", "utility_score"], ascending=[False, False]).head(limit)
    return _records(table)


@app.get("/runs/{run_id}/metrics")
def metrics(run_id: str) -> dict[str, list[dict[str, object]]]:
    run_dir = _resolve_run(run_id)
    return _load_metrics(run_dir)


@app.get("/runs/{run_id}/signals")
def run_signals(run_id: str, limit: int = Query(default=1000, ge=1, le=10000)) -> list[dict[str, object]]:
    run_dir = _resolve_run(run_id)
    table = _load_signals(run_dir)
    if table.empty:
        return []
    table = table.sort_values(["date", "utility_score"], ascending=[False, False]).head(limit)
    return _records(table)


# ----------------------------------------------------------------------------
# PRODUCTION large-move engine (locked: locks/prod_largemove_v1) — daily shortlist
# ----------------------------------------------------------------------------
def _lm_shortlist() -> pd.DataFrame:
    f = LM_PRED / "combined_shortlist.csv"
    if not f.exists():
        return pd.DataFrame()
    df = pd.read_csv(f)
    df["date"] = pd.to_datetime(df["date"])
    return df


@app.get("/prod/manifest")
def prod_manifest() -> dict[str, object]:
    """Locked config + walk-forward metrics snapshot."""
    return _read_json(LM_LOCK / "manifest.json")


@app.get("/prod/dates")
def prod_dates() -> list[str]:
    """Trading dates present in the walk-forward shortlist (descending — latest first)."""
    df = _lm_shortlist()
    if df.empty:
        return []
    return sorted(df["date"].dt.strftime("%Y-%m-%d").unique().tolist(), reverse=True)


@app.get("/prod/shortlist")
def prod_shortlist(
    date: str | None = Query(default=None),
    group: str | None = Query(default=None),
) -> list[dict[str, object]]:
    """Ranked daily picks (top-2/group, t+3 cooldown). Each row carries the realized
    outcome (actual_move_%, hit) since these are out-of-sample walk-forward predictions."""
    df = _lm_shortlist()
    if df.empty:
        return []
    if date:
        df = df[df["date"].eq(pd.Timestamp(date))]
    if group:
        df = df[df["group"].eq(group)]
    df = df.sort_values(["date", "group", "confidence"], ascending=[False, True, False])
    return _records(df)


@app.get("/prod/group/{group}")
def prod_group_predictions(
    group: str,
    top_per_day: int = Query(default=0, ge=0, le=50),
    limit: int = Query(default=2000, ge=1, le=50000),
) -> list[dict[str, object]]:
    """Full eligible-universe walk-forward predictions for a group (for deeper inspection).
    top_per_day>0 restricts to the N most-confident picks per day."""
    f = LM_PRED / f"group_{group}_predictions.csv"
    if not f.exists():
        raise HTTPException(status_code=404, detail=f"Unknown group: {group}")
    df = pd.read_csv(f)
    df["date"] = pd.to_datetime(df["date"])
    if top_per_day:
        df = df[df["rank_in_day"] <= top_per_day]
    df = df.sort_values(["date", "confidence"], ascending=[False, False]).head(limit)
    return _records(df)


# ----------------------------------------------------------------------------
# PRODUCTION v2 — direction-agnostic large-mover book (locks/prod_largemove_v2)
# ----------------------------------------------------------------------------
LM_LOCK_V2 = LM_LOCK.parent / "prod_largemove_v2"


def _v2_book() -> pd.DataFrame:
    f = LM_LOCK_V2 / "book_2024_26.csv"
    if not f.exists():
        return pd.DataFrame()
    df = pd.read_csv(f)
    df["date"] = pd.to_datetime(df["date"])
    df["symbol"] = df["symbol"].astype(str)
    # rank within (date, group) by atm_iv = the pick order; flag live (no outcome yet)
    df["pick_rank"] = df.groupby(["date", "group"])["atm_iv"].rank(ascending=False, method="first")
    df["live"] = df["move_mag"].isna()
    # direction overlay (informational; does not affect selection)
    od = LM_LOCK_V2 / "direction_overlay.csv"
    if od.exists():
        o = pd.read_csv(od)
        o["date"] = pd.to_datetime(o["date"]); o["symbol"] = o["symbol"].astype(str)
        df = df.merge(o[["date", "symbol", "group", "p_up", "dir_label", "confidence", "conf_tier"]],
                      on=["date", "symbol", "group"], how="left")
    # near-ATM option premium OHLC over the 5-day window
    pf = LM_LOCK_V2 / "book_premiums.csv"
    if pf.exists():
        p = pd.read_csv(pf)
        p["date"] = pd.to_datetime(p["date"]); p["symbol"] = p["symbol"].astype(str)
        cols = ["date", "symbol", "group", "strike", "ce_entry", "ce_high", "ce_low", "ce_close",
                "pe_entry", "pe_high", "pe_low", "pe_close", "ce_mult_best", "pe_mult_best"]
        df = df.merge(p[cols], on=["date", "symbol", "group"], how="left")
    return df


@app.get("/prod2/manifest")
def prod2_manifest() -> dict[str, object]:
    """v2 locked config + book metrics (direction-agnostic IV mover-picker)."""
    return _read_json(LM_LOCK_V2 / "manifest.json")


@app.get("/prod2/dates")
def prod2_dates() -> list[str]:
    """Trading dates in the v2 book (descending — latest first; latest are live, outcome pending)."""
    df = _v2_book()
    if df.empty:
        return []
    return sorted(df["date"].dt.strftime("%Y-%m-%d").unique().tolist(), reverse=True)


@app.get("/prod2/book")
def prod2_book(
    date: str | None = Query(default=None),
    group: str | None = Query(default=None),
) -> list[dict[str, object]]:
    """v2 daily mover book: top-3/group by atm_iv (t+3 cooldown + cap), direction-agnostic.
    Historical rows carry the realized 5-day |move| (move_mag_pct) + whipsaw flag (closed_opp);
    the latest rows are live (move_mag null = outcome pending)."""
    df = _v2_book()
    if df.empty:
        return []
    if date:
        df = df[df["date"].eq(pd.Timestamp(date))]
    if group:
        df = df[df["group"].eq(group)]
    df = df.sort_values(["date", "group", "atm_iv"], ascending=[False, True, False])
    return _records(df)


@app.get("/prod2/symbols")
def prod2_symbols() -> list[dict[str, object]]:
    """Universe symbols (with group) for the stock/price-history selectors."""
    groups = _read_json(LM_LOCK_V2 / "universe_groups.json")
    out = [{"symbol": s, "group": g} for g, syms in groups.items() for s in syms]
    return sorted(out, key=lambda r: (r["group"], r["symbol"]))


@app.get("/prod2/stock_history")
def prod2_stock_history(symbol: str) -> list[dict[str, object]]:
    """Every v2 pick for a symbol (signal history) with move, direction overlay and premium."""
    df = _v2_book()
    if df.empty:
        return []
    df = df[df["symbol"].eq(symbol.upper())].sort_values("date", ascending=False)
    return _records(df)


# ---- Sell Strategies (defined-risk premium selling) -------------------------------------------
_SELL_CACHE: dict = {"mtime": None, "contracts": None, "iv": None}
_RISK_FREE_RATE = 0.065  # flat India risk-free proxy; good enough for short-dated near-ATM IV


def _bs_price(spot: float, strike: float, years: float, sigma: float, is_call: bool) -> float:
    if sigma <= 0 or years <= 0:
        return max(0.0, (spot - strike) if is_call else (strike - spot))
    d1 = (np.log(spot / strike) + (_RISK_FREE_RATE + 0.5 * sigma ** 2) * years) / (sigma * np.sqrt(years))
    d2 = d1 - sigma * np.sqrt(years)
    if is_call:
        return spot * norm.cdf(d1) - strike * np.exp(-_RISK_FREE_RATE * years) * norm.cdf(d2)
    return strike * np.exp(-_RISK_FREE_RATE * years) * norm.cdf(-d2) - spot * norm.cdf(-d1)


def _implied_vol(price: float, spot: float, strike: float, years: float, is_call: bool) -> float | None:
    """Black-Scholes implied vol via Brent's method; None if unsolvable (price below intrinsic, etc)."""
    intrinsic = max(0.0, (spot - strike) if is_call else (strike - spot))
    if price <= intrinsic + 1e-6 or years <= 0:
        return None
    try:
        return float(brentq(lambda s: _bs_price(spot, strike, years, s, is_call) - price, 1e-4, 6.0, xtol=1e-4))
    except ValueError:
        return None


def _opt_price(row) -> float:  # leg mark: prefer close, fall back to settle
    p = row.get("close")
    return float(p) if pd.notna(p) and p > 0 else (float(row["settle"]) if pd.notna(row.get("settle")) else np.nan)


def _sell_sources():
    """Latest per-strike option chain (eod_deriv_contracts) + per-stock ATM-IV history, cached."""
    import glob
    from koscine.config import SILVER_DATA_ROOT
    root = SILVER_DATA_ROOT / "eod_deriv_contracts"
    files = sorted(glob.glob(str(root / "**" / "*.parquet"), recursive=True))
    if not files:
        return None, None
    mt = max(os.path.getmtime(f) for f in files)
    if _SELL_CACHE["mtime"] != mt:
        c = pd.concat([pd.read_parquet(f) for f in files[-2:]], ignore_index=True)
        c["date"] = pd.to_datetime(c["date"]); c["expiry"] = pd.to_datetime(c["expiry"])
        dd = pd.read_parquet(SILVER_DATA_ROOT / "eod_deriv_daily.parquet", columns=["date", "symbol", "atm_iv"])
        dd["date"] = pd.to_datetime(dd["date"]); dd["symbol"] = dd["symbol"].astype(str)
        dd = dd.sort_values(["symbol", "date"])
        dd["iv_ratio"] = dd.groupby("symbol")["atm_iv"].transform(
            lambda s: s / s.rolling(252, min_periods=60).median())
        _SELL_CACHE.update(mtime=mt, contracts=c, iv=dd)
    return _SELL_CACHE["contracts"], _SELL_CACHE["iv"]


_LOT_SIZE_CACHE: dict = {"mtime": None, "latest": None}


def _lot_sizes() -> dict:
    """symbol -> current lot size (shares per contract), from the most recent expiry_month on
    record for that symbol. NSE revises lot sizes periodically; this is the best current estimate."""
    from koscine.config import SILVER_DATA_ROOT
    f = SILVER_DATA_ROOT / "lot_size.parquet"
    if not f.exists():
        return {}
    mt = os.path.getmtime(f)
    if _LOT_SIZE_CACHE["mtime"] != mt:
        df = pd.read_parquet(f, columns=["symbol", "expiry_month", "lot"])
        latest = df.sort_values("expiry_month").groupby("symbol")["lot"].last().to_dict()
        _LOT_SIZE_CACHE.update(mtime=mt, latest=latest)
    return _LOT_SIZE_CACHE["latest"]


CONDOR_DTE_MIN, CONDOR_MIN_ROR = 9, 150.0   # safety floor (delivery-margin) + THE quality gate
MIN_RISK_FRAC = 0.10   # max_risk must be >= 10% of wing width; below that, credit~=width and the
                        # ror_pct ratio becomes numerically degenerate (blows toward infinity)


@app.get("/prod2/sell_strategies")
def prod2_sell_strategies(short_otm: float = Query(0.02, ge=0.01, le=0.08),
                          wing: float = Query(0.05, ge=0.01, le=0.10),
                          dte_min: int = Query(CONDOR_DTE_MIN, ge=0, le=30),
                          min_ror: float = Query(CONDOR_MIN_ROR, ge=0, le=1000)) -> dict[str, object]:
    """Live DEFINED-RISK iron-condor candidates on the A/B universe: sell ~short_otm% OTM CE+PE,
    buy the +wing% wings (loss always capped), on the nearest available expiry. wing=5% (not 3%):
    tested wider, similar win rate but meaningfully higher absolute rupee profit per lot. Entry
    requires DTE >= dte_min (NSE ITM delivery-margin ramps E-4..expiry: 10/25/45/70/100%+ of
    contract value -- dte_min=9 keeps a full 5-day hold clear of that window) AND entry-time
    credit/max_risk (ror_pct) > min_ror -- THE gating criterion, knowable before the trade
    (unlike a realized/backtested outcome). Ranked by ror_pct; top_picks is #1 (the backtested
    1-pick/day rule) plus #2-#3 as additional options. Backtest context (2y, pre-cost) is embedded."""
    contracts, iv = _sell_sources()
    if contracts is None:
        return {"as_of": None, "candidates": [], "note": "no option-chain data"}
    g2 = {s: g for g, syms in _read_json(LM_LOCK_V2 / "universe_groups.json").items() for s in syms}
    lots = _lot_sizes()
    last = contracts["date"].max()
    universe = _liquid_universe_today() or set(g2)
    day = contracts[(contracts["date"] == last) & contracts["opt_type"].isin(["CE", "PE"])
                    & contracts["symbol"].isin(universe) & contracts["strike"].notna()
                    & (contracts["underlying_price"] > 0)].copy()
    ivlast = iv[iv["date"] == last].set_index("symbol")["iv_ratio"].to_dict()

    out = []
    for sym, sg in day.groupby("symbol"):
        # roll forward: earliest expiry that already clears the DTE safety floor, not the nearest
        # calendar expiry -- avoids going quiet in the days right before each expiry.
        candidate_exps = sorted(sg["expiry"].unique())
        exp = next((e for e in candidate_exps if (pd.Timestamp(e) - last).days >= dte_min), None)
        if exp is None:
            continue
        chain = sg[sg["expiry"] == exp]
        u = float(chain["underlying_price"].iloc[0])
        dte = int((exp - last).days)
        ce = chain[chain["opt_type"] == "CE"]; pe = chain[chain["opt_type"] == "PE"]
        if ce.empty or pe.empty:
            continue
        def nearest(df, target):
            r = df.iloc[(df["strike"] - target).abs().argmin()]
            return r
        sce = nearest(ce, u * (1 + short_otm)); lce = nearest(ce, u * (1 + short_otm + wing))
        spe = nearest(pe, u * (1 - short_otm)); lpe = nearest(pe, u * (1 - short_otm - wing))
        psce, plce, pspe, plpe = _opt_price(sce), _opt_price(lce), _opt_price(spe), _opt_price(lpe)
        if any(np.isnan(x) for x in (psce, plce, pspe, plpe)):
            continue
        credit = (psce + pspe) - (plce + plpe)
        width = max(float(lce["strike"] - sce["strike"]), float(spe["strike"] - lpe["strike"]))
        risk = width - credit
        if credit <= 0 or risk <= 0 or risk < MIN_RISK_FRAC * width:
            continue
        ivr = ivlast.get(sym)
        lot = lots.get(sym)
        ce_credit, pe_credit = psce - plce, pspe - plpe   # per-side contribution to total credit
        richer_side = "CE" if ce_credit >= pe_credit else "PE"

        def oi_lots(row):
            v = row.get("open_int")
            if v is None or pd.isna(v) or lot is None or pd.isna(lot) or lot == 0:
                return None
            return round(float(v) / float(lot))

        out.append({
            "symbol": sym, "group": g2.get(sym, "C_liquid"), "expiry": exp.date().isoformat(), "dte": dte,
            "underlying": round(u, 1), "iv_ratio": round(float(ivr), 2) if ivr is not None and pd.notna(ivr) else None,
            "short_ce": float(sce["strike"]), "long_ce": float(lce["strike"]),
            "short_pe": float(spe["strike"]), "long_pe": float(lpe["strike"]),
            "oi_short_ce": oi_lots(sce), "oi_long_ce": oi_lots(lce),
            "oi_short_pe": oi_lots(spe), "oi_long_pe": oi_lots(lpe),
            "sell_premium": round(psce + pspe, 2), "buy_premium": round(plce + plpe, 2),
            "credit": round(credit, 2), "max_risk": round(risk, 2), "max_profit": round(credit, 2),
            "lot_size": int(lot) if lot is not None and pd.notna(lot) else None,
            "max_risk_per_lot": round(risk * lot, 1) if lot is not None and pd.notna(lot) else None,
            "max_profit_per_lot": round(credit * lot, 1) if lot is not None and pd.notna(lot) else None,
            "ror_pct": round(credit / risk * 100, 1),
            "be_low": round(float(spe["strike"]) - credit, 1), "be_high": round(float(sce["strike"]) + credit, 1),
            "richer_side": richer_side, "ce_credit": round(ce_credit, 2), "pe_credit": round(pe_credit, 2),
            "in_window": bool(dte >= dte_min and credit / risk * 100 > min_ror),
        })
    out.sort(key=lambda r: (r["in_window"], r["ror_pct"]), reverse=True)
    in_window = [c for c in out if c["in_window"]]
    # single best-of-day pick (across both groups) -- backtested at ~4.3/week, 97.8% win, +69.6%
    # mean ROR, worst -21.9% (vs the full gated pool's 89.8% win / +32.5% mean / worst -37.2%)
    # #1 is the backtested rule (1 pick/day); #2-#3 are additional options, not part of that backtest
    top_picks = sorted(in_window, key=lambda c: c["ror_pct"], reverse=True)[:3]
    return {
        "as_of": last.date().isoformat(),
        "params": {"short_otm": short_otm, "wing": wing, "dte_min": dte_min, "min_ror": min_ror},
        "backtest": {"window": "2024-08..2026-08 (2y, short 2% OTM / wing +5%, DTE>=9 w/ roll-forward, entry ror>150%, pre-cost)",
                     "ev_on_risk": 0.325, "win_rate": 0.898, "worst": "-0.37x (capped)",
                     "best_of_day": {"window": "same 2y, 1 pick/day (highest ror_pct), capped 10/week",
                                    "n": 445, "ev_on_risk": 0.696, "win_rate": 0.978, "worst": "-0.22x (capped)",
                                    "per_week": 4.3, "median_pnl_per_lot": 4121, "note": "wing widened from 3%->5%: "
                                    "similar win rate, meaningfully higher absolute Rs profit per lot"}},
        "candidates": out,
        "top_picks": top_picks,
    }


SKEW_DTE_MIN, SKEW_MIN_ROR = 15, 150.0   # safety floor (delivery-margin) + THE quality gate
# Three tiers mirroring Broken-Wing's design: wider wing -> more credit banked, fewer signals.
# t1 is the everyday base tier (>=~Rs.7k median PnL/lot floor), t2/t3 progressively rarer.
# 10% wing tested and found to plateau exactly at 8%'s payout with fewer signals -- not used.
SKEW_TIERS = [
    {"id": "t1_skew35", "label": "Tier 1 (3.5% wing)", "wing": 0.035,
     "backtest": {"per_year": 126, "median_pnl_per_lot": 7175, "win_rate": 1.0, "pct_trades_over_10k": 24.3}},
    {"id": "t2_skew6", "label": "Tier 2 (6% wing)", "wing": 0.06,
     "backtest": {"per_year": 12, "median_pnl_per_lot": 13681, "win_rate": 1.0, "pct_trades_over_10k": 62.5}},
    {"id": "t3_skew8", "label": "Tier 3 (8% wing)", "wing": 0.08,
     "backtest": {"per_year": 4, "median_pnl_per_lot": 11988, "win_rate": 1.0, "pct_trades_over_10k": 55.6}},
]
SKEW_NOTE = ("Wider wing -> more credit banked, fewer signals clear the gate (same pattern as "
             "the Broken-Wing Condor). De-duplicated against Broken-Wing: a skew candidate is "
             "dropped if the same symbol already has a Broken-Wing signal today -- skew is a "
             "secondary/complementary signal, not a duplicate (roughly a third to two-thirds of "
             "raw skew candidates overlap with Broken-Wing depending on tier, so de-dup removes "
             "a meaningful chunk -- Tier 3 is down to n=9 over 2y post-dedup, thin enough that "
             "its median should be treated as directional, not precise). 100% win rate in the "
             "backtest is flagged unverified/small-sample -- not a guarantee. Gross of costs/STT.")


def _skew_candidates(day: pd.DataFrame, short_otm: float, wing: float, dte_min: float, min_ror: float,
                     g2: dict, lots: dict, ivlast: dict, exclude_symbols: set) -> list[dict]:
    out = []
    for sym, sg in day.groupby("symbol"):
        if sym in exclude_symbols:
            continue
        candidate_exps = sorted(sg["expiry"].unique())
        exp = next((e for e in candidate_exps if (pd.Timestamp(e) - day["date"].iloc[0]).days >= dte_min), None)
        if exp is None:
            continue
        chain = sg[sg["expiry"] == exp]
        u = float(chain["underlying_price"].iloc[0])
        dte = int((exp - day["date"].iloc[0]).days)
        ce = chain[chain["opt_type"] == "CE"]; pe = chain[chain["opt_type"] == "PE"]
        if ce.empty or pe.empty:
            continue
        def nearest(df, target):
            return df.iloc[(df["strike"] - target).abs().argmin()]
        sce = nearest(ce, u * (1 + short_otm)); lce = nearest(ce, u * (1 + short_otm + wing))
        spe = nearest(pe, u * (1 - short_otm)); lpe = nearest(pe, u * (1 - short_otm - wing))
        psce, plce, pspe, plpe = _opt_price(sce), _opt_price(lce), _opt_price(spe), _opt_price(lpe)
        if any(np.isnan(x) for x in (psce, plce, pspe, plpe)):
            continue
        years = dte / 365.0
        ce_iv = _implied_vol(psce, u, float(sce["strike"]), years, True)
        pe_iv = _implied_vol(pspe, u, float(spe["strike"]), years, False)
        if ce_iv is None or pe_iv is None:
            continue
        skew = ce_iv - pe_iv
        side = "CE" if skew > 0 else "PE"
        if side == "CE":
            short_leg, long_leg, sell_premium, buy_premium = sce, lce, psce, plce
        else:
            short_leg, long_leg, sell_premium, buy_premium = spe, lpe, pspe, plpe
        credit = sell_premium - buy_premium
        width = abs(float(long_leg["strike"]) - float(short_leg["strike"]))
        risk = width - credit
        if credit <= 0 or risk <= 0 or risk < MIN_RISK_FRAC * width:
            continue
        ivr = ivlast.get(sym)
        lot = lots.get(sym)
        breakeven = float(short_leg["strike"]) + credit if side == "CE" else float(short_leg["strike"]) - credit
        out.append({
            "symbol": sym, "group": g2.get(sym, "C_top50oi"), "expiry": exp.date().isoformat(), "dte": dte,
            "underlying": round(u, 1), "iv_ratio": round(float(ivr), 2) if ivr is not None and pd.notna(ivr) else None,
            "side": side, "ce_iv": round(ce_iv, 3), "pe_iv": round(pe_iv, 3), "skew": round(skew, 3),
            "short_strike": float(short_leg["strike"]), "long_strike": float(long_leg["strike"]),
            "sell_premium": round(sell_premium, 2), "buy_premium": round(buy_premium, 2),
            "credit": round(credit, 2), "max_risk": round(risk, 2), "max_profit": round(credit, 2),
            "lot_size": int(lot) if lot is not None and pd.notna(lot) else None,
            "max_risk_per_lot": round(risk * lot, 1) if lot is not None and pd.notna(lot) else None,
            "max_profit_per_lot": round(credit * lot, 1) if lot is not None and pd.notna(lot) else None,
            "ror_pct": round(credit / risk * 100, 1), "breakeven": round(breakeven, 1),
            "in_window": bool(dte >= dte_min and credit / risk * 100 > min_ror),
        })
    return out


@app.get("/prod2/skew_strategy")
def prod2_skew_strategy(short_otm: float = Query(0.02, ge=0.01, le=0.08),
                        dte_min: int = Query(SKEW_DTE_MIN, ge=0, le=30),
                        min_ror: float = Query(SKEW_MIN_ROR, ge=0, le=1000)) -> dict[str, object]:
    """Live DEFINED-RISK single-side credit spread, evaluated at THREE wing-width tiers
    simultaneously (mirroring Broken-Wing's tiering): back out BS implied vol for the
    ~short_otm% OTM call and put separately, sell whichever side is relatively richer
    (skew = ce_iv - pe_iv), buy the tier's wing on that same side. Wider wing -> more credit
    banked, fewer signals clear the gate (same pattern found for the condor/broken-wing).
    De-duplicated against Broken-Wing Condor: a symbol already firing a Broken-Wing signal
    today is excluded here -- skew is a secondary/complementary signal source, not a duplicate.
    Entry requires DTE >= dte_min AND entry-time credit/max_risk > min_ror. Each tier's
    `fired_today` flag tells you which tier(s) are live."""
    contracts, iv = _sell_sources()
    if contracts is None:
        return {"as_of": None, "tiers": []}
    g2 = {s: g for g, syms in _read_json(LM_LOCK_V2 / "universe_groups.json").items() for s in syms}
    lots = _lot_sizes()
    last = contracts["date"].max()
    universe = _liquid_universe_today() or set(g2)
    day = contracts[(contracts["date"] == last) & contracts["opt_type"].isin(["CE", "PE"])
                    & contracts["symbol"].isin(universe) & contracts["strike"].notna()
                    & (contracts["underlying_price"] > 0)].copy()
    ivlast = iv[iv["date"] == last].set_index("symbol")["iv_ratio"].to_dict()

    # live de-dup: symbols with a Broken-Wing signal firing today (any tier), same universe/day.
    bw_a_mcap = set(_read_json(LM_LOCK_V2 / "universe_groups.json").get("A_mcap30", []))
    exclude = set(EXCLUDE_SYMBOLS)
    for tier in BW_TIERS:
        bw_out = _bw_candidates(day, 0.02, tier["wing_ce"], tier["wing_pe"], BW_DTE_MIN,
                                BW_MCAP_MIN_ROR, BW_OTHER_MIN_ROR, bw_a_mcap, g2, lots, ivlast)
        exclude |= {c["symbol"] for c in bw_out if c["in_window"]}

    tiers_out = []
    for tier in SKEW_TIERS:
        out = _skew_candidates(day, short_otm, tier["wing"], dte_min, min_ror, g2, lots, ivlast, exclude)
        out.sort(key=lambda r: (r["in_window"], r["ror_pct"]), reverse=True)
        in_window = [c for c in out if c["in_window"]]
        top_picks = sorted(in_window, key=lambda c: c["ror_pct"], reverse=True)[:3]
        tiers_out.append({
            "id": tier["id"], "label": tier["label"], "wing": tier["wing"], "fired_today": bool(in_window),
            "backtest": {**tier["backtest"], "note": SKEW_NOTE,
                        "window": "2024-08..2026-08 (2y, richer-side 2% OTM, DTE>=15 w/ roll-forward, "
                                  "entry ror>150%, de-duped vs Broken-Wing, pre-cost)"},
            "candidates": out, "top_picks": top_picks,
        })
    return {
        "as_of": last.date().isoformat(),
        "params": {"short_otm": short_otm, "dte_min": dte_min, "min_ror": min_ror},
        "excluded_by_broken_wing": sorted(exclude),
        "tiers": tiers_out,
    }


# Non-mega-cap symbols with a consistently weak (n>=3) historical track record across the
# combined Broken-Wing + Skew signal set, excluded outright: ANGELONE median Rs.608/lot over 8
# signals -- 4.6x below the next-weakest name (TMPV, Rs.2,820) -- not a fluke of a small sample.
EXCLUDE_SYMBOLS = {"ANGELONE"}

BW_DTE_MIN = 9   # same safety floor as the symmetric condor
BW_MCAP_MIN_ROR, BW_OTHER_MIN_ROR = 120.0, 150.0   # stratified entry gate: mega-caps are
    # structurally lower-IV (steadier, less wing-breach risk -- itself a quality trait for a
    # defined-risk seller) so they rarely clear a uniform 150% bar; relaxing it specifically for
    # them lifted mega-cap share from ~13% to 13-28%/tier without a win-rate cost in backtest.
_BW_TOP50_CACHE: dict = {"mtime": None, "by_date": None}


def _liquid_universe_today() -> set:
    """Today's tradable universe for every LIVE sell-strategy endpoint: the fixed Nifty50 +
    today's top-15 most liquid non-Nifty50 names by futures OI-in-lots. Delegates to
    koscine/liquid_universe.py. Note this is WIDER than the backtest universe
    (analysis/gen_*_signal_history.py uses Nifty50 only, no daily tail -- see that module's
    docstring) by design: the daily tail gives live picks some rotation without the backtest
    tracking point-in-time index membership it can't verify. Returns an empty set on failure;
    callers fall back to the static A/B groups."""
    try:
        from koscine import liquid_universe
        return liquid_universe.live_universe_today()
    except Exception as e:   # noqa: BLE001 -- a universe hiccup must not 500 the whole tab
        print(f"[api] liquid_universe unavailable ({e}); falling back to static groups")
        return set()


def _top50_oi_today() -> set:
    """Today's top-50-by-OI-in-lots symbols (fut_oi / that month's lot size -- NOT raw share OI,
    which is dominated by cheap high-share-count names). Cached on eod_deriv_daily's mtime."""
    from koscine.config import SILVER_DATA_ROOT
    f = SILVER_DATA_ROOT / "eod_deriv_daily.parquet"
    if not f.exists():
        return set()
    mt = os.path.getmtime(f)
    if _BW_TOP50_CACHE["mtime"] != mt:
        oi = pd.read_parquet(f, columns=["date", "symbol", "fut_oi"])
        oi["date"] = pd.to_datetime(oi["date"])
        oi = oi[oi["fut_oi"].notna() & (oi["fut_oi"] > 0)]
        last = oi["date"].max()
        today = oi[oi["date"] == last].copy()
        lots = _lot_sizes()
        today["lot"] = today["symbol"].map(lots)
        today = today.dropna(subset=["lot"])
        today["oi_lots"] = today["fut_oi"] / today["lot"]
        _BW_TOP50_CACHE.update(mtime=mt, by_date=set(today.nlargest(50, "oi_lots")["symbol"]))
    return _BW_TOP50_CACHE["by_date"] or set()
# Three narrow-call/wide-put tiers, all live simultaneously. t1 is the everyday base signal
# (fires most often); t2 and t3 are progressively rarer and higher-quality -- when they ALSO
# fire alongside t1 on a given day, that's a genuine higher-conviction setup, not a duplicate.
BW_TIERS = [
    {"id": "t1_2x6", "label": "Tier 1 (2%/6%)", "wing_ce": 0.02, "wing_pe": 0.06,
     "backtest": {"win_rate": 0.993, "median_pnl_per_lot": 6149, "median_pnl_per_lot_per_day": 1230,
                  "per_year": 140, "pct_trades_over_10k": 24.4}},
    {"id": "t2_3x8", "label": "Tier 2 (3%/8%)", "wing_ce": 0.03, "wing_pe": 0.08,
     "backtest": {"win_rate": 0.980, "median_pnl_per_lot": 7410, "median_pnl_per_lot_per_day": 1482,
                  "per_year": 51, "pct_trades_over_10k": 32.7}},
    {"id": "t3_2x10", "label": "Tier 3 (2%/10%)", "wing_ce": 0.02, "wing_pe": 0.10,
     "backtest": {"win_rate": 1.000, "median_pnl_per_lot": 17614, "median_pnl_per_lot_per_day": 3523,
                  "per_year": 12, "pct_trades_over_10k": 73.9}},
]
BW_NOTE = ("Asymmetric wings beat the symmetric 5%/5% condor by up to ~7.6x median profit/lot at "
           "matched entry-ror gate: max loss at expiry is set by the WIDER wing only (one side "
           "breaches at a time), so a wide put wing banks rich put-skew premium cheaply while a "
           "narrow call wing still collects decent credit. This direction (narrow call/wide put) "
           "beat the mirror image at every asymmetry level tested. More asymmetry = better quality "
           "but fewer signals -- Tier 1 is the everyday base signal; Tier 2/3 firing alongside it "
           "is a genuine higher-conviction setup. Universe = static A/B groups + today's dynamic "
           "top-50-by-OI-in-lots; entry gate is stratified (mega-caps 120%, everyone else 150%) "
           "to lift mega-cap representation without cutting frequency -- combined ~206 signals/yr "
           "across all 3 tiers, ~68/41/17 unique symbols firing per tier (vs ~28-40 before). Does "
           "NOT exclude F&O-ban-listed stocks (no ban-list data source available) -- cross-check "
           "live picks against NSE's published ban list. Gross of costs/STT.")


def _bw_candidates(day: pd.DataFrame, short_otm: float, wing_ce: float, wing_pe: float,
                   dte_min: float, mcap_min_ror: float, other_min_ror: float, a_mcap: set,
                   g2: dict, lots: dict, ivlast: dict) -> list[dict]:
    out = []
    for sym, sg in day.groupby("symbol"):
        if sym in EXCLUDE_SYMBOLS:
            continue
        candidate_exps = sorted(sg["expiry"].unique())
        exp = next((e for e in candidate_exps if (pd.Timestamp(e) - day["date"].iloc[0]).days >= dte_min), None)
        if exp is None:
            continue
        chain = sg[sg["expiry"] == exp]
        u = float(chain["underlying_price"].iloc[0])
        dte = int((exp - day["date"].iloc[0]).days)
        ce = chain[chain["opt_type"] == "CE"]; pe = chain[chain["opt_type"] == "PE"]
        if ce.empty or pe.empty:
            continue
        def nearest(df, target):
            return df.iloc[(df["strike"] - target).abs().argmin()]
        sce = nearest(ce, u * (1 + short_otm)); lce = nearest(ce, u * (1 + short_otm + wing_ce))
        spe = nearest(pe, u * (1 - short_otm)); lpe = nearest(pe, u * (1 - short_otm - wing_pe))
        psce, plce, pspe, plpe = _opt_price(sce), _opt_price(lce), _opt_price(spe), _opt_price(lpe)
        if any(np.isnan(x) for x in (psce, plce, pspe, plpe)):
            continue
        credit = (psce + pspe) - (plce + plpe)
        call_width = float(lce["strike"]) - float(sce["strike"]); put_width = float(spe["strike"]) - float(lpe["strike"])
        width = max(call_width, put_width)
        risk = width - credit
        if credit <= 0 or risk <= 0 or risk < MIN_RISK_FRAC * width:
            continue
        ivr = ivlast.get(sym)
        lot = lots.get(sym)
        ce_credit, pe_credit = psce - plce, pspe - plpe
        richer_side = "CE" if ce_credit >= pe_credit else "PE"

        def oi_lots(row):
            v = row.get("open_int")
            if v is None or pd.isna(v) or lot is None or pd.isna(lot) or lot == 0:
                return None
            return round(float(v) / float(lot))

        min_ror_here = mcap_min_ror if sym in a_mcap else other_min_ror
        out.append({
            "symbol": sym, "group": g2.get(sym, "C_top50oi"), "expiry": exp.date().isoformat(), "dte": dte,
            "underlying": round(u, 1), "iv_ratio": round(float(ivr), 2) if ivr is not None and pd.notna(ivr) else None,
            "short_ce": float(sce["strike"]), "long_ce": float(lce["strike"]),
            "short_pe": float(spe["strike"]), "long_pe": float(lpe["strike"]),
            "call_width_pct": round(call_width / u * 100, 1), "put_width_pct": round(put_width / u * 100, 1),
            "oi_short_ce": oi_lots(sce), "oi_long_ce": oi_lots(lce),
            "oi_short_pe": oi_lots(spe), "oi_long_pe": oi_lots(lpe),
            "sell_premium": round(psce + pspe, 2), "buy_premium": round(plce + plpe, 2),
            "credit": round(credit, 2), "max_risk": round(risk, 2), "max_profit": round(credit, 2),
            "lot_size": int(lot) if lot is not None and pd.notna(lot) else None,
            "max_risk_per_lot": round(risk * lot, 1) if lot is not None and pd.notna(lot) else None,
            "max_profit_per_lot": round(credit * lot, 1) if lot is not None and pd.notna(lot) else None,
            "ror_pct": round(credit / risk * 100, 1),
            "be_low": round(float(spe["strike"]) - credit, 1), "be_high": round(float(sce["strike"]) + credit, 1),
            "richer_side": richer_side, "ce_credit": round(ce_credit, 2), "pe_credit": round(pe_credit, 2),
            "min_ror_applied": min_ror_here,
            "in_window": bool(dte >= dte_min and credit / risk * 100 > min_ror_here),
        })
    return out


@app.get("/prod2/broken_wing_strategy")
def prod2_broken_wing_strategy(short_otm: float = Query(0.02, ge=0.01, le=0.08),
                               dte_min: int = Query(BW_DTE_MIN, ge=0, le=30),
                               mcap_min_ror: float = Query(BW_MCAP_MIN_ROR, ge=0, le=1000),
                               other_min_ror: float = Query(BW_OTHER_MIN_ROR, ge=0, le=1000)) -> dict[str, object]:
    """Live DEFINED-RISK broken-wing iron condor, evaluated at THREE asymmetry tiers simultaneously
    (all narrow-call/wide-put): Tier 1 2%/6% (everyday base signal, fires most often), Tier 2 3%/8%,
    Tier 3 2%/10% (rarest, highest quality). Max loss at expiry = max(call_wing, put_wing) -
    total_credit (only one side breaches at expiry, so risk is set by whichever wing is WIDER, not
    their sum). Indian equity/index options carry a persistent put skew (crash premium): a wide put
    wing buys cheap far-OTM protection while banking most of the rich put credit; a narrow call wing
    still collects decent call credit since calls aren't as richly priced. Backtested on a 7-year,
    both-direction-tested panel: this direction beat the mirror image at every asymmetry level tried.
    Each tier's `fired_today` flag tells you which tier(s) are live -- Tier 2/3 firing alongside
    Tier 1 is a genuine higher-conviction setup, not a duplicate signal.

    Universe is the UNION of the static A/B groups and today's dynamic top-50-by-OI-in-lots
    stocks (not raw share OI, which is dominated by cheap high-share-count names -- OI-in-lots
    correctly surfaces mega-caps). Entry gate is STRATIFIED: mega-caps (A_mcap30) need
    entry_ror > mcap_min_ror (120% default, lower than the uniform 150% -- they're structurally
    lower-IV/steadier, so a uniform bar under-represents them), everyone else needs > other_min_ror
    (150%, unchanged). KNOWN GAP: does not exclude F&O-ban-listed stocks -- no ban-list data
    source exists in this codebase; cross-check live picks against NSE's published ban list."""
    contracts, iv = _sell_sources()
    if contracts is None:
        return {"as_of": None, "tiers": []}
    g2 = {s: g for g, syms in _read_json(LM_LOCK_V2 / "universe_groups.json").items() for s in syms}
    a_mcap = set(_read_json(LM_LOCK_V2 / "universe_groups.json").get("A_mcap30", []))
    universe = _liquid_universe_today() or set(g2)
    lots = _lot_sizes()
    last = contracts["date"].max()
    day = contracts[(contracts["date"] == last) & contracts["opt_type"].isin(["CE", "PE"])
                    & contracts["symbol"].isin(universe) & contracts["strike"].notna()
                    & (contracts["underlying_price"] > 0)].copy()
    ivlast = iv[iv["date"] == last].set_index("symbol")["iv_ratio"].to_dict()

    tiers_out = []
    for tier in BW_TIERS:
        out = _bw_candidates(day, short_otm, tier["wing_ce"], tier["wing_pe"], dte_min,
                             mcap_min_ror, other_min_ror, a_mcap, g2, lots, ivlast)
        out.sort(key=lambda r: (r["in_window"], r["ror_pct"]), reverse=True)
        in_window = [c for c in out if c["in_window"]]
        top_picks = sorted(in_window, key=lambda c: c["ror_pct"], reverse=True)[:3]
        tiers_out.append({
            "id": tier["id"], "label": tier["label"], "wing_ce": tier["wing_ce"], "wing_pe": tier["wing_pe"],
            "fired_today": bool(in_window),
            "backtest": {**tier["backtest"], "note": BW_NOTE,
                        "window": "2024-08..2026-08 (2y, short 2% OTM, DTE>=9 w/ roll-forward, "
                                  "stratified entry gate, pre-cost)"},
            "candidates": out, "top_picks": top_picks,
        })
    return {
        "as_of": last.date().isoformat(),
        "params": {"short_otm": short_otm, "dte_min": dte_min, "mcap_min_ror": mcap_min_ror, "other_min_ror": other_min_ror},
        "universe_size": len(universe),
        "tiers": tiers_out,
    }


@app.get("/prod2/broken_wing_signal_history")
def prod2_broken_wing_signal_history(symbol: str | None = None, tier: str | None = None) -> dict[str, object]:
    """Historical broken-wing-condor signals across all 3 tiers (only dates a signal fired), each
    with entry credit / exit value / PnL / max intra-trade drawdown, tagged by `tier`. Optional
    ?tier= filter (t1_2x6 / t2_3x8 / t3_2x10)."""
    f = LM_LOCK_V2.parent / "prod_sell_strategies" / "broken_wing_signal_history.csv"
    if not f.exists():
        return {"rows": [], "summary": None}
    df = pd.read_csv(f)
    if symbol:
        df = df[df["symbol"].eq(symbol.upper())]
    if tier:
        df = df[df["tier"].eq(tier)]
    df = df.sort_values("entry_date", ascending=False)
    summary = None
    if not df.empty:
        n_completed = int((df["outcome"] != "pending").sum())
        summary = {
            "n": int(len(df)),
            "n_pending": int(len(df)) - n_completed,
            # win_rate is over COMPLETED trades only -- a still-open ("pending") trade has no
            # outcome yet and shouldn't dilute the denominator (see gen_sell_signal_history.py
            # and siblings for why some rows can now be pending: forward window not complete).
            "win_rate": round(float((df["outcome"].eq("win")).sum() / max(1, n_completed)), 3),
            "ev_ror_pct": round(float(df["ror_pct"].mean()), 1),
            "median_ror_pct": round(float(df["ror_pct"].median()), 1),
            "worst_ror_pct": round(float(df["ror_pct"].min()), 1),
            "worst_dd_pct": round(float(df["max_dd_pct"].min()), 1),
            "total_pnl": round(float(df["pnl"].sum()), 1),
        }
    return {"rows": _records(df), "summary": summary}


@app.get("/prod2/broken_wing_symbol_stats")
def prod2_broken_wing_symbol_stats(min_n: int = Query(3, ge=1, le=20)) -> dict[str, object]:
    """Per-symbol historical PnL/lot summary across ALL signals (Broken-Wing + Skew, all tiers
    combined -- the two strategies are already de-duped against each other so this is a clean
    union), for highlighting which symbols have historically fired the biggest realized profits
    -- so a live pick from a strong-track-record symbol stands out. Only symbols with lot_size
    known (2024+) and >= min_n signals are included (small samples aren't a reliable signal).
    Each entry also carries its last 10 signals (date + pnl_per_lot + outcome + tier) for a
    hover/detail view."""
    bw_f = LM_LOCK_V2.parent / "prod_sell_strategies" / "broken_wing_signal_history.csv"
    sk_f = LM_LOCK_V2.parent / "prod_sell_strategies" / "skew_signal_history.csv"
    frames = []
    if bw_f.exists():
        frames.append(pd.read_csv(bw_f)[["symbol", "entry_date", "pnl_per_lot", "ror_pct", "outcome", "tier", "lot_size"]])
    if sk_f.exists():
        frames.append(pd.read_csv(sk_f)[["symbol", "entry_date", "pnl_per_lot", "ror_pct", "outcome", "tier", "lot_size"]])
    if not frames:
        return {"symbols": {}}
    df = pd.concat(frames, ignore_index=True)
    df = df[df["lot_size"].notna() & df["pnl_per_lot"].notna()]
    out: dict[str, object] = {}
    for sym, g in df.groupby("symbol"):
        if len(g) < min_n:
            continue
        g = g.sort_values("entry_date", ascending=False)
        last10 = g.head(10)[["entry_date", "pnl_per_lot", "ror_pct", "outcome", "tier"]].to_dict("records")
        out[sym] = {
            "n": int(len(g)),
            "median_pnl_per_lot": round(float(g["pnl_per_lot"].median()), 0),
            "mean_pnl_per_lot": round(float(g["pnl_per_lot"].mean()), 0),
            "win_rate": round(float((g["outcome"].eq("win")).mean()), 3),
            "last10": last10,
        }
    return {"symbols": out}


@app.get("/prod2/skew_signal_history")
def prod2_skew_signal_history(symbol: str | None = None, tier: str | None = None) -> dict[str, object]:
    """Historical skew-strategy signals across all 3 wing-width tiers (only dates a signal
    fired: IV-rich + entry window, de-duped vs Broken-Wing), each with the side sold (CE/PE),
    entry credit / exit value / PnL / max intra-trade drawdown, tagged by `tier`. Optional
    ?symbol= and ?tier= (t1_skew4 / t2_skew6 / t3_skew8) filters."""
    f = LM_LOCK_V2.parent / "prod_sell_strategies" / "skew_signal_history.csv"
    if not f.exists():
        return {"rows": [], "summary": None}
    df = pd.read_csv(f)
    if symbol:
        df = df[df["symbol"].eq(symbol.upper())]
    if tier:
        df = df[df["tier"].eq(tier)]
    df = df.sort_values("entry_date", ascending=False)
    summary = None
    if not df.empty:
        n_completed = int((df["outcome"] != "pending").sum())
        summary = {
            "n": int(len(df)),
            "n_pending": int(len(df)) - n_completed,
            # win_rate is over COMPLETED trades only -- a still-open ("pending") trade has no
            # outcome yet and shouldn't dilute the denominator (see gen_sell_signal_history.py
            # and siblings for why some rows can now be pending: forward window not complete).
            "win_rate": round(float((df["outcome"].eq("win")).sum() / max(1, n_completed)), 3),
            "ev_ror_pct": round(float(df["ror_pct"].mean()), 1),
            "median_ror_pct": round(float(df["ror_pct"].median()), 1),
            "worst_ror_pct": round(float(df["ror_pct"].min()), 1),
            "worst_dd_pct": round(float(df["max_dd_pct"].min()), 1),
            "total_pnl": round(float(df["pnl"].sum()), 1),
        }
    return {"rows": _records(df), "summary": summary}


@app.get("/prod2/sell_signal_history")
def prod2_sell_signal_history(symbol: str | None = None) -> dict[str, object]:
    """Historical Sell-Strategy signals (only the dates a condor signal fired: IV-rich + entry
    window), each with entry credit / exit value / PnL / max intra-trade drawdown. Optional
    ?symbol= filter. Also returns the aggregate track record for the filtered set."""
    f = LM_LOCK_V2.parent / "prod_sell_strategies" / "signal_history.csv"
    if not f.exists():
        return {"rows": [], "summary": None}
    df = pd.read_csv(f)
    if symbol:
        df = df[df["symbol"].eq(symbol.upper())]
    df = df.sort_values("entry_date", ascending=False)
    summary = None
    if not df.empty:
        n_completed = int((df["outcome"] != "pending").sum())
        summary = {
            "n": int(len(df)),
            "n_pending": int(len(df)) - n_completed,
            # win_rate is over COMPLETED trades only -- a still-open ("pending") trade has no
            # outcome yet and shouldn't dilute the denominator (see gen_sell_signal_history.py
            # and siblings for why some rows can now be pending: forward window not complete).
            "win_rate": round(float((df["outcome"].eq("win")).sum() / max(1, n_completed)), 3),
            "ev_ror_pct": round(float(df["ror_pct"].mean()), 1),
            "median_ror_pct": round(float(df["ror_pct"].median()), 1),
            "worst_ror_pct": round(float(df["ror_pct"].min()), 1),
            "worst_dd_pct": round(float(df["max_dd_pct"].min()), 1),
            "total_pnl": round(float(df["pnl"].sum()), 1),
        }
    return {"rows": _records(df), "summary": summary}


CASH_LOCK = LM_LOCK_V2.parent / "prod_cash_signals"


@app.get("/prod2/cash_signals")
def prod2_cash_signals() -> dict[str, object]:
    """Live Cash Signals picks across the F&O universe: stocks breaking out of a resistance
    zone, breaking down through a support zone, or consolidating tightly inside one. Built
    offline by analysis/gen_cash_signals.py from pipeline/zones.py's support/resistance zone
    engine (gold/zones.parquet); this just serves the latest written snapshot."""
    data = _read_json(CASH_LOCK / "cash_signals.json")
    if not data:
        return {"as_of": None, "universe_size": None, "breakout": [], "breakdown": [], "consolidation": []}
    return data


@app.get("/prod2/cash_signal_history")
def prod2_cash_signal_history(signal_type: str | None = None, symbol: str | None = None) -> dict[str, object]:
    """Historical Cash Signals track record (only the dates a breakout/breakdown/consolidation
    signal actually fired), with the realized forward outcome. Optional ?signal_type=
    (breakout|breakdown|consolidation) and ?symbol= filters. Also returns the aggregate track
    record for the filtered set."""
    f = CASH_LOCK / "cash_signal_history.csv"
    if not f.exists():
        return {"rows": [], "summary": None}
    df = pd.read_csv(f)
    if signal_type:
        df = df[df["signal_type"].eq(signal_type)]
    if symbol:
        df = df[df["symbol"].eq(symbol.upper())]
    df = df.sort_values("signal_date", ascending=False)
    summary = None
    if not df.empty:
        if "fwd_return_10d" in df.columns and df["fwd_return_10d"].notna().any():
            summary = {
                "n": int(len(df)),
                "win_rate": round(float((df["outcome"].eq("win")).mean()), 3),
                "avg_fwd_return_10d": round(float(df["fwd_return_10d"].mean()), 2),
                "median_fwd_return_10d": round(float(df["fwd_return_10d"].median()), 2),
                "held_rate_10d": round(float(df["held_10d"].mean()), 3) if "held_10d" in df.columns and df["held_10d"].notna().any() else None,
            }
        else:
            summary = {
                "n": int(len(df)),
                "win_rate": round(float((df["outcome"].eq("win")).mean()), 3),
                "avg_realized_range_10d": round(float(df["realized_range_10d_pct"].mean()), 2) if "realized_range_10d_pct" in df.columns else None,
            }
    return {"rows": _records(df), "summary": summary}


NIFTY_INTRADAY_LOCK = LM_LOCK_V2.parent / "prod_nifty_intraday"
_NIFTY_5M_CACHE: dict = {"mtime": None, "df": None}


def _nifty_5m() -> pd.DataFrame:
    """Cached native 5-min NIFTY 50 index OHLC, mtime-invalidated like _market_ohlc()."""
    f = PROJECT_ROOT / "data" / "intraday" / "nifty_5m.parquet"
    if not f.exists():
        return pd.DataFrame()
    mt = os.path.getmtime(f)
    if _NIFTY_5M_CACHE["mtime"] != mt:
        df = pd.read_parquet(f)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        _NIFTY_5M_CACHE.update(mtime=mt, df=df)
    return _NIFTY_5M_CACHE["df"]


def _nifty_intraday_signals() -> dict:
    data = _read_json(NIFTY_INTRADAY_LOCK / "signals.json")
    if not data:
        return {"as_of": None, "horizon_bars": None, "entry_quantile": None, "p_cutoff": None,
                 "exit_rule": None, "backtest_summary": None, "trades": []}
    return data


@app.get("/prod2/nifty_intraday_dates")
def prod2_nifty_intraday_dates() -> dict[str, object]:
    """Dates with at least one NIFTY intraday breakout trade -- feeds the Indices tab's
    date picker. Built offline by experiments/nifty_intraday_breakout_v1/gen_indices_signals.py."""
    data = _nifty_intraday_signals()
    dates = sorted({t["date"] for t in data.get("trades", [])}, reverse=True)
    return {"dates": dates}


@app.get("/prod2/nifty_intraday_bars")
def prod2_nifty_intraday_bars(date: str = Query(...)) -> dict[str, object]:
    """5-min NIFTY OHLC for one trading day (YYYY-MM-DD), read straight from
    data/intraday/nifty_5m.parquet -- this is intentionally a separate read from
    nifty_intraday_signals so a future live 5-min poller only has to replace this one
    function, not the signals/markers path."""
    df = _nifty_5m()
    if df.empty:
        return {"date": date, "bars": []}
    day = df[df["timestamp"].dt.date.astype(str) == date].sort_values("timestamp")
    bars = [{"ts": r.timestamp.isoformat(), "open": float(r.open), "high": float(r.high),
             "low": float(r.low), "close": float(r.close)} for r in day.itertuples()]
    return {"date": date, "bars": bars}


@app.get("/prod2/nifty_intraday_bars_range")
def prod2_nifty_intraday_bars_range(end_date: str = Query(...), trading_days: int = Query(default=30, ge=1, le=120)) -> dict[str, object]:
    """5-min NIFTY OHLC for the trailing `trading_days` trading days ending on/before end_date
    (inclusive) -- feeds the Indices tab's 5m/15m charts so a 150-trailing-candle window (and
    left/right-arrow panning) can span backward across multiple sessions instead of being
    clipped to whichever single day is selected. `end_date` is always the LAST date on the
    resulting chart -- nothing after it is included."""
    df = _nifty_5m()
    if df.empty:
        return {"end_date": end_date, "bars": []}
    end = pd.Timestamp(end_date).date()
    scoped = df[df["timestamp"].dt.date <= end]
    if scoped.empty:
        return {"end_date": end_date, "bars": []}
    uniq_dates = sorted(scoped["timestamp"].dt.date.unique())[-trading_days:]
    windowed = scoped[scoped["timestamp"].dt.date >= uniq_dates[0]].sort_values("timestamp")
    bars = [{"ts": r.timestamp.isoformat(), "open": float(r.open), "high": float(r.high),
             "low": float(r.low), "close": float(r.close)} for r in windowed.itertuples()]
    return {"end_date": end_date, "bars": bars}


@app.get("/prod2/nifty_intraday_signals")
def prod2_nifty_intraday_signals(date: str | None = None) -> dict[str, object]:
    """NIFTY intraday breakout trade marker(s) (entry/exit timestamp+predicted/realized move %)
    for one date, or all dates if omitted, plus the overall backtest_summary (hit_rate / n /
    exit_reasons). Built offline by experiments/nifty_intraday_breakout_v1/gen_indices_signals.py,
    which replays the SAME non-overlap pattern-exit state machine production/nifty_live_poller.py
    runs live, over an out-of-sample window (train <= 2026-03-31, tested from 2026-04-01 on)."""
    data = _nifty_intraday_signals()
    trades = data.get("trades", [])
    if date:
        trades = [t for t in trades if t["date"] == date]
    return {**data, "trades": trades}


_NIFTY_DAILY_CACHE: dict = {"mtime": None, "df": None}


@app.get("/prod2/nifty_daily_bars")
def prod2_nifty_daily_bars(days: int = Query(default=760, ge=30, le=4000)) -> dict[str, object]:
    """NIFTY 50's own daily OHLC (silver/indices.parquet, 2015-present) -- feeds the Indices
    tab's 1D/1W timeframe views and the client-side S/R zone computation for them (the same
    srLevels() function stock charts use, at the stock-sized 2% cluster tolerance -- the 5-min/
    15-min views use a much tighter 0.10% tolerance instead, since NIFTY moves far less in
    absolute % terms within a single session). Cached on indices.parquet's mtime."""
    from koscine.config import SILVER_DATA_ROOT
    f = SILVER_DATA_ROOT / "indices.parquet"
    if not f.exists():
        return {"bars": []}
    mt = os.path.getmtime(f)
    if _NIFTY_DAILY_CACHE["mtime"] != mt:
        df = pd.read_parquet(f, columns=["date", "index_name", "open", "high", "low", "close"])
        df = df[df["index_name"] == "Nifty 50"].copy()
        df["date"] = pd.to_datetime(df["date"])
        _NIFTY_DAILY_CACHE.update(mtime=mt, df=df.sort_values("date").reset_index(drop=True))
    df = _NIFTY_DAILY_CACHE["df"].tail(days)
    bars = [{"date": r.date.strftime("%Y-%m-%d"), "open": float(r.open), "high": float(r.high),
             "low": float(r.low), "close": float(r.close)} for r in df.itertuples()]
    return {"bars": bars}


@app.get("/prod2/nifty_regime")
def prod2_nifty_regime() -> dict[str, object]:
    """Live regime read from production/nifty_live_poller.py: 'consolidation' (defined-risk
    premium selling suggested) or 'big_move_expected' (naked-option buy suggested, with a
    low-confidence direction hint), plus a concrete recommended_trade (strikes/premium off the
    live chain) when the poller has one. Empty dict if the poller hasn't run/written yet --
    this is read-only, the API never runs the poller itself."""
    return _read_json(NIFTY_INTRADAY_LOCK / "regime_live.json") or {}


@app.get("/prod2/price_history")
def prod2_price_history(symbol: str, days: int = Query(default=400, ge=20, le=4000)) -> dict[str, object]:
    """Daily price (close/high/low) for a symbol with pick markers + the ATM option premium OHLC per pick."""
    sym = symbol.upper()
    m = _market_ohlc()
    s = m[m["symbol"].eq(sym)].sort_values("date").tail(days)
    book = _v2_book()
    bsym = book[book["symbol"].eq(sym)] if not book.empty else pd.DataFrame()
    picks = set(pd.to_datetime(bsym["date"])) if not bsym.empty else set()
    vol = s["volume"].fillna(0) if "volume" in s.columns else pd.Series(0, index=s.index)
    dq = s["delivery_qty"].fillna(0) if "delivery_qty" in s.columns else pd.Series(0, index=s.index)
    series = [{"date": d.strftime("%Y-%m-%d"), "open": float(o), "close": float(c), "high": float(h), "low": float(lo),
               "volume": float(v), "delivQty": float(dqv), "picked": d in picks}
              for d, o, c, h, lo, v, dqv in zip(s["date"], s["open"], s["close"], s["high"], s["low"], vol, dq)]
    prem_cols = ["date", "strike", "ce_entry", "ce_high", "ce_low", "ce_close", "ce_mult_best",
                 "pe_entry", "pe_high", "pe_low", "pe_close", "pe_mult_best", "move_mag_pct",
                 "up_move", "down_move", "live", "dir_label", "confidence"]
    prem = (bsym[[c for c in prem_cols if c in bsym.columns]].sort_values("date", ascending=False)
            if not bsym.empty else pd.DataFrame())
    return {"symbol": sym, "series": series, "premiums": _records(prem) if not prem.empty else []}


_BANKS = {"HDFCBANK", "ICICIBANK", "SBIN", "AXISBANK", "KOTAKBANK", "INDUSINDBK", "BANKBARODA",
          "PNB", "CANBK", "AUBANK", "FEDERALBNK", "IDFCFIRSTB", "BANKINDIA", "RBLBANK"}


_UNIV_CACHE: dict = {"mtime": None, "df": None}


def _universe_frame() -> pd.DataFrame:
    """Per (date, symbol) for the A/B universe: atm_iv (rank key) + 5-day forward move (signed close & excursions).
    Auto-invalidated when the feature parquet is rewritten (daily refresh) — so the API serves fresh data."""
    from koscine3.data.sources import read_data_source
    try:
        mt = os.path.getmtime(read_data_source().path)
    except OSError:
        mt = None
    if _UNIV_CACHE["df"] is not None and _UNIV_CACHE["mtime"] == mt:
        return _UNIV_CACHE["df"]
    g2 = {s: grp for grp, syms in _read_json(LM_LOCK_V2 / "universe_groups.json").items() for s in syms}
    df = load_market_data(columns=["date", "symbol", "open", "high", "low", "close", "atm_iv"])
    df["symbol"] = df["symbol"].astype(str)
    df = df[df["symbol"].isin(g2)].sort_values(["symbol", "date"]).reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])
    g = df.groupby("symbol", sort=False)
    entry = g["open"].shift(-1)
    win_high = pd.concat([g["high"].shift(-i) for i in range(1, 6)], axis=1).max(axis=1)
    win_low = pd.concat([g["low"].shift(-i) for i in range(1, 6)], axis=1).min(axis=1)
    win_close = g["close"].shift(-5)
    df["up_move"] = (win_high - entry) / entry
    df["down_move"] = (entry - win_low) / entry
    df["signed_close"] = (win_close - entry) / entry
    df["has_fwd"] = win_close.notna()
    df["group"] = df["symbol"].map(g2)
    df["eligible"] = df["close"].ge(100) & df["atm_iv"].notna()
    out = df[["date", "symbol", "group", "eligible", "atm_iv", "signed_close", "has_fwd"]]
    _UNIV_CACHE.update(mtime=mt, df=out)
    return out


@app.get("/prod2/universe_day")
def prod2_universe_day(date: str | None = Query(default=None)) -> dict[str, object]:
    """Full ranked A & B universe for a day (rank by atm_iv) + synthetic NIFTY/BANKNIFTY = net 5-day move
    of the top-30 / of the banks within it. move5 = signed 5-day return (entry t+1 open -> t+5 close)."""
    uf = _universe_frame()
    if uf.empty:
        return {}
    if not date:
        date = uf[uf.eligible]["date"].max().strftime("%Y-%m-%d")
    d = uf[uf["date"].eq(pd.Timestamp(date)) & uf.eligible & uf.group.notna()].copy()
    book = _v2_book()
    picked = set()
    if not book.empty:
        bd = book[book["date"].eq(pd.Timestamp(date))]
        picked = {(r.group, r.symbol) for r in bd[["group", "symbol"]].itertuples(index=False)}

    def rows(grp: str) -> list[dict]:
        gg = d[d.group.eq(grp)].sort_values("atm_iv", ascending=False).reset_index(drop=True)
        return [{"rank": i + 1, "symbol": r.symbol, "atm_iv": round(float(r.atm_iv), 3),
                 "move5": None if not r.has_fwd else round(float(r.signed_close) * 100, 2),
                 "live": not bool(r.has_fwd), "picked": (grp, r.symbol) in picked}
                for i, r in gg.iterrows()]

    a = d[d.group.eq("A_mcap30")]

    def agg(sub: pd.DataFrame) -> dict:
        s = sub[sub.has_fwd]
        return {"move5": None if s.empty else round(float(s.signed_close.mean()) * 100, 2),
                "live": bool(s.empty), "n": int(len(sub))}

    return {"date": date, "A": rows("A_mcap30"), "B": rows("B_turn35"),
            "nifty": agg(a), "banknifty": agg(a[a.symbol.isin(_BANKS)])}


_LM_JOBS: dict[str, dict] = {}


def _stream_subprocess_run(kind: str, cmd: list[str], label: str, cwd: Path, env: dict, limit: int) -> None:
    """Run a subprocess with stdout+stderr merged and streamed line-by-line into
    _LM_JOBS[kind]["tail"] as it's produced, so the frontend's poll shows live progress
    instead of a blank cell until the whole process exits."""
    import time
    started = time.time()
    _LM_JOBS[kind] = {"status": "running", "started": started, "module": label, "tail": ""}
    try:
        proc = subprocess.Popen(cmd, cwd=str(cwd), env=env, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True, bufsize=1)
        lines: list[str] = []
        assert proc.stdout is not None
        for line in proc.stdout:
            lines.append(line.rstrip("\n"))
            tail = "\n".join(lines)[-limit:]
            _LM_JOBS[kind] = {"status": "running", "started": started, "module": label, "tail": tail}
        proc.wait()
        tail = "\n".join(lines)[-limit:]
        _LM_JOBS[kind] = {"status": "done" if proc.returncode == 0 else "failed", "module": label,
                          "started": started, "ended": time.time(), "tail": tail}
    except Exception as exc:  # noqa: BLE001
        _LM_JOBS[kind] = {"status": "failed", "module": label, "started": started,
                          "ended": time.time(), "error": str(exc)}


def _lm_module_run(kind: str, module: str) -> None:
    env = {**os.environ, "PYTHONPATH": f"{SRC_ROOT}{os.pathsep}{os.environ.get('PYTHONPATH', '')}"}
    _stream_subprocess_run(kind, [sys.executable, "-u", "-m", module], module, PROJECT_ROOT, env, 4000)


@app.get("/prod2/status")
def prod2_status() -> dict[str, object]:
    """Build status of v2 artifacts + any running run/retrain jobs."""
    def info(p: Path) -> dict[str, object]:
        rows = None
        if p.exists() and p.suffix == ".csv":
            rows = max(0, sum(1 for _ in p.open(encoding="utf-8")) - 1)
        return {"exists": p.exists(),
                "modified": __import__("os").path.getmtime(p) if p.exists() else None,
                "rows": rows}
    man = _read_json(LM_LOCK_V2 / "manifest.json")
    return {
        "version": man.get("version"),
        "selector": man.get("selector"),
        "book": info(LM_LOCK_V2 / "book_2024_26.csv"),
        "direction_overlay": info(LM_LOCK_V2 / "direction_overlay.csv"),
        "premiums": info(LM_LOCK_V2 / "book_premiums.csv"),
        "jobs": _LM_JOBS,
    }


@app.post("/prod2/run")
def prod2_run(background_tasks: BackgroundTasks) -> dict[str, object]:
    """Rebuild the v2 mover book (re-rank by atm_iv, refresh latest picks)."""
    if _LM_JOBS.get("book", {}).get("status") == "running":
        return {"status": "already_running", "job": "book"}
    background_tasks.add_task(_lm_module_run, "book", "koscine3.largemove.mover_v2")
    return {"status": "started", "job": "book", "detail": "rebuild v2 mover book"}


@app.post("/prod2/run_all")
def prod2_run_all(background_tasks: BackgroundTasks) -> dict[str, object]:
    """Rebuild BOTH books in one go: 5-day mover (mover_v2) + 1-day (next_day)."""
    started, skipped = [], []
    for kind, module in (("book", "koscine3.largemove.mover_v2"),
                         ("nextday", "koscine3.largemove.next_day")):
        if _LM_JOBS.get(kind, {}).get("status") == "running":
            skipped.append(kind)
        else:
            background_tasks.add_task(_lm_module_run, kind, module)
            started.append(kind)
    return {"status": "started" if started else "already_running",
            "started": started, "skipped": skipped, "detail": "rebuild 5-day mover + 1-day books"}


def _pipeline_run(start: str, end: str) -> None:
    """Full daily pipeline (fetch -> FII -> silver -> features -> books) for a date range, in one subprocess."""
    label = f"daily_pipeline {start}..{end}"
    env = {**os.environ, "PYTHONPATH": f"{SRC_ROOT}{os.pathsep}{os.environ.get('PYTHONPATH', '')}"}
    _stream_subprocess_run("refresh", [sys.executable, "-u", str(PROJECT_ROOT / "analysis" / "run_daily_pipeline.py"), start, end],
                           label, PROJECT_ROOT, env, 6000)


@app.post("/prod2/refresh")
def prod2_refresh(start: str, end: str, background_tasks: BackgroundTasks) -> dict[str, object]:
    """Full daily pipeline for [start, end]: fetch bhavcopy + FII, build silver, append features, rebuild both books."""
    try:
        s, e = pd.to_datetime(start).date(), pd.to_datetime(end).date()
    except Exception:  # noqa: BLE001
        return {"status": "error", "detail": "invalid dates (use YYYY-MM-DD)"}
    if e < s:
        return {"status": "error", "detail": "end date is before start date"}
    if _LM_JOBS.get("refresh", {}).get("status") == "running":
        return {"status": "already_running", "job": "refresh"}
    background_tasks.add_task(_pipeline_run, str(s), str(e))
    return {"status": "started", "job": "refresh", "detail": f"full pipeline {s}..{e}"}


@app.post("/prod2/retrain")
def prod2_retrain(background_tasks: BackgroundTasks) -> dict[str, object]:
    """Retrain the stage-2 direction overlay (Koscine-2.0-style)."""
    if _LM_JOBS.get("direction", {}).get("status") == "running":
        return {"status": "already_running", "job": "direction"}
    background_tasks.add_task(_lm_module_run, "direction", "koscine3.largemove.direction_stage2")
    return {"status": "started", "job": "direction", "detail": "retrain stage-2 direction overlay"}


# ----------------------------------------------------------------------------
# 1-DAY (t+1) movement model — patched alongside the 5-day v2 book
# ----------------------------------------------------------------------------
def _nextday_book() -> pd.DataFrame:
    f = LM_LOCK_V2 / "next_day_book.csv"
    if not f.exists():
        return pd.DataFrame()
    df = pd.read_csv(f)
    df["date"] = pd.to_datetime(df["date"])
    df["symbol"] = df["symbol"].astype(str)
    return df


@app.get("/prod2/nextday/universe_day")
def prod2_nextday_universe(date: str | None = Query(default=None)) -> dict[str, object]:
    """Full A & B universe ranked by PREDICTED next-day move + synthetic NIFTY/BANKNIFTY net next-day move."""
    b = _nextday_book()
    if b.empty:
        return {}
    if not date:
        date = b["date"].max().strftime("%Y-%m-%d")
    d = b[b["date"].eq(pd.Timestamp(date))].copy()

    def rows(grp: str) -> list[dict]:
        gg = d[d.group.eq(grp)].sort_values("pred_move_pct", ascending=False).reset_index(drop=True)
        return [{"rank": i + 1, "symbol": r.symbol, "atm_iv": float(r.atm_iv),
                 "pred_move": float(r.pred_move_pct),
                 "actual": None if bool(r.live) or pd.isna(r.next_signed_pct) else float(r.next_signed_pct),
                 "actual_mag": None if bool(r.live) or pd.isna(r.next_move_pct) else float(r.next_move_pct),
                 "live": bool(r.live), "picked": i < 3}
                for i, r in gg.iterrows()]

    a = d[d.group.eq("A_mcap30")]

    def agg(sub: pd.DataFrame) -> dict:
        s = sub[(~sub.live) & sub.next_signed_pct.notna()]
        return {"pred": round(float(sub.pred_move_pct.mean()), 2) if len(sub) else None,
                "move": None if s.empty else round(float(s.next_signed_pct.mean()), 2),
                "live": bool(s.empty), "n": int(len(sub))}

    return {"date": date, "A": rows("A_mcap30"), "B": rows("B_turn35"),
            "nifty": agg(a), "banknifty": agg(a[a.symbol.isin(_BANKS)])}


@app.get("/prod2/nextday/stock_history")
def prod2_nextday_stock(symbol: str) -> list[dict[str, object]]:
    b = _nextday_book()
    if b.empty:
        return []
    b = b.copy()
    b["rank"] = b.groupby(["date", "group"])["pred_move_pct"].rank(ascending=False, method="first")
    b["picked"] = b["rank"] <= 3  # top-3/group/day = a Daily-Movers pick
    return _records(b[b.symbol.eq(symbol.upper())].sort_values("date", ascending=False))


@app.get("/prod2/nextday/status")
def prod2_nextday_status() -> dict[str, object]:
    f = LM_LOCK_V2 / "next_day_book.csv"
    return {"exists": f.exists(),
            "modified": __import__("os").path.getmtime(f) if f.exists() else None,
            "rows": (max(0, sum(1 for _ in f.open(encoding="utf-8")) - 1)) if f.exists() else None,
            "jobs": _LM_JOBS}


@app.post("/prod2/nextday/run")
def prod2_nextday_run(background_tasks: BackgroundTasks) -> dict[str, object]:
    if _LM_JOBS.get("nextday", {}).get("status") == "running":
        return {"status": "already_running", "job": "nextday"}
    background_tasks.add_task(_lm_module_run, "nextday", "koscine3.largemove.next_day")
    return {"status": "started", "job": "nextday", "detail": "rebuild 1-day prediction book"}


# ----------------------------------------------------------------------------
# PRODUCTION v3 — mover-precision signals (one ranked list, top-3/day, all-IV, liquidity-gated, cost-tagged)
# ----------------------------------------------------------------------------
LM_LOCK_V3 = LM_LOCK.parent / "prod_largemove_v3"
LM_LOCK_DIR = LM_LOCK.parent / "prod_direction_v1"   # group-B PUT/CALL lean (5d)


def _v3_book(horizon: str = "5d") -> pd.DataFrame:
    h = horizon if horizon in ("5d", "1d") else "5d"
    f = LM_LOCK_V3 / f"mover_v3_book_{h}.csv"
    if not f.exists():
        return pd.DataFrame()
    df = pd.read_csv(f)
    df["date"] = pd.to_datetime(df["date"]); df["symbol"] = df["symbol"].astype(str)
    return df


def _v3_calibrated_peak_forecasts(book: pd.DataFrame) -> pd.DataFrame:
    """Attach a leakage-safe percentage forecast when legacy v3 books lack
    the raw regressor output.

    The old locked books persisted the rank ensemble, not its regression
    estimate.  For those rows we calibrate that score to realised peak moves
    using *only earlier completed v3 observations*, by group and conviction
    bucket. New books use their persisted raw prediction directly.
    """
    out = book.copy().sort_values("date")
    if "pred_move_pct" in out.columns:
        out["forecast_source"] = "v3 raw regressor"
        return out
    out["_bucket"] = pd.cut(out["conv_pctile"], bins=[0, .60, .75, .88, .96, 1.000001], labels=False,
                            include_lowest=True)
    out["pred_move_pct"] = np.nan
    out["forecast_source"] = "walk-forward calibrated v3 score"
    prior: pd.DataFrame | None = None
    for d in sorted(out.date.unique()):
        loc = out.date.eq(d)
        completed = prior.dropna(subset=["move_mag"]) if prior is not None else pd.DataFrame(columns=out.columns)
        for idx, row in out.loc[loc].iterrows():
            same = completed[(completed.group.eq(row.group)) & (completed._bucket.eq(row._bucket))]
            group = completed[completed.group.eq(row.group)]
            pool = same if len(same) >= 20 else group if len(group) >= 30 else completed
            if not pool.empty:
                out.at[idx, "pred_move_pct"] = float(pool.move_mag.mean() * 100)
        prior = out.loc[loc].copy() if prior is None else pd.concat([prior, out.loc[loc]], ignore_index=True)
    return out.drop(columns="_bucket")


def _dir_book() -> pd.DataFrame:
    f = LM_LOCK_DIR / "direction_v1_book_B.csv"
    if not f.exists():
        return pd.DataFrame()
    df = pd.read_csv(f)
    df["date"] = pd.to_datetime(df["date"]); df["symbol"] = df["symbol"].astype(str)
    return df


LM_LOCK_MOVE = LM_LOCK.parent / "prod_expected_move_v1"   # next-day expected MOVE size + Nifty/FII context


def _move_book() -> pd.DataFrame:
    f = LM_LOCK_MOVE / "expected_move_book.csv"
    if not f.exists():
        return pd.DataFrame()
    df = pd.read_csv(f)
    df["date"] = pd.to_datetime(df["date"]); df["symbol"] = df["symbol"].astype(str)
    return df


def _fwd_move_signs() -> pd.DataFrame:
    """Sign of the realized peak excursion (+1 up-dominant / -1 down-dominant) so the unsigned book move_mag can
    be shown directionally. Mirrors mover_v3's fwd_move = max(up-excursion, down-excursion). Auto-refreshes with
    the OHLC cache (mtime-keyed)."""
    import numpy as np
    _market_ohlc()   # refreshes _OHLC_CACHE and resets 'signs' to None on new data
    if _OHLC_CACHE.get("signs") is not None:
        return _OHLC_CACHE["signs"]
    g2 = {s for syms in _read_json(LM_LOCK_V2 / "universe_groups.json").values() for s in syms}
    df = _OHLC_CACHE["df"]
    df = df[df["symbol"].isin(g2)].sort_values(["symbol", "date"])   # only the 65 universe — fast
    g = df.groupby("symbol", sort=False)
    out = df[["date", "symbol"]].copy()
    for h in (1, 5):
        up = pd.concat([g["high"].shift(-i) for i in range(1, h + 1)], axis=1).max(axis=1) / df["close"] - 1
        dn = df["close"] / pd.concat([g["low"].shift(-i) for i in range(1, h + 1)], axis=1).min(axis=1) - 1
        sign = np.where(up.values >= dn.values, 1.0, -1.0)
        sign[(up.isna() | dn.isna()).values] = np.nan
        out[f"sign_{h}"] = sign
    _OHLC_CACHE["signs"] = out
    return out


@app.get("/prod3/dates")
def prod3_dates(horizon: str = Query(default="5d")) -> list[str]:
    df = _v3_book(horizon)
    return [] if df.empty else [d.strftime("%Y-%m-%d") for d in sorted(df.date.unique(), reverse=True)]


@app.get("/prod3/signals")
def prod3_signals(date: str | None = Query(default=None), horizon: str = Query(default="5d")) -> dict[str, object]:
    df = _v3_book(horizon)
    if df.empty:
        return {"date": None, "horizon": horizon, "signals": []}
    d = pd.to_datetime(date) if date else df.date.max()
    day = df[df.date.eq(d)].sort_values("rank")
    if horizon == "5d":   # attach the group-B PUT/CALL lean (direction_v1); A rows stay null (agnostic)
        db = _dir_book()
        if not db.empty:
            lean = db[db.date.eq(d)][["symbol", "p_up", "lean", "dir_conf", "dir_pctile"]]
            day = day.merge(lean, on="symbol", how="left").sort_values(["group", "rank"])
    mv = _move_book()   # next-day expected move size (magnitude) — context for sizing, both horizons
    if not mv.empty:
        day = day.merge(mv[mv.date.eq(d)][["symbol", "exp_move_pct"]], on="symbol", how="left")
    sg = _fwd_move_signs()   # direction of the realized move (sign the unsigned move_mag)
    if not sg.empty:
        col = "sign_1" if horizon == "1d" else "sign_5"
        day = day.merge(sg[["date", "symbol", col]].rename(columns={col: "move_sign"}), on=["date", "symbol"], how="left")
    return {"date": d.strftime("%Y-%m-%d"), "horizon": horizon,
            "live": bool(day["live"].any()) if "live" in day.columns else False, "signals": _records(day)}


@app.get("/signal-desk")
def signal_desk(date: str | None = Query(default=None), horizon: str = Query(default="5d")) -> dict[str, object]:
    """One target-labelled workstation for the production books.

    v3 controls the displayed shortlist; v2 is shown only as the locked
    baseline/pick overlap.  The two one-day forecasts stay distinct because
    their outcomes (intraday peak vs close-to-close) are different.
    """
    signals = _v3_calibrated_peak_forecasts(_v3_book(horizon))
    if signals.empty:
        return {"date": None, "horizon": horizon, "signals": [], "scorecard": {}}
    d = pd.to_datetime(date) if date else signals.date.max()
    day = signals[signals.date.eq(d)].copy().sort_values(["group", "rank"])

    v2 = _v2_book()
    if not v2.empty:
        baseline = v2[v2.date.eq(d)][["group", "symbol", "pick_rank"]].copy()
        baseline["v2_pick"] = True
        day = day.merge(baseline, on=["group", "symbol"], how="left")
    if "v2_pick" not in day.columns:
        day["v2_pick"] = False
    else:
        day["v2_pick"] = day["v2_pick"].notna()

    signs = _fwd_move_signs()
    sign_col = "sign_1" if horizon == "1d" else "sign_5"
    if not signs.empty:
        day = day.merge(signs[["date", "symbol", sign_col]].rename(columns={sign_col: "move_sign"}),
                        on=["date", "symbol"], how="left")
    day["actual_peak_pct"] = day["move_mag"] * 100
    day["actual_peak_signed_pct"] = day["actual_peak_pct"] * day.get("move_sign", 1)
    # The two next-day models have intentionally separate targets, so they
    # appear only in the 1-day view and never leak into the 5-day table.
    if horizon == "1d":
        mv = _move_book()
        if not mv.empty:
            day = day.merge(mv[mv.date.eq(d)][["symbol", "exp_move_pct", "fwd1_abs_pct"]].rename(
                columns={"exp_move_pct": "next_day_close_expected_pct", "fwd1_abs_pct": "next_day_close_actual_pct"}),
                on="symbol", how="left")
        nd_file = LM_LOCK_V2 / "next_day_book.csv"
        if nd_file.exists():
            nd = pd.read_csv(nd_file)
            nd["date"] = pd.to_datetime(nd["date"]); nd["symbol"] = nd["symbol"].astype(str)
            day = day.merge(nd[nd.date.eq(d)][["symbol", "pred_move_pct", "next_move_pct", "next_signed_pct"]].rename(
                columns={"pred_move_pct": "next_day_peak_expected_pct", "next_move_pct": "next_day_peak_actual_pct",
                         "next_signed_pct": "next_day_close_signed_actual_pct"}),
                on="symbol", how="left")
            day["next_day_peak_signed_actual_pct"] = day["next_day_peak_actual_pct"] * day.get("move_sign", 1)
    if horizon == "5d":
        direction = _dir_book()
        if not direction.empty:
            day = day.merge(direction[direction.date.eq(d)][["symbol", "lean", "dir_conf"]], on="symbol", how="left")

    score_path = PROJECT_ROOT / "reports" / "production_signal_scorecard.json"
    scorecard = _read_json(score_path) if score_path.exists() else {}
    return {"date": d.strftime("%Y-%m-%d"), "horizon": horizon,
            "live": bool(day["live"].any()) if "live" in day.columns else False,
            "target_notes": {
                "primary": f"v3 {horizon} regression forecast and realised peak excursion; selection rank remains a separate ensemble.",
                "pred_move_pct": f"v3 {horizon} predicted peak excursion (%), using the raw regressor when present or a leakage-safe calibration of the stored v3 score for legacy rows.",
                "actual_peak_pct": f"realised {horizon} peak excursion (%); pending while the horizon is live.",
                "next_day_peak_expected_pct": "next-session high/low excursion forecast (intraday sizing).",
                "next_day_close_expected_pct": "absolute next-day close-to-close forecast (overnight sizing).",
                "lean": "5-day B-group informational direction tilt only; null means stay direction-agnostic.",
            }, "signals": _records(day), "scorecard": scorecard.get("books", {}),
            "decision": scorecard.get("decision", {})}


@app.get("/prod3/stock_history")
def prod3_stock_history(symbol: str, horizon: str = Query(default="5d")) -> list[dict[str, object]]:
    df = _v3_book(horizon)
    return [] if df.empty else _records(df[df.symbol.eq(symbol.upper())].sort_values("date", ascending=False))


@app.get("/prod3/signal_history")
def prod3_signal_history(symbol: str | None = None, horizon: str = Query(default="5d"),
                         hit_threshold: float = Query(default=6.0)) -> dict[str, object]:
    """Historical Buy-Signal (v3 mover) picks: predicted vs realized peak move, direction-agnostic
    (this book forecasts move SIZE, not direction -- 'hit' = realized |move| >= hit_threshold%).
    Optional ?symbol= filter. Only completed (non-live) rows are included."""
    df = _v3_calibrated_peak_forecasts(_v3_book(horizon))
    if df.empty:
        return {"rows": [], "summary": None}
    if symbol:
        df = df[df["symbol"].eq(symbol.upper())]
    df = df[~df["live"].astype(bool)].copy()
    if df.empty:
        return {"rows": [], "summary": None}
    df["actual_move_pct"] = (df["move_mag"] * 100).round(2)
    df["pred_move_pct"] = df["pred_move_pct"].round(2)
    df["hit"] = df["actual_move_pct"] >= hit_threshold
    sign_col = "sign_1" if horizon == "1d" else "sign_5"
    signs = _fwd_move_signs()
    if not signs.empty and sign_col in signs.columns:
        df = df.merge(signs[["date", "symbol", sign_col]].rename(columns={sign_col: "move_sign"}),
                      on=["date", "symbol"], how="left")
        df["actual_move_signed_pct"] = (df["actual_move_pct"] * df["move_sign"].fillna(1)).round(2)
    else:
        df["actual_move_signed_pct"] = df["actual_move_pct"]
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    df = df.sort_values("date", ascending=False)
    out = df[["date", "group", "symbol", "rank", "conv_pctile", "atm_iv", "pred_move_pct",
              "actual_move_pct", "actual_move_signed_pct", "hit"]]
    summary = {
        "n": int(len(out)),
        "hit_rate": round(float(out["hit"].mean()), 3),
        "mean_pred_pct": round(float(out["pred_move_pct"].mean()), 2),
        "mean_actual_pct": round(float(out["actual_move_pct"].mean()), 2),
        "median_actual_pct": round(float(out["actual_move_pct"].median()), 2),
        "worst_actual_pct": round(float(out["actual_move_pct"].min()), 2),
    }
    return {"rows": _records(out), "summary": summary, "hit_threshold": hit_threshold}


@app.get("/prod3/status")
def prod3_status() -> dict[str, object]:
    import os
    man = _read_json(LM_LOCK_V3 / "manifest.json")
    dman = _read_json(LM_LOCK_DIR / "manifest.json")
    files = {h: (LM_LOCK_V3 / f"mover_v3_book_{h}.csv") for h in ("5d", "1d")}
    return {"version": man.get("version"), "selector": man.get("selector"), "rules": man.get("rules"),
            "horizons": man.get("horizons"),
            "books": {h: {"exists": f.exists(), "modified": os.path.getmtime(f) if f.exists() else None,
                          "rows": (max(0, sum(1 for _ in f.open(encoding="utf-8")) - 1)) if f.exists() else None}
                      for h, f in files.items()},
            "direction": {"scope": "group B (movers) — 5d PUT/CALL lean", "hit_by_year": dman.get("hit_by_year"),
                          "hit_overall": dman.get("hit_overall"), "auc": dman.get("auc_overall"), "ic": dman.get("ic_overall"),
                          "note": "small market-timing tilt; A stays direction-agnostic"} if dman else None,
            "jobs": _LM_JOBS}


@app.post("/prod3/run")
def prod3_run(background_tasks: BackgroundTasks) -> dict[str, object]:
    if _LM_JOBS.get("mover_v3", {}).get("status") == "running":
        return {"status": "already_running", "job": "mover_v3"}
    background_tasks.add_task(_lm_module_run, "mover_v3", "koscine3.largemove.mover_v3")
    return {"status": "started", "job": "mover_v3", "detail": "rebuild v3 mover-precision books (5d+1d)"}


# ----------------------------------------------------------------------------
# Tomorrow — next-day expected MOVE size per name + Nifty/FII context (the reliable magnitude edge)
# ----------------------------------------------------------------------------
@app.get("/prod_move/tomorrow")
def prod_move_tomorrow(date: str | None = Query(default=None)) -> dict[str, object]:
    mv = _move_book()
    man = _read_json(LM_LOCK_MOVE / "manifest.json")
    if mv.empty:
        return {"date": None, "context": man.get("tomorrow"), "rank_ic": man.get("rank_ic_pred_vs_realized"), "movers": []}
    d = pd.to_datetime(date) if date else mv.date.max()
    day = mv[mv.date.eq(d)].copy().sort_values("exp_move_pct", ascending=False)
    return {"date": d.strftime("%Y-%m-%d"), "context": man.get("tomorrow"),
            "rank_ic": man.get("rank_ic_pred_vs_realized"),
            "live": bool(day["live"].any()) if "live" in day.columns else False,
            "movers": _records(day[["symbol", "group", "exp_move_pct", "iv_implied_pct", "realized20_pct", "atm_iv", "live"]])}


@app.get("/prod_move/dates")
def prod_move_dates() -> list[str]:
    mv = _move_book()
    return [] if mv.empty else [d.strftime("%Y-%m-%d") for d in sorted(mv.date.unique(), reverse=True)]


@app.post("/prod_move/run")
def prod_move_run(background_tasks: BackgroundTasks) -> dict[str, object]:
    if _LM_JOBS.get("exp_move", {}).get("status") == "running":
        return {"status": "already_running", "job": "exp_move"}
    background_tasks.add_task(_lm_module_run, "exp_move", "koscine3.largemove.expected_move_v1")
    return {"status": "started", "job": "exp_move", "detail": "rebuild expected-move book (incremental)"}


def _launch_experiment(run_id: str, smoke: bool) -> None:
    command = [
        sys.executable,
        "-m",
        "koscine3.cli",
        "run-experiment",
        "--run-id",
        run_id,
    ]
    if smoke:
        command.append("--smoke")
        command.extend(["--n-estimators", "20"])
    env = {**os.environ, "PYTHONPATH": f"{SRC_ROOT}{os.pathsep}{os.environ.get('PYTHONPATH', '')}"}
    subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=False)


@app.post("/experiments/run")
def run_experiment(
    background_tasks: BackgroundTasks,
    run_id: str = Query(default="koscine3_ui_run"),
    smoke: bool = Query(default=True),
) -> dict[str, object]:
    background_tasks.add_task(_launch_experiment, run_id, smoke)
    return {"status": "started", "run_id": run_id, "smoke": smoke}
