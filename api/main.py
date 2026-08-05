from __future__ import annotations

import subprocess
import sys
import os
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import BackgroundTasks, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware


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


@app.get("/prod2/sell_strategies")
def prod2_sell_strategies(short_otm: float = Query(0.02, ge=0.01, le=0.08),
                          wing: float = Query(0.03, ge=0.01, le=0.08),
                          dte_max: int = Query(14, ge=2, le=45),
                          iv_rich: float = Query(1.1, ge=0.5, le=2.0)) -> dict[str, object]:
    """Live DEFINED-RISK iron-condor candidates on the A/B universe: sell ~short_otm% OTM CE+PE,
    buy the +wing% wings (loss always capped). Ranked by IV-richness x return-on-risk. Entry
    window flagged when DTE<=dte_max and IV is rich. Backtest context (2y, pre-cost) is embedded."""
    contracts, iv = _sell_sources()
    if contracts is None:
        return {"as_of": None, "candidates": [], "note": "no option-chain data"}
    g2 = {s: g for g, syms in _read_json(LM_LOCK_V2 / "universe_groups.json").items() for s in syms}
    last = contracts["date"].max()
    day = contracts[(contracts["date"] == last) & contracts["opt_type"].isin(["CE", "PE"])
                    & contracts["symbol"].isin(g2) & contracts["strike"].notna()
                    & (contracts["underlying_price"] > 0)].copy()
    ivlast = iv[iv["date"] == last].set_index("symbol")["iv_ratio"].to_dict()

    def price(row):  # leg mid: prefer close, fall back to settle
        p = row.get("close")
        return float(p) if pd.notna(p) and p > 0 else (float(row["settle"]) if pd.notna(row.get("settle")) else np.nan)

    out = []
    for sym, sg in day.groupby("symbol"):
        exp = sg["expiry"].min()
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
        psce, plce, pspe, plpe = price(sce), price(lce), price(spe), price(lpe)
        if any(np.isnan(x) for x in (psce, plce, pspe, plpe)):
            continue
        credit = (psce + pspe) - (plce + plpe)
        width = max(float(lce["strike"] - sce["strike"]), float(spe["strike"] - lpe["strike"]))
        risk = width - credit
        if credit <= 0 or risk <= 0:
            continue
        ivr = ivlast.get(sym)
        out.append({
            "symbol": sym, "group": g2[sym], "expiry": exp.date().isoformat(), "dte": dte,
            "underlying": round(u, 1), "iv_ratio": round(float(ivr), 2) if ivr is not None and pd.notna(ivr) else None,
            "short_ce": float(sce["strike"]), "long_ce": float(lce["strike"]),
            "short_pe": float(spe["strike"]), "long_pe": float(lpe["strike"]),
            "credit": round(credit, 2), "max_risk": round(risk, 2), "max_profit": round(credit, 2),
            "ror_pct": round(credit / risk * 100, 1),
            "be_low": round(float(spe["strike"]) - credit, 1), "be_high": round(float(sce["strike"]) + credit, 1),
            "in_window": bool(dte <= dte_max and ivr is not None and pd.notna(ivr) and ivr >= iv_rich),
        })
    out.sort(key=lambda r: (r["in_window"], (r["iv_ratio"] or 0), r["ror_pct"]), reverse=True)
    return {
        "as_of": last.date().isoformat(),
        "params": {"short_otm": short_otm, "wing": wing, "dte_max": dte_max, "iv_rich": iv_rich},
        "backtest": {"window": "2024-08..2026-08 (2y, short 2% OTM / wing +3%, DTE 5-12, pre-cost)",
                     "ev_on_risk_all": 0.46, "ev_on_risk_iv_rich": 0.68, "win_rate": 0.93, "worst": "-0.84x (capped)"},
        "candidates": out,
    }


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
        summary = {
            "n": int(len(df)),
            "win_rate": round(float((df["outcome"].eq("win")).mean()), 3),
            "ev_ror_pct": round(float(df["ror_pct"].mean()), 1),
            "median_ror_pct": round(float(df["ror_pct"].median()), 1),
            "worst_ror_pct": round(float(df["ror_pct"].min()), 1),
            "worst_dd_pct": round(float(df["max_dd_pct"].min()), 1),
            "total_pnl": round(float(df["pnl"].sum()), 1),
        }
    return {"rows": _records(df), "summary": summary}


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


def _subprocess_tail(result: subprocess.CompletedProcess[str], limit: int) -> str:
    """Return useful output for both successful and failed background jobs."""
    chunks = []
    if result.stdout:
        chunks.append(result.stdout.rstrip())
    if result.stderr:
        chunks.append("[stderr]\n" + result.stderr.rstrip())
    return "\n".join(chunks)[-limit:]


def _lm_module_run(kind: str, module: str) -> None:
    import time
    _LM_JOBS[kind] = {"status": "running", "started": time.time(), "module": module}
    env = {**os.environ, "PYTHONPATH": f"{SRC_ROOT}{os.pathsep}{os.environ.get('PYTHONPATH', '')}"}
    try:
        r = subprocess.run([sys.executable, "-m", module], cwd=str(PROJECT_ROOT), env=env,
                           capture_output=True, text=True)
        _LM_JOBS[kind] = {"status": "done" if r.returncode == 0 else "failed", "module": module,
                          "started": _LM_JOBS[kind]["started"], "ended": time.time(),
                          "tail": _subprocess_tail(r, 4000)}
    except Exception as exc:  # noqa: BLE001
        _LM_JOBS[kind] = {"status": "failed", "module": module, "error": str(exc)}


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
    import time
    label = f"daily_pipeline {start}..{end}"
    _LM_JOBS["refresh"] = {"status": "running", "started": time.time(), "module": label}
    env = {**os.environ, "PYTHONPATH": f"{SRC_ROOT}{os.pathsep}{os.environ.get('PYTHONPATH', '')}"}
    try:
        r = subprocess.run([sys.executable, "-u", str(PROJECT_ROOT / "analysis" / "run_daily_pipeline.py"), start, end],
                           cwd=str(PROJECT_ROOT), env=env, capture_output=True, text=True)
        _LM_JOBS["refresh"] = {"status": "done" if r.returncode == 0 else "failed", "module": label,
                               "started": _LM_JOBS["refresh"]["started"], "ended": time.time(),
                               "tail": _subprocess_tail(r, 6000)}
    except Exception as exc:  # noqa: BLE001
        _LM_JOBS["refresh"] = {"status": "failed", "module": label, "error": str(exc)}


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
