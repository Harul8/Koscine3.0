"""Web routes for the trade list and the ledger (requirements O-6, O-7), kept apart from the existing routes.

api/main.py includes them:

    from koscine4.api.routes import router as koscine4_router
    app.include_router(koscine4_router)

    GET  /v1/trade-lists                       the saved daily lists, newest first
    GET  /v1/trade-list/{date}?time=evening    one list, as the file holds it
    GET  /v1/ledger/open                       recommended and open trades
    GET  /v1/ledger/closed                     closed trades
    GET  /v1/ledger/{id}                       one recommendation with its fills
    POST /v1/ledger/fills                      a fill the owner typed in (O-6)
    POST /v1/ledger/{id}/not-taken             the owner did not take a recommendation
    POST /v1/ledger/{id}/close                 close a trade from its recorded fills

The routes only read files and write the ledger. None starts training or a backtest (M-8).
"""
from __future__ import annotations

import json
import math

import pandas as pd

from koscine4.daily.run import DAILY
from koscine4.ledger.book import Ledger, LedgerError
from koscine4.settings import load_settings

try:
    from fastapi import APIRouter, HTTPException
    from pydantic import BaseModel
except ImportError:  # the package works without the web stack; only these routes need it
    APIRouter = None


def clean(x):
    """JSON-safe: NaN and infinity become null, and the ledger's JSON text columns are parsed."""
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, float) and not math.isfinite(x):
        return None
    if hasattr(x, "item") and not isinstance(x, (str, bytes)):          # numpy scalars
        return clean(x.item())
    return x


LEDGER_JSON = ("legs", "reasons", "exit_rules", "reference_prices", "harness_entry_prices", "harness_exit_prices")


def ledger_row(row: dict) -> dict:
    out = dict(row)
    for k in LEDGER_JSON:
        if isinstance(out.get(k), str) and out[k]:
            try:
                out[k] = json.loads(out[k])
            except json.JSONDecodeError:
                pass
    return clean(out)


def ledger_rows(df: pd.DataFrame) -> list[dict]:
    return [ledger_row(r) for r in df.to_dict(orient="records")]


def trade_list(date: str, time: str = "evening", settings=None) -> dict:
    """The saved daily list, exactly as the file holds it."""
    settings = settings or load_settings()
    path = settings.path("records") / DAILY / f"{date}_{time}.json"
    if not path.exists():
        raise FileNotFoundError(f"No trade list for {date} ({time}).")
    return clean(json.loads(path.read_text(encoding="utf-8")))


def trade_lists(settings=None) -> list[dict]:
    """Every saved daily list, newest first: date, decision time, mode and how many new trades."""
    settings = settings or load_settings()
    folder = settings.path("records") / DAILY
    out = []
    for path in folder.glob("*.json") if folder.exists() else []:
        try:
            d = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        out.append({"date": d.get("date"), "decision_time": d.get("decision_time"), "mode": d.get("mode"),
                    "new_trades": len(d.get("new_trades", [])), "open_trades": len(d.get("open_trades", []))})
    return sorted(out, key=lambda r: (str(r["date"]), str(r["decision_time"])), reverse=True)


def recommendation(rec_id: int, settings=None) -> dict:
    ledger = Ledger(settings or load_settings())
    rec = ledger_row(ledger.get(rec_id))
    rec["fills"] = clean(ledger.fills(rec_id).to_dict(orient="records"))
    return rec


if APIRouter is not None:
    router = APIRouter(prefix="/v1", tags=["koscine4"])

    class Fill(BaseModel):
        rec_id: int
        kind: str
        leg: int
        price: float
        quantity: float
        filled_at: str
        note: str = ""

    class NotTaken(BaseModel):
        note: str = ""

    class Close(BaseModel):
        exit_date: str
        exit_reason: str

    def _bad(exc: Exception):
        raise HTTPException(status_code=404 if str(exc).startswith("No recommendation") else 400, detail=str(exc))

    @router.get("/trade-lists")
    def get_trade_lists():
        return trade_lists()

    @router.get("/trade-list/{date}")
    def get_trade_list(date: str, time: str = "evening"):
        try:
            return trade_list(date, time)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc))

    @router.get("/ledger/open")
    def get_open():
        return ledger_rows(Ledger(load_settings()).open_trades())

    @router.get("/ledger/closed")
    def get_closed():
        return ledger_rows(Ledger(load_settings()).closed())

    @router.get("/ledger/{rec_id:int}")
    def get_recommendation(rec_id: int):
        try:
            return recommendation(rec_id)
        except LedgerError as exc:
            _bad(exc)

    @router.post("/ledger/fills")
    def post_fill(fill: Fill):
        try:
            fid = Ledger(load_settings()).record_fill(fill.rec_id, fill.kind, fill.leg, fill.price, fill.quantity, fill.filled_at, fill.note)
        except LedgerError as exc:
            _bad(exc)
        return {"fill_id": fid}

    @router.post("/ledger/{rec_id:int}/not-taken")
    def post_not_taken(rec_id: int, body: NotTaken):
        try:
            Ledger(load_settings()).mark_not_taken(rec_id, body.note)
        except LedgerError as exc:
            _bad(exc)
        return {"id": rec_id, "status": "not_taken"}

    @router.post("/ledger/{rec_id:int}/close")
    def post_close(rec_id: int, body: Close):
        try:
            out = Ledger(load_settings()).close_trade(rec_id, body.exit_date, body.exit_reason)
        except LedgerError as exc:
            _bad(exc)
        return clean({"id": rec_id, "status": "closed", **out})
else:
    router = None
