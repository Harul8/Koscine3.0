"""Web routes for the trade list and the ledger (requirement O-7), kept apart from the existing routes.

To show them in the existing app, add to api/main.py:

    from koscine4.api.routes import router as koscine4_router
    app.include_router(koscine4_router)

No route here starts training or a backtest (M-8).
"""
from __future__ import annotations

import json

from koscine4.daily.run import DAILY
from koscine4.ledger.book import Ledger, LedgerError
from koscine4.settings import load_settings

try:
    from fastapi import APIRouter, HTTPException
    from pydantic import BaseModel
except ImportError:  # the package works without the web stack; only these routes need it
    APIRouter = None


def trade_list(date: str, time: str = "evening", settings=None) -> dict:
    """The saved daily list, exactly as the file holds it."""
    settings = settings or load_settings()
    path = settings.path("records") / DAILY / f"{date}_{time}.json"
    if not path.exists():
        raise FileNotFoundError(f"No trade list for {date} ({time}).")
    return json.loads(path.read_text(encoding="utf-8"))


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

    @router.get("/trade-list/{date}")
    def get_trade_list(date: str, time: str = "evening"):
        try:
            return trade_list(date, time)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc))

    @router.get("/ledger/open")
    def get_open():
        return Ledger(load_settings()).open_trades().to_dict(orient="records")

    @router.get("/ledger/closed")
    def get_closed():
        return Ledger(load_settings()).closed().to_dict(orient="records")

    @router.post("/ledger/fills")
    def post_fill(fill: Fill):
        try:
            fid = Ledger(load_settings()).record_fill(fill.rec_id, fill.kind, fill.leg, fill.price, fill.quantity, fill.filled_at, fill.note)
        except LedgerError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        return {"fill_id": fid}
else:
    router = None
