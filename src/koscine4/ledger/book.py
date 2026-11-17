"""The ledger (requirements L-1 to L-5, O-6).

One SQLite file under `records/ledger/` holds every recommendation, whether it was taken, the
fills the owner typed in, the exit and the result, and beside each closed trade what the harness
would have recorded for the same trade.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path

import numpy as np
import pandas as pd

from koscine4.settings import Settings

LEDGER_FILE = Path("ledger") / "ledger.sqlite"
_SCHEMA = """
CREATE TABLE IF NOT EXISTS recommendations (
    id INTEGER PRIMARY KEY AUTOINCREMENT, decision_date TEXT, decision_time TEXT, entry_date TEXT, strategy TEXT, book TEXT,
    instrument TEXT, legs TEXT, lots INTEGER, lot_size REAL, max_loss REAL, score REAL, reasons TEXT, exit_rules TEXT,
    reference_prices TEXT, status TEXT DEFAULT 'recommended', mode TEXT DEFAULT 'paper',
    exit_date TEXT, exit_reason TEXT, pnl REAL, result_pct REAL,
    harness_entry_prices TEXT, harness_exit_prices TEXT, harness_result_pct REAL, harness_exit_date TEXT
);
CREATE TABLE IF NOT EXISTS fills (
    id INTEGER PRIMARY KEY AUTOINCREMENT, rec_id INTEGER, kind TEXT, leg INTEGER, price REAL, quantity REAL, filled_at TEXT, note TEXT
);
CREATE TABLE IF NOT EXISTS strategy_status (
    strategy TEXT PRIMARY KEY, suspended INTEGER DEFAULT 0, reason TEXT, changed_at TEXT
);
"""


class LedgerError(RuntimeError):
    pass


class Ledger:
    def __init__(self, settings: Settings):
        self.path = settings.path("records") / LEDGER_FILE
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as con:
            con.executescript(_SCHEMA)
            con.commit()

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.path)
        con.row_factory = sqlite3.Row
        return con

    def _exec(self, sql: str, args=()) -> int:
        with closing(self._connect()) as con:
            cur = con.execute(sql, args)
            con.commit()
            return cur.lastrowid

    def frame(self, sql: str, args=()) -> pd.DataFrame:
        with closing(self._connect()) as con:
            return pd.read_sql_query(sql, con, params=args)

    # ---- recommendations and fills
    def add_recommendation(self, *, decision_date, decision_time, entry_date, strategy, book, instrument, legs, lots, lot_size,
                           max_loss, score, reasons, exit_rules, reference_prices, mode="paper") -> int:
        return self._exec(
            "INSERT INTO recommendations (decision_date, decision_time, entry_date, strategy, book, instrument, legs, lots, lot_size, max_loss,"
            " score, reasons, exit_rules, reference_prices, mode) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (str(pd.Timestamp(decision_date).date()), decision_time, str(pd.Timestamp(entry_date).date()), strategy, book, instrument,
             json.dumps(legs), int(lots), float(lot_size), float(max_loss), float(score), json.dumps(reasons, default=str),
             json.dumps(exit_rules), json.dumps(reference_prices), mode))

    def get(self, rec_id: int) -> dict:
        df = self.frame("SELECT * FROM recommendations WHERE id = ?", (rec_id,))
        if df.empty:
            raise LedgerError(f"No recommendation {rec_id}.")
        return df.iloc[0].to_dict()

    def record_fill(self, rec_id: int, kind: str, leg: int, price: float, quantity: float, filled_at, note: str = "") -> int:
        """A fill typed in by the owner (O-6). kind is "entry" or "exit"."""
        if kind not in ("entry", "exit"):
            raise LedgerError("A fill is an 'entry' or an 'exit'.")
        rec = self.get(rec_id)
        n_legs = len(json.loads(rec["legs"]))
        if not 0 <= int(leg) < n_legs:
            raise LedgerError(f"Recommendation {rec_id} has legs 0 to {n_legs - 1}.")
        fid = self._exec("INSERT INTO fills (rec_id, kind, leg, price, quantity, filled_at, note) VALUES (?,?,?,?,?,?,?)",
                         (rec_id, kind, int(leg), float(price), float(quantity), str(filled_at), note))
        if kind == "entry" and rec["status"] == "recommended":
            self._exec("UPDATE recommendations SET status = 'open' WHERE id = ?", (rec_id,))
        return fid

    def mark_not_taken(self, rec_id: int, note: str = "") -> None:
        self._exec("UPDATE recommendations SET status = 'not_taken', exit_reason = ? WHERE id = ?", (note, rec_id))

    def fills(self, rec_id: int) -> pd.DataFrame:
        return self.frame("SELECT * FROM fills WHERE rec_id = ? ORDER BY id", (rec_id,))

    def close_trade(self, rec_id: int, exit_date, exit_reason: str) -> dict:
        """Compute the result from the recorded fills and close the trade."""
        rec = self.get(rec_id)
        legs = json.loads(rec["legs"])
        f = self.fills(rec_id)
        pnl = 0.0
        for i, leg in enumerate(legs):
            side = leg[3]
            entry, exit_ = f[(f["leg"] == i) & (f["kind"] == "entry")], f[(f["leg"] == i) & (f["kind"] == "exit")]
            if entry.empty or exit_.empty:
                raise LedgerError(f"Recommendation {rec_id}: leg {i} needs both an entry and an exit fill before it can be closed.")
            avg = lambda d: float((d["price"] * d["quantity"]).sum() / d["quantity"].sum())
            pnl += side * (avg(exit_) - avg(entry)) * float(entry["quantity"].sum())
        result_pct = 100.0 * pnl / rec["max_loss"] if rec["max_loss"] else float("nan")
        self._exec("UPDATE recommendations SET status = 'closed', exit_date = ?, exit_reason = ?, pnl = ?, result_pct = ? WHERE id = ?",
                   (str(pd.Timestamp(exit_date).date()), exit_reason, pnl, result_pct, rec_id))
        return {"pnl": pnl, "result_pct": result_pct}

    def attach_harness_twin(self, rec_id: int, entry_prices, exit_prices, result_pct: float, exit_date) -> None:
        """What the harness recorded for the same trade (L-2)."""
        self._exec("UPDATE recommendations SET harness_entry_prices = ?, harness_exit_prices = ?, harness_result_pct = ?, harness_exit_date = ? WHERE id = ?",
                   (json.dumps(list(entry_prices)), json.dumps(list(exit_prices)), float(result_pct), str(pd.Timestamp(exit_date).date()), rec_id))

    # ---- state for selection and risk
    def open_trades(self) -> pd.DataFrame:
        return self.frame("SELECT * FROM recommendations WHERE status IN ('open', 'recommended') ORDER BY id")

    def started_in_week(self, date) -> dict:
        """Trades started (taken or still recommended) in the ISO week of `date`, per book."""
        date = pd.Timestamp(date)
        monday = (date - pd.Timedelta(days=date.weekday())).normalize()
        df = self.frame("SELECT book, COUNT(*) AS n FROM recommendations WHERE status IN ('open','closed','recommended') AND entry_date >= ? AND entry_date <= ? GROUP BY book",
                        (str(monday.date()), str((monday + pd.Timedelta(days=6)).date())))
        return {r["book"]: int(r["n"]) for _, r in df.iterrows()}

    def realised_pnl(self, mode: str | None = None) -> float:
        df = self.frame("SELECT pnl, mode FROM recommendations WHERE status = 'closed'")
        if mode is not None:
            df = df[df["mode"] == mode]
        return float(df["pnl"].sum()) if len(df) else 0.0

    def closed(self, strategy: str | None = None) -> pd.DataFrame:
        df = self.frame("SELECT * FROM recommendations WHERE status = 'closed' ORDER BY exit_date, id")
        return df if strategy is None else df[df["strategy"] == strategy]

    # ---- monitoring
    def differences(self) -> pd.DataFrame:
        """Live result against the harness result, trade by trade (L-2)."""
        c = self.closed()
        c = c[c["harness_result_pct"].notna()].copy()
        c["result_difference_pct"] = c["result_pct"] - c["harness_result_pct"]
        return c[["id", "strategy", "instrument", "entry_date", "exit_date", "result_pct", "harness_result_pct", "result_difference_pct"]]

    def is_suspended(self, strategy: str) -> bool:
        df = self.frame("SELECT suspended FROM strategy_status WHERE strategy = ?", (strategy,))
        return bool(len(df) and df["suspended"].iloc[0])

    def set_suspended(self, strategy: str, suspended: bool, reason: str, when) -> None:
        self._exec("INSERT INTO strategy_status (strategy, suspended, reason, changed_at) VALUES (?,?,?,?) "
                   "ON CONFLICT(strategy) DO UPDATE SET suspended = excluded.suspended, reason = excluded.reason, changed_at = excluded.changed_at",
                   (strategy, int(suspended), reason, str(when)))

    def check_suspension(self, strategy: str, backtest_mean: float, backtest_std: float, cfg, when) -> dict:
        """Suspend a strategy whose live results fall below the range its backtest predicted (L-4).

        Over the last `suspend_after_trades` live trades, the mean result is compared with the
        backtest mean less `suspend_z` standard errors. Only the owner lifts a suspension.
        """
        n = int(cfg["suspend_after_trades"])
        live = self.closed(strategy)["result_pct"].dropna().tail(n)
        if len(live) < n:
            return {"strategy": strategy, "live_trades": int(len(live)), "suspended": self.is_suspended(strategy), "checked": False}
        floor = backtest_mean - float(cfg["suspend_z"]) * backtest_std / np.sqrt(n)
        below = float(live.mean()) < floor
        if below and not self.is_suspended(strategy):
            self.set_suspended(strategy, True, f"live mean {live.mean():.1f}% over {n} trades is below the backtest floor {floor:.1f}%", when)
        return {"strategy": strategy, "live_trades": n, "live_mean_pct": float(live.mean()), "floor_pct": float(floor),
                "suspended": self.is_suspended(strategy), "checked": True}

    def monthly_report(self, month: str, backtest: dict[str, dict]) -> dict:
        """Live results for one month against the backtest range per strategy (L-3).

        backtest: strategy -> {"mean_result_pct", "std_result_pct"} from the run behind its gate record.
        """
        c = self.closed()
        c = c[c["exit_date"].astype(str).str.startswith(month)]
        out = {"month": month, "closed_trades": int(len(c)), "strategies": {}}
        for strategy, g in c.groupby("strategy"):
            b = backtest.get(strategy, {})
            n = len(g)
            mean = float(g["result_pct"].mean())
            row = {"trades": int(n), "mean_result_pct": mean, "total_pnl": float(g["pnl"].sum())}
            if b:
                half = 2.0 * float(b["std_result_pct"]) / np.sqrt(n)
                row["backtest_range_pct"] = [float(b["mean_result_pct"]) - half, float(b["mean_result_pct"]) + half]
                row["inside_range"] = bool(row["backtest_range_pct"][0] <= mean <= row["backtest_range_pct"][1])
            twin = g["harness_result_pct"].dropna()
            if len(twin):
                row["mean_difference_from_harness_pct"] = float((g.loc[twin.index, "result_pct"] - twin).mean())
            out["strategies"][strategy] = row
        return out


def feature_drift(live_row: pd.Series, history: pd.DataFrame, low_pct: float, high_pct: float) -> list[dict]:
    """Live feature values far outside the range seen in history (L-5)."""
    flags = []
    for name, value in live_row.items():
        if name not in history.columns or value != value:
            continue
        col = history[name].dropna()
        if len(col) < 100:
            continue
        lo, hi = np.percentile(col, [low_pct, high_pct])
        if value < lo or value > hi:
            flags.append({"feature": name, "value": float(value), "low": float(lo), "high": float(hi)})
    return flags
