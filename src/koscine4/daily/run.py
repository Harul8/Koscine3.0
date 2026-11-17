"""The daily run (requirements O-1 to O-6, S-11, R-7, R-9).

Steps, in order. A failed step stops the run with a message, and nothing after it happens:

    1. rebuild the store tables from the silver layer and run the data checks
    2. build today's feature rows as of the decision time
    3. score each strategy that has a gate 2 record, with the model stamped with the latest cutoff
    4. select: threshold, weekly cap, one per instrument, the two books, correlated-trade limit
    5. size each trade against the risk settings, and refuse what breaks a limit
    6. write the trade list, with today's exit instruction for every open trade

    python -m koscine4.daily.run --date 2026-09-01 --time evening
    python -m koscine4.daily.run --date 2026-09-01 --time 1430 --paper      # list without live risk settings

It never trains a model and never places an order (M-8, A7).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from koscine4.features.build import features_asof
from koscine4.harness import records
from koscine4.harness.engine import _mark
from koscine4.harness.market import MarketData
from koscine4.ledger.book import Ledger, feature_drift
from koscine4.risk import rules as risk_rules
from koscine4.selection.select import BookState, select
from koscine4.settings import Settings, SettingsError, load_settings
from koscine4.store.availability import Decision
from koscine4.store.build import build_store
from koscine4.store.checks import DataCheckError
from koscine4.strategies.base import PositionState, SpecStrategy
from koscine4.strategies.spec import DECISION_TO_ENTRY, load_spec
from koscine4.strategies.structures import Leg

DAILY = Path("daily")


class DailyRunError(RuntimeError):
    """A step of the daily run failed; the message says which and why."""


def live_strategies(settings: Settings, decision_time: str, ledger: Ledger, paper: bool) -> tuple[list[SpecStrategy], list[dict]]:
    """Strategies allowed to produce trades: a gate 2 pass on record, and not suspended (S-11, L-4)."""
    allowed, refused = [], []
    folder = settings.path("strategies")
    for path in sorted(folder.glob("*.toml")) if folder.exists() else []:
        spec = load_spec(settings, path.stem)
        if spec.decision_time != decision_time:
            continue
        if not records.has_passed(settings, spec.name, 2):
            refused.append({"strategy": spec.name, "reason": "no_gate_2_record"})
        elif spec.uses_direction and not records.has_passed(settings, spec.direction_model or spec.name, 2):
            refused.append({"strategy": spec.name, "reason": "direction_model_has_not_passed_its_gate"})      # S-12
        elif ledger.is_suspended(spec.name):
            refused.append({"strategy": spec.name, "reason": "suspended_until_the_owner_re_enables_it"})
        elif not paper and not records.has_passed(settings, spec.name, 3):
            refused.append({"strategy": spec.name, "reason": "paper_trading_gate_3_not_passed_for_a_live_list"})
        else:
            allowed.append(SpecStrategy(spec))
    return allowed, refused


def _predictions(settings: Settings, strategies: list[SpecStrategy], feature_rows: pd.DataFrame, date) -> dict[str, pd.DataFrame]:
    from koscine4.models.train import predict_latest

    names = set()
    for s in strategies:
        if s.spec.score.get("source") == "model":
            names.add(str(s.spec.score["model"]))
        if s.spec.direction_model:
            names.add(s.spec.direction_model)
    today = feature_rows[feature_rows["date"] == pd.Timestamp(date)]
    return {name: predict_latest(settings, name, today, date) for name in sorted(names)}


def exit_instructions(data: MarketData, ledger: Ledger, settings: Settings, date, decision_time: str) -> list[dict]:
    """Today's instruction for every open trade (O-3): hold, or close and why."""
    out = []
    when = "close" if decision_time == "evening" else "1515"
    cal = data.calendar
    for _, t in ledger.open_trades().iterrows():
        if t["status"] != "open":
            out.append({"id": int(t["id"]), "instrument": t["instrument"], "strategy": t["strategy"], "instruction": "awaiting entry fill"})
            continue
        legs = [Leg(pd.Timestamp(l[0]), float(l[1]), l[2], int(l[3])) for l in json.loads(t["legs"])]
        spec = load_spec(settings, t["strategy"])
        fills = ledger.fills(int(t["id"]))
        entry = [float(fills[(fills["leg"] == i) & (fills["kind"] == "entry")]["price"].mean()) for i in range(len(legs))]
        visible = data.store.last_visible("option_quotes", Decision(date, decision_time))
        marks = [_mark(data, t["instrument"], leg, visible, "close" if when == "close" else "close") for leg in legs]
        held = len(cal.between(t["entry_date"], date))
        row = {"id": int(t["id"]), "instrument": t["instrument"], "strategy": t["strategy"], "sessions_held": held}
        if any(m[0] is None for m in marks):
            row["instruction"] = "no price available today: check by hand"
        else:
            risk = float(t["max_loss"]) / (float(t["lots"]) * float(t["lot_size"]))
            state = PositionState(t["instrument"], legs, entry, [m[0] for m in marks], risk, held, pd.Timestamp(date))
            action = SpecStrategy(spec).manage(state, None)
            row["result_pct_so_far"] = round(state.result_pct, 1)
            row["instruction"] = f"CLOSE TODAY ({action.reason})" if action.exit else "hold"
        out.append(row)
    return out


def run(date, decision_time: str = "evening", settings: Settings | None = None, paper: bool = False, rebuild_store: bool = True,
        write: bool = True) -> dict:
    settings = settings or load_settings()
    date = pd.Timestamp(date).normalize()
    steps: list[str] = []

    # R-7: a live list needs every risk setting
    if not paper:
        try:
            risk = dict(settings.require_risk())
        except SettingsError as exc:
            raise DailyRunError(f"Risk settings: {exc}") from exc
    else:
        risk = dict(settings.section("risk"))

    # 1. data and checks (O-1)
    if rebuild_store:
        try:
            build_store(settings, raise_on_error=True)
        except DataCheckError as exc:
            raise DailyRunError(f"Data checks failed; nothing was scored. {exc}") from exc
    steps.append("data_checked")
    ledger = Ledger(settings)
    strategies, strategy_refusals = live_strategies(settings, decision_time, ledger, paper)

    # 2. features, one code path with the backtest (F-6)
    data = MarketData(settings, decision_time, features=pd.DataFrame(columns=["date", "instrument"]))
    asof = Decision(date, decision_time)
    last_data = data.store.last_visible("prices_raw", asof)
    if last_data is None:
        raise DailyRunError(f"No price data is visible at {date.date()} {decision_time}.")
    stale = len(pd.bdate_range(last_data, date)) - 1
    if stale > (0 if decision_time == "evening" else 1) + 1:
        raise DailyRunError(f"The newest price data is {last_data.date()}, {stale} business days before {date.date()}. Refresh the data first.")
    feats = features_asof(data.store, asof)
    today = feats[feats["date"] == date]
    data.features = feats.set_index(["date", "instrument"]).sort_index()
    steps.append("features_built")

    # L-5: live values far outside history
    drift = []
    lcfg = settings.section("ledger")
    hist = feats[feats["date"] < date]
    for _, row in today[today["instrument"].isin(["NIFTY", "BANKNIFTY"])].iterrows():
        h = hist[hist["instrument"] == row["instrument"]].drop(columns=["date", "instrument", "decision_time"])
        for flag in feature_drift(row.drop(labels=["date", "instrument", "decision_time"]), h, float(lcfg["drift_low_pct"]), float(lcfg["drift_high_pct"])):
            drift.append({"instrument": row["instrument"], **flag})

    # 3. score
    preds = _predictions(settings, strategies, feats, date) if strategies else {}
    data.predictions = {k: v.set_index(["date", "instrument"])["prediction"] for k, v in preds.items()}
    view = data.view(date, decision_time)
    candidates, refusals = [], [dict(r, stage="strategy") for r in strategy_refusals]
    for s in strategies:
        cands, refused = s.candidates(view)
        candidates += cands
        refusals += [dict(r, stage="candidate") for r in refused]
    steps.append("scored")

    # 4. select (S-7 to S-13)
    open_now = ledger.open_trades()
    state = BookState(open_instruments=set(open_now["instrument"]))
    entry_date = date if decision_time == "1430" else data.calendar.days[data.calendar.days > date][0] if (data.calendar.days > date).any() else date + pd.offsets.BDay(1)
    y, w, _ = pd.Timestamp(entry_date).isocalendar()
    state.week_counts[(y, w)] = ledger.started_in_week(entry_date)
    limits = dict(settings.section("selection"))
    for key in ("index_book_trades_per_week", "stock_book_trades_per_week", "max_correlated_trades"):
        if risk.get(key) is not None:
            limits[key] = risk[key]
    accepted, refused = select(candidates, state, entry_date, limits, groups=dict(settings.section("selection").get("groups", {})))
    refusals += [dict(r, stage="selection") for r in refused]
    steps.append("selected")

    # 5. size against the risk settings (R-1 to R-6)
    trades = []
    if accepted and all(risk.get(k) is not None for k in ("capital", "risk_per_trade_pct", "max_open_risk_pct", "drawdown_halve_pct", "drawdown_pause_pct", "expiry_safety_days")):
        multiplier, new_state = risk_rules.drawdown_rule(risk, risk_rules.read_state(settings), ledger.realised_pnl())
        if write:
            risk_rules.write_state(settings, new_state)
        open_risk = float(open_now["max_loss"].sum()) if len(open_now) else 0.0
        margin_factor = float(settings.section("risk_model")["margin_factor"])
        for c in accepted:
            if not risk_rules.expiry_ok(c, entry_date, risk):
                refusals.append({"instrument": c.instrument, "strategy": c.strategy, "score": c.score, "reason": "too_close_to_expiry", "stage": "risk"})
                continue
            sized = risk_rules.size(c, risk, open_risk, multiplier, margin_factor)
            if sized.lots < 1:
                refusals.append({"instrument": c.instrument, "strategy": c.strategy, "score": c.score, "reason": sized.refused, "stage": "risk"})
                continue
            open_risk += sized.max_loss
            spec = next(s.spec for s in strategies if s.spec.name == c.strategy)
            trades.append({
                "instrument": c.instrument, "strategy": c.strategy, "book": c.book, "structure": spec.structure,
                "legs": [{"action": "BUY" if l.side == 1 else "SELL", "expiry": str(l.expiry.date()), "strike": l.strike, "type": l.opt_type,
                          "reference_price": p} for l, p in zip(c.legs, c.reference_prices)],
                "lots": sized.lots, "lot_size": c.lot,
                "entry": {"date": str(pd.Timestamp(entry_date).date()), "window": "09:15 to 09:30" if decision_time == "evening" else "15:10 to 15:20",
                          "net_price_limit_per_share": round(c.reasons.get("net_debit", 0.0) * (1 + (0.02 if c.reasons.get("net_debit", 0.0) > 0 else -0.02)), 2)},
                "exit_rules": dict(spec.exit),
                "max_loss_rupees": round(sized.max_loss, 0), "max_loss_per_lot": round(sized.max_loss_per_lot, 0),
                "margin_estimate_rupees": round(sized.margin_estimate, 0),
                "expected_result_pct_of_max_loss": round(c.score, 2) if spec.score["source"] == "model" else None,
                "score": c.score, "reasons": c.reasons,
                "_legs_raw": [(str(l.expiry.date()), l.strike, l.opt_type, l.side) for l in c.legs], "_prices": c.reference_prices,
            })
    elif accepted:
        refusals += [{"instrument": c.instrument, "strategy": c.strategy, "score": c.score, "reason": "risk_settings_not_set", "stage": "risk"} for c in accepted]
    steps.append("sized")

    # 6. write the list (O-2 to O-4, R-9)
    exits = exit_instructions(data, ledger, settings, date, decision_time)
    best_refused = sorted([r for r in refusals if r.get("score") is not None], key=lambda r: -r["score"])[:5]
    out = {"date": str(date.date()), "decision_time": decision_time, "mode": "paper" if paper else "live",
           "data_up_to": str(last_data.date()), "steps": steps,
           "new_trades": [{k: v for k, v in t.items() if not k.startswith("_")} for t in trades],
           "no_trade_today": not trades,
           "message": "" if trades else "No new trade today. Nothing cleared the threshold and the limits.",
           "open_trades": exits, "best_refused": best_refused, "refusals": refusals, "feature_warnings": drift,
           "strategies_live": [s.spec.name for s in strategies]}
    if write:
        for t in trades:
            t["ledger_id"] = ledger.add_recommendation(
                decision_date=date, decision_time=decision_time, entry_date=entry_date, strategy=t["strategy"], book=t["book"],
                instrument=t["instrument"], legs=t["_legs_raw"], lots=t["lots"], lot_size=t["lot_size"], max_loss=t["max_loss_rupees"],
                score=t["score"], reasons=t["reasons"], exit_rules=t["exit_rules"], reference_prices=t["_prices"], mode=out["mode"])
        for t_pub, t in zip(out["new_trades"], trades):
            t_pub["ledger_id"] = t["ledger_id"]
        folder = settings.path("records") / DAILY
        folder.mkdir(parents=True, exist_ok=True)
        stem = f"{date.date()}_{decision_time}"
        (folder / f"{stem}.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
        (folder / f"{stem}.md").write_text(to_markdown(out), encoding="utf-8")
    return out


def to_markdown(out: dict) -> str:
    lines = [f"# Trade list for {out['date']} ({out['decision_time']} decision, {out['mode']})", "",
             f"Data up to {out['data_up_to']}.", ""]
    if out["no_trade_today"]:
        lines += [f"**{out['message']}**", ""]
    for t in out["new_trades"]:
        lines += [f"## {t['instrument']}: {t['structure'].replace('_', ' ')} ({t['strategy']})", "",
                  f"- Lots: {t['lots']} (lot size {t['lot_size']:g})",
                  f"- Enter: {t['entry']['date']}, {t['entry']['window']}, net price limit {t['entry']['net_price_limit_per_share']} per share",
                  f"- Maximum loss: Rs {t['max_loss_rupees']:,.0f} (Rs {t['max_loss_per_lot']:,.0f} per lot). Estimated margin: Rs {t['margin_estimate_rupees']:,.0f}",
                  f"- Expected result: {t['expected_result_pct_of_max_loss']}% of maximum loss" if t["expected_result_pct_of_max_loss"] is not None else f"- Score: {t['score']:.3f}",
                  f"- Exit rules: {t['exit_rules']}", "", "| Leg | Action | Expiry | Strike | Type | Reference price |", "|---|---|---|---|---|---|"]
        lines += [f"| {i + 1} | {l['action']} | {l['expiry']} | {l['strike']:g} | {l['type']} | {l['reference_price']} |" for i, l in enumerate(t["legs"])]
        lines.append("")
    if out["open_trades"]:
        lines += ["## Open trades", "", "| Id | Instrument | Strategy | Instruction |", "|---|---|---|---|"]
        lines += [f"| {t['id']} | {t['instrument']} | {t['strategy']} | {t['instruction']} |" for t in out["open_trades"]]
        lines.append("")
    if out["best_refused"]:
        lines += ["## Best candidates that did not trade", "", "| Instrument | Strategy | Score | Reason |", "|---|---|---|---|"]
        lines += [f"| {r['instrument']} | {r['strategy']} | {r['score']:.3f} | {r['reason']} |" for r in out["best_refused"]]
        lines.append("")
    not_live = [r for r in out["refusals"] if r.get("stage") == "strategy"]
    if not_live:
        lines += ["## Strategies not allowed to trade", "", "| Strategy | Reason |", "|---|---|"]
        lines += [f"| {r['strategy']} | {r['reason']} |" for r in not_live]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--date", required=True)
    parser.add_argument("--time", default="evening", choices=["evening", "1430"])
    parser.add_argument("--paper", action="store_true", help="paper list: gate 2 is enough, and risk settings may be unset")
    parser.add_argument("--no-rebuild", action="store_true", help="skip rebuilding the store (use only right after a build)")
    args = parser.parse_args()
    try:
        out = run(args.date, args.time, paper=args.paper, rebuild_store=not args.no_rebuild)
    except DailyRunError as exc:
        raise SystemExit(f"Daily run stopped: {exc}")
    print(to_markdown(out))


if __name__ == "__main__":
    main()
