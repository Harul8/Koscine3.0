"""The requirements closed after the first build: S-5 expected result, DR-7 direction tilt, IX-5 daily against
5-minute fills, IX-7 weekly and monthly kept apart, M-6 calibration, R-6 scenario margin, O-6/O-7 web routes,
and the tax on options exercised at expiry."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import synthetic as S
from koscine4.harness import search as grid_search
from koscine4.harness.costs import exercise_cost, order_cost, rates_on, with_slippage
from koscine4.harness.engine import price_one, run_engine
from koscine4.harness.market import MarketData, TrackRecord
from koscine4.harness.stats import summarise
from koscine4.ledger.book import Ledger
from koscine4.models.core import ModelConfig, calibrate, calibration_check, calibration_table, fit_platt
from koscine4.risk import rules as risk_rules
from koscine4.selection.select import BookState, select
from koscine4.store.surface import black_price
from koscine4.strategies.base import Candidate, SpecStrategy
from koscine4.strategies.spec import SpecError, from_dict
from koscine4.strategies.structures import BUY, SELL, Leg, max_loss_per_share

CFG = {"window_trades": 500, "min_trades": 30, "score_bands": 5, "min_band_trades": 10}
LIMITS = {"max_new_trades_per_week": 5, "max_open_per_instrument": 1}


def spec_dict(name="t_condor", **over) -> dict:
    d = dict(name=name, family=name, book="index", instruments=["NIFTY"], structure="iron_condor", decision_time="evening",
             entry_time="next_open", expiry={"min_dte": 7, "kind": "monthly"},
             strikes={"short_otm_pct": 3.0, "wing_pct": 2.0, "max_snap_pct": 1.0},
             exit={"max_hold_sessions": 5, "safety_days": 2}, liquidity={"min_volume": 100, "min_open_interest": 1, "min_premium": 0.05},
             score={"source": "take_all", "threshold": 0.0}, selection={"weekly_cap": True, "max_per_day": 1})
    d.update(over)
    return d


TILT = {"model": "t_dir", "min_confidence": 0.6,
        "up": {"put_short_otm_pct": 2.0, "call_short_otm_pct": 4.0}, "down": {"put_short_otm_pct": 4.0, "call_short_otm_pct": 2.0}}


# ---- S-5: an expected result for every candidate ------------------------------------------------------------------
def test_expected_result_uses_only_trades_closed_by_the_decision():
    tr = TrackRecord()
    days = pd.bdate_range("2024-01-01", periods=60)
    for i, d in enumerate(days):
        tr.add(d, score=float(i % 10), result_pct=float(i % 10) * 2.0 - 5.0)      # higher score, better result
    value, inputs = tr.estimate(days[20], 5.0, CFG)
    assert value != value and inputs["past_trades"] == 21                         # 21 known, below the 30 needed
    value, inputs = tr.estimate(days[59], 9.0, CFG)
    top_band = [float(i % 10) * 2.0 - 5.0 for i in range(60) if i % 10 >= 8]
    assert inputs["past_trades"] == 60 and inputs["band_trades"] == len(top_band)
    assert value == pytest.approx(np.mean(top_band))                              # mean result of past trades with a similar score
    low, _ = tr.estimate(days[59], 0.0, CFG)
    assert low < value
    later = TrackRecord.from_frame(pd.DataFrame({"exit_date": days, "score": [1.0] * 60, "result_pct": [4.0] * 60}))
    value, inputs = later.estimate(days[59], 1.0, CFG)
    assert value == pytest.approx(4.0) and "score_band" not in inputs             # one score only: the mean of all
    tr.add(days[59] + pd.Timedelta(days=5), 9.0, 1000.0)                          # a trade that closes later is never seen
    assert tr.estimate(days[59], 9.0, CFG)[0] == pytest.approx(np.mean(top_band))


def _cand(strategy, instrument, score, expected, book="index"):
    e = pd.Timestamp("2025-01-30")
    return Candidate(strategy, book, instrument, pd.Timestamp("2025-01-01"), [Leg(e, 100.0, "CE", BUY)], score, 50, [5.0], 5.0,
                     expected_result_pct=expected)


def test_selection_takes_the_best_expected_result_across_strategies():
    a = _cand("model_strategy", "NIFTY", score=0.9, expected=2.0)
    b = _cand("rank_strategy", "BANKNIFTY", score=0.1, expected=6.0)
    c = _cand("new_strategy", "AAA", score=5.0, expected=float("nan"), book="stock")
    accepted, refused = select([a, b, c], BookState(), "2025-01-02", LIMITS, max_per_day=2)
    assert [x.instrument for x in accepted] == ["BANKNIFTY", "NIFTY"]             # 6% before 2%; no expected result goes last
    assert refused[0]["reason"] == "daily_limit" and refused[0]["expected_result_pct"] != refused[0]["expected_result_pct"]
    accepted, _ = select([a, b, c], BookState(), "2025-01-02", LIMITS, max_per_day=1, rank_by="score")
    assert [x.instrument for x in accepted] == ["AAA"]
    with pytest.raises(ValueError):
        select([a], BookState(), "2025-01-02", LIMITS, rank_by="luck")


def test_backtest_candidates_carry_an_expected_result_from_the_runs_own_earlier_trades(big):
    data = MarketData(big, "evening")
    spec = from_dict(spec_dict(name="t_straddle", structure="long_straddle", strikes={"max_snap_pct": 1.0}))
    out = run_engine(SpecStrategy(spec), data, S.DAYS[30], S.DAYS[-1], LIMITS, slippage_pct=1.0)
    t = out.trades.sort_values("entry_date")
    assert t["expected_result_pct"].iloc[:30].isna().all()                         # nothing known yet
    known = t[t["expected_result_pct"].notna()]
    assert len(known) > 10
    first = known.iloc[0]
    priced = [price_one(r.legs, r.entry_prices, r.exit_prices, r.lot, r.entry_date, r.exit_date, r.exit_reason,
                        dict(big.section("costs")), 1.0)["result_pct"] for r in t.itertuples() if r.exit_date <= first["decision_date"]]
    assert len(priced) >= 30 and first["expected_result_pct"] == pytest.approx(np.mean(priced))   # take_all: one score band
    assert set(t["expiry_kind"]) == {"monthly"}


# ---- DR-7: direction tilts between two neutral variants -------------------------------------------------------------
def test_a_tilt_is_allowed_only_between_neutral_variants():
    assert from_dict(spec_dict(tilt=TILT)).tilt["model"] == "t_dir"
    with pytest.raises(SpecError, match="neutral structure"):
        from_dict(spec_dict(structure="credit_spread", side="put", strikes={"short_otm_pct": 3.0, "wing_pct": 2.0}, tilt=TILT))
    with pytest.raises(SpecError, match="may only move strikes"):
        from_dict(spec_dict(tilt=dict(TILT, up={"put_short_otm_pct": 2.0, "max_hold_sessions": 3})))
    with pytest.raises(SpecError, match="min_confidence"):
        from_dict(spec_dict(tilt=dict(TILT, min_confidence=0.4)))
    with pytest.raises(SpecError, match="must name the direction model"):
        from_dict(spec_dict(tilt=dict(TILT, model="")))


def test_the_forecast_moves_the_strikes_but_never_decides_the_trade(big):
    d = S.DAYS[200]
    spec = from_dict(spec_dict(tilt=TILT))

    def legs_with(p):
        preds = {} if p is None else {"t_dir": pd.DataFrame({"date": [d], "instrument": ["NIFTY"], "prediction": [p]})}
        cands, refused = SpecStrategy(spec).candidates(MarketData(big, "evening", predictions=preds).view(d, "evening"))
        assert len(cands) == 1, refused
        c = cands[0]
        shorts = {l.opt_type: l.strike for l in c.legs if l.side == SELL}
        return shorts, c.reasons["tilt"]

    neutral, note = legs_with(0.5)
    assert note["variant"] == "neutral" and note["reason"] == "forecast_not_confident"
    up, note = legs_with(0.8)
    assert note["variant"] == "up" and up["PE"] > neutral["PE"] and up["CE"] > neutral["CE"]       # both shorts move up
    down, note = legs_with(0.2)
    assert note["variant"] == "down" and down["PE"] < neutral["PE"] and down["CE"] < neutral["CE"]
    none, note = legs_with(None)
    assert none == neutral and note["reason"] == "no_forecast_available"                          # still a trade, untilted


# ---- IX-7: weekly and monthly are separate instruments ---------------------------------------------------------------
def test_an_index_strategy_must_say_which_expiries_it_trades():
    with pytest.raises(SpecError, match="IX-7"):
        from_dict(spec_dict(expiry={"min_dte": 7}))
    with pytest.raises(SpecError, match="expiry.kind"):
        from_dict(spec_dict(expiry={"min_dte": 7, "kind": "quarterly"}))
    stock = from_dict(spec_dict(book="stock", instruments=[], structure="long_straddle", expiry={"min_dte": 7}, strikes={"max_snap_pct": 1.0}))
    assert stock.expiry.get("kind", "any") == "any"


def test_reports_split_results_by_expiry_kind():
    e = pd.Timestamp("2025-01-30")
    rows = []
    for i, kind in enumerate(["weekly"] * 3 + ["monthly"] * 2):
        d = pd.Timestamp("2025-01-01") + pd.Timedelta(days=7 * i)
        rows.append({"instrument": "NIFTY", "decision_date": d, "entry_date": d, "exit_date": d + pd.Timedelta(days=3), "exit_reason": "time",
                     "legs": [(e, 100.0, "CE", BUY)], "entry_prices": [5.0], "exit_prices": [6.0 if kind == "weekly" else 4.0], "lot": 50.0,
                     "score": 0.0, "entry_trust": "full", "expiry_kind": kind})
    report = summarise(pd.DataFrame(rows), [], {"brokerage_per_order": 0.0, "stt_sell_pct": 0.0, "exchange_txn_pct": 0.0,
                                                "sebi_per_crore": 0.0, "stamp_buy_pct": 0.0, "gst_pct": 0.0}, 0.0)
    assert report["by_expiry_kind"]["weekly"]["trades"] == 3 and report["by_expiry_kind"]["weekly"]["mean_result_pct"] == pytest.approx(20.0)
    assert report["by_expiry_kind"]["monthly"]["mean_result_pct"] == pytest.approx(-20.0)


def test_a_grid_chooses_a_setting_for_each_expiry_kind():
    def row(cell, kind, mean, trades=50):
        return {"cell": cell, "expiry_kind": kind, "early": {"trades": trades, "mean_result_pct": mean}}
    rows = [row(0, "weekly", 1.0), row(1, "weekly", 3.0), row(2, "monthly", 2.0), row(3, "monthly", 9.0, trades=5)]
    assert grid_search.choose(rows, min_trades=40) == {"monthly": 2, "weekly": 1}                  # never pooled; too few trades not chosen


# ---- IX-5: daily fills against 5-minute fills --------------------------------------------------------------------------
def test_the_comparison_lines_up_daily_and_five_minute_results_cell_by_cell(fresh):
    res = grid_search.results_dir(fresh)
    block = lambda n, m: {"trades": n, "mean_result_pct": m, "win_rate_pct": 50.0}
    daily = {"grid": "g", "fills": "daily", "first_5m_day": "2024-10-01", "slippage_pct": 1.0, "end": "2026-10-30",
             "best_by_kind": {"monthly": 0},
             "cells": [{"cell": 0, "expiry_kind": "monthly", "all": block(300, 4.0), "overlap_5m": block(40, 3.0)},
                       {"cell": 1, "expiry_kind": "weekly", "all": block(500, 2.0), "overlap_5m": block(90, 1.0)}]}
    five = {"grid": "g", "fills": "5m", "end": "2026-10-30", "cells": [{"cell": 0, "expiry_kind": "monthly", "all": block(41, 1.5)}]}
    with pytest.raises(FileNotFoundError, match="--fills 5m"):
        (res / "search_g.json").write_text(json.dumps(daily), encoding="utf-8")
        grid_search.compare("g", fresh)
    (res / "search_g_5m.json").write_text(json.dumps(five), encoding="utf-8")
    out = grid_search.compare("g", fresh)
    c0, c1 = out["cells"]
    assert c0["chosen_for"] == "monthly" and c0["difference_pct"] == pytest.approx(-1.5) and c0["five_min_trades"] == 41
    assert c1["five_min_trades"] == 0 and c1["difference_pct"] != c1["difference_pct"]
    md = (res / "search_g_compare.md").read_text(encoding="utf-8")
    assert "5-minute minus daily" in md and "| 0 | monthly | monthly |" in md


def test_five_minute_fills_are_refused_for_an_evening_strategy(fresh):
    (fresh.root / "configs" / "strategies" / "t_eve.toml").write_text('''
name = "t_eve"
family = "t_eve"
book = "index"
instruments = ["NIFTY"]
structure = "long_straddle"
decision_time = "evening"
entry_time = "next_open"
[expiry]
min_dte = 7
kind = "monthly"
[strikes]
max_snap_pct = 1.0
[exit]
max_hold_sessions = 5
safety_days = 2
[liquidity]
min_volume = 100
min_open_interest = 1
min_premium = 1.0
[score]
source = "take_all"
threshold = 0.0
''', encoding="utf-8")
    (fresh.root / "configs" / "searches").mkdir(parents=True, exist_ok=True)
    (fresh.root / "configs" / "searches" / "t_eve_grid.toml").write_text(
        'base = "t_eve"\nselect_until = "2023-06-30"\n[grid]\n"expiry.min_dte" = [7, 14]\n', encoding="utf-8")
    with pytest.raises(ValueError, match="only for 15:15 entries"):
        grid_search.search("t_eve_grid", fresh, fills="5m")


# ---- tax on options exercised at expiry --------------------------------------------------------------------------------
def test_an_option_held_to_expiry_pays_exercise_tax_instead_of_closing_costs(big):
    costs = dict(big.section("costs"))
    e = pd.Timestamp("2025-01-30")
    legs = [(e, 100.0, "CE", BUY), (e, 110.0, "CE", SELL)]
    got = price_one(legs, [8.0, 3.0], [12.0, 2.0], 50.0, "2025-01-20", e, "expired", costs, 1.0)
    r = rates_on(costs, e)
    buy_in, sell_in = with_slippage(8.0, True, 1.0), with_slippage(3.0, False, 1.0)
    entry_costs = order_cost(buy_in, 50.0, True, r) + order_cost(sell_in, 50.0, False, r)
    tax = 12.0 * 50.0 * r["stt_exercise_pct"] / 100.0                       # only the bought leg in the money is taxed
    assert got["cost"] == pytest.approx(entry_costs + tax)
    assert got["pnl"] == pytest.approx((12.0 - buy_in - (2.0 - sell_in)) * 50.0 - entry_costs - tax)   # no exit slippage
    assert exercise_cost(-3.0, 50.0, r) == 0.0
    with pytest.raises(KeyError, match="stt_exercise_pct"):
        exercise_cost(1.0, 1.0, {})
    closed = price_one(legs, [8.0, 3.0], [12.0, 2.0], 50.0, "2025-01-20", e, "time", costs, 1.0)
    assert closed["cost"] > entry_costs and closed["cost"] != got["cost"]


# ---- R-6: scenario margin ----------------------------------------------------------------------------------------------
def test_scenario_margin_for_a_credit_spread_is_bounded_and_bought_options_need_none(big):
    cfg, date, e, spot = big.section("margin"), pd.Timestamp("2025-01-10"), pd.Timestamp("2025-01-30"), 23000.0
    t = (e - date).days / 365.0
    k_short, k_long = 22300.0, 22000.0
    legs = [Leg(e, k_short, "PE", SELL), Leg(e, k_long, "PE", BUY)]
    prices = [float(black_price(spot, k, t, 0.14, False)) for k in (k_short, k_long)]
    per_lot, inputs = risk_rules.scenario_margin(legs, prices, 75.0, spot, date, cfg, "index", 1.25)
    assert inputs["method"] == "scenario" and inputs["implied_vols"] == pytest.approx([0.14, 0.14], abs=1e-3)
    width = (k_short - k_long) * 75.0
    exposure = float(cfg["exposure_pct_index"]) / 100.0 * spot * 75.0
    assert 0 < per_lot - exposure <= width                                          # a spread never loses more than its width
    assert inputs["price_scan_pct"] >= float(cfg["min_price_scan_pct_index"])
    assert per_lot == pytest.approx(inputs["scenario_loss_per_share"] * 75.0 + exposure, rel=1e-3)
    bought = risk_rules.scenario_margin([Leg(e, k_short, "PE", BUY)], prices[:1], 75.0, spot, date, cfg, "index", 1.25)
    assert bought[0] == 0.0
    impossible, inputs = risk_rules.scenario_margin(legs, [0.0, 0.0], 75.0, spot, date, cfg, "index", 1.25)
    assert inputs["method"].startswith("fallback") and impossible == pytest.approx(risk_rules.margin_estimate(legs, [0.0, 0.0], 75.0, 1.25))


def test_sizing_uses_the_scenario_margin_when_given():
    e = pd.Timestamp("2025-01-30")
    c = Candidate("s", "index", "NIFTY", pd.Timestamp("2025-01-10"), [Leg(e, 100.0, "PE", SELL), Leg(e, 90.0, "PE", BUY)], 0.0, 50, [3.0, 1.0],
                  max_loss_per_share([Leg(e, 100.0, "PE", SELL), Leg(e, 90.0, "PE", BUY)], [3.0, 1.0]))
    risk = {"capital": 100000.0, "risk_per_trade_pct": 5.0, "max_open_risk_pct": 50.0}
    plain = risk_rules.size(c, risk, 0.0, 1.0, 1.25)                                 # 400 per lot, so 12 lots
    assert plain.lots == 12 and plain.margin_estimate == pytest.approx(12 * 400 * 1.25)
    tight = risk_rules.size(c, risk, 0.0, 1.0, 1.25, open_margin=96000.0, margin_per_lot=1000.0)
    assert tight.lots == 4 and tight.margin_estimate == pytest.approx(4000.0)


# ---- M-6: calibration correction ---------------------------------------------------------------------------------------
def test_calibration_corrects_overconfident_probabilities_without_using_unknown_outcomes():
    rng = np.random.default_rng(3)
    dates = pd.bdate_range("2020-01-01", periods=1500)
    truth = rng.uniform(0.2, 0.8, len(dates))
    y = (rng.uniform(size=len(dates)) < truth).astype(float)
    raw = 1.0 / (1.0 + np.exp(-3.0 * np.log(truth / (1.0 - truth))))              # far too sure of itself
    cutoffs = [dates[i] for i in range(200, 1500, 100)]
    cut = pd.Series(pd.NaT, index=range(len(dates)))
    for c in cutoffs:
        cut[dates > c] = c
    preds = pd.DataFrame({"date": dates, "instrument": "NIFTY", "prediction": raw, "cutoff": cut.to_numpy()}).dropna(subset=["cutoff"])
    outcomes = pd.DataFrame({"date": dates, "instrument": "NIFTY", "y": y, "window_end": dates + pd.Timedelta(days=7)})
    cfg = ModelConfig("t", "classification", "logistic", ("rv_20",), min_calibration_rows=300)
    out, fitted = calibrate(cfg, preds, outcomes)
    assert fitted[str(cutoffs[0].date())] is None                                   # too few known rows at the first cutoff
    later = out[out["cutoff"] >= cutoffs[5]]
    before = calibration_check(calibration_table(later["raw_prediction"], pd.Series(y, index=dates).reindex(later["date"]).to_numpy()), 0.05)
    after = calibration_check(calibration_table(later["prediction"], pd.Series(y, index=dates).reindex(later["date"]).to_numpy()), 0.05)
    assert after["largest_gap"] < before["largest_gap"] and fitted[str(cutoffs[5].date())].a < 0.6     # slope pulled back toward 1/3
    first = fitted[str(cutoffs[4].date())]
    known = preds.merge(outcomes, on=["date", "instrument"])
    known = known[(known["window_end"] <= cutoffs[4]) & (known["date"] <= cutoffs[4])]
    assert first is not None and first.rows == len(known)                           # only outcomes ended by the cutoff
    reg = ModelConfig("r", "regression", "ridge", ("rv_20",))
    same, none = calibrate(reg, preds, outcomes)
    assert none == {} and (same["prediction"] == same["raw_prediction"]).all()
    assert fit_platt([0.5] * 10, [1.0] * 10, 5) is None
    with pytest.raises(Exception, match="calibration"):
        ModelConfig("x", "classification", "logistic", ("rv_20",), calibration="isotonic").validate()


# ---- O-6, O-7: the web routes --------------------------------------------------------------------------------------------
def test_the_routes_list_the_trade_lists_and_record_fills(fresh, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from koscine4.api import routes

    monkeypatch.setattr(routes, "load_settings", lambda: fresh)
    folder = fresh.path("records") / "daily"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "2025-01-10_evening.json").write_text(json.dumps({"date": "2025-01-10", "decision_time": "evening", "mode": "paper",
                                                                "new_trades": [{"x": float("nan")}], "open_trades": []}), encoding="utf-8")
    ledger = Ledger(fresh)
    e = "2025-01-30"
    rid = ledger.add_recommendation(decision_date="2025-01-10", decision_time="evening", entry_date="2025-01-13", strategy="t", book="index",
                                    instrument="NIFTY", legs=[(e, 23000.0, "PE", -1), (e, 22700.0, "PE", 1)], lots=1, lot_size=75.0,
                                    max_loss=15000.0, score=0.0, reasons={"spot": 23500.0}, exit_rules={}, reference_prices=[60.0, 30.0])
    other = ledger.add_recommendation(decision_date="2025-01-10", decision_time="evening", entry_date="2025-01-13", strategy="t", book="index",
                                      instrument="BANKNIFTY", legs=[(e, 50000.0, "CE", 1)], lots=1, lot_size=30.0, max_loss=3000.0, score=0.0,
                                      reasons={}, exit_rules={}, reference_prices=[100.0])
    app = FastAPI()
    app.include_router(routes.router)
    client = TestClient(app)

    lists = client.get("/v1/trade-lists").json()
    assert lists == [{"date": "2025-01-10", "decision_time": "evening", "mode": "paper", "new_trades": 1, "open_trades": 0}]
    assert client.get("/v1/trade-list/2025-01-10").json()["new_trades"] == [{"x": None}]           # NaN is sent as null
    assert client.get("/v1/trade-list/2025-01-11").status_code == 404
    rec = client.get(f"/v1/ledger/{rid}").json()
    assert rec["legs"][0] == [e, 23000.0, "PE", -1] and rec["fills"] == [] and rec["status"] == "recommended"
    assert client.get("/v1/ledger/9999").status_code == 404
    assert [r["id"] for r in client.get("/v1/ledger/open").json()] == [rid, other]

    fill = lambda kind, leg, price: client.post("/v1/ledger/fills", json={"rec_id": rid, "kind": kind, "leg": leg, "price": price,
                                                                           "quantity": 75, "filled_at": "2025-01-13 09:20"})
    assert client.post(f"/v1/ledger/{rid}/close", json={"exit_date": "2025-01-15", "exit_reason": "time"}).status_code == 400
    assert fill("entry", 0, 62.0).status_code == 200 and fill("entry", 1, 29.0).status_code == 200
    assert fill("entry", 5, 1.0).status_code == 400                                               # no such leg
    assert client.post(f"/v1/ledger/{rid}/not-taken", json={"note": "late"}).status_code == 400   # it has fills already
    assert fill("exit", 0, 40.0).status_code == 200 and fill("exit", 1, 15.0).status_code == 200
    closed = client.post(f"/v1/ledger/{rid}/close", json={"exit_date": "2025-01-15", "exit_reason": "time"}).json()
    assert closed["pnl"] == pytest.approx(((62.0 - 40.0) - (29.0 - 15.0)) * 75.0) and closed["status"] == "closed"
    assert client.post(f"/v1/ledger/{other}/not-taken", json={"note": "missed the window"}).json()["status"] == "not_taken"
    assert [r["id"] for r in client.get("/v1/ledger/closed").json()] == [rid]
    assert client.get("/v1/ledger/open").json() == []
