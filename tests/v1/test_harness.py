"""The test harness: costs, structures, the engine, statistics, the run log, the sealed period and gates
(V-1 to V-15, S-1 to S-4, R-1, IX-2 to IX-4, IX-7, N-7, N-9)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import synthetic as S
from koscine4.harness import records
from koscine4.harness.costs import order_cost, rates_on, with_slippage
from koscine4.harness.engine import price_trades, run_engine
from koscine4.harness.leak import shuffled_label_score, too_good_columns
from koscine4.harness.market import LookAheadError, MarketData
from koscine4.harness.run import run
from koscine4.harness.search import apply_cell, cells, search
from koscine4.harness.stats import break_even_slippage, independent_count, max_drawdown, summarise, worst_run
from koscine4.harness.walkforward import Fold, GapError, check_fold, monthly_folds
from koscine4.store import Store
from koscine4.store.surface import black_price, read_quotes
from koscine4.strategies.base import Candidate, PositionState, SpecStrategy, Strategy
from koscine4.strategies.spec import SpecError, from_dict
from koscine4.strategies.structures import BUY, SELL, Leg, build_legs, choose_expiry, max_loss_per_share, net_debit, payoff_at_expiry


def spec_dict(name="t_nifty_straddle", **over) -> dict:
    d = dict(name=name, family=name, book="index", instruments=["NIFTY"], structure="long_straddle", decision_time="evening",
             entry_time="next_open", expiry={"min_dte": 7, "kind": "monthly"}, strikes={"max_snap_pct": 1.0},
             exit={"max_hold_sessions": 5, "safety_days": 2}, liquidity={"min_volume": 100, "min_open_interest": 1, "min_premium": 5.0},
             score={"source": "take_all", "threshold": 0.0}, selection={"weekly_cap": True, "max_per_day": 1})
    d.update(over)
    return d


LIMITS = {"max_new_trades_per_week": 5, "max_open_per_instrument": 1}


@pytest.fixture(scope="module")
def data(big):
    return MarketData(big, "evening")


# ---- costs (V-8) ------------------------------------------------------------------------------------------------
def test_costs_match_a_hand_calculation(big):
    costs = dict(big.section("costs"))
    r = rates_on(costs, "2025-01-15")
    assert (r["stt_sell_pct"], r["exchange_txn_pct"]) == (0.1, 0.03503)
    turnover = 100.0 * 50                                             # one lot of 50 at Rs 100
    brokerage, exchange, sebi = 20.0, turnover * 0.03503 / 100, turnover * 10 / 1e7
    gst = 0.18 * (brokerage + exchange + sebi)
    assert order_cost(100.0, 50, False, r) == pytest.approx(brokerage + exchange + sebi + gst + turnover * 0.1 / 100)       # sell: STT
    assert order_cost(100.0, 50, True, r) == pytest.approx(brokerage + exchange + sebi + gst + turnover * 0.003 / 100)      # buy: stamp duty
    assert order_cost(100.0, 50, False, r) == pytest.approx(30.6727, abs=1e-3)
    assert rates_on(costs, "2012-06-01")["stt_sell_pct"] == 0.017 and rates_on(costs, "2026-05-01")["stt_sell_pct"] == 0.15   # dated schedule
    assert with_slippage(100.0, True, 1.0) == 101.0 and with_slippage(100.0, False, 1.0) == 99.0
    assert with_slippage(2.0, True, 1.0) == 2.05 and with_slippage(100.0, True, 0.0) == 100.0                                  # at least one tick


# ---- structures and specifications (S-1 to S-3, R-1, IX-7) ------------------------------------------------------
def _chain(spot=100.0, vol=0.3):
    date, e_week, e_month = pd.Timestamp("2024-03-01"), pd.Timestamp("2024-03-14"), pd.Timestamp("2024-03-28")
    rows = []
    for e in (e_week, e_month):
        t = (e - date).days / 365.0
        for k in np.arange(80.0, 121.0, 1.0):
            for ot in ("CE", "PE"):
                rows.append({"expiry": e, "strike": k, "opt_type": ot, "close": float(black_price(spot, k, t, vol, ot == "CE")),
                             "open": 1.0, "settle": 1.0, "volume": 500, "open_int": 5000, "lot": 100.0})
    rows.append({"expiry": e_month, "strike": np.nan, "opt_type": "nan", "close": spot, "open": spot, "settle": spot, "volume": 9, "open_int": 9, "lot": 100.0})
    return pd.DataFrame(rows), date, e_week, e_month


@pytest.mark.parametrize("structure, extra, n_legs", [
    ("long_straddle", {}, 2),
    ("long_strangle", {"strikes": {"otm_pct": 5.0, "max_snap_pct": 1.0}}, 2),
    ("debit_spread", {"side": "call", "strikes": {"otm_pct": 0.0, "wing_pct": 5.0, "max_snap_pct": 1.0}, "direction_model": ""}, 2),
    ("debit_spread", {"side": "put", "strikes": {"otm_pct": 0.0, "wing_pct": 5.0, "max_snap_pct": 1.0}}, 2),
    ("credit_spread", {"side": "put", "strikes": {"short_otm_pct": 3.0, "wing_pct": 4.0, "max_snap_pct": 1.0}}, 2),
    ("credit_spread", {"side": "call", "strikes": {"short_otm_pct": 3.0, "wing_pct": 4.0, "max_snap_pct": 1.0}}, 2),
    ("iron_condor", {"strikes": {"short_otm_pct": 3.0, "wing_pct": 4.0, "max_snap_pct": 1.0}}, 4),
])
def test_maximum_loss_of_every_structure_matches_its_worst_payoff(structure, extra, n_legs):
    chain, date, _, _ = _chain()
    spec = from_dict(spec_dict(structure=structure, **extra))
    legs = build_legs(spec, chain, 100.0, date)
    assert len(legs) == n_legs
    px = [float(chain[(chain["expiry"] == l.expiry) & (chain["strike"] == l.strike) & (chain["opt_type"] == l.opt_type)]["close"].iloc[0]) for l in legs]
    worst = min(payoff_at_expiry(legs, px, u) for u in np.arange(0.0, 300.5, 0.5))
    assert max_loss_per_share(legs, px) == pytest.approx(-worst) and max_loss_per_share(legs, px) > 0          # R-1
    if all(l.side == BUY for l in legs):
        assert max_loss_per_share(legs, px) == pytest.approx(net_debit(legs, px))                              # only the premium


def test_an_uncovered_sold_option_has_no_maximum_loss_and_is_refused():
    e = pd.Timestamp("2024-03-28")
    with pytest.raises(SpecError, match="no fixed maximum loss"):
        max_loss_per_share([Leg(e, 100.0, "CE", SELL)], [3.0])
    with pytest.raises(SpecError, match="not allowed"):
        from_dict(spec_dict(structure="short_straddle"))                                                       # S-1
    with pytest.raises(SpecError, match="four-leg"):
        from_dict(spec_dict(book="stock", instruments=[], structure="iron_condor", strikes={"short_otm_pct": 3, "wing_pct": 3}))   # S-3
    with pytest.raises(SpecError, match="Supported timings"):
        from_dict(spec_dict(decision_time="evening", entry_time="1515"))                                       # A5
    with pytest.raises(SpecError, match="at most 5"):
        from_dict(spec_dict(exit={"max_hold_sessions": 8, "safety_days": 2}))                                  # A4
    with pytest.raises(SpecError, match="safety_days"):
        from_dict(spec_dict(exit={"max_hold_sessions": 5}))                                                    # R-5
    with pytest.raises(SpecError, match="min_premium"):
        from_dict(spec_dict(liquidity={"min_volume": 1, "min_open_interest": 1}))                              # S-4
    with pytest.raises(SpecError, match="direction_model"):
        from_dict(spec_dict(structure="debit_spread", side="model", strikes={"wing_pct": 3}))                  # S-12
    with pytest.raises(SpecError, match="index strategy must name"):
        from_dict(spec_dict(instruments=["AAA"]))                                                              # IX-1
    assert from_dict(spec_dict(structure="debit_spread", side="call", strikes={"wing_pct": 3})).uses_direction


def test_weekly_and_monthly_expiries_are_chosen_separately():
    chain, date, e_week, e_month = _chain()
    assert choose_expiry(chain, date, 7) == e_week
    assert choose_expiry(chain, date, 7, kind="monthly") == e_month and choose_expiry(chain, date, 7, kind="weekly") == e_week    # IX-7
    assert choose_expiry(chain, date, 20) == e_month and choose_expiry(chain, date, 60) is None
    assert build_legs(from_dict(spec_dict(strikes={"max_snap_pct": 0.1})), chain, 100.4, date) is None        # no strike near enough


# ---- the view (V-4) ---------------------------------------------------------------------------------------------
def test_a_strategy_cannot_read_past_its_decision(data, big):
    d = S.DAYS[200]
    afternoon, evening = data.view(d, "1430"), data.view(d, "evening")
    assert afternoon.visible == S.DAYS[199] and evening.visible == d
    assert afternoon.chain("NIFTY")["date"].unique().tolist() == [S.DAYS[199]]
    assert afternoon.spot("NIFTY") == data.raw_price("NIFTY", S.DAYS[199])
    with pytest.raises(LookAheadError):
        afternoon.chain_on("NIFTY", d)
    with pytest.raises(LookAheadError):
        evening.close_on("NIFTY", S.DAYS[201])
    assert len(evening.chain_on("NIFTY", d)) > 0
    assert evening.feature("NIFTY", "ret_1d") == pytest.approx(pd.read_parquet(big.path("pit") / "features" / "evening.parquet")
                                                               .set_index(["date", "instrument"]).at[(d, "NIFTY"), "ret_1d"])
    with pytest.raises(KeyError):
        evening.feature("NIFTY", "tomorrow_close")                           # F-1: unregistered
    aaa = data.view(S.DAYS[122], "evening")
    assert aaa.results_within("AAA", 5) and not data.view(S.DAYS[112], "evening").results_within("AAA", 5)     # announced on session 115


# ---- the engine (V-5, V-6, V-7, V-14, V-15) ---------------------------------------------------------------------
def test_engine_reproduces_a_trade_worked_out_by_hand(data, big):
    spec = from_dict(spec_dict())
    result = run_engine(SpecStrategy(spec), data, S.DAYS[100], S.DAYS[130], LIMITS)
    t = result.trades.iloc[0]
    q = read_quotes(big, "NIFTY")
    spot = data.raw_price("NIFTY", S.DAYS[100])
    day = q[(q["date"] == S.DAYS[100]) & q["opt_type"].isin(["CE", "PE"])]
    expiry = min(e for e in day["expiry"].unique() if (pd.Timestamp(e) - S.DAYS[100]).days >= 7)
    strike = min(day[day["expiry"] == expiry]["strike"].unique(), key=lambda k: abs(k - spot))

    def px(date, ot, field):
        return float(q[(q["date"] == date) & (q["expiry"] == expiry) & (q["strike"] == strike) & (q["opt_type"] == ot)][field].iloc[0])

    assert t["decision_date"] == S.DAYS[100] and t["entry_date"] == S.DAYS[101] and t["exit_date"] == S.DAYS[105]   # 5 sessions, entry day included
    assert t["exit_reason"] == "time" and t["legs"] == [(str(pd.Timestamp(expiry).date()), strike, "CE", 1), (str(pd.Timestamp(expiry).date()), strike, "PE", 1)]
    entry = [px(S.DAYS[101], "CE", "open"), px(S.DAYS[101], "PE", "open")]
    exit_ = [px(S.DAYS[105], "CE", "close"), px(S.DAYS[105], "PE", "close")]
    assert t["entry_prices"] == entry and t["exit_prices"] == exit_ and t["lot"] == 50

    costs = dict(big.section("costs"))
    r = rates_on(costs, S.DAYS[101])
    fee = lambda price, buy: (20.0 + price * 50 * r["exchange_txn_pct"] / 100 + price * 50 * 10 / 1e7) * 1.18 + \
        (price * 50 * r["stamp_buy_pct"] / 100 if buy else price * 50 * r["stt_sell_pct"] / 100)
    by_hand = (sum(exit_) - sum(entry)) * 50 - sum(fee(p, True) for p in entry) - sum(fee(p, False) for p in exit_)
    priced = price_trades(result.trades, costs, 0.0).iloc[0]
    assert priced["pnl_per_lot"] == pytest.approx(by_hand)                                                    # V-14, costs included
    assert priced["max_loss_per_lot"] == pytest.approx(sum(entry) * 50) and priced["result_pct"] == pytest.approx(100 * by_hand / (sum(entry) * 50))
    slipped = price_trades(result.trades, costs, 1.0).iloc[0]
    assert slipped["pnl_per_lot"] < priced["pnl_per_lot"] and slipped["max_loss_per_lot"] == pytest.approx(sum(entry) * 50 * 1.01)

    again = run_engine(SpecStrategy(spec), data, S.DAYS[100], S.DAYS[130], LIMITS)
    pd.testing.assert_frame_equal(result.trades, again.trades)                                                # V-15
    assert (result.trades["entry_trust"] == "daily_open").all()                                               # V-6
    assert (result.trades["entry_date"].iloc[1:].to_numpy() > result.trades["exit_date"].iloc[:-1].to_numpy()).all()    # one open trade per instrument


def test_a_leg_that_did_not_trade_is_skipped_and_counted(data):
    spec = from_dict(spec_dict(name="t_forced"))

    class Forced(Strategy):
        def __init__(self):
            self.spec = spec

        def candidates(self, view):
            chain = view.chain("NIFTY")
            expiry = choose_expiry(chain, view.date, 7)
            legs = [Leg(expiry, 24000.0, "CE", BUY), Leg(expiry, 24000.0, "PE", BUY)]          # far from the money: no volume
            return [Candidate("t_forced", "index", "NIFTY", view.date, legs, 1.0, 50.0, [1.0, 1.0], 2.0)], []

        def manage(self, state, view=None):
            return SpecStrategy(spec).manage(state, view)

    result = run_engine(Forced(), data, S.DAYS[100], S.DAYS[110], LIMITS)
    assert result.trades.empty
    # eleven decisions; the last one would enter after the window ends
    assert len(result.skips) == 10 and {s["reason"] for s in result.skips} == {"entry:no_volume"}             # V-5


def test_candidates_failing_the_liquidity_check_are_refused_with_a_reason(data):
    far = SpecStrategy(from_dict(spec_dict(structure="long_strangle", strikes={"otm_pct": 20.0, "max_snap_pct": 1.0})))
    cands, refused = far.candidates(data.view(S.DAYS[150], "evening"))
    assert not cands and refused[0]["reason"] == "liquidity:volume"                                           # S-4
    rich = SpecStrategy(from_dict(spec_dict(liquidity={"min_volume": 100, "min_open_interest": 1, "min_premium": 5000.0})))
    assert rich.candidates(data.view(S.DAYS[150], "evening"))[1][0]["reason"] == "liquidity:premium"
    picky = SpecStrategy(from_dict(spec_dict(score={"source": "feature", "feature": "iv_pctile", "threshold": 2.0})))
    assert picky.candidates(data.view(S.DAYS[150], "evening"))[1][0]["reason"] == "below_threshold"           # S-6
    filtered = SpecStrategy(from_dict(spec_dict(filters=[{"feature": "rv_20", "op": "<", "value": 0.0}])))
    assert filtered.candidates(data.view(S.DAYS[150], "evening"))[1][0]["reason"].startswith("filter:rv_20")


def test_a_1430_strategy_enters_the_same_day_and_is_marked_lower_trust(big):
    spec = from_dict(spec_dict(name="t_pcs", structure="credit_spread", side="put", decision_time="1430", entry_time="1515",
                               strikes={"short_otm_pct": 2.0, "wing_pct": 2.0, "max_snap_pct": 1.0},
                               liquidity={"min_volume": 100, "min_open_interest": 1, "min_premium": 1.0}))
    d1430 = MarketData(big, "1430")
    result = run_engine(SpecStrategy(spec), d1430, S.DAYS[100], S.DAYS[140], LIMITS)
    t = result.trades
    assert len(t) >= 4 and (t["entry_date"] == t["decision_date"]).all()
    assert set(t["entry_trust"]) == {"daily_close_for_1515"}                                                  # V-6
    report = summarise(t, result.skips, dict(big.section("costs")), 1.0)
    assert list(report["by_trust"]) == ["daily_close_for_1515"]
    sides = {(l[2], l[3]) for l in t["legs"].iloc[0]}
    assert sides == {("PE", -1), ("PE", 1)}


def test_exits_come_only_from_the_declared_rules():
    e = pd.Timestamp("2024-03-28")
    legs = [Leg(e, 100.0, "CE", BUY), Leg(e, 100.0, "PE", BUY)]
    s = SpecStrategy(from_dict(spec_dict(exit={"max_hold_sessions": 5, "safety_days": 2, "target_pct": 50.0, "stop_pct": 40.0})))
    state = lambda now, held, date: PositionState("NIFTY", legs, [5.0, 5.0], now, 10.0, held, pd.Timestamp(date))
    assert not s.manage(state([5.5, 5.0], 2, "2024-03-10")).exit
    assert s.manage(state([11.0, 4.0], 2, "2024-03-10")).reason == "target"       # +50% of maximum loss
    assert s.manage(state([2.0, 3.0], 2, "2024-03-10")).reason == "stop"
    assert s.manage(state([5.5, 5.0], 5, "2024-03-10")).reason == "time"
    assert s.manage(state([5.5, 5.0], 1, "2024-03-26")).reason == "safety_before_expiry"      # R-5 comes before everything
    assert state([11.0, 4.0], 1, "2024-03-10").result_pct == pytest.approx(50.0)


# ---- statistics (V-9, V-10) -------------------------------------------------------------------------------------
def _trades(rows):
    e = "2024-03-28"
    out = []
    for inst, entry, exit_, p_in, p_out in rows:
        out.append({"strategy": "s", "book": "index", "instrument": inst, "decision_date": pd.Timestamp(entry), "entry_date": pd.Timestamp(entry),
                    "exit_date": pd.Timestamp(exit_), "sessions_held": 3, "exit_reason": "time", "legs": [(e, 100.0, "CE", 1)], "lot": 100.0,
                    "entry_prices": [p_in], "exit_prices": [p_out], "score": 1.0, "entry_trust": "daily_open", "exit_trust": "daily_close",
                    "max_loss_per_share": p_in, "expiry": pd.Timestamp(e)})
    return pd.DataFrame(out)


def test_report_counts_independent_bets_and_has_every_field(big):
    t = _trades([("A", "2024-01-01", "2024-01-05", 10, 14), ("A", "2024-01-03", "2024-01-08", 10, 9),      # overlaps the first
                 ("A", "2024-01-09", "2024-01-12", 10, 12), ("B", "2024-01-03", "2024-01-08", 10, 7), ("B", "2025-02-03", "2025-02-07", 10, 13)])
    assert len(t) == 5 and independent_count(t) == 4                                                          # V-10
    assert max_drawdown(pd.Series([10.0, -30.0, -20.0, 15.0, 40.0])) == 50.0
    assert worst_run(pd.Series([5.0, -10.0, -20.0, 3.0, -5.0])) == {"losing_trades_in_a_row": 2, "total_result_pct": -30.0}
    costs = dict(big.section("costs"))
    report = summarise(t, [{"date": 1, "instrument": "A", "reason": "entry:no_volume"}, {"date": 1, "instrument": "B", "reason": "candidate:no_score"}], costs, 1.0)
    for key in ("trades", "independent_bets", "mean_result_pct", "win_rate_pct", "max_drawdown_pct", "by_year", "by_trust", "by_period",
                "skipped", "break_even_slippage_pct", "at_twice_slippage", "worst_run"):
        assert key in report                                                                                  # V-9
    assert report["trades"] == 5 and report["independent_bets"] == 4 and set(report["by_year"]) == {2024, 2025}
    assert report["skipped"]["at_entry"] == 1 and report["skipped"]["by_reason"]["candidate:no_score"] == 1
    be = break_even_slippage(t, costs)
    assert 0 < be < 20 and abs(price_trades(t, costs, be)["result_pct"].mean()) < 0.05
    assert break_even_slippage(_trades([("A", "2024-01-01", "2024-01-05", 10, 9)]), costs) == 0.0
    assert report["at_twice_slippage"]["mean_result_pct"] < report["mean_result_pct"]


# ---- run log, variants, sealed period, gates (V-11, V-12, N-7, N-9) ---------------------------------------------
def test_runs_are_logged_and_variants_counted(fresh):
    spec = from_dict(spec_dict())
    first = run(spec, S.DAYS[100], S.DAYS[200], settings=fresh)
    second = run(spec, S.DAYS[100], S.DAYS[200], settings=fresh)
    assert first["run_id"] == second["run_id"]                                                                # V-15
    assert json.dumps(first["report"], sort_keys=True, default=str) == json.dumps(second["report"], sort_keys=True, default=str)
    log = [e for e in records.read_log(fresh) if e["kind"] == "run"]
    assert len(log) == 1 and log[0]["data_snapshot"]["hash"] and log[0]["spec"]["structure"] == "long_straddle"     # V-11, N-7, S-2
    assert first["meta"]["variants_tried"] == 1
    other = run(from_dict(spec_dict(exit={"max_hold_sessions": 3, "safety_days": 2})), S.DAYS[100], S.DAYS[200], settings=fresh)
    assert other["run_id"] != first["run_id"] and other["meta"]["variants_tried"] == 2                        # a changed rule is a new variant
    out = fresh.path("runs") / first["run_id"]
    assert (out / "trades.parquet").exists() and (out / "report.json").exists()
    md = (fresh.path("records") / "results" / f"t_nifty_straddle__{first['run_id']}.md").read_text(encoding="utf-8")
    assert "Variants tried in this family: 1" in md and "Break-even slippage" in md


def test_the_sealed_months_need_an_explicit_logged_unlock(fresh):
    spec = from_dict(spec_dict())
    last = Store(fresh).calendar.days[-1]
    seal = records.seal_date(fresh, last)
    assert seal == last - pd.DateOffset(months=3)
    normal = run(spec, S.DAYS[250], None, settings=fresh)
    assert normal["trades"]["exit_date"].max() <= seal and "sealed" in normal["meta"]["sealed_note"]         # V-12
    with pytest.raises(records.SealedPeriodError):
        run(spec, S.DAYS[250], None, settings=fresh, unlock_sealed=True)
    opened = run(spec, S.DAYS[250], None, settings=fresh, unlock_sealed=True, reason="final check")
    assert opened["trades"]["exit_date"].max() > seal and "UNLOCKED" in opened["meta"]["sealed_note"]
    unlocks = [e for e in records.read_log(fresh) if e["kind"] == "sealed_period_unlocked"]
    assert len(unlocks) == 1 and unlocks[0]["reason"] == "final check"
    assert run(spec, S.DAYS[250], seal - pd.Timedelta(days=30), settings=fresh)["meta"]["sealed_note"] == ""


def test_gate_decisions_are_recorded_against_logged_runs(fresh):
    spec = from_dict(spec_dict())
    out = run(spec, S.DAYS[100], S.DAYS[250], settings=fresh)
    assert not records.has_passed(fresh, spec.name, 2)
    with pytest.raises(records.GateError, match="not found"):
        records.record_gate(fresh, spec.name, 2, "pass", ["nosuchrun000"])
    with pytest.raises(records.GateError, match="name the runs"):
        records.record_gate(fresh, spec.name, 2, "pass", [])
    records.record_gate(fresh, spec.name, 2, "pass", [out["run_id"]], note="test")
    assert records.has_passed(fresh, spec.name, 2) and not records.has_passed(fresh, spec.name, 3)            # N-9
    records.record_gate(fresh, spec.name, 2, "fail", [out["run_id"]], note="withdrawn")
    assert not records.has_passed(fresh, spec.name, 2) and len(records.read_gates(fresh, spec.name)) == 2    # appended, never edited
    checks = records.check_gate_2(out["report"], None, None, dict(fresh.section("gate2")))
    assert checks["enough_independent_bets"] is False and "positive_on_sealed_months" in checks["not_yet_measured"]


# ---- walk-forward and leak checks (V-2, V-3, V-13) --------------------------------------------------------------
def test_walk_forward_keeps_the_gap(big):
    cal = Store(big).calendar
    folds = monthly_folds(cal, S.DAYS[60], S.DAYS[300], 10, 10)
    assert len(folds) >= 10
    for f in folds:
        assert cal.index_on_or_before(f.test_start) - cal.index_on_or_before(f.cutoff) == 10                  # V-3
        assert f.test_start.day <= 7 and f.test_end >= f.test_start and f.cutoff < f.test_start
    assert all(a.test_end < b.test_start for a, b in zip(folds, folds[1:]))                                   # V-2: rolls forward, never overlaps
    with pytest.raises(GapError, match="at least 10"):
        monthly_folds(cal, S.DAYS[60], S.DAYS[300], 5, 10)
    f = folds[3]
    check_fold(f, pd.Series([f.cutoff]), cal, 10)
    with pytest.raises(GapError, match="after the cutoff"):
        check_fold(f, pd.Series([f.cutoff + pd.Timedelta(days=1)]), cal, 10)
    with pytest.raises(GapError, match="below the required"):
        check_fold(Fold(cal.days[95], cal.days[100], cal.days[120]), pd.Series([cal.days[90]]), cal, 10)


def test_shuffled_labels_score_about_zero_and_a_planted_column_is_caught():
    from sklearn.linear_model import Ridge

    rng = np.random.default_rng(3)
    n = 3000
    dates = pd.Series(np.repeat(pd.bdate_range("2023-01-02", periods=n // 20), 20))
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    y = pd.Series(0.8 * X["a"] + rng.normal(scale=0.6, size=n))
    half = dates.iloc[n // 2]
    folds = [((dates < half), (dates >= half))]
    fit_predict = lambda xt, yt, xs: Ridge().fit(xt, yt).predict(xs)
    honest = np.corrcoef(fit_predict(X[folds[0][0]], y[folds[0][0]], X[folds[0][1]]), y[folds[0][1]])[0, 1]
    assert honest > 0.5
    assert abs(shuffled_label_score(fit_predict, X, y, dates, folds)) < 0.1                                   # V-13
    X["planted"] = y + rng.normal(scale=0.01, size=n)                                                         # tomorrow's answer
    assert [c["column"] for c in too_good_columns(X, y)] == ["a", "planted"] or [c["column"] for c in too_good_columns(X, y, 0.9)] == ["planted"]
    assert [c["column"] for c in too_good_columns(X, y, 0.9)] == ["planted"]


# ---- grid search (IX-2 to IX-4) ---------------------------------------------------------------------------------
def test_grid_search_chooses_on_the_early_part_and_reports_every_cell(fresh):
    (fresh.root / "configs" / "strategies" / "t_pcs.toml").write_text('''
name = "t_pcs"
family = "t_pcs"
book = "index"
instruments = ["NIFTY"]
structure = "credit_spread"
side = "put"
decision_time = "1430"
entry_time = "1515"
[expiry]
min_dte = 7
kind = "monthly"
[strikes]
short_otm_pct = 2.0
wing_pct = 2.0
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
[selection]
weekly_cap = true
max_per_day = 1
''', encoding="utf-8")
    (fresh.root / "configs" / "searches" / "t_grid.toml").write_text('''
base = "t_pcs"
select_until = "2022-12-30"
min_trades = 3
[grid]
"strikes.short_otm_pct" = [2.0, 3.0]
"side" = ["put", "call"]
''', encoding="utf-8")
    grid = {"strikes.short_otm_pct": [2.0, 3.0], "side": ["put", "call"]}
    assert len(cells(grid)) == 4
    assert apply_cell(spec_dict(), {"strikes.max_snap_pct": 9, "side": "call"}, "g0")["strikes"]["max_snap_pct"] == 9
    report = search("t_grid", settings=fresh)
    assert len(report["cells"]) == 4 and report["best_cell"] is not None                                      # IX-4: the whole grid
    best = next(c for c in report["cells"] if c["cell"] == report["best_cell"])
    assert best["early"]["mean_result_pct"] == max(c["early"]["mean_result_pct"] for c in report["cells"] if c["early"]["trades"] >= 3)   # IX-3
    assert all(c["late"]["trades"] > 0 for c in report["cells"])
    assert records.variants_tried(fresh, "t_pcs") == 4                                                        # every cell is a logged variant
    md = (fresh.path("records") / "results" / "search_t_grid.md").read_text(encoding="utf-8")
    assert "**chosen**" in md and "2022-12-30" in md
