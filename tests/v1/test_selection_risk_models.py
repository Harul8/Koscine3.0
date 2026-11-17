"""Trade selection, risk rules and models (S-5 to S-13, DR-3, DR-7, R-1 to R-6, R-10, M-1 to M-7, U-6)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import synthetic as S
from koscine4.harness import records
from koscine4.harness.run import run
from koscine4.models.core import ModelConfig, load_latest_model, make_model
from koscine4.models.direction import research
from koscine4.models.train import train
from koscine4.outcomes import build as outcomes
from koscine4.risk import rules as risk_rules
from koscine4.selection.select import BookState, select
from koscine4.store import Store
from koscine4.strategies.base import Candidate
from koscine4.strategies.spec import from_dict
from koscine4.strategies.structures import BUY, SELL, Leg
from koscine4.universe.build import INDEX_BOOK, stock_list

E = pd.Timestamp("2024-03-28")
MON = pd.Timestamp("2024-03-04")
LIMITS = {"max_new_trades_per_week": 5, "max_open_per_instrument": 1}
GROUPS = {"banks": ["BANKNIFTY", "HDFCBANK", "ICICIBANK"]}
RISK = {"capital": 1_000_000, "risk_per_trade_pct": 1.0, "max_open_risk_pct": 3.0, "drawdown_halve_pct": 5.0,
        "drawdown_pause_pct": 10.0, "expiry_safety_days": 2}


def cand(instrument, score, book="stock", strategy="s", uses_direction=False, legs=None, prices=None, lot=50.0):
    legs = legs or [Leg(E, 100.0, "PE", SELL), Leg(E, 95.0, "PE", BUY)]
    prices = prices or [3.0, 1.0]
    return Candidate(strategy, book, instrument, MON, legs, score, lot, prices, 3.0, uses_direction=uses_direction)


# ---- selection --------------------------------------------------------------------------------------------------
def test_at_most_five_new_trades_in_a_week_best_score_first():
    cands = [cand(f"S{i}", score=i / 10) for i in range(8)]
    accepted, refused = select(cands, BookState(), MON, LIMITS)
    assert [c.instrument for c in accepted] == ["S7", "S6", "S5", "S4", "S3"]                 # S-5, S-7
    assert {r["reason"] for r in refused} == {"weekly_limit"} and len(refused) == 3          # R-9: every refusal has a reason
    state = BookState()
    for _ in range(3):
        state.record_start(MON + pd.Timedelta(days=1), "stock")
    assert len(select(cands, state, MON + pd.Timedelta(days=3), LIMITS)[0]) == 2              # three already started this week
    assert len(select(cands, state, MON + pd.Timedelta(days=7), LIMITS)[0]) == 5              # a new week
    assert select([], BookState(), MON, LIMITS) == ([], [])                                   # S-10: no trade is a valid answer
    assert len(select(cands[:2], BookState(), MON, LIMITS)[0]) == 2                           # ... and nothing is added to fill a slot


def test_one_open_trade_per_instrument():
    accepted, refused = select([cand("AAA", 0.9, strategy="a"), cand("AAA", 0.8, strategy="b"), cand("BBB", 0.5)],
                               BookState(open_instruments={"BBB"}), MON, LIMITS)
    assert [(c.instrument, c.strategy) for c in accepted] == [("AAA", "a")]                   # S-8
    assert {r["reason"] for r in refused} == {"instrument_already_has_an_open_trade"}


def test_trades_that_lose_together_are_limited_and_bank_nifty_counts_with_banks():
    limits = dict(LIMITS, max_correlated_trades=1)
    cands = [cand("BANKNIFTY", 0.9, book="index"), cand("HDFCBANK", 0.8), cand("TCS", 0.7)]
    accepted, refused = select(cands, BookState(), MON, limits, GROUPS)
    assert [c.instrument for c in accepted] == ["BANKNIFTY", "TCS"] and refused[0]["reason"] == "correlated_limit:banks"     # S-9
    accepted, refused = select(cands[1:], BookState(open_instruments={"ICICIBANK"}), MON, limits, GROUPS)
    assert [c.instrument for c in accepted] == ["TCS"]


def test_the_two_books_have_their_own_share_of_the_week():
    limits = dict(LIMITS, index_book_trades_per_week=2, stock_book_trades_per_week=3)
    cands = [cand(f"I{i}", 0.9 - i / 100, book="index") for i in range(4)] + [cand(f"S{i}", 0.5 - i / 100) for i in range(5)]
    accepted, refused = select(cands, BookState(), MON, limits)
    assert sum(c.book == "index" for c in accepted) == 2 and sum(c.book == "stock" for c in accepted) == 3    # S-13
    assert len(accepted) == 5 and {r["reason"] for r in refused} == {"weekly_limit_index_book", "weekly_limit"}
    only_stock, refused = select(cands[4:], BookState(), MON, limits)
    assert len(only_stock) == 3 and {r["reason"] for r in refused} == {"weekly_limit_stock_book"}


def test_a_strategy_without_a_gate_record_or_with_an_unproven_direction_model_is_refused():
    accepted, refused = select([cand("AAA", 0.9, strategy="ungated"), cand("BBB", 0.8, strategy="gated")], BookState(), MON, LIMITS,
                               gate_passed=lambda s: s == "gated")
    assert [c.strategy for c in accepted] == ["gated"] and refused[0]["reason"] == "strategy_has_no_gate_2_record"           # S-11
    accepted, refused = select([cand("AAA", 0.9, uses_direction=True), cand("BBB", 0.1)], BookState(), MON, LIMITS,
                               direction_passed=lambda s: False)
    assert [c.instrument for c in accepted] == ["BBB"]                                        # S-12, DR-7: direction alone cannot create a trade
    assert refused[0]["reason"] == "direction_model_has_not_passed_its_gate"


# ---- risk -------------------------------------------------------------------------------------------------------
def test_maximum_loss_and_size_match_a_hand_calculation():
    c = cand("NIFTY", 1.0, book="index")                              # sell 100 put at 3, buy 95 put at 1: width 5, credit 2
    assert risk_rules.max_loss_per_lot(c.legs, c.reference_prices, c.lot) == pytest.approx(150.0)             # R-1: (5 - 2) x 50
    sized = risk_rules.size(c, RISK, open_risk=0.0, size_multiplier=1.0, margin_factor=1.25)
    assert sized.lots == 66 and sized.max_loss == pytest.approx(9900.0)                                       # R-2: 10,000 / 150, rounded down
    assert sized.margin_estimate == pytest.approx(66 * 150 * 1.25)                                            # R-6
    big_lot = cand("NIFTY", 1.0, lot=5000.0)
    refused = risk_rules.size(big_lot, RISK, 0.0, 1.0, 1.25)
    assert refused.lots == 0 and refused.refused == "one_lot_exceeds_risk_per_trade"                          # a trade that rounds to zero lots is not taken
    assert risk_rules.size(c, RISK, 0.0, 0.5, 1.25).lots == 33                                                # halved by the drawdown rule
    long_only = cand("AAA", 1.0, legs=[Leg(E, 100.0, "CE", BUY), Leg(E, 100.0, "PE", BUY)], prices=[4.0, 3.5])
    s2 = risk_rules.size(long_only, RISK, 0.0, 1.0, 1.25)
    assert s2.max_loss_per_lot == pytest.approx(375.0) and s2.margin_estimate == 0.0                          # bought options: the premium, no margin


def test_total_open_risk_and_margin_limits_refuse_trades():
    c = cand("NIFTY", 1.0)
    assert risk_rules.size(c, RISK, open_risk=29_000.0, size_multiplier=1.0, margin_factor=1.25).lots == 6    # R-3: 1,000 of room left
    full = risk_rules.size(c, RISK, 29_950.0, 1.0, 1.25)
    assert full.lots == 0 and full.refused == "total_open_risk_limit"
    tight = risk_rules.size(c, RISK, 0.0, 1.0, 1.25, open_margin=999_900.0)
    assert tight.lots == 0 and tight.refused == "margin_exceeds_free_capital"                                 # R-6
    assert risk_rules.size(c, RISK, 0.0, 1.0, 1.25, open_margin=999_000.0).lots == 5                          # 1,000 free / 187.5 per lot
    assert risk_rules.expiry_ok(c, "2024-03-20", RISK) and not risk_rules.expiry_ok(c, "2024-03-26", RISK)    # R-5


def test_drawdown_rule_halves_then_pauses_until_the_owner_resumes(fresh):
    state = risk_rules.read_state(fresh)
    mult, state = risk_rules.drawdown_rule(RISK, state, realised_pnl=-20_000.0)
    assert mult == 1.0 and not state["paused"]
    mult, state = risk_rules.drawdown_rule(RISK, state, -60_000.0)
    assert mult == 0.5                                                                                        # R-4: first level
    mult, state = risk_rules.drawdown_rule(RISK, state, -110_000.0)
    assert mult == 0.0 and state["paused"]                                                                    # second level: no new trades
    mult, state = risk_rules.drawdown_rule(RISK, state, +50_000.0)
    assert mult == 0.0 and state["paused"]                                                                    # recovering does not lift the pause
    risk_rules.write_state(fresh, state)
    assert risk_rules.size(cand("X", 1.0), RISK, 0.0, mult, 1.25).refused == "new_trades_paused_by_drawdown_rule"
    resumed = risk_rules.resume(fresh, "reviewed the ledger")
    assert not resumed["paused"] and "owner" in resumed["note"]
    assert risk_rules.drawdown_rule(RISK, risk_rules.read_state(fresh), -110_000.0 + 0.0)[0] == 0.0           # still 11% down from capital
    assert risk_rules.drawdown_rule(RISK, risk_rules.read_state(fresh), -10_000.0)[0] == 1.0


def test_risk_settings_are_proposed_from_the_backtest_and_never_applied(fresh):
    results = [20.0, -100.0, -100.0, -60.0, 40.0, 30.0, -100.0, 50.0]
    trades = pd.DataFrame({"exit_date": pd.bdate_range("2024-01-01", periods=len(results)), "result_pct": results})
    p = risk_rules.propose(trades, capital=1_000_000, largest_acceptable_loss=100_000)
    assert p["worst_losing_run"] == {"losing_trades_in_a_row": 3, "total_result_pct": -260.0}
    assert p["largest_drawdown_pct_of_one_max_loss"] == pytest.approx(290.0)                                  # 20 down to -270
    assert p["proposed"]["risk_per_trade_pct"] == pytest.approx(100_000 / 2.9 / 1_000_000 * 100, abs=0.01)
    assert p["largest_drawdown_in_rupees_at_proposed_size"] == pytest.approx(100_000, abs=1)                  # R-10: the worst run at the proposed size
    assert p["proposed"]["drawdown_pause_pct"] == 10.0 and p["proposed"]["drawdown_halve_pct"] == 5.0
    before = (fresh.root / "configs" / "settings.toml").read_text(encoding="utf-8")
    path = risk_rules.write_proposal(fresh, p)
    assert json.loads(path.read_text(encoding="utf-8"))["status"].startswith("proposal only")
    assert (fresh.root / "configs" / "settings.toml").read_text(encoding="utf-8") == before                  # nothing takes effect by itself


# ---- models -----------------------------------------------------------------------------------------------------
MODEL = '''
name = "{name}"
kind = "regression"
algorithm = "{algorithm}"
decision_time = "evening"
min_train_rows = {min_rows}
seed = 7
features = ["rv_5", "rv_20", "atr_pct_14", "range_pct", "ret_5d", "iv_rv_ratio"]
[target]
{target}
[params]
alpha = 1.0
n_estimators = 30
num_leaves = 4
min_child_samples = 10
'''
MOVES = 'table = "moves"\ncolumn = "fwd_abs"\nhorizon = 5'


def _write_model(settings, name, algorithm="ridge", target=MOVES, min_rows=200):
    (settings.root / "configs" / "models" / f"{name}.toml").write_text(
        MODEL.format(name=name, algorithm=algorithm, target=target, min_rows=min_rows), encoding="utf-8")


def test_walk_forward_training_respects_the_cutoff_and_scores_only_the_list(fresh):
    _write_model(fresh, "t_abs")
    report = train("t_abs", settings=fresh, start=S.DAYS[120])
    preds = pd.read_parquet(fresh.path("pit") / "predictions" / "t_abs.parquet")
    cal = Store(fresh).calendar
    assert report["rows"] == len(preds) > 100 and "sealed" in report["sealed_note"]                           # stops at the seal date (V-12)
    assert preds["date"].max() <= records.seal_date(fresh, cal.days[-1])
    gap = [cal.index_on_or_before(d) - cal.index_on_or_before(c) for d, c in zip(preds["date"], preds["cutoff"])]
    assert min(gap) >= 10                                                                                     # V-3, M-4: ten sessions between training and test
    for month, g in preds.groupby(preds["date"].dt.to_period("M")):
        allowed = set(INDEX_BOOK) | set(stock_list(fresh, g["date"].min()))
        assert set(g["instrument"]) <= allowed                                                                # U-6: scored rows are the tradeable list only
    first_cutoff = sorted(p.name for p in (fresh.path("model_store") / "t_abs").iterdir())[0]
    record = json.loads((fresh.path("model_store") / "t_abs" / first_cutoff / "record.json").read_text(encoding="utf-8"))
    all_rows = outcomes.read(fresh, "next_open", first_cutoff, 5)
    assert record["train_rows"] > 0.9 * len(all_rows) > 3 * (S.TOP_N + 2) * 50                                # ... but training used every F&O stock
    assert pd.Timestamp(record["train_last_window_end"]) <= pd.Timestamp(record["cutoff"])                    # M-4
    assert record["seed"] == 7 and record["features"] == ["rv_5", "rv_20", "atr_pct_14", "range_pct", "ret_5d", "iv_rv_ratio"]   # M-7
    train("t_abs", settings=fresh, start=S.DAYS[120])
    again = pd.read_parquet(fresh.path("pit") / "predictions" / "t_abs.parquet")
    pd.testing.assert_frame_equal(preds, again)                                                               # M-7: same record, same predictions
    assert [e for e in records.read_log(fresh) if e["kind"] == "train"][0]["feature_count"] == 6              # F-10: the count is on record


def test_saved_model_reproduces_its_predictions(fresh):
    from koscine4.features import build as fb

    _write_model(fresh, "t_abs")
    train("t_abs", settings=fresh, start=S.DAYS[120])
    preds = pd.read_parquet(fresh.path("pit") / "predictions" / "t_abs.parquet")
    last_cutoff = preds["cutoff"].max()
    month = preds[preds["cutoff"] == last_cutoff]
    model, record = load_latest_model(fresh.path("model_store"), "t_abs", last_cutoff)
    feats = fb.read(record["features"], settings=fresh).merge(month[["date", "instrument"]], on=["date", "instrument"])
    got = pd.Series(model.predict(feats[record["features"]]), index=pd.MultiIndex.from_frame(feats[["date", "instrument"]]))
    want = month.set_index(["date", "instrument"])["prediction"]
    assert np.allclose(got.reindex(want.index).to_numpy(), want.to_numpy())
    a = make_model(ModelConfig("g", "regression", "gbt", ("rv_5", "rv_20"), params={"n_estimators": 20, "min_child_samples": 5}))
    b = make_model(ModelConfig("g", "regression", "gbt", ("rv_5", "rv_20"), params={"n_estimators": 20, "min_child_samples": 5}))
    X = feats[["rv_5", "rv_20"]].fillna(0.0)
    y = feats["rv_5"].fillna(0.0) * 2 + 1
    assert np.array_equal(a.fit(X, y).predict(X), b.fit(X, y).predict(X))                                     # trees are deterministic too


def test_a_model_learns_the_structure_result_and_drives_a_strategy(fresh):
    base = dict(book="stock", structure="long_straddle", decision_time="evening", entry_time="next_open", expiry={"min_dte": 7},
                strikes={"max_snap_pct": 2.5}, exit={"max_hold_sessions": 5, "safety_days": 3},
                liquidity={"min_volume": 50, "min_open_interest": 1, "min_premium": 5.0})
    take_all = from_dict(dict(base, name="t_take_all", family="t_straddle", score={"source": "take_all", "threshold": 0.0},
                              selection={"weekly_cap": False, "allow_overlap": True, "save_as_outcome": True}))
    out = run(take_all, S.DAYS[30], None, settings=fresh, unlock_sealed=True, reason="building outcomes for a test")       # M-1 baseline
    per_day = out["trades"].groupby(["instrument", "decision_date"]).size()
    assert per_day.max() == 1 and set(out["trades"]["instrument"]) == {"AAA", "BBB"}                          # HHH and GGG have no quotes stored
    assert out["report"]["independent_bets"] < out["report"]["trades"] / 3                                    # overlapping trades count once (V-10)
    assert "candidate:no_option_data" in out["report"]["skipped"]["by_reason"]
    stored = outcomes.read_structure(fresh, "t_take_all", S.DAYS[200])
    assert stored["window_end"].max() <= S.DAYS[200] and len(stored) > 150                                    # T-1, T-3
    with pytest.raises(outcomes.OutcomeError):
        outcomes.read_structure(fresh, "t_take_all", None)

    _write_model(fresh, "t_edge", algorithm="gbt", min_rows=100,
                 target='table = "structure"\nstrategy = "t_take_all"\ncolumn = "result_pct"\nhorizon = 5')
    report = train("t_edge", settings=fresh, start=S.DAYS[150])
    assert report["rows"] > 50
    driven = from_dict(dict(base, name="t_model", family="t_straddle", score={"source": "model", "model": "t_edge", "threshold": -1e9},
                            selection={"weekly_cap": True, "max_per_day": 1}))
    res = run(driven, S.DAYS[150], None, settings=fresh)
    preds = pd.read_parquet(fresh.path("pit") / "predictions" / "t_edge.parquet").set_index(["date", "instrument"])["prediction"]
    t = res["trades"]
    assert len(t) > 5 and all(t["score"].iloc[i] == pytest.approx(preds.at[(t["decision_date"].iloc[i], t["instrument"].iloc[i])]) for i in range(len(t)))   # S-5
    picks = t.groupby("decision_date").size()
    assert picks.max() == 1 and res["meta"]["variants_tried"] == 2                                            # both runs are variants of one family
    week = t.groupby([t["entry_date"].dt.isocalendar().year, t["entry_date"].dt.isocalendar().week]).size()
    assert week.max() <= 5                                                                                    # S-7 inside the harness


def test_direction_research_compares_the_four_models_by_period(fresh):
    report = research("NIFTY", 1, settings=fresh, start=S.DAYS[150])
    assert list(report["models"]) == ["always_up", "linear", "trees", "trees_recent"]                         # DR-3
    for m in report["models"].values():
        assert m["all"]["rows"] > 50 and len(m["by_period"]) >= 1 and "of" in m["years_better_than_chance"]   # DR-5
    assert 30 < report["models"]["linear"]["all"]["accuracy_pct"] < 70                                        # a random walk: nothing to find
    from koscine4.features.registry import REGISTRY
    assert all(n in REGISTRY for n in report["features"]) and len(report["features"]) <= 40                   # DR-4
    md = (fresh.path("records") / "research" / "direction_NIFTY_1d.md").read_text(encoding="utf-8")
    assert "overnight gap before entry is excluded" in md                                                     # DR-2
    assert (fresh.path("pit") / "predictions" / "direction_NIFTY_1d_trees.parquet").exists()                  # DR-6: testable through the harness
