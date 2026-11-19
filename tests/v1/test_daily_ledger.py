"""The daily run and the ledger (O-1 to O-7, L-1 to L-5, R-7, S-8, S-11, M-8)."""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import synthetic as S
from koscine4.api.routes import trade_list
from koscine4.daily.run import DailyRunError, run as daily_run, to_markdown
from koscine4.harness import records
from koscine4.harness.engine import price_trades
from koscine4.harness.gate import load_trades
from koscine4.harness.run import run as harness_run
from koscine4.ledger.book import Ledger, LedgerError, feature_drift
from koscine4.settings import load_settings, project_root
from koscine4.strategies.spec import load_spec

SPEC = '''
name = "t_nifty_straddle"
family = "t_nifty_straddle"
book = "index"
instruments = ["NIFTY"]
structure = "long_straddle"
decision_time = "evening"
entry_time = "next_open"
[expiry]
kind = "monthly"
min_dte = 7
[strikes]
max_snap_pct = 1.0
[exit]
max_hold_sessions = 5
safety_days = 2
[liquidity]
min_volume = 100
min_open_interest = 1
min_premium = 5.0
[score]
source = "take_all"
threshold = 0.0
[selection]
weekly_cap = true
max_per_day = 1
'''
RISK = '''
capital = 10000000
risk_per_trade_pct = 1.0
max_open_risk_pct = 3.0
drawdown_halve_pct = 5.0
drawdown_pause_pct = 10.0
expiry_safety_days = 2
index_book_trades_per_week = 3
stock_book_trades_per_week = 2
max_correlated_trades = 2
'''


def _with_strategy(settings, gate: int | None = 2, risk: bool = True):
    """Add a strategy file, optionally a gate record for it, and risk settings. Returns reloaded settings."""
    (settings.root / "configs" / "strategies" / "t_nifty_straddle.toml").write_text(SPEC, encoding="utf-8")
    if risk:
        path = settings.root / "configs" / "settings.toml"
        path.write_text(path.read_text(encoding="utf-8") + RISK, encoding="utf-8")          # [risk] is the last section
    settings = load_settings(root=settings.root)
    if gate is not None:
        out = harness_run("t_nifty_straddle", S.DAYS[100], S.DAYS[250], settings=settings)
        for g in range(2, gate + 1):
            records.record_gate(settings, "t_nifty_straddle", g, "pass", [out["run_id"]], note="test")
    return settings


# ---- the daily run ----------------------------------------------------------------------------------------------
def test_a_live_list_is_refused_while_any_risk_setting_is_missing(fresh):
    settings = _with_strategy(fresh, gate=3, risk=False)
    with pytest.raises(DailyRunError, match="Risk settings"):
        daily_run(S.DAYS[-1], "evening", settings=settings, write=False)                    # R-7


def test_only_a_gated_strategy_reaches_the_list(fresh):
    settings = _with_strategy(fresh, gate=None)
    out = daily_run(S.DAYS[-1], "evening", settings=settings, paper=True)
    assert out["no_trade_today"] and "No new trade today" in out["message"]                 # O-4
    assert out["strategies_live"] == [] and out["refusals"][0] == {"strategy": "t_nifty_straddle", "reason": "no_gate_2_record", "stage": "strategy"}    # S-11
    assert "Strategies not allowed to trade" in to_markdown(out)
    records.record_gate(settings, "t_nifty_straddle", 2, "pass", [harness_run("t_nifty_straddle", S.DAYS[100], S.DAYS[200], settings=settings)["run_id"]])
    paper = daily_run(S.DAYS[-1], "evening", settings=settings, paper=True, write=False)
    assert paper["strategies_live"] == ["t_nifty_straddle"] and len(paper["new_trades"]) == 1
    live = daily_run(S.DAYS[-1], "evening", settings=settings, paper=False, write=False)
    assert live["no_trade_today"] and live["refusals"][0]["reason"] == "paper_trading_gate_3_not_passed_for_a_live_list"     # gate 3 before real money


def test_the_trade_list_has_every_field_and_is_stored(fresh):
    settings = _with_strategy(fresh)
    out = daily_run(S.DAYS[-1], "evening", settings=settings, paper=True)
    assert out["steps"] == ["data_checked", "features_built", "scored", "selected", "sized"] and out["data_up_to"] == str(S.DAYS[-1].date())     # O-1
    t = out["new_trades"][0]
    for key in ("instrument", "structure", "legs", "lots", "entry", "exit_rules", "max_loss_rupees", "max_loss_per_lot",
                "expected_result_pct_of_max_loss", "score", "reasons", "ledger_id", "margin_estimate_rupees"):
        assert key in t                                                                     # O-2
    assert t["instrument"] == "NIFTY" and t["structure"] == "long_straddle" and len(t["legs"]) == 2
    assert {l["action"] for l in t["legs"]} == {"BUY"} and {l["type"] for l in t["legs"]} == {"CE", "PE"}
    assert all({"expiry", "strike", "reference_price"} <= set(l) for l in t["legs"])
    per_lot = sum(l["reference_price"] for l in t["legs"]) * 50
    assert t["max_loss_per_lot"] == pytest.approx(per_lot, abs=1) and t["lots"] == int(100_000 // per_lot) >= 1               # R-1, R-2
    assert t["entry"]["window"] == "09:15 to 09:30" and t["entry"]["net_price_limit_per_share"] > 0 and t["exit_rules"]["max_hold_sessions"] == 5
    assert t["entry"]["date"] > out["date"]                                                 # a close signal is entered next session (A5)
    md = (settings.path("records") / "daily" / f"{out['date']}_evening.md").read_text(encoding="utf-8")
    assert "Maximum loss" in md and "BUY" in md and "Exit rules" in md
    assert trade_list(out["date"], "evening", settings) == __import__("json").loads(__import__("json").dumps(out, default=str))        # O-7: the same list
    rec = Ledger(settings).get(t["ledger_id"])
    assert rec["status"] == "recommended" and rec["mode"] == "paper" and rec["lots"] == t["lots"]                             # L-1
    run_id = records.read_gates(settings, "t_nifty_straddle")[-1]["runs"][0]
    gate_trades = price_trades(load_trades(settings, run_id), dict(settings.section("costs")), 1.0)
    assert t["expected_result_inputs"]["source"] == "track_record" and t["expected_result_inputs"]["past_trades"] == len(gate_trades)
    assert t["expected_result_pct_of_max_loss"] == pytest.approx(gate_trades["result_pct"].mean(), abs=0.01)                # S-5
    assert t["margin_estimate_rupees"] == 0 and t["margin_inputs"]["method"].startswith("bought options only")              # R-6
    assert "Expected result after costs" in md and "past trades" in md


def test_a_planted_data_error_stops_the_run_before_scoring(fresh):
    settings = _with_strategy(fresh)
    path = settings.path("silver") / "eod_stock.parquet"
    stock = pd.read_parquet(path)
    stock.loc[stock.index[-3], "close"] = -5.0
    stock.to_parquet(path)
    with pytest.raises(DailyRunError, match="Data checks failed; nothing was scored"):
        daily_run(S.DAYS[-1], "evening", settings=settings, paper=True)                     # O-1
    assert not (settings.path("records") / "daily").exists()
    with pytest.raises(DailyRunError, match="Refresh the data first"):
        daily_run(S.DAYS[-1] + pd.Timedelta(days=30), "evening", settings=_with_strategy(fresh, gate=None, risk=False), paper=True, rebuild_store=False)


def test_stale_intraday_files_stop_a_1430_list_and_are_listed_on_an_evening_one(fresh):
    settings = _with_strategy(fresh, gate=None)
    evening = daily_run(S.DAYS[-1], "evening", settings=settings, paper=True, write=False)
    assert {w["check"] for w in evening["data_warnings"]} == {"intraday_missing"} and len(evening["data_warnings"]) == 4       # D-10
    assert "Data warnings" in to_markdown(evening)
    spec = SPEC.replace('"t_nifty_straddle"', '"t_nifty_pcs"').replace('decision_time = "evening"', 'decision_time = "1430"') \
               .replace('entry_time = "next_open"', 'entry_time = "1515"').replace('structure = "long_straddle"', 'structure = "credit_spread"\nside = "put"') \
               .replace("max_snap_pct = 1.0", "short_otm_pct = 2.0\nwing_pct = 2.0\nmax_snap_pct = 1.0")
    (settings.root / "configs" / "strategies" / "t_nifty_pcs.toml").write_text(spec, encoding="utf-8")
    records.record_gate(settings, "t_nifty_pcs", 2, "pass", [harness_run("t_nifty_pcs", S.DAYS[100], S.DAYS[200], settings=settings)["run_id"]])
    with pytest.raises(DailyRunError, match="Intraday data for NIFTY is not current"):
        daily_run(S.DAYS[-1], "1430", settings=settings, paper=True, write=False, rebuild_store=False)
    folder = settings.path("intraday")
    (folder / "nifty_option_chain_5m").mkdir(parents=True)
    stamp = pd.to_datetime([f"{S.DAYS[-2].date()} 15:15"])
    pd.DataFrame({"timestamp": stamp}).to_parquet(folder / "nifty_option_chain_5m" / f"{(S.DAYS[-1] + pd.Timedelta(days=30)).date()}.parquet")
    pd.DataFrame({"timestamp": stamp, "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0}).to_parquet(folder / "nifty_5m.parquet")
    out = daily_run(S.DAYS[-1], "1430", settings=settings, paper=True, write=False, rebuild_store=False)
    assert out["strategies_live"] == ["t_nifty_pcs"] and {w["instrument"] for w in out["data_warnings"]} == {"BANKNIFTY"}


def test_open_trades_get_todays_exit_instruction_and_block_a_second_trade(fresh):
    settings = _with_strategy(fresh)
    first = daily_run(S.DAYS[-6], "evening", settings=settings, paper=True)
    t = first["new_trades"][0]
    assert t["entry"]["date"] == str(S.DAYS[-5].date())
    ledger = Ledger(settings)
    qty = t["lots"] * t["lot_size"]
    for i, leg in enumerate(t["legs"]):
        ledger.record_fill(t["ledger_id"], "entry", i, leg["reference_price"] * 1.01, qty, f"{S.DAYS[-5].date()} 09:20")    # O-6
    assert ledger.get(t["ledger_id"])["status"] == "open"
    later = daily_run(S.DAYS[-1], "evening", settings=settings, paper=True)
    assert later["open_trades"][0]["id"] == t["ledger_id"] and later["open_trades"][0]["instruction"] == "CLOSE TODAY (time)"    # O-3
    assert later["open_trades"][0]["sessions_held"] == 5 and later["no_trade_today"]
    assert any(r["reason"] == "instrument_already_has_an_open_trade" for r in later["refusals"])                             # S-8 across days
    assert later["best_refused"][0]["instrument"] == "NIFTY"                                                                 # O-4
    mid = daily_run(S.DAYS[-3], "evening", settings=settings, paper=True, write=False)
    assert mid["open_trades"][0]["instruction"] == "hold"


# ---- the ledger -------------------------------------------------------------------------------------------------
def _rec(ledger, strategy="s1", instrument="NIFTY", max_loss=10_000.0, entry="2024-03-05"):
    return ledger.add_recommendation(decision_date="2024-03-04", decision_time="evening", entry_date=entry, strategy=strategy, book="index",
                                     instrument=instrument, legs=[("2024-03-28", 100.0, "PE", -1), ("2024-03-28", 95.0, "PE", 1)], lots=2,
                                     lot_size=50.0, max_loss=max_loss, score=0.4, reasons={"why": "test"}, exit_rules={"max_hold_sessions": 5},
                                     reference_prices=[3.0, 1.0])


def _round_trip(ledger, rec_id, sold_in, bought_in, sold_out, bought_out, exit_date="2024-03-11"):
    ledger.record_fill(rec_id, "entry", 0, sold_in, 100, "2024-03-05 09:20")
    ledger.record_fill(rec_id, "entry", 1, bought_in, 100, "2024-03-05 09:21")
    ledger.record_fill(rec_id, "exit", 0, sold_out, 100, f"{exit_date} 15:15")
    ledger.record_fill(rec_id, "exit", 1, bought_out, 100, f"{exit_date} 15:16")
    return ledger.close_trade(rec_id, exit_date, "time")


def test_fills_results_and_the_harness_twin_are_recorded(fresh):
    ledger = Ledger(fresh)
    rid = _rec(ledger)
    with pytest.raises(LedgerError, match="legs 0 to 1"):
        ledger.record_fill(rid, "entry", 5, 3.0, 100, "x")
    with pytest.raises(LedgerError, match="needs both"):
        ledger.close_trade(rid, "2024-03-11", "time")
    result = _round_trip(ledger, rid, 3.0, 1.0, 1.2, 0.3)
    assert result["pnl"] == pytest.approx((-1) * (1.2 - 3.0) * 100 + (0.3 - 1.0) * 100)       # sold leg gains 180, bought leg loses 70
    assert result["result_pct"] == pytest.approx(1.1)                                         # 110 on a 10,000 maximum loss
    row = ledger.get(rid)
    assert row["status"] == "closed" and row["exit_reason"] == "time" and len(ledger.fills(rid)) == 4      # L-1
    ledger.attach_harness_twin(rid, [3.1, 1.0], [1.1, 0.3], 1.4, "2024-03-11")
    diff = ledger.differences().iloc[0]
    assert diff["harness_result_pct"] == 1.4 and diff["result_difference_pct"] == pytest.approx(-0.3)      # L-2
    skipped = _rec(ledger, instrument="BANKNIFTY")
    ledger.mark_not_taken(skipped, "did not like the price")
    assert ledger.get(skipped)["status"] == "not_taken" and set(ledger.open_trades()["instrument"]) == set()
    assert ledger.started_in_week("2024-03-07") == {"index": 1} and ledger.realised_pnl() == pytest.approx(110.0)
    report = ledger.monthly_report("2024-03", {"s1": {"mean_result_pct": 1.0, "std_result_pct": 2.0}})
    s1 = report["strategies"]["s1"]
    assert report["closed_trades"] == 1 and s1["inside_range"] and s1["backtest_range_pct"] == [-3.0, 5.0]   # L-3
    assert s1["mean_difference_from_harness_pct"] == pytest.approx(-0.3)


def test_a_strategy_is_suspended_on_a_losing_run_until_the_owner_re_enables_it(fresh):
    ledger = Ledger(fresh)
    cfg = dict(fresh.section("ledger"), suspend_after_trades=6)
    for i in range(5):
        _round_trip(ledger, _rec(ledger), 3.0, 1.0, 5.0, 2.0)                                 # each loses 100: -1% of maximum loss
    early = ledger.check_suspension("s1", backtest_mean=2.0, backtest_std=3.0, cfg=cfg, when="2024-03-20")
    assert not early["checked"] and not ledger.is_suspended("s1")                             # not enough trades yet
    _round_trip(ledger, _rec(ledger), 3.0, 1.0, 5.0, 2.0)
    check = ledger.check_suspension("s1", 2.0, 3.0, cfg, "2024-03-21")
    assert check["checked"] and check["live_mean_pct"] == pytest.approx(-1.0) and check["floor_pct"] == pytest.approx(2.0 - 2.0 * 3.0 / np.sqrt(6))
    assert ledger.is_suspended("s1")                                                          # L-4
    assert ledger.check_suspension("s1", -5.0, 3.0, cfg, "2024-03-22")["suspended"]           # stays suspended whatever is measured next
    ledger.set_suspended("s1", False, "re-enabled by owner", "2024-03-25")
    assert not ledger.is_suspended("s1")


def test_a_suspended_strategy_is_left_out_of_the_daily_list(fresh):
    settings = _with_strategy(fresh)
    Ledger(settings).set_suspended("t_nifty_straddle", True, "test", "2023-08-01")
    out = daily_run(S.DAYS[-1], "evening", settings=settings, paper=True, write=False)
    assert out["no_trade_today"] and out["refusals"][0]["reason"] == "suspended_until_the_owner_re_enables_it"


def test_a_live_feature_far_outside_its_history_is_flagged():
    rng = np.random.default_rng(1)
    history = pd.DataFrame({"rv_20": rng.normal(0.2, 0.02, 500), "skew": rng.normal(0.0, 0.01, 500)})
    live = pd.Series({"rv_20": 0.9, "skew": 0.001, "unknown": 5.0})
    flags = feature_drift(live, history, 0.5, 99.5)
    assert [f["feature"] for f in flags] == ["rv_20"] and flags[0]["value"] == 0.9            # L-5


# ---- nothing starts training or a backtest by itself (M-8) ------------------------------------------------------
def test_the_daily_run_and_the_web_routes_never_train_or_backtest():
    pkg = project_root() / "src" / "koscine4"
    banned = {"koscine4.models.train": {"train"}, "koscine4.models.direction": {"research"}, "koscine4.harness.run": {"run"},
              "koscine4.harness.search": {"search"}, "koscine4.harness.engine": {"run_engine"}}
    for path in list((pkg / "daily").glob("*.py")) + list((pkg / "api").glob("*.py")) + list((pkg / "ledger").glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module in banned:
                assert not ({a.name for a in node.names} & banned[node.module]), f"{path.name} imports a training or backtest entry point"
    assert load_spec  # the daily run reads specifications; it does not change them
    for path in pkg.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "schedule." not in text and "crontab" not in text and "BackgroundTasks" not in text, path.name
