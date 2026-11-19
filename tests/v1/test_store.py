"""Phase 1: the point-in-time store (requirements D-1 to D-10)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from koscine4.store import Decision, Store
from koscine4.store.adjustments import apply_overrides, parse_action_ratio
from koscine4.store.availability import AvailabilityError, last_visible_date, source_availability
from koscine4.store.calendar import CalendarError, TradingCalendar
from koscine4.store.checks import (
    DataCheckError, check_duplicates, check_freshness, check_jumps, check_missing_sessions,
    check_price_sanity, check_stale_runs, run_price_checks,
)
from koscine4.store.events import visible_events
from koscine4.store.options import build_lot_sizes, chain_on, infer_lot_sizes
from koscine4.store.prices import adjust_prices
from koscine4.store.tables import ADJUSTMENTS, PRICES_RAW, StoreError, read_table

from conftest import DAYS, EX, true_raw


# ---- calendar ---------------------------------------------------------------------------------
def test_calendar_steps_by_sessions_not_calendar_days():
    cal = TradingCalendar.from_dates(DAYS)
    friday, monday = DAYS[4], DAYS[5]
    assert cal.shift(friday, 1) == monday
    assert cal.shift(monday, -1) == friday
    assert cal.on_or_before(friday + pd.Timedelta(days=1)) == friday          # a Saturday
    assert not cal.is_session(friday + pd.Timedelta(days=1))
    with pytest.raises(CalendarError):
        cal.shift(DAYS[-1], 1)
    with pytest.raises(CalendarError):
        cal.on_or_before(DAYS[0] - pd.Timedelta(days=1))


# ---- availability (D-3) -------------------------------------------------------------------------
@pytest.mark.parametrize(
    "rule, time, expected_offset",
    [
        ("evening", "evening", 0),         # today's close is readable this evening
        ("evening", "1430", -1),           # ...but not at 14:30 the same day
        ("evening", "preopen", -1),
        ("next_preopen", "evening", -1),   # a source that arrives next morning is not readable tonight
        ("next_preopen", "preopen", -1),   # it is readable before the next open (yesterday's row)
        ("next_preopen", "1430", -1),
        ("next_evening", "evening", -1),
        ("next_evening", "1430", -2),
    ],
)
def test_a_row_is_readable_only_after_its_source_publishes(rule, time, expected_offset):
    cal = TradingCalendar.from_dates(DAYS)
    today = DAYS[20]
    assert last_visible_date(rule, Decision(today, time), cal) == DAYS[20 + expected_offset]


def test_availability_edge_cases(world):
    cal = TradingCalendar.from_dates(DAYS)
    assert last_visible_date("evening", Decision(DAYS[0], "1430"), cal) is None          # nothing published yet
    saturday = DAYS[4] + pd.Timedelta(days=1)
    assert last_visible_date("evening", Decision(saturday, "preopen"), cal) == DAYS[4]   # holiday sees Friday
    with pytest.raises(AvailabilityError):
        Decision(DAYS[0], "noon")
    with pytest.raises(AvailabilityError, match="no entry"):
        source_availability(world.section("availability"), "some_new_feed")


def test_every_store_source_records_when_it_is_available(world):
    for source in ("prices_raw", "option_quotes", "flows_fii_dii", "flows_participant_oi", "events"):
        assert source_availability(world.section("availability"), source) in {"evening", "next_preopen", "next_evening"}


# ---- parsing declared actions --------------------------------------------------------------------
@pytest.mark.parametrize(
    "text, amount, ratio",
    [
        ("Bonus 1:1", None, 2.0),
        ("Bonus 1:2", None, 1.5),
        ("Bonus 2:1", None, 3.0),
        ("Face Value Split From Rs.10/- To Rs.2/-", None, 5.0),
        ("Face Value Split (Sub-Division) - From Rs 10/- Per Share To Rs 5/- Per Share", None, 2.0),
        ("Face Value Split From Rs.10/- To Re.1/-", None, 10.0),
        ("Fv Split Rs.10 To Rs.2", None, 5.0),
        ("SPLIT", 5.0, 5.0),
        ("Dividend - Rs 5 Per Share", 5.0, None),
        ("Annual General Meeting", None, None),
    ],
)
def test_parse_action_ratio(text, amount, ratio):
    assert parse_action_ratio(text, amount) == ratio


# ---- raw prices (D-4, D-7) -------------------------------------------------------------------------
def test_raw_prices_hold_fo_stocks_and_the_indices(built):
    settings, summary = built
    raw = read_table(settings, PRICES_RAW)
    assert set(raw["instrument"]) == {"AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG", "HHH", "NIFTY", "BANKNIFTY", "INDIAVIX"}
    assert "ZZZ" not in set(raw["instrument"])                               # never had derivatives
    assert not raw.duplicated(["instrument", "date"]).any()
    nifty = raw[raw["instrument"] == "NIFTY"]
    assert len(nifty) == len(DAYS) and (nifty["kind"] == "index").all()      # stitched across the name change
    assert summary["sessions"] == len(DAYS)


def test_silver_back_adjustment_is_undone_using_the_exchange_previous_close(built):
    settings, _ = built
    raw = read_table(settings, PRICES_RAW)
    fff = raw[raw["instrument"] == "FFF"].merge(true_raw("FFF"), on="date", suffixes=("", "_true"))
    assert np.allclose(fff["close"], fff["close_true"], rtol=2e-4)             # history is raw again
    before, after = fff[fff["date"] < EX], fff[fff["date"] >= EX]
    # includes the row just before a missing session, which has no next-day ratio of its own
    assert len(before) == 29 and before["restore_factor"].between(1.9, 2.1).all() and (after["restore_factor"] == 1.0).all()
    silver = pd.read_parquet(settings.path("silver") / "eod_stock.parquet")
    for sym in ("AAA", "BBB", "CCC", "DDD", "EEE"):                            # untouched symbols stay byte-identical
        a = raw[raw["instrument"] == sym].reset_index(drop=True)
        b = silver[silver["symbol"] == sym].reset_index(drop=True)
        assert (a["restore_factor"] == 1.0).all() and (a["close"] == b["close"]).all() and (a["open"] == b["open"]).all()


# ---- adjustments (D-1) -----------------------------------------------------------------------------
def test_each_kind_of_event_is_recognised(built):
    settings, _ = built
    adj = read_table(settings, ADJUSTMENTS).set_index("instrument")
    assert adj.loc["AAA", "kind"] == "split_or_bonus" and adj.loc["AAA", "factor"] == 0.5 and adj.loc["AAA", "applied"]
    assert adj.loc["AAA", "source"] == "declared:vendor" and adj.loc["AAA", "ex_date"] == EX
    assert adj.loc["BBB", "factor"] == pytest.approx(0.2) and adj.loc["BBB", "source"] == "declared:silver"
    assert adj.loc["CCC", "kind"] == "inferred_other" and adj.loc["CCC", "applied"]
    assert adj.loc["CCC", "factor"] == pytest.approx(0.6, rel=0.01)
    assert adj.loc["DDD", "kind"] == "unexplained_gap" and not adj.loc["DDD", "applied"]      # penny stock: listed only
    assert adj.loc["EEE", "kind"] == "declared_unconfirmed" and not adj.loc["EEE", "applied"]
    # FFF: declared twice in silver (duplicate rows), confirmed once against the restored raw price
    assert adj.loc["FFF", "kind"] == "split_or_bonus" and adj.loc["FFF", "factor"] == 0.5 and adj.loc["FFF", "source"] == "declared:silver"
    # GGG: the demerger fell inside 8 missing sessions, so it is inferred across the break
    assert adj.loc["GGG", "kind"] == "inferred_across_break" and adj.loc["GGG", "applied"]
    assert adj.loc["GGG", "factor"] == pytest.approx(0.6, rel=0.03) and adj.loc["GGG", "ex_date"] == DAYS[38]
    # HHH: a rise across a long break is listed, never applied
    assert adj.loc["HHH", "kind"] == "unexplained_gap" and not adj.loc["HHH", "applied"]
    assert adj.index.is_unique
    review = pd.read_csv(settings.path("records") / "store" / "adjustments_review.csv")
    assert set(review["instrument"]) == {"CCC", "DDD", "EEE", "GGG", "HHH"}   # confirmed declared events need no review


def test_adjusted_returns_are_ordinary_across_the_ex_date(built):
    settings, _ = built
    store = Store(settings)
    p = store.prices(["AAA", "BBB", "CCC", "FFF", "GGG"], asof=Decision(DAYS[-1]))
    ret = (p["close"] / p["prev_close"] - 1).abs()
    assert ret.max() < 0.03
    raw = store.prices(["AAA", "BBB", "CCC"], asof=Decision(DAYS[-1]), adjusted=False)
    raw_ret = raw.groupby("instrument")["close"].pct_change().abs()
    assert raw_ret.max() > 0.39                                              # the raw series still shows the gaps


def test_adjustment_uses_no_future_event(built):
    settings, _ = built
    store = Store(settings)
    before = store.prices("AAA", asof=Decision(DAYS[29]))                    # the evening before the ex-date
    raw = store.prices("AAA", asof=Decision(DAYS[29]), adjusted=False)
    assert np.allclose(before["close"], raw["close"]) and (before["adj_factor"] == 1.0).all()
    assert before["date"].max() == DAYS[29]
    after = store.prices("AAA", asof=Decision(EX))                           # the evening of the ex-date
    assert np.allclose(after.loc[after["date"] < EX, "close"], 0.5 * raw["close"])
    # returns before the event are the same whichever day you ask from
    r_before = before["close"].pct_change().dropna().to_numpy()
    r_after = after.loc[after["date"] < EX, "close"].pct_change().dropna().to_numpy()
    assert np.allclose(r_before, r_after)


def test_store_built_from_truncated_data_matches_the_as_of_read(tmp_path, built):
    """Deleting everything after day X and rebuilding gives what the full store returns as of X."""
    from conftest import write_world
    from koscine4.settings import load_settings
    from koscine4.store.build import build_store

    settings, _ = built
    cut = DAYS[35]
    full = Store(settings).prices(asof=Decision(cut))

    root2 = tmp_path / "truncated"; root2.mkdir()
    write_world(root2)
    for name in ("eod_stock", "indices"):
        f = root2 / "data" / "silver" / f"{name}.parquet"
        df = pd.read_parquet(f); df[df["date"] <= cut].to_parquet(f)
    s2 = load_settings(root=root2)
    build_store(s2, raise_on_error=False)
    part = Store(s2).prices(asof=Decision(cut))
    cols = ["instrument", "date", "open", "high", "low", "close", "volume", "adj_factor"]
    pd.testing.assert_frame_equal(full[cols], part[cols])


def test_raw_table_is_never_rewritten_by_reads(built):
    settings, _ = built
    before = read_table(settings, PRICES_RAW)
    Store(settings).prices(asof=Decision(DAYS[-1]))
    pd.testing.assert_frame_equal(before, read_table(settings, PRICES_RAW))


def test_owner_overrides_reject_and_add(built):
    settings, _ = built
    adj = read_table(settings, ADJUSTMENTS)
    out = apply_overrides(adj, {
        "reject": [{"instrument": "CCC", "ex_date": EX, "reason": "genuine fall"}],
        "add": [{"instrument": "EEE", "ex_date": DAYS[50], "factor": 0.8, "kind": "demerger", "reason": "spin-off"}],
    })
    ccc = out[out["instrument"] == "CCC"].iloc[0]
    assert not ccc["applied"] and "rejected by owner" in ccc["note"]
    added = out[(out["instrument"] == "EEE") & (out["ex_date"] == DAYS[50])].iloc[0]
    assert added["applied"] and added["factor"] == 0.8 and added["source"] == "owner"
    raw = read_table(settings, PRICES_RAW)
    adjusted = adjust_prices(raw[raw["instrument"] == "CCC"], out, DAYS[-1])
    assert (adjusted["adj_factor"] == 1.0).all()                             # rejected event no longer applied


# ---- reads respect availability (D-3) -----------------------------------------------------------------
def test_price_reads_stop_at_what_the_decision_can_see(built):
    settings, _ = built
    store = Store(settings)
    today = DAYS[40]
    assert store.prices("BBB", asof=Decision(today, "evening"))["date"].max() == today
    assert store.prices("BBB", asof=Decision(today, "1430"))["date"].max() == DAYS[39]
    assert store.prices("BBB", asof=Decision(today, "preopen"))["date"].max() == DAYS[39]
    assert store.prices("BBB", asof=Decision(DAYS[0], "1430")).empty
    with pytest.raises(TypeError):
        store.prices("BBB")                                                  # a read without a decision is not allowed


def test_flow_reads_follow_the_availability_setting(built, tmp_path):
    settings, _ = built
    store = Store(settings)
    today = DAYS[40]
    assert store.fii_dii(asof=Decision(today, "evening"))["date"].max() == today      # owner: published same evening
    assert store.fii_dii(asof=Decision(today, "1430"))["date"].max() == DAYS[39]
    assert store.participant_oi(asof=Decision(today, "evening"))["date"].max() == today

    # the documented D-3 case: a source that arrives the next morning cannot be read the same evening
    text = (settings.root / "configs" / "settings.toml").read_text(encoding="utf-8")
    (settings.root / "configs" / "settings.toml").write_text(
        text.replace('flows_fii_dii = "evening"', 'flows_fii_dii = "next_preopen"'), encoding="utf-8")
    from koscine4.settings import load_settings
    late = Store(load_settings(root=settings.root))
    assert late.fii_dii(asof=Decision(today, "evening"))["date"].max() == DAYS[39]
    assert late.fii_dii(asof=Decision(DAYS[41], "preopen"))["date"].max() == today


# ---- events (D-9) ---------------------------------------------------------------------------------------
def test_result_date_is_hidden_until_announced(built):
    settings, _ = built
    store = Store(settings)
    assert store.events(asof=Decision(DAYS[44])).empty                       # no announcement date: unknown the day before
    assert len(store.events(asof=Decision(DAYS[45]))) == 1                   # visible on the day itself
    ev = pd.DataFrame({"instrument": ["AAA"], "event_date": [DAYS[45]], "kind": ["results"], "announced_on": [DAYS[38]]})
    assert visible_events(ev, DAYS[37]).empty
    assert len(visible_events(ev, DAYS[38])) == 1                            # visible from the announcement


# ---- data checks (D-8, D-10) ------------------------------------------------------------------------------
def test_clean_data_has_no_errors_and_lists_the_unexplained_gap(built):
    settings, summary = built
    assert summary["check_errors"] == 0
    issues = pd.read_csv(settings.path("records") / "store" / "check_issues.csv")
    big = issues[issues["check"] == "large_move"]
    assert set(big["instrument"]) == {"DDD", "HHH"}                          # the adjusted ones no longer show
    assert "break in the data" in big[big["instrument"] == "HHH"]["detail"].iloc[0]
    assert "in one session" in big[big["instrument"] == "DDD"]["detail"].iloc[0]
    missing = issues[issues["check"] == "missing_sessions"]
    assert set(missing["instrument"]) == {"FFF", "GGG", "HHH"} and (missing["severity"] == "review").all()


def test_planted_errors_are_caught(built):
    settings, _ = built
    raw = read_table(settings, PRICES_RAW)
    adj = read_table(settings, ADJUSTMENTS)
    cal = TradingCalendar.from_prices(raw, 3)
    cfg = settings.section("store")

    dup = pd.concat([raw, raw.iloc[[5]]], ignore_index=True)
    assert [i.check for i in check_duplicates(dup)] == ["duplicate_row"]
    with pytest.raises(DataCheckError, match="duplicate_row"):
        run_price_checks(dup, adj, cal, cfg)

    neg = raw.copy(); neg.loc[10, "low"] = -1.0
    assert "non_positive_price" in {i.check for i in check_price_sanity(neg)}
    flipped = raw.copy(); flipped.loc[10, "high"] = flipped.loc[10, "low"] * 0.5
    assert "ohlc_out_of_order" in {i.check for i in check_price_sanity(flipped)}

    gap = raw[~((raw["instrument"] == "NIFTY") & (raw["date"] == DAYS[25]))]
    found = {(i.instrument, i.severity) for i in check_missing_sessions(gap, cal)}
    assert ("NIFTY", "error") in found                                       # a recent index hole stops the refresh
    old = {(i.instrument, i.severity) for i in check_missing_sessions(gap, cal, recent_sessions=10)}
    assert ("NIFTY", "review") in old and ("NIFTY", "error") not in old      # an old one is listed
    with pytest.raises(DataCheckError, match="missing_sessions"):
        run_price_checks(gap, adj, cal, cfg)
    stock_gap = raw[~((raw["instrument"] == "BBB") & (raw["date"] == DAYS[25]))]
    assert ("BBB", "review") in {(i.instrument, i.severity) for i in check_missing_sessions(stock_gap, cal)}

    stale = raw.copy()
    sel = (stale["instrument"] == "EEE") & (stale["date"] >= DAYS[10]) & (stale["date"] <= DAYS[17])
    stale.loc[sel, ["open", "high", "low", "close"]] = 300.0
    stale.loc[sel, "volume"] = 0.0
    assert [i.instrument for i in check_stale_runs(stale, int(cfg["stale_run_sessions"]))] == ["EEE"]

    unadjusted = check_jumps(raw, adj.assign(applied=False), float(cfg["jump_review_pct"]))
    assert {"AAA", "BBB", "CCC"} <= {i.instrument for i in unadjusted}       # without adjustment the splits show up


def test_intraday_freshness(tmp_path):
    folder = tmp_path / "nifty_option_chain_5m"; folder.mkdir()
    assert check_freshness(folder, "2026-09-01", 7)[0].check == "intraday_missing"
    (folder / "2026-08-25.parquet").write_bytes(b"")
    assert check_freshness(folder, "2026-08-28", 7) == []
    assert check_freshness(folder, "2026-09-10", 7)[0].check == "intraday_stale"
    chain = tmp_path / "chain"; chain.mkdir()                                  # option-chain files are named by expiry
    pd.DataFrame({"timestamp": pd.to_datetime(["2026-07-01 15:15", "2026-07-02 15:25"])}).to_parquet(chain / "2026-12-31.parquet")
    stale = check_freshness(chain, "2026-09-01", 7)
    assert stale[0].check == "intraday_stale" and "2026-07-02" in stale[0].detail   # a far expiry's name is not its data's date
    bars = tmp_path / "nifty_5m.parquet"
    pd.DataFrame({"timestamp": pd.to_datetime(["2026-08-31 15:25"]).tz_localize("Asia/Kolkata")}).to_parquet(bars)
    assert check_freshness(bars, "2026-09-01", 7) == [] and check_freshness(tmp_path / "none.parquet", "2026-09-01", 7)[0].check == "intraday_missing"


# ---- option quotes and lot sizes (D-5) -----------------------------------------------------------------------
def _write_contracts(settings):
    exp1, exp2 = pd.Timestamp("2024-02-29"), pd.Timestamp("2024-03-28")
    rows = []
    for d in DAYS[:40]:
        for exp, lot in ((exp1, 250), (exp2, 500)):
            for strike, mult in ((900.0, 3), (1000.0, 8), (1100.0, 5)):
                for ot in ("CE", "PE"):
                    rows.append({"date": d, "symbol": "AAA", "instrument": "OPTSTK", "expiry": exp, "strike": strike,
                                 "opt_type": ot, "open": 10.0, "high": 12.0, "low": 9.0, "close": 11.0, "settle": 11.0,
                                 "contracts": 100, "open_int": lot * mult})
    folder = settings.path("silver") / "eod_deriv_contracts" / "year=2024"
    folder.mkdir(parents=True)
    pd.DataFrame(rows).to_parquet(folder / "data.parquet")
    pd.DataFrame({"symbol": ["AAA"], "expiry_month": [pd.Timestamp("2024-03-01")], "lot": [500], "source": ["udiff"]}
                 ).to_parquet(settings.path("silver") / "lot_size.parquet")
    return exp1, exp2


def test_lot_size_is_inferred_from_open_interest_and_exchange_value_wins(built):
    settings, _ = built
    exp1, exp2 = _write_contracts(settings)
    inferred = infer_lot_sizes(pd.DataFrame({"symbol": ["X"] * 3, "expiry": [exp1] * 3, "open_int": [750, 2000, 0]}))
    assert inferred["lot"].tolist() == [250]
    lots = build_lot_sizes(settings).set_index("expiry")
    assert lots.loc[exp1, "lot"] == 250 and lots.loc[exp1, "source"] == "inferred_from_open_interest"
    assert lots.loc[exp2, "lot"] == 500 and lots.loc[exp2, "source"] == "exchange"
    assert lots["lot"].notna().all()                                         # D-5: every contract has a lot size


def test_option_chain_is_the_last_day_the_decision_can_see(built):
    settings, _ = built
    _write_contracts(settings)
    build_lot_sizes(settings)
    store = Store(settings)
    chain = store.option_chain("AAA", asof=Decision(DAYS[20], "1430"))
    assert chain["date"].unique().tolist() == [DAYS[19]]                     # yesterday's chain at 14:30
    assert len(chain) == 12 and chain["lot"].notna().all()
    assert {"instrument", "contract_type", "expiry", "strike", "opt_type", "close", "volume", "open_int", "lot"} <= set(chain.columns)
    assert chain_on(settings, "AAA", DAYS[59]).empty                         # no contracts that day


def test_missing_table_gives_a_clear_error(world):
    with pytest.raises(StoreError, match="has not been built"):
        Store(world).prices(asof=Decision(DAYS[5]))
