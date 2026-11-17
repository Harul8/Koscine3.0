"""Exchange report files, symbol changes, market value and the monthly list (D-7, D-9, D-11, U-1 to U-6)."""
from __future__ import annotations

import io
import zipfile

import numpy as np
import pandas as pd
import pytest

import synthetic as S
from koscine4.sources.exchange_reports import parse_bundle, read_symbol_changes
from koscine4.store import Decision, Store
from koscine4.store.adjustments import named_kind, parse_action, reconcile
from koscine4.store.market_value import share_multiplier, shares_for
from koscine4.store.options import chain_on
from koscine4.store.symbols import canonicalize, names_over_time, symbol_on
from koscine4.store.tables import read_table
from koscine4.universe.build import (
    INDEX_BOOK, UniverseError, build_month, build_universe, fo_members, option_activity, read_all, read_month,
    rebuild_dates, stock_list, tradeable, universe_dir,
)


def _zip(members: dict[str, str]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, text in members.items():
            zf.writestr(name, text)
    return buf.getvalue()


# ---- exchange report bundle -------------------------------------------------------------------------------------
def test_bundle_in_the_layout_used_until_october_2025():
    content = _zip({
        "Bc040110.csv": "SERIES,SYMBOL,SECURITY,RECORD_DT,BC_STRT_DT,BC_END_DT,EX_DT,ND_STRT_DT,ND_END_DT,PURPOSE\n"
                        "EQ,LITL,Lanco Infratech Limited,05/01/2010, , ,04/01/2010, , ,FV SPLIT RS.10 TO RE.1\n",
        "Bm040110.txt": "COMPANY NAME    SYMBOL     : BM DATE    : BM PURPOSE \n"
                        "Axis Bank Limited    AXISBANK   : 15-JAN-2010 : Unaudited Quarterly Results To consider\n"
                        "the results for the quarter.\n",
        "Pd040110.csv": "MKT,SERIES,SYMBOL,SECURITY,PREV_CL_PR,OPEN_PRICE,HIGH_PRICE,LOW_PRICE,CLOSE_PRICE,NET_TRDVAL,NET_TRDQTY,IND_SEC,CORP_IND,TRADES,HI_52_WK,LO_52_WK\n"
                        "Y, , ,S&P CNX Nifty,5201.05,5200.9,5238.45,5167.1,5232.2,1.0,2.0,Y, ,3.0,1,1\n"
                        "Y, , ,CNX IT,5818.4,5819.5,5889.2,5819.5,5877.25,0,0,Y, ,0,1,1\n"
                        "N,BE,PVR,PVR Ltd,185.4,176.3,191.0,176.3,186.95,1745100.7,9404,N, ,255,1,1\n"
                        "N,EQ,ACC,ACC Ltd,900,901,902,899,900.5,10,10,N, ,5,1,1\n",
        "Ix040110.csv": "INDEX_FLG,SYMBOL,SERIES,SECURITY,ISSUE_CAP,CLOSE_PRIC,MKT_CAP,WEIGHTAGE\n , , ,BANK Nifty, , , , \n"
                        "BANK Nifty,AXISBANK,EQ,AXIS BANK LIMITED,402394234,       992.10,399215319551.40,8\n",
    })
    out = parse_bundle(content, "2010-01-04")
    ca = out["corp_actions"].iloc[0]
    assert ca["symbol"] == "LITL" and ca["ex_date"] == pd.Timestamp("2010-01-04") and "SPLIT" in ca["purpose"]
    bm = out["board_meetings"].iloc[0]
    assert bm["symbol"] == "AXISBANK" and bm["meeting_date"] == pd.Timestamp("2010-01-15") and bm["file_date"] == pd.Timestamp("2010-01-04")
    assert bm["purpose"].endswith("the results for the quarter.")                      # a notice that runs over two lines
    assert out["index_prices"]["instrument"].tolist() == ["NIFTY"] and out["index_prices"]["close"].iloc[0] == 5232.2
    assert out["other_series"]["instrument"].tolist() == ["PVR"]                       # main-series rows are not repeated
    sc = out["share_counts"].iloc[0]
    assert sc["symbol"] == "AXISBANK" and sc["shares"] == 402394234 and sc["source"] == "ix"


def test_bundle_in_the_layout_used_since_october_2025():
    content = _zip({
        "bc13102025.csv": "SERIES,SYMBOL,SECURITY,RECORD_DT,BC_STRT_DT,BC_END_DT,EX_DT,ND_STRT_DT,ND_END_DT,PURPOSE\n"
                          "EQ,ABC,Abc Ltd,2025-10-17,,,2025-10-17,,,BONUS 1:1\n",
        "bm13102025.txt": "COMPANY NAME    SYMBOL     : BM DATE    : BM PURPOSE\n"
                          "Tanla Platforms Limited TANLA : 17-OCT-2025 : Financial Results/Dividend To consider\n",
        "mcap13102025.csv": "Trade Date,Symbol,Series,Security Name,Category,Last Trade Date,Face Value(Rs.),Issue Size,Close Price/Paid up value(Rs.),Market Cap(Rs.)\n"
                            "13 OCT 2025,TANLA,EQ,TANLA PLATFORMS,Listed,13 OCT 2025,1.00,134000000,650.00,87100000000.00\n",
    })
    out = parse_bundle(content, "2025-10-13", symbols={"TANLA", "ABC"})
    assert out["corp_actions"]["ex_date"].iloc[0] == pd.Timestamp("2025-10-17")
    assert out["board_meetings"]["symbol"].tolist() == ["TANLA"]
    assert out["share_counts"]["shares"].iloc[0] == 134000000 and out["share_counts"]["source"].iloc[0] == "mcap"


@pytest.mark.parametrize("text, ratio, what", [
    ("FV SPL-RS10TORS5/BON-2:1", 6.0, "both"),
    ("BON 1:1/SPLIT RS.10TORS.5", 4.0, "both"),
    ("AGM/FV SPLT FRM 10 TO 1", 10.0, "split"),
    ("BONUS 1:10", 1.1, "bonus"),
    ("SPL DIV RS 500 PER SHARE", None, ""),              # a special dividend is not a split
    ("INTDIV - RS 2.50 PER SH", None, ""),
])
def test_exchange_shorthand_is_read(text, ratio, what):
    got, kind = parse_action(text)
    assert kind == what and (got is None if ratio is None else got == pytest.approx(ratio))


def test_named_events_are_labelled_and_take_their_factor_from_the_gap(big):
    assert named_kind("DEMERGER") == "demerger_declared" and named_kind("RIGHTS 7:6@PREMIUM RS 18") == "rights_declared"
    assert named_kind("SPL DIV RS 500 PER SHARE") == "special_dividend_declared" and named_kind("DIVIDEND RS 2") is None
    cfg = big.section("store")
    d = pd.Timestamp("2024-03-04")
    gaps = pd.DataFrame({"instrument": ["X", "Y", "Z", "W"], "date": [d] * 4, "gap": [0.6, 0.93, 0.99, 0.59], "break_gap": [1.0] * 4,
                         "last_close": [500.0] * 4, "held_close": [500.0] * 4, "days_since": [1.0] * 4})
    named = pd.DataFrame({"instrument": ["X", "Y", "Z"], "ex_date": [d] * 3, "purpose": ["DEMERGER", "RIGHTS 1:5", "SCHEME OF ARRANGEMENT"]})
    declared = pd.DataFrame({"instrument": ["W"], "ex_date": [d], "ratio": [2.0], "source": ["exchange"], "what": ["bonus"]})
    out = reconcile(declared, gaps, cfg, named).set_index("instrument")
    assert out.loc["X", "kind"] == "demerger_declared" and out.loc["X", "applied"] and out.loc["X", "factor"] == 0.6
    assert out.loc["Y", "kind"] == "rights_declared" and out.loc["Y", "applied"]             # 7% below: adjusted
    assert out.loc["Z", "kind"] == "declared_no_gap" and not out.loc["Z", "applied"]         # 1% below: left alone
    # a declared 1:1 bonus with an 18% move on the day: the declared ratio is used, not the gap
    assert out.loc["W", "kind"] == "split_or_bonus" and out.loc["W", "factor"] == 0.5 and "differs" in out.loc["W", "note"]


# ---- symbol changes (D-7) ---------------------------------------------------------------------------------------
def test_symbol_changes_follow_chains_and_respect_dates():
    changes = pd.DataFrame({"old": ["SESAGOA", "SSLT", "XYZ"], "new": ["SSLT", "VEDL", "NEWXYZ"],
                            "date": pd.to_datetime(["2013-10-04", "2015-05-07", "2012-01-01"])})
    sym = pd.Series(["SESAGOA", "SSLT", "VEDL", "XYZ", "XYZ"])
    dates = pd.Series(pd.to_datetime(["2012-01-02", "2014-01-02", "2016-01-04", "2011-06-01", "2020-06-01"]))
    # the last row is a different company that took the symbol XYZ after the first one gave it up
    assert canonicalize(sym, dates, changes).tolist() == ["VEDL", "VEDL", "VEDL", "NEWXYZ", "XYZ"]
    timeline = names_over_time(changes)
    assert [symbol_on("VEDL", d, timeline) for d in ("2012-01-02", "2014-01-02", "2016-01-04")] == ["SESAGOA", "SSLT", "VEDL"]


def test_a_renamed_company_is_one_instrument_everywhere(big):
    assert len(read_symbol_changes(big)) == 1
    raw = read_table(big, "prices_raw")
    assert "AAAOLD" not in set(raw["instrument"]) and (raw["instrument"] == "AAA").sum() == S.N
    early = chain_on(big, "AAA", S.DAYS[50])                                   # the contract files hold AAAOLD on this day
    assert len(early) > 0 and set(early["instrument"]) == {"AAA"} and early["lot"].notna().all()
    events = Store(big).all_events()
    assert set(events[events["event_date"] == S.DAYS[60]]["instrument"]) == {"AAA", "BBB"}   # announced under the old name


# ---- market value (D-11) ----------------------------------------------------------------------------------------
def test_share_count_follows_a_split_using_nothing_later(big):
    events = pd.DataFrame({"ex_date": [pd.Timestamp("2024-06-03")], "factor": [0.5]})
    dates = pd.to_datetime(["2024-05-31", "2024-06-03", "2024-06-04"]).to_numpy()
    assert share_multiplier(dates, events).tolist() == [1.0, 2.0, 2.0]
    anchors = pd.DataFrame({"date": pd.to_datetime(["2024-01-01"]), "shares": [100.0]})
    cfg = big.section("store")
    got = shares_for(dates, anchors, events, cfg)
    assert got["shares"].tolist() == [100.0, 200.0, 200.0] and set(got["method"]) == {"carried_forward"}

    mv = read_table(big, "market_value")
    fff = mv[mv["instrument"] == "FFF"].set_index("date")
    assert fff.loc[S.DAYS[S.SPLIT_AT - 1], "shares"] == pytest.approx(5.0e8)
    assert fff.loc[S.DAYS[S.SPLIT_AT], "shares"] == pytest.approx(1.0e9)       # carried from the early count, doubled by the split
    assert fff.loc[S.DAYS[S.SPLIT_AT], "method"] == "carried_forward"
    assert fff.loc[S.DAYS[400], "shares"] == pytest.approx(1.0e9) and fff.loc[S.DAYS[400], "method"] == "observed"


def test_market_value_says_how_each_share_count_was_obtained(big):
    mv = read_table(big, "market_value")
    eee = mv[mv["instrument"] == "EEE"].set_index("date")
    assert eee.loc[S.DAYS[100], "method"] == "back_cast"                       # no early count exists: marked as using later data
    assert eee.loc[S.DAYS[350], "method"] == "observed"
    row = mv[(mv["instrument"] == "BBB") & (mv["date"] == S.DAYS[10])].iloc[0]
    assert row["market_value"] == pytest.approx(row["shares"] * row["close"]) and row["method"] == "observed"
    seen = Store(big).market_value(asof=Decision(S.DAYS[50], "1430"))
    assert seen["date"].max() == S.DAYS[49]


# ---- the monthly list (U-1 to U-6) ------------------------------------------------------------------------------
def _parts(settings):
    store = Store(settings)
    return store.calendar, read_table(settings, "market_value"), option_activity(settings)


def test_list_is_unchanged_when_all_later_data_is_deleted(big):
    cal, mv, act = _parts(big)
    cfg = big.section("universe")
    for rd in rebuild_dates(cal)[2::5]:
        full = build_month(rd, cal, mv, act, cfg)
        cut = build_month(rd, type(cal)(cal.days[cal.days <= rd]), mv[mv["date"] < rd], act[act["date"] < rd], cfg)   # U-1
        pd.testing.assert_frame_equal(full.reset_index(drop=True), cut.reset_index(drop=True))
        assert (full["data_up_to"] < rd).all()


def test_new_listing_faded_stock_and_stock_without_options(big):
    u = read_all(big)
    listed = u[u["in_list"]]
    assert (listed.groupby("month").size() == S.TOP_N).all()
    first_g = listed[listed["instrument"] == "GGG"]["rebuild_date"].min()
    cal = Store(big).calendar
    assert cal.index_on_or_before(first_g) - S.LIST_AT >= 16                  # 80% of 20 sessions of option trading first (U-2)
    assert "NOP" not in set(u["instrument"])                                   # the largest company, but no options: never ranked
    last_h = u[u["instrument"] == "HHH"]["rebuild_date"].max()
    assert cal.index_on_or_before(last_h) < S.FADE_AT + 20                     # dropped once its options stopped trading
    assert set(INDEX_BOOK).isdisjoint(set(u["instrument"]))
    ranked = u[u["month"] == "2022-06"].sort_values("rank")
    assert ranked["market_value"].is_monotonic_decreasing                      # ranked by market value (A1)


def test_each_month_is_saved_once_and_a_rebuild_keeps_the_old_file(fresh):
    files = sorted(universe_dir(fresh).glob("*.parquet"))
    before = {f.name: f.read_bytes() for f in files}
    again = build_universe(fresh)
    assert again["months_built"] == 0 and again["months_kept"] == len(files)   # U-3: never overwritten by a normal build
    assert {f.name: f.read_bytes() for f in sorted(universe_dir(fresh).glob("*.parquet"))} == before
    build_universe(fresh, rebuild=True)
    assert len(list((universe_dir(fresh) / "_superseded").glob("*.parquet"))) == len(files)
    top = pd.read_csv(fresh.path("records") / "universe" / "top_list.csv")
    assert top["month"].nunique() == len(files)


def test_tradeable_list_has_the_indices_and_keeps_open_trades(big):
    cal = Store(big).calendar
    for rd in rebuild_dates(cal)[1:]:
        names = tradeable(big, rd)
        assert names[:2] == list(INDEX_BOOK) and len(names) == 2 + S.TOP_N     # U-5
    late = S.DAYS[410]
    assert "HHH" not in stock_list(big, late)
    assert "HHH" in tradeable(big, late, open_instruments=["HHH"])             # U-4: stays tradeable until its trade exits
    assert Store(big).universe(asof=Decision(late)) == tradeable(big, late)


def test_list_cannot_be_read_before_its_rebuild_date(big):
    month = read_month(big, "2022-10-03")
    assert month["rebuild_date"].iloc[0] == pd.Timestamp("2022-10-03")
    with pytest.raises(UniverseError, match="before that"):
        read_month(big, "2022-10-01")                                          # a Saturday, before the first session of October
    with pytest.raises(UniverseError, match="No stock list"):
        stock_list(big, "2030-01-01")


def test_training_population_is_every_fo_stock_of_the_time(big):
    cal, _, act = _parts(big)
    members = fo_members(act, cal, S.DAYS[150])
    assert {"AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "HHH", "NOP"} <= set(members) and "GGG" not in members    # U-6
    assert len(members) > len(stock_list(big, S.DAYS[150]))
    assert np.all([m not in INDEX_BOOK for m in members])
