"""The feature registry, leak tests and outcomes (F-1 to F-10, D-2, D-6, T-1 to T-3, M-5)."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

import synthetic as S
from koscine4.features import build as fb
from koscine4.features import builders as B
from koscine4.features.leak import register_checked, truncation_test
from koscine4.features.registry import FAMILIES, MAX_MODEL_FEATURES, REGISTRY, Feature, Registry, RegistryError, UnknownFeature
from koscine4.harness.leak import too_good_columns
from koscine4.models.core import ModelConfig, ModelError
from koscine4.outcomes import build as outcomes
from koscine4.store import Decision, Store
from koscine4.store.surface import black_price, implied_vol

KEY = ["date", "instrument"]
SAMPLE = [S.DAYS[150], S.DAYS[215], S.DAYS[330], S.DAYS[405]]


@pytest.fixture(scope="module")
def loaded(big):
    store = Store(big)
    inputs = fb.load_inputs(store, Decision(store.calendar.days[-1], "evening"))
    return store, inputs, fb.all_lags(big, "evening")


# ---- registry (F-1, F-2, F-7 to F-10) ---------------------------------------------------------------------------
def test_an_unregistered_column_cannot_be_read(big):
    with pytest.raises(UnknownFeature):
        fb.read(["entry_1d_open"], settings=big)                    # the column that leaked in the old model
    with pytest.raises(UnknownFeature):
        REGISTRY.require(["ret_1d", "next_day_return"])
    got = fb.read(["ret_1d", "atm_iv"], ["AAA"], S.DAYS[100], S.DAYS[110], settings=big)
    assert list(got.columns) == KEY + ["ret_1d", "atm_iv"] and len(got) == 11


def test_every_feature_states_its_source_formula_and_lookback(big):
    sources = set(big.section("availability")) | {"intraday"}
    for f in REGISTRY.describe():
        assert f["source"] in sources and f["formula"] and f["lookback"] >= 0 and f["scale_free"]
    for family in ("implied_vs_realised", "volatility_state", "events", "regime", "structure"):     # F-8
        assert REGISTRY.names(family=family), family
    assert set(FAMILIES) >= {REGISTRY.get(n).family for n in REGISTRY.names()}
    pct = [f for f in REGISTRY.describe() if f["percentile_of"]]                                    # F-7
    assert {"rv_20_pctile", "iv_pctile", "fii_net_5d_pctile"} <= {f["name"] for f in pct}


def test_a_raw_price_level_cannot_be_registered():
    r = Registry()
    with pytest.raises(RegistryError, match="scale-free"):
        r.register(Feature("close_price", "trend", "price", "prices_raw", "close", 0, scale_free=False))     # F-9
    r.register(Feature("x", "trend", "price", "prices_raw", "close / open - 1", 0))
    with pytest.raises(RegistryError, match="already"):
        r.register(Feature("x", "trend", "price", "prices_raw", "close / open - 1", 0))


def test_a_model_may_use_at_most_forty_features():
    names = tuple(REGISTRY.names(decision_time="evening"))
    assert len(names) > MAX_MODEL_FEATURES
    with pytest.raises(ModelError, match="at most"):
        ModelConfig(name="m", kind="regression", algorithm="ridge", features=names).validate()               # F-10
    with pytest.raises(ModelError, match="not valid for a evening"):
        ModelConfig(name="m", kind="regression", algorithm="ridge", features=("intraday_ret_1430",)).validate()   # F-5
    with pytest.raises(ModelError):
        ModelConfig(name="m", kind="regression", algorithm="neural_net", features=names[:5]).validate()      # M-3


# ---- leak tests (F-3, F-4) --------------------------------------------------------------------------------------
def test_no_registered_feature_changes_when_later_data_is_removed(loaded):
    _, inputs, lags = loaded
    problems = truncation_test(lambda inp: fb.compute(inp, "evening", lags), inputs, SAMPLE)
    assert problems == []                                                                                     # F-3


def test_no_1430_feature_changes_when_later_data_is_removed(big):
    store = Store(big)
    inputs = fb.load_inputs(store, Decision(store.calendar.days[-1], "evening"))
    lags = fb.all_lags(big, "1430")
    assert truncation_test(lambda inp: fb.compute(inp, "1430", lags), inputs, SAMPLE[:2]) == []


def test_a_feature_built_from_tomorrow_is_refused_and_flagged_if_forced(loaded):
    _, inputs, lags = loaded

    def with_planted(inp):
        out = fb.compute(inp, "evening", lags)
        p = inp.prices.sort_values(["instrument", "date"])
        p["tomorrow_return"] = p.groupby("instrument")["close"].shift(-1) / p["close"] - 1.0
        return out.merge(p[KEY + ["tomorrow_return"]], on=KEY, how="left")

    planted = Feature("tomorrow_return", "trend", "price", "prices_raw", "close tomorrow / close - 1", 1)
    with pytest.raises(RegistryError, match="after the decision date"):
        register_checked(Registry(), planted, with_planted, inputs, SAMPLE[:2])                               # the registry refuses it
    forced = Registry()
    forced.register(planted)                                                                                  # forced in ...
    flagged = truncation_test(with_planted, inputs, SAMPLE[:2])
    assert {p["feature"] for p in flagged} == {"tomorrow_return"}                                             # ... and caught (F-4)
    honest = Feature("ret_1d_again", "trend", "price", "prices_raw", "close / previous close - 1", 1)
    register_checked(Registry(), honest, lambda inp: fb.compute(inp, "evening", lags).assign(ret_1d_again=lambda d: d["ret_1d"]), inputs, SAMPLE[:1])


# ---- decision times (F-5, F-6) ----------------------------------------------------------------------------------
def test_a_1430_decision_uses_no_price_from_its_own_day(loaded, big):
    _, inputs, _ = loaded
    lags = fb.all_lags(big, "1430")
    assert lags["prices_raw"] == 1 and fb.all_lags(big, "evening")["prices_raw"] == 0
    base = fb.compute(inputs, "1430", lags)
    last = inputs.calendar.days[-1]
    shocked = inputs.prices.copy()
    hit = shocked["date"] == last
    shocked.loc[hit, ["open", "high", "low", "close"]] *= 1.5                 # change today's prices completely
    again = fb.compute(replace(inputs, prices=shocked), "1430", lags)
    names = [n for n in REGISTRY.names(decision_time="1430") if REGISTRY.get(n).group != "intraday"]
    a = base[base["date"] == last].set_index("instrument")[names]
    b = again[again["date"] == last].set_index("instrument")[names]
    pd.testing.assert_frame_equal(a, b)                                       # F-5: nothing moved
    ev = fb.read(["ret_1d", "rv_20"], ["BBB"], settings=big).set_index("date")
    at = fb.read(["ret_1d", "rv_20"], ["BBB"], decision_time="1430", settings=big).set_index("date")
    d, prev = S.DAYS[200], S.DAYS[199]
    assert at.loc[d, "ret_1d"] == pytest.approx(ev.loc[prev, "ret_1d"])      # what 14:30 sees is yesterday's evening row


def test_the_live_row_equals_the_backtest_row_once_it_is_history(big):
    store = Store(big)
    day = S.DAYS[180]                                                         # before the FFF split, which later rescales its prices
    live = fb.features_asof(store, Decision(day, "evening"))
    live = live[live["date"] == day].set_index("instrument").drop(columns=["date", "decision_time"])
    names = list(live.columns)
    stored = fb.read(names, None, day, day, settings=big).set_index("instrument")[names]
    pd.testing.assert_frame_equal(live.sort_index(), stored.sort_index(), check_exact=False, rtol=1e-9)      # F-6


def test_features_do_not_depend_on_the_price_level(loaded):
    _, inputs, lags = loaded
    scaled = inputs.prices.copy()
    hit = scaled["instrument"] == "AAA"
    scaled.loc[hit, ["open", "high", "low", "close", "prev_close"]] *= 100.0
    deriv = inputs.deriv.copy()
    deriv.loc[deriv["instrument"] == "AAA", "fut_close"] *= 100.0
    a = fb.compute(inputs, "evening", lags)
    b = fb.compute(replace(inputs, prices=scaled, deriv=deriv), "evening", lags)
    own = [n for n in REGISTRY.names(decision_time="evening") if REGISTRY.get(n).group in ("price", "option", "event")]
    x = a[a["instrument"] == "AAA"].set_index("date")[own]
    y = b[b["instrument"] == "AAA"].set_index("date")[own]
    pd.testing.assert_frame_equal(x, y, check_exact=False, rtol=1e-9)                                         # F-9


def test_a_result_date_counts_only_after_it_is_announced(big):
    f = fb.read(["days_to_results", "results_in_5d", "days_since_results"], ["AAA"], settings=big).set_index("date")
    meeting, told = 125, 125 - S.ANNOUNCE_AHEAD
    assert f.loc[S.DAYS[told - 1], "days_to_results"] == 1.0                  # not announced yet: unknown
    assert f.loc[S.DAYS[told], "days_to_results"] == pytest.approx(S.ANNOUNCE_AHEAD / 30.0)
    assert f.loc[S.DAYS[meeting - 3], "results_in_5d"] == 1.0 and f.loc[S.DAYS[told], "results_in_5d"] == 0.0
    assert f.loc[S.DAYS[meeting], "days_since_results"] == 0.0 and f.loc[S.DAYS[meeting + 6], "days_since_results"] == pytest.approx(0.1)
    ccc = fb.read(["days_to_results"], ["CCC"], settings=big).set_index("date")
    assert (ccc.loc[:S.DAYS[149], "days_to_results"] == 1.0).all()            # no announcement date known: never seen coming


def test_intraday_state_at_1430_ignores_later_bars():
    ts = pd.to_datetime(["2024-01-02 09:15", "2024-01-02 12:00", "2024-01-02 14:25", "2024-01-02 14:30", "2024-01-02 15:25"])
    bars = pd.DataFrame({"timestamp": ts, "instrument": "NIFTY", "open": [100, 101, 102, 103, 120.0], "high": [101, 103, 104, 130, 140.0],
                         "low": [99, 100, 101, 102, 119.0], "close": [101, 102, 103, 125, 139.0]})
    prices = pd.DataFrame({"date": pd.to_datetime(["2024-01-01"]), "instrument": ["NIFTY"], "kind": "index", "close": [98.0]})
    inp = B.Inputs(prices=prices, surface=pd.DataFrame(), deriv=pd.DataFrame(), fii_dii=pd.DataFrame(), participant=pd.DataFrame(),
                   events=pd.DataFrame(), calendar=None, intraday=bars)
    row = B.intraday_features(inp).iloc[0]
    assert row["intraday_ret_1430"] == pytest.approx(103 / 100 - 1)           # the 14:25 bar ends at 14:30; the 14:30 bar is excluded
    assert row["intraday_range_1430"] == pytest.approx((104 - 99) / 100)
    assert row["gap_open_today"] == pytest.approx(100 / 98 - 1)


# ---- option surface (D-6) ---------------------------------------------------------------------------------------
def test_implied_volatility_is_recovered_from_formula_prices(big):
    k = np.array([90.0, 100.0, 110.0])
    for vol in (0.12, 0.35, 0.9):
        prices = black_price(100.0, k, 0.08, vol, [False, True, True])
        assert implied_vol(prices, 100.0, k, 0.08, [False, True, True]) == pytest.approx([vol] * 3, abs=1e-6)
    assert np.isnan(implied_vol(0.0, 100.0, 100.0, 0.1, True))
    surface = Store(big).surface(asof=Decision(S.DAYS[-1]))
    near = surface[(surface["dte"] >= 7) & (surface["dte"] <= 40)]
    for instrument, spec in S.OPTION_SPECS.items():
        got = near[near["instrument"] == instrument]["atm_iv"]
        assert len(got) > 300 and got.median() == pytest.approx(spec[4], abs=0.01)       # the volatility the prices were made with
    assert (surface["forward_source"] == "futures").all()
    assert Store(big).surface(asof=Decision(S.DAYS[50], "1430"))["date"].max() == S.DAYS[49]


# ---- outcomes (T-1 to T-3, D-2, M-5) ----------------------------------------------------------------------------
def test_outcomes_match_a_hand_calculation(big):
    store = Store(big)
    px = store.prices(asof=Decision(S.DAYS[-1])).set_index(["instrument", "date"])
    o5 = outcomes.read(big, "next_open", S.DAYS[-1], 5).set_index(["instrument", "date"])
    o1 = outcomes.read(big, "1515", S.DAYS[-1], 1).set_index(["instrument", "date"])
    for instrument, i in (("AAA", 40), ("BBB", 133), ("NIFTY", 250), ("DDD", 301), ("FFF", 198)):        # the last spans the split
        d = S.DAYS[i]
        entry = px.loc[(instrument, S.DAYS[i + 1]), "open"]                    # next session's open: the overnight gap is not in the outcome (T-2)
        ret = px.loc[(instrument, S.DAYS[i + 5]), "close"] / entry - 1.0
        row = o5.loc[(instrument, d)]
        assert row["fwd_ret"] == pytest.approx(ret) and row["fwd_abs"] == pytest.approx(abs(ret)) and row["fwd_up"] == float(ret > 0)
        assert row["entry_date"] == S.DAYS[i + 1] and row["window_end"] == S.DAYS[i + 5]
        hi = max(px.loc[(instrument, S.DAYS[i + k]), "high"] for k in range(1, 6)) / entry - 1.0
        lo = min(px.loc[(instrument, S.DAYS[i + k]), "low"] for k in range(1, 6)) / entry - 1.0
        assert row["within_2pct"] == float(max(abs(hi), abs(lo)) <= 0.02)
        same_day = px.loc[(instrument, S.DAYS[i + 1]), "close"] / px.loc[(instrument, d), "close"] - 1.0
        assert o1.loc[(instrument, d), "fwd_ret"] == pytest.approx(same_day)       # 14:30 decision: entry stands at that day's close
    row = o5.loc[("AAA", S.DAYS[200])]
    assert row["implied"] == pytest.approx(0.26 * np.sqrt(5 / 252), abs=0.003) and row["excess"] == pytest.approx(row["fwd_abs"] - row["implied"])
    assert abs(o5.loc[("FFF", S.DAYS[198]), "fwd_ret"]) < 0.2                      # the split is not a 50% move


def test_an_outcome_is_readable_only_after_its_window_has_passed(big):
    with pytest.raises(outcomes.OutcomeError, match="cutoff"):
        outcomes.read(big, "next_open", None, 5)
    cutoff = S.DAYS[300]
    o = outcomes.read(big, "next_open", cutoff, 5)
    assert o["window_end"].max() == cutoff and o["date"].max() == S.DAYS[295] and o["fwd_ret"].notna().all()        # T-3
    everything = outcomes.read(big, "next_open", S.DAYS[-1] + pd.Timedelta(days=30), 5)
    assert everything["date"].max() == S.DAYS[-6]                              # the last five sessions are incomplete and never returned
    with pytest.raises(outcomes.OutcomeError):
        outcomes.read(big, "next_open", cutoff, 4)


def test_features_and_outcomes_share_no_column_and_no_information(big):
    feats = pd.read_parquet(fb.table_path(big, "evening"))
    moves = pd.read_parquet(outcomes.moves_path(big, "next_open"))
    assert (set(feats.columns) & set(moves.columns)) == set(KEY)               # D-2: separate tables, joined only through read()
    assert fb.table_path(big, "evening").parent != outcomes.moves_path(big, "next_open").parent
    o = outcomes.read(big, "next_open", S.DAYS[-1], 5)
    joined = feats.merge(o[KEY + ["fwd_ret"]], on=KEY)
    assert too_good_columns(joined.drop(columns=KEY + ["decision_time", "fwd_ret"]), joined["fwd_ret"]) == []
    assert outcomes.overlap_weight(5) * 5 == pytest.approx(1.0)                # M-5: five overlapping rows count as one event
