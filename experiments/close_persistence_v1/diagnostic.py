"""Close-persistence experiment - DIAGNOSTIC phase (no modeling yet).

Answers, on the locked PROD OOS predictions (read-only):
  (a) reproduce the 25/50/25 close split on PROD top-2 + define/verify the top-3-mover metric
  (b) ORACLE bound  - of N candidates/day, can we even pick 2 that close >= thr?  (reachability)
  (c) separability  - do side-margin / trend-alignment signals beat confidence at picking closers?
  (d) tension       - does optimizing close cost the top-3-mover (magnitude) property?

All artifacts/logs are sandbox-local. PROD is consumed read-only.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))   # shared koscine3.data

import numpy as np
import pandas as pd

from koscine3.data.sources import load_market_data

PROD_PRED = HERE.parents[1] / "locks" / "prod_largemove_v1" / "predictions"
THR = {"A_mcap30": 0.03, "B_turn35": 0.04}
COOLDOWN = 3
pd.set_option("display.width", 240)
pd.set_option("display.max_columns", 40)


# --------------------------------------------------------------------------- data
def build_pool() -> tuple[pd.DataFrame, dict]:
    pool = pd.concat(
        [pd.read_csv(PROD_PRED / f"group_{b}_predictions.csv", parse_dates=["date"]) for b in THR],
        ignore_index=True,
    )
    pool["symbol"] = pool["symbol"].astype(str)

    cols = ["date", "symbol", "open", "close", "ret_20d_cs_rank", "close_sma50_dist"]
    m = load_market_data(columns=cols)
    m["symbol"] = m["symbol"].astype(str)
    m = m.sort_values(["symbol", "date"])
    g = m.groupby("symbol", sort=False)
    cm = pd.DataFrame({
        "date": m["date"].values, "symbol": m["symbol"].values,
        "entry_open": g["open"].shift(-1).values, "win_close": g["close"].shift(-5).values,
    })
    feat = m[["date", "symbol", "ret_20d_cs_rank", "close_sma50_dist"]].drop_duplicates(["date", "symbol"])
    cal = {pd.Timestamp(d): i for i, d in enumerate(sorted(m["date"].unique()))}

    pool = pool.merge(cm, on=["date", "symbol"], how="left").merge(feat, on=["date", "symbol"], how="left")
    pool["close_move"] = np.where(
        pool.side.eq("long"),
        (pool.win_close - pool.entry_open) / pool.entry_open,
        (pool.entry_open - pool.win_close) / pool.entry_open,
    )

    # side margin = conviction of chosen side over the other side
    wide = pool.pivot_table(index=["date", "symbol"], columns="side", values="confidence")
    wide = wide.rename(columns={"long": "c_long", "short": "c_short"}).reset_index()
    pool = pool.merge(wide, on=["date", "symbol"], how="left")
    pool["side_margin"] = np.where(pool.side.eq("long"), pool.c_long - pool.c_short, pool.c_short - pool.c_long)

    # trend / momentum alignment (long wants uptrend, short wants downtrend)
    pool["trend_sma"] = np.where(pool.side.eq("long"), pool.close_sma50_dist, -pool.close_sma50_dist)
    pool["mom_align"] = np.where(pool.side.eq("long"), pool.ret_20d_cs_rank - 0.5, 0.5 - pool.ret_20d_cs_rank)

    def z(s):
        return (s - s.mean()) / (s.std() + 1e-9)
    pool["blend"] = 0.5 * z(pool["side_margin"]) + 0.5 * z(pool["mom_align"])

    # top-3-mover rank: per (day, group) rank stocks by their larger-side ceiling (magnitude)
    mag = pool.groupby(["date", "group", "symbol"], as_index=False)["actual_move_%"].max()
    mag = mag.rename(columns={"actual_move_%": "stock_mag"})
    mag["mover_rank"] = mag.groupby(["date", "group"])["stock_mag"].rank(ascending=False, method="first")
    pool = pool.merge(mag[["date", "group", "symbol", "mover_rank"]], on=["date", "group", "symbol"], how="left")
    return pool, cal


# ----------------------------------------------------------------------- selection
def select(dfg: pd.DataFrame, N: int, signal: str, cal: dict, n_pick: int = 2) -> pd.DataFrame:
    """Per day: gate top-N available (cooldown-respecting) by confidence, then pick n_pick by `signal`."""
    last: dict[str, int] = {}
    keep: list[int] = []
    for day, g in dfg.groupby("date", sort=True):
        i = cal[pd.Timestamp(day)]
        avail = g[g["symbol"].map(lambda s: (i - last.get(s, -10**9)) >= COOLDOWN)]
        poolN = avail.sort_values("confidence", ascending=False).head(N)
        ranked = poolN.sort_values(signal, ascending=False)
        seen: set[str] = set()
        for idx, s in zip(ranked.index, ranked["symbol"]):
            if s in seen:
                continue
            keep.append(idx); seen.add(s)
            if len(seen) >= n_pick:
                break
        for s in seen:
            last[s] = i
    return dfg.loc[keep]


def outcomes(sel: pd.DataFrame) -> dict:
    s = sel.dropna(subset=["close_move"]).copy()
    if s.empty:
        return dict(trades=0, above=0, small=0, opp=0, top3=0, peak=0)
    above = s.close_move >= s.threshold
    small = (s.close_move >= 0) & (s.close_move < s.threshold)
    return dict(
        trades=len(s),
        above=round(above.mean() * 100, 1),
        small=round(small.mean() * 100, 1),
        opp=round((s.close_move < 0).mean() * 100, 1),
        top3=round((s.mover_rank <= 3).mean() * 100, 1),
        peak=round(s.hit.mean() * 100, 1),
    )


def select_all(pool, N, signal, cal):
    return pd.concat([select(pool[pool.group.eq(b)], N, signal, cal) for b in THR], ignore_index=True)


# ----------------------------------------------------------------------------- run
def main():
    pool, cal = build_pool()
    print(f"pool rows={len(pool)} | with close_move={pool.close_move.notna().mean()*100:.0f}% "
          f"| dates={pool.date.nunique()}\n")

    # (a) baseline PROD top-2 (signal=confidence, N=2) -------------------------------
    print("=" * 92)
    print("(a) BASELINE  - PROD top-2/group (rank by confidence, t+3 cooldown): CLOSE outcomes")
    print("=" * 92)
    rows = []
    for b in THR:
        sel = select(pool[pool.group.eq(b)], 2, "confidence", cal)
        o = outcomes(sel); o["scope"] = f"{b} (>={int(THR[b]*100)}%)"; rows.append(o)
        s = sel.dropna(subset=["close_move"]); s = s.assign(yr=s.date.dt.year)
        for yr, d in s.groupby("yr"):
            oo = outcomes(d); oo["scope"] = f"  {b} {yr}"; rows.append(oo)
    base_all = select_all(pool, 2, "confidence", cal)
    o = outcomes(base_all); o["scope"] = "COMBINED"; rows.append(o)
    df = pd.DataFrame(rows)[["scope", "trades", "above", "small", "opp", "top3", "peak"]]
    print(df.to_string(index=False))
    print("\nabove=closed >= thr (favorable) | small=closed 0..thr | opp=closed opposite | "
          "top3=stock among day's 3 biggest movers in group | peak=ceiling hit (PROD target)")

    # (b)+(c)+(d) selection-rule comparison at N=7 ----------------------------------
    print("\n" + "=" * 92)
    print("(b/c/d) SELECTION COMPARISON @ N=7 candidate pool (combined groups)")
    print("=" * 92)
    rows = []
    for sig, label in [("confidence", "confidence (=PROD)"), ("side_margin", "side_margin"),
                       ("trend_sma", "trend_sma"), ("mom_align", "mom_align"),
                       ("blend", "blend(margin+mom)"), ("close_move", "ORACLE(close)")]:
        o = outcomes(select_all(pool, 7, sig, cal)); o["pick_by"] = label; rows.append(o)
    df = pd.DataFrame(rows)[["pick_by", "trades", "above", "small", "opp", "top3", "peak"]]
    print(df.to_string(index=False))
    print("\nORACLE = cheat: picks the 2 actual best-closers in the pool -> upper bound for any re-ranker.")

    # frontier across N --------------------------------------------------------------
    print("\n" + "=" * 92)
    print("FRONTIER across candidate-pool size N (combined): baseline vs blend vs ORACLE")
    print("=" * 92)
    rows = []
    for N in (5, 7, 10):
        for sig, label in [("confidence", "confidence"), ("blend", "blend"), ("close_move", "ORACLE")]:
            o = outcomes(select_all(pool, N, sig, cal)); o["N"] = N; o["pick_by"] = label; rows.append(o)
    df = pd.DataFrame(rows)[["N", "pick_by", "trades", "above", "opp", "top3", "peak"]]
    print(df.to_string(index=False))

    # raw separability: does close-outcome vary across a signal within the N=7 pool? --
    print("\n" + "=" * 92)
    print("RAW SEPARABILITY - within N=7 confidence pool, split by signal tertiles")
    print("=" * 92)
    # candidate pool = gate top-7 by confidence, keep all (n_pick=7), then inspect close vs signal
    cand = pd.concat([select(pool[pool.group.eq(b)], 7, "confidence", cal, n_pick=7) for b in THR],
                     ignore_index=True).dropna(subset=["close_move"])
    for sig in ["side_margin", "mom_align", "trend_sma", "blend"]:
        cand["_t"] = pd.qcut(cand[sig], 3, labels=["low", "mid", "high"], duplicates="drop")
        g = cand.groupby("_t", observed=True).apply(
            lambda d: pd.Series({"above%": (d.close_move >= d.threshold).mean() * 100,
                                 "opp%": (d.close_move < 0).mean() * 100,
                                 "n": len(d)}))
        print(f"\n[{sig}]")
        print(g.round(1).to_string())


if __name__ == "__main__":
    main()
