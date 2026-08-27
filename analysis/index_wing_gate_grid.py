"""Wing x gate grid for index credit spreads, with an out-of-sample split.

The single-config runs showed the result swings hard on wing width -- narrow wings lose to fixed
costs, wide wings run out of liquidity -- so a hand-picked tier set can easily flatter itself.
This searches the grid properly and then asks the only question that matters about the winner:
does it hold up on data it was not chosen on?

The in-sample/out-of-sample split is chronological (first half / second half), and the config is
selected ON THE FIRST HALF ONLY, then applied unchanged to the second. A config that looks good
in-sample and dies out-of-sample is the expected outcome of grid-searching a thin dataset -- the
point of this script is to find that out rather than ship it.

Usage:
    python analysis/index_wing_gate_grid.py --index nifty
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"C:\Users\rahul\Koscine 3.0")
SHORT_OTM = 0.02
DTE_MIN, FWD, SAFE_DTE = 7, 5, 4
MIN_RISK_FRAC = 0.10
MAX_SNAP_ERR_PCT = 0.35

BROKERAGE_PER_ORDER, GST_PCT = 20.0, 0.18
STT_SELL_PCT, EXCH_TXN_PCT = 0.0625 / 100, 0.03503 / 100
SEBI_PCT, STAMP_PCT = 10.0 / 1e7, 0.003 / 100

WINGS = [(0.005, 0.01), (0.01, 0.02), (0.015, 0.03), (0.02, 0.04), (0.025, 0.05), (0.03, 0.06)]
GATES = [10, 15, 20, 25, 30, 40]


def costs(sell_p, buy_p, exit_v, credit, lot, n_legs=4):
    ge = (sell_p + buy_p) * lot
    turn = ge + ge * np.clip(exit_v / np.maximum(credit, 1e-9), 0, 3)
    brok = BROKERAGE_PER_ORDER * n_legs * 2
    return (brok + STT_SELL_PCT * turn * .5 + EXCH_TXN_PCT * turn + SEBI_PCT * turn
            + STAMP_PCT * turn * .5 + GST_PCT * (brok + EXCH_TXN_PCT * turn + SEBI_PCT * turn))


def build(index, entry):
    files = sorted(glob.glob(str(ROOT / "data/intraday" / f"{index}_option_chain_5m" / "*.parquet")))
    if not files:
        raise SystemExit(f"no {index} chain data")
    keep = ["timestamp", "expiry", "strike", "underlying_close",
            "ce_close", "ce_volume", "ce_lot_size", "pe_close", "pe_volume"]
    out = []
    for f in files:
        x = pd.read_parquet(f, columns=keep)
        x = x[x["timestamp"].dt.strftime("%H:%M") == entry]
        if not x.empty:
            out.append(x)
    c = pd.concat(out, ignore_index=True)
    c["date"] = c["timestamp"].dt.strftime("%Y-%m-%d")
    c["expiry"] = pd.to_datetime(c["expiry"]).dt.strftime("%Y-%m-%d")
    return c


def run(c, wce, wpe):
    """all candidate spreads for one wing config (gate applied later)"""
    ce = {(r.date, r.expiry, r.strike): (r.ce_close, r.ce_volume)
          for r in c[c["ce_close"].notna()].itertuples()}
    pe = {(r.date, r.expiry, r.strike): (r.pe_close, r.pe_volume)
          for r in c[c["pe_close"].notna()].itertuples()}
    lot = float(c["ce_lot_size"].dropna().iloc[0])
    spot = c.drop_duplicates(["date"]).set_index("date")["underlying_close"].to_dict()
    st: dict = {}
    for r in c.itertuples():
        st.setdefault((r.date, r.expiry), []).append(r.strike)
    st = {k: np.sort(np.unique(v)) for k, v in st.items()}
    exp_by: dict = {}
    for (d, e) in st:
        exp_by.setdefault(d, set()).add(e)
    days = sorted(exp_by)
    dpos = {d: i for i, d in enumerate(days)}

    def snap(d, e, tgt, u, strict):
        s = st.get((d, e))
        if s is None or not len(s):
            return None
        k = s[np.abs(s - tgt).argmin()]
        return None if strict and abs(k - tgt) / u * 100 > MAX_SNAP_ERR_PCT else float(k)

    rows = []
    for d in days:
        u = spot.get(d)
        if u is None or not np.isfinite(u):
            continue
        e = next((x for x in sorted(exp_by[d])
                  if (pd.Timestamp(x) - pd.Timestamp(d)).days >= DTE_MIN), None)
        if e is None:
            continue
        ks = [snap(d, e, u * (1 + SHORT_OTM), u, True), snap(d, e, u * (1 + SHORT_OTM + wce), u, False),
              snap(d, e, u * (1 - SHORT_OTM), u, True), snap(d, e, u * (1 - SHORT_OTM - wpe), u, False)]
        if None in ks or ks[0] == ks[1] or ks[2] == ks[3]:
            continue
        lg = [ce.get((d, e, ks[0])), ce.get((d, e, ks[1])),
              pe.get((d, e, ks[2])), pe.get((d, e, ks[3]))]
        if any(x is None for x in lg) or not all(x[1] and x[1] > 0 for x in lg):
            continue
        p = [x[0] for x in lg]
        sp, bp = p[0] + p[2], p[1] + p[3]
        credit = sp - bp
        width = max(ks[1] - ks[0], ks[2] - ks[3])
        risk = width - credit
        if credit <= 0 or risk <= 0 or risk < MIN_RISK_FRAC * width:
            continue
        # forward walk
        vals, exited = [], False
        for k in range(1, FWD + 1):
            if dpos[d] + k >= len(days):
                break
            d2 = days[dpos[d] + k]
            if (pd.Timestamp(e) - pd.Timestamp(d2)).days <= SAFE_DTE:
                exited = True
            l2 = [ce.get((d2, e, ks[0])), ce.get((d2, e, ks[1])),
                  pe.get((d2, e, ks[2])), pe.get((d2, e, ks[3]))]
            if all(x is not None for x in l2):
                vals.append(max(0.0, min(width, (l2[0][0] - l2[1][0]) + (l2[2][0] - l2[3][0]))))
            if exited:
                break
        if not vals:
            continue
        pnl = credit - vals[-1]
        rows.append({"date": d, "credit": credit, "risk": risk, "sp": sp, "bp": bp,
                     "exit": vals[-1], "pnl_lot": pnl * lot, "risk_lot": risk * lot,
                     "ror": credit / risk * 100, "lot": lot, "win": pnl > 0,
                     "vol": min(x[1] for x in lg)})
    return pd.DataFrame(rows)


def score(df, gate):
    s = df[df["ror"] > gate]
    if len(s) < 10:
        return None
    cost = costs(s["sp"], s["bp"], s["exit"], s["credit"], s["lot"])
    net = s["pnl_lot"] - cost
    return {"n": len(s), "win": s["win"].mean() * 100,
            "gross_pct": s["pnl_lot"].sum() / s["risk_lot"].sum() * 100,
            "net_pct": net.sum() / s["risk_lot"].sum() * 100,
            "net_total": net.sum(), "med_vol": s["vol"].median()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="nifty")
    ap.add_argument("--entry", default="15:15")
    args = ap.parse_args()
    print(f"loading {args.index} ...", flush=True)
    c = build(args.index, args.entry)
    days = sorted(c["date"].unique())
    split = days[len(days) // 2]
    print(f"  {c['date'].nunique()} days; IS <= {split} < OOS\n", flush=True)

    cache = {}
    print("=" * 100)
    print(f"{args.index.upper()} -- NET of costs, % of risk (full sample). "
          f"blank = fewer than 10 trades")
    print("=" * 100)
    print(f"  {'wings':>12} " + "".join(f"{'g' + str(g):>11}" for g in GATES))
    print("  " + "-" * 80)
    for wce, wpe in WINGS:
        df = run(c, wce, wpe)
        cache[(wce, wpe)] = df
        cells = []
        for g in GATES:
            r = score(df, g)
            cells.append(f"{r['net_pct']:>10.2f}%" if r else f"{'--':>11}")
        print(f"  {f'{wce*100:g}x{wpe*100:g}%':>12} " + "".join(cells))

    print(f"\n  {'wings':>12} " + "".join(f"{'g' + str(g):>11}" for g in GATES) + "   (trade counts)")
    print("  " + "-" * 80)
    for wce, wpe in WINGS:
        cells = []
        for g in GATES:
            r = score(cache[(wce, wpe)], g)
            cells.append(f"{r['n']:>11,}" if r else f"{'--':>11}")
        print(f"  {f'{wce*100:g}x{wpe*100:g}%':>12} " + "".join(cells))

    # pick the best config on the FIRST HALF only, then test it on the second
    print()
    print("=" * 100)
    print("OUT-OF-SAMPLE: config chosen on the first half, applied unchanged to the second")
    print("=" * 100)
    best, best_net = None, -1e9
    for (wce, wpe), df in cache.items():
        ins = df[df["date"] <= split]
        for g in GATES:
            r = score(ins, g)
            if r and r["net_pct"] > best_net:
                best_net, best = r["net_pct"], (wce, wpe, g, r)
    if best is None:
        print("  no config had enough in-sample trades")
        return
    wce, wpe, g, r_is = best
    print(f"  chosen: wings {wce*100:g}%x{wpe*100:g}%, gate {g}%")
    print(f"    IN-SAMPLE : n={r_is['n']:,}  win {r_is['win']:.1f}%  "
          f"gross {r_is['gross_pct']:+.2f}%  NET {r_is['net_pct']:+.2f}% of risk")
    oos = cache[(wce, wpe)]
    r_oos = score(oos[oos["date"] > split], g)
    if r_oos:
        print(f"    OUT-OF-SAMPLE: n={r_oos['n']:,}  win {r_oos['win']:.1f}%  "
              f"gross {r_oos['gross_pct']:+.2f}%  NET {r_oos['net_pct']:+.2f}% of risk")
        verdict = "HOLDS UP" if r_oos["net_pct"] > 0 else "FAILS OUT-OF-SAMPLE"
        print(f"\n  => {verdict}")
    else:
        print("    OUT-OF-SAMPLE: too few trades to judge")


if __name__ == "__main__":
    main()
