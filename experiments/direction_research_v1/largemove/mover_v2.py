"""PRODUCTION v2 — direction-agnostic large-mover picker.

Rationale (see experiments/persistence_flow_v1/DIRECTION_RESEARCH.md): direction is unpredictable
(coin flip, ~0.52 AUC across ~25 academic signals); magnitude is predictable and IMPLIED VOL is the
single best ranker (no model — incl. quarterly-retrained — beats it). So v2 ranks eligible stocks by
atm_iv, per group, with t+3 cooldown + a per-stock trade-share cap, taking top-3/group/day. The book
is direction-agnostic (trade via straddle/strangle, or single side + exit-at-peak).

    python -m largemove.mover_v2          # build book + summary + lock locks/prod_largemove_v2/
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from koscine3.data.sources import load_market_data
from largemove.config import LOCK_DIR as LOCK_V1

VERSION = "prod_largemove_v2"
RANKER = "atm_iv"
COOLDOWN = 3
CAP_FRAC = 0.15          # max share of a group's trades one name may take (non-binding belt; cooldown caps ~11%)
DEPTH = 3               # picks per group per day
MIN_UNDERLYING = 100.0
WINDOW = 5
BOOK_YEARS = (2024, 2025, 2026)
GROUP_SIZE = {"A_mcap30": 30, "B_turn35": 35}

ROOT = Path(__file__).resolve().parents[3]
PKG = ROOT / "src" / "koscine3" / "largemove"
LOCK_V2 = ROOT / "locks" / VERSION


def load_book() -> pd.DataFrame:
    groups = json.loads((LOCK_V1 / "universe_groups.json").read_text())
    g2 = {s: g for g, syms in groups.items() for s in syms}
    m = load_market_data(columns=["date", "symbol", "open", "high", "low", "close", "atm_iv"])
    m["symbol"] = m["symbol"].astype(str)
    m = m[m.symbol.isin(g2)].sort_values(["symbol", "date"]).reset_index(drop=True)
    g = m.groupby("symbol", sort=False)
    entry = g["open"].shift(-1)
    H = pd.concat([g["high"].shift(-i) for i in range(1, WINDOW + 1)], axis=1).max(axis=1)
    L = pd.concat([g["low"].shift(-i) for i in range(1, WINDOW + 1)], axis=1).min(axis=1)
    c5 = g["close"].shift(-WINDOW)
    m["up_move"] = (H - entry) / entry
    m["down_move"] = (entry - L) / entry
    m["move_mag"] = m[["up_move", "down_move"]].max(axis=1)
    signed = (c5 - entry) / entry
    peak_up = m.up_move > m.down_move
    m["closed_opp"] = (peak_up & (signed < 0)) | (~peak_up & (signed > 0))
    m["group"] = m.symbol.map(g2)
    m["eligible"] = m.close.ge(MIN_UNDERLYING) & m.atm_iv.notna()
    m["year"] = m.date.dt.year
    return m


def select(eg: pd.DataFrame, cooldown: int = COOLDOWN, cap_frac: float = CAP_FRAC, depth: int = DEPTH) -> pd.DataFrame:
    """Rank by atm_iv, top-`depth`/day, t+`cooldown` cooldown, per-stock cap on share of trades."""
    last, cum, keep, total = {}, {}, [], 0
    for di, day in enumerate(sorted(eg.date.unique())):
        g = eg[eg.date == day].sort_values(RANKER, ascending=False)
        picked = 0
        for idx, s in zip(g.index, g.symbol):
            if di - last.get(s, -10**9) < cooldown:
                continue
            if total > 0 and cum.get(s, 0) >= cap_frac * total:
                continue
            keep.append(idx); last[s] = di; cum[s] = cum.get(s, 0) + 1; total += 1; picked += 1
            if picked >= depth:
                break
    return eg.loc[keep]


def build_picks(m: pd.DataFrame) -> pd.DataFrame:
    ev = m[m.eligible & m.group.notna() & m.year.isin(BOOK_YEARS)].copy()
    picks = pd.concat([select(ev[ev.group == grp]) for grp in GROUP_SIZE], ignore_index=True)
    return picks.sort_values(["date", "group", RANKER], ascending=[True, True, False])


def summarize(picks: pd.DataFrame) -> dict:
    yrs = (picks.date.max() - picks.date.min()).days / 365.25
    out = {}
    for grp in GROUP_SIZE:
        d = picks[picks.group == grp].dropna(subset=["move_mag"])
        vc = d.symbol.value_counts()
        out[grp] = {"trades": int(len(d)), "per_yr": round(len(d) / yrs),
                    "move_ge6_pct": round((d.move_mag >= 0.06).mean() * 100, 1),
                    "move_ge8_pct": round((d.move_mag >= 0.08).mean() * 100, 1),
                    "avg_move_pct": round(d.move_mag.mean() * 100, 1),
                    "closed_opp_pct": round(d.closed_opp.mean() * 100, 1),
                    "coverage": f"{d.symbol.nunique()}/{GROUP_SIZE[grp]}",
                    "max_stock_share_pct": round(vc.iloc[0] / len(d) * 100, 1),
                    "top5_share_pct": round(vc.head(5).sum() / len(d) * 100, 1),
                    "top5_names": [f"{s}:{int(c)}" for s, c in vc.head(5).items()]}
    return out


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def lock() -> None:
    LOCK_V2.mkdir(parents=True, exist_ok=True)
    m = load_book()
    picks = build_picks(m)
    summ = summarize(picks)
    picks_out = picks[["date", "group", "symbol", "atm_iv", "move_mag", "up_move", "down_move",
                       "closed_opp", "year"]].copy()
    picks_out["move_mag_pct"] = (picks_out.move_mag * 100).round(2)
    picks_out.to_csv(LOCK_V2 / "book_2024_26.csv", index=False)
    (LOCK_V2 / "universe_groups.json").write_text((LOCK_V1 / "universe_groups.json").read_text(), encoding="utf-8")

    manifest = {
        "version": VERSION, "supersedes": "prod_largemove_v1 (kept for reference)",
        "strategy": "direction-agnostic large-mover picker; trade via straddle/strangle or single-side + exit-at-peak",
        "selector": {"ranker": RANKER, "rule": "rank eligible by atm_iv per group",
                     "cooldown_trading_days": COOLDOWN, "per_stock_trade_cap": CAP_FRAC,
                     "picks_per_group_per_day": DEPTH},
        "eligibility": {"min_underlying": MIN_UNDERLYING, "requires_optionable": True,
                        "window_days": WINDOW},
        "groups": {g: GROUP_SIZE[g] for g in GROUP_SIZE},
        "why": ("direction unpredictable (coin flip ~0.52 AUC across ~25 academic signals incl. Pan-Poteshman, "
                "George-Hwang, Cremers-Weinbaum, order-flow); magnitude predictable and atm_iv is the best ranker "
                "(no model, incl. quarterly-retrained, beats it); t+3 cooldown de-concentrates (cap is a non-binding belt)."),
        "book_metrics_2024_26": summ,
        "metric_notes": ("move_ge6/8 = 5-day |move| (direction-agnostic, what straddle/peak-exit captures); "
                         "closed_opp = whipsaw (spiked one way, closed the other vs realized side; ex-ante side is ~50/50)."),
    }
    (LOCK_V2 / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _write_doc(manifest)

    ledger = ["path,sha256"]
    for p in [PKG / "mover_v2.py", LOCK_V2 / "universe_groups.json", LOCK_V2 / "manifest.json",
              LOCK_V2 / "PROD_LOCK_v2.md"]:
        if p.exists():
            ledger.append(f"{p.relative_to(ROOT).as_posix()},{_sha(p)}")
    (LOCK_V2 / "checksums_sha256.csv").write_text("\n".join(ledger) + "\n", encoding="utf-8")

    print(f"Locked {VERSION} -> {LOCK_V2}")
    for grp, s in summ.items():
        print(f"  {grp}: {s['per_yr']}/yr | move>=6% {s['move_ge6_pct']} | >=8% {s['move_ge8_pct']} | "
              f"whipsaw {s['closed_opp_pct']} | cover {s['coverage']} | top-name {s['max_stock_share_pct']}%")


def _write_doc(man: dict) -> None:
    s = man["book_metrics_2024_26"]
    lines = [
        f"# PRODUCTION LOCK — `{VERSION}` (direction-agnostic large-mover picker)", "",
        "Supersedes v1 for selection (v1 kept locked for reference). **Frozen** — experiments must clone, not edit.", "",
        "## Strategy", man["strategy"], "",
        f"**Selector:** rank eligible stocks by `{RANKER}` within each group; **t+{COOLDOWN} cooldown** + "
        f"**{int(CAP_FRAC*100)}% per-stock trade cap**; take **top-{DEPTH}/group/day**. "
        "Eligibility = optionable (atm_iv present) AND close ≥ ₹100.", "",
        "## Why (research)", man["why"], "",
        "## Book metrics (2024–26, out-of-sample by construction — atm_iv is observed, no fitting)",
        "| group | trades/yr | move ≥6% | move ≥8% | avg move | whipsaw | coverage | top-name | top-5 share |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for g in ("A_mcap30", "B_turn35"):
        d = s[g]
        lines.append(f"| {g} | {d['per_yr']} | {d['move_ge6_pct']}% | {d['move_ge8_pct']}% | {d['avg_move_pct']}% | "
                     f"{d['closed_opp_pct']}% | {d['coverage']} | {d['max_stock_share_pct']}% | {d['top5_share_pct']}% |")
    lines += ["", man["metric_notes"], "",
              "## Artifacts", "- `book_2024_26.csv` — daily picks with move outcomes",
              "- `manifest.json` · `universe_groups.json` · `checksums_sha256.csv`", ""]
    (LOCK_V2 / "PROD_LOCK_v2.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    lock()
