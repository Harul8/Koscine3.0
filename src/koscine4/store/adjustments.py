"""Corporate-action adjustment factors (requirement D-1).

Four sources, reconciled against the price series itself:

1. The exchange's own corporate-action file, downloaded by `koscine4.sources.exchange_reports`.
   It gives a ratio for splits and bonuses, and names demergers and schemes without a ratio.
2. Declared splits and bonuses from the vendor files in `raw/corp_actions_yf`.
3. Declared splits and bonuses from the silver `corp_actions` table.
4. The opening gap on each day: open / previous close.

A declared event is used only when the price gap near its date confirms it; the factor is then
the exact declared one. A large gap with no declared event (a demerger, or a split missing from
both sources) is inferred from the gap and listed for review. Every event carries the date it
took effect, so history can be adjusted as of any date without using later information.
"""
from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from koscine4.settings import Settings
from koscine4.sources.exchange_reports import read_reports
from koscine4.store.symbols import canonical_frame, load_changes
from koscine4.store.tables import ADJUSTMENT_COLUMNS, ADJUSTMENTS, write_table

# Share-count multiples a split or bonus usually takes; used to recognise an undeclared one.
CLEAN_RATIOS = (1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 10.0, 20.0)
OVERRIDES_FILE = Path("configs") / "adjustment_overrides.toml"
REVIEW_FILE = Path("store") / "adjustments_review.csv"

_BONUS = re.compile(r"bon(?:us)?[^0-9:]{0,3}(\d+)\s*:\s*(\d+)", re.IGNORECASE)
# "split", "splt", "spl" (but not "spl div", a special dividend), "sub-division", "consolidation"
_SPLIT_WORD = re.compile(r"split|splt|spl(?![a-z]*\.?\s*div)|sub[- ]?div|consol", re.IGNORECASE)
# the two face values: "RS.10 TO RE.1", "RS10TORS5", "FRM 10 TO 1", "Rs.10/- To Rs.2/-"
_FROM_TO = re.compile(r"(\d+(?:\.\d+)?)\s*(?:/-)?\s*(?:per\s*sh(?:are)?\.?\s*)?TO\s*(?:R[SE]\.?\s*)?(\d+(?:\.\d+)?)", re.IGNORECASE)
# A value transfer the exchange names but gives no ratio for; the factor comes from the opening gap.
_NAMED = (
    ("demerger_declared", ("demerg", "de-merg", "scheme of arr", "arrangement", "spin off", "spin-off",
                           "capital reduc", "reduction of cap", "hive off", "hiving off")),
    ("rights_declared", ("rights", "right issue", "rts ")),
    ("special_dividend_declared", ("spl div", "spl.div", "special div", "spcl div")),
)


def parse_action(action_type: str, amount: float | None = None) -> tuple[float | None, str]:
    """(share-count multiple, what) for a corporate-action description; (None, "") if it is not a split or bonus.

    what is "bonus", "split" or "both".
    "Bonus 1:1" -> 2.0.  "Face Value Split From Rs.10/- To Rs.2/-" -> 5.0.  "BON 1:1/SPLIT RS.10TORS.5" -> 4.0.
    A bare "SPLIT" row carries the multiple in `amount`.
    """
    text = str(action_type or "")
    bonus = split = None
    m = _BONUS.search(text)
    if m and int(m.group(2)) > 0:
        bonus = (int(m.group(1)) + int(m.group(2))) / int(m.group(2))
    w = _SPLIT_WORD.search(text)
    if w:
        ft = _FROM_TO.search(text[w.start():])
        if ft and float(ft.group(1)) > 0 and float(ft.group(2)) > 0:
            split = float(ft.group(1)) / float(ft.group(2))
        elif bonus is None and amount is not None and np.isfinite(amount) and amount > 0:
            split = float(amount)
    if bonus and split:
        return bonus * split, "both"
    if bonus:
        return bonus, "bonus"
    if split:
        return split, "split"
    return None, ""


def parse_action_ratio(action_type: str, amount: float | None = None) -> float | None:
    return parse_action(action_type, amount)[0]


def named_kind(purpose: str) -> str | None:
    """The kind of a value transfer the exchange names without a ratio, or None."""
    low = str(purpose or "").lower()
    for kind, words in _NAMED:
        if any(w in low for w in words):
            return kind
    return None


def is_no_ratio_event(purpose: str) -> bool:
    return named_kind(purpose) is not None


DECLARED_COLUMNS = ["instrument", "ex_date", "ratio", "source", "what"]


def load_declared_exchange(settings: Settings) -> tuple[pd.DataFrame, pd.DataFrame]:
    """From the exchange's corporate-action file: (splits and bonuses with a ratio, events named without one)."""
    ca = read_reports(settings, "corp_actions")
    if ca.empty:
        return pd.DataFrame(columns=DECLARED_COLUMNS), pd.DataFrame(columns=["instrument", "ex_date", "purpose"])
    ca = ca[ca["series"].isin(["EQ", "BE", "BZ"]) & ca["ex_date"].notna()]
    ca = ca.drop_duplicates(["symbol", "ex_date", "purpose"])
    with_ratio, without = [], []
    for r in ca.itertuples(index=False):
        ratio, what = parse_action(r.purpose)
        if ratio and abs(ratio - 1.0) > 0.005:
            with_ratio.append({"instrument": r.symbol, "ex_date": pd.Timestamp(r.ex_date).normalize(), "ratio": float(ratio), "source": "exchange", "what": what})
        elif ratio is None and is_no_ratio_event(r.purpose):
            without.append({"instrument": r.symbol, "ex_date": pd.Timestamp(r.ex_date).normalize(), "purpose": str(r.purpose).strip()})
    named = pd.DataFrame(without, columns=["instrument", "ex_date", "purpose"]).drop_duplicates(["instrument", "ex_date"])
    return pd.DataFrame(with_ratio, columns=DECLARED_COLUMNS).drop_duplicates(), named


def load_declared_vendor(raw_dir: Path) -> pd.DataFrame:
    """Splits and bonuses from `raw/corp_actions_yf/yf_<SYMBOL>.json` (fields: Date, Stock Splits)."""
    rows = []
    folder = Path(raw_dir) / "corp_actions_yf"
    if folder.exists():
        for fp in sorted(folder.glob("yf_*.json")):
            try:
                items = json.loads(fp.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            symbol = fp.stem[3:].upper()
            for it in items if isinstance(items, list) else []:
                ratio = it.get("Stock Splits")
                date = str(it.get("Date", ""))[:10]
                if ratio and float(ratio) > 0 and float(ratio) != 1.0 and date:
                    rows.append({"instrument": symbol, "ex_date": pd.Timestamp(date), "ratio": float(ratio), "source": "vendor", "what": "both"})
    return pd.DataFrame(rows, columns=DECLARED_COLUMNS)


def load_declared_silver(silver_dir: Path) -> pd.DataFrame:
    """Splits and bonuses parsed from the silver `corp_actions` table."""
    path = Path(silver_dir) / "corp_actions.parquet"
    cols = DECLARED_COLUMNS
    if not path.exists():
        return pd.DataFrame(columns=cols)
    ca = pd.read_parquet(path)
    rows = []
    for r in ca.itertuples(index=False):
        ratio, what = parse_action(r.action_type, getattr(r, "amount", None))
        if ratio and abs(ratio - 1.0) > 0.005 and pd.notna(r.ex_date):
            rows.append({"instrument": str(r.symbol).upper(), "ex_date": pd.Timestamp(r.ex_date).normalize(), "ratio": float(ratio), "source": "silver", "what": what})
    return pd.DataFrame(rows, columns=cols)


def opening_gaps(prices_raw: pd.DataFrame) -> pd.DataFrame:
    """Two gaps for every stock row.

    gap        open / the exchange's previous close. Valid even when earlier rows are missing.
    break_gap  the exchange's previous close / the close of the last row we hold. It is 1 when
               no session is missing in between; otherwise it is the whole move across the break,
               which may contain a corporate action we cannot see.
    """
    p = prices_raw[prices_raw["kind"] == "stock"][["instrument", "date", "open", "close", "prev_close"]].sort_values(["instrument", "date"])
    g = p.groupby("instrument", sort=False)
    p["held_close"] = g["close"].shift(1)
    p["days_since"] = (p["date"] - g["date"].shift(1)).dt.days
    p["gap"] = p["open"] / p["prev_close"]
    p["break_gap"] = (p["prev_close"] / p["held_close"]).fillna(1.0)
    p["close_gap"] = p["close"] / p["prev_close"]
    p = p.rename(columns={"prev_close": "last_close"})
    return p[(p["open"] > 0) & (p["last_close"] > 0)][["instrument", "date", "gap", "break_gap", "last_close", "held_close", "days_since", "close_gap"]]


def reconcile(declared: pd.DataFrame, gaps: pd.DataFrame, cfg: Mapping[str, float],
              named: pd.DataFrame | None = None) -> pd.DataFrame:
    """Turn declared events and observed gaps into one table of adjustment events.

    `named` holds value transfers the exchange announced without a ratio (demergers, rights issues,
    special dividends). A large gap beside one is labelled with it; a smaller gap on its ex-date is
    applied when the open is at least `demerger_min_gap` below the previous close.
    """
    down, up = float(cfg["gap_down_ratio"]), float(cfg["gap_up_ratio"])
    window = pd.Timedelta(days=int(cfg["declared_search_days"]))
    tol, clean_tol = float(cfg["declared_match_tolerance"]), float(cfg["clean_ratio_tolerance"])
    loose = float(cfg.get("declared_loose_tolerance", 0.25))
    min_price = float(cfg["min_price_for_inferred"])
    max_break = int(cfg["max_break_days_for_inferred"])
    restore_tol = float(cfg["restore_tolerance"])
    named = named if named is not None else pd.DataFrame(columns=["instrument", "ex_date", "purpose"])
    named_by = {k: v for k, v in named.groupby("instrument", sort=False)} if len(named) else {}
    named_used: set[tuple[str, pd.Timestamp]] = set()

    def named_near(instrument, date, back_days: int = 0):
        """The purpose of a named event near `date` (or inside the break that ended on it), else None."""
        ev = named_by.get(instrument)
        if ev is None:
            return None
        lo = date - max(window, pd.Timedelta(days=back_days))
        hit = ev[(ev["ex_date"] >= lo) & (ev["ex_date"] <= date + window)]
        if hit.empty:
            return None
        row = hit.iloc[(hit["ex_date"] - date).abs().argsort().iloc[0]]
        named_used.add((instrument, row["ex_date"]))
        return str(row["purpose"])

    if "close_gap" not in gaps.columns:
        gaps = gaps.assign(close_gap=np.nan)
    settle_tol = float(cfg.get("named_open_close_tolerance", 0.15))

    def named_factor(open_gap: float, close_gap: float) -> tuple[float, str]:
        """Factor for a named event. On some demerger days the first print is stale and the price is found
        later in the session; when the close disagrees with the open this much, the close is the better measure."""
        if close_gap == close_gap and close_gap > 0 and abs(close_gap / open_gap - 1.0) > settle_tol and close_gap < open_gap:
            return float(close_gap), "factor taken from the close, because the price was found after the open"
        return float(open_gap), "factor taken from the opening gap"

    by_inst = {k: v.set_index("date") for k, v in gaps.groupby("instrument", sort=False)}
    used: set[tuple[str, pd.Timestamp]] = set()
    out: list[dict] = []

    if len(declared):
        declared = declared.copy()
        if "what" not in declared.columns:
            declared["what"] = "both"
        declared["ratio"] = declared["ratio"].round(6)
        declared = declared.drop_duplicates(["instrument", "ex_date", "ratio", "source", "what"]).assign(combined=False)
        # a bonus and a split announced separately for the same day act as one event: try their product first
        pairs = []
        for (inst, ex, src), g in declared.groupby(["instrument", "ex_date", "source"], sort=False):
            bon, spl = g[g["what"] == "bonus"]["ratio"].unique(), g[g["what"] == "split"]["ratio"].unique()
            pairs += [{"instrument": inst, "ex_date": ex, "ratio": round(float(x * y), 6), "source": src, "what": "both", "combined": True}
                      for x in bon for y in spl]
        if pairs:
            declared = pd.concat([declared, pd.DataFrame(pairs)], ignore_index=True)
        declared = declared.sort_values(["instrument", "ex_date", "combined", "source"], ascending=[True, True, False, True])

    def allowed_error(factor: float) -> float:
        # the gap must be nearer the declared factor than to no change at all, and never further than `tol`
        return min(tol, 0.5 * abs(1.0 - factor) / min(factor, 1.0))

    unmatched = []
    for ev in declared.itertuples(index=False):
        g = by_inst.get(ev.instrument)
        if g is None:
            continue
        near = g[(g.index >= ev.ex_date - window) & (g.index <= ev.ex_date + window)]
        if near.empty:
            continue                                   # outside the price history we hold
        factor = 1.0 / ev.ratio
        err = (near["gap"] / factor - 1.0).abs()
        best = err.idxmin()
        if err.loc[best] <= allowed_error(factor):
            if (ev.instrument, best) in used:
                continue                               # same event already taken from another source
            used.add((ev.instrument, best))
            out.append({"instrument": ev.instrument, "ex_date": best, "factor": factor, "kind": "split_or_bonus",
                        "source": f"declared:{ev.source}", "observed_gap": float(near.loc[best, "gap"]),
                        "declared_ratio": ev.ratio, "applied": True, "note": ""})
        else:
            unmatched.append((ev, factor, float(near.loc[best, "gap"])))

    def declared_for(instrument, date, gap, back_days: int = 0, limit: float = loose):
        """A declared split or bonus, not yet matched, whose ratio is within `limit` of an observed gap."""
        lo = date - max(window, pd.Timedelta(days=back_days))
        best = None
        for ev, factor, _ in unmatched:
            if ev.instrument == instrument and lo <= ev.ex_date <= date + window and abs(gap / factor - 1.0) <= limit:
                if best is None or abs(gap / factor - 1.0) < abs(gap / best[1] - 1.0):
                    best = (ev, factor)
        return best

    # A large move hidden inside a stretch of missing rows: it may hold a corporate action, but
    # it also holds every ordinary move of those days, so only a short break is trusted.
    across = gaps[((gaps["break_gap"] <= down) | (gaps["break_gap"] >= up)) & ((gaps["break_gap"] - 1.0).abs() > restore_tol)]
    for r in across.itertuples(index=False):
        days = int(r.days_since) if pd.notna(r.days_since) else -1
        gap = float(r.break_gap)
        hit = declared_for(r.instrument, r.date, gap, back_days=max(days, 0), limit=tol) if days >= 0 else None
        if hit:
            ev, factor = hit
            used.add((r.instrument, r.date))
            out.append({"instrument": r.instrument, "ex_date": r.date, "factor": factor, "kind": "split_or_bonus",
                        "source": f"declared:{ev.source}", "observed_gap": gap, "declared_ratio": ev.ratio, "applied": True,
                        "note": f"declared for {ev.ex_date.date()}; seen across a {days}-day break in the data"})
            continue
        ok = 0 <= days <= max_break and r.held_close >= min_price and gap <= down
        purpose = named_near(r.instrument, r.date, back_days=max(days, 0)) if ok else None
        out.append({"instrument": r.instrument, "ex_date": r.date, "factor": gap,
                    "kind": (named_kind(purpose) if purpose else "inferred_across_break") if ok else "unexplained_gap",
                    "source": "declared:exchange+price" if purpose else "price",
                    "observed_gap": gap, "declared_ratio": np.nan, "applied": bool(ok),
                    "note": (f"exchange: {purpose}; " if purpose else "") + f"move across a {days}-day break in the data" + ("" if ok else "; not applied")})

    # an instrument's first row is its listing day: the "previous close" is the issue price, not a trade
    big = gaps[((gaps["gap"] <= down) | (gaps["gap"] >= up)) & gaps["held_close"].notna()]
    for r in big.itertuples(index=False):
        if (r.instrument, r.date) in used:
            continue
        gap = float(r.gap)
        hit = declared_for(r.instrument, r.date, gap)
        if hit:                                        # the exchange gives the ratio; the rest of the gap is the day's move
            ev, factor = hit
            used.add((r.instrument, r.date))
            out.append({"instrument": r.instrument, "ex_date": r.date, "factor": factor, "kind": "split_or_bonus",
                        "source": f"declared:{ev.source}", "observed_gap": gap, "declared_ratio": ev.ratio, "applied": True,
                        "note": f"opening gap differs from the declared ratio by {abs(gap / factor - 1.0) * 100:.0f}%"})
            continue
        if r.last_close < min_price:
            out.append({"instrument": r.instrument, "ex_date": r.date, "factor": gap, "kind": "unexplained_gap",
                        "source": "price", "observed_gap": gap, "declared_ratio": np.nan, "applied": False,
                        "note": "previous close below the minimum price"})
            continue
        purpose = named_near(r.instrument, r.date)
        if purpose:
            factor, how = named_factor(gap, float(r.close_gap))
            out.append({"instrument": r.instrument, "ex_date": r.date, "factor": factor, "kind": named_kind(purpose),
                        "source": "declared:exchange+price", "observed_gap": gap, "declared_ratio": np.nan, "applied": True,
                        "note": f"exchange: {purpose}; {how}"})
            continue
        snapped = next((1.0 / c for c in CLEAN_RATIOS if abs(gap * c - 1.0) <= clean_tol), None)
        out.append({"instrument": r.instrument, "ex_date": r.date,
                    "factor": snapped if snapped is not None else gap,
                    "kind": "inferred_split" if snapped is not None else "inferred_other",
                    "source": "price", "observed_gap": gap, "declared_ratio": np.nan, "applied": True,
                    "note": "no declared event; inferred from the opening gap"})

    # declared with a ratio, and nothing in the prices near the date
    taken = {(o["instrument"], pd.Timestamp(o["ex_date"])) for o in out if o["applied"]}
    listed: set[tuple[str, pd.Timestamp]] = set()
    for ev, factor, seen_gap in unmatched:
        if any(k[0] == ev.instrument and abs((k[1] - ev.ex_date).days) <= window.days for k in taken):
            continue
        if (ev.instrument, ev.ex_date) in listed:
            continue
        listed.add((ev.instrument, ev.ex_date))
        out.append({"instrument": ev.instrument, "ex_date": ev.ex_date, "factor": factor, "kind": "declared_unconfirmed",
                    "source": f"declared:{ev.source}", "observed_gap": seen_gap, "declared_ratio": ev.ratio, "applied": False,
                    "note": "declared, but no matching price gap near the date"})

    # A named event whose gap is too small for the rules above (a demerger of a small business, a rights issue).
    min_drop = float(cfg.get("demerger_min_gap", 0.05))
    for ev in named.itertuples(index=False):
        if (ev.instrument, ev.ex_date) in named_used:
            continue
        g = by_inst.get(ev.instrument)
        if g is None or ev.ex_date not in g.index:
            continue
        if any(k[0] == ev.instrument and abs((k[1] - ev.ex_date).days) <= window.days for k in taken):
            continue
        row = g.loc[ev.ex_date]
        gap = float(row["gap"])
        factor, how = named_factor(gap, float(row["close_gap"]))
        apply = factor <= 1.0 - min_drop and float(row["last_close"]) >= min_price
        out.append({"instrument": ev.instrument, "ex_date": ev.ex_date, "factor": factor if apply else gap,
                    "kind": named_kind(ev.purpose) if apply else "declared_no_gap",
                    "source": "declared:exchange+price" if apply else "declared:exchange",
                    "observed_gap": gap, "declared_ratio": np.nan, "applied": bool(apply),
                    "note": f"exchange: {ev.purpose}" + (f"; {how}" if apply else "; opening gap too small to adjust")})

    df = pd.DataFrame(out, columns=ADJUSTMENT_COLUMNS)
    df["ex_date"] = pd.to_datetime(df["ex_date"])
    df["applied"] = df["applied"].astype(bool)
    return df.sort_values(["instrument", "ex_date"]).reset_index(drop=True)


def load_overrides(root: Path) -> dict:
    path = Path(root) / OVERRIDES_FILE
    if not path.exists():
        return {"reject": [], "add": []}
    with path.open("rb") as fh:
        raw = tomllib.load(fh)
    return {"reject": list(raw.get("reject", [])), "add": list(raw.get("add", []))}


def apply_overrides(adjustments: pd.DataFrame, overrides: Mapping[str, list]) -> pd.DataFrame:
    """The owner's corrections: `reject` switches an event off, `add` supplies one the data cannot show."""
    df = adjustments.copy()
    for item in overrides.get("reject", []):
        hit = (df["instrument"] == str(item["instrument"])) & (df["ex_date"] == pd.Timestamp(item["ex_date"]))
        df.loc[hit, "applied"] = False
        df.loc[hit, "note"] = "rejected by owner: " + str(item.get("reason", ""))
    extra = []
    for item in overrides.get("add", []):
        inst, ex = str(item["instrument"]), pd.Timestamp(item["ex_date"])
        df = df[~((df["instrument"] == inst) & (df["ex_date"] == ex))]
        extra.append({"instrument": inst, "ex_date": ex, "factor": float(item["factor"]), "kind": str(item.get("kind", "owner_added")),
                      "source": "owner", "observed_gap": np.nan, "declared_ratio": np.nan, "applied": True,
                      "note": "added by owner: " + str(item.get("reason", ""))})
    if extra:
        df = pd.concat([df, pd.DataFrame(extra, columns=ADJUSTMENT_COLUMNS)], ignore_index=True)
    df["applied"] = df["applied"].astype(bool)
    return df.sort_values(["instrument", "ex_date"]).reset_index(drop=True)


def build_adjustments(settings: Settings, prices_raw: pd.DataFrame) -> pd.DataFrame:
    exchange, named = load_declared_exchange(settings)
    changes = load_changes(settings)
    sources = [exchange, load_declared_vendor(settings.path("raw")), load_declared_silver(settings.path("silver"))]
    sources = [canonical_frame(s, changes, date="ex_date") for s in sources if len(s)]
    declared = pd.concat(sources, ignore_index=True) if sources else exchange
    named = canonical_frame(named, changes, date="ex_date")
    df = reconcile(declared, opening_gaps(prices_raw), settings.section("store"), named)
    df = apply_overrides(df, load_overrides(settings.root))
    write_table(settings, ADJUSTMENTS, df)
    confirmed = (df["kind"] == "split_or_bonus") & (df["note"] == "")
    review = df[~confirmed & (df["kind"] != "declared_no_gap")]
    review_path = settings.path("records") / REVIEW_FILE
    review_path.parent.mkdir(parents=True, exist_ok=True)
    review.to_csv(review_path, index=False)
    return df
