"""Build the store tables from the silver layer and run the data checks.

    python -m koscine4.store.build            # prices, adjustments, checks
    python -m koscine4.store.build --lots     # also rebuild the lot-size table (reads every contract year)
"""
from __future__ import annotations

import argparse

from koscine4.settings import Settings, load_settings
from koscine4.store.adjustments import build_adjustments
from koscine4.store.calendar import TradingCalendar
from koscine4.store.checks import DataCheckError, issues_frame, run_price_checks
from koscine4.store.market_value import build_market_value
from koscine4.store.options import build_lot_sizes
from koscine4.store.prices import build_prices_raw

ISSUES_FILE = "store/check_issues.csv"


def build_store(settings: Settings | None = None, lots: bool = False, raise_on_error: bool = True) -> dict:
    settings = settings or load_settings()
    cfg = settings.section("store")
    raw = build_prices_raw(settings)
    adjustments = build_adjustments(settings, raw)
    market_value = build_market_value(settings, raw, adjustments)
    calendar = TradingCalendar.from_prices(raw, int(cfg["min_symbols_for_session"]))
    issues = run_price_checks(raw, adjustments, calendar, cfg, raise_on_error=False)
    out = settings.path("records") / ISSUES_FILE
    out.parent.mkdir(parents=True, exist_ok=True)
    issues_frame(issues).to_csv(out, index=False)
    summary = {
        "price_rows": len(raw),
        "instruments": int(raw["instrument"].nunique()),
        "sessions": len(calendar),
        "first_session": str(calendar.days[0].date()),
        "last_session": str(calendar.days[-1].date()),
        "adjustments_applied": int(adjustments["applied"].sum()),
        "adjustments_by_kind": adjustments.groupby("kind").size().to_dict(),
        "market_value_rows": len(market_value),
        "market_value_by_method": market_value["method"].value_counts().to_dict() if len(market_value) else {},
        "check_errors": sum(i.severity == "error" for i in issues),
        "check_reviews": sum(i.severity == "review" for i in issues),
    }
    if lots:
        summary["lot_rows"] = len(build_lot_sizes(settings))
    errors = [i for i in issues if i.severity == "error"]
    if errors and raise_on_error:
        raise DataCheckError(errors)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lots", action="store_true", help="also rebuild the lot-size table")
    parser.add_argument("--no-fail", action="store_true", help="report check errors without stopping")
    args = parser.parse_args()
    for key, value in build_store(lots=args.lots, raise_on_error=not args.no_fail).items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
