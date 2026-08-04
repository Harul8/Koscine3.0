"""Generate the PROD lock artifacts: manifest.json, PROD_LOCK.md, checksums_sha256.csv.

    python -m largemove.lock

Deterministic: metrics are recomputed from the saved predictions, no wall-clock timestamps,
so re-running on unchanged inputs reproduces identical checksums. The ledger covers the engine
code + universe + manifest + lock doc — the *definition* of PROD. Models/predictions are
reproducible outputs and are intentionally excluded from the integrity ledger.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from largemove.config import (
    PROD, VERSION, LOCK_DIR, MODELS_DIR, PREDICTIONS_DIR,
    XGB_CLF_PARAMS, XGB_REG_PARAMS,
)

ROOT = Path(__file__).resolve().parents[3]
PKG = ROOT / "src" / "koscine3" / "largemove"


def _metrics() -> dict:
    out = {}
    for group, thr in PROD.group_thresholds:
        f = PREDICTIONS_DIR / f"group_{group}_predictions.csv"
        if not f.exists():
            continue
        p = pd.read_csv(f)
        d1 = p[p.rank_in_day == 1]
        by_year = (
            d1.assign(y=pd.to_datetime(d1.date).dt.year)
            .groupby("y")["hit"].mean().mul(100).round(1).to_dict()
        )
        out[group] = {
            "threshold": thr,
            "eligible_rows": int(len(p)),
            "prec_at_1_overall": round(float(d1["hit"].mean()) * 100, 1),
            "prec_at_1_by_year": {str(k): v for k, v in by_year.items()},
        }
    return out


def _peak() -> dict | None:
    f = PREDICTIONS_DIR / "peak_kpi_summary.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None


def _manifest() -> dict:
    trained = (MODELS_DIR / "trained_through.txt")
    peak = _peak()
    return {
        "version": VERSION,
        "contract": {
            "window_days": PROD.window_days,
            "cooldown_trading_days": PROD.cooldown_trading_days,
            "min_underlying": PROD.min_underlying,
            "requires_optionable": PROD.requires_optionable,
            "label": "ceiling (max favorable move over window) >= per-group threshold",
            "entry": "t+1 open",
            "walk_forward": "base-fit < T-1, isotonic-calibrate on T-1, predict T",
            "test_years": list(PROD.test_years),
            "eval_end": PROD.eval_end,
        },
        "groups": {g: t for g, t in PROD.group_thresholds},
        "features": list(PROD.features),
        "model": {
            "type": "per (group, side): XGBoost classifier (isotonic-calibrated) -> confidence/rank "
                    "+ XGBoost regressor -> expected move",
            "clf_params": XGB_CLF_PARAMS,
            "reg_params": XGB_REG_PARAMS,
        },
        "trained_through": trained.read_text().strip() if trained.exists() else None,
        "walk_forward_metrics": _metrics(),
        "selection": "per-group daily top-2 by confidence, per-stock t+3 cooldown, no hard daily cap",
        "strategy": (peak or {}).get("strategy", "exit-at-peak (capture favorable peak via long options)"),
        "exit_at_peak_kpis": peak,
        "decisions": [
            "close_persistence_v1 experiment: re-ranking to optimize the t+5 CLOSE was REJECTED — "
            "close direction is not predictable from EOD features (OOS AUC ~0.52; close-above lift only 23->25%). "
            "Exit-at-peak retained as the objective; persistence/flow-data path spun off as a fresh experiment."
        ],
    }


def _ledger_files() -> list[Path]:
    files = sorted(PKG.glob("*.py"))
    files += [LOCK_DIR / "universe_groups.json", LOCK_DIR / "manifest.json", LOCK_DIR / "PROD_LOCK.md"]
    return files


def _sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _lock_doc(man: dict) -> str:
    g = man["groups"]
    m = man["walk_forward_metrics"]
    lines = [
        f"# PRODUCTION LOCK — `{VERSION}`",
        "",
        "This pipeline is **frozen**. Do not edit `src/koscine3/largemove/` or this lock directory.",
        "Run experiments by cloning (`python experiments/clone_prod.py <name>`); see "
        "`experiments/EXPERIMENT_POLICY.md`. Verify integrity with `python experiments/clone_prod.py --verify`.",
        "",
        "## What it does",
        "Daily ranked shortlist of stocks likely to make a large **favorable** move, traded as long options "
        "(convexity: capped downside = premium, uncapped upside).",
        "",
        "## Universe (2 groups)",
        "- **A_mcap30** — top-30 by market cap (mega-caps). Threshold "
        f"**{g['A_mcap30']*100:.0f}%**.",
        "- **B_turn35** — next-35 by turnover (the movers). Threshold "
        f"**{g['B_turn35']*100:.0f}%**.",
        "",
        "## Contract",
        f"- Entry **t+1 open**, **{man['contract']['window_days']}-day** window; label = ceiling ≥ threshold.",
        f"- Per-stock **t+{man['contract']['cooldown_trading_days']} cooldown**; eligibility = optionable "
        f"(atm_iv present) AND close ≥ ₹{man['contract']['min_underlying']:.0f} (non-penny), point-in-time.",
        f"- Walk-forward: {man['contract']['walk_forward']}; test years "
        f"{man['contract']['test_years']}, eval through {man['contract']['eval_end']}.",
        "",
        "## Model",
        "Per (group, side): isotonic-calibrated **XGBoost classifier** → confidence/rank, plus **XGBoost "
        "regressor** → expected move. 20 lean, level-dominated features (ablation-validated).",
        "",
        "## Walk-forward result (out-of-sample, top-1/day)",
        "| group | threshold | prec@1 | by year |",
        "|---|---|---|---|",
    ]
    for grp, d in m.items():
        by = ", ".join(f"{k}: {v}%" for k, v in d["prec_at_1_by_year"].items())
        lines.append(f"| {grp} | ≥{d['threshold']*100:.0f}% | **{d['prec_at_1_overall']}%** | {by} |")
    peak = man.get("exit_at_peak_kpis") or {}
    lines += [
        "",
        f"Trained through: `{man['trained_through']}`.",
        "",
        "## Strategy & official KPI — exit-at-peak",
        "Buy long options on the shortlist; **exit at the favorable peak inside the 5-day window** "
        "(downside capped at premium). The official KPI is **peak capture**, not the t+5 close.",
        "",
        "| group | peak-hit | avg peak (hits) | median day→peak | % peak by day2 | median first-cross day |",
        "|---|---|---|---|---|---|",
    ]
    bg = peak.get("by_group", {})
    for grp in man["groups"]:
        d = bg.get(grp)
        if d:
            lines.append(
                f"| {grp} | **{d['peak_hit_pct']}%** | {d['avg_peak_among_hits_pct']}% | "
                f"{d['median_days_to_peak']:.0f} | {d['pct_peak_by_day2']}% | {d['median_first_cross_day']:.0f} |"
            )
    c = peak.get("combined")
    if c:
        lines.append(
            f"| **combined** | **{c['peak_hit_pct']}%** | {c['avg_peak_among_hits_pct']}% | "
            f"{c['median_days_to_peak']:.0f} | {c['pct_peak_by_day2']}% | {c['median_first_cross_day']:.0f} |"
        )
    lines += [
        "",
        "Read: the move typically **crosses the threshold early (median ~day 2)** then **peaks ~day 4** — "
        "an early cross signal with room for the peak to develop. See `predictions/peak_capture_report.csv`.",
        "",
        "## Decisions",
    ]
    for dec in man.get("decisions", []):
        lines.append(f"- {dec}")
    lines += [
        "",
        "## Artifacts in this directory",
        "- `universe_groups.json` — locked A/B membership · `mcap_universe_dhan.csv` — market-cap source",
        "- `manifest.json` — machine-readable config + metrics snapshot",
        "- `models/` — production models (`{group}_{side}.joblib`) · `predictions/` — walk-forward OOS + shortlist + peak KPIs",
        "- `checksums_sha256.csv` — integrity ledger (engine code + config + universe + manifest + this doc)",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    LOCK_DIR.mkdir(parents=True, exist_ok=True)
    man = _manifest()
    (LOCK_DIR / "manifest.json").write_text(json.dumps(man, indent=2), encoding="utf-8")
    (LOCK_DIR / "PROD_LOCK.md").write_text(_lock_doc(man), encoding="utf-8")

    rows = ["path,sha256"]
    for p in _ledger_files():
        if p.exists():
            rows.append(f"{p.relative_to(ROOT).as_posix()},{_sha256(p)}")
    (LOCK_DIR / "checksums_sha256.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")

    print(f"Locked {VERSION}: wrote manifest.json, PROD_LOCK.md, checksums_sha256.csv ({len(rows)-1} files).")
    for grp, d in man["walk_forward_metrics"].items():
        print(f"  {grp}: prec@1 {d['prec_at_1_overall']}%  by-year {d['prec_at_1_by_year']}")


if __name__ == "__main__":
    main()
