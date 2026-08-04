"""Koscine 3.0 — Large-Move Options Engine (PRODUCTION).

Two-group daily ranked shortlist of stocks likely to make a large favorable move,
to be traded as long options. LOCKED pipeline — see locks/prod_largemove_v1/.

Group A = top-30 by market cap, target >=4% move proxy at >=3% threshold (mega-caps).
Group B = next-35 by turnover, target >=4% move (the movers).
Per (stock, side): calibrated classifier P(move>=thr) = confidence (ranker) + regressor = expected move.
Rank by confidence, per-stock t+3 cooldown, no hard daily limit.
"""
from koscine3.largemove.config import LargeMoveConfig, PROD  # noqa: F401
