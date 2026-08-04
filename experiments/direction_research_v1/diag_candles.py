"""Last attempt: practitioner price-action / candlestick patterns for direction.
Construct ~20 candle & chart-pattern signals from OHLC(V), measure their ACTUAL forward up-rate
(vs base) at h=1 and h=5, plus a multivariate XGB on all candle features (OOS). Broad universe."""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parents[1] / "src"))
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from koscine3.data.sources import load_market_data

pd.set_option("display.width", 200)


def main():
    m = load_market_data(columns=["date", "symbol", "open", "high", "low", "close", "volume"])
    m["symbol"] = m["symbol"].astype(str)
    m = m.sort_values(["symbol", "date"]).reset_index(drop=True)
    g = m.groupby("symbol", sort=False)
    for h in (1, 2, 3, 5):
        m[f"dir{h}"] = (g["close"].shift(-h) / m["close"] - 1.0 > 0).astype(float)

    rng = (m.high - m.low).replace(0, np.nan)
    o, c, h_, l_ = m.open, m.close, m.high, m.low
    m["body_frac"] = (c - o) / o
    m["body_to_range"] = (c - o).abs() / rng
    m["upper_wick"] = (h_ - np.maximum(o, c)) / rng
    m["lower_wick"] = (np.minimum(o, c) - l_) / rng
    m["clv"] = ((c - l_) - (h_ - c)) / rng
    m["range_pct"] = rng / c
    c1, o1, h1, l1 = g["close"].shift(1), g["open"].shift(1), g["high"].shift(1), g["low"].shift(1)
    c2, c3 = g["close"].shift(2), g["close"].shift(3)
    v20 = g["volume"].transform(lambda s: s.rolling(20, min_periods=10).mean())
    hh20 = g["high"].transform(lambda s: s.rolling(20, min_periods=10).max())
    ll20 = g["low"].transform(lambda s: s.rolling(20, min_periods=10).min())
    m["vol_surge"] = m.volume / v20
    green, red = c > o, c < o
    small = m.body_to_range < 0.35

    pats = {
        # bullish (expect forward up-rate > base)
        "bull_engulf": green & (c1 < o1) & (c >= o1) & (o <= c1),
        "hammer": small & (m.lower_wick > 0.55) & (m.upper_wick < 0.2),
        "gap_up": o > c1 * 1.005,
        "three_up": (c > c1) & (c1 > c2) & (c2 > c3),
        "new_high_20": c >= hh20,
        "marubozu_green": green & (m.body_to_range > 0.9),
        "close_near_high": m.clv > 0.6,
        "bull_engulf_volup": green & (c1 < o1) & (c >= o1) & (o <= c1) & (m.vol_surge > 1.5),
        # bearish (expect forward up-rate < base)
        "bear_engulf": red & (c1 > o1) & (c <= o1) & (o >= c1),
        "shooting_star": small & (m.upper_wick > 0.55) & (m.lower_wick < 0.2),
        "gap_down": o < c1 * 0.995,
        "three_down": (c < c1) & (c1 < c2) & (c2 < c3),
        "new_low_20": c <= ll20,
        "marubozu_red": red & (m.body_to_range > 0.9),
        "close_near_low": m.clv < -0.6,
    }
    m = m[(m.close >= 100) & (m.date.dt.year >= 2022)].copy()
    for k, v in pats.items():
        m[k] = v.reindex(m.index).fillna(False)

    print(f"rows={len(m)}  base P(up): h1={m.dir1.mean():.3f} h5={m.dir5.mean():.3f}\n")
    print("=" * 86)
    print("PATTERN forward up-rate (firing rows) vs base — lift = up_rate - base")
    print("=" * 86)
    rows = []
    for k in pats:
        d = m[m[k]]
        if len(d) < 300:
            continue
        rows.append({"pattern": k, "n": len(d),
                     "up1%": round(d.dir1.mean() * 100, 1), "lift1": round((d.dir1.mean() - m.dir1.mean()) * 100, 1),
                     "up5%": round(d.dir5.mean() * 100, 1), "lift5": round((d.dir5.mean() - m.dir5.mean()) * 100, 1)})
    res = pd.DataFrame(rows).sort_values("lift5", key=lambda s: s.abs(), ascending=False)
    print(res.to_string(index=False))

    # multivariate ML on candle features (OOS time-split)
    feats = ["body_frac", "body_to_range", "upper_wick", "lower_wick", "clv", "range_pct", "vol_surge"] + list(pats)
    m[feats] = m[feats].replace([np.inf, -np.inf], np.nan).astype(float)
    for tgt in ("dir1", "dir5"):
        d = m.dropna(subset=[tgt])
        tr, te = d[d.date.dt.year < 2025], d[d.date.dt.year >= 2025]
        clf = XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.03, subsample=0.8,
                            colsample_bytree=0.8, tree_method="hist", device="cuda", verbosity=0,
                            reg_lambda=5).fit(tr[feats], tr[tgt])
        p = clf.predict_proba(te[feats])[:, 1]
        te2 = te.assign(p=p, dec=pd.qcut(pd.Series(p).rank(method="first"), 10, labels=False).values)
        top, bot = te2[te2.dec == 9], te2[te2.dec == 0]
        print(f"\nML candle-only {tgt}: OOS AUC(2025-26)={roc_auc_score(te[tgt], p):.4f}  "
              f"top-decile up={top[tgt].mean()*100:.1f}%  bot-decile up={bot[tgt].mean()*100:.1f}%")


if __name__ == "__main__":
    main()
