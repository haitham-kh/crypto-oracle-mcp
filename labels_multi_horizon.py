"""
labels_multi_horizon.py - Multi-horizon, fee-aware triple-barrier labels.

For each sample bar i and each horizon h in HORIZONS, computes:

    y_h         : 1 if a real trade with the simulator's exact barriers
                  closes net-of-fees > 0 within h bars, else 0.
    net_h       : the realised net-of-fees % return at first-hit / time exit.
    ret_h       : the *raw* signed forward log-return over h bars (regression
                  target, not barrier-bound). Used by the magnitude regressor.
    sigma_h     : realised std of 1m log returns over [i+1, i+h] (× sqrt(h))
                  used as the "predicted vol" target.
    rejected_h  : True if the setup's TP could not clear MIN_TP_TO_COST_RATIO.

A row is *kept for training* if at least ONE horizon was not rejected. Rows
where every horizon is rejected give the model nothing useful and are dropped.

Why multi-horizon: at any given moment one horizon almost always dominates
(e.g. tight chop favours 60m, expansions favour 720m). Letting the model
choose lets the strategy harvest each regime's natural cadence.
"""
from __future__ import annotations
import numpy as np

from trading_config import (
    ROUND_TRIP_COST, SL_MULT, MIN_TP_TO_COST_RATIO,
    HORIZONS, horizon_atr_scale, regime_tp_mult,
)


def label_multi_horizon(close, high, low, atr_arr, hurst_arr, sample_idx,
                        horizons=HORIZONS):
    """Compute multi-horizon fee-aware labels + regression targets.

    Returns a dict whose keys are 'y_<h>', 'net_<h>', 'ret_<h>', 'sigma_<h>'
    for each h in horizons, plus 'tp_<h>', 'sl_<h>', 'reason_<h>'. Each
    value is an (k,) numpy array.
    """
    n = len(close); k = len(sample_idx)
    cost = ROUND_TRIP_COST
    INF = np.iinfo(np.int32).max

    out = {}
    for h in horizons:
        out[f"y_{h}"]     = np.full(k, np.nan, dtype=np.float64)
        out[f"net_{h}"]   = np.full(k, np.nan, dtype=np.float64)
        out[f"ret_{h}"]   = np.full(k, np.nan, dtype=np.float64)
        out[f"sigma_{h}"] = np.full(k, np.nan, dtype=np.float64)
        out[f"tp_{h}"]    = np.full(k, np.nan, dtype=np.float64)
        out[f"sl_{h}"]    = np.full(k, np.nan, dtype=np.float64)
        out[f"reason_{h}"] = np.full(k, -1, dtype=np.int8)
    out["any_valid"] = np.zeros(k, dtype=bool)

    log_close = np.log(np.maximum(close, 1e-12))

    for kk, i in enumerate(sample_idx):
        atr = atr_arr[i]
        if not np.isfinite(atr) or atr <= 0:
            continue
        entry = close[i]
        if entry <= 0:
            continue

        h_val = hurst_arr[i] if np.isfinite(hurst_arr[i]) else 0.5
        any_valid = False

        for h in horizons:
            end = min(i + h, n - 1)
            if end <= i:
                continue
            scale = horizon_atr_scale(h)
            tp_mult = regime_tp_mult(h_val) * scale
            sl_mult = SL_MULT * scale
            tp_p = tp_mult * atr / entry
            sl_p = sl_mult * atr / entry

            # ── Forward return + realized vol regression targets ─────────────
            ret_h   = float(log_close[end] - log_close[i])
            window  = log_close[i + 1:end + 1] - log_close[i:end]
            sigma_h = float(np.std(window) * np.sqrt(max(end - i, 1))) if end > i else 0.0
            out[f"ret_{h}"][kk]   = ret_h
            out[f"sigma_{h}"][kk] = sigma_h

            if tp_p < MIN_TP_TO_COST_RATIO * cost:
                # un-tradable horizon → leave classification fields NaN
                continue
            any_valid = True

            tp_price = entry + tp_mult * atr
            sl_price = entry - sl_mult * atr
            fwd_hi = high[i + 1:end + 1]
            fwd_lo = low[i + 1:end + 1]
            tp_hits = np.where(fwd_hi >= tp_price)[0]
            sl_hits = np.where(fwd_lo <= sl_price)[0]
            tp_t = tp_hits[0] if len(tp_hits) else INF
            sl_t = sl_hits[0] if len(sl_hits) else INF

            if tp_t < sl_t:
                gross = tp_p; reason = 0
            elif sl_t < tp_t:
                gross = -sl_p; reason = 1
            elif tp_t == sl_t and tp_t != INF:
                gross = -sl_p; reason = 1   # ambiguous → assume SL (worst case)
            else:
                gross = (close[end] - entry) / entry; reason = 2

            net = gross - cost
            out[f"y_{h}"][kk]     = 1.0 if net > 0 else 0.0
            out[f"net_{h}"][kk]   = net
            out[f"tp_{h}"][kk]    = tp_p
            out[f"sl_{h}"][kk]    = sl_p
            out[f"reason_{h}"][kk] = reason

        out["any_valid"][kk] = any_valid

    return out
