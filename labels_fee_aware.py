"""
labels_fee_aware.py - Triple-barrier labels that match the simulator exactly.

Label = 1  iff a trade opened at bar i would close fee-positive within the
holding horizon, using the same regime-conditioned ATR barriers and the same
round-trip cost the simulator/live executor will pay.

Returns:
    y      : (k,) float  in {0,1} or NaN if unlabelable
    tp_pct : (k,) float  TP as fraction of entry
    sl_pct : (k,) float  SL as fraction of entry
    net_pct: (k,) float  realised net % return at exit (for EV diagnostics)
    exit_reason : (k,) int   0 = TP hit, 1 = SL hit, 2 = time exit
"""
from __future__ import annotations
import numpy as np

from trading_config import (
    ROUND_TRIP_COST, ATR_HORIZON_SCALE, SL_MULT,
    FORWARD_BARS, MIN_TP_TO_COST_RATIO, regime_tp_mult,
)


def label_fee_aware(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    atr_arr: np.ndarray,
    hurst_arr: np.ndarray,
    sample_idx: np.ndarray,
    fwd: int = FORWARD_BARS,
):
    """Compute fee-aware triple-barrier labels.

    A trade is labeled 1 only when its NET-OF-FEES return at exit > 0.
    Setups whose TP cannot clear MIN_TP_TO_COST_RATIO * cost are returned
    as NaN so the model never trains on un-tradable points.
    """
    n = len(close)
    k = len(sample_idx)
    y = np.full(k, np.nan, dtype=np.float64)
    tp_pct = np.full(k, np.nan, dtype=np.float64)
    sl_pct = np.full(k, np.nan, dtype=np.float64)
    net_pct = np.full(k, np.nan, dtype=np.float64)
    exit_reason = np.full(k, -1, dtype=np.int8)

    cost = ROUND_TRIP_COST

    for kk, i in enumerate(sample_idx):
        atr = atr_arr[i]
        if not np.isfinite(atr) or atr <= 0:
            continue
        entry = close[i]
        if entry <= 0:
            continue
        end = min(i + fwd, n - 1)
        if end <= i:
            continue

        h = hurst_arr[i] if np.isfinite(hurst_arr[i]) else 0.5
        tp_mult = regime_tp_mult(h) * ATR_HORIZON_SCALE
        sl_mult = SL_MULT * ATR_HORIZON_SCALE

        tp_p = tp_mult * atr / entry
        sl_p = sl_mult * atr / entry

        # Reject setups that can't clear fees with margin -> no label.
        if tp_p < MIN_TP_TO_COST_RATIO * cost:
            continue

        tp_price = entry + tp_mult * atr
        sl_price = entry - sl_mult * atr

        fwd_hi = high[i + 1:end + 1]
        fwd_lo = low[i + 1:end + 1]

        tp_hits = np.where(fwd_hi >= tp_price)[0]
        sl_hits = np.where(fwd_lo <= sl_price)[0]
        tp_t = tp_hits[0] if len(tp_hits) else np.iinfo(np.int32).max
        sl_t = sl_hits[0] if len(sl_hits) else np.iinfo(np.int32).max

        if tp_t < sl_t:
            gross = tp_p
            reason = 0
        elif sl_t < tp_t:
            gross = -sl_p
            reason = 1
        elif tp_t == sl_t and tp_t != np.iinfo(np.int32).max:
            # Same bar, can't tell which hit first → assume worst case (SL).
            gross = -sl_p
            reason = 1
        else:
            # Time exit at end-of-horizon close
            gross = (close[end] - entry) / entry
            reason = 2

        net = gross - cost
        net_pct[kk] = net
        tp_pct[kk] = tp_p
        sl_pct[kk] = sl_p
        exit_reason[kk] = reason
        y[kk] = 1.0 if net > 0 else 0.0

    return y, tp_pct, sl_pct, net_pct, exit_reason
