"""
regime_filter.py - Causal market-regime classifier used to gate trading.

Produces one of: TREND_UP, TREND_DOWN, RANGE, EXPANSION, CHAOS, LOW_LIQUIDITY
at every bar. Uses only quantities that the simulator already computes:

    hurst              (autocorrelation of returns; > 0.55 = trending)
    trend_efficiency   (net move / sum of |moves| over a window)
    rv_1h              (1-hour realized vol)
    rv_zscore_1d       (where current 1h-vol sits in 1-day window)
    vol_burst_1h       (1h vol over 1d mean vol)
    range_position     (where price sits in 20-bar high-low range)

Decision tree (intentionally simple — interpretable & low-overfitting):

  CHAOS         : rv_zscore_1d > 2.5  AND  abs(hurst-0.5) < 0.05
                  (very high vol with no autocorr → no edge)
  LOW_LIQUIDITY : vol_burst_1h < -0.4 AND rv_1h < 0.3rd-percentile-of-train
                  (dead market, fees dominate any move)
  EXPANSION     : rv_zscore_1d > 1.5  AND  trend_efficiency > 0.30
                  (volatile but directional → favour 720m horizon)
  TREND_UP      : hurst > 0.55  AND  range_position > 0
  TREND_DOWN    : hurst > 0.55  AND  range_position < 0
  RANGE         : everything else

The regime is purely a *gate*, not a feature: the model already sees the
underlying signals. It exists to keep capital out of impossible market
states, which is what disciplined human traders do well and ML models
are bad at.
"""
from __future__ import annotations
import numpy as np


CHAOS         = "CHAOS"
LOW_LIQUIDITY = "LOW_LIQUIDITY"
EXPANSION     = "EXPANSION"
TREND_UP      = "TREND_UP"
TREND_DOWN    = "TREND_DOWN"
RANGE         = "RANGE"

ALL_REGIMES = [CHAOS, LOW_LIQUIDITY, EXPANSION, TREND_UP, TREND_DOWN, RANGE]


def classify_regime(hurst, trend_efficiency, rv_1h, rv_zscore_1d,
                    vol_burst_1h, range_position):
    """Classify a single bar's regime. All inputs are scalars."""
    h  = 0.5 if (hurst is None or hurst != hurst) else float(hurst)
    te = 0.0 if (trend_efficiency is None or trend_efficiency != trend_efficiency) else float(trend_efficiency)
    rv = 0.0 if (rv_1h is None or rv_1h != rv_1h) else float(rv_1h)
    rz = 0.0 if (rv_zscore_1d is None or rv_zscore_1d != rv_zscore_1d) else float(rv_zscore_1d)
    vb = 0.0 if (vol_burst_1h is None or vol_burst_1h != vol_burst_1h) else float(vol_burst_1h)
    rp = 0.0 if (range_position is None or range_position != range_position) else float(range_position)

    if rz > 2.5 and abs(h - 0.5) < 0.05:
        return CHAOS
    if vb < -0.4 and rv < 0.0005:        # 0.05% per hour vol → near dead
        return LOW_LIQUIDITY
    if rz > 1.5 and te > 0.30:
        return EXPANSION
    if h > 0.55:
        return TREND_UP if rp >= 0 else TREND_DOWN
    return RANGE


def classify_regime_array(hurst_arr, te_arr, rv_1h_arr, rv_z_1d_arr,
                          vol_burst_1h_arr, range_pos_arr):
    """Vectorised version. Returns an (n,) numpy array of regime strings."""
    n = len(hurst_arr)
    out = np.full(n, RANGE, dtype=object)
    for i in range(n):
        out[i] = classify_regime(
            hurst_arr[i], te_arr[i], rv_1h_arr[i], rv_z_1d_arr[i],
            vol_burst_1h_arr[i], range_pos_arr[i],
        )
    return out
