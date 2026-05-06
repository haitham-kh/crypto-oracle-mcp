"""
features_v4.py - Expanded feature engine for v4.

Builds on features_v3 (29 cols) and adds features grouped by what kind of
edge they target:

  Microstructure (multi-window OFI/CVD)  ........ 8
  Volume profile / liquidity ................... 4
  Realized vol regime .......................... 5
  Range expansion / compression ................ 4
  Time of day / session ........................ 4
  BTC interaction .............................. 4
  Multi-window momentum z-scores ............... 4

Total v4 = 29 (v3 kept) + 33 (new) = 62 features.

All features are causal (use only data ≤ index i). Every feature is
clipped to a bounded range so the GBDTs see well-behaved inputs.
"""
from __future__ import annotations
import numpy as np

from features_v3 import (
    FEATURE_NAMES_V3, build_v3_from_v2, _ema, _rsi, _roll_mean, _roll_std,
)

# ── New v4 feature names (33) ───────────────────────────────────────────────
FEATURE_NAMES_V4_EXTRA = [
    # Microstructure (8)
    "ofi_z_5m", "ofi_z_15m", "ofi_z_1h", "ofi_z_4h", "ofi_z_1d",
    "cvd_z_1h", "cvd_z_4h", "taker_buy_ratio_1h",
    # Volume profile / liquidity (4)
    "poc_distance_1d", "vol_concentration_1d", "vol_zscore_5m", "vol_burst_1h",
    # Realized vol regime (5)
    "rv_5m", "rv_1h", "rv_ratio_short_long", "rv_zscore_1d", "vol_of_vol_1d",
    # Range expansion / compression (4)
    "nr7_flag", "inside_bar_streak", "range_expansion_ratio", "atr_ratio_short_long",
    # Time / session (4)
    "session_asia", "session_eu", "session_us", "is_weekend",
    # BTC interaction (4)
    "btc_rv_1h", "btc_return_1h", "btc_return_4h", "rolling_beta_btc_1d",
    # Multi-window momentum z-scores (4)
    "ret_z_15m", "ret_z_1h", "ret_z_4h", "ret_z_1d",
]
FEATURE_NAMES_V4 = FEATURE_NAMES_V3 + FEATURE_NAMES_V4_EXTRA


# ── Causal helpers ──────────────────────────────────────────────────────────

def _causal_zscore(x, window):
    """Rolling z-score of x using the trailing `window` values (causal)."""
    n = len(x)
    out = np.full(n, np.nan)
    if n < window:
        return out
    m = _roll_mean(x, window)
    s = _roll_std(x, window)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(s > 1e-12, (x - m) / s, 0.0)
    out = np.nan_to_num(np.clip(out, -5, 5), nan=0.0)
    return out


def _rolling_log_return(close, window):
    n = len(close)
    out = np.zeros(n)
    valid = (close[:-window] > 0) & (close[window:] > 0) if n > window else None
    if n > window and valid is not None:
        with np.errstate(divide="ignore", invalid="ignore"):
            out[window:] = np.log(close[window:] / np.maximum(close[:-window], 1e-12))
    return out


def _rolling_realized_vol(close, window):
    """Annualized-style realized vol over `window` 1-min bars (in % units).
    σ = std(log returns) * sqrt(window)."""
    n = len(close)
    out = np.zeros(n)
    if n < window + 1:
        return out
    rets = np.zeros(n)
    rets[1:] = np.log(np.maximum(close[1:], 1e-12) / np.maximum(close[:-1], 1e-12))
    s = _roll_std(rets, window)
    out[:] = np.nan_to_num(s * np.sqrt(max(window, 1)), nan=0.0)
    return out


def _rolling_beta(asset_ret, btc_ret, window):
    """Causal rolling beta of asset returns vs btc returns over `window`."""
    n = len(asset_ret)
    out = np.zeros(n)
    if n < window + 1:
        return out
    a = np.nan_to_num(asset_ret, nan=0.0)
    b = np.nan_to_num(btc_ret, nan=0.0)
    a_mean = _roll_mean(a, window)
    b_mean = _roll_mean(b, window)
    ab = _roll_mean(a * b, window)
    bb = _roll_mean(b * b, window)
    with np.errstate(divide="ignore", invalid="ignore"):
        cov = ab - a_mean * b_mean
        var_b = np.maximum(bb - b_mean ** 2, 1e-12)
        beta = cov / var_b
    out[:] = np.nan_to_num(np.clip(beta, -3, 3), nan=0.0)
    return out


def _volume_profile_poc(volume, close, window, stride=30):
    """Approximate VPVR: bin trailing `window` bars of volume by close price.
    Returns (poc_distance, concentration) ∈ [-1, 1] each.

    Speed-critical: uses np.bincount instead of Python inner loop, and
    only re-computes every `stride` bars (forward-fills between). Volume
    profile is slow-moving (1-day window) so this is a safe approximation
    that turns an O(n*window) Python double-loop into O(n/stride * window).
    """
    n = len(close); n_bins = 24
    poc_d = np.zeros(n)
    conc  = np.zeros(n)
    if n < window:
        return poc_d, conc

    last_poc_d = 0.0
    last_conc  = 0.0
    for i in range(window, n):
        if (i - window) % stride == 0:
            cl_w = close[i - window:i]
            v_w  = volume[i - window:i]
            lo = float(cl_w.min()); hi = float(cl_w.max())
            if hi > lo:
                edges = np.linspace(lo, hi, n_bins + 1)
                bins = np.clip(np.searchsorted(edges, cl_w, side="right") - 1,
                               0, n_bins - 1)
                # Vectorised bin accumulation (was a Python loop before)
                bin_vol = np.bincount(bins, weights=v_w, minlength=n_bins)
                total = bin_vol.sum()
                if total > 0:
                    poc_idx = int(np.argmax(bin_vol))
                    poc_price = (edges[poc_idx] + edges[poc_idx + 1]) / 2
                    last_poc_d = float(np.clip((close[i] - poc_price) /
                                               max(close[i], 1e-12), -0.5, 0.5)) * 2
                    top3 = float(np.partition(bin_vol, -3)[-3:].sum())
                    last_conc = float(top3 / total)
        poc_d[i] = last_poc_d
        conc[i]  = last_conc
    return poc_d, conc


def _nr7(high, low):
    """1 if today's range is the narrowest in last 7 bars (vectorised)."""
    n = len(high); out = np.zeros(n)
    if n < 7: return out
    rng = high - low
    # rolling min over previous 7 bars (exclusive of current)
    try:
        from numpy.lib.stride_tricks import sliding_window_view
        win = sliding_window_view(rng[:-1], 7)        # shape (n-7, 7)
        prev_min = win.min(axis=1)                    # shape (n-7,)
        out[7:] = (rng[7:] <= prev_min[:n-7]).astype(np.float64)
    except Exception:
        for i in range(7, n):
            out[i] = 1.0 if rng[i] <= rng[i - 7:i].min() else 0.0
    return out


def _inside_bar_streak(high, low):
    """Streak length of consecutive inside bars (vectorised cumcount)."""
    n = len(high); out = np.zeros(n)
    if n < 2: return out
    inside = np.zeros(n, dtype=bool)
    inside[1:] = (high[1:] <= high[:-1]) & (low[1:] >= low[:-1])
    # cumulative count that resets on False (idiomatic numpy)
    streak = np.zeros(n, dtype=np.int64)
    cum = 0
    for i in range(n):                       # tight scalar loop, ~3s for 1.7M
        cum = cum + 1 if inside[i] else 0
        streak[i] = cum
    out[:] = np.minimum(streak, 10) / 10.0
    return out


# ── Main builder ────────────────────────────────────────────────────────────

def build_v4_extras(close, high, low, volume, ofi, taker_buy_volume,
                    timestamps_ms, btc_close=None):
    """Compute the 33 v4-extra features. Returns (n, 33) array.

    Inputs are 1-minute resolution numpy arrays of equal length n.
    `timestamps_ms` is float ms; used for hour-of-day and weekend flags.
    `btc_close` (same length) gives BTC-aligned closes for beta/return
    interactions; if None, BTC features are zeroed.
    """
    n = len(close)
    F = np.zeros((n, len(FEATURE_NAMES_V4_EXTRA)), dtype=np.float64)

    # log returns at 1m for downstream rolling stats
    log_ret = np.zeros(n)
    log_ret[1:] = np.log(np.maximum(close[1:], 1e-12) / np.maximum(close[:-1], 1e-12))

    # ── Microstructure ──────────────────────────────────────────────────────
    # OFI z-scores at multiple windows
    F[:, 0] = _causal_zscore(ofi, 5)
    F[:, 1] = _causal_zscore(ofi, 15)
    F[:, 2] = _causal_zscore(ofi, 60)
    F[:, 3] = _causal_zscore(ofi, 240)
    F[:, 4] = _causal_zscore(ofi, 1440)

    # CVD = cumulative OFI; z-score its rate-of-change
    cvd = np.cumsum(np.nan_to_num(ofi, nan=0.0))
    cvd_d_1h  = np.zeros(n); cvd_d_1h[60:]   = cvd[60:]   - cvd[:-60]
    cvd_d_4h  = np.zeros(n); cvd_d_4h[240:]  = cvd[240:]  - cvd[:-240]
    F[:, 5] = _causal_zscore(cvd_d_1h, 1440)
    F[:, 6] = _causal_zscore(cvd_d_4h, 1440)

    # Taker-buy share over 1h: ratio of buy vol to total vol
    sum_v_1h = _roll_mean(volume, 60) * 60
    sum_b_1h = _roll_mean(taker_buy_volume, 60) * 60
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(sum_v_1h > 1e-9, sum_b_1h / sum_v_1h, 0.5)
    F[:, 7] = np.nan_to_num(np.clip((ratio - 0.5) * 2, -1, 1), nan=0.0)

    # ── Volume profile / liquidity ──────────────────────────────────────────
    poc_d, conc = _volume_profile_poc(volume, close, window=1440)  # 1-day VPVR
    F[:, 8]  = np.nan_to_num(poc_d, nan=0.0)
    F[:, 9]  = np.nan_to_num(np.clip(conc * 2 - 1, -1, 1), nan=0.0)
    F[:, 10] = _causal_zscore(volume, 5)
    # Volume burst: ratio of last-60-min volume to its 1-day mean
    vol_1h = _roll_mean(volume, 60)
    vol_1d = _roll_mean(volume, 1440)
    with np.errstate(divide="ignore", invalid="ignore"):
        burst = np.where(vol_1d > 1e-9, vol_1h / vol_1d, 1.0)
    F[:, 11] = np.nan_to_num(np.clip(np.log(np.maximum(burst, 1e-6)), -2, 2) / 2, nan=0.0)

    # ── Realized volatility regime ──────────────────────────────────────────
    rv_5m  = _rolling_realized_vol(close, 5)
    rv_1h  = _rolling_realized_vol(close, 60)
    rv_4h  = _rolling_realized_vol(close, 240)
    rv_1d  = _rolling_realized_vol(close, 1440)
    F[:, 12] = np.nan_to_num(np.clip(rv_5m * 100, 0, 5) / 5, nan=0.0)   # in % units
    F[:, 13] = np.nan_to_num(np.clip(rv_1h * 100, 0, 5) / 5, nan=0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        rv_ratio = np.where(rv_4h > 1e-9, rv_1h / rv_4h, 1.0)
    F[:, 14] = np.nan_to_num(np.clip(np.log(np.maximum(rv_ratio, 1e-6)), -1.5, 1.5) / 1.5, nan=0.0)
    F[:, 15] = _causal_zscore(rv_1h, 1440)
    # Vol of vol: rolling std of rv_1h
    vov = _roll_std(rv_1h, 1440)
    F[:, 16] = np.nan_to_num(np.clip(vov * 100, 0, 2) / 2, nan=0.0)

    # ── Range expansion / compression ───────────────────────────────────────
    F[:, 17] = _nr7(high, low)
    F[:, 18] = _inside_bar_streak(high, low)
    rng = high - low
    rng_short = _roll_mean(rng, 60)
    rng_long  = _roll_mean(rng, 1440)
    with np.errstate(divide="ignore", invalid="ignore"):
        rng_ratio = np.where(rng_long > 1e-9, rng_short / rng_long, 1.0)
    F[:, 19] = np.nan_to_num(np.clip(np.log(np.maximum(rng_ratio, 1e-6)), -2, 2) / 2, nan=0.0)
    # ATR ratio short vs long (proxy for vol regime change)
    from features_v3 import _roll_mean as _rm
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)),
                                           np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr_short = _rm(tr, 60)
    atr_long  = _rm(tr, 1440)
    with np.errstate(divide="ignore", invalid="ignore"):
        atr_ratio = np.where(atr_long > 1e-9, atr_short / atr_long, 1.0)
    F[:, 20] = np.nan_to_num(np.clip(np.log(np.maximum(atr_ratio, 1e-6)), -2, 2) / 2, nan=0.0)

    # ── Time of day / session ───────────────────────────────────────────────
    # 1m timestamps in ms; UTC hour and weekday.
    secs = (timestamps_ms / 1000).astype(np.int64)
    hour = ((secs // 3600) % 24).astype(np.int32)
    dow  = (((secs // 86400) + 4) % 7).astype(np.int32)  # 1970-01-01 was Thu (=3)
    # Sessions in UTC: Asia 0-7, EU 7-15, US 13-22, overlap allowed.
    F[:, 21] = ((hour >= 0)  & (hour < 8 )).astype(np.float64) * 2 - 1
    F[:, 22] = ((hour >= 7)  & (hour < 16)).astype(np.float64) * 2 - 1
    F[:, 23] = ((hour >= 13) & (hour < 22)).astype(np.float64) * 2 - 1
    F[:, 24] = ((dow == 5) | (dow == 6)).astype(np.float64) * 2 - 1

    # ── BTC interaction ─────────────────────────────────────────────────────
    if btc_close is not None and len(btc_close) == n:
        btc_logret = np.zeros(n)
        btc_logret[1:] = np.log(np.maximum(btc_close[1:], 1e-12) /
                                 np.maximum(btc_close[:-1], 1e-12))
        btc_rv_1h = _rolling_realized_vol(btc_close, 60)
        F[:, 25] = np.nan_to_num(np.clip(btc_rv_1h * 100, 0, 3) / 3, nan=0.0)
        # 1h / 4h cumulative log returns of BTC
        btc_r1h = np.zeros(n); btc_r1h[60:]  = np.log(np.maximum(btc_close[60:], 1e-12) /
                                                      np.maximum(btc_close[:-60], 1e-12))
        btc_r4h = np.zeros(n); btc_r4h[240:] = np.log(np.maximum(btc_close[240:], 1e-12) /
                                                      np.maximum(btc_close[:-240], 1e-12))
        F[:, 26] = np.nan_to_num(np.clip(btc_r1h * 100, -10, 10) / 10, nan=0.0)
        F[:, 27] = np.nan_to_num(np.clip(btc_r4h * 100, -15, 15) / 15, nan=0.0)
        # 1-day rolling beta of asset vs BTC (1m returns)
        F[:, 28] = _rolling_beta(log_ret, btc_logret, 1440) / 3.0  # already clipped to [-3,3]

    # ── Multi-window momentum z-scores ──────────────────────────────────────
    r_15m = _rolling_log_return(close, 15)
    r_1h  = _rolling_log_return(close, 60)
    r_4h  = _rolling_log_return(close, 240)
    r_1d  = _rolling_log_return(close, 1440)
    F[:, 29] = _causal_zscore(r_15m, 1440)
    F[:, 30] = _causal_zscore(r_1h,  1440)
    F[:, 31] = _causal_zscore(r_4h,  1440)
    F[:, 32] = _causal_zscore(r_1d,  1440)

    return F


def build_v4_from_components(F_v2, close, high, low, volume, ofi,
                             taker_buy_volume, timestamps_ms,
                             atr_arr, btc_close=None):
    """Build the full v4 (n, 62) feature matrix from raw inputs.

    Reuses the v3 pipeline (29 cols) and appends 33 new cols.
    """
    F_v3 = build_v3_from_v2(F_v2, close, high, low, atr_arr)             # (n, 29)
    F_extras = build_v4_extras(close, high, low, volume, ofi,
                               taker_buy_volume, timestamps_ms, btc_close)  # (n, 33)
    return np.hstack([F_v3, F_extras])
