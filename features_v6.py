"""
features_v6.py — V6 Feature Engine
====================================
Adds 36 real technical signal features on top of the V5 stack (94 features).
All features are CAUSAL (no look-ahead bias) and clipped to [-1, 1].

Feature Groups & Tier Ratings:
  Donchian Channels    (6)  — S+ tier: structural range & breakout definition
  Anchored VWAP        (3)  — S  tier: institutional fair-value anchors
  Volume Profile VAH   (4)  — S  tier: auction-market value area (70% of volume)
  Liquidity Sweeps     (5)  — S  tier: stop-hunt detection via wick rejection
  SMA Trends           (6)  — A- tier: macro directional bias stack
  Order Flow Extras    (4)  — A  tier: large-trade delta & pressure shift
  Fibonacci Levels     (5)  — B  tier: confluence detector (38.2 / 50 / 61.8 / 78.6)
  Bollinger Extras     (3)  — B  tier: squeeze/expansion volatility regime

V6 total: 94 (V5) + 36 (V6-extras) = 130 features.

Usage:
    from features_v6 import FEATURE_NAMES_V6, FEATURE_NAMES_V6_EXTRA, build_v6_features
    F_v6 = build_v6_features(close, high, low, volume, ofi, tbv, timestamps_ms, atr_arr)
    # F_v6.shape == (n, 36)
"""
from __future__ import annotations
import numpy as np

from features_v4 import _causal_zscore, _roll_mean, _roll_std
from features_v5 import FEATURE_NAMES_V5

# ─────────────────────────────────────────────────────────────────────────────
# Feature names
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_NAMES_V6_EXTRA = [
    # ── Donchian Channels (6) — S+ ──────────────────────────────────────────
    "don_position",        # Price position in 20-bar Donchian channel [-1=low, +1=high]
    "don_upper_dist",      # Distance above upper band (positive = breakout)
    "don_lower_dist",      # Distance below lower band (positive = breakdown)
    "don_mid_dist",        # Distance from channel midpoint
    "don_width_z",         # Z-score of channel width (volatility regime indicator)
    "don_breakout",        # +1 upward breakout, -1 downward, 0 inside

    # ── Anchored VWAP (3) — S ───────────────────────────────────────────────
    "avwap_session_dist",  # Distance from daily-anchored VWAP
    "avwap_weekly_dist",   # Distance from weekly-anchored VWAP
    "avwap_monthly_dist",  # Distance from monthly-anchored VWAP

    # ── Volume Profile VAH/VAL (4) — S ──────────────────────────────────────
    "vp_poc_dist",         # Distance from Point of Control (48-bin, 1440-bar window)
    "vp_vah_dist",         # Distance from Value Area High (70% volume boundary)
    "vp_val_dist",         # Distance from Value Area Low (70% volume boundary)
    "vp_area_width",       # Normalized value area width (low = tight, high = wide)

    # ── Liquidity Sweeps (5) — S ─────────────────────────────────────────────
    "sweep_bull_flag",     # +1 if this bar is a bullish sweep (wick below low, close above)
    "sweep_bear_flag",     # +1 if this bar is a bearish sweep (wick above high, close below)
    "sweep_bull_count_z",  # Z-score of bullish sweep count in rolling 60-bar window
    "sweep_bear_count_z",  # Z-score of bearish sweep count in rolling 60-bar window
    "sweep_strength",      # Wick extension beyond key level as fraction of ATR [-1, +1]

    # ── SMA Trends (6) — A- ──────────────────────────────────────────────────
    "sma20_dist",          # Price distance from 20-bar SMA (scaled)
    "sma50_dist",          # Price distance from 50-bar SMA (scaled)
    "sma200_dist",         # Price distance from 200-bar SMA (scaled)
    "sma20_50_signal",     # +1 if price>SMA20>SMA50 (bull), -1 if price<SMA20<SMA50 (bear)
    "sma50_200_signal",    # +1 golden-cross territory, -1 death-cross territory
    "sma_alignment",       # +1 all three SMAs stacked bullish, -1 all bearish

    # ── Order Flow Extras (4) — A ────────────────────────────────────────────
    "large_trade_buy_ratio",  # Share of large-trade volume that is taker-buy
    "delta_acceleration",     # CVD acceleration: 15m delta minus 60m delta (z-scored)
    "of_imbalance_z",         # Taker buy ratio z-score vs 4h rolling window
    "taker_pressure_shift",   # Change in taker buy ratio over 30 bars

    # ── Fibonacci Retracements (5) — B ───────────────────────────────────────
    "fib_382_dist",        # Price distance from 38.2% retracement of 100-bar swing
    "fib_500_dist",        # Price distance from 50.0% retracement
    "fib_618_dist",        # Price distance from 61.8% retracement (golden pocket)
    "fib_786_dist",        # Price distance from 78.6% retracement
    "fib_confluence",      # +1 price is near any fib level, -1 far from all

    # ── Bollinger Band Extras (3) — B ────────────────────────────────────────
    "bb_width_z",          # Bandwidth z-score vs 100-bar history
    "bb_squeeze_flag",     # +1 in Bollinger squeeze (width < 20th pctile), -1 otherwise
    "bb_expansion_flag",   # +1 in expansion (width > 80th pctile), -1 otherwise
]

FEATURE_NAMES_V6 = FEATURE_NAMES_V5 + FEATURE_NAMES_V6_EXTRA
N_V6_EXTRA = len(FEATURE_NAMES_V6_EXTRA)  # == 36


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _sliding_max(arr, window):
    """Causal rolling maximum using stride_tricks (fast vectorised)."""
    n = len(arr)
    out = np.full(n, np.nan)
    try:
        from numpy.lib.stride_tricks import sliding_window_view
        out[window - 1:] = sliding_window_view(arr, window).max(axis=1)
    except Exception:
        for i in range(window - 1, n):
            out[i] = arr[i - window + 1:i + 1].max()
    return out


def _sliding_min(arr, window):
    """Causal rolling minimum using stride_tricks (fast vectorised)."""
    n = len(arr)
    out = np.full(n, np.nan)
    try:
        from numpy.lib.stride_tricks import sliding_window_view
        out[window - 1:] = sliding_window_view(arr, window).min(axis=1)
    except Exception:
        for i in range(window - 1, n):
            out[i] = arr[i - window + 1:i + 1].min()
    return out


def _anchored_vwap(close, volume, timestamps_ms, period='day'):
    """Compute VWAP reset at each period boundary (day / week / month).

    Vectorised cumsum-difference approach: O(n + B) where B = number of resets.
    CAUSAL: at bar i, VWAP uses all bars from period start through i (inclusive).
    """
    n = len(close)
    secs = (timestamps_ms / 1000).astype(np.int64)

    if period == 'day':
        period_id = secs // 86400
    elif period == 'week':
        # Thursday = Unix epoch day 0; Monday offset = +3 days
        period_id = (secs + 3 * 86400) // (7 * 86400)
    elif period == 'month':
        # Approximate month boundary every 30 days (±1 day drift is fine for VWAP)
        period_id = secs // (30 * 86400)
    else:
        raise ValueError(f"Unknown period: {period}")

    boundaries = np.zeros(n, dtype=bool)
    boundaries[0] = True
    boundaries[1:] = period_id[1:] != period_id[:-1]

    pv = close.astype(np.float64) * volume.astype(np.float64)
    cumsum_pv  = np.cumsum(pv)
    cumsum_vol = np.cumsum(volume.astype(np.float64))

    # For each group, subtract the cumsum value at the last bar of the previous group
    anchor_pv  = np.zeros(n, dtype=np.float64)
    anchor_vol = np.zeros(n, dtype=np.float64)

    boundary_idx = np.where(boundaries)[0]
    for j, bi in enumerate(boundary_idx):
        end = boundary_idx[j + 1] if j + 1 < len(boundary_idx) else n
        prev_pv  = cumsum_pv[bi - 1]  if bi > 0 else 0.0
        prev_vol = cumsum_vol[bi - 1] if bi > 0 else 0.0
        anchor_pv[bi:end]  = prev_pv
        anchor_vol[bi:end] = prev_vol

    group_pv  = cumsum_pv  - anchor_pv
    group_vol = cumsum_vol - anchor_vol
    vwap = (group_pv / np.maximum(group_vol, 1e-9)).astype(np.float32)
    return vwap


def _vpvr_value_area(volume, close, window=1440, stride=30, n_bins=48, va_pct=0.70):
    """Volume Profile with Value Area High/Low.

    Returns (poc_dist, vah_dist, val_dist, area_width) — all in [-1, 1].
    Uses 48 price bins (vs 24 in V4) for better resolution.
    Recomputes every `stride` bars and forward-fills between (same as V4 POC).
    """
    n = len(close)
    poc_d = np.zeros(n, dtype=np.float32)
    vah_d = np.zeros(n, dtype=np.float32)
    val_d = np.zeros(n, dtype=np.float32)
    aw    = np.zeros(n, dtype=np.float32)

    last_poc = last_vah = last_val = last_aw = 0.0

    for i in range(window, n):
        # Forward-fill between recompute points
        poc_d[i] = last_poc
        vah_d[i] = last_vah
        val_d[i] = last_val
        aw[i]    = last_aw

        if (i - window) % stride != 0:
            continue

        cl_w = close[i - window:i]
        v_w  = volume[i - window:i]
        lo = float(cl_w.min())
        hi = float(cl_w.max())
        cur = float(close[i])

        if hi <= lo or cur <= 0:
            continue

        edges = np.linspace(lo, hi, n_bins + 1)
        bins  = np.clip(np.searchsorted(edges, cl_w, side='right') - 1, 0, n_bins - 1)
        bv    = np.bincount(bins, weights=v_w, minlength=n_bins).astype(np.float64)
        tot   = bv.sum()
        if tot <= 0:
            continue

        poc_i = int(np.argmax(bv))
        poc_p = (edges[poc_i] + edges[poc_i + 1]) / 2.0
        last_poc = float(np.clip((cur - poc_p) / cur * 20, -1, 1))

        # Expand from POC until 70% of volume is captured
        target = tot * va_pct
        vah_i = poc_i
        val_i = poc_i
        captured = bv[poc_i]

        while captured < target:
            up_vol = bv[vah_i + 1] if vah_i + 1 < n_bins else 0.0
            dn_vol = bv[val_i - 1] if val_i - 1 >= 0    else 0.0
            if up_vol <= 0 and dn_vol <= 0:
                break
            if up_vol >= dn_vol and vah_i + 1 < n_bins:
                vah_i += 1
                captured += up_vol
            elif val_i - 1 >= 0:
                val_i -= 1
                captured += dn_vol
            else:
                if vah_i + 1 < n_bins:
                    vah_i += 1
                    captured += up_vol
                else:
                    break

        vah_p = (edges[vah_i] + edges[vah_i + 1]) / 2.0
        val_p = (edges[val_i] + edges[val_i + 1]) / 2.0

        last_vah = float(np.clip((cur - vah_p) / cur * 20, -1, 1))
        last_val = float(np.clip((cur - val_p) / cur * 20, -1, 1))
        # area_width: fraction of current price; map to [-1,1]
        last_aw  = float(np.clip((vah_p - val_p) / max(cur, 1e-10) * 10, 0, 1)) * 2 - 1

        poc_d[i] = last_poc
        vah_d[i] = last_vah
        val_d[i] = last_val
        aw[i]    = last_aw

    return poc_d, vah_d, val_d, aw


# ─────────────────────────────────────────────────────────────────────────────
# Main V6 builder
# ─────────────────────────────────────────────────────────────────────────────

def build_v6_features(close, high, low, volume, ofi, taker_buy_volume,
                      timestamps_ms, atr_arr):
    """Compute the 36 V6 features. Returns float32 array of shape (n, 36).

    All inputs are 1-minute-resolution numpy arrays of equal length n.
    `timestamps_ms`  — epoch milliseconds (int64 or float64)
    `atr_arr`        — precomputed ATR-14 from V2 feature builder
    """
    n = len(close)
    F = np.zeros((n, N_V6_EXTRA), dtype=np.float32)
    col = 0  # running column index

    close   = np.asarray(close,            dtype=np.float64)
    high    = np.asarray(high,             dtype=np.float64)
    low     = np.asarray(low,              dtype=np.float64)
    volume  = np.asarray(volume,           dtype=np.float64)
    ofi     = np.asarray(ofi,              dtype=np.float64)
    tbv     = np.asarray(taker_buy_volume, dtype=np.float64)
    atr_arr = np.asarray(atr_arr,          dtype=np.float64)
    ts_ms   = np.asarray(timestamps_ms,    dtype=np.int64)

    # ── 1. Donchian Channels (6 features, S+ tier) ────────────────────────────
    DON = 20  # standard 20-bar Donchian

    don_upper = _sliding_max(high, DON)   # highest high of last 20 bars
    don_lower = _sliding_min(low,  DON)   # lowest low of last 20 bars
    don_range = np.maximum(don_upper - don_lower, 1e-10)
    don_mid   = (don_upper + don_lower) / 2.0

    # don_position: price location in channel [-1 = at lower, +1 = at upper]
    F[:, col] = np.nan_to_num(
        np.clip((close - don_lower) / don_range * 2 - 1, -1, 1), nan=0.0
    ).astype(np.float32)
    col += 1  # 0

    # don_upper_dist: how far above upper band (positive = breakout)
    F[:, col] = np.nan_to_num(
        np.clip((close - don_upper) / np.maximum(close, 1e-10) * 20, -1, 1), nan=0.0
    ).astype(np.float32)
    col += 1  # 1

    # don_lower_dist: how far below lower band (positive = breakdown)
    F[:, col] = np.nan_to_num(
        np.clip((don_lower - close) / np.maximum(close, 1e-10) * 20, -1, 1), nan=0.0
    ).astype(np.float32)
    col += 1  # 2

    # don_mid_dist: distance from midpoint
    F[:, col] = np.nan_to_num(
        np.clip((close - don_mid) / np.maximum(don_mid, 1e-10) * 10, -1, 1), nan=0.0
    ).astype(np.float32)
    col += 1  # 3

    # don_width_z: z-score of channel width over 1-day rolling window
    F[:, col] = np.clip(
        _causal_zscore(don_range, 1440) / 3.0, -1, 1
    ).astype(np.float32)
    col += 1  # 4

    # don_breakout: compare close to PRIOR bar's channel boundaries (causal)
    breakout = np.zeros(n, dtype=np.float32)
    if n > DON:
        prev_upper = don_upper[DON - 1:-1]
        prev_lower = don_lower[DON - 1:-1]
        cur_close  = close[DON:]
        breakout[DON:] = np.where(
            cur_close > prev_upper, 1.0,
            np.where(cur_close < prev_lower, -1.0, 0.0)
        )
    F[:, col] = breakout
    col += 1  # 5

    # ── 2. Anchored VWAP (3 features, S tier) ────────────────────────────────
    avwap_day   = _anchored_vwap(close, volume, ts_ms, 'day')
    avwap_week  = _anchored_vwap(close, volume, ts_ms, 'week')
    avwap_month = _anchored_vwap(close, volume, ts_ms, 'month')

    def _vwap_dist(cl, vwap, scale=20.0):
        d = (cl - vwap) / np.maximum(vwap, 1e-10)
        return np.nan_to_num(np.clip(d * scale, -1, 1), nan=0.0).astype(np.float32)

    F[:, col] = _vwap_dist(close, avwap_day);   col += 1  # 6
    F[:, col] = _vwap_dist(close, avwap_week);  col += 1  # 7
    F[:, col] = _vwap_dist(close, avwap_month); col += 1  # 8

    # ── 3. Volume Profile VAH/VAL (4 features, S tier) ───────────────────────
    poc_d, vah_d, val_d, aw = _vpvr_value_area(
        volume, close, window=1440, stride=30, n_bins=48, va_pct=0.70
    )
    F[:, col] = np.clip(poc_d, -1, 1); col += 1  # 9
    F[:, col] = np.clip(vah_d, -1, 1); col += 1  # 10
    F[:, col] = np.clip(val_d, -1, 1); col += 1  # 11
    F[:, col] = np.clip(aw,   -1, 1);  col += 1  # 12

    # ── 4. Liquidity Sweeps (5 features, S tier) ─────────────────────────────
    # A sweep occurs when price wicks beyond a key level (prev N-bar high/low)
    # but closes back inside — revealing stop-hunt followed by reversal.
    SWEEP_PERIOD = 20   # Key level = max/min of prior 20 bars (not including current)

    sweep_bull = np.zeros(n, dtype=np.float32)   # bullish: low swept, close above
    sweep_bear = np.zeros(n, dtype=np.float32)   # bearish: high swept, close below
    sweep_str  = np.zeros(n, dtype=np.float32)   # wick magnitude vs ATR

    if n > SWEEP_PERIOD + 1:
        try:
            from numpy.lib.stride_tricks import sliding_window_view
            # Prior N bars (exclude current bar → shift by 1)
            key_high_win = sliding_window_view(high[:-1], SWEEP_PERIOD)  # shape (n-period, period)
            key_low_win  = sliding_window_view(low[:-1],  SWEEP_PERIOD)
            key_high = key_high_win.max(axis=1)  # length = n - SWEEP_PERIOD
            key_low  = key_low_win.min(axis=1)

            start = SWEEP_PERIOD + 1
            ch = high[start:]
            cl_bar = low[start:]
            cc = close[start:]
            at = np.maximum(atr_arr[start:], 1e-10)

            kh = key_high[:len(ch)]
            kl = key_low[:len(cl_bar)]

            # Bearish sweep: wick above prior range top, close below it
            bear_mask = (ch > kh) & (cc < kh)
            # Bullish sweep: wick below prior range bottom, close above it
            bull_mask = (cl_bar < kl) & (cc > kl)

            sweep_bear[start:] = bear_mask.astype(np.float32)
            sweep_bull[start:] = bull_mask.astype(np.float32)

            bear_ext = np.where(bear_mask, (ch - kh) / at, 0.0)
            bull_ext = np.where(bull_mask, (kl - cl_bar) / at, 0.0)
            sweep_str[start:] = np.clip(bear_ext + bull_ext, 0, 3).astype(np.float32) / 3.0
        except Exception:
            # Fallback scalar loop
            for i in range(SWEEP_PERIOD + 1, n):
                kh = high[i - SWEEP_PERIOD:i].max()
                kl = low[i  - SWEEP_PERIOD:i].min()
                at = max(atr_arr[i], 1e-10)
                if high[i] > kh and close[i] < kh:
                    sweep_bear[i] = 1.0
                    sweep_str[i] = min((high[i] - kh) / at, 1.0)
                elif low[i] < kl and close[i] > kl:
                    sweep_bull[i] = 1.0
                    sweep_str[i] = min((kl - low[i]) / at, 1.0)

    # Remap binary flags: 0/1 → -1/+1
    F[:, col] = sweep_bull * 2 - 1; col += 1  # 13
    F[:, col] = sweep_bear * 2 - 1; col += 1  # 14

    # Rolling count of sweeps in 60-bar window, z-scored
    bull_count_1h = _roll_mean(sweep_bull, 60) * 60
    bear_count_1h = _roll_mean(sweep_bear, 60) * 60
    F[:, col] = np.clip(_causal_zscore(bull_count_1h, 1440) / 3, -1, 1).astype(np.float32)
    col += 1  # 15
    F[:, col] = np.clip(_causal_zscore(bear_count_1h, 1440) / 3, -1, 1).astype(np.float32)
    col += 1  # 16

    # Sweep strength: last wick extension (already [0,1] → remap to [-1,+1])
    F[:, col] = (sweep_str * 2 - 1).astype(np.float32)
    col += 1  # 17

    # ── 5. SMA Trends (6 features, A- tier) ──────────────────────────────────
    sma20  = _roll_mean(close, 20)
    sma50  = _roll_mean(close, 50)
    sma200 = _roll_mean(close, 200)

    def _sma_dist(cl, sma, scale):
        d = (cl - sma) / np.maximum(sma, 1e-10)
        return np.nan_to_num(np.clip(d * scale, -1, 1), nan=0.0).astype(np.float32)

    F[:, col] = _sma_dist(close, sma20,  20.0); col += 1  # 18
    F[:, col] = _sma_dist(close, sma50,  10.0); col += 1  # 19
    F[:, col] = _sma_dist(close, sma200,  5.0); col += 1  # 20

    # sma20_50_signal: full bull = price>SMA20>SMA50, full bear = price<SMA20<SMA50
    above20 = (close > sma20).astype(np.float32)
    above50 = (close > sma50).astype(np.float32)
    sma20_gt_50 = (sma20 > sma50).astype(np.float32)
    sig_2050 = np.where(
        (sma20_gt_50 > 0) & (above50 > 0),  1.0,
        np.where((sma20_gt_50 < 1) & (above50 < 1), -1.0, 0.0)
    )
    F[:, col] = np.nan_to_num(sig_2050, nan=0.0).astype(np.float32)
    col += 1  # 21

    # sma50_200_signal: golden (+1) vs death (-1) cross territory
    sma50_gt_200 = (sma50 > sma200).astype(np.float32) * 2 - 1
    F[:, col] = np.nan_to_num(sma50_gt_200, nan=0.0).astype(np.float32)
    col += 1  # 22

    # sma_alignment: all three SMAs and price stacked
    bull_align = (close > sma20) & (sma20 > sma50) & (sma50 > sma200)
    bear_align = (close < sma20) & (sma20 < sma50) & (sma50 < sma200)
    align = np.where(bull_align, 1.0, np.where(bear_align, -1.0, 0.0))
    F[:, col] = np.nan_to_num(align, nan=0.0).astype(np.float32)
    col += 1  # 23

    # ── 6. Order Flow Extras (4 features, A tier) ────────────────────────────
    taker_sell = np.maximum(volume - tbv, 0.0)
    avg_vol_60 = _roll_mean(volume, 60)

    # Large-trade threshold: 2x average bar volume
    large_thr  = 2.0 * avg_vol_60
    large_buy  = np.where(tbv > large_thr, tbv, 0.0)
    large_sell = np.where(taker_sell > large_thr, taker_sell, 0.0)

    large_sum_60     = _roll_mean(large_buy + large_sell, 60) * 60
    large_buy_sum_60 = _roll_mean(large_buy, 60) * 60
    with np.errstate(divide='ignore', invalid='ignore'):
        lbr = np.where(large_sum_60 > 1e-9, large_buy_sum_60 / large_sum_60, 0.5)
    F[:, col] = np.nan_to_num(
        np.clip((lbr - 0.5) * 2, -1, 1), nan=0.0
    ).astype(np.float32)
    col += 1  # 24

    # delta_acceleration: fast CVD delta vs slow CVD delta
    cvd = np.cumsum(np.nan_to_num(ofi, nan=0.0))
    delta_15m = np.zeros(n); delta_15m[15:]  = cvd[15:]  - cvd[:-15]
    delta_60m = np.zeros(n); delta_60m[60:]  = cvd[60:]  - cvd[:-60]
    delta_accel = _causal_zscore(delta_15m - delta_60m / 4.0, 1440)
    F[:, col] = np.clip(delta_accel / 3.0, -1, 1).astype(np.float32)
    col += 1  # 25

    # of_imbalance_z: taker buy ratio z-score over 4h window
    tbv_ratio = tbv / np.maximum(volume, 1e-9)
    of_imb_z  = _causal_zscore(tbv_ratio, 240)
    F[:, col] = np.clip(of_imb_z / 3.0, -1, 1).astype(np.float32)
    col += 1  # 26

    # taker_pressure_shift: change in taker buy ratio over 30 bars
    tbv_shifted = np.zeros(n)
    if n > 30:
        tbv_shifted[30:] = tbv_ratio[:-30]
    pressure_shift = tbv_ratio - tbv_shifted
    F[:, col] = np.clip(pressure_shift * 5.0, -1, 1).astype(np.float32)
    col += 1  # 27

    # ── 7. Fibonacci Retracements (5 features, B tier) ───────────────────────
    FIB_LOOKBACK = 100  # 100-bar swing high/low for fib levels

    swing_hi = _sliding_max(high, FIB_LOOKBACK)
    swing_lo = _sliding_min(low,  FIB_LOOKBACK)
    swing_range = np.maximum(swing_hi - swing_lo, 1e-10)

    # Fib levels measured down from swing high (standard retracement convention)
    fib_382 = swing_hi - 0.382 * swing_range
    fib_500 = swing_hi - 0.500 * swing_range
    fib_618 = swing_hi - 0.618 * swing_range
    fib_786 = swing_hi - 0.786 * swing_range

    def _fib_dist(cl, level, scale=10.0):
        d = (cl - level) / np.maximum(cl, 1e-10)
        return np.nan_to_num(np.clip(d * scale, -1, 1), nan=0.0).astype(np.float32)

    F[:, col] = _fib_dist(close, fib_382, 10); col += 1  # 28
    F[:, col] = _fib_dist(close, fib_500, 10); col += 1  # 29
    F[:, col] = _fib_dist(close, fib_618, 10); col += 1  # 30
    F[:, col] = _fib_dist(close, fib_786, 10); col += 1  # 31

    # Fibonacci confluence: how close is price to ANY of the four key levels
    fib_levels = np.stack([fib_382, fib_500, fib_618, fib_786], axis=1)
    # Fill NaN fib levels (warmup period) with current close so dist = 0 → neutral
    cl_2d = close[:, None]
    fib_safe = np.where(np.isnan(fib_levels), cl_2d, fib_levels)
    min_fib_dist_pct = np.min(
        np.abs(cl_2d - fib_safe) / np.maximum(cl_2d, 1e-10), axis=1
    )
    # +1 = right at a fib level, -1 = far from all
    confluence = np.clip(1.0 - min_fib_dist_pct * 20, -1, 1)
    F[:, col] = np.nan_to_num(confluence, nan=0.0).astype(np.float32)
    col += 1  # 32

    # ── 8. Bollinger Band Extras (3 features, B tier) ────────────────────────
    sma20_bb = _roll_mean(close, 20)
    std20_bb  = _roll_std(close, 20)
    bb_width  = 4.0 * std20_bb  # total width = 2σ above + 2σ below

    # Bandwidth z-score vs 100-bar history
    bb_width_z_arr = _causal_zscore(bb_width, 100)
    F[:, col] = np.clip(bb_width_z_arr / 3.0, -1, 1).astype(np.float32)
    col += 1  # 33

    # Squeeze flag: bandwidth in lowest ~20th percentile (z < -0.84 ≈ 20th pctile N(0,1))
    squeeze   = (bb_width_z_arr < -0.84).astype(np.float32) * 2 - 1
    F[:, col] = squeeze
    col += 1  # 34

    # Expansion flag: bandwidth in highest ~20th percentile (z > +0.84)
    expansion = (bb_width_z_arr > 0.84).astype(np.float32) * 2 - 1
    F[:, col] = expansion
    col += 1  # 35

    assert col == N_V6_EXTRA, f"V6 feature count mismatch: built {col}, expected {N_V6_EXTRA}"
    return F


def compute_v6_signal_strength(F_v6):
    """Compute a scalar signal-strength score [0, 1] for each sample row.

    Used for sample weighting during training: rows where V6 signals are in
    strong agreement receive higher weight (up to 2.33x) so the GBDTs spend
    more learning capacity on those samples — effectively giving V6 features
    70% of the model's effective weight.

    Combines:
      - Donchian breakout magnitude
      - AVWAP session/weekly agreement
      - Liquidity sweep flags
      - SMA alignment score
    """
    # Map [-1,1] → [0,1] for unsigned magnitude
    don_pos    = np.abs(F_v6[:, 0])                       # don_position
    don_bk     = np.abs(F_v6[:, 5])                       # don_breakout
    avwap_s    = np.abs(F_v6[:, 6])                       # avwap_session_dist
    avwap_w    = np.abs(F_v6[:, 7])                       # avwap_weekly_dist
    sweep_bull = (F_v6[:, 13] + 1) / 2                    # [-1,1] → [0,1]
    sweep_bear = (F_v6[:, 14] + 1) / 2
    sma_align  = np.abs(F_v6[:, 23])                      # sma_alignment

    # Weighted average of key signals (S+ and S tier dominate)
    strength = (
        0.25 * don_bk
        + 0.20 * don_pos
        + 0.15 * avwap_s
        + 0.10 * avwap_w
        + 0.15 * np.maximum(sweep_bull, sweep_bear)
        + 0.15 * sma_align
    )
    return np.clip(strength, 0, 1).astype(np.float32)
