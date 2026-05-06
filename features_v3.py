"""
features_v3.py — Phase A feature upgrades
==========================================
Adds 7 price-structure features + 3 interactions to the v2 feature set.
Drops 6 dead/weak features. Total: 29 features.
"""
import numpy as np

FEATURE_NAMES_V3 = [
    # Flow (6)
    "ofi_score", "ofi_acceleration", "absorption_index",
    "cvd_divergence_15m", "cvd_divergence_1h", "cvd_divergence_4h",
    # Regime (4)
    "hurst", "trend_efficiency", "persistence_score", "vol_state",
    # Accumulation (2)
    "accum_probability", "distrib_probability",
    # Volume (2)
    "volume_zscore", "atr_normalized_ofi",
    # Time (4)
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    # Cross-market (1)
    "btc_return_60m",
    # NEW: Price structure (7)
    "price_vs_ema200", "bb_position", "rsi_14",
    "atr_percentile", "return_momentum_5", "return_momentum_60",
    "range_position",
    # NEW: Interactions (3)
    "hurst_x_ofi", "volstate_x_absorption", "te_x_cvd4h",
]

# Map v2 columns to v3 columns (v2 has 25 cols, we keep 19 of them)
# v2 indices to KEEP: 0,1,3, 7,8,9, 10,11,12, 14, 16,17, 18,19, 20,21,22,23, 24
V2_KEEP_INDICES = [0, 1, 3, 7, 8, 9, 10, 11, 12, 14, 16, 17, 18, 19, 20, 21, 22, 23, 24]
# That's 19 features from v2


def _ema(x, period):
    """Exponential moving average."""
    out = np.full(len(x), np.nan)
    k = 2.0 / (period + 1)
    out[period - 1] = np.mean(x[:period])
    for i in range(period, len(x)):
        out[i] = x[i] * k + out[i - 1] * (1 - k)
    return out


def _rsi(close, period=14):
    """RSI indicator."""
    n = len(close)
    out = np.full(n, np.nan)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = np.full(n - 1, np.nan)
    avg_loss = np.full(n - 1, np.nan)
    if period > len(gain):
        return out
    avg_gain[period - 1] = gain[:period].mean()
    avg_loss[period - 1] = loss[:period].mean()
    for i in range(period, len(gain)):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gain[i]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + loss[i]) / period
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = np.where(avg_loss > 0, avg_gain / avg_loss, 100.0)
    rsi_vals = 100.0 - 100.0 / (1.0 + rs)
    out[1:] = rsi_vals
    return out


def _roll_mean(x, w):
    out = np.full(len(x), np.nan)
    cs = np.cumsum(np.nan_to_num(x, nan=0.0))
    out[w - 1:] = (cs[w - 1:] - np.concatenate([[0], cs[:-w]])) / w
    return out


def _roll_std(x, w):
    m = _roll_mean(x, w)
    m2 = _roll_mean(x ** 2, w)
    var = m2 - m ** 2
    return np.sqrt(np.maximum(var, 0))


def _roll_percentile(x, val, w):
    """Rolling percentile rank of val within window w of array x."""
    n = len(x)
    out = np.full(n, np.nan)
    for i in range(w, n):
        window = x[i - w:i + 1]
        valid = window[~np.isnan(window)]
        if len(valid) > 0:
            out[i] = np.sum(valid <= val[i]) / len(valid)
    return out


def compute_price_structure_features(close, high, low, atr_arr):
    """Compute 7 price-structure features. Returns (n, 7) array."""
    n = len(close)
    F = np.full((n, 7), np.nan)

    # 0: price_vs_ema200 — distance from 200 EMA as fraction
    ema200 = _ema(close, 200)
    with np.errstate(divide='ignore', invalid='ignore'):
        F[:, 0] = np.clip((close - ema200) / np.maximum(ema200, 1e-10), -0.5, 0.5) * 2  # [-1, 1]

    # 1: bb_position — where in Bollinger Bands (0=lower, 1=upper)
    sma20 = _roll_mean(close, 20)
    std20 = _roll_std(close, 20)
    bb_upper = sma20 + 2 * std20
    bb_lower = sma20 - 2 * std20
    bb_range = np.maximum(bb_upper - bb_lower, 1e-10)
    F[:, 1] = np.clip((close - bb_lower) / bb_range, 0, 1) * 2 - 1  # [-1, 1]

    # 2: rsi_14 — normalized to [-1, 1]
    rsi = _rsi(close, 14)
    F[:, 2] = (rsi - 50) / 50  # [-1, 1]

    # 3: atr_percentile — current ATR rank in 100-bar window
    atr_pctile = _roll_percentile(atr_arr, atr_arr, 100)
    F[:, 3] = atr_pctile * 2 - 1  # [-1, 1]

    # 4: return_momentum_5 — 5-bar return
    F[5:, 4] = np.clip((close[5:] - close[:-5]) / np.maximum(close[:-5], 1e-10) * 100, -5, 5) / 5

    # 5: return_momentum_60 — 60-bar return
    F[60:, 5] = np.clip((close[60:] - close[:-60]) / np.maximum(close[:-60], 1e-10) * 100, -10, 10) / 10

    # 6: range_position — price in 20-bar high/low range
    for i in range(20, n):
        hi20 = high[i - 19:i + 1].max()
        lo20 = low[i - 19:i + 1].min()
        rng = hi20 - lo20
        if rng > 0:
            F[i, 6] = (close[i] - lo20) / rng * 2 - 1  # [-1, 1]

    return F


def compute_interactions(F_v3_partial):
    """Compute 3 interaction features from the v3 feature matrix.
    Input should have columns in FEATURE_NAMES_V3 order (first 26 cols).
    Returns (n, 3) array."""
    n = len(F_v3_partial)
    I = np.full((n, 3), 0.0)

    # hurst_x_ofi = hurst(idx 6) * ofi_score(idx 0)
    hurst = F_v3_partial[:, 6]
    ofi = F_v3_partial[:, 0]
    I[:, 0] = np.nan_to_num(hurst * ofi, nan=0.0)

    # volstate_x_absorption = vol_state(idx 9) * absorption_index(idx 2)
    vs = F_v3_partial[:, 9]
    ab = F_v3_partial[:, 2]
    I[:, 1] = np.nan_to_num(vs * ab, nan=0.0)

    # te_x_cvd4h = trend_efficiency(idx 7) * cvd_divergence_4h(idx 5)
    # Note: in v3 order, cvd_divergence_4h is at index 5, trend_efficiency at 7
    te = F_v3_partial[:, 7]
    cvd4h = F_v3_partial[:, 5]
    I[:, 2] = np.nan_to_num(te * cvd4h, nan=0.0)

    return I


def build_v3_from_v2(F_v2, close, high, low, atr_arr):
    """Convert v2 feature matrix (n,25) to v3 (n,29).
    Drops 6 dead features, adds 7 price-structure + 3 interactions."""
    # Keep 19 from v2
    F_kept = F_v2[:, V2_KEEP_INDICES]  # (n, 19)

    # Price structure features (7)
    F_price = compute_price_structure_features(close, high, low, atr_arr)  # (n, 7)

    # Combine kept + price = 26 cols
    F_partial = np.hstack([F_kept, F_price])  # (n, 26)

    # Interactions (3)
    F_inter = compute_interactions(F_partial)  # (n, 3)

    # Final: 26 + 3 = 29
    return np.hstack([F_partial, F_inter])
