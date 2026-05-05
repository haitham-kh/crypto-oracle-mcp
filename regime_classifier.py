from __future__ import annotations
"""
REGIME CLASSIFIER — Data-Driven Market Regime Detection
=======================================================
Replaces ADX-only heuristic with multi-signal regime identification.

Regimes:
  TRENDING_UP    — persistent upward drift, ADX > 25, Hurst > 0.55
  TRENDING_DOWN  — persistent downward drift, ADX > 25, Hurst > 0.55
  RANGING        — oscillating, ADX < 20, Hurst < 0.5, negative autocorrelation
  EXPANSION      — volatility breakout from compression
  LOW_LIQUIDITY  — volume < 30% of 20d avg → signals unreliable

Outputs:
  - regime classification
  - confidence (0-100)
  - recommended strategy type
  - which signal categories dominate in this regime
"""

import numpy as np
from typing import Dict, List, Any, Optional


# ─────────────────────────────────────────────────────────────────────────────
# HURST EXPONENT (Rescaled Range Analysis)
# ─────────────────────────────────────────────────────────────────────────────

def compute_hurst(prices: np.ndarray, min_n: int = 20) -> Optional[float]:
    """
    Compute Hurst exponent via R/S analysis.

    H > 0.55 → trending (persistent)
    H ≈ 0.50 → random walk (unpredictable)
    H < 0.45 → mean-reverting (anti-persistent)

    Returns None if insufficient data.
    """
    if prices is None or len(prices) < min_n:
        return None

    returns = np.diff(np.log(prices + 1e-10))
    n = len(returns)
    if n < 10:
        return None

    # Use multiple sub-series lengths
    lags = [max(10, n // 8), max(10, n // 4), max(10, n // 2)]
    rs_values = []
    lag_values = []

    for lag in lags:
        if lag > n:
            continue
        sub_returns = returns[-lag:]
        mean_r = np.mean(sub_returns)
        deviation = np.cumsum(sub_returns - mean_r)
        R = np.max(deviation) - np.min(deviation)
        S = np.std(sub_returns, ddof=1)
        if S > 0 and R > 0:
            rs_values.append(np.log(R / S))
            lag_values.append(np.log(lag))

    if len(rs_values) < 2:
        return None

    # Linear regression of log(R/S) vs log(lag) → slope = Hurst exponent
    try:
        coeffs = np.polyfit(lag_values, rs_values, 1)
        hurst = float(np.clip(coeffs[0], 0.01, 0.99))
        return round(hurst, 3)
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# PRICE AUTOCORRELATION
# ─────────────────────────────────────────────────────────────────────────────

def compute_return_autocorrelation(prices: np.ndarray, lag: int = 1) -> Optional[float]:
    """
    First-order autocorrelation of returns.

    Positive autocorrelation → trending (returns tend to continue)
    Negative autocorrelation → mean-reverting (returns tend to reverse)
    """
    if prices is None or len(prices) < lag + 5:
        return None
    returns = np.diff(prices) / (prices[:-1] + 1e-10)
    if len(returns) < lag + 2:
        return None
    r_t = returns[lag:]
    r_lag = returns[:-lag]
    if np.std(r_t) == 0 or np.std(r_lag) == 0:
        return None
    corr = float(np.corrcoef(r_t, r_lag)[0, 1])
    return round(corr, 4)


# ─────────────────────────────────────────────────────────────────────────────
# VOLATILITY REGIME
# ─────────────────────────────────────────────────────────────────────────────

def compute_volatility_regime(prices: np.ndarray) -> Dict[str, Any]:
    """
    Detect if volatility is expanding, contracting, or stable.

    Expansion = recent 5-period vol > 1.5× prior 20-period vol → breakout regime
    Contraction = recent vol < 0.5× prior vol → compression, coiling
    """
    if prices is None or len(prices) < 25:
        return {"regime": "unknown", "expansion": False, "contraction": False}

    returns = np.diff(np.log(prices + 1e-10))
    if len(returns) < 20:
        return {"regime": "unknown", "expansion": False, "contraction": False}

    recent_vol = float(np.std(returns[-5:])) if len(returns) >= 5 else 0.0
    prior_vol = float(np.std(returns[-25:-5])) if len(returns) >= 25 else float(np.std(returns))

    if prior_vol == 0:
        return {"regime": "unknown", "expansion": False, "contraction": False}

    vol_ratio = recent_vol / prior_vol

    if vol_ratio > 1.5:
        regime = "expanding"
        expansion = True
        contraction = False
    elif vol_ratio < 0.5:
        regime = "contracting"
        expansion = False
        contraction = True
    else:
        regime = "stable"
        expansion = False
        contraction = False

    return {
        "regime": regime,
        "expansion": expansion,
        "contraction": contraction,
        "vol_ratio": round(vol_ratio, 3),
        "recent_vol_pct": round(recent_vol * 100, 4),
        "prior_vol_pct": round(prior_vol * 100, 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# HIGHER HIGHS / HIGHER LOWS STRUCTURE
# ─────────────────────────────────────────────────────────────────────────────

def detect_price_structure(candles: List[Dict]) -> Dict[str, Any]:
    """
    Detect HH/HL (uptrend) or LH/LL (downtrend) structure.

    Uses last 3 significant swing points to determine structural bias.
    """
    if not candles or len(candles) < 15:
        return {"structure": "unknown"}

    closes = [float(c.get("close") or 0) for c in candles]
    highs = [float(c.get("high") or 0) for c in candles]
    lows = [float(c.get("low") or 0) for c in candles]

    # Find swing highs and lows (local maxima/minima with order=3)
    order = 3
    swing_highs = []
    swing_lows = []

    for i in range(order, len(highs) - order):
        window_h = highs[i - order: i + order + 1]
        window_l = lows[i - order: i + order + 1]
        if highs[i] == max(window_h):
            swing_highs.append((i, highs[i]))
        if lows[i] == min(window_l):
            swing_lows.append((i, lows[i]))

    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return {"structure": "unclear", "swing_highs": [], "swing_lows": []}

    # Last 2 swing highs and lows
    sh1, sh2 = swing_highs[-2][1], swing_highs[-1][1]
    sl1, sl2 = swing_lows[-2][1], swing_lows[-1][1]

    hh = sh2 > sh1  # Higher High
    hl = sl2 > sl1  # Higher Low
    lh = sh2 < sh1  # Lower High
    ll = sl2 < sl1  # Lower Low

    if hh and hl:
        structure = "uptrend"
        detail = f"HH ({sh2:.4g} > {sh1:.4g}) + HL ({sl2:.4g} > {sl1:.4g})"
    elif lh and ll:
        structure = "downtrend"
        detail = f"LH ({sh2:.4g} < {sh1:.4g}) + LL ({sl2:.4g} < {sl1:.4g})"
    elif hh and ll:
        structure = "volatile"
        detail = "HH but LL — expanding range, no clear bias"
    elif lh and hl:
        structure = "consolidating"
        detail = "LH + HL — contracting range, coiling for breakout"
    else:
        structure = "mixed"
        detail = "Ambiguous structure — no clear HH/HL or LH/LL pattern"

    return {
        "structure": structure,
        "detail": detail,
        "last_swing_high": round(sh2, 8),
        "prev_swing_high": round(sh1, 8),
        "last_swing_low": round(sl2, 8),
        "prev_swing_low": round(sl1, 8),
        "higher_high": bool(hh),
        "higher_low": bool(hl),
        "lower_high": bool(lh),
        "lower_low": bool(ll),
    }


# ─────────────────────────────────────────────────────────────────────────────
# VOLUME LIQUIDITY CHECK
# ─────────────────────────────────────────────────────────────────────────────

def check_volume_liquidity(candles: List[Dict]) -> Dict[str, Any]:
    """
    Check if current volume is sufficient for reliable signals.
    Low volume markets produce false breakouts and unreliable TA.
    """
    if not candles or len(candles) < 5:
        return {"adequate": True, "vol_ratio": 1.0}

    volumes = [float(c.get("volume") or 0) for c in candles]
    current_vol = volumes[-1] if volumes else 0
    avg_vol_20 = float(np.mean(volumes[-20:])) if len(volumes) >= 20 else float(np.mean(volumes))

    if avg_vol_20 == 0:
        return {"adequate": False, "vol_ratio": 0.0, "reason": "zero_avg_volume"}

    vol_ratio = current_vol / avg_vol_20

    return {
        "adequate": vol_ratio >= 0.30,
        "vol_ratio": round(vol_ratio, 3),
        "current_volume": round(current_vol, 2),
        "avg_volume_20": round(avg_vol_20, 2),
        "low_liquidity": vol_ratio < 0.30,
        "volume_spike": vol_ratio > 2.5,
    }


# ─────────────────────────────────────────────────────────────────────────────
# TREND EFFICIENCY RATIO
# ─────────────────────────────────────────────────────────────────────────────

def compute_trend_efficiency(prices: np.ndarray, window: int = 14) -> Optional[float]:
    """
    Trend Efficiency Ratio (Kaufman's ER).
    ER = |net_price_change| / sum(|individual_candle_changes|)

    ER = 1.0 → perfectly trending (no noise)
    ER = 0.0 → pure noise / chop
    ER > 0.6 → strong trend signal reliable
    ER < 0.3 → trend signals unreliable, oscillators preferred
    """
    if prices is None or len(prices) < window + 1:
        return None

    prices = prices[-window - 1:]
    net_change = abs(float(prices[-1]) - float(prices[0]))
    path_length = float(np.sum(np.abs(np.diff(prices))))

    if path_length == 0:
        return None

    er = net_change / path_length
    return round(float(er), 4)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN REGIME CLASSIFIER
# ─────────────────────────────────────────────────────────────────────────────

def classify_regime(candles_4h: List[Dict], candles_1d: List[Dict]) -> Dict[str, Any]:
    """
    Classify market regime from 4H and 1D candles.

    Priority order:
    1. LOW_LIQUIDITY overrides all (signals are unreliable)
    2. EXPANSION (volatility breakout) takes precedence in trend detection
    3. TRENDING_UP / TRENDING_DOWN based on Hurst + structure + ADX
    4. RANGING default if no trend confirmed

    Returns:
      regime: str
      confidence: int (0-100)
      strategy_bias: str (what works in this regime)
      signal_weights: Dict (how to weight different signal types)
      all_metrics: Dict (raw values for diagnostics)
    """
    # Use 4H as primary, fall back to 1D for longer-horizon metrics
    primary = candles_4h or candles_1d
    secondary = candles_1d or candles_4h

    if not primary:
        return {
            "regime": "UNKNOWN",
            "confidence": 0,
            "strategy_bias": "no_trade",
            "signal_weights": {},
        }

    closes_primary = np.array([float(c.get("close") or 0) for c in primary])
    closes_secondary = np.array([float(c.get("close") or 0) for c in secondary])

    # Compute all metrics
    hurst_4h = compute_hurst(closes_primary)
    hurst_1d = compute_hurst(closes_secondary)
    hurst = hurst_4h or hurst_1d

    autocorr = compute_return_autocorrelation(closes_primary, lag=1)
    vol_regime = compute_volatility_regime(closes_primary)
    structure = detect_price_structure(primary)
    liq = check_volume_liquidity(primary)
    er = compute_trend_efficiency(closes_primary, window=14)

    # ADX from indicators if available (fallback: not used)
    # We compute it inline from candle data
    adx_val = _compute_adx_simple(primary)

    all_metrics = {
        "hurst_4h": hurst_4h,
        "hurst_1d": hurst_1d,
        "return_autocorr_1lag": autocorr,
        "vol_regime": vol_regime.get("regime"),
        "vol_ratio": vol_regime.get("vol_ratio"),
        "price_structure": structure.get("structure"),
        "trend_efficiency_ratio": er,
        "adx_approx": adx_val,
        "volume_liquidity_ratio": liq.get("vol_ratio"),
        "volume_adequate": liq.get("adequate"),
    }

    confidence_factors: List[int] = []

    # ── Rule 1: Low Liquidity overrides everything
    if not liq.get("adequate"):
        return {
            "regime": "LOW_LIQUIDITY",
            "confidence": 85,
            "strategy_bias": "reduce_size_or_avoid",
            "interpretation": "Volume < 30% of 20-period average — all signals have elevated false positive rate",
            "signal_weights": {
                "ofi": 0.3, "cvd": 0.3, "technical": 0.0,
                "derivatives_context": 0.0, "macro": 0.4,
            },
            "all_metrics": all_metrics,
        }

    # ── Rule 2: Expansion (volatility breakout)
    if vol_regime.get("expansion"):
        direction_hint = structure.get("structure", "mixed")
        return {
            "regime": "EXPANSION",
            "confidence": 70,
            "strategy_bias": "breakout_follow" if er and er > 0.5 else "wait_for_confirmation",
            "interpretation": f"Volatility expanding {vol_regime.get('vol_ratio', 0):.1f}×. {direction_hint} structure.",
            "signal_weights": {
                "ofi": 0.40, "cvd": 0.35, "technical": 0.10,
                "derivatives_context": 0.05, "macro": 0.10,
            },
            "all_metrics": all_metrics,
        }

    # ── Rule 3: Trending
    trend_evidence = 0
    if hurst and hurst > 0.55:
        trend_evidence += 2
    if adx_val and adx_val > 25:
        trend_evidence += 2
    if er and er > 0.55:
        trend_evidence += 1
    if autocorr and autocorr > 0.1:
        trend_evidence += 1

    if trend_evidence >= 3:
        struct = structure.get("structure", "mixed")
        if struct == "uptrend":
            regime = "TRENDING_UP"
        elif struct == "downtrend":
            regime = "TRENDING_DOWN"
        else:
            # Use price vs mid of range
            current = float(closes_primary[-1]) if len(closes_primary) > 0 else 0
            mid = float(np.mean([np.max(closes_primary), np.min(closes_primary)]))
            regime = "TRENDING_UP" if current > mid else "TRENDING_DOWN"

        confidence = min(90, 40 + trend_evidence * 10)
        return {
            "regime": regime,
            "confidence": confidence,
            "strategy_bias": "trend_follow",
            "interpretation": (
                f"Hurst {hurst:.2f} + ADX ~{adx_val:.0f} + ER {er:.2f} confirm directional trend. "
                f"Structure: {struct}."
            ),
            "signal_weights": {
                "ofi": 0.30, "cvd": 0.30, "technical": 0.20,
                "derivatives_context": 0.10, "macro": 0.10,
            },
            "all_metrics": all_metrics,
        }

    # ── Rule 4: Ranging (default)
    range_evidence = 0
    if hurst and hurst < 0.50:
        range_evidence += 2
    if autocorr and autocorr < -0.05:
        range_evidence += 2
    if vol_regime.get("contraction"):
        range_evidence += 1
    if er and er < 0.35:
        range_evidence += 1

    confidence = min(80, 40 + range_evidence * 8)

    return {
        "regime": "RANGING",
        "confidence": confidence,
        "strategy_bias": "mean_reversion_or_wait",
        "interpretation": (
            f"Hurst {hurst:.2f} + autocorr {autocorr:.2f} indicate range-bound price action. "
            f"Oscillators dominate; trend signals downweighted."
        ),
        "signal_weights": {
            "ofi": 0.35, "cvd": 0.30, "technical": 0.15,
            "derivatives_context": 0.05, "macro": 0.15,
        },
        "all_metrics": all_metrics,
    }


def _compute_adx_simple(candles: List[Dict], period: int = 14) -> Optional[float]:
    """Lightweight ADX computation without full indicator library."""
    if not candles or len(candles) < period + 5:
        return None
    try:
        highs = np.array([float(c.get("high") or 0) for c in candles])
        lows = np.array([float(c.get("low") or 0) for c in candles])
        closes = np.array([float(c.get("close") or 0) for c in candles])

        # True Range
        prev_closes = closes[:-1]
        cur_highs = highs[1:]
        cur_lows = lows[1:]
        tr = np.maximum(cur_highs - cur_lows,
               np.maximum(np.abs(cur_highs - prev_closes),
                          np.abs(cur_lows - prev_closes)))

        # Directional Movement
        up_moves = highs[1:] - highs[:-1]
        down_moves = lows[:-1] - lows[1:]
        plus_dm = np.where((up_moves > down_moves) & (up_moves > 0), up_moves, 0.0)
        minus_dm = np.where((down_moves > up_moves) & (down_moves > 0), down_moves, 0.0)

        # Smooth with Wilder's method
        def wilder_smooth(arr, n):
            result = np.zeros(len(arr))
            result[n - 1] = np.sum(arr[:n])
            for i in range(n, len(arr)):
                result[i] = result[i - 1] - result[i - 1] / n + arr[i]
            return result

        atr14 = wilder_smooth(tr, period)
        plus14 = wilder_smooth(plus_dm, period)
        minus14 = wilder_smooth(minus_dm, period)

        with np.errstate(divide="ignore", invalid="ignore"):
            plus_di = 100 * plus14 / np.where(atr14 > 0, atr14, 1)
            minus_di = 100 * minus14 / np.where(atr14 > 0, atr14, 1)
            dx = 100 * np.abs(plus_di - minus_di) / np.where(plus_di + minus_di > 0, plus_di + minus_di, 1)

        adx = wilder_smooth(dx[period:], period)
        if len(adx) > 0:
            return round(float(adx[-1]), 2)
        return None
    except Exception:
        return None
