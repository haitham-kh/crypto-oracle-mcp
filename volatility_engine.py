from __future__ import annotations
"""
VOLATILITY ENGINE — Trend Sustainability & Continuation Probability
===================================================================
Replaces ATR-only volatility with a full volatility regime + trend
persistence model. Outputs probabilities of continuation vs mean reversion.

Key outputs:
  - Realized volatility (rolling, annualized)
  - Volatility expansion/contraction state
  - Trend persistence score
  - P(trend continuation) vs P(mean reversion)
  - Failure mode flags (fake breakout, low-vol chop)
"""

import numpy as np
from typing import Dict, List, Any, Optional
from regime_classifier import compute_trend_efficiency, compute_hurst


# ─────────────────────────────────────────────────────────────────────────────
# REALIZED VOLATILITY
# ─────────────────────────────────────────────────────────────────────────────

def compute_realized_vol(prices: np.ndarray, window: int = 20) -> Dict[str, Any]:
    """
    Rolling realized volatility (annualized %).

    Uses log returns for better distributional properties.
    Returns multiple windows to detect vol regime transitions.
    """
    if prices is None or len(prices) < window + 2:
        return {"available": False}

    log_returns = np.diff(np.log(prices + 1e-10))

    def _ann_vol(ret_slice):
        return float(np.std(ret_slice) * np.sqrt(365) * 100)

    rv_5 = _ann_vol(log_returns[-5:]) if len(log_returns) >= 5 else None
    rv_20 = _ann_vol(log_returns[-20:]) if len(log_returns) >= 20 else None
    rv_60 = _ann_vol(log_returns[-60:]) if len(log_returns) >= 60 else None

    # Vol of vol (how much volatility itself is changing)
    if len(log_returns) >= 10:
        rolling_stds = [np.std(log_returns[max(0, i-5):i]) * np.sqrt(365) * 100
                        for i in range(5, len(log_returns) + 1)]
        vol_of_vol = float(np.std(rolling_stds)) if len(rolling_stds) > 1 else 0.0
    else:
        vol_of_vol = None

    # Current vs long-term vol
    if rv_5 and rv_60:
        vol_ratio = rv_5 / rv_60
        if vol_ratio > 1.5:
            state = "expanding"
        elif vol_ratio < 0.6:
            state = "contracting"
        else:
            state = "stable"
    else:
        vol_ratio = None
        state = "unknown"

    return {
        "available": True,
        "realized_vol_5d_pct": round(rv_5, 2) if rv_5 else None,
        "realized_vol_20d_pct": round(rv_20, 2) if rv_20 else None,
        "realized_vol_60d_pct": round(rv_60, 2) if rv_60 else None,
        "vol_state": state,
        "vol_ratio_short_vs_long": round(vol_ratio, 3) if vol_ratio else None,
        "vol_of_vol_annualized_pct": round(vol_of_vol, 2) if vol_of_vol is not None else None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# TREND PERSISTENCE METRICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_trend_persistence(candles: List[Dict]) -> Dict[str, Any]:
    """
    Quantify how persistent the current trend is.

    Metrics:
    1. Trend efficiency ratio (Kaufman): 0-1, higher = more directional
    2. Consecutive same-direction closes: trend momentum
    3. Average candle body / total range: noise ratio
    4. Hurst exponent: persistent (> 0.55) vs mean-reverting (< 0.45)
    """
    if not candles or len(candles) < 15:
        return {"available": False}

    closes = np.array([float(c.get("close") or 0) for c in candles])
    opens = np.array([float(c.get("open") or 0) for c in candles])
    highs = np.array([float(c.get("high") or 0) for c in candles])
    lows = np.array([float(c.get("low") or 0) for c in candles])

    # Trend efficiency ratio
    er = compute_trend_efficiency(closes, window=14)

    # Consecutive same-direction closes (last 10)
    directions = [1 if closes[i] > closes[i-1] else -1 for i in range(1, len(closes))]
    directions = directions[-10:]
    max_run = current_run = 0
    run_direction = 0
    for d in directions:
        if d == run_direction:
            current_run += 1
        else:
            current_run = 1
            run_direction = d
        max_run = max(max_run, current_run)

    recent_direction = run_direction
    run_length = current_run

    # Body/range ratio (noise metric): high = directional, low = choppy
    bodies = np.abs(closes - opens)
    ranges = highs - lows
    with np.errstate(divide="ignore", invalid="ignore"):
        body_ratios = np.where(ranges > 0, bodies / ranges, 0.0)
    avg_body_ratio = float(np.mean(body_ratios[-14:])) if len(body_ratios) >= 14 else float(np.mean(body_ratios))

    # Hurst
    hurst = compute_hurst(closes)

    # Composite persistence score (0-100)
    persistence_score = 0.0
    if er is not None:
        persistence_score += er * 35
    if hurst is not None:
        normalized_hurst = max(0, (hurst - 0.4) / 0.3)  # 0 at H=0.4, 1 at H=0.7+
        persistence_score += normalized_hurst * 35
    persistence_score += min(run_length / 7, 1.0) * 15
    persistence_score += avg_body_ratio * 15

    return {
        "available": True,
        "trend_efficiency_ratio": er,
        "hurst_exponent": hurst,
        "current_run_length": run_length,
        "current_run_direction": "up" if recent_direction == 1 else "down",
        "avg_body_range_ratio": round(avg_body_ratio, 3),
        "persistence_score": round(min(100, persistence_score), 1),
        "interpretation": (
            "Strong trend persistence — continuation more likely than reversal"
            if persistence_score > 60
            else "Moderate persistence" if persistence_score > 40
            else "Low persistence — choppy, mean reversion favored"
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# BREAKOUT QUALITY ASSESSMENT
# ─────────────────────────────────────────────────────────────────────────────

def assess_breakout_quality(candles: List[Dict], breakout_level: Optional[float] = None) -> Dict[str, Any]:
    """
    Is a recent breakout real or fake?

    Real breakout indicators:
    - Volume expands (> 1.5× avg) on and after the breakout candle
    - OFI confirms direction (if trade data available)
    - Price closes convincingly beyond the level (> 0.3% beyond)
    - Multiple candles sustain beyond the level

    Fake breakout indicators:
    - Volume spike on breakout but immediately drops
    - Price returns to level within 2-3 candles
    - Wick rejection (wick > 2× body on breakout candle)
    """
    if not candles or len(candles) < 10:
        return {"available": False}

    closes = [float(c.get("close") or 0) for c in candles]
    volumes = [float(c.get("volume") or 0) for c in candles]
    opens = [float(c.get("open") or 0) for c in candles]
    highs = [float(c.get("high") or 0) for c in candles]
    lows = [float(c.get("low") or 0) for c in candles]

    avg_vol_20 = float(np.mean(volumes[-20:])) if len(volumes) >= 20 else float(np.mean(volumes))
    if avg_vol_20 == 0:
        return {"available": False}

    # Last 3 candle volume signature
    recent_vols = volumes[-3:]
    vol_ratios = [v / avg_vol_20 for v in recent_vols]
    vol_expanding = vol_ratios[-1] > 1.5

    # Last candle structure
    last_close = closes[-1]
    last_open = opens[-1]
    last_high = highs[-1]
    last_low = lows[-1]
    body = abs(last_close - last_open)
    full_range = last_high - last_low
    upper_wick = last_high - max(last_close, last_open)
    lower_wick = min(last_close, last_open) - last_low

    wick_rejection = (upper_wick > body * 2) or (lower_wick > body * 2) if body > 0 else False

    # Close strength (how far into range the close is)
    if full_range > 0:
        if last_close > last_open:  # Bullish candle
            close_strength = (last_close - last_low) / full_range
        else:  # Bearish candle
            close_strength = (last_high - last_close) / full_range
    else:
        close_strength = 0.5

    # If breakout_level provided, check follow-through
    follow_through = None
    if breakout_level and last_close > 0:
        pct_beyond = (last_close - breakout_level) / breakout_level * 100
        follow_through = {
            "pct_beyond_level": round(pct_beyond, 3),
            "convincing": abs(pct_beyond) > 0.3,
        }

    quality_score = 0
    if vol_expanding:
        quality_score += 40
    if not wick_rejection:
        quality_score += 25
    if close_strength > 0.6:
        quality_score += 25
    if follow_through and follow_through.get("convincing"):
        quality_score += 10

    return {
        "available": True,
        "quality_score": quality_score,
        "quality_label": (
            "high" if quality_score >= 70 else
            "medium" if quality_score >= 40 else "low"
        ),
        "volume_expanding": vol_expanding,
        "vol_ratios_last_3": [round(v, 2) for v in vol_ratios],
        "wick_rejection": wick_rejection,
        "close_strength": round(close_strength, 3),
        "follow_through": follow_through,
        "interpretation": (
            "Strong breakout — volume + close structure confirm direction"
            if quality_score >= 70
            else "Moderate breakout — monitor for follow-through volume"
            if quality_score >= 40
            else "Weak/fake breakout suspected — volume and structure do not confirm"
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# CONTINUATION vs MEAN REVERSION PROBABILITY
# ─────────────────────────────────────────────────────────────────────────────

def compute_direction_probability(
    persistence: Dict,
    vol_data: Dict,
    regime: str,
    ofi_score: float = 0.0,
    accum_score: float = 50.0,
) -> Dict[str, Any]:
    """
    Combine persistence, volatility, and flow signals to estimate:
    - P(trend continuation)
    - P(mean reversion)

    These are heuristic base rates — they MUST be calibrated against
    historical data via feature_validator.py before being used in sizing.

    The model here is:
    - Base rate: 55% continuation in trending, 50% in ranging
    - Adjusted by persistence score (higher = more continuation)
    - Adjusted by vol state (expansion favors continuation of last move)
    - Adjusted by flow (OFI and CVD supporting vs opposing)
    """
    # Base rates by regime
    base_rates = {
        "TRENDING_UP": 0.58,
        "TRENDING_DOWN": 0.58,
        "RANGING": 0.50,
        "EXPANSION": 0.55,
        "LOW_LIQUIDITY": 0.50,
        "UNKNOWN": 0.50,
    }
    p_base = base_rates.get(regime, 0.50)

    adjustments: List[Dict] = []

    # Persistence adjustment: +/- 8%
    persist_score = persistence.get("persistence_score", 50) if persistence.get("available") else 50
    persist_adj = (persist_score - 50) / 50 * 0.08
    p_base += persist_adj
    adjustments.append({"factor": "trend_persistence", "adjustment": round(persist_adj, 4)})

    # Vol expansion: favors continuation by up to +5%
    vol_state = vol_data.get("vol_state") if vol_data.get("available") else "stable"
    if vol_state == "expanding":
        p_base += 0.04
        adjustments.append({"factor": "vol_expanding", "adjustment": 0.04})
    elif vol_state == "contracting":
        p_base -= 0.02
        adjustments.append({"factor": "vol_contracting", "adjustment": -0.02})

    # OFI alignment: up to +/- 6% if strong signal
    ofi_adj = max(-0.06, min(0.06, ofi_score / 100 * 0.10))
    p_base += ofi_adj
    adjustments.append({"factor": "ofi_alignment", "adjustment": round(ofi_adj, 4)})

    # Accumulation/distribution: up to +/- 6%
    accum_adj = (accum_score - 50) / 50 * 0.06
    p_base += accum_adj
    adjustments.append({"factor": "accum_distribution", "adjustment": round(accum_adj, 4)})

    p_continuation = float(np.clip(p_base, 0.20, 0.80))
    p_mean_reversion = 1.0 - p_continuation

    return {
        "p_continuation": round(p_continuation, 4),
        "p_mean_reversion": round(p_mean_reversion, 4),
        "p_continuation_pct": round(p_continuation * 100, 1),
        "p_mean_reversion_pct": round(p_mean_reversion * 100, 1),
        "adjustments": adjustments,
        "calibration_warning": (
            "⚠️ These probabilities are heuristic base rates. "
            "They require empirical backtesting via feature_validator.py to be trusted."
        ),
        "dominant_outcome": (
            "continuation" if p_continuation > 0.55
            else "mean_reversion" if p_mean_reversion > 0.55
            else "uncertain"
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def analyze_volatility_sustainability(
    candles: List[Dict],
    regime: str,
    ofi_score: float = 0.0,
    accum_score: float = 50.0,
) -> Dict[str, Any]:
    """
    Full volatility and trend sustainability analysis.

    Args:
        candles: OHLCV candles (4H recommended)
        regime: Current regime string from regime_classifier
        ofi_score: Net buying pressure score from spot_flow_engine (-100..+100)
        accum_score: Accumulation probability from accumulation_engine (0..100)
    """
    closes = np.array([float(c.get("close") or 0) for c in candles]) if candles else np.array([])

    rv = compute_realized_vol(closes) if len(closes) >= 10 else {"available": False}
    persistence = compute_trend_persistence(candles)
    breakout_quality = assess_breakout_quality(candles)
    direction_probs = compute_direction_probability(
        persistence, rv, regime, ofi_score, accum_score
    )

    return {
        "realized_vol": rv,
        "trend_persistence": persistence,
        "breakout_quality": breakout_quality,
        "direction_probabilities": direction_probs,
        "summary": {
            "vol_state": rv.get("vol_state") if rv.get("available") else "unknown",
            "persistence_score": persistence.get("persistence_score") if persistence.get("available") else None,
            "p_continuation_pct": direction_probs["p_continuation_pct"],
            "dominant_outcome": direction_probs["dominant_outcome"],
        },
    }
