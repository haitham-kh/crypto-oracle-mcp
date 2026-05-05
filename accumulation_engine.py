from __future__ import annotations
"""
ACCUMULATION / DISTRIBUTION ENGINE — Multi-timescale CVD Analysis
=================================================================
Detects stealth accumulation and distribution using cumulative volume delta
at 15m, 1h, and 4h independently. Also builds volume-at-price profile
as a proxy for real accumulation zones.

Key outputs:
  - Accumulation probability score (0-100)
  - Distribution probability score (0-100)
  - Volume clustering at price (mini VPVR)
  - Multi-timescale CVD divergence signals
"""

import numpy as np
from typing import Dict, List, Any, Optional


# ─────────────────────────────────────────────────────────────────────────────
# CVD COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────

def compute_cvd_from_candles(candles: List[Dict]) -> Dict[str, Any]:
    """
    Compute Cumulative Volume Delta from candles.

    If taker_buy_volume / taker_sell_volume are present: use them directly.
    Otherwise: estimate delta = (close - open) / (high - low) * volume (proxy).

    Returns:
      cvd_series: list of cumulative delta values
      cvd_current: final value
      cvd_trend: 'rising' | 'falling' | 'flat'
      cvd_momentum: rate of change (last 5 vs prior 5)
    """
    if not candles or len(candles) < 10:
        return {"available": False}

    has_taker = "taker_buy_volume" in (candles[0] if candles else {})
    deltas = []

    for c in candles:
        vol = float(c.get("volume") or 0)
        if has_taker:
            tbv = float(c.get("taker_buy_volume") or 0)
            tsv = float(c.get("taker_sell_volume") or 0)
            delta = tbv - tsv
        else:
            # Proxy: candle body direction determines delta sign
            o = float(c.get("open") or 0)
            cl = float(c.get("close") or 0)
            h = float(c.get("high") or 0)
            l = float(c.get("low") or 0)
            body_range = h - l if h > l else 1e-10
            direction = (cl - o) / body_range  # -1 to +1
            delta = direction * vol
        deltas.append(delta)

    deltas = np.array(deltas)
    cvd = np.cumsum(deltas)

    if len(cvd) < 5:
        return {"available": False}

    # Trend via linear regression on last 10 values
    tail = cvd[-10:]
    x = np.arange(len(tail))
    slope = np.polyfit(x, tail, 1)[0]
    mean_abs = float(np.mean(np.abs(cvd))) if np.mean(np.abs(cvd)) > 0 else 1.0
    normalized_slope = slope / mean_abs

    if normalized_slope > 0.05:
        trend = "rising"
    elif normalized_slope < -0.05:
        trend = "falling"
    else:
        trend = "flat"

    # Momentum: last 5 vs prior 5
    if len(cvd) >= 10:
        recent_avg = float(np.mean(cvd[-5:]))
        prior_avg = float(np.mean(cvd[-10:-5]))
        momentum = recent_avg - prior_avg
        momentum_direction = "accelerating" if momentum > 0 and trend == "rising" else (
            "decelerating" if momentum < 0 and trend == "rising" else
            "accelerating_down" if momentum < 0 and trend == "falling" else "stable"
        )
    else:
        momentum = 0.0
        momentum_direction = "stable"

    return {
        "available": True,
        "cvd_series": [round(float(v), 4) for v in cvd[-20:]],  # last 20 values
        "cvd_current": round(float(cvd[-1]), 4),
        "cvd_trend": trend,
        "cvd_momentum": round(float(momentum), 4),
        "cvd_momentum_direction": momentum_direction,
        "normalized_slope": round(float(normalized_slope), 6),
        "has_taker_data": has_taker,
        "candle_count": len(candles),
    }


# ─────────────────────────────────────────────────────────────────────────────
# PRICE-CVD DIVERGENCE (MULTI-TIMESCALE)
# ─────────────────────────────────────────────────────────────────────────────

def detect_price_cvd_divergence(cvd_data: Dict, candles: List[Dict]) -> Dict[str, Any]:
    """
    Detect divergence between price and CVD using swing pivots.

    Stealth accumulation: price flat/down, CVD rising → hidden buying
    Stealth distribution: price flat/up, CVD falling → hidden selling

    Uses a robust swing-pivot approach (min separation = 6 candles).
    """
    if not cvd_data.get("available") or not candles or len(candles) < 20:
        return {"divergence": "none", "confidence": 0}

    closes = np.array([float(c.get("close") or 0) for c in candles])
    cvd_series = np.array(cvd_data.get("cvd_series") or [])

    if len(cvd_series) < 10:
        return {"divergence": "none", "confidence": 0}

    # Align lengths — use last N of both
    n = min(len(closes), len(cvd_series))
    closes = closes[-n:]
    cvd_series = cvd_series[-n:]

    order = 4  # minimum pivot separation

    def find_pivots(arr):
        lows, highs = [], []
        for i in range(order, len(arr) - order):
            window = arr[i - order: i + order + 1]
            if arr[i] == min(window):
                lows.append((i, float(arr[i])))
            if arr[i] == max(window):
                highs.append((i, float(arr[i])))
        return lows, highs

    price_lows, price_highs = find_pivots(closes)
    cvd_lows, cvd_highs = find_pivots(cvd_series)

    # Bullish divergence: price lower low, CVD higher low
    divergence = "none"
    confidence = 0

    if len(price_lows) >= 2 and len(cvd_lows) >= 2:
        pl1, pl2 = price_lows[-2], price_lows[-1]
        if pl2[0] - pl1[0] >= 6:
            cl1 = min(cvd_lows, key=lambda x: abs(x[0] - pl1[0]))
            cl2 = min(cvd_lows, key=lambda x: abs(x[0] - pl2[0]))
            if (pl2[1] < pl1[1] and cl2[1] > cl1[1] and
                    abs(cl1[0] - pl1[0]) <= 4 and abs(cl2[0] - pl2[0]) <= 4):
                price_drop_pct = abs(pl2[1] - pl1[1]) / pl1[1] * 100 if pl1[1] > 0 else 0
                cvd_rise_pct = abs(cl2[1] - cl1[1]) / (abs(cl1[1]) + 1e-10) * 100
                divergence = "bullish_accumulation"
                confidence = min(100, int(price_drop_pct * 5 + cvd_rise_pct * 3))

    # Bearish divergence: price higher high, CVD lower high
    if divergence == "none" and len(price_highs) >= 2 and len(cvd_highs) >= 2:
        ph1, ph2 = price_highs[-2], price_highs[-1]
        if ph2[0] - ph1[0] >= 6:
            ch1 = min(cvd_highs, key=lambda x: abs(x[0] - ph1[0]))
            ch2 = min(cvd_highs, key=lambda x: abs(x[0] - ph2[0]))
            if (ph2[1] > ph1[1] and ch2[1] < ch1[1] and
                    abs(ch1[0] - ph1[0]) <= 4 and abs(ch2[0] - ph2[0]) <= 4):
                price_rise_pct = abs(ph2[1] - ph1[1]) / ph1[1] * 100 if ph1[1] > 0 else 0
                cvd_drop_pct = abs(ch2[1] - ch1[1]) / (abs(ch1[1]) + 1e-10) * 100
                divergence = "bearish_distribution"
                confidence = min(100, int(price_rise_pct * 5 + cvd_drop_pct * 3))

    interpretations = {
        "bullish_accumulation": "CVD rising while price flat/down — stealth buying, expect delayed upward move",
        "bearish_distribution": "CVD falling while price flat/up — stealth selling, expect delayed downward move",
        "none": "No CVD-price divergence detected",
    }

    return {
        "divergence": divergence,
        "confidence": confidence,
        "interpretation": interpretations.get(divergence, ""),
    }


# ─────────────────────────────────────────────────────────────────────────────
# VOLUME CLUSTERING AT PRICE (Mini VPVR)
# ─────────────────────────────────────────────────────────────────────────────

def compute_volume_profile(candles: List[Dict], n_buckets: int = 20) -> Dict[str, Any]:
    """
    Volume Profile / VPVR proxy: where is volume clustered in price space?

    High-volume price zones = proven support/resistance with real transaction history.
    Low-volume zones = fragile, fast-move potential.

    Returns:
      value_area_high, value_area_low (70% of volume lives here)
      point_of_control (price with most volume)
      high_volume_nodes: price levels with > 150% avg volume
      low_volume_nodes: price levels with < 50% avg volume
    """
    if not candles or len(candles) < 10:
        return {"available": False}

    closes = [float(c.get("close") or 0) for c in candles]
    highs = [float(c.get("high") or 0) for c in candles]
    lows = [float(c.get("low") or 0) for c in candles]
    volumes = [float(c.get("volume") or 0) for c in candles]

    price_min = min(lows)
    price_max = max(highs)
    if price_max <= price_min:
        return {"available": False}

    bucket_size = (price_max - price_min) / n_buckets
    volume_by_bucket = [0.0] * n_buckets

    for i, c in enumerate(candles):
        # Distribute candle volume across the price range it covered
        lo = float(c.get("low") or 0)
        hi = float(c.get("high") or 0)
        vol = volumes[i]
        if hi == lo or vol == 0:
            continue
        for b in range(n_buckets):
            bucket_lo = price_min + b * bucket_size
            bucket_hi = bucket_lo + bucket_size
            overlap = max(0, min(hi, bucket_hi) - max(lo, bucket_lo))
            frac = overlap / (hi - lo)
            volume_by_bucket[b] += vol * frac

    total_vol = sum(volume_by_bucket)
    if total_vol == 0:
        return {"available": False}

    avg_bucket_vol = total_vol / n_buckets

    # Point of control
    poc_idx = int(np.argmax(volume_by_bucket))
    poc_price = price_min + (poc_idx + 0.5) * bucket_size

    # Value area: 70% of total volume around POC
    sorted_buckets = sorted(enumerate(volume_by_bucket), key=lambda x: -x[1])
    va_vol = 0.0
    va_indices = []
    for idx, vol in sorted_buckets:
        va_vol += vol
        va_indices.append(idx)
        if va_vol >= total_vol * 0.70:
            break
    va_lo = price_min + min(va_indices) * bucket_size
    va_hi = price_min + (max(va_indices) + 1) * bucket_size

    # High and low volume nodes
    hvn = []
    lvn = []
    for b in range(n_buckets):
        bucket_price = price_min + (b + 0.5) * bucket_size
        ratio = volume_by_bucket[b] / avg_bucket_vol if avg_bucket_vol > 0 else 1.0
        if ratio > 1.5:
            hvn.append({"price": round(bucket_price, 8), "volume_ratio": round(ratio, 2)})
        elif ratio < 0.5:
            lvn.append({"price": round(bucket_price, 8), "volume_ratio": round(ratio, 2)})

    return {
        "available": True,
        "point_of_control": round(poc_price, 8),
        "value_area_high": round(va_hi, 8),
        "value_area_low": round(va_lo, 8),
        "high_volume_nodes": sorted(hvn, key=lambda x: -x["volume_ratio"])[:5],
        "low_volume_nodes": sorted(lvn, key=lambda x: x["volume_ratio"])[:5],
        "candle_count": len(candles),
        "price_range": {"low": round(price_min, 8), "high": round(price_max, 8)},
    }


# ─────────────────────────────────────────────────────────────────────────────
# ACCUMULATION / DISTRIBUTION PROBABILITY SCORES
# ─────────────────────────────────────────────────────────────────────────────

def compute_accum_distribution_scores(
    cvd_15m: Dict,
    cvd_1h: Dict,
    cvd_4h: Dict,
    div_15m: Dict,
    div_1h: Dict,
    div_4h: Dict,
    volume_profile: Dict,
    current_price: float,
) -> Dict[str, Any]:
    """
    Synthesize all CVD signals into accumulation and distribution probability scores.

    Accumulation score (0-100):
      - Multiple timeframes show CVD rising + price flat or slightly down
      - CVD divergence detected on higher timeframes
      - Price near value area low or HVN support

    Distribution score (0-100):
      - Multiple timeframes show CVD falling + price flat or slightly up
      - CVD bearish divergence on higher timeframes
      - Price near value area high or resistance
    """
    accum_score = 0.0
    distrib_score = 0.0
    signals = []

    # --- CVD trend contributions ---
    timeframe_weights = {"4h": 1.0, "1h": 0.6, "15m": 0.3}

    for tf, cvd, weight in [("4h", cvd_4h, 1.0), ("1h", cvd_1h, 0.6), ("15m", cvd_15m, 0.3)]:
        if not cvd.get("available"):
            continue
        trend = cvd.get("cvd_trend", "flat")
        if trend == "rising":
            accum_score += 20 * weight
            signals.append(f"CVD {tf} rising (buy pressure)")
        elif trend == "falling":
            distrib_score += 20 * weight
            signals.append(f"CVD {tf} falling (sell pressure)")

    # --- CVD divergence contributions ---
    for tf, div, weight in [("4h", div_4h, 1.5), ("1h", div_1h, 1.0), ("15m", div_15m, 0.5)]:
        divergence = div.get("divergence", "none")
        conf = div.get("confidence", 0) / 100.0
        if divergence == "bullish_accumulation":
            accum_score += 25 * weight * conf
            signals.append(f"CVD/price divergence {tf}: stealth accumulation")
        elif divergence == "bearish_distribution":
            distrib_score += 25 * weight * conf
            signals.append(f"CVD/price divergence {tf}: stealth distribution")

    # --- Volume profile context ---
    if volume_profile.get("available") and current_price:
        val = volume_profile.get("value_area_low")
        vah = volume_profile.get("value_area_high")
        poc = volume_profile.get("point_of_control")
        if val and current_price < val * 1.02:
            accum_score += 10
            signals.append("Price at/below value area low — high-vol support zone")
        if vah and current_price > vah * 0.98:
            distrib_score += 10
            signals.append("Price at/above value area high — high-vol resistance zone")
        if poc and abs(current_price - poc) / poc < 0.01:
            signals.append("Price at Point of Control — maximum volume area, expect decision")

    # Normalize to 0-100
    accum_score = min(100.0, accum_score)
    distrib_score = min(100.0, distrib_score)

    # Dominant signal
    if accum_score > distrib_score + 20:
        dominant = "accumulation"
    elif distrib_score > accum_score + 20:
        dominant = "distribution"
    else:
        dominant = "neutral"

    return {
        "accumulation_probability": round(accum_score, 1),
        "distribution_probability": round(distrib_score, 1),
        "dominant_signal": dominant,
        "signal_reasons": signals,
        "interpretation": {
            "accumulation": "Strong evidence of stealth buying — price likely to follow CVD upward",
            "distribution": "Strong evidence of stealth selling — price likely to follow CVD downward",
            "neutral": "No clear accumulation or distribution signal",
        }.get(dominant, ""),
    }


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def analyze_accumulation(
    candles_15m: List[Dict],
    candles_1h: List[Dict],
    candles_4h: List[Dict],
    current_price: float,
) -> Dict[str, Any]:
    """
    Full accumulation/distribution analysis. Entry point for tool integration.

    Args:
        candles_15m, candles_1h, candles_4h: OHLCV with optional taker volumes
        current_price: Current market price

    Returns:
        Structured accumulation/distribution report
    """
    cvd_15m = compute_cvd_from_candles(candles_15m)
    cvd_1h = compute_cvd_from_candles(candles_1h)
    cvd_4h = compute_cvd_from_candles(candles_4h)

    div_15m = detect_price_cvd_divergence(cvd_15m, candles_15m)
    div_1h = detect_price_cvd_divergence(cvd_1h, candles_1h)
    div_4h = detect_price_cvd_divergence(cvd_4h, candles_4h)

    # Use 4h candles for volume profile (most representative)
    vprofile = compute_volume_profile(candles_4h or candles_1h, n_buckets=20)

    scores = compute_accum_distribution_scores(
        cvd_15m, cvd_1h, cvd_4h,
        div_15m, div_1h, div_4h,
        vprofile, current_price
    )

    return {
        "cvd": {"15m": cvd_15m, "1h": cvd_1h, "4h": cvd_4h},
        "divergence": {"15m": div_15m, "1h": div_1h, "4h": div_4h},
        "volume_profile": vprofile,
        "scores": scores,
        "summary": {
            "accumulation_probability": scores["accumulation_probability"],
            "distribution_probability": scores["distribution_probability"],
            "dominant_signal": scores["dominant_signal"],
            "top_reason": scores["signal_reasons"][0] if scores["signal_reasons"] else "No signal",
        },
    }
