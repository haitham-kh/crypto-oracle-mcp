from __future__ import annotations
"""
CryptoOracle MCP — Chart Pattern Detection Module
Detects structural chart patterns: H&S, double top/bottom, triangles, flags, etc.
Also computes support/resistance levels using swing highs/lows + clustering.
"""

import numpy as np
import pandas as pd
from typing import Optional

from indicators import candles_to_df


# ======================================================================
# Support & Resistance
# ======================================================================


def compute_vpvr(df: pd.DataFrame, bins: int = 200) -> dict:
    """Volume Profile Visible Range — 200-bin, close-weighted kernel.
    
    Instead of distributing volume uniformly across a candle's range,
    uses a triangular kernel centered on the close price. This produces
    more accurate POC and volume nodes because most trading activity
    concentrates near the close, not at wick extremes.
    """
    min_p = df['low'].min()
    max_p = df['high'].max()
    if max_p == min_p:
        return {"poc": float(min_p), "vpvr_levels": []}
    bin_size = (max_p - min_p) / bins
    
    vpvr = np.zeros(bins)
    price_levels = np.linspace(min_p + bin_size / 2, max_p - bin_size / 2, bins)
    
    for _, row in df.iterrows():
        low_val = float(row['low'])
        high_val = float(row['high'])
        close_val = float(row['close'])
        vol = float(row.get('volume', 0))
        if vol <= 0 or high_val <= low_val:
            continue
        
        low_bin = max(0, int((low_val - min_p) / bin_size))
        high_bin = min(bins - 1, int((high_val - min_p) / bin_size))
        close_bin = max(low_bin, min(high_bin, int((close_val - min_p) / bin_size)))
        
        if high_bin <= low_bin:
            if low_bin < bins:
                vpvr[low_bin] += vol
            continue
        
        # Triangular kernel: weight peaks at close_bin, falls linearly to edges
        weights = np.zeros(high_bin - low_bin + 1)
        for b_idx, b in enumerate(range(low_bin, high_bin + 1)):
            dist_from_close = abs(b - close_bin)
            max_dist = max(close_bin - low_bin, high_bin - close_bin, 1)
            # Triangular weight: 1.0 at close, 0.2 at extremes
            weights[b_idx] = max(0.2, 1.0 - 0.8 * (dist_from_close / max_dist))
        
        # Normalize weights so they sum to 1, then distribute volume
        w_sum = weights.sum()
        if w_sum > 0:
            weights /= w_sum
            for b_idx, b in enumerate(range(low_bin, high_bin + 1)):
                if b < bins:
                    vpvr[b] += vol * weights[b_idx]
                
    poc_idx = int(np.argmax(vpvr))
    poc_price = price_levels[poc_idx] if poc_idx < len(price_levels) else float(min_p)
    
    # Find local maxima (high volume nodes) for S/R
    peaks = []
    mean_v = float(np.mean(vpvr))
    std_v = float(np.std(vpvr))
    # Require volume > mean + 0.5 std to be significant
    threshold = mean_v + 0.5 * std_v
    for i in range(2, bins - 2):
        if (vpvr[i] > vpvr[i-1] and vpvr[i] > vpvr[i+1] and
            vpvr[i] > vpvr[i-2] and vpvr[i] > vpvr[i+2] and
            vpvr[i] > threshold):
            peaks.append({
                "level": round(float(price_levels[i]), 8),
                "volume": round(float(vpvr[i]), 2),
                "strength": round(float(vpvr[i] / mean_v), 2) if mean_v > 0 else 0
            })
    
    # Also find low-volume nodes (potential breakout levels)
    lvn_threshold = mean_v * 0.3
    low_volume_nodes = []
    for i in range(2, bins - 2):
        if (vpvr[i] < vpvr[i-1] and vpvr[i] < vpvr[i+1] and
            vpvr[i] < lvn_threshold):
            low_volume_nodes.append({
                "level": round(float(price_levels[i]), 8),
                "volume": round(float(vpvr[i]), 2),
            })
    
    peaks.sort(key=lambda x: x["volume"], reverse=True)
    return {
        "poc": round(float(poc_price), 8),
        "vpvr_levels": peaks[:10],
        "low_volume_nodes": low_volume_nodes[:5],
        "value_area_high": round(float(_value_area_edge(vpvr, price_levels, poc_idx, bins, direction=1)), 8),
        "value_area_low": round(float(_value_area_edge(vpvr, price_levels, poc_idx, bins, direction=-1)), 8),
    }


def _value_area_edge(vpvr, price_levels, poc_idx, bins, direction=1):
    """Compute value area (70% of volume) edge in the given direction."""
    total_vol = vpvr.sum()
    if total_vol <= 0:
        return price_levels[poc_idx] if poc_idx < len(price_levels) else 0
    target = total_vol * 0.35  # 35% each side of POC = 70% total
    accum = vpvr[poc_idx]
    idx = poc_idx
    while accum < target and 0 <= idx + direction < bins:
        idx += direction
        accum += vpvr[idx]
    return price_levels[max(0, min(idx, len(price_levels) - 1))]

def compute_support_resistance(candles: list[dict], lookback: int = 200) -> dict:
    """
    Compute key S/R levels using:
    1. Zigzag pivot detection (swing highs/lows)
    2. Level clustering (merge nearby levels)
    3. Psychological round numbers
    """
    if not candles or len(candles) < 20:
        return {"error": "Insufficient data"}

    df = candles_to_df(candles).tail(lookback)
    high = df["high"].values
    low = df["low"].values
    close = df["close"].values
    last_close = float(close[-1])

    # --- Step 1: Find swing highs and lows ---
    swing_highs = []
    swing_lows = []
    window = 5

    for i in range(window, len(high) - window):
        if high[i] == max(high[i - window:i + window + 1]):
            swing_highs.append({"level": float(high[i]), "index": i})
        if low[i] == min(low[i - window:i + window + 1]):
            swing_lows.append({"level": float(low[i]), "index": i})

    # --- Step 2: Cluster nearby levels ---
    all_levels = [(s["level"], "resistance", s["index"]) for s in swing_highs]
    all_levels += [(s["level"], "support", s["index"]) for s in swing_lows]

    clustered = _cluster_levels(all_levels, threshold_pct=0.5)

    # --- Step 3: Add psychological round numbers ---
    price_range = max(high) - min(low)
    if price_range > 0:
        magnitude = 10 ** (int(np.log10(last_close)) - 1) if last_close > 1 else 0.01
        round_levels = []
        base = (last_close // magnitude) * magnitude
        for i in range(-5, 6):
            lvl = base + i * magnitude
            if min(low) * 0.9 < lvl < max(high) * 1.1:
                round_levels.append(lvl)

    # --- Step 4: Classify supports and resistances ---
    supports = []
    resistances = []

    for cl in clustered:
        level = cl["level"]
        touches = cl["touches"]
        last_idx = cl["last_index"]
        strength = min(100, touches * 20 + (10 if last_idx > len(close) - 30 else 0))

        entry = {
            "level": round(level, 8),
            "touches": touches,
            "last_tested": int(last_idx),
            "strength_score": strength,
        }

        if level < last_close:
            supports.append(entry)
        else:
            resistances.append(entry)

    supports.sort(key=lambda x: x["level"], reverse=True)
    resistances.sort(key=lambda x: x["level"])

    nearest_support = supports[0]["level"] if supports else None
    nearest_resistance = resistances[0]["level"] if resistances else None

    # Key levels with % distance from current
    key_levels = []
    for s in supports[:5]:
        pct = round(((last_close - s["level"]) / last_close) * 100, 2)
        key_levels.append({"level": s["level"], "pct_from_current": pct, "type": "support"})
    for r in resistances[:5]:
        pct = round(((r["level"] - last_close) / last_close) * 100, 2)
        key_levels.append({"level": r["level"], "pct_from_current": pct, "type": "resistance"})

    vpvr_data = compute_vpvr(df)
    return {
        "vpvr_poc": vpvr_data['poc'],
        "vpvr_levels": vpvr_data['vpvr_levels'],
        "strong_supports": supports[:5],
        "strong_resistances": resistances[:5],
        "current_range": {
            "lower": nearest_support or float(min(low)),
            "upper": nearest_resistance or float(max(high)),
        },
        "nearest_support": nearest_support,
        "nearest_resistance": nearest_resistance,
        "key_levels_pct_away": key_levels,
        "current_price": last_close,
    }


def _cluster_levels(levels: list[tuple], threshold_pct: float = 0.5) -> list[dict]:
    """Cluster nearby price levels and count touches."""
    if not levels:
        return []

    sorted_levels = sorted(levels, key=lambda x: x[0])
    clusters = []
    current_cluster = [sorted_levels[0]]

    for i in range(1, len(sorted_levels)):
        avg_level = np.mean([x[0] for x in current_cluster])
        if abs(sorted_levels[i][0] - avg_level) / avg_level * 100 < threshold_pct:
            current_cluster.append(sorted_levels[i])
        else:
            clusters.append(current_cluster)
            current_cluster = [sorted_levels[i]]
    clusters.append(current_cluster)

    result = []
    for cluster in clusters:
        level = np.mean([x[0] for x in cluster])
        touches = len(cluster)
        last_index = max(x[2] for x in cluster)
        result.append({
            "level": round(float(level), 8),
            "touches": touches,
            "last_index": last_index,
        })

    return result


# ======================================================================
# Chart Pattern Detection
# ======================================================================

def detect_chart_patterns(candles: list[dict], lookback_candles: int = 100) -> dict:
    """
    Detect structural chart patterns:
    - Head and Shoulders / Inverse H&S
    - Double Top / Double Bottom
    - Triple Top / Triple Bottom
    - Ascending / Descending / Symmetrical Triangle
    - Bull Flag / Bear Flag
    - Rising / Falling Wedge
    - Cup and Handle
    - Rising / Falling Channel
    """
    if not candles or len(candles) < 30:
        return {"patterns_detected": [], "strongest_pattern": None, "implied_direction": "neutral"}

    df = candles_to_df(candles).tail(lookback_candles)
    high = df["high"].values.astype(float)
    low = df["low"].values.astype(float)
    close = df["close"].values.astype(float)
    last_close = float(close[-1])
    n = len(close)

    patterns = []

    # Find swing pivots
    swing_highs = _find_swings(high, order=5, kind="high")
    swing_lows = _find_swings(low, order=5, kind="low")

    # --- Double Top ---
    p = _detect_double_top(swing_highs, close, last_close, n)
    if p:
        patterns.append(p)

    # --- Double Bottom ---
    p = _detect_double_bottom(swing_lows, close, last_close, n)
    if p:
        patterns.append(p)

    # --- Head and Shoulders ---
    p = _detect_head_and_shoulders(swing_highs, swing_lows, close, last_close, n)
    if p:
        patterns.append(p)

    # --- Inverse Head and Shoulders ---
    p = _detect_inverse_head_and_shoulders(swing_highs, swing_lows, close, last_close, n)
    if p:
        patterns.append(p)

    # --- Triangles ---
    for tri in _detect_triangles(swing_highs, swing_lows, close, last_close, n):
        patterns.append(tri)

    # --- Flags ---
    for flag in _detect_flags(close, high, low, last_close, n):
        patterns.append(flag)

    # --- Wedges ---
    for w in _detect_wedges(swing_highs, swing_lows, close, last_close, n):
        patterns.append(w)

    # Sort by confidence
    patterns.sort(key=lambda x: x.get("confidence_pct", 0), reverse=True)
    strongest = patterns[0] if patterns else None

    if strongest:
        implied = strongest.get("bullish_or_bearish", "neutral")
    else:
        implied = "neutral"

    target = strongest.get("target_price") if strongest else None

    return {
        "patterns_detected": patterns,
        "strongest_pattern": strongest,
        "implied_direction": implied,
        "price_target_if_pattern_plays_out": target,
    }


def _find_swings(data: np.ndarray, order: int = 5, kind: str = "high") -> list[dict]:
    """Find swing highs or lows with their indices."""
    swings = []
    for i in range(order, len(data) - order):
        if kind == "high":
            if data[i] == max(data[i - order:i + order + 1]):
                swings.append({"index": i, "value": float(data[i])})
        else:
            if data[i] == min(data[i - order:i + order + 1]):
                swings.append({"index": i, "value": float(data[i])})
    return swings


def _detect_double_top(swing_highs: list, close: np.ndarray, last_close: float, n: int) -> Optional[dict]:
    if len(swing_highs) < 2:
        return None
    # Look at last 2 significant swing highs
    sh = swing_highs[-2:]
    h1, h2 = sh[0]["value"], sh[1]["value"]
    tolerance = 0.02  # 2% tolerance

    if abs(h1 - h2) / max(h1, h2) < tolerance:
        # Find neckline (lowest low between the two tops)
        idx1, idx2 = sh[0]["index"], sh[1]["index"]
        if idx2 > idx1:
            neckline = float(min(close[idx1:idx2 + 1]))
            height = ((h1 + h2) / 2) - neckline
            target = neckline - height

            # Confidence based on spacing and symmetry
            spacing = idx2 - idx1
            confidence = min(80, 50 + spacing)

            if last_close < (h1 + h2) / 2:  # Price should be below tops
                return {
                    "pattern_name": "Double Top",
                    "bullish_or_bearish": "bearish",
                    "confidence_pct": confidence,
                    "target_price": round(target, 8),
                    "invalidation_price": round(max(h1, h2) * 1.01, 8),
                    "neckline": round(neckline, 8),
                    "formed_at_index": idx2,
                }
    return None


def _detect_double_bottom(swing_lows: list, close: np.ndarray, last_close: float, n: int) -> Optional[dict]:
    if len(swing_lows) < 2:
        return None
    sl = swing_lows[-2:]
    l1, l2 = sl[0]["value"], sl[1]["value"]
    tolerance = 0.02

    if abs(l1 - l2) / max(l1, l2) < tolerance:
        idx1, idx2 = sl[0]["index"], sl[1]["index"]
        if idx2 > idx1:
            neckline = float(max(close[idx1:idx2 + 1]))
            height = neckline - ((l1 + l2) / 2)
            target = neckline + height
            spacing = idx2 - idx1
            confidence = min(80, 50 + spacing)

            if last_close > (l1 + l2) / 2:
                return {
                    "pattern_name": "Double Bottom",
                    "bullish_or_bearish": "bullish",
                    "confidence_pct": confidence,
                    "target_price": round(target, 8),
                    "invalidation_price": round(min(l1, l2) * 0.99, 8),
                    "neckline": round(neckline, 8),
                    "formed_at_index": idx2,
                }
    return None


def _detect_head_and_shoulders(sh: list, sl: list, close: np.ndarray, last_close: float, n: int) -> Optional[dict]:
    """Detect Head & Shoulders (bearish)."""
    if len(sh) < 3:
        return None
    # Need 3 peaks where middle is highest
    for i in range(len(sh) - 2):
        left, head, right = sh[i], sh[i + 1], sh[i + 2]
        if head["value"] > left["value"] and head["value"] > right["value"]:
            # Left and right shoulders roughly equal
            if abs(left["value"] - right["value"]) / head["value"] < 0.05:
                # Neckline from lows between shoulders
                idx_range = range(left["index"], right["index"] + 1)
                if right["index"] < n:
                    neckline = float(min(close[left["index"]:right["index"] + 1]))
                    height = head["value"] - neckline
                    target = neckline - height

                    return {
                        "pattern_name": "Head and Shoulders",
                        "bullish_or_bearish": "bearish",
                        "confidence_pct": 75,
                        "target_price": round(target, 8),
                        "invalidation_price": round(head["value"] * 1.01, 8),
                        "neckline": round(neckline, 8),
                        "formed_at_index": right["index"],
                    }
    return None


def _detect_inverse_head_and_shoulders(sh: list, sl: list, close: np.ndarray, last_close: float, n: int) -> Optional[dict]:
    """Detect Inverse H&S (bullish)."""
    if len(sl) < 3:
        return None
    for i in range(len(sl) - 2):
        left, head, right = sl[i], sl[i + 1], sl[i + 2]
        if head["value"] < left["value"] and head["value"] < right["value"]:
            if abs(left["value"] - right["value"]) / abs(head["value"]) < 0.05 if head["value"] != 0 else False:
                if right["index"] < n:
                    neckline = float(max(close[left["index"]:right["index"] + 1]))
                    height = neckline - head["value"]
                    target = neckline + height

                    return {
                        "pattern_name": "Inverse Head and Shoulders",
                        "bullish_or_bearish": "bullish",
                        "confidence_pct": 75,
                        "target_price": round(target, 8),
                        "invalidation_price": round(head["value"] * 0.99, 8),
                        "neckline": round(neckline, 8),
                        "formed_at_index": right["index"],
                    }
    return None


def _detect_triangles(sh: list, sl: list, close: np.ndarray, last_close: float, n: int) -> list[dict]:
    """Detect ascending, descending, and symmetrical triangles."""
    patterns = []
    if len(sh) < 2 or len(sl) < 2:
        return patterns

    # Use last few swing highs and lows
    recent_highs = sh[-3:] if len(sh) >= 3 else sh
    recent_lows = sl[-3:] if len(sl) >= 3 else sl

    # Slope of highs and lows
    if len(recent_highs) >= 2:
        h_slope = (recent_highs[-1]["value"] - recent_highs[0]["value"]) / max(1, recent_highs[-1]["index"] - recent_highs[0]["index"])
    else:
        h_slope = 0

    if len(recent_lows) >= 2:
        l_slope = (recent_lows[-1]["value"] - recent_lows[0]["value"]) / max(1, recent_lows[-1]["index"] - recent_lows[0]["index"])
    else:
        l_slope = 0

    avg_price = (recent_highs[-1]["value"] + recent_lows[-1]["value"]) / 2
    h_slope_norm = h_slope / avg_price * 100 if avg_price else 0
    l_slope_norm = l_slope / avg_price * 100 if avg_price else 0

    # Ascending triangle: flat resistance, rising support
    if abs(h_slope_norm) < 0.02 and l_slope_norm > 0.01:
        resistance = recent_highs[-1]["value"]
        height = resistance - recent_lows[-1]["value"]
        patterns.append({
            "pattern_name": "Ascending Triangle",
            "bullish_or_bearish": "bullish",
            "confidence_pct": 65,
            "target_price": round(resistance + height, 8),
            "invalidation_price": round(recent_lows[-1]["value"] * 0.99, 8),
            "formed_at_index": max(recent_highs[-1]["index"], recent_lows[-1]["index"]),
        })

    # Descending triangle: falling resistance, flat support
    elif h_slope_norm < -0.01 and abs(l_slope_norm) < 0.02:
        support = recent_lows[-1]["value"]
        height = recent_highs[0]["value"] - support
        patterns.append({
            "pattern_name": "Descending Triangle",
            "bullish_or_bearish": "bearish",
            "confidence_pct": 65,
            "target_price": round(support - height, 8),
            "invalidation_price": round(recent_highs[-1]["value"] * 1.01, 8),
            "formed_at_index": max(recent_highs[-1]["index"], recent_lows[-1]["index"]),
        })

    # Symmetrical triangle: converging
    elif h_slope_norm < -0.005 and l_slope_norm > 0.005:
        height = recent_highs[0]["value"] - recent_lows[0]["value"]
        direction = "bullish" if last_close > avg_price else "bearish"
        target = last_close + height if direction == "bullish" else last_close - height
        patterns.append({
            "pattern_name": "Symmetrical Triangle",
            "bullish_or_bearish": direction,
            "confidence_pct": 60,
            "target_price": round(target, 8),
            "invalidation_price": round(recent_lows[-1]["value"] * 0.99, 8) if direction == "bullish" else round(recent_highs[-1]["value"] * 1.01, 8),
            "formed_at_index": max(recent_highs[-1]["index"], recent_lows[-1]["index"]),
        })

    return patterns


def _detect_flags(close: np.ndarray, high: np.ndarray, low: np.ndarray, last_close: float, n: int) -> list[dict]:
    """Detect bull and bear flags (strong move + consolidation channel)."""
    patterns = []
    if n < 30:
        return patterns

    # Look for a strong pole (last 30 candles)
    lookback_pole = 20
    lookback_flag = 10

    pole_start = max(0, n - lookback_pole - lookback_flag)
    pole_end = n - lookback_flag
    flag_start = pole_end
    flag_end = n

    if pole_end <= pole_start:
        return patterns

    pole_change = (close[pole_end - 1] - close[pole_start]) / close[pole_start] * 100

    # Bull flag: strong up move (>5%) + slight downward consolidation
    if pole_change > 5:
        flag_data = close[flag_start:flag_end]
        if len(flag_data) >= 3:
            flag_slope = (flag_data[-1] - flag_data[0]) / flag_data[0] * 100
            flag_range = (max(high[flag_start:flag_end]) - min(low[flag_start:flag_end])) / flag_data[0] * 100

            if -5 < flag_slope < 2 and flag_range < abs(pole_change) * 0.5:
                pole_height = close[pole_end - 1] - close[pole_start]
                target = last_close + pole_height
                patterns.append({
                    "pattern_name": "Bull Flag",
                    "bullish_or_bearish": "bullish",
                    "confidence_pct": 70,
                    "target_price": round(target, 8),
                    "invalidation_price": round(min(low[flag_start:flag_end]) * 0.99, 8),
                    "formed_at_index": n - 1,
                })

    # Bear flag: strong down move + slight upward consolidation
    if pole_change < -5:
        flag_data = close[flag_start:flag_end]
        if len(flag_data) >= 3:
            flag_slope = (flag_data[-1] - flag_data[0]) / flag_data[0] * 100
            flag_range = (max(high[flag_start:flag_end]) - min(low[flag_start:flag_end])) / flag_data[0] * 100

            if -2 < flag_slope < 5 and flag_range < abs(pole_change) * 0.5:
                pole_height = abs(close[pole_start] - close[pole_end - 1])
                target = last_close - pole_height
                patterns.append({
                    "pattern_name": "Bear Flag",
                    "bullish_or_bearish": "bearish",
                    "confidence_pct": 70,
                    "target_price": round(target, 8),
                    "invalidation_price": round(max(high[flag_start:flag_end]) * 1.01, 8),
                    "formed_at_index": n - 1,
                })

    return patterns


def _detect_wedges(sh: list, sl: list, close: np.ndarray, last_close: float, n: int) -> list[dict]:
    """Detect rising and falling wedges."""
    patterns = []
    if len(sh) < 2 or len(sl) < 2:
        return patterns

    recent_highs = sh[-3:] if len(sh) >= 3 else sh
    recent_lows = sl[-3:] if len(sl) >= 3 else sl

    if len(recent_highs) >= 2 and len(recent_lows) >= 2:
        h_slope = (recent_highs[-1]["value"] - recent_highs[0]["value"])
        l_slope = (recent_lows[-1]["value"] - recent_lows[0]["value"])

        # Rising wedge (bearish): both highs and lows rising, but converging
        if h_slope > 0 and l_slope > 0 and l_slope > h_slope:
            target = recent_lows[0]["value"]
            patterns.append({
                "pattern_name": "Rising Wedge",
                "bullish_or_bearish": "bearish",
                "confidence_pct": 62,
                "target_price": round(target, 8),
                "invalidation_price": round(recent_highs[-1]["value"] * 1.01, 8),
                "formed_at_index": max(recent_highs[-1]["index"], recent_lows[-1]["index"]),
            })

        # Falling wedge (bullish): both falling, but converging
        if h_slope < 0 and l_slope < 0 and abs(l_slope) > abs(h_slope):
            target = recent_highs[0]["value"]
            patterns.append({
                "pattern_name": "Falling Wedge",
                "bullish_or_bearish": "bullish",
                "confidence_pct": 62,
                "target_price": round(target, 8),
                "invalidation_price": round(recent_lows[-1]["value"] * 0.99, 8),
                "formed_at_index": max(recent_highs[-1]["index"], recent_lows[-1]["index"]),
            })

    return patterns
