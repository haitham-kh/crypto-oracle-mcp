from __future__ import annotations
"""
FAILURE MODE DETECTOR — Market Condition Viability Filter
=========================================================
Detects conditions that invalidate normal trading signals.
When failure modes are active, confidence is reduced and position
sizing should be cut proportionally.

Failure modes:
  LOW_VOLUME_DEAD_MARKET   — signals unreliable, spreads wide
  FAKE_BREAKOUT            — price breaks level without volume confirmation
  NEWS_SPIKE               — abnormal price move without OFI support
  WASH_TRADING             — volume anomalously high with no price impact
  REGIME_TRANSITION        — signals from different regimes are conflicting

Output:
  confidence_multiplier: 0.0 (avoid) to 1.0 (full confidence)
  failure_modes: list of detected issues
  recommended_action: str
"""

import numpy as np
from typing import Dict, List, Any, Optional


def detect_low_volume(candles: List[Dict]) -> Optional[Dict[str, Any]]:
    """
    Dead market detection: current volume < 20% of 20-period avg.
    Signals in dead markets are unreliable — spreads widen, moves are random.
    """
    if not candles or len(candles) < 5:
        return None

    volumes = [float(c.get("volume") or 0) for c in candles]
    current = volumes[-1] if volumes else 0
    avg_20 = float(np.mean(volumes[-20:])) if len(volumes) >= 20 else float(np.mean(volumes))

    if avg_20 == 0:
        return None

    ratio = current / avg_20
    if ratio < 0.20:
        return {
            "type": "LOW_VOLUME_DEAD_MARKET",
            "severity": "high",
            "confidence_penalty": 0.7,
            "detail": f"Volume {ratio:.0%} of 20-period avg — market is asleep. Signals unreliable.",
        }
    elif ratio < 0.35:
        return {
            "type": "LOW_VOLUME_DEAD_MARKET",
            "severity": "medium",
            "confidence_penalty": 0.4,
            "detail": f"Volume {ratio:.0%} of avg — below normal liquidity. Reduce size.",
        }
    return None


def detect_fake_breakout(candles: List[Dict], key_levels: List[float]) -> Optional[Dict[str, Any]]:
    """
    Fake breakout: price pierces a key level but volume does NOT expand.

    Heuristic rule:
    - Last candle high (or low) exceeds a key level
    - Volume on that candle < 1.0× avg (no conviction)
    - Candle closes back below (above) the level (rejection)
    """
    if not candles or len(candles) < 5 or not key_levels:
        return None

    closes = [float(c.get("close") or 0) for c in candles]
    highs = [float(c.get("high") or 0) for c in candles]
    lows = [float(c.get("low") or 0) for c in candles]
    opens = [float(c.get("open") or 0) for c in candles]
    volumes = [float(c.get("volume") or 0) for c in candles]

    avg_vol = float(np.mean(volumes[-20:])) if len(volumes) >= 20 else float(np.mean(volumes))
    if avg_vol == 0:
        return None

    last_high = highs[-1]
    last_low = lows[-1]
    last_close = closes[-1]
    last_open = opens[-1]
    last_vol = volumes[-1]
    vol_ratio = last_vol / avg_vol

    for level in key_levels:
        # Bullish fake: high pierced above level but close is back below
        if last_high > level and last_close < level and vol_ratio < 1.0:
            return {
                "type": "FAKE_BREAKOUT",
                "severity": "high",
                "confidence_penalty": 0.5,
                "detail": (
                    f"Price pierced {level:.6g} intracandle but closed back below "
                    f"({last_close:.6g}). Volume only {vol_ratio:.1f}× avg — no conviction."
                ),
                "level": level,
                "direction": "bullish_trap",
            }
        # Bearish fake: low pierced below level but close is back above
        if last_low < level and last_close > level and vol_ratio < 1.0:
            return {
                "type": "FAKE_BREAKOUT",
                "severity": "high",
                "confidence_penalty": 0.5,
                "detail": (
                    f"Price wicked below {level:.6g} but recovered to {last_close:.6g}. "
                    f"Volume {vol_ratio:.1f}× avg — stop-hunt suspected."
                ),
                "level": level,
                "direction": "bearish_trap",
            }

    return None


def detect_news_spike(candles: List[Dict], ofi_score: float = 0.0) -> Optional[Dict[str, Any]]:
    """
    News spike: large rapid price move without corresponding OFI.

    Signature:
    - Price change on last 1-3 candles > 3% 
    - OFI score is neutral (< ±20) — no sustained aggressive buying/selling
    - Volume is elevated but OFI is not directional

    News spikes mean: price moved for a news reason, not accumulated flow.
    Technical signals are unreliable immediately after.
    """
    if not candles or len(candles) < 3:
        return None

    closes = [float(c.get("close") or 0) for c in candles]
    volumes = [float(c.get("volume") or 0) for c in candles]

    # 3-candle price change
    start_price = closes[-4] if len(closes) >= 4 else closes[0]
    end_price = closes[-1]
    if start_price == 0:
        return None

    pct_change = abs(end_price - start_price) / start_price * 100
    avg_vol = float(np.mean(volumes[-20:])) if len(volumes) >= 20 else float(np.mean(volumes))
    vol_spike = (volumes[-1] / avg_vol) > 1.5 if avg_vol > 0 else False

    if pct_change > 3.0 and vol_spike and abs(ofi_score) < 20:
        return {
            "type": "NEWS_SPIKE",
            "severity": "medium",
            "confidence_penalty": 0.5,
            "detail": (
                f"Price moved {pct_change:.1f}% in 3 candles with vol spike, "
                f"but OFI is only {ofi_score:+.0f} — likely news-driven, not accumulated flow. "
                f"Wait for OFI to confirm before entering."
            ),
            "pct_move": round(pct_change, 2),
            "ofi_score": ofi_score,
        }
    return None


def detect_wash_trading(candles: List[Dict], trades: Optional[List[Dict]] = None) -> Optional[Dict[str, Any]]:
    """
    Wash trading: volume anomalously high but price barely moves.

    If volume > 5× avg AND price range < 20% of avg range:
    This is a red flag for fake volume / wash trading.
    OFI from trade tape should be near zero if wash trading.
    """
    if not candles or len(candles) < 10:
        return None

    volumes = [float(c.get("volume") or 0) for c in candles]
    highs = [float(c.get("high") or 0) for c in candles]
    lows = [float(c.get("low") or 0) for c in candles]

    avg_vol = float(np.mean(volumes[-20:])) if len(volumes) >= 20 else float(np.mean(volumes))
    avg_range = float(np.mean([h - l for h, l in zip(highs[-20:], lows[-20:])]))

    if avg_vol == 0 or avg_range == 0:
        return None

    last_vol_ratio = volumes[-1] / avg_vol
    last_range = highs[-1] - lows[-1]
    range_ratio = last_range / avg_range

    if last_vol_ratio > 5.0 and range_ratio < 0.2:
        ofi_suspicious = trades is not None  # placeholder
        return {
            "type": "WASH_TRADING_SUSPECTED",
            "severity": "medium",
            "confidence_penalty": 0.35,
            "detail": (
                f"Volume {last_vol_ratio:.1f}× avg but price range only {range_ratio:.1%} of avg. "
                f"Volume is not producing price discovery — possible wash trading or algo-driven."
            ),
            "vol_ratio": round(last_vol_ratio, 2),
            "range_ratio": round(range_ratio, 3),
        }
    return None


def detect_regime_conflict(regime: str, mtf_scores: Dict[str, float]) -> Optional[Dict[str, Any]]:
    """
    Regime transition / timeframe conflict.

    If 15m and 1d give opposite signals (> 40 score gap), regime is transitioning.
    Trading during transitions has negative expected value without clear confirmation.
    """
    if not mtf_scores or len(mtf_scores) < 2:
        return None

    scores = [v for v in mtf_scores.values() if v is not None]
    if len(scores) < 2:
        return None

    score_range = max(scores) - min(scores)
    has_conflict = max(scores) > 25 and min(scores) < -25

    if has_conflict:
        return {
            "type": "REGIME_CONFLICT",
            "severity": "medium",
            "confidence_penalty": 0.30,
            "detail": (
                f"Timeframe signals diverge sharply (range: {score_range:.0f} pts). "
                f"Lower TF signals oppose higher TF structure. "
                f"Wait for timeframe alignment before entering."
            ),
            "score_range": round(score_range, 1),
            "scores": {k: round(v, 1) for k, v in mtf_scores.items() if v is not None},
        }
    return None


# ─────────────────────────────────────────────────────────────────────────────
# AGGREGATE FAILURE CHECK
# ─────────────────────────────────────────────────────────────────────────────

def detect_all_failure_modes(
    candles: List[Dict],
    key_levels: List[float],
    ofi_score: float,
    regime: str,
    mtf_scores: Dict[str, float],
    trades: Optional[List[Dict]] = None,
) -> Dict[str, Any]:
    """
    Run all failure mode detectors and aggregate results.

    Returns:
      confidence_multiplier: 0.0 to 1.0 (multiply with model confidence)
      failure_modes: list of detected issues
      recommended_action: 'trade_normal' | 'reduce_size' | 'wait' | 'avoid'
    """
    failures: List[Dict] = []

    # Run each detector
    checks = [
        detect_low_volume(candles),
        detect_fake_breakout(candles, key_levels),
        detect_news_spike(candles, ofi_score),
        detect_wash_trading(candles, trades),
        detect_regime_conflict(regime, mtf_scores),
    ]

    for check in checks:
        if check is not None:
            failures.append(check)

    # Aggregate confidence penalty (multiplicative)
    multiplier = 1.0
    for f in failures:
        penalty = f.get("confidence_penalty", 0)
        multiplier *= (1.0 - penalty)

    multiplier = max(0.0, min(1.0, multiplier))

    # Recommended action
    if multiplier < 0.25:
        action = "avoid"
    elif multiplier < 0.50:
        action = "wait"
    elif multiplier < 0.75:
        action = "reduce_size"
    else:
        action = "trade_normal"

    high_severity = [f for f in failures if f.get("severity") == "high"]
    medium_severity = [f for f in failures if f.get("severity") == "medium"]

    return {
        "confidence_multiplier": round(multiplier, 3),
        "failure_modes": failures,
        "high_severity_count": len(high_severity),
        "medium_severity_count": len(medium_severity),
        "recommended_action": action,
        "clean_conditions": len(failures) == 0,
        "summary": (
            "Market conditions are clean — proceed with normal sizing."
            if not failures
            else f"{len(failures)} failure mode(s) detected: "
            + ", ".join(f.get("type", "UNKNOWN") for f in failures)
            + f". Confidence reduced to {multiplier:.0%}."
        ),
    }
