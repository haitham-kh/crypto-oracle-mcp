from __future__ import annotations
"""
ORACLE ENGINE — Fused Wolf / Insider / Quant analysis pipeline.

Consumes the structured report produced by ``tool_full_coin_intelligence_report``
and deterministically executes all nine phases of the ORACLE spec:

    Phase 1 — Regime detection
    Phase 2 — Multi-Timeframe signal engine (MTF)
    Phase 3 — Smart-Money divergence (DPI + WFI)
    Phase 4 — Pattern confidence engine
    Phase 5 — Macro adjustment
    Phase 6 — Composite score + signal
    Phase 7 — Bayesian scenarios + 24H expected value
    Phase 8 — Trade execution plan (entry / SL / TP / Kelly / size)
    Phase 9 — Risk audit

Returns both a structured dict and a fully formatted Markdown brief that
matches the FINAL OUTPUT FORMAT section of ``prompts/analysis_oracle.md``.
"""
from typing import Any, Dict, List, Optional, Tuple

# ----------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------

def _g(d: Optional[Dict], *keys, default=None):
    """Safe nested dict getter."""
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k)
        if cur is None:
            return default
    return cur if cur is not None else default


def _num(x, default=None):
    try:
        if x is None:
            return default
        return float(x)
    except (TypeError, ValueError):
        return default


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _fmt_price(x: Optional[float]) -> str:
    if x is None:
        return "n/a"
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "n/a"
    if v >= 100:
        return f"{v:,.2f}"
    if v >= 1:
        return f"{v:.4f}"
    if v >= 0.01:
        return f"{v:.6f}"
    return f"{v:.8f}"


def _pct(x: Optional[float], digits: int = 2) -> str:
    if x is None:
        return "n/a"
    try:
        return f"{float(x):+.{digits}f}%"
    except (TypeError, ValueError):
        return "n/a"


# ----------------------------------------------------------------------
# Phase 1 — Regime detection
# ----------------------------------------------------------------------

def detect_regime(ta_4h: Dict, ta_1d: Dict) -> Dict[str, Any]:
    """Classify as TRENDING / RANGING / VOLATILE / CHAOTIC per the spec."""
    adx_4h = _num(_g(ta_4h, "adx"))
    adx_1d = _num(_g(ta_1d, "adx"))
    atr_pct_4h = _num(_g(ta_4h, "atr_pct"))
    bb_squeeze_4h = bool(_g(ta_4h, "bb_squeeze", default=False))
    bb_bw = _num(_g(ta_4h, "bb_bandwidth"))

    # Representative ADX: prefer 4H, fall back to 1D
    adx = adx_4h if adx_4h is not None else adx_1d

    # CHAOTIC proxy: ATR% > 8 (crypto scale) -> chaotic
    chaotic = atr_pct_4h is not None and atr_pct_4h >= 8.0

    if chaotic:
        regime = "CHAOTIC"
        reason = f"ATR on 4H is {atr_pct_4h:.2f}% of price — volatility well above normal, standard signals unreliable."
        multiplier = 0.50
    elif bb_squeeze_4h:
        regime = "VOLATILE"
        reason = "Bollinger squeeze active on 4H — breakout pending, direction unconfirmed until volume expansion."
        multiplier = 0.70
    elif adx is not None and adx > 25:
        regime = "TRENDING"
        reason = f"ADX {adx:.1f} on 4H/1D confirms directional trend — momentum tools are primary."
        multiplier = 1.00
    elif adx is not None and adx < 20:
        regime = "RANGING"
        reason = f"ADX {adx:.1f} indicates consolidation — oscillators dominate, trend tools downweighted."
        multiplier = 0.85
    else:
        regime = "RANGING"
        reason = "ADX in transition zone — treating as range until directional break confirmed."
        multiplier = 0.85

    return {
        "regime": regime,
        "reason": reason,
        "multiplier": multiplier,
        "adx_4h": adx_4h,
        "adx_1d": adx_1d,
        "atr_pct_4h": atr_pct_4h,
        "bb_squeeze_4h": bb_squeeze_4h,
        "bb_bandwidth_4h": bb_bw,
    }


# ----------------------------------------------------------------------
# Phase 2 — Multi-Timeframe signal engine
# ----------------------------------------------------------------------

def score_timeframe(ta: Dict, regime: str) -> Dict[str, Any]:
    """Compute raw signal score (-100..+100) for one timeframe."""
    if not isinstance(ta, dict) or ta.get("error"):
        return {"score": 0, "contributions": [], "available": False}

    contribs: List[Tuple[str, float]] = []

    def add(label: str, pts: float):
        if pts:
            contribs.append((label, round(pts, 2)))

    # Weights for regime-sensitive grouping
    trend_w = 0.5 if regime == "RANGING" else 1.0
    osc_w = 1.3 if regime == "RANGING" else 1.0

    # -------- Trend --------
    if _g(ta, "price_vs_ema200") == "above":
        add("Price > EMA200", +20 * trend_w)
    elif _g(ta, "price_vs_ema200") == "below":
        add("Price < EMA200", -20 * trend_w)

    if _g(ta, "golden_cross_active"):
        add("Golden cross", +15 * trend_w)
    if _g(ta, "death_cross_active"):
        add("Death cross", -15 * trend_w)

    # Ichimoku
    pvc = _g(ta, "ichimoku", "price_vs_cloud")
    cc = _g(ta, "ichimoku", "cloud_color")
    if pvc == "above" and cc == "bullish":
        add("Ichimoku price>cloud (green)", +10 * trend_w)
    elif pvc == "below" and cc == "bearish":
        add("Ichimoku price<cloud (red)", -10 * trend_w)
    tk = _g(ta, "ichimoku", "tk_cross")
    if tk == "bullish":
        add("TK cross bullish", +8 * trend_w)
    elif tk == "bearish":
        add("TK cross bearish", -8 * trend_w)

    # ADX + DI
    adx = _num(_g(ta, "adx"))
    dp = _num(_g(ta, "di_plus"))
    dm = _num(_g(ta, "di_minus"))
    if adx is not None and dp is not None and dm is not None and adx > 25:
        if dp > dm:
            add("ADX>25 DI+>DI-", +6 * trend_w)
        elif dm > dp:
            add("ADX>25 DI->DI+", -6 * trend_w)

    # -------- Momentum (oscillators) --------
    rsi = _num(_g(ta, "rsi"))
    if rsi is not None:
        # RSI 45-55 is truly neutral — don't award points for doing nothing
        if 45 <= rsi <= 55:
            pass  # Neutral — no signal
        elif 55 < rsi <= 65:
            add("RSI upper momentum", +8 * osc_w)
        elif 35 <= rsi < 45:
            add("RSI lower momentum", -8 * osc_w)
        elif rsi < 30:
            add("RSI oversold reversal", +12 * osc_w)
        elif rsi > 70:
            add("RSI overbought", -12 * osc_w)

    mc = _g(ta, "macd_crossover")
    hist = _num(_g(ta, "macd_histogram"))
    if mc == "bullish_cross":
        add("MACD bull cross", +15)
    elif mc == "bearish_cross":
        add("MACD bear cross", -15)
    ml = _num(_g(ta, "macd_line"))
    ms = _num(_g(ta, "macd_signal"))
    if ml is not None and ms is not None:
        if ml > 0 and ms > 0:
            add("MACD above zero", +8)
        elif ml < 0 and ms < 0:
            add("MACD below zero", -8)

    k = _num(_g(ta, "stochrsi_k"))
    d = _num(_g(ta, "stochrsi_d"))
    if k is not None and d is not None:
        if k > d and k < 0.2:
            add("StochRSI oversold cross", +10 * osc_w)
        elif k < d and k > 0.8:
            add("StochRSI overbought cross", -10 * osc_w)

    cmf = _num(_g(ta, "cmf"))
    if cmf is not None:
        if cmf > 0.1:
            add("CMF strong inflow", +8)
        elif cmf < -0.1:
            add("CMF strong outflow", -8)

    mfi = _num(_g(ta, "mfi"))
    if mfi is not None:
        if 40 <= mfi <= 60:
            add("MFI healthy", +6)
        elif mfi > 80 or mfi < 20:
            add("MFI extreme", -6)

    cci = _num(_g(ta, "cci"))
    if cci is not None:
        if -110 < cci < -80:
            add("CCI recovery", +5)
        elif 80 < cci < 110:
            add("CCI exhaustion", -5)

    # -------- Volatility / Structure --------
    pb = _num(_g(ta, "bb_percent_b"))
    vol_vs_avg = _num(_g(ta, "volume_vs_avg"))
    if pb is not None and rsi is not None and vol_vs_avg is not None:
        if pb < 0.1 and rsi < 40 and vol_vs_avg > 1:
            add("Lower BB + RSI<40 + vol rising", +10 * osc_w)
        elif pb > 0.9 and rsi > 60 and vol_vs_avg > 1:
            add("Upper BB + RSI>60 + vol rising", -10 * osc_w)

    if _g(ta, "bb_squeeze") and _g(ta, "volume_spike"):
        if pb is not None and pb > 0.8:
            add("BB squeeze up-break + vol spike", +8)
        elif pb is not None and pb < 0.2:
            add("BB squeeze down-break + vol spike", -8)

    if _g(ta, "price_vs_vwap") == "above":
        add("Price > VWAP", +6)
    elif _g(ta, "price_vs_vwap") == "below":
        add("Price < VWAP", -6)

    # -------- Volume (regime-aware) --------
    if _g(ta, "volume_spike"):
        td = _g(ta, "trend_direction")
        if regime == "RANGING":
            # In range: volume spike at resistance = rejection (bearish)
            #           volume spike at support = bounce (bullish)
            if pb is not None:
                if pb > 0.8:  # Near upper BB / resistance
                    add("Vol spike at resistance (rejection)", -10 * osc_w)
                elif pb < 0.2:  # Near lower BB / support
                    add("Vol spike at support (bounce)", +10 * osc_w)
                else:
                    pass  # Midrange volume spike is ambiguous in range
        else:
            # In trend: volume confirms direction
            if td == "bullish":
                add("Volume spike up-candle", +10)
            elif td == "bearish":
                add("Volume spike down-candle", -10)

    obv_t = _g(ta, "obv_trend")
    if obv_t == "up":
        add("OBV higher highs", +8)
    elif obv_t == "down":
        add("OBV lower lows", -8)

    # -------- CVD (Cumulative Volume Delta) --------
    cvd_div = _g(ta, "cvd_divergence")
    if cvd_div == "bullish_accumulation":
        add("CVD bullish divergence (stealth accumulation)", +15)
    elif cvd_div == "bearish_distribution":
        add("CVD bearish divergence (stealth distribution)", -15)

    cvd_trend = _g(ta, "cvd_trend")
    if cvd_trend == "rising":
        add("CVD trend rising (buy pressure)", +6)
    elif cvd_trend == "falling":
        add("CVD trend falling (sell pressure)", -6)

    # -------- Divergences (powerful) --------
    tf_name = _g(ta, "timeframe") or ""
    div = _g(ta, "rsi_divergence")
    if div == "bullish":
        add("RSI bull divergence", +20 if tf_name in ("4h", "1d") else 12)
    elif div == "bearish":
        add("RSI bear divergence", -20 if tf_name in ("4h", "1d") else -12)

    total = sum(pts for _, pts in contribs)
    total = _clamp(total, -100, 100)
    return {
        "score": round(total, 2),
        "contributions": contribs,
        "available": True,
        "rsi": rsi,
        "macd_crossover": mc,
        "bb_percent_b": pb,
        "volume_vs_avg": vol_vs_avg,
        "trend_direction": _g(ta, "trend_direction"),
        "rsi_divergence": div,
        "cvd_divergence": cvd_div,
    }


def compute_mtf(technical_analysis: Dict, regime: str) -> Dict[str, Any]:
    """Score all 4 timeframes and weight them."""
    weights = {"1d": 0.40, "4h": 0.30, "1h": 0.20, "15m": 0.10}
    per_tf: Dict[str, Dict] = {}
    weighted = 0.0
    total_w = 0.0
    for tf, w in weights.items():
        ta = technical_analysis.get(tf) if isinstance(technical_analysis, dict) else None
        s = score_timeframe(ta or {}, regime)
        per_tf[tf] = s
        if s["available"]:
            weighted += s["score"] * w
            total_w += w
    mtf = (weighted / total_w) if total_w > 0 else 0.0
    # Regime filter on composite
    if regime == "VOLATILE":
        mtf *= 0.6
    elif regime == "CHAOTIC":
        mtf *= 0.3
    return {"per_tf": per_tf, "mtf_score": round(_clamp(mtf, -100, 100), 2)}


# ----------------------------------------------------------------------
# Phase 3 — Smart Money Divergence Engine (DPI + WFI)
# ----------------------------------------------------------------------

def compute_smart_money(report: Dict, current_price: Optional[float]) -> Dict[str, Any]:
    derivs = report.get("derivatives") or {}
    fr = derivs.get("funding_rates") or {}
    oi = derivs.get("open_interest") or {}
    whale = report.get("whale_and_onchain") or {}
    ob = report.get("order_book") or {}

    contribs: List[Tuple[str, float]] = []

    def add(label: str, pts: float):
        if pts:
            contribs.append((label, round(pts, 2)))

    # --- DPI ---
    # Funding (already as percent in fetcher)
    f = _num(fr.get("current_funding_rate"))
    if f is not None:
        if f > 0.1:
            add("Funding >+0.1% (extreme long crowd)", -20)
        elif f > 0.05:
            add("Funding >+0.05% (long crowded)", -15)
        elif f < -0.02:
            add("Funding <-0.02% (short crowded)", +20)
        elif abs(f) <= 0.01:
            add("Funding ~0 (accumulation)", +15)

    # OI vs price change — use per-TF signals as proxy for direction
    oi_val = _num(oi.get("open_interest_usdt"))
    # We do not have delta-OI/delta-price here directly; skip unless available.

    # Long/short ratio
    ls = _num(oi.get("long_short_ratio"))
    if ls is not None:
        if ls < 0.8:
            add("L/S<0.8 squeeze fuel", +10)
        elif ls > 1.5:
            add("L/S>1.5 flush fuel", -10)

    # Liquidation walls — approximate from order book walls relative to price
    bid_wall = _num(ob.get("bid_wall_at"))
    ask_wall = _num(ob.get("ask_wall_at"))
    if current_price and ask_wall and ask_wall > current_price:
        dist = (ask_wall - current_price) / current_price
        if dist < 0.05:
            add("Ask wall / short liq cluster just above", +15)
    if current_price and bid_wall and bid_wall < current_price:
        dist = (current_price - bid_wall) / current_price
        if dist < 0.05:
            add("Bid wall / long liq cluster just below", -15)

    # --- WFI ---
    net_flow = _num(whale.get("exchange_net_flow_24h"))
    if net_flow is not None:
        if net_flow < 0:
            add("Exchange net outflow (accumulation)", +20)
        elif net_flow > 0:
            add("Exchange net inflow (distribution)", -20)

    buys = _num(whale.get("whale_buy_count"))
    sells = _num(whale.get("whale_sell_count"))
    if buys is not None and sells is not None and (buys + sells) >= 3:
        share_buy = buys / (buys + sells)
        if share_buy > 0.7:
            add("Whale buys >70%", +15)
        elif share_buy < 0.3:
            add("Whale sells >70%", -15)

    total = sum(p for _, p in contribs)
    total = _clamp(total, -100, 100)

    return {
        "score": round(total, 2),
        "contributions": contribs,
        "funding_rate_pct": f,
        "funding_sentiment": fr.get("funding_sentiment"),
        "open_interest_usdt": oi_val,
        "long_short_ratio": ls,
        "net_flow_24h": net_flow,
        "net_flow_interpretation": whale.get("net_flow_interpretation"),
        "whale_buy_count": buys,
        "whale_sell_count": sells,
    }


# ----------------------------------------------------------------------
# Phase 4 — Pattern confidence
# ----------------------------------------------------------------------

PATTERN_TABLE = {
    "inverse_head_and_shoulders": (+25, 75, "bullish"),
    "inverse_h&s": (+25, 75, "bullish"),
    "head_and_shoulders": (-25, 75, "bearish"),
    "double_bottom": (+20, 72, "bullish"),
    "double_top": (-20, 72, "bearish"),
    "triple_bottom": (+20, 70, "bullish"),
    "triple_top": (-20, 70, "bearish"),
    "bull_flag": (+18, 70, "bullish"),
    "bear_flag": (-18, 70, "bearish"),
    "ascending_triangle": (+15, 65, "bullish"),
    "descending_triangle": (-15, 65, "bearish"),
    "cup_and_handle": (+15, 65, "bullish"),
    "symmetrical_triangle": (0, 58, "neutral"),
    "falling_wedge": (+12, 62, "bullish"),
    "rising_wedge": (-12, 62, "bearish"),
    "bullish_engulfing": (+10, 62, "bullish"),
    "bearish_engulfing": (-10, 62, "bearish"),
    "morning_star": (+12, 65, "bullish"),
    "evening_star": (-12, 65, "bearish"),
    "hammer": (+10, 60, "bullish"),
    "shooting_star": (-10, 60, "bearish"),
}


def _pattern_key(name: str) -> str:
    return (name or "").strip().lower().replace(" ", "_").replace("-", "_")


def compute_pattern_score(chart_patterns: Dict, mtf_score: float) -> Dict[str, Any]:
    strongest = _g(chart_patterns, "strongest_pattern") or {}
    name = strongest.get("pattern") or strongest.get("name") or strongest.get("type") or ""
    key = _pattern_key(name)
    base = PATTERN_TABLE.get(key)
    direction = "neutral"
    confidence = None
    raw = 0.0
    conflict = False

    if base is not None:
        points, confidence, direction = base
        # Scale by the pattern's detected confidence if provided
        detected_conf = _num(strongest.get("confidence_pct")) or confidence
        scale = (detected_conf or confidence) / 100.0
        raw = points * scale
        # Conflict check vs MTF direction
        mtf_dir = "bullish" if mtf_score > 10 else ("bearish" if mtf_score < -10 else "neutral")
        if direction != "neutral" and mtf_dir != "neutral" and direction != mtf_dir:
            conflict = True
            raw = raw * 0.5

    return {
        "score": round(_clamp(raw, -100, 100), 2),
        "pattern_name": name or "none",
        "pattern_direction": direction,
        "base_confidence_pct": confidence,
        "conflicts_with_mtf": conflict,
    }


# ----------------------------------------------------------------------
# Phase 5 — Macro adjustment
# ----------------------------------------------------------------------

def compute_macro(report: Dict, is_btc: bool, beta: Optional[float]) -> Dict[str, Any]:
    fg = _g(report, "macro_context", "fear_greed") or {}
    gm = _g(report, "macro_context", "global_market") or {}
    fg_val = _num(fg.get("current_value"))
    fg_trend = fg.get("trend")
    fg_adj = 0
    if fg_val is not None:
        if fg_val < 15:
            fg_adj = +20
        elif fg_val < 30:
            fg_adj = +10
        elif fg_val < 45:
            fg_adj = +5
        elif fg_val <= 55:
            fg_adj = 0
        elif fg_val <= 70:
            fg_adj = -5
        elif fg_val <= 85:
            fg_adj = -15
        else:
            fg_adj = -25

    btc_dom = _num(gm.get("btc_dominance_pct"))
    btc_dom_adj = 0  # without a reliable 7d dominance delta we default to 0
    # If the report carries a 7d change we honor it
    btc_dom_7d_change = _num(gm.get("btc_dominance_change_7d"))
    if btc_dom_7d_change is not None and not is_btc:
        if btc_dom_7d_change > 1:
            btc_dom_adj = -10
        elif btc_dom_7d_change < -1:
            btc_dom_adj = +10

    macro_total = fg_adj + btc_dom_adj

    size_haircut = 0.0
    if beta is not None and beta > 2.0:
        size_haircut = 0.25  # reduce recommended size by 25%

    return {
        "score": round(_clamp(macro_total, -100, 100), 2),
        "fear_greed_value": fg_val,
        "fear_greed_trend": fg_trend,
        "fear_greed_adjustment": fg_adj,
        "btc_dominance_pct": btc_dom,
        "btc_dominance_adjustment": btc_dom_adj,
        "beta_vs_btc": beta,
        "size_haircut_from_beta": size_haircut,
    }


# ----------------------------------------------------------------------
# Sentiment sub-score (small)
# ----------------------------------------------------------------------

def compute_sentiment_score(report: Dict) -> Dict[str, Any]:
    s = report.get("sentiment") or {}
    trend_rank = _num(s.get("trending_rank_coingecko"))
    votes_up = _num(s.get("sentiment_votes_up_pct"))
    score = 0.0
    if trend_rank is not None:
        if trend_rank <= 5:
            score += 15
        elif trend_rank <= 15:
            score += 8
    if votes_up is not None:
        if votes_up > 75:
            score += 10
        elif votes_up > 60:
            score += 5
        elif votes_up < 40:
            score -= 5
    # ── News headline scoring (was previously fetched but never consumed) ──
    news = report.get("news") or {}
    headlines = news.get("news", []) if isinstance(news, dict) else []
    if headlines:
        # Simple keyword-based sentiment from headlines
        bullish_keywords = {"surge", "rally", "soar", "bull", "breakout", "pump", "adoption",
                           "partnership", "upgrade", "launch", "moon", "ath", "record"}
        bearish_keywords = {"crash", "dump", "bear", "hack", "exploit", "ban", "lawsuit",
                           "sec", "fraud", "collapse", "plunge", "sell-off", "selloff"}
        bull_hits = 0
        bear_hits = 0
        for h in headlines:
            title = (h.get("title") or "").lower()
            for kw in bullish_keywords:
                if kw in title:
                    bull_hits += 1
            for kw in bearish_keywords:
                if kw in title:
                    bear_hits += 1
        if bull_hits > bear_hits + 1:
            score += 10
        elif bear_hits > bull_hits + 1:
            score -= 10

    return {"score": round(_clamp(score, -100, 100), 2),
            "trending_rank": trend_rank,
            "sentiment_votes_up_pct": votes_up,
            "news_headlines_count": len(headlines)}


# ----------------------------------------------------------------------
# Phase 6 — Composite + signal label
# ----------------------------------------------------------------------

def composite(mtf: float, smart: float, pattern: float, sentiment: float,
              macro: float, regime_multiplier: float) -> Dict[str, Any]:
    raw = (mtf * 0.40) + (smart * 0.25) + (pattern * 0.15) + (sentiment * 0.10) + (macro * 0.10)
    final = _clamp(raw * regime_multiplier, -100, 100)
    if final >= 70:
        sig = "STRONG_BUY"
    elif final >= 45:
        sig = "BUY"
    elif final >= 20:
        sig = "WEAK_BUY"
    elif final > -20:
        sig = "NEUTRAL"
    elif final > -45:
        sig = "WEAK_SELL"
    elif final > -70:
        sig = "SELL"
    else:
        sig = "STRONG_SELL"
    return {
        "raw_pre_regime": round(raw, 2),
        "final_score": round(final, 2),
        "signal": sig,
        "contributions": {
            "mtf": round(mtf * 0.40, 2),
            "smart_money": round(smart * 0.25, 2),
            "pattern": round(pattern * 0.15, 2),
            "sentiment": round(sentiment * 0.10, 2),
            "macro": round(macro * 0.10, 2),
        },
    }


# ----------------------------------------------------------------------
# Phase 7 — Heuristic scenarios + EV24H
# NOTE: Probabilities are score-derived heuristics, NOT Bayesian posteriors.
# They require empirical calibration via backtesting to be trustworthy.
# ----------------------------------------------------------------------

def build_scenarios(current_price: Optional[float], atr_4h: Optional[float],
                    sr: Dict, final_score: float) -> Dict[str, Any]:
    if not current_price:
        return {"error": "no current price"}
    atr = atr_4h or (current_price * 0.02)
    res = (sr or {}).get("strong_resistances") or []
    sup = (sr or {}).get("strong_supports") or []
    r1 = _num((res[0] or {}).get("level")) if res else None
    r2 = _num((res[1] or {}).get("level")) if len(res) > 1 else None
    s1 = _num((sup[0] or {}).get("level")) if sup else None
    s2 = _num((sup[1] or {}).get("level")) if len(sup) > 1 else None

    bull_24h = (r1 if r1 and r1 > current_price else current_price + 1.5 * atr)
    bull_7d = (r2 if r2 and r2 > bull_24h else bull_24h + 2.5 * atr)
    bear_24h = (s1 if s1 and s1 < current_price else current_price - 1.5 * atr)
    bear_7d = (s2 if s2 and s2 < bear_24h else bear_24h - 2.5 * atr)

    # Probabilities skewed by final_score (-100..+100 -> probability tilt)
    tilt = final_score / 100.0  # -1..+1
    base_prob = 0.55 - 0.10 * abs(tilt)  # stronger edge reduces base weight
    remaining = 1.0 - base_prob
    # Split remaining between bull/bear weighted by sign of tilt
    bull_share = 0.5 + 0.5 * tilt  # 0..1
    bull_prob = remaining * bull_share
    bear_prob = remaining * (1 - bull_share)

    def pct(a: float) -> float:
        return (a - current_price) / current_price * 100.0

    base_mid = current_price  # base case midpoint = roughly current price
    base_low = current_price - 0.75 * atr
    base_high = current_price + 0.75 * atr

    bull_pct = pct(bull_24h)
    bear_pct = pct(bear_24h)
    base_pct_mid = pct(base_mid)

    ev_24h = (bull_prob * bull_pct) + (base_prob * base_pct_mid) + (bear_prob * bear_pct)

    return {
        "bull": {
            "prob_pct": round(bull_prob * 100, 1),
            "target_24h": bull_24h,
            "target_24h_pct": round(bull_pct, 2),
            "target_7d": bull_7d,
            "target_7d_pct": round(pct(bull_7d), 2),
            "trigger": f"Close > {_fmt_price(r1 or bull_24h)} on 4H with volume >150% of 20-period average",
        },
        "base": {
            "prob_pct": round(base_prob * 100, 1),
            "range_24h_low": base_low,
            "range_24h_high": base_high,
            "range_7d_low": current_price - 2 * atr,
            "range_7d_high": current_price + 2 * atr,
            "trigger": "Structure continuation without catalyst — oscillation between nearest S/R",
        },
        "bear": {
            "prob_pct": round(bear_prob * 100, 1),
            "target_24h": bear_24h,
            "target_24h_pct": round(bear_pct, 2),
            "target_7d": bear_7d,
            "target_7d_pct": round(pct(bear_7d), 2),
            "trigger": f"Close < {_fmt_price(s1 or bear_24h)} on 4H on expanding volume",
        },
        "ev_24h_pct": round(ev_24h, 2),
    }


# ----------------------------------------------------------------------
# Phase 8 — Trade execution plan
# ----------------------------------------------------------------------

def build_trade_plan(signal: str, current_price: Optional[float], atr_4h: Optional[float],
                     vwap_4h: Optional[float], sr: Dict, final_score: float,
                     beta: Optional[float], size_haircut: float) -> Dict[str, Any]:
    if not current_price:
        return {"error": "no price"}

    direction = "FLAT"
    if signal in ("STRONG_BUY", "BUY", "WEAK_BUY"):
        direction = "LONG"
    elif signal in ("STRONG_SELL", "SELL", "WEAK_SELL"):
        direction = "SHORT"

    setup_quality = {
        "STRONG_BUY": "A+", "BUY": "A", "WEAK_BUY": "B",
        "NEUTRAL": "No Trade",
        "WEAK_SELL": "B", "SELL": "A", "STRONG_SELL": "A+",
    }.get(signal, "No Trade")

    atr = atr_4h or (current_price * 0.02)
    res = (sr or {}).get("strong_resistances") or []
    sup = (sr or {}).get("strong_supports") or []
    nearest_support = _num((sup[0] or {}).get("level")) if sup else (current_price - atr)
    nearest_resistance = _num((res[0] or {}).get("level")) if res else (current_price + atr)
    # Sanity-check VWAP: if it's more than 15% away from price (bad data / misaligned
    # timeframe), fall back to current price.
    ideal_entry = vwap_4h if (vwap_4h and abs(vwap_4h - current_price) / current_price < 0.15) else current_price

    # When no tradable edge, emit blank tiers — do not print fake numbers.
    if direction == "FLAT":
        blank = {"entry_lower": None, "entry_upper": None, "entry_ideal": None,
                 "stop_loss": None, "sl_pct": None,
                 "tp1": None, "tp1_pct": None, "tp2": None, "tp2_pct": None,
                 "tp3": None, "tp3_pct": None,
                 "rr_tp1": 1.5, "rr_tp2": 2.5, "rr_tp3": 4.0,
                 "risk_per_unit": 0}
        return {
            "direction": "FLAT",
            "setup_quality": setup_quality,
            "conservative": dict(blank),
            "moderate": dict(blank),
            "aggressive": dict(blank),
            "kelly": {"win_prob": "n/a", "avg_rr": "n/a",
                      "full_kelly_pct": "n/a", "half_kelly_pct": "n/a",
                      "half_kelly_adj_pct": "n/a"},
            "position_size": {"capital_example_usdt": 10000.0, "risk_pct": 1.0,
                              "risk_usdt": 100.0, "position_usdt": 0,
                              "pct_of_capital": 0, "beta_haircut_applied": False},
            "invalidation": {"level": None, "timeframe": "4H", "direction": "n/a"},
        }

    # Estimate slippage + fees for realistic R:R
    vol_24h = _num(_g(sr, "total_volume_usd")) or 0
    fee_pct = 0.10  # taker fee (Binance/MEXC)
    slippage_pct = 0.05 if vol_24h > 50_000_000 else (0.10 if vol_24h > 10_000_000 else 0.30)
    round_trip_cost_pct = (fee_pct + slippage_pct) * 2  # entry + exit

    def build_tier(sl_mult: float, entry_offset_mult: float) -> Dict[str, Any]:
        """Build trade tier with differentiated entry zones per risk tolerance.
        Conservative: deeper pullback entry (larger offset from market).
        Aggressive: near-market entry.
        """
        if direction == "LONG":
            tier_entry = ideal_entry - entry_offset_mult * atr
            sl = min(tier_entry - sl_mult * atr, (nearest_support or tier_entry) * 0.995)
            risk = tier_entry - sl
            tp1 = tier_entry + 1.5 * risk
            tp2 = tier_entry + 2.5 * risk
            tp3 = tier_entry + 4.0 * risk
        elif direction == "SHORT":
            tier_entry = ideal_entry + entry_offset_mult * atr
            sl = max(tier_entry + sl_mult * atr, (nearest_resistance or tier_entry) * 1.005)
            risk = sl - tier_entry
            tp1 = tier_entry - 1.5 * risk
            tp2 = tier_entry - 2.5 * risk
            tp3 = tier_entry - 4.0 * risk
        else:
            tier_entry = ideal_entry
            sl = ideal_entry
            risk = atr
            tp1 = tp2 = tp3 = ideal_entry
        risk = max(risk, 1e-12)
        # Effective R:R after fees and slippage
        eff_risk = risk + (tier_entry * round_trip_cost_pct / 100)
        eff_rr_tp1 = round(abs(tp1 - tier_entry) / eff_risk, 2) if eff_risk > 0 else 0
        eff_rr_tp2 = round(abs(tp2 - tier_entry) / eff_risk, 2) if eff_risk > 0 else 0
        return {
            "entry_ideal": round(tier_entry, 8),
            "entry_lower": round(tier_entry - 0.3 * atr, 8),
            "entry_upper": round(tier_entry + 0.3 * atr, 8),
            "stop_loss": round(sl, 8),
            "sl_pct": round(abs(sl - tier_entry) / tier_entry * 100, 2),
            "tp1": round(tp1, 8),
            "tp1_pct": round((tp1 - tier_entry) / tier_entry * 100, 2),
            "tp2": round(tp2, 8),
            "tp2_pct": round((tp2 - tier_entry) / tier_entry * 100, 2),
            "tp3": round(tp3, 8),
            "tp3_pct": round((tp3 - tier_entry) / tier_entry * 100, 2),
            "rr_tp1": 1.5,
            "rr_tp2": 2.5,
            "rr_tp3": 4.0,
            "effective_rr_tp1_after_fees": eff_rr_tp1,
            "effective_rr_tp2_after_fees": eff_rr_tp2,
            "risk_per_unit": round(risk, 8),
            "estimated_slippage_pct": slippage_pct,
            "round_trip_cost_pct": round(round_trip_cost_pct, 3),
        }

    # Conservative: wait for deeper pullback (0.5 ATR offset), tight SL
    # Moderate: near VWAP entry, standard SL
    # Aggressive: market entry (0 offset), wide SL
    conservative = build_tier(1.5, 0.5)
    moderate = build_tier(2.0, 0.15)
    aggressive = build_tier(2.5, 0.0)

    # Kelly: score-to-probability mapping (UNCALIBRATED HEURISTIC)
    # WARNING: This p is derived from composite score, not empirical win rate.
    # Until backtested, treat as directional guidance only.
    # Hard-cap at 1-2% risk regardless of Kelly output.
    p = _clamp(0.5 + (final_score / 200.0), 0.10, 0.85)  # narrower bounds than before
    # Compute effective R:R from moderate tier (after fees)
    b = moderate.get("effective_rr_tp2_after_fees") or 2.0
    f_full = (p * b - (1 - p)) / b
    f_full = max(0.0, f_full)
    f_half = f_full * 0.5
    # Apply beta haircut
    f_half_adj = f_half * (1 - size_haircut)
    # Hard cap: never risk more than 3% per trade regardless of Kelly
    f_half_adj = min(f_half_adj, 0.03)

    # Position size (example $10k capital, 1% risk)
    capital = 10000.0
    risk_pct = 0.01 if setup_quality in ('B', 'No Trade') else 0.02
    risk_usdt = capital * risk_pct
    risk_per_unit = moderate["risk_per_unit"] if moderate["risk_per_unit"] > 0 else atr
    position_units = risk_usdt / risk_per_unit if risk_per_unit > 0 else 0
    position_usdt = position_units * ideal_entry
    # Cap to half-kelly % of capital
    kelly_cap_usdt = capital * f_half_adj
    if kelly_cap_usdt > 0:
        position_usdt = min(position_usdt, kelly_cap_usdt)

    invalidation_level = conservative["stop_loss"]

    return {
        "direction": direction,
        "setup_quality": setup_quality,
        "conservative": conservative,
        "moderate": moderate,
        "aggressive": aggressive,
        "kelly": {
            "win_prob": round(p * 100, 1),
            "avg_rr": b,
            "full_kelly_pct": round(f_full * 100, 2),
            "half_kelly_pct": round(f_half * 100, 2),
            "half_kelly_adj_pct": round(f_half_adj * 100, 2),
        },
        "position_size": {
            "capital_example_usdt": capital,
            "risk_pct": 1.0,
            "risk_usdt": risk_usdt,
            "position_usdt": round(position_usdt, 2),
            "pct_of_capital": round(position_usdt / capital * 100, 2),
            "beta_haircut_applied": size_haircut > 0,
        },
        "invalidation": {
            "level": invalidation_level,
            "timeframe": "4H",
            "direction": "below" if direction == "LONG" else ("above" if direction == "SHORT" else "n/a"),
        },
    }


# ----------------------------------------------------------------------
# Phase 9 — Risk audit
# ----------------------------------------------------------------------

def risk_audit(report: Dict, smart: Dict, macro: Dict, mtf: Dict,
               per_tf_ta: Dict, completeness_pct: float,
               missing_sections: List[str]) -> Dict[str, Any]:
    audit = {}

    # 1. Crowding
    f = smart.get("funding_rate_pct")
    audit["funding_crowding"] = (
        {"status": "WARNING", "detail": f"Funding {f:+.4f}% — leveraged longs overcrowded."}
        if f is not None and f > 0.05 else {"status": "CLEAR"}
    )

    # 2. Liquidity
    vol_24h = _num(_g(report, "metadata", "market_data", "total_volume", "usd"))
    if vol_24h is None:
        vol_24h = _num(_g(report, "metadata", "total_volume_usd"))
    audit["liquidity"] = (
        {"status": "WARNING", "detail": f"24h volume ${vol_24h:,.0f} < $5M — slippage risk."}
        if vol_24h is not None and vol_24h < 5_000_000 else {"status": "CLEAR"}
    )

    # 3. Whale concentration
    top10 = _num(_g(report, "top_holders", "top10_holders_pct"))
    audit["whale_concentration"] = (
        {"status": "WARNING", "detail": f"Top 10 wallets hold {top10:.1f}% of supply."}
        if top10 is not None and top10 > 40 else {"status": "CLEAR"}
    )

    # 4. Token unlock
    days = _num(_g(report, "upcoming_events", "days_until_next_event"))
    audit["token_unlock"] = (
        {"status": "WARNING", "detail": f"Major event in {int(days)}d."}
        if days is not None and days <= 14 else {"status": "CLEAR"}
    )

    # 5. Beta
    beta = macro.get("beta_vs_btc")
    audit["beta"] = (
        {"status": "WARNING", "detail": f"Beta {beta:.2f} — BTC exposure amplified."}
        if beta is not None and beta > 2.0 else {"status": "CLEAR"}
    )

    # 6. MTF divergence
    per_tf = mtf.get("per_tf", {})
    scores = [v.get("score") for v in per_tf.values() if v.get("available")]
    divergent = len(scores) >= 2 and (max(scores) > 25 and min(scores) < -25)
    audit["mtf_divergence"] = (
        {"status": "WARNING", "detail": "Timeframes disagree sharply."}
        if divergent else {"status": "CLEAR"}
    )

    # 7. Volume confirmation (4H)
    vol_4h = _num(_g(per_tf_ta.get("4h") or {}, "volume_vs_avg"))
    audit["volume_confirmation"] = (
        {"status": "UNCONFIRMED", "detail": f"4H volume {vol_4h:.2f}x avg — below average."}
        if vol_4h is not None and vol_4h < 1.0 else {"status": "CONFIRMED"}
    )

    # 8. Data completeness
    audit["data_completeness"] = (
        {"status": "FULL"} if completeness_pct >= 95
        else {"status": "PARTIAL", "detail": f"{completeness_pct:.0f}% — missing: {', '.join(missing_sections) or 'n/a'}"}
    )

    # Overall rating
    warnings = sum(1 for k, v in audit.items() if v.get("status") in ("WARNING", "UNCONFIRMED", "PARTIAL"))
    if warnings >= 4:
        overall = "EXTREME"
    elif warnings >= 3:
        overall = "HIGH"
    elif warnings >= 1:
        overall = "MODERATE"
    else:
        overall = "LOW"
    audit["overall_risk"] = overall
    return audit


# ----------------------------------------------------------------------
# Narrative helpers — Wolf / Insider / Verdict prose
# ----------------------------------------------------------------------

def wolf_read(report: Dict, ta_4h: Dict, current_price: Optional[float]) -> str:
    ob = report.get("order_book") or {}
    rt = report.get("trade_flow") or {}
    imb = _num(ob.get("imbalance_ratio"))
    bwall = _num(ob.get("bid_wall_at"))
    awall = _num(ob.get("ask_wall_at"))
    buy_pct = _num(rt.get("buy_volume_pct"))
    sell_pct = _num(rt.get("sell_volume_pct"))
    accel = _num(rt.get("trade_acceleration_score"))
    vwap = _num(ta_4h.get("vwap"))
    pvw = ta_4h.get("price_vs_vwap")

    parts = []
    if imb is not None:
        if imb > 1.3:
            parts.append(f"Bid depth outweighs asks {imb:.2f}× — buyers absorbing supply, not chasing.")
        elif imb < 0.77:
            parts.append(f"Ask depth dominates {1/imb:.2f}× — sellers stacked, rallies likely sold.")
        else:
            parts.append(f"Book near balance ({imb:.2f}) — no side has clear commitment.")
    if awall and current_price:
        parts.append(f"Visible ask wall at {_fmt_price(awall)} acts as magnetic resistance above.")
    if bwall and current_price:
        parts.append(f"Bid wall at {_fmt_price(bwall)} is the line the market must hold.")
    if buy_pct is not None and sell_pct is not None:
        if buy_pct > 60:
            parts.append(f"Tape prints {buy_pct:.0f}% aggressive buys — flow is directional up.")
        elif sell_pct > 60:
            parts.append(f"Tape prints {sell_pct:.0f}% aggressive sells — distribution is active.")
    if accel is not None:
        if accel > 1.5:
            parts.append(f"Trade pace accelerating {accel:.2f}× — conviction is building.")
        elif accel < 0.7:
            parts.append(f"Trade pace decelerating ({accel:.2f}×) — interest fading, watch for reversal bait.")
    if pvw:
        parts.append(f"Price is {pvw} VWAP ({_fmt_price(vwap)}) — intraday institutional bias leans {'bullish' if pvw=='above' else 'bearish'}.")
    if not parts:
        parts.append("Microstructure data is thin — read with caution until book and tape return cleaner signal.")
    return " ".join(parts[:5])


def insider_read(smart: Dict, report: Dict) -> str:
    lines = []
    f = smart.get("funding_rate_pct")
    if f is not None:
        if f > 0.05:
            lines.append(f"Funding at {f:+.4f}% — longs are paying a tax; spot upside is rented, not owned.")
        elif f < -0.02:
            lines.append(f"Funding at {f:+.4f}% — shorts are bleeding, squeeze fuel is priced in.")
        else:
            lines.append(f"Funding near neutral ({f:+.4f}%) — no leveraged distortion, base rate respected.")
    ls = smart.get("long_short_ratio")
    if ls is not None:
        if ls > 1.5:
            lines.append(f"Long/short {ls:.2f} — crowd is long, reflexive flush risk is real.")
        elif ls < 0.8:
            lines.append(f"Long/short {ls:.2f} — crowd is short, any strength gets chased.")
    nf = smart.get("net_flow_24h")
    if nf is not None:
        lines.append(f"Net exchange flow {nf:+.2f} — {smart.get('net_flow_interpretation') or 'mixed'}.")
    wb = smart.get("whale_buy_count")
    ws = smart.get("whale_sell_count")
    if wb is not None and ws is not None:
        lines.append(f"Whales: {int(wb)} buy prints vs {int(ws)} sell prints in the 24h window.")
    return " ".join(lines) or "Derivatives and on-chain streams incomplete — insider lens runs with reduced resolution."


def verdict(signal: str, final_score: float, regime: str, ev_24h: float,
            conflicts: List[str]) -> str:
    lean = {
        "STRONG_BUY": "The edge is asymmetric to the upside — full conviction long.",
        "BUY": "Odds favor the long side with standard sizing.",
        "WEAK_BUY": "A marginal long edge is present — wait for trigger before committing.",
        "NEUTRAL": "No edge. Do not trade. Capital is a position.",
        "WEAK_SELL": "Marginal short edge — tighten or reduce long exposure.",
        "SELL": "Conditions favor the short side — reduce exposure.",
        "STRONG_SELL": "Downside edge is asymmetric — avoid longs entirely.",
    }.get(signal, "No clean read.")
    pieces = [
        f"Regime: {regime}. Composite score {final_score:+.1f}.",
        lean,
        f"24H expected value {ev_24h:+.2f}% — {'favorable' if ev_24h > 0 else 'unfavorable'} edge.",
    ]
    if conflicts:
        pieces.append("Watch for: " + "; ".join(conflicts[:3]) + ".")
    pieces.append("Trade the plan, respect the invalidation, size for survival.")
    return " ".join(pieces)


# ----------------------------------------------------------------------
# Extended helpers — Snapshot / Confluence / Risk profile / Conviction
# ----------------------------------------------------------------------

SIGNAL_EMOJI = {
    "STRONG_BUY": "🟢🟢", "BUY": "🟢", "WEAK_BUY": "🟡↑",
    "NEUTRAL": "⚪",
    "WEAK_SELL": "🟡↓", "SELL": "🔴", "STRONG_SELL": "🔴🔴",
}


def _fg_label(v: Optional[float]) -> str:
    if v is None:
        return "n/a"
    if v < 25:
        return "Extreme Fear"
    if v < 45:
        return "Fear"
    if v <= 55:
        return "Neutral"
    if v <= 75:
        return "Greed"
    return "Extreme Greed"


def compute_market_snapshot(report: Dict, current_price: Optional[float]) -> Dict[str, Any]:
    """Pull headline market stats (perf, mcap, volume, ATH/ATL) from CoinGecko metadata."""
    meta = report.get("metadata") or {}
    md = meta.get("market_data") or {}

    def _usd(field: str):
        v = md.get(field)
        if isinstance(v, dict):
            return _num(v.get("usd"))
        return _num(v)

    snap: Dict[str, Any] = {
        "market_cap_usd": _usd("market_cap"),
        "market_cap_rank": _num(md.get("market_cap_rank") or meta.get("market_cap_rank")),
        "fdv_usd": _usd("fully_diluted_valuation"),
        "total_volume_usd": _usd("total_volume"),
        "circulating_supply": _num(md.get("circulating_supply")),
        "max_supply": _num(md.get("max_supply")),
        "ath_usd": _usd("ath"),
        "ath_change_pct": _num((md.get("ath_change_percentage") or {}).get("usd")
                               if isinstance(md.get("ath_change_percentage"), dict)
                               else md.get("ath_change_percentage")),
        "atl_usd": _usd("atl"),
        "change_24h_pct": _num((md.get("price_change_percentage_24h_in_currency") or {}).get("usd")
                               if isinstance(md.get("price_change_percentage_24h_in_currency"), dict)
                               else md.get("price_change_percentage_24h")),
        "change_7d_pct": _num((md.get("price_change_percentage_7d_in_currency") or {}).get("usd")
                              if isinstance(md.get("price_change_percentage_7d_in_currency"), dict)
                              else md.get("price_change_percentage_7d")),
        "change_30d_pct": _num((md.get("price_change_percentage_30d_in_currency") or {}).get("usd")
                               if isinstance(md.get("price_change_percentage_30d_in_currency"), dict)
                               else md.get("price_change_percentage_30d")),
    }
    # Volume / mcap turnover ratio (proxy for liquidity health)
    if snap["total_volume_usd"] and snap["market_cap_usd"]:
        snap["volume_mcap_ratio"] = round(snap["total_volume_usd"] / snap["market_cap_usd"], 4)
    else:
        snap["volume_mcap_ratio"] = None
    return snap


def compute_confluence(mtf: Dict, smart: Dict, pattern: Dict, sentiment: Dict,
                       macro: Dict) -> Dict[str, Any]:
    """Aggregate every contribution across phases into bullish/bearish factor lists."""
    bullish: List[Tuple[str, float, str]] = []
    bearish: List[Tuple[str, float, str]] = []

    def absorb(items: List[Tuple[str, float]], origin: str):
        for label, pts in items or []:
            try:
                p = float(pts)
            except (TypeError, ValueError):
                continue
            if p > 0:
                bullish.append((label, p, origin))
            elif p < 0:
                bearish.append((label, p, origin))

    # MTF contributions are nested per timeframe
    for tf, blk in (mtf.get("per_tf") or {}).items():
        if blk and blk.get("available"):
            absorb(blk.get("contributions") or [], f"MTF/{tf.upper()}")
    absorb(smart.get("contributions") or [], "Smart-Money")
    # Pattern: synthesise a pseudo-contribution if direction known
    if pattern.get("score"):
        absorb([(f"Pattern: {pattern.get('pattern_name','?')}", pattern["score"])], "Pattern")
    if sentiment.get("score"):
        absorb([("Sentiment composite", sentiment["score"])], "Sentiment")
    if macro.get("score"):
        absorb([("Macro composite", macro["score"])], "Macro")

    bullish.sort(key=lambda r: -r[1])
    bearish.sort(key=lambda r: r[1])
    return {
        "bullish_factors": [{"label": l, "points": round(p, 2), "origin": o} for l, p, o in bullish[:10]],
        "bearish_factors": [{"label": l, "points": round(p, 2), "origin": o} for l, p, o in bearish[:10]],
        "bullish_count": len(bullish),
        "bearish_count": len(bearish),
        "net_factors": len(bullish) - len(bearish),
    }


def compute_risk_profile(report: Dict) -> Dict[str, Any]:
    """Lift risk_metrics from the report (already computed by tool_calculate_risk_metrics)."""
    rm = report.get("risk_metrics") or {}
    if not isinstance(rm, dict) or not rm:
        return {"available": False}
    return {
        "available": True,
        "volatility_30d_pct": _num(rm.get("volatility_30d") or rm.get("annualized_volatility")),
        "sharpe_ratio": _num(rm.get("sharpe_ratio")),
        "sortino_ratio": _num(rm.get("sortino_ratio")),
        "max_drawdown_pct": _num(rm.get("max_drawdown_30d") or rm.get("max_drawdown")),
        "var_95_pct": _num(rm.get("var_95")),
        "cvar_95_pct": _num(rm.get("cvar_95")),
        "calmar_ratio": _num(rm.get("calmar_ratio")),
        "beta_vs_btc": _num(rm.get("beta_vs_btc")),
        "correlation_btc": _num(rm.get("correlation_btc")),
        "risk_category": rm.get("risk_category"),
    }


def compute_watchlist_triggers(current_price: Optional[float], atr_4h: Optional[float],
                               sr: Dict, ob: Dict) -> Dict[str, Any]:
    """When signal is FLAT, emit concrete bias-flip triggers based on S/R + ATR."""
    if not current_price:
        return {"available": False}
    res = (sr or {}).get("strong_resistances") or []
    sup = (sr or {}).get("strong_supports") or []
    r1 = _num((res[0] or {}).get("level")) if res else None
    s1 = _num((sup[0] or {}).get("level")) if sup else None
    atr = atr_4h or (current_price * 0.02)
    ask_wall = _num((ob or {}).get("ask_wall_at"))
    bid_wall = _num((ob or {}).get("bid_wall_at"))

    bull_break = r1 if (r1 and r1 > current_price) else (current_price + atr)
    bear_break = s1 if (s1 and s1 < current_price) else (current_price - atr)

    return {
        "available": True,
        "go_long_if": {
            "trigger": f"4H close > ${_fmt_price(bull_break)} on volume ≥ 150% of 20-period avg",
            "level": bull_break,
            "confirms": "trend continuation; targets next R + ATR projection",
        },
        "go_short_if": {
            "trigger": f"4H close < ${_fmt_price(bear_break)} on expanding volume",
            "level": bear_break,
            "confirms": "structure breakdown; targets next S + ATR projection",
        },
        "stand_aside_until": (
            f"price compresses inside ${_fmt_price(bear_break)}–${_fmt_price(bull_break)} "
            f"with funding/whale flow alignment"
        ),
        "ask_wall": ask_wall,
        "bid_wall": bid_wall,
    }


def compute_conviction(completeness_pct: float, mtf: Dict, regime: str,
                       audit: Dict, comp: Dict) -> Dict[str, Any]:
    """Synthesize a confidence label combining data, MTF agreement, regime stability, risk audit."""
    score = 50.0
    notes: List[str] = []

    # Data completeness
    if completeness_pct >= 95:
        score += 15
    elif completeness_pct >= 80:
        score += 5
    else:
        score -= 15
        notes.append(f"data {completeness_pct:.0f}% (reduced confidence)")

    # MTF agreement (std dev across timeframes)
    per_tf = mtf.get("per_tf") or {}
    scores = [v.get("score") for v in per_tf.values() if v.get("available")]
    if len(scores) >= 2:
        spread = max(scores) - min(scores)
        if spread < 30:
            score += 15
            notes.append("timeframes aligned")
        elif spread < 60:
            score += 0
        else:
            score -= 15
            notes.append("timeframes disagree")

    # Regime
    if regime == "TRENDING":
        score += 15
    elif regime == "RANGING":
        score += 5
    elif regime == "VOLATILE":
        score -= 10
        notes.append("breakout pending")
    elif regime == "CHAOTIC":
        score -= 25
        notes.append("vol too high to commit")

    # Risk audit warnings
    warns = sum(1 for k, v in audit.items()
                if isinstance(v, dict) and v.get("status") in ("WARNING", "UNCONFIRMED", "PARTIAL"))
    score -= warns * 5
    score = _clamp(score, 0, 100)
    if score >= 70:
        label = "HIGH"
    elif score >= 45:
        label = "MEDIUM"
    elif score >= 25:
        label = "LOW"
    else:
        label = "VERY_LOW"
    return {"score": round(score, 1), "label": label, "notes": notes}


def compute_catalysts(report: Dict) -> Dict[str, Any]:
    """Extract upcoming events worth surfacing in the brief."""
    ev = report.get("upcoming_events") or {}
    days = _num(ev.get("days_until_next_event"))
    events = ev.get("events") or []
    return {
        "days_until_next": days,
        "events": events[:3],
        "has_near_term": days is not None and days <= 14,
    }


def _run_quant_engine(report: Dict, ta_all: Dict, current_price: Optional[float]) -> Dict[str, Any]:
    """
    Phase 10 — Quant Engine.
    Runs: spot flow, accumulation/distribution, data-driven regime,
    volatility sustainability, failure mode detection, and EV model.

    This is the authoritative signal for position sizing decisions.
    It replaces the heuristic composite for EV-driven sizing,
    while the heuristic phases 1-9 continue providing supporting context.

    NOTE: The EV model is UNCALIBRATED until backtested with real historical
    data via the feature_validator module. Treat P(up) as directional guidance,
    not calibrated probability, until training data has been collected.
    """
    result: Dict[str, Any] = {"available": False, "error": None}

    try:
        from spot_flow_engine import analyze_spot_flow
        from accumulation_engine import analyze_accumulation
        from regime_classifier import classify_regime
        from volatility_engine import analyze_volatility_sustainability
        from failure_detector import detect_all_failure_modes
        from ev_model import predict_ev, build_feature_vector

        # --- Extract raw data from report ---
        candles_15m = _extract_candles(ta_all, "15m", report)
        candles_1h = _extract_candles(ta_all, "1h", report)
        candles_4h = _extract_candles(ta_all, "4h", report)
        candles_1d = _extract_candles(ta_all, "1d", report)

        trades = _g(report, "trade_flow") or {}
        raw_trades = trades.get("trades_sample") or []
        order_book = report.get("order_book") or {}
        primary_candles = candles_4h or candles_1h or []

        # --- Spot Flow Analysis ---
        spot_flow = {}
        if raw_trades and primary_candles and current_price:
            spot_flow = analyze_spot_flow(raw_trades, primary_candles, order_book, current_price)

        # --- Accumulation / Distribution ---
        accum = {}
        if (candles_15m or candles_1h or candles_4h) and current_price:
            accum = analyze_accumulation(candles_15m, candles_1h, candles_4h, current_price)

        # --- Data-Driven Regime Classification ---
        regime_quant = {}
        if primary_candles:
            regime_quant = classify_regime(candles_4h or [], candles_1d or [])

        # --- Volatility Engine ---
        vol_engine = {}
        if primary_candles:
            ofi_score = _g(spot_flow, "net_buying_pressure", "net_buying_pressure_score") or 0.0
            accum_score = _g(accum, "scores", "accumulation_probability") or 50.0
            regime_name = regime_quant.get("regime", "UNKNOWN")
            vol_engine = analyze_volatility_sustainability(
                primary_candles, regime_name, ofi_score, accum_score
            )

        # --- Failure Mode Detection ---
        sr = report.get("support_resistance") or {}
        sup_levels = [s.get("level") for s in (sr.get("strong_supports") or [])[:3] if s.get("level")]
        res_levels = [r.get("level") for r in (sr.get("strong_resistances") or [])[:3] if r.get("level")]
        key_levels = [l for l in sup_levels + res_levels if l is not None]

        per_tf_scores = {}
        for tf in ["15m", "1h", "4h", "1d"]:
            ta = ta_all.get(tf) or {}
            if ta and not ta.get("error"):
                per_tf_scores[tf] = _num(ta.get("signal_score")) or 0.0

        failure_modes = detect_all_failure_modes(
            candles=primary_candles,
            key_levels=key_levels,
            ofi_score=_g(spot_flow, "net_buying_pressure", "net_buying_pressure_score") or 0.0,
            regime=regime_quant.get("regime", "UNKNOWN"),
            mtf_scores=per_tf_scores,
            trades=raw_trades or None,
        )

        # --- EV Model ---
        ev_pred = {}
        macro_data = {
            "fear_greed_value": _g(report, "macro_context", "fear_greed", "current_value") or 50.0
        }

        # Determine ATR-based price targets
        atr_4h = _num(_g(ta_all, "4h", "atr")) or (current_price * 0.02 if current_price else 1.0)
        target_up = (atr_4h / (current_price or 1.0)) * 100 * 1.5   # 1.5×ATR as TP1
        target_down = (atr_4h / (current_price or 1.0)) * 100 * 1.0  # 1.0×ATR as SL

        ev_pred = predict_ev(
            ofi=spot_flow.get("ofi") or {},
            absorption=spot_flow.get("absorption") or {},
            accum=accum,
            regime=regime_quant,
            vol=vol_engine,
            macro=macro_data,
            target_up_pct=max(0.5, target_up),
            target_down_pct=max(0.3, target_down),
        )

        # Apply failure mode confidence multiplier to EV signal
        conf_mult = failure_modes.get("confidence_multiplier", 1.0)
        if ev_pred:
            ev_pred["confidence_after_failure_filter"] = round(
                ev_pred.get("model_confidence", 0) * conf_mult, 4
            )
            ev_pred["failure_mode_applied"] = conf_mult < 1.0

        result = {
            "available": True,
            "spot_flow": spot_flow.get("summary") or {},
            "spot_flow_detail": {
                "ofi": spot_flow.get("ofi") or {},
                "absorption": spot_flow.get("absorption") or {},
                "icebergs": spot_flow.get("icebergs") or {},
            },
            "accumulation": {
                "summary": accum.get("summary") or {},
                "scores": (accum.get("scores") or {}),
                "cvd_trends": {
                    tf: (accum.get("cvd") or {}).get(tf, {}).get("cvd_trend", "n/a")
                    for tf in ["15m", "1h", "4h"]
                },
                "divergences": {
                    tf: (accum.get("divergence") or {}).get(tf, {}).get("divergence", "none")
                    for tf in ["15m", "1h", "4h"]
                },
            },
            "regime_quant": {
                "regime": regime_quant.get("regime"),
                "confidence": regime_quant.get("confidence"),
                "strategy_bias": regime_quant.get("strategy_bias"),
                "interpretation": regime_quant.get("interpretation"),
                "metrics": regime_quant.get("all_metrics") or {},
            },
            "volatility": {
                "vol_state": (vol_engine.get("realized_vol") or {}).get("vol_state"),
                "persistence_score": (vol_engine.get("trend_persistence") or {}).get("persistence_score"),
                "p_continuation_pct": (vol_engine.get("direction_probabilities") or {}).get("p_continuation_pct"),
                "dominant_outcome": (vol_engine.get("direction_probabilities") or {}).get("dominant_outcome"),
            },
            "failure_modes": {
                "confidence_multiplier": failure_modes.get("confidence_multiplier"),
                "recommended_action": failure_modes.get("recommended_action"),
                "clean_conditions": failure_modes.get("clean_conditions"),
                "failures_detected": [f.get("type") for f in failure_modes.get("failure_modes") or []],
                "summary": failure_modes.get("summary"),
            },
            "ev_model": ev_pred,
            "quant_signal": {
                "signal": ev_pred.get("signal", "NEUTRAL"),
                "p_up_pct": ev_pred.get("p_up_pct"),
                "p_down_pct": ev_pred.get("p_down_pct"),
                "ev_net_pct": ev_pred.get("ev_net_pct"),
                "is_positive_ev": ev_pred.get("is_positive_ev"),
                "top_5_features": ev_pred.get("top_5_features") or [],
                "regime": regime_quant.get("regime"),
                "confidence_after_failure_filter": ev_pred.get("confidence_after_failure_filter"),
                "recommended_action": failure_modes.get("recommended_action"),
                "calibration_warning": ev_pred.get("calibration_warning"),
                "model_calibrated": ev_pred.get("model_calibrated", False),
            },
        }

    except ImportError as e:
        result = {
            "available": False,
            "error": f"Quant module import failed: {e}. Run: pip install scipy",
        }
    except Exception as e:
        result = {
            "available": False,
            "error": f"Quant engine error: {e}",
        }

    return result


def _extract_candles(ta_all: Dict, timeframe: str, report: Dict) -> List[Dict]:
    """Extract raw candle data from stored OHLCV if available in report."""
    ta = ta_all.get(timeframe) or {}
    candles = ta.get("_candles") or []
    return candles if isinstance(candles, list) else []


def analyze(report: Dict) -> Dict[str, Any]:
    """Run Phases 1-10 on the structured intelligence report.

    Phases 1-9: Legacy heuristic pipeline (kept for full markdown brief)
    Phase 10: Quant engine — EV model, spot flow, accumulation, regime,
              volatility sustainability, failure mode detection.

    Use quant_analysis.quant_signal for EV-based sizing decisions.
    Use phases 1-9 for the supporting analysis narrative.
    """
    ta_all = report.get("technical_analysis") or {}
    ta_4h = ta_all.get("4h") or {}
    ta_1d = ta_all.get("1d") or {}
    ta_1h = ta_all.get("1h") or {}
    ta_15m = ta_all.get("15m") or {}

    # Current price
    price_block = report.get("current_price") or {}
    current_price = _num(price_block.get("average_price") or
                         price_block.get("binance_price") or
                         price_block.get("mexc_price") or
                         price_block.get("coingecko_price"))

    symbol = (_g(report, "collection_metadata", "symbol")
              or _g(report, "coin_profile", "symbol") or "")
    name = (_g(report, "collection_metadata", "coin")
            or _g(report, "coin_profile", "name") or symbol)
    is_btc = symbol.upper().startswith("BTC")

    # Risk metrics (beta, vol, etc.)
    rm = report.get("risk_metrics") or {}
    beta = _num(rm.get("beta_vs_btc"))

    # Phase 1
    regime_info = detect_regime(ta_4h, ta_1d)

    # Phase 2
    mtf = compute_mtf(ta_all, regime_info["regime"])

    # Phase 3 — Derivatives score capped at 40% weight
    smart_raw = compute_smart_money(report, current_price)
    smart = dict(smart_raw)
    smart["score"] = round(_clamp(smart_raw["score"] * 0.40, -40, 40), 2)
    smart["_derivatives_capped"] = True
    smart["_note"] = "Derivatives signals capped at 40% — funding/OI are spot context, not spot edge."

    # Phase 4
    pattern = compute_pattern_score(report.get("chart_patterns") or {}, mtf["mtf_score"])

    # Phase 5
    macro = compute_macro(report, is_btc, beta)

    # Sentiment sub-score (feeds 10% weight)
    sent = compute_sentiment_score(report)

    # Phase 6
    comp = composite(mtf["mtf_score"], smart["score"], pattern["score"],
                     sent["score"], macro["score"], regime_info["multiplier"])

    # Data completeness
    col_meta = report.get("collection_metadata") or {}
    completeness = _num(col_meta.get("data_completeness_pct")) or 0.0
    missing = [f.get("tool") for f in (col_meta.get("failed_tools") or [])]

    # Phase 7
    scenarios = build_scenarios(current_price, _num(ta_4h.get("atr")),
                                report.get("support_resistance") or {},
                                comp["final_score"])

    # Phase 8
    plan = build_trade_plan(
        comp["signal"], current_price, _num(ta_4h.get("atr")), _num(ta_4h.get("vwap")),
        report.get("support_resistance") or {}, comp["final_score"],
        beta, macro["size_haircut_from_beta"],
    )

    # Phase 9
    audit = risk_audit(report, smart, macro, mtf, ta_all, completeness, missing)

    # Phase 10 — Quant Engine (authoritative EV-based signal)
    quant_analysis = _run_quant_engine(report, ta_all, current_price)

    # Conflicts list for verdict narrative
    conflicts = []
    if pattern["conflicts_with_mtf"]:
        conflicts.append(f"pattern ({pattern['pattern_name']}) contradicts MTF")
    if audit["mtf_divergence"]["status"] != "CLEAR":
        conflicts.append("MTF timeframes diverge")
    if audit["volume_confirmation"]["status"] != "CONFIRMED":
        conflicts.append("volume below average")
    if audit["funding_crowding"]["status"] != "CLEAR":
        conflicts.append("funding crowded long")

    # Extended sections (snapshot, confluence, risk profile, watchlist, conviction, catalysts)
    market_snapshot = compute_market_snapshot(report, current_price)
    confluence = compute_confluence(mtf, smart, pattern, sent, macro)
    risk_profile = compute_risk_profile(report)
    watchlist = compute_watchlist_triggers(
        current_price, _num(ta_4h.get("atr")),
        report.get("support_resistance") or {},
        report.get("order_book") or {},
    )
    conviction = compute_conviction(completeness, mtf, regime_info["regime"], audit, comp)
    catalysts = compute_catalysts(report)

    result = {
        "symbol": symbol,
        "name": name,
        "current_price": current_price,
        "price_sources": {
            "binance": price_block.get("binance_price"),
            "mexc": price_block.get("mexc_price"),
            "coingecko": price_block.get("coingecko_price"),
            "spread_pct": price_block.get("price_spread_pct"),
        },
        "data_completeness_pct": completeness,
        "missing_sections": missing,
        "regime": regime_info,
        "mtf": mtf,
        "smart_money": smart,
        "pattern": pattern,
        "macro": macro,
        "sentiment_score": sent,
        "composite": comp,
        "scenarios": scenarios,
        "trade_plan": plan,
        "risk_audit": audit,
        "quant_analysis": quant_analysis,
        "market_snapshot": market_snapshot,
        "confluence": confluence,
        "risk_profile": risk_profile,
        "watchlist_triggers": watchlist,
        "conviction": conviction,
        "catalysts": catalysts,
        "narratives": {
            "wolf": wolf_read(report, ta_4h, current_price),
            "insider": insider_read(smart, report),
            "verdict": verdict(comp["signal"], comp["final_score"],
                               regime_info["regime"], scenarios.get("ev_24h_pct", 0.0),
                               conflicts),
        },
    }
    result["markdown"] = render_markdown(result, report)

    # ── State Persistence + Paper Trade Logging ──
    try:
        from utils import signal_store, paper_logger
        # Log signal to state history
        signal_store.append(symbol, {
            "signal": comp["signal"],
            "score": comp["final_score"],
            "price": current_price,
            "regime": regime_info["regime"],
        })
        # Check for signal flip
        flip = signal_store.detect_signal_flip(symbol, comp["signal"])
        if flip:
            result["signal_flip"] = {"from": flip, "to": comp["signal"]}

        # Paper trade logging for calibration dataset
        if current_price:
            paper_logger.log_signal(
                symbol, comp["signal"], comp["final_score"],
                current_price, regime_info["regime"],
                conviction.get("score", 0)
            )
            # Reconcile old entries that are now 24h+ old
            paper_logger.reconcile(symbol, current_price)
    except Exception:
        pass  # State persistence should never break analysis

    return result


# ----------------------------------------------------------------------
# Markdown rendering — matches FINAL OUTPUT FORMAT template
# ----------------------------------------------------------------------

def _zone(rsi: Optional[float]) -> str:
    if rsi is None:
        return "n/a"
    if rsi > 70:
        return "overbought"
    if rsi > 60:
        return "upper"
    if rsi > 40:
        return "neutral"
    if rsi > 30:
        return "lower"
    return "oversold"


def _bb_zone(pb: Optional[float]) -> str:
    if pb is None:
        return "n/a"
    if pb > 0.9:
        return "upper band"
    if pb > 0.6:
        return "upper half"
    if pb > 0.4:
        return "mid"
    if pb > 0.1:
        return "lower half"
    return "lower band"


def _sig_label(sig: str) -> str:
    return sig.replace("_", " ")


def render_markdown(result: Dict, report: Dict) -> str:
    import datetime as dt

    sym = result["symbol"]
    name = result["name"] or sym
    price = result["current_price"]
    ps = result["price_sources"]
    regime = result["regime"]
    mtf = result["mtf"]
    smart = result["smart_money"]
    pat = result["pattern"]
    macro = result["macro"]
    comp = result["composite"]
    sc = result["scenarios"]
    plan = result["trade_plan"]
    audit = result["risk_audit"]
    narr = result["narratives"]

    ta_all = report.get("technical_analysis") or {}
    per_tf = mtf.get("per_tf", {})

    ts = dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    def tf_row(tf: str) -> str:
        ta = ta_all.get(tf) or {}
        s = per_tf.get(tf) or {}
        trend_dir = ta.get("trend_direction") or "n/a"
        rsi = _num(ta.get("rsi"))
        mc = ta.get("macd_crossover") or "none"
        pb = _num(ta.get("bb_percent_b"))
        vv = _num(ta.get("volume_vs_avg"))
        score = s.get("score", 0) if s.get("available") else "n/a"
        rsi_s = f"{rsi:.1f} {_zone(rsi)}" if rsi is not None else "n/a"
        vv_s = f"{vv:.2f}x" if vv is not None else "n/a"
        return f"| {tf.upper()} | {trend_dir} | {rsi_s} | {mc} | {_bb_zone(pb)} | {vv_s} | {score}/100 |"

    # Key levels table
    sr = report.get("support_resistance") or {}
    res = sr.get("strong_resistances") or []
    sup = sr.get("strong_supports") or []

    def lvl_row(label: str, obj: Optional[Dict], lvl_type: str) -> str:
        if not obj:
            return f"| {label} | n/a | {lvl_type} |  |  |"
        strength_score = _num(obj.get("strength_score")) or 0
        strength = "High" if strength_score >= 60 else ("Med" if strength_score >= 30 else "Low")
        return f"| {label} | {_fmt_price(obj.get('level'))} | {lvl_type} | Swing cluster | {strength} |"

    # Liq walls from order book
    ob = report.get("order_book") or {}
    bid_wall = ob.get("bid_wall_at")
    ask_wall = ob.get("ask_wall_at")

    # Derivatives values for insider table
    fr_pct = smart.get("funding_rate_pct")
    oi_usd = smart.get("open_interest_usdt")
    ls_ratio = smart.get("long_short_ratio")

    # Ichimoku state 4H
    ichi = _g(ta_all, "4h", "ichimoku") or {}
    atr_4h = _num(_g(ta_all, "4h", "atr"))
    atr_pct_4h = _num(_g(ta_all, "4h", "atr_pct"))
    if not isinstance(plan, dict) or "error" in plan or "conservative" not in plan:
        # Stub plan when no price/OHLCV was available — keeps renderer safe
        empty_tier = {"entry_lower": None, "entry_upper": None, "entry_ideal": None,
                      "stop_loss": None, "sl_pct": None,
                      "tp1": None, "tp1_pct": None, "tp2": None, "tp2_pct": None,
                      "tp3": None, "tp3_pct": None, "risk_per_unit": 0}
        plan = {
            "direction": "FLAT",
            "setup_quality": "No Trade",
            "conservative": dict(empty_tier),
            "moderate": dict(empty_tier),
            "aggressive": dict(empty_tier),
            "kelly": {"win_prob": "n/a", "avg_rr": "n/a", "full_kelly_pct": "n/a",
                      "half_kelly_pct": "n/a", "half_kelly_adj_pct": "n/a"},
            "position_size": {"position_usdt": "n/a", "pct_of_capital": "n/a"},
            "invalidation": {"level": None, "timeframe": "4H", "direction": "n/a"},
        }
    sl_cons = plan.get("conservative", {}).get("stop_loss")

    # EV24h
    ev = sc.get("ev_24h_pct", 0.0)
    ev_fav = "favorable edge" if ev > 0 else "unfavorable edge"

    # Build plan rows
    def plan_cell(tier: str, key: str, pct_key: Optional[str] = None) -> str:
        t = plan.get(tier) or {}
        v = t.get(key)
        if v is None:
            return "n/a"
        if pct_key:
            p = t.get(pct_key)
            return f"${_fmt_price(v)} ({_pct(p)})"
        return f"${_fmt_price(v)}"

    kelly = plan.get("kelly") or {}
    pos = plan.get("position_size") or {}
    inv = plan.get("invalidation") or {}

    # Risk audit rendering
    def ra(label: str, key: str) -> str:
        blk = audit.get(key) or {}
        st = blk.get("status", "n/a")
        detail = blk.get("detail")
        return f"- **{label}:** {st}" + (f" — {detail}" if detail else "")

    # Pre-pull extended sections
    snap = result.get("market_snapshot") or {}
    confl = result.get("confluence") or {}
    rprof = result.get("risk_profile") or {}
    watch = result.get("watchlist_triggers") or {}
    conv = result.get("conviction") or {}
    cats = result.get("catalysts") or {}

    sig = comp["signal"]
    badge = SIGNAL_EMOJI.get(sig, "")
    spread_str = ps.get("spread_pct")
    if isinstance(spread_str, (int, float)):
        spread_str = f"{spread_str:.4f}%"
    elif spread_str is None:
        spread_str = "n/a"
    else:
        spread_str = f"{spread_str}%"

    md = []
    md.append(f"# ORACLE INTELLIGENCE BRIEF — {name} ({sym})")
    md.append(f"**Timestamp:** {ts}  ")
    # Signal banner — at-a-glance call
    md.append(
        f"**Signal:** {badge} **{_sig_label(sig)}** · "
        f"Score **{comp['final_score']:+.1f}/100** · "
        f"EV24H **{sc.get('ev_24h_pct', 0):+.2f}%** · "
        f"Conviction **{conv.get('label', 'n/a')}** ({conv.get('score','n/a')}/100)  "
    )
    def _src_price(v):
        return f"${_fmt_price(v)}" if v else "n/a"
    md.append(
        f"**Price:** ${_fmt_price(price)} "
        f"(Binance: {_src_price(ps.get('binance'))} | MEXC: {_src_price(ps.get('mexc'))} | "
        f"CoinGecko: {_src_price(ps.get('coingecko'))} | Spread: {spread_str})  "
    )
    md.append(f"**Market Regime:** {regime['regime']} — {regime['reason']}  ")
    miss = ", ".join(result.get("missing_sections") or []) or "none"
    md.append(f"**Data Completeness:** {result['data_completeness_pct']:.0f}% | Missing: {miss}")
    md.append("\n---\n")

    # ----- MARKET SNAPSHOT -----
    if any(snap.get(k) is not None for k in ("market_cap_usd", "total_volume_usd",
                                              "change_24h_pct", "change_7d_pct", "change_30d_pct",
                                              "ath_usd")):
        md.append("## MARKET SNAPSHOT\n")
        md.append("| Metric | Value | Metric | Value |")
        md.append("|---|---|---|---|")
        def _bigusd(v):
            if v is None:
                return "n/a"
            try:
                v = float(v)
            except (TypeError, ValueError):
                return "n/a"
            if v >= 1e9:
                return f"${v/1e9:.2f}B"
            if v >= 1e6:
                return f"${v/1e6:.2f}M"
            if v >= 1e3:
                return f"${v/1e3:.2f}K"
            return f"${v:.2f}"
        rank = snap.get("market_cap_rank")
        rank_s = f"#{int(rank)}" if rank is not None else "n/a"
        md.append(f"| Market Cap | {_bigusd(snap.get('market_cap_usd'))} (rank {rank_s}) | "
                  f"24h Volume | {_bigusd(snap.get('total_volume_usd'))} |")
        vmr = snap.get("volume_mcap_ratio")
        vmr_s = f"{vmr*100:.2f}%" if vmr is not None else "n/a"
        md.append(f"| FDV | {_bigusd(snap.get('fdv_usd'))} | "
                  f"Vol/MCap | {vmr_s} |")
        md.append(f"| 24h % | {_pct(snap.get('change_24h_pct'))} | "
                  f"7d % | {_pct(snap.get('change_7d_pct'))} |")
        md.append(f"| 30d % | {_pct(snap.get('change_30d_pct'))} | "
                  f"From ATH | {_pct(snap.get('ath_change_pct'))} |")
        cs = snap.get("circulating_supply"); ms = snap.get("max_supply")
        if cs is not None or ms is not None:
            cs_s = f"{cs:,.0f}" if cs is not None else "n/a"
            ms_s = f"{ms:,.0f}" if ms is not None else "uncapped"
            sup_pct = f"{(cs/ms*100):.1f}%" if (cs and ms) else "n/a"
            md.append(f"| Circ. Supply | {cs_s} | Max Supply | {ms_s} ({sup_pct} circ.) |")
        md.append("\n---\n")

    md.append("## THE WOLF'S READ — Market Microstructure")
    md.append("> *Order flow, crowd psychology, price action.*\n")
    md.append(narr["wolf"])
    md.append("\n---\n")

    md.append("## THE INSIDER'S READ — Smart Money & Derivatives")
    md.append("> *What the money is doing, not what it says.*\n")
    md.append(f"**Funding Rate:** {_pct(fr_pct, 4) if fr_pct is not None else 'n/a'} per period → "
              f"{smart.get('funding_sentiment') or 'n/a'} → "
              f"{'overcrowded longs' if (fr_pct or 0) > 0.05 else ('shorts bleeding' if (fr_pct or 0) < -0.02 else 'neutral positioning')}  ")
    md.append(f"**OI:** ${_fmt_price(oi_usd)}  ")
    whale = report.get("whale_and_onchain") or {}
    nf_val = smart.get("net_flow_24h")
    nf_interp = whale.get("net_flow_interpretation")
    if nf_val is not None:
        nf_line = f"**Exchange Flow (24h):** {nf_interp or 'mixed'} ({nf_val:+,.2f})  "
    elif nf_interp and nf_interp not in ("data_unavailable", "n/a"):
        nf_line = f"**Exchange Flow (24h):** {nf_interp}  "
    else:
        nf_line = "**Exchange Flow (24h):** n/a (source unavailable)  "
    md.append(nf_line)
    md.append(f"**Whale Activity:** {int(smart.get('whale_buy_count') or 0)} buys / "
              f"{int(smart.get('whale_sell_count') or 0)} sells in 24h  ")
    md.append(f"**Long/Short Ratio:** {ls_ratio if ls_ratio is not None else 'n/a'}\n")
    md.append(narr["insider"])
    md.append("\n---\n")

    md.append("## THE QUANT'S ANALYSIS — Multi-Timeframe Signal Engine\n")
    md.append(f"### Regime: {regime['regime']}\n")
    md.append("| Timeframe | Trend | RSI | MACD | BB | Volume | Score |")
    md.append("|---|---|---|---|---|---|---|")
    for tf in ("1d", "4h", "1h", "15m"):
        md.append(tf_row(tf))
    md.append(f"\n**Weighted MTF Score: {mtf['mtf_score']:+.2f}/100**\n")
    md.append(f"**Dominant Pattern:** {pat['pattern_name']} — {pat['pattern_direction']} — "
              f"Confidence: {pat.get('base_confidence_pct') or 'n/a'}%  ")
    div_flags = [tf for tf, s in per_tf.items() if s.get("rsi_divergence") in ("bullish", "bearish")]
    md.append(f"**Divergences Detected:** {', '.join(div_flags) if div_flags else 'none'}  ")
    md.append(f"**Ichimoku State (4H):** Price {ichi.get('price_vs_cloud', 'n/a')} cloud | "
              f"Cloud {ichi.get('cloud_color', 'n/a')} | TK: {ichi.get('tk_cross', 'none')}  ")
    atr_pct_str = f"{atr_pct_4h:.2f}%" if atr_pct_4h is not None else "n/a"
    # If trade plan is FLAT/blank, surface a hypothetical ATR-anchored SL so the line is informative.
    if sl_cons is None and price and atr_4h:
        hypo_long = price - 1.5 * atr_4h
        hypo_short = price + 1.5 * atr_4h
        sl_str = (f"hypothetical long ${_fmt_price(hypo_long)} / "
                  f"short ${_fmt_price(hypo_short)} (1.5×ATR)")
    else:
        sl_str = f"${_fmt_price(sl_cons)}"
    md.append(f"**ATR (4H):** ${_fmt_price(atr_4h)} "
              f"({atr_pct_str} of price) → SL for conservative: {sl_str}")
    md.append("\n---\n")

    md.append("## COMPOSITE SIGNAL SCORECARD\n")
    md.append("| Engine | Raw Score | Weight | Contribution |")
    md.append("|---|---|---|---|")
    md.append(f"| MTF Technical | {mtf['mtf_score']:+.2f}/100 | 40% | {comp['contributions']['mtf']:+.2f} |")
    md.append(f"| Smart Money (DPI+WFI) | {smart['score']:+.2f}/100 | 25% | {comp['contributions']['smart_money']:+.2f} |")
    md.append(f"| Chart Patterns | {pat['score']:+.2f}/100 | 15% | {comp['contributions']['pattern']:+.2f} |")
    md.append(f"| Sentiment | {result['sentiment_score']['score']:+.2f}/100 | 10% | {comp['contributions']['sentiment']:+.2f} |")
    md.append(f"| Macro Adjustment | {macro['score']:+.2f}/100 | 10% | {comp['contributions']['macro']:+.2f} |")
    md.append(f"| **Regime Multiplier** | ×{regime['multiplier']:.2f} | — | applied |")
    md.append(f"| **FINAL SCORE** | **{comp['final_score']:+.2f}/100** | — | **{_sig_label(comp['signal'])}** |")
    md.append("\n---\n")

    # ----- CONFLUENCE MAP -----
    bull_f = confl.get("bullish_factors") or []
    bear_f = confl.get("bearish_factors") or []
    if bull_f or bear_f:
        md.append("## CONFLUENCE MAP — What Aligns, What Conflicts\n")
        md.append(f"**Bullish factors:** {confl.get('bullish_count', 0)} · "
                  f"**Bearish factors:** {confl.get('bearish_count', 0)} · "
                  f"**Net:** {confl.get('net_factors', 0):+d}\n")
        md.append("| # | Bullish | +pts | Origin | Bearish | −pts | Origin |")
        md.append("|---|---|---|---|---|---|---|")
        for i in range(max(len(bull_f), len(bear_f), 1)):
            b = bull_f[i] if i < len(bull_f) else None
            r = bear_f[i] if i < len(bear_f) else None
            b_label = b["label"] if b else "—"
            b_pts = f"{b['points']:+.1f}" if b else ""
            b_org = b["origin"] if b else ""
            r_label = r["label"] if r else "—"
            r_pts = f"{r['points']:+.1f}" if r else ""
            r_org = r["origin"] if r else ""
            md.append(f"| {i+1} | {b_label} | {b_pts} | {b_org} | {r_label} | {r_pts} | {r_org} |")
            if i >= 6:
                break
        md.append("\n---\n")

    md.append("## KEY PRICE LEVELS\n")
    md.append("| Level | Price | Type | Method | Strength |")
    md.append("|---|---|---|---|---|")
    md.append(lvl_row("R3", res[2] if len(res) > 2 else None, "Resistance"))
    md.append(lvl_row("R2", res[1] if len(res) > 1 else None, "Resistance"))
    md.append(lvl_row("R1", res[0] if res else None, "Resistance"))
    md.append(f"| **NOW** | **${_fmt_price(price)}** | Current | — | — |")
    md.append(lvl_row("S1", sup[0] if sup else None, "Support"))
    md.append(lvl_row("S2", sup[1] if len(sup) > 1 else None, "Support"))
    md.append(lvl_row("S3", sup[2] if len(sup) > 2 else None, "Support"))
    md.append(f"| Liquidation Wall (Longs) | ${_fmt_price(bid_wall)} | Danger | Order book | "
              f"{'High' if bid_wall else 'n/a'} |")
    md.append(f"| Liquidation Wall (Shorts) | ${_fmt_price(ask_wall)} | Magnet | Order book | "
              f"{'High' if ask_wall else 'n/a'} |")
    md.append("\n---\n")

    md.append("## TRADE EXECUTION PLAN\n")
    md.append(f"**Direction:** {plan.get('direction', 'FLAT')}  ")
    md.append(f"**Setup Quality:** {plan.get('setup_quality', 'No Trade')}  ")
    md.append(f"**Expected Value (24H):** {ev:+.2f}% → {ev_fav}\n")

    # When FLAT and we have watchlist triggers, replace the empty Conservative/Moderate/Aggressive
    # table with concrete bias-flip triggers — far more useful than a wall of "n/a".
    if plan.get("direction") == "FLAT" and watch.get("available"):
        gl = watch.get("go_long_if") or {}
        gs = watch.get("go_short_if") or {}
        md.append("> *No edge right now. Track these conditional triggers; do not pre-empt them.*\n")
        md.append("| Bias Flip | Trigger | Confirms |")
        md.append("|---|---|---|")
        md.append(f"| **GO LONG** | {gl.get('trigger','n/a')} | {gl.get('confirms','')} |")
        md.append(f"| **GO SHORT** | {gs.get('trigger','n/a')} | {gs.get('confirms','')} |")
        md.append(f"| **STAND ASIDE** | {watch.get('stand_aside_until','until clarity')} | preserve capital |")
        md.append("\n*Trade tiers, Kelly sizing, and invalidation are intentionally blank — "
                  "they activate only when a directional signal fires.*")
        md.append("\n---\n")
    else:
        md.append("| Parameter | Conservative | Moderate | Aggressive |")
        md.append("|---|---|---|---|")
        md.append("| Entry Zone | "
                  f"${_fmt_price(plan['conservative']['entry_lower'])}–${_fmt_price(plan['conservative']['entry_upper'])} | "
                  f"${_fmt_price(plan['moderate']['entry_lower'])}–${_fmt_price(plan['moderate']['entry_upper'])} | "
                  f"${_fmt_price(plan['aggressive']['entry_ideal'])} (market) |")
        md.append(f"| Stop Loss | {plan_cell('conservative', 'stop_loss', 'sl_pct')} | "
                  f"{plan_cell('moderate', 'stop_loss', 'sl_pct')} | "
                  f"{plan_cell('aggressive', 'stop_loss', 'sl_pct')} |")
        md.append(f"| TP1 — 40% | {plan_cell('conservative','tp1','tp1_pct')} | "
                  f"{plan_cell('moderate','tp1','tp1_pct')} | {plan_cell('aggressive','tp1','tp1_pct')} |")
        md.append(f"| TP2 — 35% | {plan_cell('conservative','tp2','tp2_pct')} | "
                  f"{plan_cell('moderate','tp2','tp2_pct')} | {plan_cell('aggressive','tp2','tp2_pct')} |")
        md.append(f"| TP3 — trail 25% | {plan_cell('conservative','tp3','tp3_pct')} | "
                  f"{plan_cell('moderate','tp3','tp3_pct')} | {plan_cell('aggressive','tp3','tp3_pct')} |")
        md.append("| R:R Ratio | 1:1.5 / 1:2.5 / 1:4.0 | 1:1.5 / 1:2.5 / 1:4.0 | 1:1.5 / 1:2.5 / 1:4.0 |")
        md.append("")
        md.append(f"**Kelly Fraction:** Win prob {kelly.get('win_prob','n/a')}% | "
                  f"Avg R:R {kelly.get('avg_rr','n/a')} → Full Kelly: {kelly.get('full_kelly_pct','n/a')}% | "
                  f"Half Kelly (recommended): {kelly.get('half_kelly_pct','n/a')}%  ")
        md.append(f"**Position Size ($10k capital, 1% risk):** "
                  f"${pos.get('position_usdt','n/a')} "
                  f"({pos.get('pct_of_capital','n/a')}% of capital)  ")
        md.append(f"**Invalidation:** Price closes {inv.get('direction','n/a')} "
                  f"${_fmt_price(inv.get('level'))} on {inv.get('timeframe','4H')} — exit immediately.")
        md.append("\n---\n")

    md.append("## PRICE PREDICTION\n")
    md.append("| Scenario | Probability | 24H Target | 7D Target | Trigger |")
    md.append("|---|---|---|---|---|")
    b = sc.get("bull", {})
    base = sc.get("base", {})
    br = sc.get("bear", {})
    md.append(f"| Bull | {b.get('prob_pct','n/a')}% | "
              f"${_fmt_price(b.get('target_24h'))} ({_pct(b.get('target_24h_pct'))}) | "
              f"${_fmt_price(b.get('target_7d'))} ({_pct(b.get('target_7d_pct'))}) | "
              f"{b.get('trigger','')} |")
    md.append(f"| Base Case | {base.get('prob_pct','n/a')}% | "
              f"${_fmt_price(base.get('range_24h_low'))}–${_fmt_price(base.get('range_24h_high'))} | "
              f"${_fmt_price(base.get('range_7d_low'))}–${_fmt_price(base.get('range_7d_high'))} | "
              f"{base.get('trigger','')} |")
    md.append(f"| Bear | {br.get('prob_pct','n/a')}% | "
              f"${_fmt_price(br.get('target_24h'))} ({_pct(br.get('target_24h_pct'))}) | "
              f"${_fmt_price(br.get('target_7d'))} ({_pct(br.get('target_7d_pct'))}) | "
              f"{br.get('trigger','')} |")
    md.append(f"\n**24H Expected Value: {ev:+.2f}%** — {ev_fav}")
    md.append("\n---\n")

    # ----- RISK PROFILE (statistical, lifted from risk_metrics) -----
    if rprof.get("available"):
        md.append("## RISK PROFILE — Statistical\n")
        def _rv(v, suffix=""):
            if v is None:
                return "n/a"
            try:
                return f"{float(v):.2f}{suffix}"
            except (TypeError, ValueError):
                return "n/a"
        md.append("| Metric | Value | Metric | Value |")
        md.append("|---|---|---|---|")
        md.append(f"| 30d Volatility | {_rv(rprof.get('volatility_30d_pct'),'%')} | "
                  f"Risk Category | {rprof.get('risk_category') or 'n/a'} |")
        md.append(f"| Sharpe Ratio | {_rv(rprof.get('sharpe_ratio'))} | "
                  f"Sortino Ratio | {_rv(rprof.get('sortino_ratio'))} |")
        md.append(f"| Max Drawdown | {_rv(rprof.get('max_drawdown_pct'),'%')} | "
                  f"Calmar Ratio | {_rv(rprof.get('calmar_ratio'))} |")
        md.append(f"| VaR 95% (1d) | {_rv(rprof.get('var_95_pct'),'%')} | "
                  f"CVaR 95% | {_rv(rprof.get('cvar_95_pct'),'%')} |")
        md.append(f"| Beta vs BTC | {_rv(rprof.get('beta_vs_btc'))} | "
                  f"Correlation BTC | {_rv(rprof.get('correlation_btc'))} |")
        md.append("\n---\n")

    # ----- CATALYSTS (only if any near-term events surfaced) -----
    if cats.get("events"):
        md.append("## UPCOMING CATALYSTS\n")
        for e in cats["events"]:
            title = (e or {}).get("title") or (e or {}).get("name") or "event"
            when = (e or {}).get("date") or (e or {}).get("when") or ""
            md.append(f"- **{title}** {('— ' + str(when)) if when else ''}")
        if cats.get("days_until_next") is not None:
            md.append(f"\n*Next event in ~{int(cats['days_until_next'])} days.*")
        md.append("\n---\n")

    md.append("## RISK AUDIT\n")
    md.append(ra("Funding rate crowding", "funding_crowding"))
    md.append(ra("Liquidity", "liquidity"))
    md.append(ra("Whale concentration", "whale_concentration"))
    md.append(ra("Token unlock pressure", "token_unlock"))
    md.append(ra("Beta / BTC amplification", "beta"))
    md.append(ra("MTF divergence", "mtf_divergence"))
    md.append(ra("Volume confirmation", "volume_confirmation"))
    md.append(ra("Data completeness", "data_completeness"))
    md.append(f"\n**Overall Risk Rating:** {audit['overall_risk']}")
    md.append("\n---\n")

    md.append("## ORACLE'S VERDICT\n")
    md.append(f"{badge} **{_sig_label(sig)}** · Conviction **{conv.get('label','n/a')}** "
              f"({conv.get('score','n/a')}/100) · Risk: **{audit.get('overall_risk','?')}**\n")
    md.append(narr["verdict"])
    if conv.get("notes"):
        md.append(f"\n*Conviction drivers:* {', '.join(conv['notes'])}.")

    # Key watch levels — directional triggers from S/R + walls
    watch_lines = []
    if watch.get("available"):
        gl = watch.get("go_long_if") or {}
        gs = watch.get("go_short_if") or {}
        if gl.get("level"):
            watch_lines.append(f"Bullish trigger: **${_fmt_price(gl['level'])}** (4H close + volume)")
        if gs.get("level"):
            watch_lines.append(f"Bearish trigger: **${_fmt_price(gs['level'])}** (4H close + volume)")
    if ask_wall:
        watch_lines.append(f"Magnetic ask wall: **${_fmt_price(ask_wall)}**")
    if bid_wall:
        watch_lines.append(f"Defended bid wall: **${_fmt_price(bid_wall)}**")
    if watch_lines:
        md.append("\n**Watch Levels:**")
        for w in watch_lines[:4]:
            md.append(f"- {w}")

    if cats.get("has_near_term") and cats.get("days_until_next") is not None:
        md.append(f"\n**⚠ Catalyst window:** event in ~{int(cats['days_until_next'])}d — "
                  f"size down or hedge into it.")

    md.append("\n---\n")
    md.append("*This analysis is generated from quantitative algorithms and market data. "
              "It is educational intelligence, not financial advice. Markets are probabilistic, "
              "not deterministic. Manage your risk. Size your positions. "
              "Never risk capital you cannot afford to lose.*")

    return "\n".join(md)
