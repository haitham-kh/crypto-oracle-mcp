from __future__ import annotations
"""
CryptoOracle MCP — Technical Indicators Module
Computes 25+ indicators from OHLCV candle data using the `ta` library.
Compatible with Python 3.9+.
"""

import numpy as np
import pandas as pd
from ta import trend, momentum, volatility, volume as vol_ind
from typing import Optional, List, Dict, Any


def candles_to_df(candles: List[Dict]) -> pd.DataFrame:
    """Convert candle dicts into a pandas DataFrame."""
    df = pd.DataFrame(candles)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if "open_time" in df.columns:
        df["datetime"] = pd.to_datetime(df["open_time"], unit="ms")
        df.set_index("datetime", inplace=True)
    return df


def _sl(series, default=None):
    """Get last non-NaN value."""
    if series is None or series.empty:
        return default
    val = series.dropna()
    if val.empty:
        return default
    v = float(val.iloc[-1])
    if np.isnan(v):
        return default
    return round(v, 8)


def compute_all_indicators(candles: List[Dict]) -> Dict[str, Any]:
    """Compute full suite of TA indicators from OHLCV candle list."""
    if not candles or len(candles) < 30:
        return {"error": "Insufficient candle data (need >=30)"}

    df = candles_to_df(candles)
    c, h, l, o, v = df["close"], df["high"], df["low"], df["open"], df["volume"]
    lc = float(c.iloc[-1])
    r = {}

    # ── TREND ──
    r["ema9"] = _sl(trend.ema_indicator(c, window=9))
    r["ema21"] = _sl(trend.ema_indicator(c, window=21))
    r["ema50"] = _sl(trend.ema_indicator(c, window=50))
    r["ema200"] = _sl(trend.ema_indicator(c, window=200))
    r["sma20"] = _sl(trend.sma_indicator(c, window=20))
    r["sma50"] = _sl(trend.sma_indicator(c, window=50))
    r["sma200"] = _sl(trend.sma_indicator(c, window=200))

    r["price_vs_ema200"] = "above" if (r["ema200"] and lc > r["ema200"]) else "below"
    r["golden_cross_active"] = (r["ema50"] or 0) > (r["ema200"] or 1e18)
    r["death_cross_active"] = (r["ema50"] or 1e18) < (r["ema200"] or 0)
    r["ema_cross_status"] = "bullish" if (r["ema9"] or 0) > (r["ema21"] or 1e18) else "bearish"

    # ADX
    adx_ind = trend.ADXIndicator(h, l, c, window=14)
    r["adx"] = _sl(adx_ind.adx())
    r["di_plus"] = _sl(adx_ind.adx_pos())
    r["di_minus"] = _sl(adx_ind.adx_neg())

    adx_v = r["adx"]
    r["trend_strength"] = "strong" if (adx_v and adx_v > 40) else ("moderate" if (adx_v and adx_v > 25) else "weak")
    dp, dm = r.get("di_plus"), r.get("di_minus")
    r["trend_direction"] = ("bullish" if (dp or 0) > (dm or 0) else "bearish") if dp is not None and dm is not None else "neutral"

    # Ichimoku
    try:
        ichi = trend.IchimokuIndicator(h, l, window1=9, window2=26, window3=52)
        tenkan = _sl(ichi.ichimoku_conversion_line())
        kijun = _sl(ichi.ichimoku_base_line())
        sa = _sl(ichi.ichimoku_a())
        sb = _sl(ichi.ichimoku_b())
        if sa and sb:
            ct, cb = max(sa, sb), min(sa, sb)
            pvc = "above" if lc > ct else ("below" if lc < cb else "inside")
            cc = "bullish" if sa > sb else "bearish"
        else:
            pvc, cc = "unknown", "unknown"
        tkc = ("bullish" if (tenkan or 0) > (kijun or 1e18) else "bearish") if tenkan and kijun else "none"
        r["ichimoku"] = {"tenkan": tenkan, "kijun": kijun, "senkou_a": sa, "senkou_b": sb,
                         "price_vs_cloud": pvc, "cloud_color": cc, "tk_cross": tkc}
    except Exception:
        r["ichimoku"] = {"price_vs_cloud": "unknown", "cloud_color": "unknown", "tk_cross": "none"}

    # ── MOMENTUM ──
    rsi_s = momentum.rsi(c, window=14)
    rv = _sl(rsi_s)
    r["rsi"] = rv
    r["rsi_zone"] = ("overbought" if rv > 70 else "upper" if rv > 60 else "neutral" if rv > 40 else "lower" if rv > 30 else "oversold") if rv else "unknown"
    r["rsi_divergence"] = _detect_rsi_divergence(c, rsi_s)

    macd_ind = trend.MACD(c, window_slow=26, window_fast=12, window_sign=9)
    ml, ms, mh_s = macd_ind.macd(), macd_ind.macd_signal(), macd_ind.macd_diff()
    r["macd_line"] = _sl(ml)
    r["macd_signal"] = _sl(ms)
    r["macd_histogram"] = _sl(mh_s)
    r["macd_above_signal"] = (r["macd_line"] or 0) > (r["macd_signal"] or 0)
    # Crossover
    if ml is not None and ms is not None and len(ml.dropna()) >= 2 and len(ms.dropna()) >= 2:
        pm, ps2 = float(ml.dropna().iloc[-2]), float(ms.dropna().iloc[-2])
        cm, cs2 = r["macd_line"] or 0, r["macd_signal"] or 0
        r["macd_crossover"] = "bullish_cross" if pm < ps2 and cm > cs2 else ("bearish_cross" if pm > ps2 and cm < cs2 else "none")
    else:
        r["macd_crossover"] = "none"

    # StochRSI
    try:
        sri = momentum.StochRSIIndicator(c, window=14, smooth1=3, smooth2=3)
        r["stochrsi_k"] = _sl(sri.stochrsi_k())
        r["stochrsi_d"] = _sl(sri.stochrsi_d())
    except Exception:
        r["stochrsi_k"], r["stochrsi_d"] = None, None

    r["cci"] = _sl(trend.cci(h, l, c, window=20))
    r["williams_r"] = _sl(momentum.williams_r(h, l, c, lbp=14))
    r["roc"] = _sl(momentum.roc(c, window=10))
    r["mfi"] = _sl(vol_ind.money_flow_index(h, l, c, v, window=14))
    try:
        r["cmf"] = _sl(vol_ind.chaikin_money_flow(h, l, c, v, window=21))
    except Exception:
        r["cmf"] = None

    # KDJ
    try:
        stoch = momentum.StochasticOscillator(h, l, c, window=9, smooth_window=3)
        k = _sl(stoch.stoch())
        d = _sl(stoch.stoch_signal())
        r["kdj_k"], r["kdj_d"] = k, d
        r["kdj_j"] = round(3 * (k or 0) - 2 * (d or 0), 4) if k and d else None
    except Exception:
        r["kdj_k"], r["kdj_d"], r["kdj_j"] = None, None, None

    # ── VOLATILITY ──
    bb = volatility.BollingerBands(c, window=20, window_dev=2)
    r["bb_upper"] = _sl(bb.bollinger_hband())
    r["bb_middle"] = _sl(bb.bollinger_mavg())
    r["bb_lower"] = _sl(bb.bollinger_lband())
    r["bb_percent_b"] = _sl(bb.bollinger_pband())
    bw = bb.bollinger_wband()
    r["bb_bandwidth"] = _sl(bw)
    if bw is not None and len(bw.dropna()) >= 20:
        r["bb_squeeze"] = float(bw.dropna().iloc[-1]) <= float(bw.dropna().tail(20).min()) * 1.05
    else:
        r["bb_squeeze"] = False

    atr_s = volatility.average_true_range(h, l, c, window=14)
    av = _sl(atr_s)
    r["atr"] = av
    r["atr_pct"] = round((av / lc) * 100, 4) if av and lc else None

    try:
        kc = volatility.KeltnerChannel(h, l, c, window=20, window_atr=20, multiplier=2)
        r["kc_upper"] = _sl(kc.keltner_channel_hband())
        r["kc_basis"] = _sl(kc.keltner_channel_mband())
        r["kc_lower"] = _sl(kc.keltner_channel_lband())
    except Exception:
        r["kc_upper"], r["kc_basis"], r["kc_lower"] = None, None, None

    try:
        dc = volatility.DonchianChannel(h, l, c, window=20)
        r["dc_upper"] = _sl(dc.donchian_channel_hband())
        r["dc_mid"] = _sl(dc.donchian_channel_mband())
        r["dc_lower"] = _sl(dc.donchian_channel_lband())
    except Exception:
        r["dc_upper"], r["dc_mid"], r["dc_lower"] = None, None, None

    if len(c) >= 20:
        lr = np.log(c / c.shift(1)).dropna()
        r["historical_volatility_annualized"] = round(float(lr.tail(20).std() * np.sqrt(365) * 100), 2)
    else:
        r["historical_volatility_annualized"] = None

    # ── VOLUME ──
    obv_s = vol_ind.on_balance_volume(c, v)
    r["obv"] = _sl(obv_s)
    r["obv_trend"] = _obv_trend(obv_s)

    # Rolling 24h VWAP (approximates intraday anchor)
    tp = (h + l + c) / 3
    tf = (candles[-1].get("open_time", 0) - candles[-2].get("open_time", 0)) if len(candles) >= 2 else 0
    # candles per 24h given the TF in ms
    per_day = max(1, int(round((24 * 3600 * 1000) / tf))) if tf else 24
    window = max(2, min(per_day, len(tp)))
    tpv_roll = (tp * v).rolling(window, min_periods=2).sum()
    v_roll = v.rolling(window, min_periods=2).sum()
    vwap = tpv_roll / v_roll
    r["vwap"] = _sl(vwap)
    r["price_vs_vwap"] = "above" if (r["vwap"] and lc > r["vwap"]) else "below"

    vma = trend.sma_indicator(v, window=20)
    vma_v = _sl(vma)
    cv = float(v.iloc[-1])
    if vma_v and vma_v > 0:
        r["volume_vs_avg"] = round(cv / vma_v, 2)
        r["volume_spike"] = cv > (vma_v * 2.5)
    else:
        r["volume_vs_avg"], r["volume_spike"] = None, False

    if "taker_buy_volume" in df.columns:
        delta = df["taker_buy_volume"].fillna(0) - df["taker_sell_volume"].fillna(0)
        cvd = delta.cumsum()
        r["cvd"] = _sl(cvd)
        # CVD divergence: price making higher highs but CVD making lower highs = distribution
        r["cvd_divergence"] = _detect_cvd_divergence(c, cvd)
        # CVD trend (last 10 candles)
        r["cvd_trend"] = _cvd_trend(cvd)
        # Net delta ratio for recent candles (buy pressure vs sell pressure)
        recent_delta = delta.tail(20)
        buy_delta = recent_delta[recent_delta > 0].sum()
        sell_delta = abs(recent_delta[recent_delta < 0].sum())
        total_delta = buy_delta + sell_delta
        r["cvd_buy_pressure_pct"] = round(float(buy_delta / total_delta * 100), 2) if total_delta > 0 else 50.0
    else:
        r["cvd"] = None
        r["cvd_divergence"] = "none"
        r["cvd_trend"] = "flat"
        r["cvd_buy_pressure_pct"] = None

    # ── CANDLESTICK PATTERNS ──
    r["candlestick_patterns"] = _detect_candles(df)
    r["pivot_points"] = _pivots(df)

    # ── SIGNAL ──
    sc = _score(r, lc)
    r["signal_score"] = sc
    r["signal_summary"] = "STRONG_BUY" if sc >= 50 else "BUY" if sc >= 20 else "NEUTRAL" if sc >= -20 else "SELL" if sc >= -50 else "STRONG_SELL"
    return r


def _detect_rsi_divergence(close, rsi_s):
    """Pivot-based RSI divergence detection.
    
    Finds swing lows/highs in both price and RSI, then checks if
    consecutive pivots diverge in direction. Requires pivots separated
    by at least 10 candles to filter noise.
    """
    if rsi_s is None or len(rsi_s.dropna()) < 30:
        return "none"
    try:
        c_vals = close.values[-40:]
        r_vals = rsi_s.dropna().values[-40:]
        if len(c_vals) < 20 or len(r_vals) < 20:
            return "none"
        # Trim to same length
        min_len = min(len(c_vals), len(r_vals))
        c_vals = c_vals[-min_len:]
        r_vals = r_vals[-min_len:]
        
        # Find swing lows (for bullish divergence)
        order = 5
        price_lows = []
        rsi_lows = []
        for i in range(order, len(c_vals) - order):
            if c_vals[i] == min(c_vals[i - order:i + order + 1]):
                price_lows.append((i, float(c_vals[i])))
            if r_vals[i] == min(r_vals[i - order:i + order + 1]):
                rsi_lows.append((i, float(r_vals[i])))
        
        # Find swing highs (for bearish divergence)
        price_highs = []
        rsi_highs = []
        for i in range(order, len(c_vals) - order):
            if c_vals[i] == max(c_vals[i - order:i + order + 1]):
                price_highs.append((i, float(c_vals[i])))
            if r_vals[i] == max(r_vals[i - order:i + order + 1]):
                rsi_highs.append((i, float(r_vals[i])))
        
        # Bullish divergence: price makes lower low, RSI makes higher low
        if len(price_lows) >= 2 and len(rsi_lows) >= 2:
            pl1, pl2 = price_lows[-2], price_lows[-1]
            # Require at least 10 candles between pivots
            if pl2[0] - pl1[0] >= 10:
                # Find RSI lows closest to the price low indices
                rl1 = min(rsi_lows, key=lambda x: abs(x[0] - pl1[0]))
                rl2 = min(rsi_lows, key=lambda x: abs(x[0] - pl2[0]))
                if (pl2[1] < pl1[1] and  # price lower low
                    rl2[1] > rl1[1] and  # RSI higher low
                    abs(rl1[0] - pl1[0]) <= 3 and  # pivots aligned within 3 candles
                    abs(rl2[0] - pl2[0]) <= 3):
                    return "bullish"
        
        # Bearish divergence: price makes higher high, RSI makes lower high
        if len(price_highs) >= 2 and len(rsi_highs) >= 2:
            ph1, ph2 = price_highs[-2], price_highs[-1]
            if ph2[0] - ph1[0] >= 10:
                rh1 = min(rsi_highs, key=lambda x: abs(x[0] - ph1[0]))
                rh2 = min(rsi_highs, key=lambda x: abs(x[0] - ph2[0]))
                if (ph2[1] > ph1[1] and  # price higher high
                    rh2[1] < rh1[1] and  # RSI lower high
                    abs(rh1[0] - ph1[0]) <= 3 and
                    abs(rh2[0] - ph2[0]) <= 3):
                    return "bearish"
    except Exception:
        pass
    return "none"


def _detect_cvd_divergence(close, cvd_series):
    """Detect CVD vs price divergence using swing pivots.
    
    Price making higher highs while CVD makes lower highs = stealth distribution.
    Price making lower lows while CVD makes higher lows = stealth accumulation.
    """
    if cvd_series is None or len(cvd_series.dropna()) < 20:
        return "none"
    try:
        c_vals = close.values[-30:]
        cvd_vals = cvd_series.dropna().values[-30:]
        min_len = min(len(c_vals), len(cvd_vals))
        c_vals = c_vals[-min_len:]
        cvd_vals = cvd_vals[-min_len:]
        
        order = 4
        # Check for bearish CVD divergence (distribution)
        price_highs = []
        cvd_highs = []
        for i in range(order, len(c_vals) - order):
            if c_vals[i] == max(c_vals[i - order:i + order + 1]):
                price_highs.append((i, float(c_vals[i])))
            if cvd_vals[i] == max(cvd_vals[i - order:i + order + 1]):
                cvd_highs.append((i, float(cvd_vals[i])))
        
        if len(price_highs) >= 2 and len(cvd_highs) >= 2:
            ph1, ph2 = price_highs[-2], price_highs[-1]
            if ph2[0] - ph1[0] >= 8:
                ch1 = min(cvd_highs, key=lambda x: abs(x[0] - ph1[0]))
                ch2 = min(cvd_highs, key=lambda x: abs(x[0] - ph2[0]))
                if ph2[1] > ph1[1] and ch2[1] < ch1[1]:
                    return "bearish_distribution"
        
        # Check for bullish CVD divergence (accumulation)
        price_lows = []
        cvd_lows = []
        for i in range(order, len(c_vals) - order):
            if c_vals[i] == min(c_vals[i - order:i + order + 1]):
                price_lows.append((i, float(c_vals[i])))
            if cvd_vals[i] == min(cvd_vals[i - order:i + order + 1]):
                cvd_lows.append((i, float(cvd_vals[i])))
        
        if len(price_lows) >= 2 and len(cvd_lows) >= 2:
            pl1, pl2 = price_lows[-2], price_lows[-1]
            if pl2[0] - pl1[0] >= 8:
                cl1 = min(cvd_lows, key=lambda x: abs(x[0] - pl1[0]))
                cl2 = min(cvd_lows, key=lambda x: abs(x[0] - pl2[0]))
                if pl2[1] < pl1[1] and cl2[1] > cl1[1]:
                    return "bullish_accumulation"
    except Exception:
        pass
    return "none"


def _cvd_trend(cvd_series):
    """Determine CVD trend direction over last 10 candles."""
    if cvd_series is None or len(cvd_series.dropna()) < 10:
        return "flat"
    vals = cvd_series.dropna().tail(10).values
    # Simple linear regression slope
    x = np.arange(len(vals))
    slope = np.polyfit(x, vals, 1)[0]
    mean_cvd = np.mean(np.abs(vals)) if np.mean(np.abs(vals)) > 0 else 1
    normalized_slope = slope / mean_cvd
    if normalized_slope > 0.05:
        return "rising"
    elif normalized_slope < -0.05:
        return "falling"
    return "flat"


def _obv_trend(obv_s):
    if obv_s is None or len(obv_s.dropna()) < 5:
        return "flat"
    r = obv_s.dropna().tail(5).values
    if all(r[i] <= r[i+1] for i in range(len(r)-1)):
        return "up"
    if all(r[i] >= r[i+1] for i in range(len(r)-1)):
        return "down"
    return "flat"


def _detect_candles(df):
    pats = []
    if len(df) < 3:
        return pats
    o, h2, l2, c2 = df["open"].values, df["high"].values, df["low"].values, df["close"].values
    body = abs(c2[-1] - o[-1])
    uw = h2[-1] - max(o[-1], c2[-1])
    lw = min(o[-1], c2[-1]) - l2[-1]
    fr = h2[-1] - l2[-1]
    if fr == 0:
        return pats
    bp = body / fr
    if bp < 0.1:
        pats.append("doji")
    if lw > 2 * body and uw < body * 0.5 and c2[-1] > o[-1]:
        pats.append("hammer")
    if uw > 2 * body and lw < body * 0.5 and c2[-1] < o[-1]:
        pats.append("shooting_star")
    if uw > 2 * body:
        pats.append("bearish_pin_bar")
    if lw > 2 * body:
        pats.append("bullish_pin_bar")
    if len(df) >= 2:
        pb = abs(c2[-2] - o[-2])
        if c2[-2] < o[-2] and c2[-1] > o[-1] and body > pb and o[-1] <= c2[-2] and c2[-1] >= o[-2]:
            pats.append("bullish_engulfing")
        if c2[-2] > o[-2] and c2[-1] < o[-1] and body > pb and o[-1] >= c2[-2] and c2[-1] <= o[-2]:
            pats.append("bearish_engulfing")
    if len(df) >= 3:
        if all(c2[-i] > o[-i] for i in range(1, 4)) and c2[-1] > c2[-2] > c2[-3]:
            pats.append("three_white_soldiers")
        if all(c2[-i] < o[-i] for i in range(1, 4)) and c2[-1] < c2[-2] < c2[-3]:
            pats.append("three_black_crows")
    return pats


def _pivots(df):
    if len(df) < 2:
        return {}
    h2, l2, c2 = float(df["high"].iloc[-2]), float(df["low"].iloc[-2]), float(df["close"].iloc[-2])
    p = (h2 + l2 + c2) / 3
    return {"pivot": round(p, 8), "R1": round(2*p - l2, 8), "R2": round(p + (h2-l2), 8),
            "S1": round(2*p - h2, 8), "S2": round(p - (h2-l2), 8)}


def _score(ind, lc):
    s = 0
    rsi = ind.get("rsi")
    if rsi:
        if 40 <= rsi <= 60: s += 20
        elif rsi < 30: s += 10
        elif rsi > 70: s -= 10
    mc = ind.get("macd_crossover", "none")
    if mc == "bullish_cross": s += 15
    elif mc == "bearish_cross": s -= 15
    if ind.get("macd_histogram") and ind["macd_histogram"] > 0: s += 5
    elif ind.get("macd_histogram") and ind["macd_histogram"] < 0: s -= 5
    if ind.get("price_vs_ema200") == "above": s += 20
    else: s -= 20
    if ind.get("golden_cross_active"): s += 15
    if ind.get("death_cross_active"): s -= 15
    if ind.get("bb_squeeze"):
        pb = ind.get("bb_percent_b")
        if pb and pb > 0.8: s += 10
        elif pb and pb < 0.2: s -= 10
    if ind.get("volume_spike"):
        if ind.get("trend_direction") == "bullish": s += 10
        elif ind.get("trend_direction") == "bearish": s -= 10
    if ind.get("price_vs_vwap") == "above": s += 10
    else: s -= 10
    bp = {"hammer","bullish_engulfing","three_white_soldiers","bullish_pin_bar"}
    brp = {"shooting_star","bearish_engulfing","three_black_crows","bearish_pin_bar"}
    for p in ind.get("candlestick_patterns", []):
        if p in bp: s += 15; break
    for p in ind.get("candlestick_patterns", []):
        if p in brp: s -= 15; break
    mfi = ind.get("mfi")
    if mfi and mfi > 50: s += 10
    elif mfi and mfi < 50: s -= 10
    cmf = ind.get("cmf")
    if cmf and cmf > 0: s += 10
    elif cmf and cmf < 0: s -= 10
    return max(-100, min(100, s))
