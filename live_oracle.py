"""
live_oracle.py — Live analysis using XGBoost quant model
========================================================
1. Downloads latest 1m candles for a given SYMBOL + BTC from Binance
2. Runs the trained XGBoost EV model (v2) for P(up) prediction
3. Pulls current price, funding rate, order book
4. Computes leverage-aware TP/SL for 100x and 20x scenarios
"""
from __future__ import annotations
import os, sys, json, time, datetime, math
import numpy as np
import requests
import argparse

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, os.path.dirname(__file__))

BINANCE_BASE = "https://api.binance.com"
FAPI_BASE = "https://fapi.binance.com"

# ── Data Download ────────────────────────────────────────────────────────

def get_klines(symbol, interval="1m", limit=500):
    """Download recent klines from Binance."""
    url = f"{BINANCE_BASE}/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    candles = []
    for k in r.json():
        candles.append({
            "timestamp": int(k[0]),
            "open": float(k[1]), "high": float(k[2]),
            "low": float(k[3]), "close": float(k[4]),
            "volume": float(k[5]),
            "taker_buy_volume": float(k[9]),
        })
    return candles

def get_ticker(symbol):
    """Get 24h ticker stats."""
    url = f"{BINANCE_BASE}/api/v3/ticker/24hr"
    r = requests.get(url, params={"symbol": symbol}, timeout=10)
    r.raise_for_status()
    return r.json()

def get_funding_rate(symbol):
    """Get current funding rate from futures."""
    try:
        url = f"{FAPI_BASE}/fapi/v1/fundingRate"
        r = requests.get(url, params={"symbol": symbol, "limit": 8}, timeout=10)
        r.raise_for_status()
        return r.json()
    except:
        return []

def get_order_book(symbol, limit=50):
    """Get order book depth."""
    try:
        url = f"{BINANCE_BASE}/api/v3/depth"
        r = requests.get(url, params={"symbol": symbol, "limit": limit}, timeout=10)
        r.raise_for_status()
        return r.json()
    except:
        return {"bids": [], "asks": []}

def get_open_interest(symbol):
    """Get futures open interest."""
    try:
        url = f"{FAPI_BASE}/fapi/v1/openInterest"
        r = requests.get(url, params={"symbol": symbol}, timeout=10)
        r.raise_for_status()
        return r.json()
    except:
        return {}

# ── Build Arrays ─────────────────────────────────────────────────────────

def build_arrays(candles):
    ts = np.array([c["timestamp"] for c in candles], dtype=np.float64)
    o = np.array([c["open"] for c in candles], dtype=np.float64)
    h = np.array([c["high"] for c in candles], dtype=np.float64)
    l = np.array([c["low"] for c in candles], dtype=np.float64)
    c = np.array([c["close"] for c in candles], dtype=np.float64)
    v = np.array([x["volume"] for x in candles], dtype=np.float64)
    tbv = np.array([x.get("taker_buy_volume", 0) for x in candles], dtype=np.float64)
    tsv = v - tbv
    ofi = tbv - tsv  # = 2*tbv - v
    return ts, o, h, l, c, v, ofi

# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Live analysis for a given symbol using XGBoost Quant Model")
    parser.add_argument("symbol", type=str, nargs='?', default="PEPEUSDT", help="The trading symbol (e.g., BTCUSDT, PEPEUSDT)")
    args = parser.parse_args()
    
    symbol = args.symbol.upper()
    if not symbol.endswith("USDT"):
        symbol += "USDT"

    print("=" * 70)
    print(f"  🔮 {symbol} LIVE ORACLE — XGBoost Quant Model Analysis")
    print(f"  Timestamp: {datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print("=" * 70)

    # Step 1: Download data
    print(f"\n--- Downloading {symbol} 1m candles (1000 bars = ~16.7 hours) ---")
    try:
        symbol_candles = get_klines(symbol, "1m", 1000)
    except Exception as e:
        print(f"  ERROR: Failed to download {symbol} klines: {e}")
        return
        
    print(f"  Got {len(symbol_candles)} candles")
    
    print("--- Downloading BTCUSDT 1m candles (matching window) ---")
    btc_candles = get_klines("BTCUSDT", "1m", 1000)
    print(f"  Got {len(btc_candles)} candles")

    # Step 2: Get live market data
    print("\n--- Fetching live market data ---")
    try:
        ticker = get_ticker(symbol)
        current_price = float(ticker["lastPrice"])
        price_change_24h = float(ticker["priceChangePercent"])
        volume_24h = float(ticker["quoteVolume"])
        high_24h = float(ticker["highPrice"])
        low_24h = float(ticker["lowPrice"])
        
        print(f"  Current Price:    ${current_price:.10f}")
        print(f"  24h Change:       {price_change_24h:+.2f}%")
        print(f"  24h Volume:       ${volume_24h:,.0f}")
        print(f"  24h High:         ${high_24h:.10f}")
        print(f"  24h Low:          ${low_24h:.10f}")
    except Exception as e:
        print(f"  ERROR: Failed to get ticker for {symbol}: {e}")
        return

    # Funding rate
    funding_data = get_funding_rate(symbol)
    if funding_data:
        current_funding = float(funding_data[-1]["fundingRate"])
        avg_funding = np.mean([float(f["fundingRate"]) for f in funding_data])
        print(f"\n  Funding Rate:     {current_funding*100:.4f}%")
        print(f"  Avg Funding (24h): {avg_funding*100:.4f}%")
        funding_sentiment = "longs_paying" if current_funding > 0 else "shorts_paying"
        extreme_funding = abs(current_funding) > 0.001
        print(f"  Sentiment:        {funding_sentiment}")
        if extreme_funding:
            print(f"  ⚠️  EXTREME FUNDING — potential squeeze risk!")
    else:
        current_funding = 0
        avg_funding = 0
        funding_sentiment = "unknown"
        extreme_funding = False

    # Order book
    ob = get_order_book(symbol, 50)
    if ob["bids"] and ob["asks"]:
        total_bid = sum(float(b[0]) * float(b[1]) for b in ob["bids"])
        total_ask = sum(float(a[0]) * float(a[1]) for a in ob["asks"])
        imbalance = total_bid / max(total_ask, 1e-10)
        spread = (float(ob["asks"][0][0]) - float(ob["bids"][0][0])) / current_price * 100
        print(f"\n  Bid Depth (USDT):  ${total_bid:,.0f}")
        print(f"  Ask Depth (USDT):  ${total_ask:,.0f}")
        print(f"  Imbalance Ratio:   {imbalance:.3f} ({'buy pressure' if imbalance > 1.1 else 'sell pressure' if imbalance < 0.9 else 'balanced'})")
        print(f"  Spread:            {spread:.4f}%")
    else:
        imbalance = 1.0

    # Open interest
    oi_data = get_open_interest(symbol)
    if oi_data:
        oi_usdt = float(oi_data.get("openInterest", 0)) * current_price
        print(f"\n  Open Interest:     ${oi_usdt:,.0f}")

    # Step 3: Run XGBoost model
    print("\n" + "=" * 70)
    print("  🧠 XGBOOST MODEL PREDICTION (v2 — trained on 2.3M samples)")
    print("=" * 70)

    from train_ev_model_v2 import build_features_v2, _atr

    ts, op, hi, lo, cl, vol, ofi = build_arrays(symbol_candles)
    _, _, _, _, btc_cl, _, _ = build_arrays(btc_candles)
    
    # Align BTC to target symbol timestamps
    btc_ts_arr = np.array([c["timestamp"] for c in btc_candles], dtype=np.float64)
    btc_aligned = np.interp(ts, btc_ts_arr, btc_cl)

    F, atr_arr = build_features_v2(cl, hi, lo, vol, ofi, ts, btc_aligned)
    
    # Get the latest valid feature vector
    latest_idx = len(cl) - 1
    while latest_idx > 0 and np.any(np.isnan(F[latest_idx])):
        latest_idx -= 1

    if np.any(np.isnan(F[latest_idx])):
        print("  ERROR: Cannot compute features (insufficient data)")
        return

    # Load model
    import xgboost as xgb
    meta_path = os.path.join(os.path.dirname(__file__), "data", "ev_model_v2_meta.json")
    model_path = os.path.join(os.path.dirname(__file__), "data", "ev_model_xgb.json")
    
    if not os.path.exists(meta_path) or not os.path.exists(model_path):
        print("  ERROR: Model files not found in data/ directory.")
        return
        
    with open(meta_path) as f:
        meta = json.load(f)
    feature_names = meta["feature_names"]
    
    model = xgb.Booster()
    model.load_model(model_path)
    
    F_latest = F[latest_idx:latest_idx+1]
    dmat = xgb.DMatrix(F_latest, feature_names=feature_names)
    p_up = float(model.predict(dmat)[0])
    p_down = 1.0 - p_up

    print(f"\n  P(price up in next 60 bars):   {p_up*100:.1f}%")
    print(f"  P(price down in next 60 bars): {p_down*100:.1f}%")
    print(f"  Model confidence:              {abs(p_up - 0.5)*200:.1f}%")

    # Signal classification
    if p_up >= 0.60:
        signal = "🟢 STRONG BUY"
    elif p_up >= 0.55:
        signal = "🟢 BUY"
    elif p_up <= 0.40:
        signal = "🔴 STRONG SELL"
    elif p_up <= 0.45:
        signal = "🔴 SELL"
    else:
        signal = "🟡 NEUTRAL"
    print(f"  Signal:                        {signal}")

    # Feature breakdown
    importance = model.get_score(importance_type='gain')
    print(f"\n  Key Feature Values (latest bar):")
    feat_vals = {}
    for i, fname in enumerate(feature_names):
        feat_vals[fname] = F[latest_idx, i]
    
    sorted_feats = sorted(feat_vals.items(), key=lambda x: abs(x[1]), reverse=True)
    for fname, val in sorted_feats[:8]:
        direction = "↑" if val > 0 else "↓" if val < 0 else "—"
        print(f"    {fname:<25} {val:>8.4f}  {direction}")

    # ATR for TP/SL
    atr = atr_arr[latest_idx]
    if np.isnan(atr) or atr <= 0:
        # Fallback: compute from recent data
        atr = np.nanmean(hi[-14:] - lo[-14:])
    
    atr_pct = (atr / current_price) * 100
    hurst_val = F[latest_idx, 10]  # hurst feature

    # Regime-conditioned barriers (matching training logic)
    if hurst_val > 0.1:  # trending (raw > 0.55)
        tp_mult, sl_mult = 2.0, 1.0
        regime = "TRENDING"
    elif hurst_val < -0.1:  # ranging (raw < 0.45)
        tp_mult, sl_mult = 1.0, 1.0
        regime = "RANGING"
    else:
        tp_mult, sl_mult = 1.5, 1.0
        regime = "NEUTRAL"

    print(f"\n  ATR (14):          {atr:.12f} ({atr_pct:.3f}%)")
    print(f"  Hurst regime:      {regime} (raw={hurst_val:.3f})")

    # Step 4: Multi-bar scan (last 50 bars)
    print("\n" + "=" * 70)
    print("  📊 MULTI-BAR MODEL SCAN (last 50 bars)")
    print("=" * 70)
    
    scan_start = max(250, latest_idx - 50)
    scan_indices = list(range(scan_start, latest_idx + 1))
    valid_scan = []
    for si in scan_indices:
        if not np.any(np.isnan(F[si])):
            valid_scan.append(si)
    
    if valid_scan:
        F_scan = F[np.array(valid_scan)]
        dmat_scan = xgb.DMatrix(F_scan, feature_names=feature_names)
        p_up_scan = model.predict(dmat_scan)
        
        avg_p_up = float(np.mean(p_up_scan))
        recent_p_up = float(np.mean(p_up_scan[-10:]))  # last 10 bars
        trend_shift = recent_p_up - avg_p_up
        buy_signals = int(np.sum(p_up_scan >= 0.55))
        sell_signals = int(np.sum(p_up_scan <= 0.45))
        
        print(f"  Bars scanned:          {len(valid_scan)}")
        print(f"  Avg P(up):             {avg_p_up*100:.1f}%")
        print(f"  Recent P(up) (10 bar): {recent_p_up*100:.1f}%")
        print(f"  Trend shift:           {trend_shift*100:+.1f}%")
        print(f"  Buy signals (>55%):    {buy_signals}")
        print(f"  Sell signals (<45%):   {sell_signals}")

    # Step 5: Leverage analysis
    print("\n" + "=" * 70)
    print("  💰 LEVERAGE SCENARIO ANALYSIS ($1 POSITION)")
    print("=" * 70)

    for leverage in [100, 20]:
        position_size = 1.0 * leverage  # notional
        liq_distance_pct = 100.0 / leverage * 0.80  # ~80% of margin (maintenance margin)
        
        # For long:
        sl_price = current_price * (1 - sl_mult * atr / current_price)
        tp1_price = current_price * (1 + tp_mult * atr / current_price)
        tp2_price = current_price * (1 + tp_mult * 1.5 * atr / current_price)
        liq_price = current_price * (1 - liq_distance_pct / 100)
        
        sl_pct = (sl_price - current_price) / current_price * 100
        tp1_pct = (tp1_price - current_price) / current_price * 100
        tp2_pct = (tp2_price - current_price) / current_price * 100
        liq_pct = (liq_price - current_price) / current_price * 100
        
        # PnL calculations
        sl_loss_usd = abs(sl_pct) / 100 * position_size
        tp1_gain_usd = tp1_pct / 100 * position_size
        tp2_gain_usd = tp2_pct / 100 * position_size
        
        # EV calculation
        fee_pct = 0.05  # taker fee on perps
        fees = position_size * fee_pct / 100 * 2  # entry + exit
        
        ev_gross = p_up * tp1_gain_usd - p_down * sl_loss_usd
        ev_net = ev_gross - fees
        
        # Risk of liquidation
        # How many ATRs away is liquidation?
        atr_to_liq = liq_distance_pct / atr_pct if atr_pct > 0 else 0
        
        # Historical: what % of 60-bar windows see a move > liq_distance_pct against entry?
        # Check from our candle data
        max_drawdowns = []
        for i in range(max(0, len(cl)-200), len(cl)-60):
            entry = cl[i]
            future_low = lo[i+1:i+61].min()
            dd = (entry - future_low) / entry * 100
            max_drawdowns.append(dd)
        
        if max_drawdowns:
            liq_probability = np.mean(np.array(max_drawdowns) >= liq_distance_pct) * 100
        else:
            liq_probability = 50.0
        
        # Win probability adjusted for liquidation
        effective_p_win = p_up * (1 - liq_probability/100)
        
        print(f"\n  ┌─── {leverage}x LONG ────────────────────────────────────┐")
        print(f"  │  Margin:          $1.00")
        print(f"  │  Notional:        ${position_size:.0f}")
        print(f"  │  Current Price:   ${current_price:.10f}")
        print(f"  │")
        print(f"  │  Stop Loss:       ${sl_price:.10f}  ({sl_pct:.3f}%)")
        print(f"  │  Take Profit 1:   ${tp1_price:.10f}  (+{tp1_pct:.3f}%)")
        print(f"  │  Take Profit 2:   ${tp2_price:.10f}  (+{tp2_pct:.3f}%)")
        print(f"  │  Liquidation:     ${liq_price:.10f}  ({liq_pct:.3f}%)")
        print(f"  │")
        print(f"  │  ATRs to Liq:     {atr_to_liq:.1f} ATRs")
        print(f"  │  Liq Risk (60bar): {liq_probability:.1f}%")
        print(f"  │")
        print(f"  │  If TP1 hit:      +${tp1_gain_usd:.3f}")
        print(f"  │  If SL hit:       -${sl_loss_usd:.3f}")
        print(f"  │  Fees (round):    -${fees:.4f}")
        print(f"  │")
        print(f"  │  EV (gross):      ${ev_gross:+.4f}")
        print(f"  │  EV (net):        ${ev_net:+.4f}")
        print(f"  │  Effective P(win): {effective_p_win*100:.1f}%")
        
        # Verdict
        if ev_net > 0 and liq_probability < 30:
            verdict = "✅ PLAYABLE"
        elif ev_net > 0 and liq_probability < 50:
            verdict = "⚠️  MARGINAL (high liq risk)"
        elif ev_net > 0:
            verdict = "❌ AVOID (liq risk too high)"
        else:
            verdict = "❌ NEGATIVE EV"
        
        print(f"  │  VERDICT:         {verdict}")
        print(f"  └──────────────────────────────────────────────────┘")

    # Step 6: Final Oracle Verdict
    print("\n" + "=" * 70)
    print("  🔮 ORACLE FINAL VERDICT")
    print("=" * 70)
    
    # Composite score
    signals = {
        "model_p_up": p_up,
        "funding_bullish": 1 if current_funding < 0 else (0.5 if current_funding < 0.0005 else 0),
        "ob_bullish": 1 if imbalance > 1.1 else (0.5 if imbalance > 0.95 else 0),
        "trend_momentum": 1 if (valid_scan and trend_shift > 0.02) else (0.5 if (valid_scan and trend_shift > -0.02) else 0),
    }
    
    composite = (
        signals["model_p_up"] * 0.50 +
        signals["funding_bullish"] * 0.15 +
        signals["ob_bullish"] * 0.15 +
        signals["trend_momentum"] * 0.20
    )
    
    print(f"\n  Model P(up):          {signals['model_p_up']*100:.1f}%  (weight: 50%)")
    print(f"  Funding signal:       {signals['funding_bullish']*100:.0f}%    (weight: 15%)")
    print(f"  Order book signal:    {signals['ob_bullish']*100:.0f}%    (weight: 15%)")
    print(f"  Trend momentum:       {signals['trend_momentum']*100:.0f}%    (weight: 20%)")
    print(f"\n  COMPOSITE SCORE:      {composite*100:.1f}%")
    
    if composite > 0.60:
        print(f"\n  DIRECTION: 🟢 LONG")
    elif composite < 0.40:
        print(f"\n  DIRECTION: 🔴 SHORT")
    else:
        print(f"\n  DIRECTION: 🟡 SKIP / WAIT")
    
    # Recommendation
    print(f"\n  {'─'*50}")
    print(f"  RECOMMENDATION FOR $1 FUN TRADE:")
    
    # Compare 100x vs 20x
    print(f"\n  100x: Maximum payout potential but liquidation is")
    print(f"        only ~{100/100*0.80:.2f}% away — a tiny wick kills you.")
    print(f"        Even with favorable direction, liquidation")
    print(f"        probability is very high on meme coins.")
    print(f"")
    print(f"  20x:  Liquidation at ~{100/20*0.80:.1f}% away — much more")
    print(f"        survivable. You catch the same direction")
    print(f"        with real chance of hitting TP.")

    # Save results
    results = {
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "symbol": symbol,
        "current_price": current_price,
        "price_change_24h_pct": price_change_24h,
        "volume_24h_usdt": volume_24h,
        "model_p_up": round(p_up, 4),
        "model_p_down": round(p_down, 4),
        "signal": signal,
        "regime": regime,
        "atr": float(atr),
        "atr_pct": round(atr_pct, 4),
        "hurst": round(float(hurst_val), 4),
        "funding_rate": round(current_funding, 6) if funding_data else None,
        "ob_imbalance": round(imbalance, 4),
        "composite_score": round(composite, 4),
        "feature_values": {k: round(float(v), 4) for k, v in sorted_feats[:10]},
    }
    
    out_path = os.path.join(os.path.dirname(__file__), f"{symbol.lower()}_oracle_result.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved -> {out_path}")


if __name__ == "__main__":
    main()
