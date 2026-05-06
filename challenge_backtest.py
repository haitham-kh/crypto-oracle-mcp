"""
challenge_backtest.py - True Out-of-Sample Model Challenge
==========================================================
Downloads 6 months of 1-min data for a coin NOT in the training set,
runs the XGBoost model on it, and compares predictions to reality.

Flow:
  Months 1-2: warmup/lookback for feature computation
  Month 3:    PREDICT -> compare to actual outcomes
  Months 4-5: warmup/lookback
  Month 6:    PREDICT -> compare to actual outcomes

Usage:
  python challenge_backtest.py LUNCUSDT
  python challenge_backtest.py APTUSDT
"""
from __future__ import annotations
import os, sys, time, json, datetime, math
import numpy as np
import requests

sys.path.insert(0, os.path.dirname(__file__))

BINANCE_BASE = "https://api.binance.com"
DATA_DIR = os.path.join(os.path.dirname(__file__), "challenge_data")


# ── Download ─────────────────────────────────────────────────────────────

def download_klines(symbol: str, start_ms: int, end_ms: int) -> list:
    """Download 1m klines from Binance REST API with pagination."""
    all_candles = []
    current = start_ms
    total_expected = (end_ms - start_ms) // 60000
    print(f"  Downloading {symbol} 1m candles ({total_expected:,} expected)...")

    while current < end_ms:
        url = f"{BINANCE_BASE}/api/v3/klines"
        params = {"symbol": symbol, "interval": "1m", "startTime": current,
                  "endTime": end_ms, "limit": 1500}
        try:
            r = requests.get(url, params=params, timeout=15)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            print(f"    Retry after error: {e}")
            time.sleep(2)
            continue

        if not data:
            break

        for k in data:
            all_candles.append({
                "timestamp": int(k[0]),
                "open": float(k[1]), "high": float(k[2]),
                "low": float(k[3]), "close": float(k[4]),
                "volume": float(k[5]),
                "taker_buy_volume": float(k[9]),
            })

        current = int(data[-1][0]) + 60000  # next minute after last candle
        if len(all_candles) % 50000 < 1500:
            print(f"    {len(all_candles):,} candles...", flush=True)
        time.sleep(0.1)  # rate limit

    print(f"  Done: {len(all_candles):,} candles downloaded")
    return all_candles


def get_month_boundaries(n_months=6):
    """Return start/end timestamps for the last N complete months."""
    now = datetime.datetime.utcnow()
    # Start from the 1st of the current month and go back n_months
    end_month = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    months = []
    for i in range(n_months, 0, -1):
        # Go back i months from end_month
        y = end_month.year
        m = end_month.month - i
        while m <= 0:
            m += 12; y -= 1
        start = datetime.datetime(y, m, 1)
        # Next month
        nm = m + 1; ny = y
        if nm > 12: nm = 1; ny += 1
        end = datetime.datetime(ny, nm, 1)
        months.append((start, end))
    return months


# ── Feature builder (reuse v2 logic) ─────────────────────────────────────

def build_arrays(candles):
    """Convert candle list to numpy arrays."""
    ts = np.array([c["timestamp"] for c in candles], dtype=np.float64)
    o = np.array([c["open"] for c in candles], dtype=np.float64)
    h = np.array([c["high"] for c in candles], dtype=np.float64)
    l = np.array([c["low"] for c in candles], dtype=np.float64)
    c = np.array([c["close"] for c in candles], dtype=np.float64)
    v = np.array([x["volume"] for x in candles], dtype=np.float64)
    tbv = np.array([x.get("taker_buy_volume", 0) for x in candles], dtype=np.float64)
    # OFI proxy: taker_buy - taker_sell
    tsv = v - tbv
    ofi = tbv - tsv  # = 2*tbv - v
    return ts, o, h, l, c, v, ofi


def run_predictions(candles_study, candles_predict, btc_study, btc_predict,
                    model, feature_names, label=""):
    """Run XGBoost model on predict window, compare to actual outcomes."""
    import xgboost as xgb
    from train_ev_model_v2 import build_features_v2, _atr

    # Combine study + predict for feature warmup
    all_candles = candles_study + candles_predict
    ts, op, hi, lo, cl, vol, ofi = build_arrays(all_candles)
    n_total = len(cl)
    n_study = len(candles_study)
    n_predict = len(candles_predict)

    # BTC alignment
    btc_all = btc_study + btc_predict
    _, _, _, _, btc_cl, _, _ = build_arrays(btc_all)
    btc_aligned = np.interp(ts, np.array([c["timestamp"] for c in btc_all], dtype=np.float64), btc_cl)

    # Build features
    F, atr_arr = build_features_v2(cl, hi, lo, vol, ofi, ts, btc_aligned)

    # Only predict on the predict window (after study period)
    predict_start = n_study
    predict_end = n_total

    # Sample every 10 bars, skip first 250 of predict window if possible
    warmup_in_predict = min(250, n_predict // 4)
    sample_start = predict_start + warmup_in_predict
    forward_bars = 60

    idx = np.arange(sample_start, predict_end - forward_bars, 10)
    if len(idx) == 0:
        print(f"  {label}: no valid prediction samples")
        return None

    F_samp = F[idx]
    valid = ~np.any(np.isnan(F_samp), axis=1)
    idx_v = idx[valid]
    F_v = F_samp[valid]

    if len(idx_v) == 0:
        print(f"  {label}: all samples have NaN features")
        return None

    # Run XGBoost predictions
    dmat = xgb.DMatrix(F_v, feature_names=feature_names)
    p_up = model.predict(dmat)

    # Compute actual outcomes (triple-barrier, regime-conditioned)
    hurst_arr = F[:, 10]  # hurst feature
    actual = np.full(len(idx_v), np.nan)
    for k, i in enumerate(idx_v):
        atr = atr_arr[i]
        if np.isnan(atr) or atr <= 0: continue
        entry = cl[i]
        end = min(i + forward_bars, n_total - 1)
        if end <= i: continue

        h_val = hurst_arr[i] if not np.isnan(hurst_arr[i]) else 0.5
        if h_val > 0.55: tp_m, sl_m = 2.0, 1.0
        elif h_val < 0.45: tp_m, sl_m = 1.0, 1.0
        else: tp_m, sl_m = 1.5, 1.0

        tp = entry + tp_m * atr
        sl = entry - sl_m * atr
        fh = hi[i+1:end+1]
        fl = lo[i+1:end+1]
        tp_hit = np.where(fh >= tp)[0]
        sl_hit = np.where(fl <= sl)[0]
        tp_t = tp_hit[0] if len(tp_hit) > 0 else np.inf
        sl_t = sl_hit[0] if len(sl_hit) > 0 else np.inf

        if tp_t == np.inf and sl_t == np.inf:
            actual[k] = 1 if cl[end] > entry else 0
        elif tp_t <= sl_t:
            actual[k] = 1
        else:
            actual[k] = 0

    # Filter out NaN actuals
    mask = ~np.isnan(actual)
    p_up = p_up[mask]
    actual = actual[mask]
    idx_final = idx_v[mask]

    if len(actual) == 0:
        print(f"  {label}: no labelled outcomes")
        return None

    # ── Compute all metrics ──────────────────────────────────────────────
    preds_binary = (p_up >= 0.5).astype(int)
    accuracy = float(np.mean(preds_binary == actual))
    actual_wr = float(actual.mean())

    from scipy.stats import spearmanr
    ic, ic_pval = spearmanr(p_up, actual)

    # Confidence buckets
    high_conf_buy = p_up >= 0.55
    high_conf_sell = p_up <= 0.45
    neutral = (~high_conf_buy) & (~high_conf_sell)

    results = {
        "label": label,
        "total_predictions": len(actual),
        "actual_win_rate": round(actual_wr * 100, 1),
        "model_accuracy": round(accuracy * 100, 1),
        "information_coefficient": round(float(ic), 4),
        "ic_p_value": round(float(ic_pval), 6),
        "ic_significant": ic_pval < 0.05,
    }

    # High-confidence BUY signals
    if high_conf_buy.sum() > 0:
        buy_actual_wr = float(actual[high_conf_buy].mean())
        results["high_conf_buy_count"] = int(high_conf_buy.sum())
        results["high_conf_buy_win_rate"] = round(buy_actual_wr * 100, 1)
        results["high_conf_buy_edge"] = round((buy_actual_wr - actual_wr) * 100, 1)
    else:
        results["high_conf_buy_count"] = 0

    # High-confidence SELL/AVOID signals
    if high_conf_sell.sum() > 0:
        sell_actual_wr = float(actual[high_conf_sell].mean())
        results["high_conf_avoid_count"] = int(high_conf_sell.sum())
        results["high_conf_avoid_win_rate"] = round(sell_actual_wr * 100, 1)
        results["high_conf_avoid_edge"] = round((actual_wr - sell_actual_wr) * 100, 1)
    else:
        results["high_conf_avoid_count"] = 0

    # Simulated PnL: follow high-conf buys, skip low-conf
    pnl_per_trade = []
    for k in range(len(idx_final)):
        if not high_conf_buy[k]: continue
        i = idx_final[k]
        atr = atr_arr[i]
        if np.isnan(atr) or atr <= 0: continue
        entry = cl[i]
        h_val = hurst_arr[i] if not np.isnan(hurst_arr[i]) else 0.5
        if h_val > 0.55: tp_m = 2.0
        elif h_val < 0.45: tp_m = 1.0
        else: tp_m = 1.5

        if actual[k] == 1:
            pnl_per_trade.append(tp_m * atr / entry * 100)  # win in %
        else:
            pnl_per_trade.append(-1.0 * atr / entry * 100)  # loss in %

    if pnl_per_trade:
        pnl_arr = np.array(pnl_per_trade)
        results["sim_trades"] = len(pnl_arr)
        results["sim_total_pnl_pct"] = round(float(pnl_arr.sum()), 2)
        results["sim_avg_pnl_pct"] = round(float(pnl_arr.mean()), 4)
        results["sim_win_rate_pct"] = round(float((pnl_arr > 0).mean() * 100), 1)
        results["sim_profit_factor"] = round(
            float(pnl_arr[pnl_arr > 0].sum() / max(abs(pnl_arr[pnl_arr < 0].sum()), 0.001)), 2)
        results["sim_max_win_pct"] = round(float(pnl_arr.max()), 3)
        results["sim_max_loss_pct"] = round(float(pnl_arr.min()), 3)

    return results


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    symbol = sys.argv[1] if len(sys.argv) > 1 else "LUNCUSDT"
    print("=" * 65)
    print(f"  MODEL CHALLENGE: {symbol}")
    print(f"  True out-of-sample backtest (coin NOT in training data)")
    print("=" * 65)

    # Check training coins
    meta_path = os.path.join(os.path.dirname(__file__), "data", "ev_model_v2_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        if symbol in meta.get("training_coins", []):
            print(f"\n  WARNING: {symbol} WAS in the training set!")
            print(f"  This is still a valid time-window test but not truly unseen.\n")
        else:
            print(f"\n  CONFIRMED: {symbol} was NOT in the training data.")
            print(f"  This is a true out-of-sample test.\n")

    # Get 6 month boundaries
    months = get_month_boundaries(6)
    print("  Test periods:")
    for i, (s, e) in enumerate(months):
        role = "STUDY" if i in [0,1,3,4] else "PREDICT"
        print(f"    Month {i+1}: {s.strftime('%Y-%m-%d')} to {e.strftime('%Y-%m-%d')} [{role}]")

    # Download data
    overall_start = int(months[0][0].timestamp() * 1000)
    overall_end = int(months[-1][1].timestamp() * 1000)

    os.makedirs(DATA_DIR, exist_ok=True)
    cache_file = os.path.join(DATA_DIR, f"{symbol}_challenge.json")
    btc_cache = os.path.join(DATA_DIR, "BTCUSDT_challenge.json")

    print(f"\n--- Downloading {symbol} ---")
    if os.path.exists(cache_file):
        print(f"  Using cached data from {cache_file}")
        with open(cache_file) as f:
            candles = json.load(f)
    else:
        candles = download_klines(symbol, overall_start, overall_end)
        with open(cache_file, "w") as f:
            json.dump(candles, f)

    print(f"\n--- Downloading BTCUSDT (for cross-market feature) ---")
    if os.path.exists(btc_cache):
        print(f"  Using cached data from {btc_cache}")
        with open(btc_cache) as f:
            btc_candles = json.load(f)
    else:
        btc_candles = download_klines("BTCUSDT", overall_start, overall_end)
        with open(btc_cache, "w") as f:
            json.dump(btc_candles, f)

    # Split candles by month
    def split_by_month(clist, months):
        result = []
        for s, e in months:
            s_ms = int(s.timestamp() * 1000)
            e_ms = int(e.timestamp() * 1000)
            month_candles = [c for c in clist if s_ms <= c["timestamp"] < e_ms]
            result.append(month_candles)
        return result

    coin_months = split_by_month(candles, months)
    btc_months = split_by_month(btc_candles, months)

    print(f"\n  Candles per month:")
    for i, mc in enumerate(coin_months):
        print(f"    Month {i+1}: {len(mc):>8,} candles  |  BTC: {len(btc_months[i]):>8,}")

    # Load XGBoost model
    print("\n--- Loading XGBoost model ---")
    import xgboost as xgb
    model_path = os.path.join(os.path.dirname(__file__), "data", "ev_model_xgb.json")
    model = xgb.Booster()
    model.load_model(model_path)
    feature_names = meta.get("feature_names", [])
    print(f"  Model loaded: {len(feature_names)} features, {meta.get('total_samples', 0):,} training samples")

    # ── Challenge Round 1: Study months 1-2, Predict month 3 ─────────
    print("\n" + "=" * 65)
    print("  ROUND 1: Study Nov-Dec 2025 -> Predict Jan 2026")
    print("=" * 65)

    study1 = coin_months[0] + coin_months[1]
    predict1 = coin_months[2]
    btc_study1 = btc_months[0] + btc_months[1]
    btc_predict1 = btc_months[2]

    r1 = run_predictions(study1, predict1, btc_study1, btc_predict1,
                         model, feature_names, label="Round 1 (Month 3)")

    # ── Challenge Round 2: Study months 4-5, Predict month 6 ─────────
    print("\n" + "=" * 65)
    print("  ROUND 2: Study Feb-Mar 2026 -> Predict Apr 2026")
    print("=" * 65)

    study2 = coin_months[3] + coin_months[4]
    predict2 = coin_months[5]
    btc_study2 = btc_months[3] + btc_months[4]
    btc_predict2 = btc_months[5]

    r2 = run_predictions(study2, predict2, btc_study2, btc_predict2,
                         model, feature_names, label="Round 2 (Month 6)")

    # ── Final Report ─────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  CHALLENGE RESULTS")
    print("=" * 65)

    for r in [r1, r2]:
        if r is None:
            print("  [Round failed - insufficient data]")
            continue
        print(f"\n  --- {r['label']} ---")
        print(f"  Total predictions:          {r['total_predictions']:,}")
        print(f"  Actual win rate (baseline): {r['actual_win_rate']}%")
        print(f"  Model accuracy:             {r['model_accuracy']}%")
        print(f"  Information Coefficient:    {r['information_coefficient']}")
        print(f"  IC statistically significant: {'YES' if r['ic_significant'] else 'NO'} (p={r['ic_p_value']})")

        if r.get("high_conf_buy_count", 0) > 0:
            print(f"\n  High-confidence BUY signals: {r['high_conf_buy_count']}")
            print(f"    Win rate when model says BUY:  {r['high_conf_buy_win_rate']}%")
            print(f"    Edge over random:              {r['high_conf_buy_edge']:+.1f}%")

        if r.get("high_conf_avoid_count", 0) > 0:
            print(f"\n  High-confidence AVOID signals: {r['high_conf_avoid_count']}")
            print(f"    Win rate when model says AVOID: {r['high_conf_avoid_win_rate']}%")
            print(f"    Edge (avoided losses):          {r['high_conf_avoid_edge']:+.1f}%")

        if r.get("sim_trades"):
            print(f"\n  Simulated Trading (high-conf buys only):")
            print(f"    Trades taken:     {r['sim_trades']}")
            print(f"    Total PnL:        {r['sim_total_pnl_pct']:+.2f}%")
            print(f"    Avg PnL/trade:    {r['sim_avg_pnl_pct']:+.4f}%")
            print(f"    Win rate:         {r['sim_win_rate_pct']}%")
            print(f"    Profit factor:    {r['sim_profit_factor']}")
            print(f"    Best trade:       {r['sim_max_win_pct']:+.3f}%")
            print(f"    Worst trade:      {r['sim_max_loss_pct']:+.3f}%")

    # Combined IC
    if r1 and r2:
        avg_ic = (r1["information_coefficient"] + r2["information_coefficient"]) / 2
        print(f"\n  Combined average IC: {avg_ic:.4f}")
        if avg_ic > 0.05:
            print("  VERDICT: Model has REAL predictive power on unseen coin")
        elif avg_ic > 0.02:
            print("  VERDICT: Model has WEAK but present signal on unseen coin")
        else:
            print("  VERDICT: Model has NO meaningful signal on this coin")

    # Save results
    results_path = os.path.join(DATA_DIR, f"{symbol}_challenge_results.json")
    with open(results_path, "w") as f:
        json.dump({"round1": r1, "round2": r2}, f, indent=2)
    print(f"\n  Results saved -> {results_path}")


if __name__ == "__main__":
    main()
