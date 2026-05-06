"""
multi_coin_blind_test.py - Multi-Coin Blind Robustness Test
============================================================
Downloads 6 months of 1-min data for 3 coins NOT in the training set,
runs the XGBoost model, and compares predictions to reality.

Coins tested (all unseen during training):
  1. DOTUSDT  (Polkadot - large cap infrastructure)
  2. APTUSDT  (Aptos - newer L1 chain)
  3. NEARUSDT (NEAR Protocol - mid cap L1)

Flow per coin:
  Study months 1-2 -> Predict month 3
  Study months 4-5 -> Predict month 6

Usage:
  python multi_coin_blind_test.py
"""
from __future__ import annotations
import os, sys, time, json, datetime
import numpy as np
import requests

sys.path.insert(0, os.path.dirname(__file__))

BINANCE_BASE = "https://api.binance.com"
DATA_DIR = os.path.join(os.path.dirname(__file__), "challenge_data")
TEST_COINS = ["DOTUSDT", "APTUSDT", "NEARUSDT"]


def download_klines(symbol, start_ms, end_ms):
    all_candles = []
    current = start_ms
    total_expected = (end_ms - start_ms) // 60000
    print(f"  Downloading {symbol} ({total_expected:,} candles)...", flush=True)
    while current < end_ms:
        params = {"symbol": symbol, "interval": "1m", "startTime": current,
                  "endTime": end_ms, "limit": 1500}
        try:
            r = requests.get(f"{BINANCE_BASE}/api/v3/klines", params=params, timeout=15)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            print(f"    Retry: {e}")
            time.sleep(2)
            continue
        if not data:
            break
        for k in data:
            all_candles.append({
                "timestamp": int(k[0]), "open": float(k[1]), "high": float(k[2]),
                "low": float(k[3]), "close": float(k[4]), "volume": float(k[5]),
                "taker_buy_volume": float(k[9]),
            })
        current = int(data[-1][0]) + 60000
        if len(all_candles) % 50000 < 1500:
            print(f"    {len(all_candles):,}...", flush=True)
        time.sleep(0.1)
    print(f"    Done: {len(all_candles):,} candles")
    return all_candles


def get_month_boundaries(n_months=6):
    now = datetime.datetime.utcnow()
    end_month = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    months = []
    for i in range(n_months, 0, -1):
        y, m = end_month.year, end_month.month - i
        while m <= 0:
            m += 12; y -= 1
        start = datetime.datetime(y, m, 1)
        nm, ny = m + 1, y
        if nm > 12:
            nm, ny = 1, y + 1
        end = datetime.datetime(ny, nm, 1)
        months.append((start, end))
    return months


def build_arrays(candles):
    ts = np.array([c["timestamp"] for c in candles], dtype=np.float64)
    h = np.array([c["high"] for c in candles], dtype=np.float64)
    l = np.array([c["low"] for c in candles], dtype=np.float64)
    c = np.array([c["close"] for c in candles], dtype=np.float64)
    v = np.array([x["volume"] for x in candles], dtype=np.float64)
    tbv = np.array([x.get("taker_buy_volume", 0) for x in candles], dtype=np.float64)
    ofi = 2 * tbv - v
    return ts, h, l, c, v, ofi


def run_predictions(candles_study, candles_predict, btc_study, btc_predict,
                    model, feature_names, label="", use_v3=False):
    import xgboost as xgb
    from train_ev_model_v2 import build_features_v2

    all_candles = candles_study + candles_predict
    ts, hi, lo, cl, vol, ofi = build_arrays(all_candles)
    n_total = len(cl)
    n_study = len(candles_study)
    n_predict = len(candles_predict)

    btc_all = btc_study + btc_predict
    btc_ts_arr = np.array([c["timestamp"] for c in btc_all], dtype=np.float64)
    btc_cl_arr = np.array([c["close"] for c in btc_all], dtype=np.float64)
    btc_aligned = np.interp(ts, btc_ts_arr, btc_cl_arr)

    F_v2, atr_arr = build_features_v2(cl, hi, lo, vol, ofi, ts, btc_aligned)

    # Convert to v3 if needed
    if use_v3:
        from features_v3 import build_v3_from_v2
        F = build_v3_from_v2(F_v2, cl, hi, lo, atr_arr)
    else:
        F = F_v2

    warmup_in_predict = min(250, n_predict // 4)
    sample_start = n_study + warmup_in_predict
    forward_bars = 60
    idx = np.arange(sample_start, n_total - forward_bars, 10)
    if len(idx) == 0:
        return None

    F_samp = F[idx]
    valid = ~np.any(np.isnan(F_samp), axis=1)
    idx_v = idx[valid]
    F_v = F_samp[valid]
    if len(idx_v) == 0:
        return None

    dmat = xgb.DMatrix(F_v, feature_names=feature_names)
    p_up = model.predict(dmat)

    hurst_arr = F_v2[:, 10]  # hurst is always at v2 index 10
    actual = np.full(len(idx_v), np.nan)
    for k, i in enumerate(idx_v):
        atr = atr_arr[i]
        if np.isnan(atr) or atr <= 0:
            continue
        entry = cl[i]
        end = min(i + forward_bars, n_total - 1)
        if end <= i:
            continue
        h_val = hurst_arr[i] if not np.isnan(hurst_arr[i]) else 0.5
        if h_val > 0.55:
            tp_m, sl_m = 2.0, 1.0
        elif h_val < 0.45:
            tp_m, sl_m = 1.0, 1.0
        else:
            tp_m, sl_m = 1.5, 1.0
        tp = entry + tp_m * atr
        sl = entry - sl_m * atr
        fh, fl = hi[i+1:end+1], lo[i+1:end+1]
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

    mask = ~np.isnan(actual)
    p_up, actual, idx_final = p_up[mask], actual[mask], idx_v[mask]
    if len(actual) == 0:
        return None

    preds_binary = (p_up >= 0.5).astype(int)
    accuracy = float(np.mean(preds_binary == actual))
    actual_wr = float(actual.mean())

    from scipy.stats import spearmanr
    ic, ic_pval = spearmanr(p_up, actual)

    high_conf_buy = p_up >= 0.55
    high_conf_sell = p_up <= 0.45

    r = {
        "label": label,
        "total_predictions": int(len(actual)),
        "actual_win_rate": round(actual_wr * 100, 1),
        "model_accuracy": round(accuracy * 100, 1),
        "ic": round(float(ic), 4),
        "ic_pval": round(float(ic_pval), 6),
        "ic_significant": bool(ic_pval < 0.05),
    }

    if high_conf_buy.sum() > 0:
        buy_wr = float(actual[high_conf_buy].mean())
        r["buy_signals"] = int(high_conf_buy.sum())
        r["buy_win_rate"] = round(buy_wr * 100, 1)
        r["buy_edge"] = round((buy_wr - actual_wr) * 100, 1)

    if high_conf_sell.sum() > 0:
        sell_wr = float(actual[high_conf_sell].mean())
        r["avoid_signals"] = int(high_conf_sell.sum())
        r["avoid_win_rate"] = round(sell_wr * 100, 1)
        r["avoid_edge"] = round((actual_wr - sell_wr) * 100, 1)

    # Simulated PnL
    pnl = []
    for k in range(len(idx_final)):
        if not high_conf_buy[k]:
            continue
        i = idx_final[k]
        atr = atr_arr[i]
        if np.isnan(atr) or atr <= 0:
            continue
        entry = cl[i]
        h_val = hurst_arr[i] if not np.isnan(hurst_arr[i]) else 0.5
        tp_m = 2.0 if h_val > 0.55 else (1.0 if h_val < 0.45 else 1.5)
        if actual[k] == 1:
            pnl.append(float(tp_m * atr / entry * 100))
        else:
            pnl.append(float(-1.0 * atr / entry * 100))

    if pnl:
        pa = np.array(pnl)
        wins = pa[pa > 0]
        losses = pa[pa < 0]
        r["sim_trades"] = len(pa)
        r["sim_pnl_pct"] = round(float(pa.sum()), 2)
        r["sim_avg_pnl"] = round(float(pa.mean()), 4)
        r["sim_win_rate"] = round(float((pa > 0).mean() * 100), 1)
        r["sim_profit_factor"] = round(float(wins.sum() / max(abs(losses.sum()), 0.001)), 2)

    return r


def main():
    print("=" * 65)
    print("  MULTI-COIN BLIND ROBUSTNESS TEST")
    print("  3 unseen coins x 6 months x 2 prediction rounds")
    print("=" * 65)

    # Auto-detect best model: v3 > v2
    use_v3 = False
    v3_meta = os.path.join(os.path.dirname(__file__), "data", "ev_model_v3_meta.json")
    v3_model = os.path.join(os.path.dirname(__file__), "data", "ev_model_xgb_v3.json")
    v2_meta = os.path.join(os.path.dirname(__file__), "data", "ev_model_v2_meta.json")
    v2_model = os.path.join(os.path.dirname(__file__), "data", "ev_model_xgb.json")

    if os.path.exists(v3_meta) and os.path.exists(v3_model):
        meta_path, model_path = v3_meta, v3_model
        use_v3 = True
    else:
        meta_path, model_path = v2_meta, v2_model

    with open(meta_path) as f:
        meta = json.load(f)
    training_coins = meta.get("training_coins", [])
    model_ver = meta.get("model_type", "v2")

    print(f"\n  Model version: {model_ver} ({'v3 Phase A' if use_v3 else 'v2'})")
    print(f"  Training coins ({len(training_coins)}): {', '.join(training_coins)}")
    print(f"  Test coins (UNSEEN):  {', '.join(TEST_COINS)}")
    for tc in TEST_COINS:
        status = "NOT in training" if tc not in training_coins else "WARNING: IN TRAINING"
        print(f"    {tc}: {status}")

    months = get_month_boundaries(6)
    print(f"\n  Period: {months[0][0].strftime('%Y-%m-%d')} to {months[-1][1].strftime('%Y-%m-%d')}")
    for i, (s, e) in enumerate(months):
        role = "STUDY" if i in [0, 1, 3, 4] else ">>PREDICT<<"
        print(f"    Month {i+1}: {s.strftime('%b %Y')} [{role}]")

    overall_start = int(months[0][0].timestamp() * 1000)
    overall_end = int(months[-1][1].timestamp() * 1000)
    os.makedirs(DATA_DIR, exist_ok=True)

    # Download BTC first (shared)
    btc_cache = os.path.join(DATA_DIR, "BTCUSDT_challenge.json")
    if os.path.exists(btc_cache):
        print(f"\n  BTC data cached, loading...")
        with open(btc_cache) as f:
            btc_candles = json.load(f)
        print(f"    {len(btc_candles):,} candles")
    else:
        print(f"\n--- Downloading BTC reference ---")
        btc_candles = download_klines("BTCUSDT", overall_start, overall_end)
        with open(btc_cache, "w") as f:
            json.dump(btc_candles, f)

    # Download test coins
    coin_data = {}
    for sym in TEST_COINS:
        cache = os.path.join(DATA_DIR, f"{sym}_challenge.json")
        if os.path.exists(cache):
            print(f"\n  {sym} cached, loading...")
            with open(cache) as f:
                coin_data[sym] = json.load(f)
            print(f"    {len(coin_data[sym]):,} candles")
        else:
            print(f"\n--- Downloading {sym} ---")
            coin_data[sym] = download_klines(sym, overall_start, overall_end)
            with open(cache, "w") as f:
                json.dump(coin_data[sym], f)

    # Load model
    print("\n--- Loading model ---")
    import xgboost as xgb
    model = xgb.Booster()
    model.load_model(model_path)
    feature_names = meta.get("feature_names", [])
    print(f"  Loaded: {meta.get('total_samples',0):,} samples, {len(feature_names)} features, v3={use_v3}")

    # Split helper
    def split_by_month(clist):
        result = []
        for s, e in months:
            s_ms, e_ms = int(s.timestamp() * 1000), int(e.timestamp() * 1000)
            result.append([c for c in clist if s_ms <= c["timestamp"] < e_ms])
        return result

    btc_months = split_by_month(btc_candles)

    # Run all coins
    all_results = {}
    for sym in TEST_COINS:
        print(f"\n{'='*65}")
        print(f"  TESTING: {sym}")
        print(f"{'='*65}")

        cm = split_by_month(coin_data[sym])
        for i, mc in enumerate(cm):
            print(f"    Month {i+1}: {len(mc):,} candles")

        # Round 1
        r1 = run_predictions(
            cm[0] + cm[1], cm[2], btc_months[0] + btc_months[1], btc_months[2],
            model, feature_names, f"{sym} Round 1 (Month 3)", use_v3=use_v3)

        # Round 2
        r2 = run_predictions(
            cm[3] + cm[4], cm[5], btc_months[3] + btc_months[4], btc_months[5],
            model, feature_names, f"{sym} Round 2 (Month 6)", use_v3=use_v3)

        all_results[sym] = {"round1": r1, "round2": r2}

    # ── FINAL REPORT ─────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  MULTI-COIN BLIND TEST — FINAL REPORT")
    print("=" * 65)

    header = f"{'Coin':<12} {'Round':<8} {'IC':>8} {'Sig?':>5} {'Accuracy':>9} {'BuyWR':>7} {'Edge':>7} {'PF':>6} {'PnL%':>8}"
    print(f"\n  {header}")
    print(f"  {'-' * len(header)}")

    all_ics = []
    all_pnls = []
    all_pfs = []

    for sym in TEST_COINS:
        for rkey, rlabel in [("round1", "R1"), ("round2", "R2")]:
            r = all_results[sym].get(rkey)
            if r is None:
                print(f"  {sym:<12} {rlabel:<8} {'FAILED':>8}")
                continue

            ic_str = f"{r['ic']:.4f}"
            sig = "YES" if r.get("ic_significant") else "NO"
            acc = f"{r['model_accuracy']}%"
            bwr = f"{r.get('buy_win_rate', 'N/A')}%"
            edge = f"+{r.get('buy_edge', 0):.1f}%"
            pf = f"{r.get('sim_profit_factor', 'N/A')}"
            pnl = f"{r.get('sim_pnl_pct', 0):+.2f}%"

            print(f"  {sym:<12} {rlabel:<8} {ic_str:>8} {sig:>5} {acc:>9} {bwr:>7} {edge:>7} {pf:>6} {pnl:>8}")

            all_ics.append(r["ic"])
            if r.get("sim_pnl_pct") is not None:
                all_pnls.append(r["sim_pnl_pct"])
            if r.get("sim_profit_factor") is not None:
                all_pfs.append(r["sim_profit_factor"])

    # Summary
    print(f"\n  {'='*50}")
    print(f"  SUMMARY ACROSS ALL COINS & ROUNDS")
    print(f"  {'='*50}")
    if all_ics:
        avg_ic = np.mean(all_ics)
        min_ic = np.min(all_ics)
        max_ic = np.max(all_ics)
        print(f"  Average IC:        {avg_ic:.4f}  (min={min_ic:.4f}, max={max_ic:.4f})")
        print(f"  All ICs positive:  {'YES' if min_ic > 0 else 'NO'}")
    if all_pnls:
        print(f"  Total sim PnL:     {sum(all_pnls):+.2f}%  across {len(all_pnls)} rounds")
        print(f"  Avg PnL per round: {np.mean(all_pnls):+.2f}%")
        print(f"  Profitable rounds: {sum(1 for p in all_pnls if p > 0)}/{len(all_pnls)}")
    if all_pfs:
        print(f"  Avg profit factor: {np.mean(all_pfs):.2f}")

    # Verdict
    print(f"\n  VERDICT:")
    if all_ics and np.mean(all_ics) > 0.05:
        if min(all_ics) > 0:
            print(f"  >> MODEL PASSES MULTI-COIN BLIND TEST <<")
            print(f"  >> All ICs positive, average IC = {avg_ic:.4f} (above 0.05 threshold) <<")
        else:
            print(f"  >> MIXED RESULTS: Some rounds show signal, others don't <<")
    else:
        print(f"  >> MODEL FAILS BLIND TEST: IC too low across unseen coins <<")

    # Save
    tag = "v3" if use_v3 else "v2"
    results_path = os.path.join(DATA_DIR, f"multi_coin_blind_results_{tag}.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Results saved: {results_path}")


if __name__ == "__main__":
    main()
