"""
blind_backtest.py - Realistic futures backtest on a randomly chosen altcoin.
Fully blind, high slippage, normal fees, and real liquidation rules.
"""
from __future__ import annotations
import os, sys, json, datetime, time, pickle, random, gc
import numpy as np
import polars as pl
import xgboost as xgb
import requests

sys.path.insert(0, os.path.dirname(__file__))

from trading_config import (
    CHECK_EVERY_MINS, HORIZONS,
    horizon_atr_scale, regime_tp_mult, SL_MULT,
    MIN_TP_TO_COST_RATIO, MIN_EV_PCT,
    RISK_PER_TRADE_PCT, MAX_POSITION_PCT, KELLY_FRACTION,
    RISK_AVERSION_TARGET, MAX_OPEN_POSITIONS_TOTAL,
    DAILY_LOSS_HALT_PCT, WEEKLY_LOSS_HALT_PCT,
    HALT_REGIMES, MAX_SL_PCT, ASSET_MAX_POSITION_SCALE,
    HORIZON_REGIME_EV_SCALE, BTC_QUIET_4H_THRESHOLD, BTC_QUIET_EV_MULTIPLIER,
    FEE_PCT_PER_SIDE, SLIPPAGE_PCT_PER_SIDE, ROUND_TRIP_COST
)
from features_v5 import FEATURE_NAMES_V5, build_v5_full
from features_v6 import FEATURE_NAMES_V6_EXTRA, build_v6_features
from regime_filter import classify_regime
from test_oos_btc_v5 import build_sim_basket_state, _as_compact_arrays, load_v5_models, ensure_perp_data

BINANCE_BASE = "https://api.binance.com"
DATA_DIR = os.path.join(os.path.dirname(__file__), "challenge_data")
PERP_DIR = os.path.join(os.path.dirname(__file__), "data", "perp")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "data")

# ── Execution Costs Aligned to config/training ──
# Imported from trading_config
MMR = 0.005 # 0.5% Maintenance Margin Rate
STARTING_CAPITAL = 10000.0

COIN_ONEHOT_NAMES = [
    "AAVEUSDT", "ADAUSDT", "AVAXUSDT", "BNBUSDT", "BTCUSDT",
    "DOGEUSDT", "ETHUSDT", "LINKUSDT", "PEPEUSDT", "SHIBUSDT",
    "SOLUSDT", "UNIUSDT", "WIFUSDT", "XRPUSDT",
]
ALL_FEATURE_NAMES    = FEATURE_NAMES_V5 + [f"coin_is_{s}" for s in COIN_ONEHOT_NAMES]  # 94 features
ALL_FEATURE_NAMES_V6 = ALL_FEATURE_NAMES + FEATURE_NAMES_V6_EXTRA                     # 130 features

# V6 blending: V6 = 70%, V5 = 30%  (user-specified)
V6_WEIGHT = 0.70
V5_WEIGHT = 0.30
# V6 micro gate threshold
V6_MICRO_GATE = 0.50

# Select a random coin that is not BTC or ETH and has local parquet files
CANDIDATE_COINS = ["AAVEUSDT", "ADAUSDT", "AVAXUSDT", "BNBUSDT", "DOGEUSDT", "LINKUSDT", "SOLUSDT", "UNIUSDT", "WIFUSDT", "XRPUSDT"]
RANDOM_COIN = random.choice(CANDIDATE_COINS)
if len(sys.argv) > 1:
    arg = sys.argv[1].upper()
    if arg in CANDIDATE_COINS:
        RANDOM_COIN = arg
    elif arg + "USDT" in CANDIDATE_COINS:
        RANDOM_COIN = arg + "USDT"

# OOS Blind Month: May 2025
START_DATE_STR = "2025-05-01"
END_DATE_STR = "2025-06-01"

# Warmup start (10 days prior to load indicators)
WARMUP_DATE_STR = "2025-04-20"

# Feature Indices
IDX_HURST     = ALL_FEATURE_NAMES.index("hurst")
IDX_TE        = ALL_FEATURE_NAMES.index("trend_efficiency")
IDX_RANGE_POS = ALL_FEATURE_NAMES.index("range_position")
IDX_RV1H      = ALL_FEATURE_NAMES.index("rv_1h")
IDX_RV_Z_1D   = ALL_FEATURE_NAMES.index("rv_zscore_1d")
IDX_VOL_BURST = ALL_FEATURE_NAMES.index("vol_burst_1h")

def download_klines(symbol, start_ms, end_ms):
    candles = []; current = start_ms
    total = (end_ms - start_ms) // 60000
    print(f"  Downloading {symbol} ({total:,} candles)...", flush=True)
    retry_count = 0
    while current < end_ms:
        try:
            r = requests.get(f"{BINANCE_BASE}/api/v3/klines",
                             params={"symbol": symbol, "interval": "1m",
                                     "startTime": current, "endTime": end_ms,
                                     "limit": 1500}, timeout=15)
            r.raise_for_status(); data = r.json()
            retry_count = 0
        except Exception as e:
            retry_count += 1
            if retry_count > 3:
                print(f"    Skipping {symbol} after 3 failed retries: {e}")
                break
            print(f"    retry {retry_count}/3: {e}"); time.sleep(2); continue
        if not data: break
        for k in data:
            candles.append({
                "timestamp": int(k[0]), "open": float(k[1]), "high": float(k[2]),
                "low": float(k[3]), "close": float(k[4]), "volume": float(k[5]),
                "taker_buy_volume": float(k[9]),
            })
        current = int(data[-1][0]) + 60000
        time.sleep(0.1)
    print(f"    done: {len(candles):,}")
    return candles

def _coin_onehot(symbol, n):
    block = np.zeros((n, len(COIN_ONEHOT_NAMES)), dtype=np.float32)
    if symbol in COIN_ONEHOT_NAMES:
        block[:, COIN_ONEHOT_NAMES.index(symbol)] = 1.0
    return block

def mean_variance_size(equity, mu_pct, sigma_pct, p_up, tp_pct, sl_pct):
    sigma = max(abs(float(sigma_pct)), 1e-4)
    mv = float(mu_pct) / (sigma ** 2)
    pos_mv = max(0.0, equity * RISK_AVERSION_TARGET * mv)
    b = float(tp_pct) / max(float(sl_pct), 1e-9)
    p = float(p_up); q = 1 - p
    f_star = max(0.0, (p * b - q) / max(b, 1e-9))
    pos_kelly = equity * KELLY_FRACTION * f_star
    pos_risk = equity * RISK_PER_TRADE_PCT / max(float(sl_pct), 1e-9)
    pos_cap  = equity * MAX_POSITION_PCT
    return float(max(0.0, min(pos_mv, pos_kelly, pos_risk, pos_cap)))

def load_v6_models_backtest():
    """Load V6_full + V6_micro models for backtest blending. Returns (full_models, micro_models)."""
    import pickle, json as _json
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    meta_path = os.path.join(data_dir, "v6_meta.json")
    if not os.path.exists(meta_path):
        return None, None
    with open(meta_path) as f:
        meta = _json.load(f)
    horizons = meta.get("horizons", [60, 720])
    full_m, micro_m = {}, {}
    for h in horizons:
        full_m[h] = {}; micro_m[h] = {}
        for direction in ["long", "short"]:
            for tag, store in [("full", full_m), ("micro", micro_m)]:
                cp = os.path.join(data_dir, f"v6_{tag}_clf_{direction}_h{h}.json")
                pp = os.path.join(data_dir, f"v6_{tag}_calib_{direction}_h{h}.pkl")
                if not os.path.exists(cp) or not os.path.exists(pp):
                    return None, None
                clf = xgb.Booster(); clf.load_model(cp)
                with open(pp, "rb") as fp:
                    calib = pickle.load(fp)
                try:
                    p_thr = meta["horizon_results"][str(h)][direction][tag]["p_threshold"]
                except (KeyError, TypeError):
                    p_thr = 0.56
                store[h][direction] = {"clf": clf, "calib": calib, "p_threshold": float(p_thr)}
    return full_m, micro_m


def main():
    print("=" * 80)
    print("  BLIND OUT-OF-SAMPLE FUTURES BACKTEST (V5 + V6 blend)")
    print(f"  Chosen Coin: {RANDOM_COIN}")
    print(f"  Test Period: {START_DATE_STR} to {END_DATE_STR}")
    print(f"  Slippage: {SLIPPAGE_PCT_PER_SIDE*10000:.0f} bps/side  |  Fees: {FEE_PCT_PER_SIDE*10000:.0f} bps/side")
    print(f"  Round Trip Cost: {ROUND_TRIP_COST*100:.2f}%  |  MMR: {MMR*100:.1f}%")
    print("=" * 80)

    # Make sure we have the required perp data parquets
    print("\nChecking perp data parquets...")
    ensure_perp_data([RANDOM_COIN, "BTCUSDT"])

    # Convert times to milliseconds
    d_warmup = datetime.datetime.strptime(WARMUP_DATE_STR, "%Y-%m-%d")
    d_start = datetime.datetime.strptime(START_DATE_STR, "%Y-%m-%d")
    d_end = datetime.datetime.strptime(END_DATE_STR, "%Y-%m-%d")

    warmup_ms = int(d_warmup.timestamp() * 1000)
    start_ms = int(d_start.timestamp() * 1000)
    end_ms = int(d_end.timestamp() * 1000)

    # 1. Download Spot data
    print("\n[1/5] Downloading Spot data from Binance...")
    target_candles = download_klines(RANDOM_COIN, warmup_ms, end_ms)
    btc_candles = download_klines("BTCUSDT", warmup_ms, end_ms)

    if not target_candles or not btc_candles:
        print("ERROR: Download failed.")
        return

    # Extract target array components
    tc_n = len(target_candles)
    tc_ts = np.fromiter((c["timestamp"] for c in target_candles), dtype=np.int64, count=tc_n)
    tc_op = np.fromiter((c["open"] for c in target_candles), dtype=np.float32, count=tc_n)
    tc_hi = np.fromiter((c["high"] for c in target_candles), dtype=np.float32, count=tc_n)
    tc_lo = np.fromiter((c["low"] for c in target_candles), dtype=np.float32, count=tc_n)
    tc_cl = np.fromiter((c["close"] for c in target_candles), dtype=np.float32, count=tc_n)
    tc_vo = np.fromiter((c["volume"] for c in target_candles), dtype=np.float32, count=tc_n)
    tc_tb = np.fromiter((c["taker_buy_volume"] for c in target_candles), dtype=np.float32, count=tc_n)

    # Extract BTC components
    btc_n = len(btc_candles)
    btc_ts = np.fromiter((c["timestamp"] for c in btc_candles), dtype=np.int64, count=btc_n)
    btc_cl = np.fromiter((c["close"] for c in btc_candles), dtype=np.float32, count=btc_n)

    # Align BTC close to target timestamps
    btc_cl_aligned = np.interp(tc_ts.astype(np.float64), btc_ts.astype(np.float64), btc_cl.astype(np.float64))

    # 2. Build cross-sectional basket state
    print("\n[2/5] Building Cross-Sectional Basket State...")
    coin_data = {
        RANDOM_COIN: (tc_ts, tc_cl),
        "BTCUSDT": (btc_ts, btc_cl)
    }
    basket_state = build_sim_basket_state(coin_data)

    # 3. Compute V5 Features on the fly
    print("\n[3/5] Computing V5 Features on the fly...")
    from train_ev_model_v2 import build_features_v2
    # Compute true OFI
    tc_ofi = (2.0 * tc_tb - tc_vo).astype(np.float64)
    
    print("  Calculating V2/V4 features...")
    F_v2, atr_arr = build_features_v2(
        tc_cl.astype(np.float64), 
        tc_hi.astype(np.float64), 
        tc_lo.astype(np.float64), 
        tc_vo.astype(np.float64), 
        tc_ofi, 
        tc_ts.astype(np.float64), 
        btc_cl_aligned
    )
    
    print("  Stacking V5 features...")
    X_features = build_v5_full(
        RANDOM_COIN, F_v2, tc_cl.astype(np.float64), 
        tc_hi.astype(np.float64), tc_lo.astype(np.float64), 
        tc_vo.astype(np.float64), tc_ofi, tc_tb.astype(np.float64),
        tc_ts, atr_arr, btc_cl_aligned, PERP_DIR, basket_state
    )

    onehot = _coin_onehot(RANDOM_COIN, tc_n)
    X_v5 = np.hstack([X_features, onehot]).astype(np.float32)  # (n, 94)
    print(f"  V5 features shape: {X_v5.shape}")

    # 3b. Build V6 features on top of V5
    print("  Building V6 feature block (36 extra features)...")
    try:
        F_v6_block = build_v6_features(
            tc_cl.astype(np.float64), tc_hi.astype(np.float64), tc_lo.astype(np.float64),
            tc_vo.astype(np.float64), tc_ofi, tc_tb.astype(np.float64),
            tc_ts, atr_arr
        )  # (n, 36)
        X_v6 = np.hstack([X_v5, F_v6_block]).astype(np.float32)  # (n, 130)
        print(f"  V6 full stack shape: {X_v6.shape}")
        V6_AVAILABLE = True
    except Exception as e:
        print(f"  [WARN] V6 feature build failed: {e}  — running V5-only mode.")
        F_v6_block = None
        X_v6 = None
        V6_AVAILABLE = False

    # Backward compat: keep X pointing to V5 for non-blended code paths
    X = X_v5

    # 4. Load Models
    print("\n[4/5] Loading trained models (V5 + V6 if available)...")
    meta, models = load_v5_models()
    if not models:
        print("ERROR: V5 models not found.")
        return
    v6_full_models, v6_micro_models = load_v6_models_backtest()
    if v6_full_models is not None:
        print(f"  V6 models loaded — blend: V5×{V5_WEIGHT:.0%} + V6×{V6_WEIGHT:.0%}")
    else:
        print("  V6 models not found — running V5-only (run train_v6.py to enable).")
        V6_AVAILABLE = False

    # Filter out horizons we want to study
    _HORIZON_MODELS = [(h, m) for h, m in models.items() if h in HORIZONS]

    # Pre-compute BTC 4h momentum lookup
    print("  Building BTC momentum filter...")
    _btc_4h_mom = np.zeros(len(btc_ts), dtype=np.float32)
    if len(btc_cl) > 240:
        _btc_4h_mom[240:] = (np.log(btc_cl[240:]) - np.log(btc_cl[:-240])).astype(np.float32)
    _ct_arr = np.array(tc_ts, dtype=np.int64)
    _bi_arr = np.searchsorted(btc_ts, _ct_arr, side="right") - 1
    _btc_4h_mom_aligned = _btc_4h_mom[np.clip(_bi_arr, 0, len(_btc_4h_mom)-1)]

    # 5. Run simulation
    print("\n[5/5] Running Simulation loop...")
    capital = STARTING_CAPITAL
    equity_curve = [capital]
    all_trades = []
    
    # Track open position expiry
    open_until_ts = 0
    
    # Sample every CHECK_EVERY_MINS (training cadence) — not every raw 1m bar
    _all_oos_ts = [t for t in tc_ts if t >= start_ms and t < end_ms]
    check_times   = _all_oos_ts[::CHECK_EVERY_MINS]
    check_indices = [int(np.searchsorted(tc_ts, t)) for t in check_times]
    
    # Filter stats
    filter_stats = {
        "p_below_threshold": 0,
        "ev_below_threshold": 0,
        "tp_too_small": 0,
        "candidates_passed": 0,
        "liquidations": 0
    }

    for step, idx in enumerate(check_indices):
        check_ts = tc_ts[idx]
        
        # Check if already in trade
        if check_ts < open_until_ts:
            continue
            
        f_row = X[idx]
        if np.any(np.isnan(f_row)):
            continue
            
        entry = float(tc_cl[idx])
        atr_val = atr_arr[idx]
        if atr_val <= 0:
            continue
            
        hurst = float(F_v2[idx, 10])

        # Regime classification
        rz = float(f_row[IDX_RV_Z_1D])
        if rz > 2.5 and abs(hurst - 0.5) < 0.05:
            regime = "CHAOS"
        else:
            vb = float(f_row[IDX_VOL_BURST])
            rv = float(f_row[IDX_RV1H])
            if vb < -0.4 and rv < 0.0005:
                regime = "LOW_LIQUIDITY"
            else:
                te = float(f_row[IDX_TE])
                if rz > 1.5 and te > 0.30:
                    regime = "EXPANSION"
                elif hurst > 0.55:
                    regime = "TREND_UP" if float(f_row[IDX_RANGE_POS]) >= 0 else "TREND_DOWN"
                else:
                    regime = "RANGE"

        if regime in HALT_REGIMES:
            continue

        # Evaluate horizons
        candidates = []
        stacked_v5  = X_v5[idx:idx+1]   # (1, 94)
        stacked_v6  = X_v6[idx:idx+1]  if (V6_AVAILABLE and X_v6 is not None) else None  # (1, 130)
        stacked_v6m = F_v6_block[idx:idx+1] if (V6_AVAILABLE and F_v6_block is not None) else None  # (1, 36)
        # backward compat alias
        stacked_feats = stacked_v5

        for h, m_group in _HORIZON_MODELS:
            for direction in ["long", "short"]:
                m = m_group.get(direction)
                if m is None: continue

                # V5 probability
                p_v5_raw = m["clf"].inplace_predict(stacked_v5)
                p_v5_cal = float(m["calib"].predict(p_v5_raw)[0])
                mu_hat   = float(m["reg"].inplace_predict(stacked_v5)[0])

                # V6 blending
                if (V6_AVAILABLE and v6_full_models is not None
                        and stacked_v6 is not None
                        and h in v6_full_models and direction in v6_full_models[h]):
                    try:
                        mv6 = v6_full_models[h][direction]
                        p_v6_raw = mv6["clf"].inplace_predict(stacked_v6)
                        p_v6_cal = float(mv6["calib"].predict(p_v6_raw)[0])
                        # Micro gate
                        gate = True
                        if (v6_micro_models and stacked_v6m is not None
                                and h in v6_micro_models and direction in v6_micro_models[h]):
                            mv6m = v6_micro_models[h][direction]
                            p_m_raw = mv6m["clf"].inplace_predict(stacked_v6m)
                            gate = float(mv6m["calib"].predict(p_m_raw)[0]) >= V6_MICRO_GATE
                        if gate:
                            p_cal = V5_WEIGHT * p_v5_cal + V6_WEIGHT * p_v6_cal
                            thr_model = (V5_WEIGHT * float(m["p_threshold"])
                                         + V6_WEIGHT * float(mv6["p_threshold"]))
                        else:
                            p_cal = p_v5_cal
                            thr_model = float(m["p_threshold"])
                    except Exception:
                        p_cal = p_v5_cal
                        thr_model = float(m["p_threshold"])
                else:
                    p_cal = p_v5_cal
                    thr_model = float(m["p_threshold"])  # strict V5 threshold

                scale = horizon_atr_scale(h)
                
                tp_mult = regime_tp_mult(hurst) * scale
                sl_mult = SL_MULT * scale
                tp_pct = tp_mult * atr_val / entry
                sl_pct = sl_mult * atr_val / entry
                sl_pct = min(sl_pct, MAX_SL_PCT)
                
                if tp_pct < MIN_TP_TO_COST_RATIO * ROUND_TRIP_COST:
                    filter_stats["tp_too_small"] += 1
                    continue
                    
                if p_cal < thr_model:
                    filter_stats["p_below_threshold"] += 1
                    continue
                    
                if direction == "short":
                    mu_hat = -mu_hat
                    
                ev = p_cal * tp_pct - (1 - p_cal) * sl_pct - ROUND_TRIP_COST
                if ev < MIN_EV_PCT:
                    filter_stats["ev_below_threshold"] += 1
                    continue

                _h_scale = HORIZON_REGIME_EV_SCALE.get(regime, {}).get(h, 1.0)
                _effective_ev = ev * _h_scale
                
                if direction == "long":
                    tp_price = entry + tp_mult * atr_val
                    sl_price = entry * (1.0 - sl_pct)
                else:
                    tp_price = entry - tp_mult * atr_val
                    sl_price = entry * (1.0 + sl_pct)

                candidates.append({
                    "horizon": h, "direction": direction, "p_up": p_cal, "mu_hat": mu_hat,
                    "tp_pct": tp_pct, "sl_pct": sl_pct,
                    "tp_price": tp_price, "sl_price": sl_price,
                    "ev": ev, "effective_ev": _effective_ev,
                    "hurst": hurst, "regime": regime
                })

        # Apply market momentum filter
        _btc_mom = _btc_4h_mom_aligned[idx]
        _active_min_ev = MIN_EV_PCT * (BTC_QUIET_EV_MULTIPLIER if abs(_btc_mom) < BTC_QUIET_4H_THRESHOLD else 1.0)
        candidates = [c for c in candidates if c["ev"] >= _active_min_ev]

        if not candidates:
            continue

        candidates.sort(key=lambda c: -c["effective_ev"])
        c = candidates[0]
        filter_stats["candidates_passed"] += 1

        # Position Sizing
        window = max(min(c["horizon"], idx), 5)
        log_w = np.diff(np.log(np.maximum(tc_cl[idx-window:idx+1], 1e-12)))
        realized_sigma = float(np.std(log_w) * np.sqrt(c["horizon"])) if len(log_w) > 1 else c["sl_pct"]
        atr_sigma = float(c["sl_pct"] / SL_MULT)
        sigma_pct = max(realized_sigma, atr_sigma, 1e-4)
        mu_pct = max(c["mu_hat"], 0.0)

        desired_nominal_position = mean_variance_size(capital, mu_pct, sigma_pct, c["p_up"], c["tp_pct"], c["sl_pct"])
        asset_scale = ASSET_MAX_POSITION_SCALE.get(RANDOM_COIN, 1.0)
        desired_nominal_position *= asset_scale

        # --- DYNAMIC FUTURES LEVERAGE (20x - 50x) ---
        confidence_surplus = max(0.0, c["p_up"] - 0.59)
        leverage = 20.0 + (confidence_surplus / 0.11) * 30.0
        leverage = min(50.0, max(20.0, leverage))

        # Enforce leverage cap to guarantee stop loss is reached BEFORE isolated liquidation
        # Liquidation distance = 1/leverage - MMR
        # We need: 1/leverage - MMR > sl_pct => 1/leverage > sl_pct + MMR => leverage < 1/(sl_pct + MMR)
        leverage_cap = 1.0 / (c["sl_pct"] + MMR)
        leverage = min(leverage, leverage_cap)

        margin_required = desired_nominal_position / leverage
        margin_required = min(margin_required, capital)

        if desired_nominal_position < 10.0: # Minimum nominal position size $10
            continue

        position = margin_required * leverage

        # ─── REAL FUTURES LIQUIDATION RULES ───
        if c["direction"] == "long":
            liq_price = entry * (1.0 - 1.0 / leverage + MMR)
        else:
            liq_price = entry * (1.0 + 1.0 / leverage - MMR)

        end_idx = min(idx + c["horizon"], tc_n - 1)
        fwd_hi = tc_hi[idx+1:end_idx+1]
        fwd_lo = tc_lo[idx+1:end_idx+1]
        fwd_cl = tc_cl[idx+1:end_idx+1]

        # Scan candle by candle to check for Liquidation vs SL vs TP
        reason = "time_exit"
        gross = 0.0
        close_offset = len(fwd_hi) - 1

        is_liq_closer = (1.0 / leverage - MMR) < c["sl_pct"]

        for t_step in range(len(fwd_cl)):
            cur_hi = fwd_hi[t_step]
            cur_lo = fwd_lo[t_step]
            cur_cl = fwd_cl[t_step]

            # We must evaluate the loss boundary closer to the entry first
            if is_liq_closer:
                # 1. Check Liquidation First
                if c["direction"] == "long" and cur_lo <= liq_price:
                    reason = "LIQUIDATION"
                    gross = -1.0
                    close_offset = t_step
                    filter_stats["liquidations"] += 1
                    break
                elif c["direction"] == "short" and cur_hi >= liq_price:
                    reason = "LIQUIDATION"
                    gross = -1.0
                    close_offset = t_step
                    filter_stats["liquidations"] += 1
                    break

                # 2. Check Stop Loss Second
                if c["direction"] == "long" and cur_lo <= c["sl_price"]:
                    reason = "stop_loss"
                    gross = -c["sl_pct"]
                    close_offset = t_step
                    break
                elif c["direction"] == "short" and cur_hi >= c["sl_price"]:
                    reason = "stop_loss"
                    gross = -c["sl_pct"]
                    close_offset = t_step
                    break
            else:
                # 1. Check Stop Loss First (standard case when leverage is capped)
                if c["direction"] == "long" and cur_lo <= c["sl_price"]:
                    reason = "stop_loss"
                    gross = -c["sl_pct"]
                    close_offset = t_step
                    break
                elif c["direction"] == "short" and cur_hi >= c["sl_price"]:
                    reason = "stop_loss"
                    gross = -c["sl_pct"]
                    close_offset = t_step
                    break

                # 2. Check Liquidation Second
                if c["direction"] == "long" and cur_lo <= liq_price:
                    reason = "LIQUIDATION"
                    gross = -1.0
                    close_offset = t_step
                    filter_stats["liquidations"] += 1
                    break
                elif c["direction"] == "short" and cur_hi >= liq_price:
                    reason = "LIQUIDATION"
                    gross = -1.0
                    close_offset = t_step
                    filter_stats["liquidations"] += 1
                    break

            # 3. Check Take Profit
            if c["direction"] == "long" and cur_hi >= c["tp_price"]:
                reason = "take_profit"
                gross = c["tp_pct"]
                close_offset = t_step
                break
            elif c["direction"] == "short" and cur_lo <= c["tp_price"]:
                reason = "take_profit"
                gross = c["tp_pct"]
                close_offset = t_step
                break

        # Time exit if none hit
        if reason == "time_exit":
            exit_price = float(tc_cl[end_idx])
            if c["direction"] == "long":
                gross = (exit_price - entry) / entry
            else:
                gross = (entry - exit_price) / entry
            close_offset = min(c["horizon"], len(fwd_hi)-1)

        # Blended runner logic only applies to non-liquidated, non-SL, non-time-exits at TP
        if reason == "take_profit" and c["horizon"] >= 240:
            tp_t = close_offset
            runner_hi = fwd_hi[tp_t:]
            runner_lo = fwd_lo[tp_t:]
            
            if c["direction"] == "long":
                tp2_price = entry * (1.0 + c["tp_pct"] * 1.5)
                be_sl_price = entry * (1.0 + ROUND_TRIP_COST)
                hit_tp2 = np.where(runner_hi >= tp2_price)[0]
                hit_be = np.where(runner_lo <= be_sl_price)[0]
            else:
                tp2_price = entry * (1.0 - c["tp_pct"] * 1.5)
                be_sl_price = entry * (1.0 - ROUND_TRIP_COST)
                hit_tp2 = np.where(runner_lo <= tp2_price)[0]
                hit_be = np.where(runner_hi >= be_sl_price)[0]
                
            t2_t = hit_tp2[0] if len(hit_tp2) else np.iinfo(np.int32).max
            be_t = hit_be[0] if len(hit_be) else np.iinfo(np.int32).max
            
            # Check liquidation on the runner phase
            liq_t = np.iinfo(np.int32).max
            if c["direction"] == "long":
                liq_hits = np.where(runner_lo <= liq_price)[0]
            else:
                liq_hits = np.where(runner_hi >= liq_price)[0]
            if len(liq_hits):
                liq_t = liq_hits[0]

            if liq_t < t2_t and liq_t < be_t:
                runner_gross = -1.0
                reason = "runner_LIQUIDATED"
                close_offset = tp_t + liq_t
                filter_stats["liquidations"] += 1
            elif t2_t == np.iinfo(np.int32).max and be_t == np.iinfo(np.int32).max:
                runner_exit = float(tc_cl[end_idx])
                if c["direction"] == "long":
                    runner_gross = (runner_exit - entry) / entry
                else:
                    runner_gross = (entry - runner_exit) / entry
                reason = "partial_tp1_time_exit"
                close_offset = len(fwd_hi) - 1
            elif t2_t < be_t:
                runner_gross = c["tp_pct"] * 1.5
                reason = "partial_tp1_and_tp2"
                close_offset = tp_t + t2_t
            else:
                runner_gross = ROUND_TRIP_COST
                reason = "partial_tp1_stopped_be"
                close_offset = tp_t + be_t
                
            gross = (0.5 * c["tp_pct"]) + (0.5 * runner_gross)

        # Net Return & PnL calculation
        total_fees = position * (FEE_PCT_PER_SIDE * 2)
        total_slippage = position * (SLIPPAGE_PCT_PER_SIDE * 2)
        
        if "LIQUIDATED" in reason or reason == "LIQUIDATION":
            pnl = -margin_required
        else:
            pnl = (position * gross) - total_fees - total_slippage

        capital += pnl
        equity_curve.append(capital)
        
        open_until_ts = tc_ts[idx + close_offset + 1]
        hold_mins = close_offset + 1
        close_ts = tc_ts[idx + close_offset]
        
        d_open = datetime.datetime.utcfromtimestamp(check_ts/1000).strftime('%Y-%m-%d %H:%M')
        d_close = datetime.datetime.utcfromtimestamp(close_ts/1000).strftime('%Y-%m-%d %H:%M')
        
        trade_record = {
            "sym": RANDOM_COIN,
            "direction": c["direction"],
            "horizon": c["horizon"],
            "entry": entry,
            "margin": margin_required,
            "leverage": leverage,
            "size": position,
            "gross": gross,
            "pnl": pnl,
            "reason": reason,
            "hold_mins": hold_mins
        }
        all_trades.append(trade_record)

        print(f"  [TRADE] {RANDOM_COIN:<10} {c['direction'].upper():<5} | Margin: ${margin_required:>7,.2f} ({leverage:.1f}x Lev) -> Size: ${position:>9,.2f} | Open: {d_open} -> Close: {d_close} | "
              f"Result: {reason:<20} | PnL: ${pnl:>+8,.2f} | Cap: ${capital:>10,.2f} | Hold: {hold_mins}m", flush=True)

    # Calculate Drawdown
    peaks = np.maximum.accumulate(equity_curve)
    drawdowns = (peaks - equity_curve) / peaks * 100
    max_dd = drawdowns.max()

    # Final Report
    print("\n" + "="*80)
    print("  FINAL REPORT - BLIND FUTURES TEST")
    print("="*80)
    print(f"  Asset:               {RANDOM_COIN}")
    print(f"  Starting Capital:    $ 10,000.00")
    print(f"  Final Capital:       $ {capital:>10,.2f}")
    total_pnl = capital - STARTING_CAPITAL
    print(f"  Total P&L:           $ {total_pnl:>+10,.2f}  ({total_pnl/STARTING_CAPITAL*100:>+.2f}%)")
    print(f"  Max Drawdown:        {max_dd:.2f}%")
    print(f"  Total Trades:        {len(all_trades)}")
    
    if len(all_trades) > 0:
        wins = [t for t in all_trades if t["pnl"] > 0]
        losses = [t for t in all_trades if t["pnl"] <= 0]
        win_rate = len(wins) / len(all_trades) * 100
        avg_win = np.mean([t["pnl"] for t in wins]) if wins else 0.0
        avg_loss = np.mean([t["pnl"] for t in losses]) if losses else 0.0
        gross_wins = sum(t["pnl"] for t in wins)
        gross_losses = -sum(t["pnl"] for t in losses)
        profit_factor = gross_wins / max(gross_losses, 1e-9)
        
        print(f"  Win Rate:            {win_rate:.1f}%")
        print(f"  Avg Win:             $ {avg_win:>10,.2f}")
        print(f"  Avg Loss:            $ {avg_loss:>10,.2f}")
        print(f"  Profit Factor:       {profit_factor:.2f}")
        print(f"  Liquidations:        {filter_stats['liquidations']}")
    else:
        print("  No trades executed.")
    print("="*80)
    print("\n  FILTER REJECTION BREAKDOWN:")
    total_checks = len(check_indices)
    for k, v in filter_stats.items():
        if k == "liquidations": continue
        pct = v / max(total_checks, 1) * 100
        print(f"    {k:<30}: {v:>7,}  ({pct:.1f}% of {total_checks:,} checks)")
    print(f"    {'candidates_passed':<30}: {filter_stats['candidates_passed']:>7,}")

if __name__ == "__main__":
    main()
