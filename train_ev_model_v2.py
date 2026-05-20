"""
train_ev_model_v2.py — XGBoost + Expanded Features + Better Labels
===================================================================
Upgrades over v1:
  1. XGBoost gradient boosting (finds non-linear interactions)
  2. 7 new features: volume_zscore, atr_norm_ofi, hour_sin/cos, dow_sin/cos, btc_return_60m
  3. Regime-conditioned triple-barrier labels (wider TP in trends, tighter in ranges)
  4. BTC cross-market awareness (loads BTC data alongside each coin)
"""
from __future__ import annotations
import os, glob, time, json, math
import numpy as np
import polars as pl
import xgboost as xgb
from typing import Optional

PROCESSED_DIR = r"E:\training data for quant\processed_features"
MODEL_PATH    = os.path.join(os.path.dirname(__file__), "data", "ev_model_xgb.json")
META_PATH     = os.path.join(os.path.dirname(__file__), "data", "ev_model_v2_meta.json")

SAMPLE_EVERY  = 10
WARMUP        = 250
FORWARD_BARS  = 60
ATR_PERIOD    = 14

# 25 features total
FEATURE_NAMES_V2 = [
    "ofi_score", "ofi_acceleration", "large_trade_ofi", "absorption_index",
    "cvd_trend_15m", "cvd_trend_1h", "cvd_trend_4h",
    "cvd_divergence_15m", "cvd_divergence_1h", "cvd_divergence_4h",
    "hurst", "trend_efficiency", "persistence_score",
    "depth_imbalance", "vol_state", "fear_greed_norm",
    "accum_probability", "distrib_probability",
    # NEW v2 features
    "volume_zscore", "atr_normalized_ofi",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "btc_return_60m",
]

# ── Helpers ──────────────────────────────────────────────────────────────

def _roll_mean(x, w):
    out = np.full(len(x), np.nan)
    cs = np.cumsum(np.nan_to_num(x, nan=0.0))
    out[w-1:] = (cs[w-1:] - np.concatenate([[0], cs[:-w]])) / w
    return out

def _roll_std(x, w):
    m = _roll_mean(x, w)
    m2 = _roll_mean(x**2, w)
    var = m2 - m**2
    return np.sqrt(np.maximum(var, 0))

def _atr(high, low, close, p=14):
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    tr[1:] = np.maximum.reduce([high[1:]-low[1:], np.abs(high[1:]-close[:-1]), np.abs(low[1:]-close[:-1])])
    atr = np.full(n, np.nan)
    atr[p-1] = tr[:p].mean()
    k = (p-1)/p
    for i in range(p, n):
        atr[i] = atr[i-1]*k + tr[i]/p
    return atr

def _hurst(prices):
    if len(prices) < 50: return None
    rets = np.diff(np.log(prices + 1e-10))
    n = len(rets)
    lags, rs = [], []
    for lag in [max(10, n//8), max(10, n//4), max(10, n//2)]:
        if lag >= n: continue
        sub = rets[-lag:]
        dev = np.cumsum(sub - sub.mean())
        R = dev.max() - dev.min()
        S = sub.std(ddof=1)
        if S > 0 and R > 0:
            lags.append(np.log(lag)); rs.append(np.log(R/S))
    if len(rs) < 2: return None
    return float(np.clip(np.polyfit(lags, rs, 1)[0], 0.01, 0.99))


# ── Feature Builder (25 features) ───────────────────────────────────────

def build_features_v2(close, high, low, volume, ofi, timestamps_ms, btc_close=None):
    """Build (n, 25) feature matrix."""
    n = len(close)
    F = np.full((n, 25), np.nan)
    vol_safe = np.where(volume > 0, volume, 1.0)
    ofi_score = np.clip(ofi / vol_safe, -1, 1)
    atr_arr = _atr(high, low, close, ATR_PERIOD)

    # 0: ofi_score
    F[:, 0] = ofi_score
    # 1: ofi_acceleration
    F[5:, 1] = np.clip(ofi_score[5:] - ofi_score[:-5], -1, 1)
    # 2: large_trade_ofi (unavailable)
    F[:, 2] = 0.0
    # 3: absorption_index
    rng = np.maximum(high - low, 1e-10)
    vol_ma = _roll_mean(volume, 20)
    rng_ma = _roll_mean(rng, 20)
    with np.errstate(invalid='ignore', divide='ignore'):
        abs_score = np.where((vol_ma>0)&(rng_ma>0), (volume/vol_ma)/np.maximum(rng/rng_ma, 0.1), 1.0)
    # Vectorised 20-bar rolling min/max (was 1.75M-iter Python list comp).
    try:
        from numpy.lib.stride_tricks import sliding_window_view
        pad_lo = np.concatenate([np.full(19, close[0]), close])
        pad_hi = np.concatenate([np.full(19, close[0]), close])
        low20 = sliding_window_view(pad_lo, 20).min(axis=1)
        high20 = sliding_window_view(pad_hi, 20).max(axis=1)
    except Exception:
        low20 = np.array([close[max(0,i-19):i+1].min() for i in range(n)])
        high20 = np.array([close[max(0,i-19):i+1].max() for i in range(n)])
    rng20 = np.maximum(high20 - low20, 1e-10)
    ppos = (close - low20) / rng20
    sign = np.where(ppos < 0.35, 1.0, np.where(ppos > 0.65, -1.0, 0.0))
    F[:, 3] = np.clip((abs_score - 1.0) / 2.0, 0, 1) * sign

    # 4-6: cvd_trend 15m/1h/4h
    cvd = np.cumsum(ofi)
    for col, w in zip([4,5,6], [15,60,240]):
        t = np.full(n, np.nan)
        denom = np.where(np.abs(cvd) > 0, np.abs(cvd), 1.0)
        t[w:] = np.clip((cvd[w:] - cvd[:-w]) / denom[w:], -1, 1)
        F[:, col] = t

    # 7-9: cvd_divergence 15m/1h/4h
    for col, w in zip([7,8,9], [15,60,240]):
        div = np.where((close[w:] > close[:-w]) & (cvd[w:] < cvd[:-w]), -0.7,
              np.where((close[w:] < close[:-w]) & (cvd[w:] > cvd[:-w]),  0.7, 0.0))
        F[w:, col] = div

    # 10: hurst (every 60 bars, forward-filled)
    hurst_arr = np.full(n, np.nan)
    for i in range(200, n, 60):
        h = _hurst(close[max(0, i-200):i+1])
        if h is not None: hurst_arr[i] = (h - 0.5) * 2.0
    last = np.nan
    for i in range(n):
        if not np.isnan(hurst_arr[i]): last = hurst_arr[i]
        elif not np.isnan(last): hurst_arr[i] = last
    F[:, 10] = hurst_arr

    # 11: trend_efficiency (Kaufman ER)
    diff_abs = np.abs(np.diff(close, prepend=close[0]))
    path_cs = np.cumsum(diff_abs)
    path_lag = np.concatenate([np.zeros(14), path_cs[:-14]])
    path_14 = path_cs - path_lag
    close_lag = np.concatenate([np.full(14, close[0]), close[:-14]])
    net_14 = np.abs(close - close_lag)
    with np.errstate(invalid='ignore', divide='ignore'):
        er = np.where(path_14 > 0, net_14 / path_14, 0.0)
    te = np.full(n, np.nan)
    te[14:] = np.clip((er[14:] - 0.5) * 2.0, -1, 1)
    F[:, 11] = te

    # 12: persistence_score
    ofi_dir = np.sign(ofi_score)
    pos_cs = np.cumsum((ofi_dir > 0).astype(float))
    pos_roll = pos_cs - np.concatenate([np.zeros(20), pos_cs[:-20]])
    F[20:, 12] = np.clip((pos_roll[20:] / 20.0 - 0.5) * 2.0, -1, 1)

    # 13: depth_imbalance (unavailable)
    F[:, 13] = 0.0
    # 14: vol_state
    log_ret = np.full(n, 0.0)
    log_ret[1:] = np.diff(np.log(np.maximum(close, 1e-10)))
    ret_sq = log_ret**2
    recent_var = _roll_mean(ret_sq, 5)
    prior_var = _roll_mean(ret_sq, 25)
    with np.errstate(invalid='ignore', divide='ignore'):
        vr = np.where(prior_var > 0, recent_var / prior_var, 1.0)
    F[25:, 14] = np.where(vr[25:] > 1.5, 1.0, np.where(vr[25:] < 0.5, -1.0, 0.0))

    # 15: fear_greed_norm (unavailable)
    F[:, 15] = 0.0
    # 16-17: accum/distrib probability
    F[60:, 16] = np.where((close[60:] < close[:-60]) & (cvd[60:] > cvd[:-60]), 0.7,
                 np.where((close[60:] > close[:-60]) & (cvd[60:] < cvd[:-60]), -0.5, 0.0))
    F[:, 17] = -F[:, 16]

    # ── NEW V2 FEATURES ──────────────────────────────────────────────────

    # 18: volume_zscore (rolling 20-bar)
    vol_mean = _roll_mean(volume, 20)
    vol_std = _roll_std(volume, 20)
    with np.errstate(invalid='ignore', divide='ignore'):
        vz = np.where(vol_std > 0, (volume - vol_mean) / vol_std, 0.0)
    F[:, 18] = np.clip(vz, -3, 3) / 3.0  # normalize to [-1, 1]

    # 19: atr_normalized_ofi (ofi relative to volatility)
    atr_pct = np.where(close > 0, atr_arr / close, 0.01)
    with np.errstate(invalid='ignore', divide='ignore'):
        atr_norm = np.where(atr_pct > 0, ofi_score / (atr_pct * 100), 0.0)
    F[:, 19] = np.clip(atr_norm, -1, 1)

    # 20-21: hour_sin, hour_cos (cyclical time-of-day)
    if timestamps_ms is not None and len(timestamps_ms) == n:
        hours = (timestamps_ms / 1000 / 3600) % 24  # UTC hour as float
        F[:, 20] = np.sin(2 * np.pi * hours / 24)
        F[:, 21] = np.cos(2 * np.pi * hours / 24)
    else:
        F[:, 20] = 0.0; F[:, 21] = 0.0

    # 22-23: dow_sin, dow_cos (cyclical day-of-week)
    if timestamps_ms is not None and len(timestamps_ms) == n:
        # Unix epoch (Jan 1 1970) was a Thursday (day 3), so (days + 3) % 7
        days = timestamps_ms / 1000 / 86400
        dow = (days + 3) % 7  # 0=Mon, 6=Sun
        F[:, 22] = np.sin(2 * np.pi * dow / 7)
        F[:, 23] = np.cos(2 * np.pi * dow / 7)
    else:
        F[:, 22] = 0.0; F[:, 23] = 0.0

    # 24: btc_return_60m (cross-market BTC momentum)
    if btc_close is not None and len(btc_close) == n:
        btc_ret = np.full(n, np.nan)
        btc_ret[60:] = (btc_close[60:] - btc_close[:-60]) / np.maximum(btc_close[:-60], 1e-10)
        F[:, 24] = np.clip(btc_ret * 10, -1, 1)  # scale: 10% BTC move → 1.0
    else:
        F[:, 24] = 0.0

    return F, atr_arr


# ── Regime-Conditioned Labels ────────────────────────────────────────────

def label_bars_v2(close, high, low, atr_arr, hurst_arr, sample_idx, fwd=60):
    """Triple-barrier with regime-conditioned TP/SL multipliers."""
    labels = np.full(len(sample_idx), np.nan)
    n = len(close)

    for k, i in enumerate(sample_idx):
        atr = atr_arr[i]
        if np.isnan(atr) or atr <= 0: continue
        entry = close[i]
        end = min(i + fwd, n - 1)
        if end <= i: continue

        # Regime-conditioned barriers
        h = hurst_arr[i] if not np.isnan(hurst_arr[i]) else 0.5
        if h > 0.55:    # trending → let winners run
            tp_mult, sl_mult = 2.0, 1.0
        elif h < 0.45:  # ranging → tight TP
            tp_mult, sl_mult = 1.0, 1.0
        else:            # neutral
            tp_mult, sl_mult = 1.5, 1.0

        tp = entry + tp_mult * atr
        sl = entry - sl_mult * atr
        future_high = high[i+1:end+1]
        future_low = low[i+1:end+1]
        tp_hit = np.where(future_high >= tp)[0]
        sl_hit = np.where(future_low <= sl)[0]
        tp_t = tp_hit[0] if len(tp_hit) > 0 else np.inf
        sl_t = sl_hit[0] if len(sl_hit) > 0 else np.inf

        if tp_t == np.inf and sl_t == np.inf:
            labels[k] = 1 if close[end] > entry else 0
        elif tp_t <= sl_t:
            labels[k] = 1
        else:
            labels[k] = 0
    return labels


# ── Load BTC data (shared across all coins) ─────────────────────────────

def load_btc_close():
    """Load all BTC parquets, return dict of {month: close_array}."""
    pattern = os.path.join(PROCESSED_DIR, "BTCUSDT_1m_features_*.parquet")
    files = sorted(glob.glob(pattern))
    # Exclude 2025 and 2026 for now so they remain completely unseen for later testing
    files = [f for f in files if "2025-" not in f and "2026-" not in f]
    all_dfs = []
    for f in files:
        try: all_dfs.append(pl.read_parquet(f))
        except: pass
    if not all_dfs: return None, None
    df = pl.concat(all_dfs).sort("timestamp")
    ts = df["timestamp"].dt.cast_time_unit("ms").cast(pl.Int64).to_numpy()
    close = df["close"].to_numpy().astype(np.float64)
    return ts, close


# ── Per-coin processor ───────────────────────────────────────────────────

def process_coin(symbol, btc_ts, btc_close):
    pattern = os.path.join(PROCESSED_DIR, f"{symbol}_1m_features_*.parquet")
    files = sorted(glob.glob(pattern))
    if not files: return np.empty((0, 25)), np.empty(0)

    dfs = []
    for f in files:
        try: dfs.append(pl.read_parquet(f))
        except: pass
    if not dfs: return np.empty((0, 25)), np.empty(0)

    df = pl.concat(dfs).sort("timestamp")
    close = df["close"].to_numpy().astype(np.float64)
    high = df["high"].to_numpy().astype(np.float64)
    low = df["low"].to_numpy().astype(np.float64)
    volume = df["volume"].to_numpy().astype(np.float64)
    ofi_raw = df["ofi"].to_numpy().astype(np.float64)
    ts_ms = df["timestamp"].cast(pl.Int64).to_numpy().astype(np.float64)
    n = len(close)
    print(f"  {symbol}: {n:,} bars across {len(files)} months", flush=True)

    # Align BTC close to this coin's timestamps via nearest-neighbor
    btc_aligned = None
    if btc_ts is not None and btc_close is not None and symbol != "BTCUSDT":
        btc_aligned = np.interp(ts_ms, btc_ts.astype(np.float64), btc_close)

    F, atr_arr = build_features_v2(close, high, low, volume, ofi_raw, ts_ms, btc_aligned)

    # Hurst array for regime-conditioned labels (reuse from feature col 10)
    hurst_raw = F[:, 10].copy()

    # Sample indices
    idx = np.arange(WARMUP, n - FORWARD_BARS, SAMPLE_EVERY)
    F_samp = F[idx]
    valid = ~np.any(np.isnan(F_samp), axis=1)
    idx_v = idx[valid]
    F_v = F_samp[valid]
    if len(idx_v) == 0:
        print(f"  {symbol}: no valid rows"); return np.empty((0,25)), np.empty(0)

    y = label_bars_v2(close, high, low, atr_arr, hurst_raw, idx_v, fwd=FORWARD_BARS)
    labelled = ~np.isnan(y)
    X_out, y_out = F_v[labelled], y[labelled]
    wr = y_out.mean() if len(y_out) > 0 else 0
    print(f"  {symbol}: {len(X_out):,} samples | win-rate {wr:.1%}", flush=True)
    return X_out, y_out


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print("=" * 60)
    print("EV Model V2 Training — XGBoost + Expanded Features")
    print("=" * 60)

    # Load BTC data for cross-market feature
    print("\nLoading BTC reference data...")
    btc_ts, btc_close = load_btc_close()
    if btc_ts is not None:
        print(f"  BTC reference: {len(btc_ts):,} bars loaded")
    else:
        print("  WARNING: No BTC data found, btc_return_60m will be zeroed")

    # Discover coins
    all_files = glob.glob(os.path.join(PROCESSED_DIR, "*.parquet"))
    symbols = sorted({os.path.basename(f).split("_1m_")[0] for f in all_files})
    print(f"\nCoins: {len(symbols)}")

    all_X, all_y = [], []
    for sym in symbols:
        print(f"\n[{sym}]")
        X, y = process_coin(sym, btc_ts, btc_close)
        if len(X) > 0:
            all_X.append(X); all_y.append(y)

    if not all_X:
        print("ERROR: no data"); return

    X = np.vstack(all_X)
    y = np.concatenate(all_y)
    print(f"\n{'='*60}")
    print(f"Total samples: {len(X):,}  |  Features: {X.shape[1]}")
    print(f"Win-rate: {y.mean():.1%}")
    print(f"{'='*60}")

    # Walk-forward split (80/20)
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Train XGBoost
    print("\nTraining XGBoost...")
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=FEATURE_NAMES_V2)
    dtest = xgb.DMatrix(X_test, label=y_test, feature_names=FEATURE_NAMES_V2)

    params = {
        "objective": "binary:logistic",
        "eval_metric": ["logloss", "auc"],
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 50,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "tree_method": "hist",  # fast on CPU
        "nthread": 4,
        "seed": 42,
        "verbosity": 1,
    }

    model = xgb.train(
        params, dtrain,
        num_boost_round=300,
        evals=[(dtrain, "train"), (dtest, "test")],
        early_stopping_rounds=30,
        verbose_eval=50,
    )

    # Evaluate
    p_test = model.predict(dtest)
    preds = (p_test >= 0.5).astype(int)
    accuracy = float(np.mean(preds == y_test))

    from scipy.stats import spearmanr
    ic, _ = spearmanr(p_test, y_test)

    # Also train logistic for comparison
    print("\nTraining Logistic Regression (comparison)...")
    from ev_model import LogisticEVModel, FEATURE_NAMES
    # Pad/trim features to match original 18 if needed
    X_train_lr = X_train[:, :18]
    X_test_lr = X_test[:, :18]
    lr_model = LogisticEVModel()
    lr_result = lr_model.fit(X_train_lr, y_train, learning_rate=0.01, n_epochs=200)
    lr_ic = lr_result.get("information_coefficient", 0)

    print(f"\n{'='*60}")
    print(f"RESULTS COMPARISON")
    print(f"{'='*60}")
    print(f"{'Metric':<30} {'Logistic (v1)':<18} {'XGBoost (v2)':<18}")
    print(f"{'-'*66}")
    print(f"{'OOS Accuracy':<30} {lr_result.get('out_of_sample_accuracy',0):.1%}{'':<13} {accuracy:.1%}")
    print(f"{'Information Coefficient':<30} {lr_ic:.4f}{'':<13} {ic:.4f}")
    print(f"{'Features':<30} {'18':<18} {'25':<18}")
    print(f"{'Model Type':<30} {'Linear':<18} {'Non-linear':<18}")

    improvement = ((ic - lr_ic) / max(abs(lr_ic), 0.001)) * 100
    print(f"\nIC improvement: {improvement:+.0f}%")

    # Feature importance
    importance = model.get_score(importance_type='gain')
    sorted_imp = sorted(importance.items(), key=lambda x: -x[1])
    print(f"\nTop 10 Feature Importance (gain):")
    for feat, gain in sorted_imp[:10]:
        bar = "#" * min(40, int(gain / max(1, sorted_imp[0][1]) * 40))
        print(f"  {feat:<28} {gain:>10.1f}  {bar}")

    # Save XGBoost model
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    model.save_model(MODEL_PATH)

    # Save metadata
    meta = {
        "model_type": "xgboost",
        "feature_names": FEATURE_NAMES_V2,
        "n_features": len(FEATURE_NAMES_V2),
        "training_coins": symbols,
        "total_samples": int(len(X)),
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "oos_accuracy": round(accuracy, 4),
        "information_coefficient": round(float(ic), 4),
        "lr_ic_comparison": round(float(lr_ic), 4),
        "best_iteration": model.best_iteration,
        "feature_importance_top10": {k: round(v, 2) for k, v in sorted_imp[:10]},
        "unavailable_features_zeroed": [
            "large_trade_ofi", "depth_imbalance", "fear_greed_norm"
        ],
        "regime_conditioned_labels": True,
        "forward_bars": FORWARD_BARS,
        "sample_every": SAMPLE_EVERY,
    }
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)

    elapsed = time.time() - t0
    print(f"\n[OK] XGBoost model saved -> {MODEL_PATH}")
    print(f"[OK] Metadata saved -> {META_PATH}")
    print(f"  Training time: {elapsed/60:.1f} minutes")

if __name__ == "__main__":
    main()
