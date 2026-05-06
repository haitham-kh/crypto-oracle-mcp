#!/usr/bin/env python3
"""
train_ev_model.py
=================
Calibrates the LogisticEVModel from the 40-month 1-minute OHLCV+OFI dataset.

Pipeline:
  1. Load all parquet files per coin (polars, fast)
  2. Compute the 18 ev_model features via vectorised numpy
  3. Label each sampled bar with a simplified triple-barrier method
  4. Fit LogisticEVModel (walk-forward 80/20 split)
  5. Save calibrated weights → data/ev_model_weights.json

Run from the crypto-oracle-mcp directory:
    python train_ev_model.py

Features derived from parquet columns (timestamp, open, high, low, close, volume, ofi):
  Available (14):  ofi_score, ofi_acceleration, absorption_index,
                   cvd_trend_15m/1h/4h, cvd_divergence_15m/1h/4h,
                   hurst, trend_efficiency, persistence_score,
                   vol_state, accum_probability, distrib_probability
  Unavailable (3): large_trade_ofi, depth_imbalance, fear_greed_norm → 0.0 (neutral)
  Model learns zero weight for unavailable features during training;
  they are populated normally by the live oracle at inference time.
"""
from __future__ import annotations

import os
import glob
import time
import json
import numpy as np
import polars as pl
from typing import Optional

# ── paths ──────────────────────────────────────────────────────────────────
PROCESSED_DIR = r"E:\training data for quant\processed_features"
WEIGHTS_PATH  = os.path.join(os.path.dirname(__file__), "data", "ev_model_weights.json")

# ── config ─────────────────────────────────────────────────────────────────
SAMPLE_EVERY  = 10    # sample 1 row per 10 minutes → manageable matrix size
WARMUP        = 250   # skip first N bars per coin (feature warm-up)
FORWARD_BARS  = 60    # triple-barrier horizon (60 min)
TP_MULT       = 1.5   # profit-take = TP_MULT × ATR
SL_MULT       = 1.0   # stop-loss   = SL_MULT  × ATR
ATR_PERIOD    = 14


# ─────────────────────────────────────────────────────────────────────────────
# Vectorised helpers
# ─────────────────────────────────────────────────────────────────────────────

def _roll_mean(x: np.ndarray, w: int) -> np.ndarray:
    """O(n) rolling mean."""
    out = np.full(len(x), np.nan)
    cs  = np.cumsum(np.nan_to_num(x, nan=0.0))
    out[w - 1:] = (cs[w - 1:] - np.concatenate([[0], cs[:-w]])) / w
    return out


def _atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, p: int = 14) -> np.ndarray:
    """Wilder ATR."""
    n  = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    tr[1:] = np.maximum.reduce([
        high[1:] - low[1:],
        np.abs(high[1:] - close[:-1]),
        np.abs(low[1:]  - close[:-1]),
    ])
    atr = np.full(n, np.nan)
    atr[p - 1] = tr[:p].mean()
    k = (p - 1) / p
    for i in range(p, n):
        atr[i] = atr[i - 1] * k + tr[i] / p
    return atr


def _hurst(prices: np.ndarray) -> Optional[float]:
    """R/S Hurst exponent on a price sub-series."""
    if len(prices) < 50:
        return None
    rets = np.diff(np.log(prices + 1e-10))
    n    = len(rets)
    lags, rs_vals = [], []
    for lag in [max(10, n // 8), max(10, n // 4), max(10, n // 2)]:
        if lag >= n:
            continue
        sub  = rets[-lag:]
        mean = sub.mean()
        dev  = np.cumsum(sub - mean)
        R    = dev.max() - dev.min()
        S    = sub.std(ddof=1)
        if S > 0 and R > 0:
            lags.append(np.log(lag))
            rs_vals.append(np.log(R / S))
    if len(rs_vals) < 2:
        return None
    return float(np.clip(np.polyfit(lags, rs_vals, 1)[0], 0.01, 0.99))


# ─────────────────────────────────────────────────────────────────────────────
# Feature matrix builder
# ─────────────────────────────────────────────────────────────────────────────

def build_features(close: np.ndarray, high: np.ndarray, low: np.ndarray,
                   volume: np.ndarray, ofi: np.ndarray) -> np.ndarray:
    """Return (n, 18) feature matrix matching ev_model.FEATURE_NAMES order."""
    n = len(close)
    F = np.full((n, 18), np.nan)

    # ── 0  ofi_score ─────────────────────────────────────────────────────────
    vol_safe  = np.where(volume > 0, volume, 1.0)
    ofi_score = np.clip(ofi / vol_safe, -1.0, 1.0)
    F[:, 0]   = ofi_score

    # ── 1  ofi_acceleration ───────────────────────────────────────────────────
    accel     = np.full(n, np.nan)
    accel[5:] = np.clip(ofi_score[5:] - ofi_score[:-5], -1.0, 1.0)
    F[:, 1]   = accel

    # ── 2  large_trade_ofi  (unavailable) ────────────────────────────────────
    F[:, 2] = 0.0

    # ── 3  absorption_index ───────────────────────────────────────────────────
    rng      = np.maximum(high - low, 1e-10)
    vol_ma   = _roll_mean(volume, 20)
    rng_ma   = _roll_mean(rng,    20)
    with np.errstate(invalid='ignore', divide='ignore'):
        abs_score = np.where(
            (vol_ma > 0) & (rng_ma > 0),
            (volume / vol_ma) / np.maximum(rng / rng_ma, 0.1),
            1.0)
    low20  = np.array([close[max(0, i - 19):i + 1].min() for i in range(n)])
    high20 = np.array([close[max(0, i - 19):i + 1].max() for i in range(n)])
    rng20  = np.maximum(high20 - low20, 1e-10)
    ppos   = (close - low20) / rng20
    sign   = np.where(ppos < 0.35, 1.0, np.where(ppos > 0.65, -1.0, 0.0))
    F[:, 3] = np.clip((abs_score - 1.0) / 2.0, 0.0, 1.0) * sign

    # ── 4-6  cvd_trend 15m / 1h / 4h ─────────────────────────────────────────
    cvd = np.cumsum(ofi)
    for col, w in zip([4, 5, 6], [15, 60, 240]):
        trend       = np.full(n, np.nan)
        past_cvd    = np.full(n, np.nan)
        past_cvd[w:] = cvd[:-w]
        denom       = np.where(np.abs(cvd) > 0, np.abs(cvd), 1.0)
        trend[w:]   = np.clip((cvd[w:] - past_cvd[w:]) / denom[w:], -1.0, 1.0)
        F[:, col]   = trend

    # ── 7-9  cvd_divergence 15m / 1h / 4h ────────────────────────────────────
    for col, w in zip([7, 8, 9], [15, 60, 240]):
        past_p    = np.full(n, np.nan)
        past_c    = np.full(n, np.nan)
        past_p[w:] = close[:-w]
        past_c[w:] = cvd[:-w]
        p_chg      = close - past_p
        c_chg      = cvd   - past_c
        div        = np.where(
            (p_chg < 0) & (c_chg > 0),  0.7,   # bullish accumulation
            np.where(
                (p_chg > 0) & (c_chg < 0), -0.7,  # bearish distribution
                0.0))
        div[:w]    = np.nan
        F[:, col]  = div

    # ── 10  hurst  (computed every 60 bars, forward-filled) ──────────────────
    hurst_arr = np.full(n, np.nan)
    for i in range(200, n, 60):
        h = _hurst(close[max(0, i - 200): i + 1])
        if h is not None:
            hurst_arr[i] = (h - 0.5) * 2.0      # normalise to [-1, +1]
    # forward fill
    last = np.nan
    for i in range(n):
        if not np.isnan(hurst_arr[i]):
            last = hurst_arr[i]
        elif not np.isnan(last):
            hurst_arr[i] = last
    F[:, 10] = hurst_arr

    # ── 11  trend_efficiency (Kaufman ER, window=14) ─────────────────────────
    te        = np.full(n, np.nan)
    diff_abs  = np.abs(np.diff(close, prepend=close[0]))   # shape (n,)
    path_cs   = np.cumsum(diff_abs)                        # shape (n,)
    # 14-bar path = path_cs[i] - path_cs[i-14];  pad with zeros for first 14
    path_lag  = np.concatenate([np.zeros(14), path_cs[:-14]])  # shape (n,)
    path_14   = path_cs - path_lag                             # shape (n,)
    # 14-bar net change
    close_lag = np.concatenate([np.full(14, close[0]), close[:-14]])  # shape (n,)
    net_14    = np.abs(close - close_lag)                              # shape (n,)
    with np.errstate(invalid='ignore', divide='ignore'):
        er    = np.where(path_14 > 0, net_14 / path_14, 0.0)
    te[14:]   = np.clip((er[14:] - 0.5) * 2.0, -1.0, 1.0)
    F[:, 11]  = te

    # ── 12  persistence_score (directional OFI consistency, 20 bars) ─────────
    ofi_dir    = np.sign(ofi_score)
    pos_cs     = np.cumsum((ofi_dir > 0).astype(float))
    pos_roll20 = pos_cs - np.concatenate([np.zeros(20), pos_cs[:-20]])
    persist    = np.full(n, np.nan)
    persist[20:] = np.clip((pos_roll20[20:] / 20.0 - 0.5) * 2.0, -1.0, 1.0)
    F[:, 12]   = persist

    # ── 13  depth_imbalance  (unavailable) ───────────────────────────────────
    F[:, 13] = 0.0

    # ── 14  vol_state  (recent 5-bar vol vs prior 20-bar vol) ────────────────
    log_ret        = np.full(n, 0.0)
    log_ret[1:]    = np.diff(np.log(np.maximum(close, 1e-10)))
    ret_sq         = log_ret ** 2
    recent_var     = _roll_mean(ret_sq,  5)
    prior_var      = _roll_mean(ret_sq, 25)
    with np.errstate(invalid='ignore', divide='ignore'):
        vol_ratio  = np.where(prior_var > 0, recent_var / prior_var, 1.0)
    vol_state      = np.where(vol_ratio > 1.5, 1.0, np.where(vol_ratio < 0.5, -1.0, 0.0))
    vol_state[:25] = np.nan
    F[:, 14]       = vol_state

    # ── 15  fear_greed_norm  (unavailable) ───────────────────────────────────
    F[:, 15] = 0.0

    # ── 16  accum_probability (60-bar CVD vs price) ───────────────────────────
    past_p60       = np.full(n, np.nan);  past_p60[60:] = close[:-60]
    past_c60       = np.full(n, np.nan);  past_c60[60:] = cvd[:-60]
    p60            = close - past_p60
    c60            = cvd   - past_c60
    accum          = np.where((p60 < 0) & (c60 > 0),  0.7,
                    np.where((p60 > 0) & (c60 < 0), -0.5, 0.0))
    accum[:60]     = np.nan
    F[:, 16]       = accum

    # ── 17  distrib_probability (inverse: -accum_probability) ────────────────
    F[:, 17] = -accum

    return F


# ─────────────────────────────────────────────────────────────────────────────
# Triple-barrier labeller  (vectorised)
# ─────────────────────────────────────────────────────────────────────────────

def label_bars(close: np.ndarray, high: np.ndarray, low: np.ndarray,
               atr_arr: np.ndarray, sample_idx: np.ndarray,
               fwd: int = 60, tp_mult: float = 1.5, sl_mult: float = 1.0) -> np.ndarray:
    """Return binary labels (1=win, 0=loss) for sampled indices."""
    labels = np.full(len(sample_idx), np.nan)
    n      = len(close)

    for k, i in enumerate(sample_idx):
        atr = atr_arr[i]
        if np.isnan(atr) or atr <= 0:
            continue
        entry = close[i]
        tp    = entry + tp_mult * atr
        sl    = entry - sl_mult * atr
        end   = min(i + fwd, n - 1)
        if end <= i:
            continue
        future_high = high[i + 1: end + 1]
        future_low  = low[i  + 1: end + 1]
        tp_hit = np.where(future_high >= tp)[0]
        sl_hit = np.where(future_low  <= sl)[0]
        tp_t   = tp_hit[0] if len(tp_hit) > 0 else np.inf
        sl_t   = sl_hit[0] if len(sl_hit) > 0 else np.inf

        if tp_t == np.inf and sl_t == np.inf:
            labels[k] = 1 if close[end] > entry else 0
        elif tp_t <= sl_t:
            labels[k] = 1
        else:
            labels[k] = 0

    return labels


# ─────────────────────────────────────────────────────────────────────────────
# Per-coin processor
# ─────────────────────────────────────────────────────────────────────────────

def process_coin(symbol: str) -> tuple[np.ndarray, np.ndarray]:
    """Load all months for one coin, compute features + labels. Returns (X, y)."""
    pattern = os.path.join(PROCESSED_DIR, f"{symbol}_1m_features_*.parquet")
    files   = sorted(glob.glob(pattern))
    if not files:
        return np.empty((0, 18)), np.empty(0)

    dfs = []
    for f in files:
        try:
            dfs.append(pl.read_parquet(f))
        except Exception as e:
            print(f"  WARN: skip {os.path.basename(f)} — {e}")

    if not dfs:
        return np.empty((0, 18)), np.empty(0)

    df = pl.concat(dfs).sort("timestamp")

    close  = df["close"].to_numpy().astype(np.float64)
    high   = df["high"].to_numpy().astype(np.float64)
    low    = df["low"].to_numpy().astype(np.float64)
    volume = df["volume"].to_numpy().astype(np.float64)
    ofi    = df["ofi"].to_numpy().astype(np.float64)
    n      = len(close)

    print(f"  {symbol}: {n:,} bars across {len(files)} months", flush=True)

    # Build feature matrix
    F   = build_features(close, high, low, volume, ofi)
    atr = _atr(high, low, close, ATR_PERIOD)

    # Sample indices (skip warmup, step every N)
    idx = np.arange(WARMUP, n - FORWARD_BARS, SAMPLE_EVERY)

    # Drop rows where ANY feature is NaN
    F_samp   = F[idx]
    valid    = ~np.any(np.isnan(F_samp), axis=1)
    idx_v    = idx[valid]
    F_v      = F_samp[valid]

    if len(idx_v) == 0:
        print(f"  {symbol}: no valid rows after NaN filter")
        return np.empty((0, 18)), np.empty(0)

    # Label
    y = label_bars(close, high, low, atr, idx_v,
                   fwd=FORWARD_BARS, tp_mult=TP_MULT, sl_mult=SL_MULT)

    # Drop unlabelled rows
    labelled = ~np.isnan(y)
    X_out    = F_v[labelled]
    y_out    = y[labelled]

    pos_rate = y_out.mean() if len(y_out) > 0 else 0
    print(f"  {symbol}: {len(X_out):,} labelled samples | win-rate {pos_rate:.1%}", flush=True)
    return X_out, y_out


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print("=" * 60)
    print("EV Model Training — reading parquet dataset")
    print(f"Processed dir : {PROCESSED_DIR}")
    print(f"Sample every  : {SAMPLE_EVERY} bars")
    print(f"Forward horizon: {FORWARD_BARS} bars (minutes)")
    print("=" * 60)

    # Discover coins
    all_files = glob.glob(os.path.join(PROCESSED_DIR, "*.parquet"))
    symbols   = sorted({os.path.basename(f).split("_1m_")[0] for f in all_files})
    print(f"\nCoins found: {len(symbols)}")
    for s in symbols:
        print(f"  {s}")

    # Collect X, y across all coins
    all_X, all_y = [], []
    for sym in symbols:
        print(f"\n[{sym}]")
        X, y = process_coin(sym)
        if len(X) > 0:
            all_X.append(X)
            all_y.append(y)

    if not all_X:
        print("\nERROR: no data collected. Check PROCESSED_DIR path.")
        return

    X = np.vstack(all_X)
    y = np.concatenate(all_y)

    print(f"\n{'=' * 60}")
    print(f"Total samples : {len(X):,}")
    print(f"Feature shape : {X.shape}")
    print(f"Overall win-rate: {y.mean():.1%}  (ideal ≈ 40-60%)")
    print(f"{'=' * 60}")

    # Fit model
    print("\nFitting LogisticEVModel ...")
    from ev_model import LogisticEVModel, FEATURE_NAMES

    assert X.shape[1] == len(FEATURE_NAMES), \
        f"Feature count mismatch: got {X.shape[1]}, expected {len(FEATURE_NAMES)}"

    model  = LogisticEVModel()
    result = model.fit(X, y, learning_rate=0.01, n_epochs=200, l2_lambda=0.01)

    if "error" in result:
        print(f"Training error: {result['error']}")
        return

    print(f"\nTraining complete:")
    print(f"  Train samples      : {result['train_samples']:,}")
    print(f"  Test samples       : {result['test_samples']:,}")
    print(f"  OOS accuracy       : {result['out_of_sample_accuracy']:.1%}")
    print(f"  Information Coeff  : {result['information_coefficient']:.4f}  (>0.05 = useful)")
    print(f"  Final loss         : {result['final_loss']:.6f}")

    # Top features by weight
    weights_sorted = sorted(model.weights.items(), key=lambda x: abs(x[1]), reverse=True)
    print("\n  Top 10 feature weights:")
    for feat, w in weights_sorted[:10]:
        bar = "█" * int(abs(w) * 20)
        sign = "+" if w >= 0 else "-"
        print(f"    {feat:<28} {sign}{abs(w):.4f}  {bar}")

    # Save weights
    os.makedirs(os.path.dirname(WEIGHTS_PATH), exist_ok=True)
    model.save_weights(WEIGHTS_PATH, metadata={
        "training_coins"     : symbols,
        "total_samples"      : int(len(X)),
        "oos_accuracy"       : result["out_of_sample_accuracy"],
        "information_coeff"  : result["information_coefficient"],
        "sample_every_n_bars": SAMPLE_EVERY,
        "forward_bars"       : FORWARD_BARS,
        "tp_mult"            : TP_MULT,
        "sl_mult"            : SL_MULT,
        "unavailable_features_set_to_zero": [
            "large_trade_ofi", "depth_imbalance", "fear_greed_norm"
        ],
    })

    elapsed = time.time() - t0
    print(f"\n✓ Weights saved → {WEIGHTS_PATH}")
    print(f"  Training time: {elapsed / 60:.1f} minutes")
    print("\nThe oracle will load these weights automatically on next start.")
    print("Run the oracle server and look for 'CALIBRATED MODEL' in its output.\n")


if __name__ == "__main__":
    main()
