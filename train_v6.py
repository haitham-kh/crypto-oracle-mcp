"""
train_v6.py — V6 Model Training Pipeline
==========================================
Trains TWO complementary model sets from the V5 + V6 feature stack:

  V6_full  (130 features = 94 V5 + 36 V6):
      Primary model. Replaces V5 in the signal chain.
      Trained with sample-weighting that amplifies rows where V6 signals
      are in strong agreement (~2.33x boost), giving V6 signals ~70% effective
      learning weight while V5 features remain as context.

  V6_micro (36 features = V6 only):
      Gate model. Used as a secondary confirmation signal.
      A trade is only taken if both V6_full AND V6_micro agree directionally.

Architecture at inference (live engine):
    p_v5       = V5_clf(X_94)              # existing V5 models
    p_v6_full  = V6_full_clf(X_130)        # NEW: full-stack
    p_final    = 0.30 * p_v5 + 0.70 * p_v6_full   # user-specified 70/30 blend

Outputs:
    data/v6_full_clf_{long,short}_h{60,720}.json
    data/v6_full_calib_{long,short}_h{60,720}.pkl
    data/v6_full_reg_h{60,720}.json
    data/v6_micro_clf_{long,short}_h{60,720}.json
    data/v6_micro_calib_{long,short}_h{60,720}.pkl
    data/v6_meta.json

Prerequisites:
    Run the Colab unified pipeline first to generate processed training parquets
    in:  <ROOT>/processed/<symbol>_training_data.parquet
    Then run this script locally (or on Colab with GPU).

    python train_v6.py [--processed-dir /path/to/processed]
"""
from __future__ import annotations
import os, sys, glob, json, time, pickle, argparse
import numpy as np
import polars as pl
import xgboost as xgb

sys.path.insert(0, os.path.dirname(__file__))

from features_v6 import (
    FEATURE_NAMES_V6, FEATURE_NAMES_V6_EXTRA, N_V6_EXTRA,
    build_v6_features, compute_v6_signal_strength
)
from features_v5 import FEATURE_NAMES_V5
from train_ev_model_v2 import build_features_v2
from trading_config import (
    HORIZONS, ROUND_TRIP_COST, WARMUP_BARS, SAMPLE_EVERY,
    MIN_EV_PCT, MIN_P_UP_DEFAULT,
)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

DATA_DIR   = os.path.join(os.path.dirname(__file__), "data")
PERP_DIR   = os.path.join(DATA_DIR, "perp")
BASKET_FP  = os.path.join(DATA_DIR, "basket", "basket_state.parquet")
META_PATH  = os.path.join(DATA_DIR, "v6_meta.json")

# Default processed dir (Colab output). Override with --processed-dir CLI arg.
DEFAULT_PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

COIN_ONEHOT_NAMES = [
    "AAVEUSDT", "ADAUSDT", "AVAXUSDT", "BNBUSDT", "BTCUSDT",
    "DOGEUSDT", "ETHUSDT", "LINKUSDT", "PEPEUSDT", "SHIBUSDT",
    "SOLUSDT",  "UNIUSDT", "WIFUSDT",  "XRPUSDT",
]

N_V5        = len(FEATURE_NAMES_V5)          # 80
N_ONEHOT    = len(COIN_ONEHOT_NAMES)         # 14
N_V5_FULL   = N_V5 + N_ONEHOT               # 94  (matches existing V5 stack)
N_V6_FULL   = N_V5_FULL + N_V6_EXTRA        # 130

ALL_FEATURE_NAMES_V6 = FEATURE_NAMES_V6 + [f"coin_is_{s}" for s in COIN_ONEHOT_NAMES]

# ─────────────────────────────────────────────────────────────────────────────
# XGBoost hyperparameters
# (slightly deeper than V5 to exploit the richer 130-feature stack)
# ─────────────────────────────────────────────────────────────────────────────

XGB_CLF_PARAMS = {
    "objective":        "binary:logistic",
    "eval_metric":      ["logloss", "auc"],
    "max_depth":        7,
    "learning_rate":    0.030,
    "subsample":        0.80,
    "colsample_bytree": 0.65,   # slight reduction forces feature diversity
    "colsample_bylevel":0.80,
    "min_child_weight": 80,
    "reg_alpha":        0.20,
    "reg_lambda":       1.5,
    "tree_method":      "hist",
    "nthread":          4,
    "seed":             42,
    "verbosity":        1,
}

XGB_REG_PARAMS = {
    "objective":        "reg:squarederror",
    "eval_metric":      "rmse",
    "max_depth":        7,
    "learning_rate":    0.030,
    "subsample":        0.80,
    "colsample_bytree": 0.65,
    "colsample_bylevel":0.80,
    "min_child_weight": 80,
    "reg_alpha":        0.20,
    "reg_lambda":       1.5,
    "tree_method":      "hist",
    "nthread":          4,
    "seed":             43,
    "verbosity":        0,
}

XGB_MICRO_PARAMS = {
    "objective":        "binary:logistic",
    "eval_metric":      ["logloss", "auc"],
    "max_depth":        5,
    "learning_rate":    0.035,
    "subsample":        0.80,
    "colsample_bytree": 0.80,
    "min_child_weight": 60,
    "reg_alpha":        0.10,
    "reg_lambda":       1.0,
    "tree_method":      "hist",
    "nthread":          4,
    "seed":             44,
    "verbosity":        0,
}

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _coin_onehot(symbol, n):
    block = np.zeros((n, N_ONEHOT), dtype=np.float32)
    if symbol in COIN_ONEHOT_NAMES:
        block[:, COIN_ONEHOT_NAMES.index(symbol)] = 1.0
    return block


def recency_weights(ts, anchor_ts, half_life_months=12):
    age_m = np.maximum((anchor_ts - ts) / (30.44 * 24 * 3600 * 1000), 0)
    lam = np.log(2) / half_life_months
    w = np.exp(-lam * age_m)
    return w / w.mean()


def pick_threshold(p_cal, net_pct, min_trades):
    grid = np.round(np.arange(0.50, 0.81, 0.01), 2)
    best = {"threshold": MIN_P_UP_DEFAULT, "ev_pct": -1e9, "n": 0,
            "win_rate": 0.0, "profit_factor": 0.0}
    rows = []
    for thr in grid:
        m = p_cal >= thr
        n = int(m.sum())
        if n < min_trades:
            rows.append((float(thr), n, 0.0, 0.0, 0.0))
            continue
        nets = net_pct[m]
        ev   = float(np.nanmean(nets))
        wr   = float((nets > 0).mean())
        wins   = nets[nets > 0].sum()
        losses = -nets[nets <= 0].sum()
        pf = float(wins / max(losses, 1e-9))
        rows.append((float(thr), n, ev * 100, wr * 100, pf))
        if ev > best["ev_pct"]:
            best = {"threshold": float(thr), "ev_pct": ev * 100,
                    "n": n, "win_rate": wr * 100, "profit_factor": pf}
    return best, rows


# ─────────────────────────────────────────────────────────────────────────────
# Per-coin data loading — two modes
# ─────────────────────────────────────────────────────────────────────────────

def _load_from_colab_parquet(symbol, processed_dir):
    """
    Load from Colab-generated training_data.parquet.
    These contain columns: timestamp_ms, X_0..X_93, y_60, net_60, y_short_60,
    net_short_60, ret_60, y_720, net_720, ..., any_valid.
    We extract X_0..X_93 (V5 features already computed) and recompute V6 on top.
    """
    fp = os.path.join(processed_dir, f"{symbol}_training_data.parquet")
    if not os.path.exists(fp):
        return None
    try:
        df = pl.read_parquet(fp)
    except Exception as e:
        print(f"  [{symbol}] Failed to read {fp}: {e}")
        return None

    # Extract pre-computed V5 features
    x_cols = [c for c in df.columns if c.startswith("X_")]
    if len(x_cols) < N_V5_FULL:
        print(f"  [{symbol}] Only {len(x_cols)} X columns, expected >= {N_V5_FULL}. Skipping.")
        return None

    x_cols_sorted = sorted(x_cols, key=lambda c: int(c.split("_")[1]))
    X_v5_full = df.select(x_cols_sorted[:N_V5_FULL]).to_numpy().astype(np.float32)
    ts = df["timestamp_ms"].to_numpy().astype(np.float64)

    label_data = {}
    for h in HORIZONS:
        for k in [f"y_{h}", f"net_{h}", f"y_short_{h}", f"net_short_{h}", f"ret_{h}"]:
            if k in df.columns:
                label_data[k] = df[k].to_numpy().astype(np.float32)
            else:
                label_data[k] = np.full(len(df), np.nan, dtype=np.float32)

    any_valid = df["any_valid"].to_numpy().astype(bool) if "any_valid" in df.columns else np.ones(len(df), dtype=bool)

    print(f"  [{symbol}] {len(df):,} samples from Colab parquet")
    ohlcv_path = os.path.join(processed_dir, f"{symbol}_1m_ohlcv.parquet")
    return {
        "X_v5_full":   X_v5_full,  # (n, 94)
        "ts":          ts,
        "labels":      label_data,
        "any_valid":   any_valid,
        "n":           len(df),
        "_symbol":     symbol,
        "_ohlcv_path": ohlcv_path,  # companion file from download_v6_ohlcv.py
    }


def _load_from_processed_features(symbol, processed_dir, btc_ts, btc_close, basket_state, perp_dir):
    """
    Load from raw monthly feature parquets (E-drive format from train_v5.py).
    Recomputes full V5 features on the fly.
    """
    from features_v5 import build_v5_full
    from labels_multi_horizon import label_multi_horizon

    pattern = os.path.join(processed_dir, f"{symbol}_1m_features_*.parquet")
    files = sorted(glob.glob(pattern))
    files = [f for f in files if "2025-" not in f and "2026-" not in f]
    if not files:
        return None

    dfs = []
    for f in files:
        try:
            dfs.append(pl.read_parquet(f))
        except Exception:
            pass
    if not dfs:
        return None

    df = pl.concat(dfs).sort("timestamp")
    close = df["close"].to_numpy().astype(np.float64)
    high  = df["high"].to_numpy().astype(np.float64)
    low   = df["low"].to_numpy().astype(np.float64)
    vol   = df["volume"].to_numpy().astype(np.float64)
    ofi_r = df["ofi"].to_numpy().astype(np.float64)
    ts_ms = df["timestamp"].dt.cast_time_unit("ms").cast(pl.Int64).to_numpy().astype(np.float64)
    tbv   = df["taker_buy_volume"].to_numpy().astype(np.float64) if "taker_buy_volume" in df.columns else (ofi_r + vol) / 2.0
    n = len(close)
    print(f"  [{symbol}] {n:,} raw bars from monthly parquets")

    btc_aligned = None
    if btc_ts is not None and symbol != "BTCUSDT":
        btc_aligned = np.interp(ts_ms, btc_ts.astype(np.float64), btc_close)

    F_v2, atr_arr = build_features_v2(close, high, low, vol, ofi_r, ts_ms, btc_aligned)
    F_v5 = build_v5_full(symbol, F_v2, close, high, low, vol, ofi_r, tbv,
                         ts_ms, atr_arr, btc_aligned, perp_dir, basket_state)
    oh = _coin_onehot(symbol, n)
    X_v5_full = np.hstack([F_v5, oh]).astype(np.float32)

    hurst_raw = F_v2[:, 10].copy()
    max_h = max(HORIZONS)
    idx = np.arange(WARMUP_BARS, n - max_h, SAMPLE_EVERY)
    X_samp = X_v5_full[idx]
    valid = ~np.any(np.isnan(X_samp), axis=1)
    idx_v = idx[valid]
    X_v5_samp = X_samp[valid]
    if len(idx_v) == 0:
        return None

    labels = label_multi_horizon(close, high, low, atr_arr, hurst_raw, idx_v)
    keep = labels["any_valid"]
    if keep.sum() == 0:
        return None

    label_data = {}
    for h in HORIZONS:
        for k in [f"y_{h}", f"net_{h}", f"y_short_{h}", f"net_short_{h}", f"ret_{h}"]:
            label_data[k] = labels[k][keep]

    return {
        "X_v5_full": X_v5_samp[keep],
        "ts":        ts_ms[idx_v[keep]],
        "labels":    label_data,
        "any_valid": np.ones(keep.sum(), dtype=bool),
        "n":         int(keep.sum()),
        # Also store raw arrays so V6 can be computed on the sampled indices
        "_raw_close": close, "_raw_high": high, "_raw_low": low,
        "_raw_vol":   vol,   "_raw_ofi":  ofi_r,"_raw_tbv":  tbv,
        "_raw_ts":    ts_ms, "_raw_atr":  atr_arr,
        "_raw_idx":   idx_v[keep],
    }


def add_v6_features(bundle):
    """Compute V6 features and append to the bundle's X_v5_full matrix.

    Three modes (tried in order):
      A. Raw arrays bundled  → compute V6 on full series, slice to sample indices
      B. Companion OHLCV parquet exists  → load + compute (Mode B is the standard
         path when using download_v6_ohlcv.py + existing training_data parquets)
      C. No raw OHLCV  → zeros (V6_full behaves like V5; warns user)
    """
    if "_raw_close" in bundle:
        # Mode A — raw monthly parquets bundled full OHLCV arrays
        cl  = bundle["_raw_close"]; hi  = bundle["_raw_high"]
        lo  = bundle["_raw_low"];   vo  = bundle["_raw_vol"]
        ofi = bundle["_raw_ofi"];   tbv = bundle["_raw_tbv"]
        ts  = bundle["_raw_ts"];    atr = bundle["_raw_atr"]
        idx = bundle["_raw_idx"]
        F_v6_full = build_v6_features(cl, hi, lo, vo, ofi, tbv, ts, atr)
        F_v6_samp = F_v6_full[idx]

    elif "_ohlcv_path" in bundle and os.path.exists(bundle["_ohlcv_path"]):
        # Mode B — companion OHLCV downloaded by download_v6_ohlcv.py
        print(f"    Loading companion OHLCV: {os.path.basename(bundle['_ohlcv_path'])}")
        try:
            odf = pl.read_parquet(bundle["_ohlcv_path"]).sort("timestamp_ms")
            cl_f  = odf["close"].to_numpy().astype(np.float64)
            hi_f  = odf["high"].to_numpy().astype(np.float64)
            lo_f  = odf["low"].to_numpy().astype(np.float64)
            vo_f  = odf["volume"].to_numpy().astype(np.float64)
            tbv_f = odf["taker_buy_volume"].to_numpy().astype(np.float64)
            ofi_f = (odf["ofi"].to_numpy().astype(np.float64)
                     if "ofi" in odf.columns else 2 * tbv_f - vo_f)
            ts_f  = odf["timestamp_ms"].to_numpy().astype(np.int64)

            # Compute ATR-14 via V2 builder (only need atr_arr output)
            _, atr_f = build_features_v2(
                cl_f, hi_f, lo_f, vo_f, ofi_f, ts_f.astype(np.float64), None
            )

            # Compute all 36 V6 features on the full 1m series
            F_v6_full = build_v6_features(cl_f, hi_f, lo_f, vo_f, ofi_f, tbv_f, ts_f, atr_f)

            # Map training timestamps → nearest row in OHLCV series
            train_ts = bundle["ts"].astype(np.int64)
            pos = np.searchsorted(ts_f, train_ts, side="left")
            pos = np.clip(pos, 0, len(ts_f) - 1)
            valid_match = np.abs(ts_f[pos] - train_ts) <= 5 * 60_000  # ±5 min tolerance
            F_v6_samp = np.zeros((bundle["n"], N_V6_EXTRA), dtype=np.float32)
            F_v6_samp[valid_match] = F_v6_full[pos[valid_match]]
            n_ok = int(valid_match.sum())
            print(f"    V6 timestamp match: {n_ok}/{bundle['n']} rows ({n_ok/bundle['n']*100:.1f}%)")
        except Exception as e:
            print(f"    [WARN] Companion OHLCV load failed: {e} — zeroing V6 features")
            F_v6_samp = np.zeros((bundle["n"], N_V6_EXTRA), dtype=np.float32)

    else:
        # Mode C — no OHLCV available; zero-fill V6 (still trains, just V5 features only)
        ohlcv_hint = bundle.get("_ohlcv_path", "unknown")
        print(f"    [WARN] Companion OHLCV not found: {os.path.basename(ohlcv_hint)}")
        print( "    [WARN] Run:  python download_v6_ohlcv.py  (free, no API key needed)")
        print( "    [WARN] V6 features set to zero — model will train on V5 features only.")
        F_v6_samp = np.zeros((bundle["n"], N_V6_EXTRA), dtype=np.float32)

    X_v6_full  = np.hstack([bundle["X_v5_full"], F_v6_samp]).astype(np.float32)
    bundle["X_v6_full"]   = X_v6_full   # (n, 130)
    bundle["X_v6_only"]   = F_v6_samp   # (n, 36)
    bundle["v6_strength"] = compute_v6_signal_strength(F_v6_samp)  # (n,)
    return bundle


# ─────────────────────────────────────────────────────────────────────────────
# Horizon training
# ─────────────────────────────────────────────────────────────────────────────

def train_one_horizon(h, X_tr, ts_tr, y_tr, net_tr,
                      X_va, y_va, net_va,
                      X_te, y_te, net_te,
                      ret_tr, ret_va, ret_te,
                      v6_strength_tr,
                      feature_names, anchor_ts,
                      direction, model_tag, tag_prefix):
    """Train classifier + regressor for one horizon/direction/model-set.

    Returns result dict or None if not enough data.
    """
    print(f"\n  ── {tag_prefix} | h={h}m | {direction.upper()} ─────────────────────────────────")

    m_tr = ~np.isnan(y_tr); m_va = ~np.isnan(y_va); m_te = ~np.isnan(y_te)
    n_tr, n_va, n_te = int(m_tr.sum()), int(m_va.sum()), int(m_te.sum())
    print(f"    samples — train {n_tr:,}, val {n_va:,}, test {n_te:,}")
    if n_tr < 3000 or n_va < 300:
        print("    SKIPPED (insufficient data).")
        return None

    # Recency weighting × V6 signal strength (for full model only)
    w_base = recency_weights(ts_tr[m_tr], anchor_ts)
    if model_tag == "full":
        # Boost rows where V6 signals are strong: up to 2.33x base weight
        # → effectively gives V6-driven rows 70% of learning budget
        v6_boost = 1.0 + 2.33 * v6_strength_tr[m_tr]
        sample_w = w_base * v6_boost
        sample_w = sample_w / sample_w.mean()   # renormalise to keep scale
    else:
        sample_w = w_base

    dtr = xgb.DMatrix(X_tr[m_tr], label=y_tr[m_tr], weight=sample_w, feature_names=feature_names)
    dva = xgb.DMatrix(X_va[m_va], label=y_va[m_va], feature_names=feature_names)
    dte = xgb.DMatrix(X_te[m_te], label=y_te[m_te], feature_names=feature_names)

    params = XGB_CLF_PARAMS if model_tag == "full" else XGB_MICRO_PARAMS
    clf = xgb.train(
        params, dtr, num_boost_round=800,
        evals=[(dtr, "train"), (dva, "val")],
        early_stopping_rounds=40, verbose_eval=100,
    )

    p_va_raw = clf.predict(dva)
    p_te_raw = clf.predict(dte)
    from sklearn.isotonic import IsotonicRegression
    calib = IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds="clip")
    calib.fit(p_va_raw, y_va[m_va])
    p_va_cal = calib.predict(p_va_raw)
    p_te_cal = calib.predict(p_te_raw)

    best, _ = pick_threshold(p_va_cal, net_va[m_va],
                             min_trades=max(100, n_va // 1000))
    print(f"    best thr={best['threshold']:.2f}  val EV={best['ev_pct']:+.3f}%  "
          f"WR={best['win_rate']:.1f}%  PF={best['profit_factor']:.2f}  n={best['n']}")

    m_t = p_te_cal >= best["threshold"]
    if m_t.sum() > 0:
        nets_t = net_te[m_te][m_t]
        ev_t = float(nets_t.mean()) * 100
        wr_t = float((nets_t > 0).mean()) * 100
        wins_t   = nets_t[nets_t > 0].sum()
        losses_t = -nets_t[nets_t <= 0].sum()
        pf_t = float(wins_t / max(losses_t, 1e-9))
        print(f"    OOS test: n={int(m_t.sum())}, EV={ev_t:+.3f}%, WR={wr_t:.1f}%, PF={pf_t:.2f}")
    else:
        ev_t = wr_t = pf_t = 0.0
        print("    OOS test: no trades cleared threshold.")

    # Calibration reliability
    for lo_p, hi_p in [(0.45, 0.55), (0.55, 0.65), (0.65, 0.75), (0.75, 1.00)]:
        mm = (p_va_cal >= lo_p) & (p_va_cal < hi_p)
        if mm.sum() > 30:
            print(f"      p∈[{lo_p:.2f},{hi_p:.2f})  n={int(mm.sum()):>5}  "
                  f"pred~{p_va_cal[mm].mean():.3f}  actual={y_va[m_va][mm].mean():.3f}")

    # Save classifier + calibrator
    clf_path   = os.path.join(DATA_DIR, f"v6_{model_tag}_clf_{direction}_h{h}.json")
    calib_path = os.path.join(DATA_DIR, f"v6_{model_tag}_calib_{direction}_h{h}.pkl")
    clf.save_model(clf_path)
    with open(calib_path, "wb") as f:
        pickle.dump(calib, f)

    # Regressor (one per horizon, not per direction)
    rank_ic = 0.0
    if direction == "long" and model_tag == "full":
        rm_tr = ~np.isnan(ret_tr); rm_va = ~np.isnan(ret_va)
        if rm_tr.sum() > 1000:
            w_tr_r = recency_weights(ts_tr[rm_tr], anchor_ts)
            dtr_r = xgb.DMatrix(X_tr[rm_tr], label=ret_tr[rm_tr], weight=w_tr_r, feature_names=feature_names)
            dva_r = xgb.DMatrix(X_va[rm_va], label=ret_va[rm_va], feature_names=feature_names)
            reg = xgb.train(XGB_REG_PARAMS, dtr_r, num_boost_round=800,
                            evals=[(dtr_r, "train"), (dva_r, "val")],
                            early_stopping_rounds=40, verbose_eval=False)
            reg_path = os.path.join(DATA_DIR, f"v6_full_reg_h{h}.json")
            reg.save_model(reg_path)
            rm_te = ~np.isnan(ret_te)
            if rm_te.sum() > 0:
                p_reg = reg.predict(xgb.DMatrix(X_te[rm_te], feature_names=feature_names))
                from scipy.stats import spearmanr
                rank_ic, _ = spearmanr(p_reg, ret_te[rm_te])
            print(f"    regressor rank IC: {rank_ic:+.3f}")

    # Feature importance (top 12)
    try:
        imp = clf.get_score(importance_type="gain")
        ranked = sorted(imp.items(), key=lambda x: -x[1])[:12]
        print("    top-12 features (gain):")
        for fname, g in ranked:
            marker = " ◀ V6" if fname in FEATURE_NAMES_V6_EXTRA else ""
            print(f"      {fname:<32} {g:>10.1f}{marker}")
    except Exception:
        pass

    return {
        "horizon":            int(h),
        "direction":          direction,
        "model_tag":          model_tag,
        "clf_path":           os.path.relpath(clf_path,   os.path.dirname(__file__)),
        "calib_path":         os.path.relpath(calib_path, os.path.dirname(__file__)),
        "p_threshold":        best["threshold"],
        "val_ev_pct":         best["ev_pct"],
        "val_win_rate_pct":   best["win_rate"],
        "val_profit_factor":  best["profit_factor"],
        "val_n_trades":       best["n"],
        "test_ev_pct":        ev_t,
        "test_win_rate_pct":  wr_t,
        "test_profit_factor": pf_t,
        "test_n_trades":      int(m_t.sum()) if m_t is not None else 0,
        "regressor_rank_ic":  float(rank_ic),
        "n_train": n_tr, "n_val": n_va, "n_test": n_te,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train V6 models")
    parser.add_argument("--processed-dir", default=DEFAULT_PROCESSED_DIR,
                        help="Directory containing training parquets")
    parser.add_argument("--skip-micro", action="store_true",
                        help="Skip V6 micro-model training (saves ~50% time)")
    args = parser.parse_args()
    processed_dir = args.processed_dir

    t0 = time.time()
    print("=" * 80)
    print("  CRYPTO ORACLE V6 — TRAINING PIPELINE")
    print(f"  Processed dir : {processed_dir}")
    print(f"  Horizons      : {HORIZONS}")
    print(f"  V6_full feat  : {N_V6_FULL}  (V5={N_V5_FULL} + V6_extras={N_V6_EXTRA})")
    print(f"  V6_micro feat : {N_V6_EXTRA}")
    print(f"  Round-trip cost: {ROUND_TRIP_COST*100:.2f}%  MIN_EV: {MIN_EV_PCT*100:.2f}%")
    print("=" * 80)

    # ── Detect data source ────────────────────────────────────────────────────
    # Mode A: Colab training_data.parquet files
    colab_parquets = glob.glob(os.path.join(processed_dir, "*_training_data.parquet"))
    # Mode B: Raw monthly feature parquets (E-drive / train_v5 style)
    raw_parquets   = glob.glob(os.path.join(processed_dir, "*_1m_features_*.parquet"))

    if colab_parquets:
        print(f"\n[DATA] Found {len(colab_parquets)} Colab training_data parquets in {processed_dir}")
        use_colab = True
        symbols = sorted({os.path.basename(f).split("_training_data")[0] for f in colab_parquets})
    elif raw_parquets:
        print(f"\n[DATA] Found {len(raw_parquets)} raw monthly parquets in {processed_dir}")
        use_colab = False

        # Load BTC reference & basket for raw mode
        from train_ev_model_v2 import load_btc_close
        btc_ts, btc_close = load_btc_close()
        if btc_ts is not None:
            print(f"  BTC reference: {len(btc_ts):,} bars")
        basket_state = None
        if os.path.exists(BASKET_FP):
            basket_state = pl.read_parquet(BASKET_FP)
            print(f"  Basket state: {basket_state.height:,} rows")
        symbols = sorted({os.path.basename(f).split("_1m_")[0] for f in raw_parquets})
    else:
        print(f"\n[ERROR] No training data found in {processed_dir}")
        print("  Run the Colab unified pipeline first to generate training parquets.")
        sys.exit(1)

    print(f"  Coins: {len(symbols)} — {', '.join(symbols)}\n")

    # ── Load & build V6 features per coin ────────────────────────────────────
    bundles = []
    for sym in symbols:
        print(f"\n[Coin] {sym}")
        if use_colab:
            b = _load_from_colab_parquet(sym, processed_dir)
        else:
            b = _load_from_processed_features(sym, processed_dir,
                                              btc_ts, btc_close,
                                              basket_state, PERP_DIR)
        if b is None:
            print(f"  SKIP — no data")
            continue

        print(f"  Building V6 features...")
        b = add_v6_features(b)
        bundles.append(b)
        print(f"  X_v6_full shape: {b['X_v6_full'].shape}")

    if not bundles:
        print("\n[ERROR] All coins skipped — no data loaded.")
        sys.exit(1)

    # ── Stack all bundles ─────────────────────────────────────────────────────
    X_full   = np.vstack([b["X_v6_full"]  for b in bundles])  # (N, 130)
    X_micro  = np.vstack([b["X_v6_only"]  for b in bundles])  # (N, 36)
    v6_str   = np.concatenate([b["v6_strength"] for b in bundles])  # (N,)
    ts_all   = np.concatenate([b["ts"] for b in bundles])

    per_h = {}
    for h in HORIZONS:
        for k in [f"y_{h}", f"net_{h}", f"y_short_{h}", f"net_short_{h}", f"ret_{h}"]:
            per_h[k] = np.concatenate([b["labels"].get(k, np.full(b["n"], np.nan)) for b in bundles])

    # Sort by timestamp (temporal ordering is critical for valid OOS evaluation)
    order  = np.argsort(ts_all, kind="mergesort")
    X_full = X_full[order]; X_micro = X_micro[order]
    v6_str = v6_str[order]; ts_all  = ts_all[order]
    for k in list(per_h.keys()):
        per_h[k] = per_h[k][order]

    N = len(X_full)
    train_end = int(N * 0.70); val_end = int(N * 0.85)
    sl_tr = slice(0, train_end); sl_va = slice(train_end, val_end); sl_te = slice(val_end, N)

    import datetime as dt
    def _fmt(ms): return dt.datetime.utcfromtimestamp(ms / 1000).strftime("%Y-%m-%d")
    print(f"\nTotal: {N:,} samples  ×  {X_full.shape[1]} full features  /  {X_micro.shape[1]} micro features")
    print(f"  train : {train_end:>8,}  {_fmt(ts_all[0])} → {_fmt(ts_all[train_end-1])}")
    print(f"  val   : {val_end-train_end:>8,}  {_fmt(ts_all[train_end])} → {_fmt(ts_all[val_end-1])}")
    print(f"  test  : {N-val_end:>8,}  {_fmt(ts_all[val_end])} → {_fmt(ts_all[-1])}")

    anchor_ts = float(ts_all.max())

    # ── Train models ──────────────────────────────────────────────────────────
    full_feat_names  = ALL_FEATURE_NAMES_V6
    micro_feat_names = FEATURE_NAMES_V6_EXTRA

    horizon_results = {}
    for h in HORIZONS:
        horizon_results[str(h)] = {}
        ret_arr = per_h[f"ret_{h}"]

        for direction in ["long", "short"]:
            if direction == "long":
                y_arr  = per_h[f"y_{h}"]
                net_arr= per_h[f"net_{h}"]
            else:
                y_arr  = per_h[f"y_short_{h}"]
                net_arr= per_h[f"net_short_{h}"]

            direction_results = {}

            # ── V6_full model (130 features) ──────────────────────────────
            res_full = train_one_horizon(
                h,
                X_full[sl_tr], ts_all[sl_tr], y_arr[sl_tr], net_arr[sl_tr],
                X_full[sl_va],                y_arr[sl_va], net_arr[sl_va],
                X_full[sl_te],                y_arr[sl_te], net_arr[sl_te],
                ret_arr[sl_tr], ret_arr[sl_va], ret_arr[sl_te],
                v6_str[sl_tr],
                full_feat_names, anchor_ts, direction, "full",
                f"V6_full",
            )
            if res_full:
                direction_results["full"] = res_full

            # ── V6_micro model (36 features) ─────────────────────────────
            if not args.skip_micro:
                res_micro = train_one_horizon(
                    h,
                    X_micro[sl_tr], ts_all[sl_tr], y_arr[sl_tr], net_arr[sl_tr],
                    X_micro[sl_va],                y_arr[sl_va], net_arr[sl_va],
                    X_micro[sl_te],                y_arr[sl_te], net_arr[sl_te],
                    ret_arr[sl_tr], ret_arr[sl_va], ret_arr[sl_te],
                    v6_str[sl_tr],
                    micro_feat_names, anchor_ts, direction, "micro",
                    f"V6_micro",
                )
                if res_micro:
                    direction_results["micro"] = res_micro

            horizon_results[str(h)][direction] = direction_results

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  SUMMARY")
    print("=" * 80)
    hdr = f"  {'model':<18} {'h':>4} {'dir':>6} {'val_EV%':>9} {'val_WR%':>9} {'val_PF':>7} {'test_EV%':>10} {'test_WR%':>10} {'test_PF':>8}"
    print(hdr)
    for h_str, dirs in horizon_results.items():
        for direction, models in dirs.items():
            for tag, r in models.items():
                label = f"V6_{tag}"
                print(f"  {label:<18} {h_str:>4} {direction:>6} "
                      f"{r['val_ev_pct']:>+9.3f} {r['val_win_rate_pct']:>9.1f} "
                      f"{r['val_profit_factor']:>7.2f} {r['test_ev_pct']:>+10.3f} "
                      f"{r['test_win_rate_pct']:>10.1f} {r['test_profit_factor']:>8.2f}")

    # ── Save meta ─────────────────────────────────────────────────────────────
    meta = {
        "model_type":          "xgboost_v6_full_plus_micro",
        "v6_full_feature_names": ALL_FEATURE_NAMES_V6,
        "v6_micro_feature_names": FEATURE_NAMES_V6_EXTRA,
        "n_features_v6_full":  N_V6_FULL,
        "n_features_v6_micro": N_V6_EXTRA,
        "n_features_v5_full":  N_V5_FULL,
        "horizons":            HORIZONS,
        "training_coins":      symbols,
        "coin_onehot_order":   COIN_ONEHOT_NAMES,
        "total_samples":       int(N),
        "split":               {"train": int(train_end), "val": int(val_end - train_end), "test": int(N - val_end)},
        "round_trip_cost":     ROUND_TRIP_COST,
        "min_ev_pct_required": MIN_EV_PCT,
        "blending":            {"v5_weight": 0.30, "v6_weight": 0.70},
        "horizon_results":     horizon_results,
    }
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(META_PATH, "w") as fp:
        json.dump(meta, fp, indent=2)
    print(f"\n[OK] Meta saved → {META_PATH}")
    print(f"  Total time: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
