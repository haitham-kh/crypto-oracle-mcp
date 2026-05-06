"""
train_v4.py - Multi-horizon, EV-driven training pipeline.

Per horizon h ∈ HORIZONS we train TWO models:

    classifier_h  : P(fee-positive trade at horizon h | features)
    regressor_h   : E[forward log-return over h bars | features]

Plus an isotonic calibrator on validation. At decision time the simulator
picks the horizon with the highest realised EV:

    EV_h = p_h_cal * tp_h_pct - (1 - p_h_cal) * sl_h_pct - cost

Per-horizon thresholds are chosen by maximising validation EV with a
minimum-trade-count floor; both threshold and calibrator are persisted.

Training pipeline:
  1. Per coin: read parquets → v4 features (62) → multi-horizon labels.
  2. Concatenate across all coins, sort globally by timestamp.
  3. Temporal walk-forward split 70 / 15 / 15.
  4. Recency-weight training rows on the global newest timestamp.
  5. For each horizon: train classifier + regressor, calibrate, tune thr,
     report OOS EV on the held-out test slice.
  6. Persist: per-horizon model files, calibrators, and a single meta JSON
     containing thresholds, EV diagnostics, and feature names.

Run with:
    python train_v4.py
"""
from __future__ import annotations
import os, sys, glob, json, time, pickle
import numpy as np
import polars as pl
import xgboost as xgb

sys.path.insert(0, os.path.dirname(__file__))
from train_ev_model_v2 import build_features_v2, load_btc_close, PROCESSED_DIR
from features_v4 import FEATURE_NAMES_V4, build_v4_from_components
from labels_multi_horizon import label_multi_horizon
from trading_config import (
    HORIZONS, ROUND_TRIP_COST,
    WARMUP_BARS, SAMPLE_EVERY, MIN_EV_PCT, MIN_P_UP_DEFAULT,
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
META_PATH = os.path.join(DATA_DIR, "ev_model_v4_meta.json")


# ── Per-coin processing ─────────────────────────────────────────────────────

def process_coin(symbol, btc_ts, btc_close):
    """Return dict with X (k,62), ts (k,), and per-horizon label arrays."""
    pattern = os.path.join(PROCESSED_DIR, f"{symbol}_1m_features_*.parquet")
    files = sorted(glob.glob(pattern))
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
    ofi_raw = df["ofi"].to_numpy().astype(np.float64)
    ts_ms = df["timestamp"].cast(pl.Int64).to_numpy().astype(np.float64)
    # taker_buy_volume column may or may not exist; fall back to volume/2
    if "taker_buy_volume" in df.columns:
        tbv = df["taker_buy_volume"].to_numpy().astype(np.float64)
    else:
        # If only OFI is stored, taker_buy = (OFI + total_vol) / 2
        tbv = (ofi_raw + vol) / 2.0
    n = len(close)
    print(f"  {symbol}: {n:,} bars, {len(files)} months", flush=True)

    t0 = time.time()
    btc_aligned = None
    if btc_ts is not None and btc_close is not None and symbol != "BTCUSDT":
        btc_aligned = np.interp(ts_ms, btc_ts.astype(np.float64), btc_close)

    print(f"    [{symbol}] building v2 features...", flush=True)
    F_v2, atr_arr = build_features_v2(close, high, low, vol, ofi_raw, ts_ms, btc_aligned)
    print(f"    [{symbol}] v2 done ({time.time()-t0:.1f}s); building v4 extras...", flush=True)
    t1 = time.time()
    X = build_v4_from_components(F_v2, close, high, low, vol, ofi_raw, tbv,
                                 ts_ms, atr_arr, btc_close=btc_aligned)
    print(f"    [{symbol}] v4 features done ({time.time()-t1:.1f}s, "
          f"shape {X.shape}); sampling + labelling...", flush=True)
    hurst_raw = F_v2[:, 10].copy()

    max_h = max(HORIZONS)
    idx = np.arange(WARMUP_BARS, n - max_h, SAMPLE_EVERY)
    X_samp = X[idx]
    valid_feat = ~np.any(np.isnan(X_samp), axis=1)
    idx_v = idx[valid_feat]
    X_v = X_samp[valid_feat]
    if len(idx_v) == 0:
        return None

    t2 = time.time()
    labels = label_multi_horizon(close, high, low, atr_arr, hurst_raw, idx_v)
    print(f"    [{symbol}] labels done ({time.time()-t2:.1f}s); "
          f"total {time.time()-t0:.1f}s for this coin", flush=True)
    keep = labels["any_valid"]
    if keep.sum() == 0:
        return None

    out = {"X": X_v[keep], "ts": ts_ms[idx_v[keep]]}
    for h in HORIZONS:
        out[f"y_{h}"]     = labels[f"y_{h}"][keep]
        out[f"net_{h}"]   = labels[f"net_{h}"][keep]
        out[f"ret_{h}"]   = labels[f"ret_{h}"][keep]
        out[f"sigma_{h}"] = labels[f"sigma_{h}"][keep]
        out[f"tp_{h}"]    = labels[f"tp_{h}"][keep]
        out[f"sl_{h}"]    = labels[f"sl_{h}"][keep]

    counts = {h: int((~np.isnan(out[f"y_{h}"])).sum()) for h in HORIZONS}
    print(f"  {symbol}: kept {keep.sum():,} | "
          + " ".join(f"h={h}:{counts[h]:,}" for h in HORIZONS), flush=True)
    return out


# ── Recency weighting ───────────────────────────────────────────────────────

def recency_weights(ts, anchor_ts, half_life_months=12):
    age_m = np.maximum((anchor_ts - ts) / (30.44 * 24 * 3600 * 1000), 0)
    lam = np.log(2) / half_life_months
    w = np.exp(-lam * age_m)
    return w / w.mean()


# ── Per-horizon training ────────────────────────────────────────────────────

XGB_CLF_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": ["logloss", "auc"],
    "max_depth": 6,
    "learning_rate": 0.04,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 80,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "tree_method": "hist",
    "nthread": 4,
    "seed": 42,
    "verbosity": 1,
}
XGB_REG_PARAMS = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "max_depth": 6,
    "learning_rate": 0.04,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 80,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "tree_method": "hist",
    "nthread": 4,
    "seed": 43,
    "verbosity": 0,
}


def pick_threshold(p_cal, net_pct, min_trades):
    grid = np.round(np.arange(0.50, 0.81, 0.01), 2)
    best = {"threshold": MIN_P_UP_DEFAULT, "ev_pct": -1e9, "n": 0,
            "win_rate": 0.0, "profit_factor": 0.0}
    rows = []
    for thr in grid:
        m = p_cal >= thr
        n = int(m.sum())
        if n < min_trades:
            rows.append((float(thr), n, 0.0, 0.0, 0.0)); continue
        nets = net_pct[m]
        ev = float(np.nanmean(nets))
        wr = float((nets > 0).mean())
        wins = nets[nets > 0].sum()
        losses = -nets[nets <= 0].sum()
        pf = float(wins / max(losses, 1e-9))
        rows.append((float(thr), n, ev * 100, wr * 100, pf))
        if ev > best["ev_pct"]:
            best = {"threshold": float(thr), "ev_pct": ev * 100,
                    "n": n, "win_rate": wr * 100, "profit_factor": pf}
    return best, rows


def train_horizon(h, X_train, ts_train, y_train, X_val, y_val, net_val,
                  X_test, y_test, net_test, ret_train, ret_val, ret_test,
                  feature_names, anchor_ts):
    """Train classifier + regressor for a single horizon. Returns dict."""
    print(f"\n  ── horizon {h}m ───────────────────────────────────────────")
    # Classifier mask: only rows with valid y_h
    m_tr = ~np.isnan(y_train); m_va = ~np.isnan(y_val); m_te = ~np.isnan(y_test)
    n_tr, n_va, n_te = int(m_tr.sum()), int(m_va.sum()), int(m_te.sum())
    print(f"    classifier samples — train {n_tr:,}, val {n_va:,}, test {n_te:,}")
    if n_tr < 5000 or n_va < 500:
        print("    SKIPPED: not enough samples for this horizon.")
        return None

    w_tr = recency_weights(ts_train[m_tr], anchor_ts)
    dtr = xgb.DMatrix(X_train[m_tr], label=y_train[m_tr], weight=w_tr,
                      feature_names=feature_names)
    dva = xgb.DMatrix(X_val[m_va],   label=y_val[m_va],   feature_names=feature_names)
    dte = xgb.DMatrix(X_test[m_te],  label=y_test[m_te],  feature_names=feature_names)

    print(f"    training classifier...")
    clf = xgb.train(XGB_CLF_PARAMS, dtr,
                    num_boost_round=600,
                    evals=[(dtr, "train"), (dva, "val")],
                    early_stopping_rounds=40,
                    verbose_eval=100)

    p_va_raw = clf.predict(dva); p_te_raw = clf.predict(dte)
    from sklearn.isotonic import IsotonicRegression
    calib = IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds="clip")
    calib.fit(p_va_raw, y_val[m_va])
    p_va_cal = calib.predict(p_va_raw)
    p_te_cal = calib.predict(p_te_raw)

    # ── Threshold tuning by EV on val ──────────────────────────────────────
    best, rows = pick_threshold(p_va_cal, net_val[m_va],
                                min_trades=max(200, n_va // 1000))
    print("    thr  trades       EV%   WR%      PF")
    for thr, n, ev_p, wr_p, pf in rows[::3]:
        flag = " <<" if abs(thr - best["threshold"]) < 1e-9 else ""
        print(f"    {thr:0.2f} {n:>7} {ev_p:>+7.3f} {wr_p:>5.1f} {pf:>6.2f}{flag}")
    print(f"    chosen p_thr={best['threshold']:.2f}  val EV={best['ev_pct']:+.3f}%/trade  "
          f"WR={best['win_rate']:.1f}%  PF={best['profit_factor']:.2f}  n={best['n']}")

    # ── Test slice diagnostics ─────────────────────────────────────────────
    m_t = p_te_cal >= best["threshold"]
    if m_t.sum() > 0:
        nets_t = net_test[m_te][m_t]
        ev_t = float(nets_t.mean()) * 100
        wr_t = float((nets_t > 0).mean()) * 100
        wins_t = nets_t[nets_t > 0].sum()
        losses_t = -nets_t[nets_t <= 0].sum()
        pf_t = float(wins_t / max(losses_t, 1e-9))
        print(f"    OOS test: n={int(m_t.sum())}, EV={ev_t:+.3f}%, "
              f"WR={wr_t:.1f}%, PF={pf_t:.2f}")
    else:
        ev_t = wr_t = pf_t = 0.0
        print("    OOS test: no trades cleared the threshold.")

    # ── Reliability on val ─────────────────────────────────────────────────
    print("    calibration reliability (val):")
    for lo, hi in [(0.45,0.55),(0.55,0.60),(0.60,0.70),(0.70,1.00)]:
        mm = (p_va_cal >= lo) & (p_va_cal < hi)
        if mm.sum() > 30:
            print(f"      p∈[{lo:.2f},{hi:.2f})  n={int(mm.sum()):>5}  "
                  f"pred~{p_va_cal[mm].mean():.3f}  actual={y_val[m_va][mm].mean():.3f}")

    # ── Regressor (forward log-return) ─────────────────────────────────────
    # The regressor is trained on ALL rows (no fee-rejection filter): the
    # target is always defined.
    rm_tr = ~np.isnan(ret_train); rm_va = ~np.isnan(ret_val); rm_te = ~np.isnan(ret_test)
    w_tr_r = recency_weights(ts_train[rm_tr], anchor_ts)
    dtr_r = xgb.DMatrix(X_train[rm_tr], label=ret_train[rm_tr], weight=w_tr_r,
                        feature_names=feature_names)
    dva_r = xgb.DMatrix(X_val[rm_va],   label=ret_val[rm_va],   feature_names=feature_names)
    print(f"    training regressor (log-return target)...")
    reg = xgb.train(XGB_REG_PARAMS, dtr_r,
                    num_boost_round=400,
                    evals=[(dtr_r, "train"), (dva_r, "val")],
                    early_stopping_rounds=40,
                    verbose_eval=100)

    # Spearman of reg predictions vs realised return on val (rank IC)
    from scipy.stats import spearmanr
    pred_val = reg.predict(dva_r)
    rank_ic, _ = spearmanr(pred_val, ret_val[rm_va])
    print(f"    regressor rank-IC (val) = {rank_ic:.4f}")

    # ── Persist this horizon's artefacts ───────────────────────────────────
    clf_path  = os.path.join(DATA_DIR, f"v4_clf_h{h}.json")
    reg_path  = os.path.join(DATA_DIR, f"v4_reg_h{h}.json")
    cal_path  = os.path.join(DATA_DIR, f"v4_calib_h{h}.pkl")
    clf.save_model(clf_path); reg.save_model(reg_path)
    with open(cal_path, "wb") as fp: pickle.dump(calib, fp)

    return {
        "horizon": int(h),
        "clf_path": os.path.relpath(clf_path, os.path.dirname(__file__)),
        "reg_path": os.path.relpath(reg_path, os.path.dirname(__file__)),
        "calib_path": os.path.relpath(cal_path, os.path.dirname(__file__)),
        "p_threshold": best["threshold"],
        "val_ev_pct": best["ev_pct"],
        "val_win_rate_pct": best["win_rate"],
        "val_profit_factor": best["profit_factor"],
        "val_n_trades": best["n"],
        "test_ev_pct": ev_t,
        "test_win_rate_pct": wr_t,
        "test_profit_factor": pf_t,
        "test_n_trades": int(m_t.sum()) if m_t is not None else 0,
        "regressor_rank_ic": float(rank_ic),
        "n_train": n_tr, "n_val": n_va, "n_test": n_te,
    }


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print("=" * 78)
    print("  EV Model V4 — multi-horizon, fee-aware, classifier+regressor ensemble")
    print(f"  Horizons: {HORIZONS}   round-trip cost: {ROUND_TRIP_COST*100:.2f}%")
    print(f"  Features: {len(FEATURE_NAMES_V4)} (v3=29 + v4-extras=33)")
    print("=" * 78)

    print("\nLoading BTC reference...")
    btc_ts, btc_close = load_btc_close()
    if btc_ts is not None:
        print(f"  BTC: {len(btc_ts):,} bars")

    all_files = glob.glob(os.path.join(PROCESSED_DIR, "*.parquet"))
    symbols = sorted({os.path.basename(f).split("_1m_")[0] for f in all_files})
    print(f"Coins: {len(symbols)}\n")

    bundles = []
    for sym in symbols:
        b = process_coin(sym, btc_ts, btc_close)
        if b is not None:
            bundles.append(b)

    # Concatenate everything
    X = np.vstack([b["X"] for b in bundles])
    ts_all = np.concatenate([b["ts"] for b in bundles])
    per_h = {}
    for h in HORIZONS:
        per_h[f"y_{h}"]   = np.concatenate([b[f"y_{h}"]   for b in bundles])
        per_h[f"net_{h}"] = np.concatenate([b[f"net_{h}"] for b in bundles])
        per_h[f"ret_{h}"] = np.concatenate([b[f"ret_{h}"] for b in bundles])

    # Temporal walk-forward split
    order = np.argsort(ts_all, kind="mergesort")
    X = X[order]; ts_all = ts_all[order]
    for k in list(per_h.keys()):
        per_h[k] = per_h[k][order]

    n = len(X)
    train_end = int(n * 0.70); val_end = int(n * 0.85)
    sl_tr = slice(0, train_end); sl_va = slice(train_end, val_end); sl_te = slice(val_end, n)

    import datetime as dt
    def _fmt(ms): return dt.datetime.utcfromtimestamp(ms/1000).strftime("%Y-%m-%d")
    print(f"\nTotal: {n:,} samples × {X.shape[1]} features")
    print(f"  train  {train_end:>8,}  {_fmt(ts_all[0])}  →  {_fmt(ts_all[train_end-1])}")
    print(f"  val    {val_end-train_end:>8,}  {_fmt(ts_all[train_end])}  →  {_fmt(ts_all[val_end-1])}")
    print(f"  test   {n-val_end:>8,}  {_fmt(ts_all[val_end])}  →  {_fmt(ts_all[-1])}")

    anchor_ts = float(ts_all.max())
    horizon_results = {}
    for h in HORIZONS:
        y = per_h[f"y_{h}"]; net = per_h[f"net_{h}"]; ret = per_h[f"ret_{h}"]
        res = train_horizon(
            h,
            X[sl_tr], ts_all[sl_tr], y[sl_tr],
            X[sl_va],                y[sl_va], net[sl_va],
            X[sl_te],                y[sl_te], net[sl_te],
            ret[sl_tr], ret[sl_va], ret[sl_te],
            FEATURE_NAMES_V4, anchor_ts,
        )
        if res is not None:
            horizon_results[str(h)] = res

    # Summary
    print("\n" + "=" * 78)
    print("  SUMMARY (best-EV horizon wins at decision time)")
    print("=" * 78)
    print(f"  {'h':<5}{'val_EV%':>9}{'val_WR%':>9}{'val_PF':>8}{'val_n':>9}"
          f"{'test_EV%':>10}{'test_WR%':>10}{'test_PF':>9}{'test_n':>9}{'regIC':>8}")
    for h_str, r in horizon_results.items():
        print(f"  {h_str:<5}{r['val_ev_pct']:>+9.3f}{r['val_win_rate_pct']:>9.1f}"
              f"{r['val_profit_factor']:>8.2f}{r['val_n_trades']:>9}"
              f"{r['test_ev_pct']:>+10.3f}{r['test_win_rate_pct']:>10.1f}"
              f"{r['test_profit_factor']:>9.2f}{r['test_n_trades']:>9}"
              f"{r['regressor_rank_ic']:>+8.3f}")

    meta = {
        "model_type": "xgboost_v4_multi_horizon_clf_reg",
        "feature_names": FEATURE_NAMES_V4,
        "n_features": len(FEATURE_NAMES_V4),
        "horizons": HORIZONS,
        "training_coins": symbols,
        "total_samples": int(n),
        "split": {"train": int(train_end),
                  "val": int(val_end - train_end),
                  "test": int(n - val_end)},
        "round_trip_cost": ROUND_TRIP_COST,
        "min_ev_pct_required_for_trade": MIN_EV_PCT,
        "horizon_results": horizon_results,
    }
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(META_PATH, "w") as fp: json.dump(meta, fp, indent=2)
    print(f"\n[OK] meta saved -> {META_PATH}")
    print(f"  Total time: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
