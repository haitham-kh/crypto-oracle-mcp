"""
train_ev_model_v3.py - Fee-aware, EV-driven training pipeline.

Goal: predict the probability that a real trade — opened at bar i with the
exact ATR/horizon/fees the simulator and live executor use — closes
fee-positive. Then pick a probability threshold that maximises net-of-fees
expected value on a held-out *temporally* later slice.

Major upgrades vs the previous v3:
  1. Fee-aware labels (labels_fee_aware.py) match the simulator 1:1.
  2. Proper TEMPORAL walk-forward split (sort by timestamp globally first).
  3. Sample weights anchored on the global newest timestamp.
  4. Isotonic probability calibration on a dedicated validation slice.
  5. Threshold chosen by maximising realised expectancy on validation.
  6. Saves: model, calibrator, meta with chosen threshold + EV diagnostics.
"""
from __future__ import annotations
import os, sys, glob, time, json, pickle
import numpy as np
import polars as pl
import xgboost as xgb

sys.path.insert(0, os.path.dirname(__file__))
from train_ev_model_v2 import (
    build_features_v2, load_btc_close, _atr,
    PROCESSED_DIR,
)
from features_v3 import FEATURE_NAMES_V3, build_v3_from_v2
from labels_fee_aware import label_fee_aware
from trading_config import (
    FORWARD_BARS, SAMPLE_EVERY, WARMUP_BARS, ROUND_TRIP_COST,
    MIN_EV_PCT, MIN_P_UP_DEFAULT,
)

DATA_DIR   = os.path.join(os.path.dirname(__file__), "data")
MODEL_PATH = os.path.join(DATA_DIR, "ev_model_xgb_v3.json")
META_PATH  = os.path.join(DATA_DIR, "ev_model_v3_meta.json")
CALIB_PATH = os.path.join(DATA_DIR, "ev_model_v3_calibrator.pkl")


# ── Per-coin processing ─────────────────────────────────────────────────────

def process_coin_v3(symbol, btc_ts, btc_close):
    """Process one coin into fee-aware-labelled feature rows.

    Returns:
        X     : (k, 29) features
        y     : (k,)   {0,1}
        ts    : (k,)   ms timestamps (for temporal split + recency weights)
        tp    : (k,)   tp_pct of each label
        sl    : (k,)   sl_pct of each label
        netp  : (k,)   realised net% return at exit (oracle for EV diag)
    """
    pattern = os.path.join(PROCESSED_DIR, f"{symbol}_1m_features_*.parquet")
    files = sorted(glob.glob(pattern))
    if not files:
        return [np.empty((0,)) for _ in range(6)]

    dfs = []
    for f in files:
        try:
            dfs.append(pl.read_parquet(f))
        except Exception:
            pass
    if not dfs:
        return [np.empty((0,)) for _ in range(6)]

    df = pl.concat(dfs).sort("timestamp")
    close = df["close"].to_numpy().astype(np.float64)
    high  = df["high"].to_numpy().astype(np.float64)
    low   = df["low"].to_numpy().astype(np.float64)
    volume = df["volume"].to_numpy().astype(np.float64)
    ofi_raw = df["ofi"].to_numpy().astype(np.float64)
    ts_ms = df["timestamp"].cast(pl.Int64).to_numpy().astype(np.float64)
    n = len(close)
    print(f"  {symbol}: {n:,} bars, {len(files)} months", flush=True)

    btc_aligned = None
    if btc_ts is not None and btc_close is not None and symbol != "BTCUSDT":
        btc_aligned = np.interp(ts_ms, btc_ts.astype(np.float64), btc_close)

    F_v2, atr_arr = build_features_v2(close, high, low, volume, ofi_raw, ts_ms, btc_aligned)
    F_v3 = build_v3_from_v2(F_v2, close, high, low, atr_arr)

    hurst_raw = F_v2[:, 10].copy()
    idx = np.arange(WARMUP_BARS, n - FORWARD_BARS, SAMPLE_EVERY)
    F_samp = F_v3[idx]
    valid_feat = ~np.any(np.isnan(F_samp), axis=1)
    idx_v = idx[valid_feat]
    F_v = F_samp[valid_feat]
    if len(idx_v) == 0:
        return [np.empty((0,)) for _ in range(6)]

    y, tp_p, sl_p, net_p, _ = label_fee_aware(
        close, high, low, atr_arr, hurst_raw, idx_v, fwd=FORWARD_BARS
    )
    labelled = ~np.isnan(y)
    if labelled.sum() == 0:
        return [np.empty((0,)) for _ in range(6)]

    X_out  = F_v[labelled]
    y_out  = y[labelled]
    ts_out = ts_ms[idx_v[labelled]]
    tp_out = tp_p[labelled]
    sl_out = sl_p[labelled]
    net_out = net_p[labelled]

    wr = y_out.mean()
    rejected = (~labelled).sum()
    print(f"  {symbol}: {len(X_out):,} labelled ({rejected:,} rejected by fee filter) | WR {wr:.1%} | "
          f"avg net {net_out.mean()*100:+.3f}%", flush=True)
    return X_out, y_out, ts_out, tp_out, sl_out, net_out


# ── Recency weights anchored on global newest timestamp ─────────────────────

def compute_recency_weights(timestamps_ms, anchor_ts, half_life_months=12):
    age_months = (anchor_ts - timestamps_ms) / (30.44 * 24 * 3600 * 1000)
    age_months = np.maximum(age_months, 0)
    lam = np.log(2) / half_life_months
    w = np.exp(-lam * age_months)
    return w / w.mean()


# ── Threshold tuning by realised expectancy ─────────────────────────────────

def pick_threshold_by_ev(p_cal, tp_pct, sl_pct, net_pct,
                         min_trades=200, candidates=None):
    """Choose the probability threshold that maximises realised net expectancy
    on the validation slice, subject to a minimum number of trades.
    """
    if candidates is None:
        candidates = np.round(np.arange(0.50, 0.81, 0.01), 2)

    best = {"threshold": MIN_P_UP_DEFAULT, "ev_pct": -1e9, "n": 0,
            "win_rate": 0.0, "profit_factor": 0.0}
    table = []
    for thr in candidates:
        mask = p_cal >= thr
        n = int(mask.sum())
        if n < min_trades:
            table.append((float(thr), n, 0.0, 0.0, 0.0))
            continue
        nets = net_pct[mask]
        ev = float(nets.mean())
        wr = float((nets > 0).mean())
        wins = nets[nets > 0].sum()
        losses = -nets[nets <= 0].sum()
        pf = float(wins / max(losses, 1e-9))
        table.append((float(thr), n, ev * 100, wr * 100, pf))
        # Tie-break: higher EV, then more trades.
        if ev > best["ev_pct"] or (
            abs(ev - best["ev_pct"]) < 1e-12 and n > best["n"]
        ):
            best = {"threshold": float(thr), "ev_pct": ev * 100,
                    "n": n, "win_rate": wr * 100, "profit_factor": pf}

    print("\n  Threshold sweep (validation, net of fees):")
    print("  thr   trades       EV%      WR%       PF")
    for thr, n, ev_p, wr_p, pf in table:
        flag = " <<" if abs(thr - best["threshold"]) < 1e-9 else ""
        print(f"  {thr:0.2f}  {n:>6}  {ev_p:>+7.3f}  {wr_p:>5.1f}  {pf:>6.2f}{flag}")
    return best


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print("=" * 72)
    print("  EV Model V3 — fee-aware, EV-driven training")
    print(f"  Horizon {FORWARD_BARS} bars | round-trip cost {ROUND_TRIP_COST*100:.2f}%")
    print("=" * 72)

    print("\nLoading BTC reference...")
    btc_ts, btc_close = load_btc_close()
    if btc_ts is not None:
        print(f"  BTC: {len(btc_ts):,} bars")

    all_files = glob.glob(os.path.join(PROCESSED_DIR, "*.parquet"))
    symbols = sorted({os.path.basename(f).split("_1m_")[0] for f in all_files})
    print(f"Coins: {len(symbols)}\n")

    Xs, ys, tss, tps, sls, nets = [], [], [], [], [], []
    for sym in symbols:
        X, y, ts, tp, sl, net = process_coin_v3(sym, btc_ts, btc_close)
        if len(X) > 0:
            Xs.append(X); ys.append(y); tss.append(ts)
            tps.append(tp); sls.append(sl); nets.append(net)

    X = np.vstack(Xs)
    y = np.concatenate(ys)
    ts_all = np.concatenate(tss)
    tp_all = np.concatenate(tps)
    sl_all = np.concatenate(sls)
    net_all = np.concatenate(nets)

    # ── TEMPORAL walk-forward split (sort by timestamp globally) ────────────
    order = np.argsort(ts_all, kind="mergesort")
    X = X[order]; y = y[order]; ts_all = ts_all[order]
    tp_all = tp_all[order]; sl_all = sl_all[order]; net_all = net_all[order]

    n = len(X)
    train_end = int(n * 0.70)
    val_end   = int(n * 0.85)
    X_train, y_train, ts_train = X[:train_end], y[:train_end], ts_all[:train_end]
    X_val,   y_val,   ts_val   = X[train_end:val_end], y[train_end:val_end], ts_all[train_end:val_end]
    X_test,  y_test            = X[val_end:],  y[val_end:]
    tp_val, sl_val, net_val    = tp_all[train_end:val_end], sl_all[train_end:val_end], net_all[train_end:val_end]
    tp_test, sl_test, net_test = tp_all[val_end:], sl_all[val_end:], net_all[val_end:]

    import datetime as dt
    def _fmt(ms):
        return dt.datetime.utcfromtimestamp(ms/1000).strftime("%Y-%m-%d")
    print(f"\nTotal: {n:,} samples | features {X.shape[1]} | global WR {y.mean():.1%}")
    print(f"  train  {len(X_train):>8,}  {_fmt(ts_train[0])}  →  {_fmt(ts_train[-1])}")
    print(f"  val    {len(X_val):>8,}  {_fmt(ts_val[0])}  →  {_fmt(ts_val[-1])}")
    print(f"  test   {len(X_test):>8,}  {_fmt(ts_all[val_end])}  →  {_fmt(ts_all[-1])}")

    anchor_ts = float(ts_all.max())
    weights = compute_recency_weights(ts_train, anchor_ts, half_life_months=12)
    print(f"\nRecency weights: oldest={weights.min():.3f}, newest={weights.max():.3f}")

    # ── Train ───────────────────────────────────────────────────────────────
    print("\nTraining XGBoost...")
    dtrain = xgb.DMatrix(X_train, label=y_train, weight=weights,
                         feature_names=FEATURE_NAMES_V3)
    dval   = xgb.DMatrix(X_val,   label=y_val,   feature_names=FEATURE_NAMES_V3)
    dtest  = xgb.DMatrix(X_test,  label=y_test,  feature_names=FEATURE_NAMES_V3)

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
        "tree_method": "hist",
        "nthread": 4,
        "seed": 42,
        "verbosity": 1,
    }
    model = xgb.train(
        params, dtrain,
        num_boost_round=600,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=40,
        verbose_eval=50,
    )

    # ── Calibrate on validation ─────────────────────────────────────────────
    p_val_raw = model.predict(dval)
    from sklearn.isotonic import IsotonicRegression
    calibrator = IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds="clip")
    calibrator.fit(p_val_raw, y_val)
    p_val_cal = calibrator.predict(p_val_raw)

    from scipy.stats import spearmanr
    ic_raw, _ = spearmanr(p_val_raw, y_val)
    ic_cal, _ = spearmanr(p_val_cal, y_val)
    print(f"\nValidation IC: raw={ic_raw:.4f}  calibrated={ic_cal:.4f}")

    # Reliability check
    print("  Calibration reliability (predicted vs actual on val):")
    for lo, hi in [(0.45,0.55),(0.55,0.60),(0.60,0.65),(0.65,0.75),(0.75,1.00)]:
        m = (p_val_cal >= lo) & (p_val_cal < hi)
        if m.sum() > 30:
            print(f"    p∈[{lo:.2f},{hi:.2f})  n={m.sum():>5}  predicted~{p_val_cal[m].mean():.3f}  actual={y_val[m].mean():.3f}")

    # ── Pick threshold by realised EV on validation ────────────────────────
    best = pick_threshold_by_ev(p_val_cal, tp_val, sl_val, net_val,
                                min_trades=max(200, len(p_val_cal)//500))
    chosen_thr = best["threshold"]
    print(f"\n  Chosen threshold: p_up >= {chosen_thr:.2f}  "
          f"(val EV {best['ev_pct']:+.3f}% / trade, WR {best['win_rate']:.1f}%, "
          f"PF {best['profit_factor']:.2f}, n={best['n']})")

    # ── Test the chosen threshold on the truly held-out test slice ─────────
    p_test_raw = model.predict(dtest)
    p_test_cal = calibrator.predict(p_test_raw)
    mask_t = p_test_cal >= chosen_thr
    if mask_t.sum() > 0:
        nets_t = net_test[mask_t]
        ev_t = nets_t.mean() * 100
        wr_t = (nets_t > 0).mean() * 100
        wins_t = nets_t[nets_t > 0].sum()
        losses_t = -nets_t[nets_t <= 0].sum()
        pf_t = wins_t / max(losses_t, 1e-9)
        print(f"\n  Out-of-sample TEST @ thr {chosen_thr:.2f}: "
              f"n={int(mask_t.sum())}, EV {ev_t:+.3f}%, WR {wr_t:.1f}%, PF {pf_t:.2f}")
    else:
        ev_t = wr_t = pf_t = 0.0
        print("\n  No test trades cleared the threshold.")

    # ── Persist ─────────────────────────────────────────────────────────────
    os.makedirs(DATA_DIR, exist_ok=True)
    model.save_model(MODEL_PATH)
    with open(CALIB_PATH, "wb") as f:
        pickle.dump(calibrator, f)

    importance = model.get_score(importance_type="gain")
    sorted_imp = sorted(importance.items(), key=lambda x: -x[1])

    meta = {
        "model_type": "xgboost_v3_fee_aware",
        "feature_names": FEATURE_NAMES_V3,
        "n_features": len(FEATURE_NAMES_V3),
        "training_coins": symbols,
        "total_samples": int(n),
        "split": {"train": int(len(X_train)), "val": int(len(X_val)), "test": int(len(X_test))},
        "label_definition": {
            "type": "fee_aware_triple_barrier",
            "round_trip_cost": ROUND_TRIP_COST,
            "forward_bars": int(FORWARD_BARS),
            "atr_horizon_scale": "see trading_config.py",
        },
        "validation_ic_raw": float(ic_raw),
        "validation_ic_calibrated": float(ic_cal),
        "chosen_threshold": float(chosen_thr),
        "validation_ev_pct": best["ev_pct"],
        "validation_win_rate_pct": best["win_rate"],
        "validation_profit_factor": best["profit_factor"],
        "validation_n_trades": best["n"],
        "test_ev_pct": float(ev_t),
        "test_win_rate_pct": float(wr_t),
        "test_profit_factor": float(pf_t),
        "test_n_trades": int(mask_t.sum()) if mask_t is not None else 0,
        "min_ev_pct_required_for_trade": MIN_EV_PCT,
        "best_iteration": int(model.best_iteration),
        "feature_importance_top15": {k: round(v, 2) for k, v in sorted_imp[:15]},
        "recency_half_life_months": 12,
        "calibration": "isotonic_on_validation",
    }
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n[OK] model       -> {MODEL_PATH}")
    print(f"[OK] calibrator  -> {CALIB_PATH}")
    print(f"[OK] meta        -> {META_PATH}")
    print(f"  Total time: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
