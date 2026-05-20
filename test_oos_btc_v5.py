"""
trading_simulation.py - V5 out-of-sample backtest.

Same simulation skeleton as v4 (per-coin features → multi-horizon EV
selection → mean-variance sizing → regime gates) but powered by the v5
feature stack and v5 models:

    62 v4  + 12 perp  + 6 cross-sectional  + 14 coin-onehot  =  94 features

At startup we:
  1. Download spot klines for the 5 sim coins (existing behaviour).
  2. Auto-download perp data for the 5 sim coins from data.binance.vision
     into data/perp/ (skipped if already present).
  3. Build an in-memory basket-state table from the 5 sim coins for the
     cross-sectional features (sim basket = the 5 sim coins; this is the
     same definition the model would use live).
  4. Pre-compute v5 features per coin.
  5. Load v5 meta/models and run the decision loop.

Outputs simulation_v5_results.json with full per-trade audit trail.
"""
from __future__ import annotations
import os, sys, json, datetime, time, pickle
import concurrent.futures
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from trading_config import (
    CHECK_EVERY_MINS, HORIZONS, ROUND_TRIP_COST,
    horizon_atr_scale, regime_tp_mult, SL_MULT,
    MIN_TP_TO_COST_RATIO, MIN_EV_PCT,
    RISK_PER_TRADE_PCT, MAX_POSITION_PCT, KELLY_FRACTION,
    RISK_AVERSION_TARGET,
    MAX_OPEN_POSITIONS_TOTAL,
    DAILY_LOSS_HALT_PCT, WEEKLY_LOSS_HALT_PCT,
    HALT_REGIMES,
    MAX_SL_PCT, ASSET_MAX_POSITION_SCALE,
    HORIZON_REGIME_EV_SCALE,
    BTC_QUIET_4H_THRESHOLD, BTC_QUIET_EV_MULTIPLIER,
)
from features_v5 import (
    FEATURE_NAMES_V5, build_v5_full,
)
from regime_filter import classify_regime
import polars as pl

BINANCE_BASE = "https://api.binance.com"
DATA_DIR = os.path.join(os.path.dirname(__file__), "challenge_data")
PERP_DIR = os.path.join(os.path.dirname(__file__), "data", "perp")
COINS = ["BTCUSDT"]
STARTING_CAPITAL = 10000.0

COIN_ONEHOT_NAMES = [
    "AAVEUSDT", "ADAUSDT", "AVAXUSDT", "BNBUSDT", "BTCUSDT",
    "DOGEUSDT", "ETHUSDT", "LINKUSDT", "PEPEUSDT", "SHIBUSDT",
    "SOLUSDT", "UNIUSDT", "WIFUSDT", "XRPUSDT",
]
ALL_FEATURE_NAMES = FEATURE_NAMES_V5 + [f"coin_is_{s}" for s in COIN_ONEHOT_NAMES]


# ── Data loading (spot) ─────────────────────────────────────────────────────

def download_klines(symbol, start_ms, end_ms):
    import requests
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
        if len(candles) % 50000 < 1500:
            print(f"    {len(candles):,}...", flush=True)
        time.sleep(0.1)
    print(f"    done: {len(candles):,}")
    return candles


def load_or_download(symbol, start_ms, end_ms):
    os.makedirs(DATA_DIR, exist_ok=True)
    pkl_path = os.path.join(DATA_DIR, f"{symbol}_challenge.pkl")
    json_path = os.path.join(DATA_DIR, f"{symbol}_challenge.json")
    
    if os.path.exists(pkl_path):
        with open(pkl_path, "rb") as f: data = pickle.load(f)
        print(f"  {symbol}: cached ({len(data):,} candles)")
        return data
        
    # Auto-migrate slow JSONs to fast PKLs
    if os.path.exists(json_path):
        with open(json_path) as f: data = json.load(f)
        with open(pkl_path, "wb") as f: pickle.dump(data, f)
        os.remove(json_path)
        print(f"  {symbol}: migrated to fast cache ({len(data):,} candles)")
        return data
        
    data = download_klines(symbol, start_ms, end_ms)
    with open(pkl_path, "wb") as f: pickle.dump(data, f)
    return data


def get_month_boundaries(n_months=6):
    now = datetime.datetime.utcnow()
    end_month = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    months = []
    for i in range(n_months, 0, -1):
        y, m = end_month.year, end_month.month - i
        while m <= 0: m += 12; y -= 1
        start = datetime.datetime(y, m, 1)
        nm, ny = m + 1, y
        if nm > 12: nm, ny = 1, y + 1
        months.append((start, datetime.datetime(ny, nm, 1)))
    return months


# ── Perp data: ensure parquets exist for sim coins ──────────────────────────

def ensure_perp_data(symbols):
    """Download funding + metrics for each sim coin if missing."""
    missing = [s for s in symbols
               if not (os.path.exists(os.path.join(PERP_DIR, f"{s}_funding.parquet")) and
                       os.path.exists(os.path.join(PERP_DIR, f"{s}_metrics.parquet")))]
    if not missing:
        print("  Perp data: cached for all sim coins.")
        return
    print(f"  Perp data: downloading for {missing} ...")
    import download_perp_data as dp
    orig_coins = dp.COINS
    dp.COINS = missing
    try:
        dp.main()
    finally:
        dp.COINS = orig_coins


# ── Sim-time basket-state builder (in-memory, fast) ─────────────────────────

def _as_compact_arrays(coin_data):
    """Normalize coin_data into {sym: (ts_int64, cl_float32)} regardless of input form.

    Accepts either:
      - dict[sym] -> list[{"timestamp": int, "close": float, ...}]
      - dict[sym] -> (ts_int64_array, cl_float32_array)  ← preferred (low RAM)
    """
    out = {}
    for s, v in coin_data.items():
        if isinstance(v, tuple) and len(v) == 2:
            ts, cl = v
            out[s] = (np.asarray(ts, dtype=np.int64),
                      np.asarray(cl, dtype=np.float32))
        else:
            n = len(v)
            ts = np.fromiter((c["timestamp"] for c in v), dtype=np.int64, count=n)
            cl = np.fromiter((c["close"]     for c in v), dtype=np.float32, count=n)
            out[s] = (ts, cl)
    return out


def _ffill_inplace(a):
    """Vectorized forward-fill of NaNs in 1-D float array. Leading NaNs untouched."""
    mask = ~np.isnan(a)
    if not mask.any():
        return a
    idx = np.where(mask, np.arange(a.size, dtype=np.int64), 0)
    np.maximum.accumulate(idx, out=idx)
    a[:] = a[idx]
    return a


def _row_ranks_chunked(M, chunk=200_000):
    """Per-row percentile ranks of an (n,k) float matrix containing NaNs.
    Returns float32 matrix; rows with fewer than 2 finite values are all-NaN.
    Memory-bounded by `chunk` (peak ~chunk*k*8 bytes for argsort)."""
    n, k = M.shape
    out = np.full_like(M, np.nan, dtype=np.float32)
    col_idx = np.arange(k, dtype=np.int64)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        block = M[s:e]                         # view, no copy
        mask = ~np.isnan(block)
        kper = mask.sum(axis=1)
        # NaNs sort to the end -> they get the highest ranks but are masked out below
        filled = np.where(mask, block, np.inf).astype(np.float32, copy=False)
        order = np.argsort(filled, axis=1, kind="stable")
        ranks = np.empty_like(order)
        rows = np.arange(e - s, dtype=np.int64)[:, None]
        ranks[rows, order] = col_idx[None, :]
        denom = np.maximum(kper - 1, 1).astype(np.float32)
        block_out = ranks.astype(np.float32) / denom[:, None]
        block_out[~mask] = np.nan
        block_out[kper <= 1] = np.nan
        out[s:e] = block_out
    return out


def build_sim_basket_state(coin_data):
    """Lightweight cross-sectional basket state on a unified 1m grid.

    Memory-safe for 45 coins x 1.7M minutes:
      - accepts compact (ts, cl) tuples directly (no dict-list rebuild),
      - builds the timestamp union with numpy (no Python set of 76M ints),
      - forward-fills via np.maximum.accumulate (no Python per-bar loop),
      - computes per-row ranks in chunks (bounded peak RAM),
      - emits the long-form polars frame in one shot (no 45-frame concat).
    """
    print(f"\nBuilding sim-time basket state ({len(coin_data)}-coin universe)...")
    arrays = _as_compact_arrays(coin_data)
    syms = list(arrays.keys())
    n_syms = len(syms)
    if n_syms == 0:
        return pl.DataFrame(schema={
            "timestamp": pl.Int64, "symbol": pl.Utf8,
            "ret_rank_1h": pl.Float32, "ret_rank_4h": pl.Float32,
            "rv_rank_1h": pl.Float32, "coin_alpha_4h": pl.Float32,
            "basket_mom_4h": pl.Float32, "n_coins": pl.Float64,
        })

    # ── Unified timestamp grid (numpy union, no Python set) ────────────────
    ts_arr = np.unique(np.concatenate([arrays[s][0] for s in syms]))
    n = ts_arr.size
    print(f"  union grid: {n:,} unique minutes")

    # ── Build (n, n_syms) close matrix on the grid, forward-filled per coin ─
    cl_mat = np.full((n, n_syms), np.nan, dtype=np.float32)
    for ci, s in enumerate(syms):
        cts, ccl = arrays[s]
        # de-duplicate on timestamp (keep first occurrence)
        unq_ts, unq_idx = np.unique(cts, return_index=True)
        idx = np.searchsorted(ts_arr, unq_ts)
        col = np.full(n, np.nan, dtype=np.float32)
        col[idx] = ccl[unq_idx]
        _ffill_inplace(col)
        cl_mat[:, ci] = col
        del col, unq_ts, unq_idx, idx
    del arrays

    # ── Returns: r1h = log-diff over 60 bars, r4h = over 240 bars ──────────
    log_cl = np.log(np.maximum(cl_mat, 1e-12, dtype=np.float32))
    R1 = np.zeros_like(log_cl, dtype=np.float32)
    R4 = np.zeros_like(log_cl, dtype=np.float32)
    if n > 60:
        R1[60:]  = log_cl[60:]  - log_cl[:-60]
    if n > 240:
        R4[240:] = log_cl[240:] - log_cl[:-240]

    # ── Realised vol: 60-bar rolling std of 1m log-returns × sqrt(60) ──────
    V1 = np.zeros_like(log_cl, dtype=np.float32)
    if n > 60:
        win = 60
        for i in range(n_syms):
            m1 = np.zeros(n, dtype=np.float32)
            m1[1:] = log_cl[1:, i] - log_cl[:-1, i]
            c1 = np.cumsum(m1, dtype=np.float64)
            c2 = np.cumsum(m1 * m1, dtype=np.float64)
            sum1 = c1[win-1:] - np.concatenate([[0.0], c1[:-win]])
            sum2 = c2[win-1:] - np.concatenate([[0.0], c2[:-win]])
            mean = sum1 / win
            var = np.maximum(sum2 / win - mean * mean, 0.0)
            V1[win-1:, i] = (np.sqrt(var) * np.sqrt(win)).astype(np.float32)
    del log_cl, cl_mat

    # ── Cross-sectional aggregates ─────────────────────────────────────────
    n_coins = (~np.isnan(R1)).sum(axis=1).astype(np.float64)
    with np.errstate(invalid="ignore", all="ignore"):
        med_r4 = np.nan_to_num(np.nanmedian(R4, axis=1), nan=0.0).astype(np.float32)

    # ── Vectorized chunked rank computation (no Python per-row loop) ───────
    rk_r1 = _row_ranks_chunked(R1)
    rk_r4 = _row_ranks_chunked(R4)
    rk_v1 = _row_ranks_chunked(V1)

    # ── Long-form output: build column arrays once, single polars frame ────
    coin_alpha = (R4 - med_r4[:, None]).astype(np.float32)
    del R1, V1, R4
    # ── Long-form output: build column arrays one by one to avoid OOM ────
    valid = ~np.isnan(rk_r4)            # (n, n_syms) bool
    n_valid = int(valid.sum())
    if n_valid == 0:
        del rk_r1, rk_r4, rk_v1, coin_alpha, valid
        return pl.DataFrame(schema={
            "timestamp": pl.Int64, "symbol": pl.Utf8,
            "ret_rank_1h": pl.Float32, "ret_rank_4h": pl.Float32,
            "rv_rank_1h": pl.Float32, "coin_alpha_4h": pl.Float32,
            "basket_mom_4h": pl.Float32, "n_coins": pl.Float64,
        })

    import gc
    out_dict = {}

    # 1. Base columns
    ts_col = np.empty(n_valid, dtype=np.int64)
    sym_col = np.empty(n_valid, dtype=object)
    ncoins_col = np.empty(n_valid, dtype=np.float32)
    basket_col = np.empty(n_valid, dtype=np.float32)
    
    offset = 0
    for i in range(n_syms):
        mask = valid[:, i]
        n_i = mask.sum()
        if n_i == 0: continue
        (idx,) = np.nonzero(mask)
        ts_col[offset:offset+n_i] = ts_arr[idx]
        sym_col[offset:offset+n_i] = syms[i]
        ncoins_col[offset:offset+n_i] = n_coins[idx].astype(np.float32)
        basket_col[offset:offset+n_i] = med_r4[idx]
        offset += n_i

    out_dict["timestamp"] = ts_col
    out_dict["symbol"] = sym_col
    out_dict["n_coins"] = ncoins_col
    out_dict["basket_mom_4h"] = basket_col

    # 2. rk1
    rk1_col = np.empty(n_valid, dtype=np.float32)
    offset = 0
    for i in range(n_syms):
        mask = valid[:, i]
        n_i = mask.sum()
        if n_i > 0:
            rk1_col[offset:offset+n_i] = rk_r1[mask, i]
            offset += n_i
    del rk_r1; gc.collect()
    out_dict["ret_rank_1h"] = rk1_col

    # 3. rk4
    rk4_col = np.empty(n_valid, dtype=np.float32)
    offset = 0
    for i in range(n_syms):
        mask = valid[:, i]
        n_i = mask.sum()
        if n_i > 0:
            rk4_col[offset:offset+n_i] = rk_r4[mask, i]
            offset += n_i
    del rk_r4; gc.collect()
    out_dict["ret_rank_4h"] = rk4_col

    # 4. rkv
    rkv_col = np.empty(n_valid, dtype=np.float32)
    offset = 0
    for i in range(n_syms):
        mask = valid[:, i]
        n_i = mask.sum()
        if n_i > 0:
            rkv_col[offset:offset+n_i] = rk_v1[mask, i]
            offset += n_i
    del rk_v1; gc.collect()
    out_dict["rv_rank_1h"] = rkv_col

    # 5. alpha
    alpha_col = np.empty(n_valid, dtype=np.float32)
    offset = 0
    for i in range(n_syms):
        mask = valid[:, i]
        n_i = mask.sum()
        if n_i > 0:
            alpha_col[offset:offset+n_i] = coin_alpha[mask, i]
            offset += n_i
    del coin_alpha, valid; gc.collect()
    out_dict["coin_alpha_4h"] = alpha_col

    out = pl.DataFrame(out_dict)
    del out_dict
    gc.collect()
    print(f"  basket state: {out.height:,} rows over {n} unique minutes")
    return out


# ── Per-coin feature precompute ─────────────────────────────────────────────

def _coin_onehot_block(symbol, n):
    block = np.zeros((n, len(COIN_ONEHOT_NAMES)), dtype=np.float32)
    if symbol in COIN_ONEHOT_NAMES:
        block[:, COIN_ONEHOT_NAMES.index(symbol)] = 1.0
    return block


def precompute_features(coin_syms, btc_ts_arr, btc_cl_arr, basket_state,
                        load_candles_fn, feats_cache_dir):
    """Per-coin v5 feature precompute (low-RAM streaming).

    Args:
      coin_syms: list of symbol names.
      btc_ts_arr, btc_cl_arr: BTC reference arrays (numpy int64 / float64).
        These replace the old dict-list ``btc_data`` (~850 MB → ~28 MB).
      basket_state: cross-sectional polars frame from build_sim_basket_state().
      load_candles_fn: callable(sym) -> list[dict]. ONLY invoked when a coin's
        feature cache is missing. After extracting numpy arrays we immediately
        drop the dict list and gc, so peak RAM stays around one coin's worth.
      feats_cache_dir: directory for ``{sym}_v5_meta.pkl`` + ``{sym}_v5_F.npy``.
    """
    print("\nPre-computing v5 features...")
    from train_ev_model_v2 import build_features_v2
    import gc

    btc_ts_f64 = np.asarray(btc_ts_arr, dtype=np.float64)
    btc_cl_f64 = np.asarray(btc_cl_arr, dtype=np.float64)
    out = {}

    os.makedirs(feats_cache_dir, exist_ok=True)

    for sym in coin_syms:
        coin_meta_path = os.path.join(feats_cache_dir, f"{sym}_v5_meta.pkl")
        coin_f_path    = os.path.join(feats_cache_dir, f"{sym}_v5_F.npy")

        # Auto-purge legacy RAM-heavy pickle caches
        old_cache_path = os.path.join(feats_cache_dir, f"{sym}_v5_feats.pkl")
        if os.path.exists(old_cache_path):
            os.remove(old_cache_path)

        if os.path.exists(coin_meta_path) and os.path.exists(coin_f_path):
            with open(coin_meta_path, "rb") as f:
                out[sym] = pickle.load(f)
            # mmap the 100MB+ matrix → 0 RAM
            out[sym]["F"] = np.load(coin_f_path, mmap_mode='r')
            print(f"  {sym}: loaded from cache ({len(out[sym]['ts']):,} bars) [mmap]")
            continue

        # ── Cache miss: load candles for THIS coin only, free immediately. ──
        print(f"  {sym}: cache miss — loading full candles...")
        candles = load_candles_fn(sym)
        if not candles or len(candles) < 2000:
            print(f"  {sym}: skipping (insufficient data: {len(candles) if candles else 0})")
            del candles
            gc.collect()
            continue

        t0 = time.time()
        n_c = len(candles)
        ts     = np.fromiter((c["timestamp"]               for c in candles), dtype=np.float64, count=n_c)
        hi     = np.fromiter((c["high"]                    for c in candles), dtype=np.float32, count=n_c)
        lo     = np.fromiter((c["low"]                     for c in candles), dtype=np.float32, count=n_c)
        cl     = np.fromiter((c["close"]                   for c in candles), dtype=np.float32, count=n_c)
        vo     = np.fromiter((c["volume"]                  for c in candles), dtype=np.float32, count=n_c)
        tbv    = np.fromiter((c.get("taker_buy_volume", 0) for c in candles), dtype=np.float32, count=n_c)
        ts_int = np.fromiter((c["timestamp"]               for c in candles), dtype=np.int64,   count=n_c)
        # Free the ~850MB Python dict list before building features
        del candles
        gc.collect()

        ofi = (2.0 * tbv - vo).astype(np.float32)
        btc_aligned = np.interp(ts, btc_ts_f64, btc_cl_f64)
        F_v2, atr = build_features_v2(cl, hi, lo, vo, ofi, ts, btc_aligned)
        F_feats = build_v5_full(sym, F_v2, cl, hi, lo, vo, ofi, tbv, ts, atr,
                                btc_aligned, PERP_DIR, basket_state)
        F_full = np.hstack([F_feats, _coin_onehot_block(sym, len(cl))]).astype(np.float32)
        hurst = F_v2[:, 10].copy().astype(np.float32)

        out[sym] = {
            "ts":  ts_int,
            "hi": hi, "lo": lo, "cl": cl,
            "atr": atr.astype(np.float32), "hurst": hurst,
        }

        with open(coin_meta_path, "wb") as f:
            pickle.dump(out[sym], f)
        np.save(coin_f_path, F_full)

        # Drop big intermediates before next coin
        del F_full, F_feats, F_v2, ofi, btc_aligned, vo, tbv, ts
        gc.collect()
        out[sym]["F"] = np.load(coin_f_path, mmap_mode='r')
        print(f"  {sym}: {n_c:,} bars × 94 feats  ({time.time()-t0:.1f}s)")
    return out


# ── Model loading ───────────────────────────────────────────────────────────

def load_v5_models():
    import xgboost as xgb
    here = os.path.dirname(__file__)
    meta_path = os.path.join(here, "data", "ev_model_v5_meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"v5 meta not found: {meta_path}\n"
                                "Run `python train_v5.py` first.")
    with open(meta_path) as f: meta = json.load(f)
    horizons = meta["horizons"]
    models = {}
    for h in horizons:
        r_group = meta["horizon_results"].get(str(h))
        if r_group is None:
            continue
        models[h] = {}
        for direction in ["long", "short"]:
            r = r_group.get(direction)
            if r is None:
                continue
            clf = xgb.Booster(); clf.load_model(os.path.join(here, r["clf_path"]))
            reg = xgb.Booster(); reg.load_model(os.path.join(here, r["reg_path"]))
            with open(os.path.join(here, r["calib_path"]), "rb") as f:
                calib = pickle.load(f)
            models[h][direction] = {"clf": clf, "reg": reg, "calib": calib,
                         "p_threshold": float(r["p_threshold"])}
            print(f"  h={h:>4}m ({direction}) thr={r['p_threshold']:.2f}  "
                  f"val_EV={r['val_ev_pct']:+.3f}%  test_EV={r['test_ev_pct']:+.3f}%  "
                  f"regIC={r['regressor_rank_ic']:+.3f}")
    return meta, models


# ── Indices for regime classifier ───────────────────────────────────────────

def _name_idx(name):
    return ALL_FEATURE_NAMES.index(name)

IDX_HURST     = _name_idx("hurst")
IDX_TE        = _name_idx("trend_efficiency")
IDX_RANGE_POS = _name_idx("range_position")
IDX_RV1H      = _name_idx("rv_1h")
IDX_RV_Z_1D   = _name_idx("rv_zscore_1d")
IDX_VOL_BURST = _name_idx("vol_burst_1h")


# ── Sizing ──────────────────────────────────────────────────────────────────

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


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    global COINS
    print("=" * 78)
    print("  ORACLE V5 SIMULATION — perp + cross-sectional + regime-gated")
    # Will update COINS dynamically later if loaded from cache, but printing current COINS here for now.
    print(f"  Capital ${STARTING_CAPITAL:,.0f}  coins {', '.join(COINS)}")
    cadence_str = f"{CHECK_EVERY_MINS}m" if CHECK_EVERY_MINS < 60 else f"{CHECK_EVERY_MINS//60}h"
    print(f"  Decision every {cadence_str}  horizons {HORIZONS}m  "
          f"cost {ROUND_TRIP_COST*100:.2f}%")
    print("=" * 78)

    # TRUE-OOS window: fully after the v5 training set (train ends 2025-04-27).
    # Both trained coins (14) AND untrained coins (31) are date-OOS here.
    d_start = datetime.datetime.strptime("2025-02-01", "%Y-%m-%d")
    d_end   = datetime.datetime.strptime("2025-05-01", "%Y-%m-%d")
    overall_start = int(d_start.timestamp() * 1000)
    overall_end   = int(d_end.timestamp() * 1000)

    # ── LOW-RAM data loading ────────────────────────────────────────────────
    # Original code did `coin_data = {sym: load_or_download(...) for sym in COINS}`
    # which materialised every coin (~850 MB / 1.7M-candle PKL) simultaneously.
    # On 21-month × 45-coin runs this allocates ~38 GB of Python dicts → OOM.
    # We now (1) PEEK each PKL for length, freeing immediately;
    #       (2) compact-load BTC as numpy ts+cl only;
    #       (3) lazy-load full candles per-coin only on cache miss.
    import gc

    print("\nScanning spot data (low-RAM peek)...")
    candle_counts = {}
    for sym in list(COINS):
        pkl_path  = os.path.join(DATA_DIR, f"{sym}_challenge.pkl")
        json_path = os.path.join(DATA_DIR, f"{sym}_challenge.json")
        n = 0
        if os.path.exists(pkl_path):
            try:
                with open(pkl_path, "rb") as f:
                    _d = pickle.load(f)
                n = len(_d) if isinstance(_d, list) else 0
                del _d
                gc.collect()
            except Exception as ex:
                print(f"  {sym}: peek failed ({ex})")
                n = 0
        elif os.path.exists(json_path):
            # JSON migration path is rare; do the full load+cache once, then drop.
            _d = load_or_download(sym, overall_start, overall_end)
            n = len(_d) if isinstance(_d, list) else 0
            del _d
            gc.collect()
        candle_counts[sym] = n
        print(f"  {sym}: {n:,} candles")

    print("\nEnsuring perp data for sim coins...")
    ensure_perp_data(COINS)

    # Filter incomplete data (no full candles in RAM)
    valid_coins = []
    for sym in list(COINS):
        if candle_counts.get(sym, 0) < 10000:
            print(f"  Removing {sym} (insufficient spot data)")
            continue
        f_path = os.path.join(PERP_DIR, f"{sym}_funding.parquet")
        m_path = os.path.join(PERP_DIR, f"{sym}_metrics.parquet")
        if not os.path.exists(f_path) or not os.path.exists(m_path):
            print(f"  Removing {sym} (missing perp data)")
            continue
        valid_coins.append(sym)
    COINS = valid_coins

    # Compact BTC reference: ts + close as numpy (replaces ~850 MB dict-list)
    print("\nCompact-loading BTC reference (ts + close only)...")
    _btc_pkl = os.path.join(DATA_DIR, "BTCUSDT_challenge.pkl")
    if os.path.exists(_btc_pkl):
        with open(_btc_pkl, "rb") as f:
            _btc_raw = pickle.load(f)
    else:
        _btc_raw = load_or_download("BTCUSDT", overall_start, overall_end)
    _n_btc = len(_btc_raw)
    btc_ts_arr = np.fromiter((c["timestamp"] for c in _btc_raw), dtype=np.int64,   count=_n_btc)
    btc_cl_arr = np.fromiter((c["close"]     for c in _btc_raw), dtype=np.float64, count=_n_btc)
    del _btc_raw
    gc.collect()
    print(f"  BTC: {_n_btc:,} bars  ({(btc_ts_arr.nbytes + btc_cl_arr.nbytes) / 1e6:.1f} MB)")

    # Lazy loader — only invoked by precompute_features / basket-build on cache miss
    def _lazy_load_full(sym):
        return load_or_download(sym, overall_start, overall_end)

    # Start trading almost immediately (14-day warmup)
    study_end_ms = overall_start + (14 * 24 * 60 * 60 * 1000)
    trade_end_ms = overall_end

    d_start = datetime.datetime.utcfromtimestamp(overall_start/1000)
    d_study = datetime.datetime.utcfromtimestamp(study_end_ms/1000)
    d_trade = datetime.datetime.utcfromtimestamp(trade_end_ms/1000)

    print(f"\n  Study:  {d_start.strftime('%b %Y')} - {d_study.strftime('%b %Y')}")
    print(f"  Trade:  {d_study.strftime('%b %Y')} - {d_trade.strftime('%b %Y')}")

    feats_cache_dir = r"E:\training data for quant\sim_cache_oos\per_coin_feats"
    os.makedirs(feats_cache_dir, exist_ok=True)
    basket_cache_path = os.path.join(feats_cache_dir, "basket_state.pkl")

    if os.path.exists(basket_cache_path):
        print("\nLoading basket state from cache...")
        with open(basket_cache_path, "rb") as f:
            basket_state = pickle.load(f)
    else:
        # Stream compact (ts, cl) tuples per coin — basket builder accepts them natively.
        print("\nBasket cache MISS — streaming compact (ts, close) per coin...")
        coin_compact = {}
        _total_mb = 0.0
        for sym in COINS:
            pkl_path = os.path.join(DATA_DIR, f"{sym}_challenge.pkl")
            with open(pkl_path, "rb") as f:
                raw = pickle.load(f)
            n = len(raw)
            ts = np.fromiter((c["timestamp"] for c in raw), dtype=np.int64,   count=n)
            cl = np.fromiter((c["close"]     for c in raw), dtype=np.float32, count=n)
            del raw
            gc.collect()
            coin_compact[sym] = (ts, cl)
            _total_mb += (ts.nbytes + cl.nbytes) / 1e6
            print(f"  {sym}: {n:,} compact rows ({_total_mb:.0f} MB cumulative)")
        basket_state = build_sim_basket_state(coin_compact)
        del coin_compact
        gc.collect()
        with open(basket_cache_path, "wb") as f:
            pickle.dump(basket_state, f)

    feats = precompute_features(COINS, btc_ts_arr, btc_cl_arr, basket_state,
                                _lazy_load_full, feats_cache_dir)

    # Speed: Pre-load all mmap feature arrays fully into RAM.
    # Each coin F matrix is ~100MB; total ~2-4GB for 55 coins.
    # On low-end hardware we load only if available memory allows.
    # We check by loading one and measuring, then decide.
    print("\nPromoting mmap arrays to RAM for max speed...")
    import ctypes
    def get_available_ram_gb():
        try:
            import ctypes.wintypes
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong),
                            ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_uint64),
                            ("ullAvailPhys", ctypes.c_uint64),
                            ("ullTotalPageFile", ctypes.c_uint64),
                            ("ullAvailPageFile", ctypes.c_uint64),
                            ("ullTotalVirtual", ctypes.c_uint64),
                            ("ullAvailVirtual", ctypes.c_uint64),
                            ("ullAvailExtendedVirtual", ctypes.c_uint64)]
            stat = MEMORYSTATUSEX(); stat.dwLength = ctypes.sizeof(stat)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            return stat.ullAvailPhys / 1e9
        except Exception:
            return 2.0  # conservative fallback
    avail_gb = get_available_ram_gb()
    total_feat_gb = sum(f["F"].nbytes for f in feats.values()) / 1e9
    print(f"  Available RAM: {avail_gb:.1f}GB  Feature arrays: {total_feat_gb:.1f}GB")
    if avail_gb > total_feat_gb * 1.5:  # need 1.5x headroom
        for sym in feats:
            feats[sym]["F"] = np.array(feats[sym]["F"])  # force full RAM copy
        print(f"  Promoted {len(feats)} coins to RAM (fast random access)")
    else:
        print(f"  Keeping mmap (not enough RAM headroom)")

    print("\nLoading v5 models...")
    meta, models = load_v5_models()
    if not models:
        print("ERROR: no models loaded. Run train_v5.py first."); return

    import xgboost as xgb
    feat_names = meta["feature_names"]

    check_ms = CHECK_EVERY_MINS * 60 * 1000
    check_times = list(range(study_end_ms, trade_end_ms, check_ms))
    print(f"\n  Decision points: {len(check_times)} × {len(COINS)} coins")

    # OPT-A (i3): No ThreadPoolExecutor.
    # Previous code submitted 4 horizon predicts in parallel via a persistent pool,
    # but XGBoost.inplace_predict already uses all cores via OpenMP. On a 2-core i3
    # this stacked 4×OMP_threads against 2 hardware threads → kernel thrashing and
    # thermal throttling, *plus* per-iteration future.submit / future.result overhead
    # (~0.2-0.5 ms × 4 × 90k iters = 72-180s wasted). Serial predicts are faster on
    # weak hardware and produce byte-identical outputs.

    # Speed: Pre-allocate a reusable batch buffer for max 55 coins × 94 features.
    # This avoids a fresh np.ascontiguousarray allocation every 10 minutes.
    _MAX_BATCH = len(COINS) + 5
    _N_FEATS = len(ALL_FEATURE_NAMES)
    _batch_buf = np.zeros((_MAX_BATCH, _N_FEATS), dtype=np.float32)
    # OPT-D (i3): Hoist HORIZONS-derived constants out of the 4M-iter inner path.
    _MAX_H = max(HORIZONS)
    _HORIZON_MODELS = [(h, m) for h, m in models.items() if h in HORIZONS]

    # Feature 3: Pre-compute BTC 4h momentum lookup aligned to check_times.
    # This tells us if the market is in an active or quiet state at each decision point.
    # Built once before the loop: O(N) searchsorted, zero cost inside the loop.
    print("\nBuilding BTC momentum filter...")
    # Reuse the compact BTC arrays loaded above (no dict-list re-extraction).
    _btc_ts = btc_ts_arr
    _btc_cl = btc_cl_arr.astype(np.float32, copy=False)
    # OPT-1: Fully vectorized — no Python loop, no per-step searchsorted.
    # Align every check_time to a BTC index in one batch call, then index.
    _ct_arr  = np.array(check_times, dtype=np.int64)
    _bi_arr  = np.searchsorted(_btc_ts, _ct_arr, side="right") - 1   # shape (N,)
    _valid   = (_bi_arr >= 240) & (_bi_arr < len(_btc_cl))
    _btc_4h_mom = np.zeros(len(check_times), dtype=np.float32)
    _v_idx  = _bi_arr[_valid]
    _btc_4h_mom[_valid] = (_btc_cl[_v_idx] - _btc_cl[_v_idx - 240]) / np.maximum(_btc_cl[_v_idx - 240], 1e-9)
    print(f"  BTC momentum filter: {(np.abs(_btc_4h_mom) < BTC_QUIET_4H_THRESHOLD).mean()*100:.1f}% of decision points are 'quiet'")

    capital = STARTING_CAPITAL
    peak = capital; max_dd = 0.0; fees_paid = 0.0
    all_trades = []
    open_until_ts = {sym: 0 for sym in COINS}

    DAY_MS = 24*60*60*1000; WEEK_MS = 7*DAY_MS
    day_start_ms = study_end_ms; day_start_cap = capital
    week_start_ms = study_end_ms; week_start_cap = capital
    week_num = 0; weekly = []
    day_halted = False; week_halted = False
    # OPT-3: O(1) per-week trade counter — avoids scanning all_trades each week.
    _week_trade_count = 0
    regime_counts = {}

    print("\n" + "=" * 78)
    print("  RUNNING")
    print("=" * 78)

    # Logging counters
    filter_stats = {
        "position_open": 0,
        "insufficient_data": 0,
        "non_finite_features": 0,
        "invalid_atr": 0,
        "halt_regime": 0,
        "tp_too_small": 0,
        "p_below_threshold": 0,
        "ev_below_threshold": 0,
        "candidates_passed": 0,
    }

    # Fast pointer cache to avoid 1.6 million binary searches
    coin_idx_map = {sym: 1500 for sym in COINS}

    for ci, check_ts in enumerate(check_times):
        while check_ts >= day_start_ms + DAY_MS:
            day_start_ms += DAY_MS; day_start_cap = capital; day_halted = False
        while check_ts >= week_start_ms + WEEK_MS:
            week_num += 1
            wk_pnl = capital - week_start_cap
            n_wk = _week_trade_count          # OPT-3: O(1) read
            _week_trade_count = 0              # reset for next week
            d = datetime.datetime.utcfromtimestamp(week_start_ms/1000)
            print(f"  week {week_num:>2} ({d.strftime('%b %d')}): "
                  f"{n_wk:>3} trades  PnL ${wk_pnl:>+8,.2f}  cap ${capital:>10,.2f}", flush=True)
            weekly.append({"week": week_num, "capital": round(capital,2),
                           "pnl": round(wk_pnl,2), "trades": n_wk})
            week_start_ms += WEEK_MS; week_start_cap = capital; week_halted = False

        if not day_halted and (day_start_cap - capital)/max(day_start_cap,1e-9) >= DAILY_LOSS_HALT_PCT:
            day_halted = True
        if not week_halted and (week_start_cap - capital)/max(week_start_cap,1e-9) >= WEEKLY_LOSS_HALT_PCT:
            week_halted = True
        if day_halted or week_halted:
            continue

        candidates = []
        
        # ── BATCH PREPARATION ──
        valid_cands = []
        feats_list = []
        
        for sym in COINS:
            if open_until_ts[sym] > check_ts:                       # OPT-C: dict[]
                filter_stats["position_open"] += 1
                continue
            if sym not in feats:
                continue
            d = feats[sym]
            ts_arr = d["ts"]
            n_ts = len(ts_arr)

            # Fast-path: check_ts increments by exactly 10 minutes (600,000ms).
            # If candles are perfect 1m, the index advances by exactly 10.
            curr_idx = coin_idx_map[sym]
            guess = curr_idx + 10
            if guess < n_ts and ts_arr[guess] == check_ts:
                idx = guess
            else:
                idx = int(np.searchsorted(ts_arr, check_ts, side="right") - 1)
            coin_idx_map[sym] = max(0, idx)

            if idx < 1500 or idx + _MAX_H >= n_ts:                  # OPT-D: hoist max()
                filter_stats["insufficient_data"] += 1
                continue
            f_row = d["F"][idx]
            if not np.isfinite(f_row[0]): # Just check the first feature for speed
                filter_stats["non_finite_features"] += 1
                continue
            atr_val = d["atr"][idx]
            if atr_val <= 0:
                filter_stats["invalid_atr"] += 1
                continue
            entry = float(d["cl"][idx])
            hurst = float(d["hurst"][idx])

            # Inlined regime classification (0 function call overhead)
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

            regime_counts[regime] = regime_counts.get(regime, 0) + 1
            if regime in HALT_REGIMES:
                filter_stats["halt_regime"] += 1
                continue
                
            valid_cands.append({"sym": sym, "idx": idx, "entry": entry, "atr_val": atr_val, "hurst": hurst, "regime": regime})
            feats_list.append(f_row)

        # ── BATCH PREDICTION ──
        if len(valid_cands) > 0:
            # OPT-B (i3): Single C-level stack into the preallocated buffer.
            # Replaces a Python `for _bi, _frow in enumerate(...): _batch_buf[_bi]=_frow`
            # which executed up to 45 attribute+index ops per outer iter (~4M total).
            n_cands = len(valid_cands)
            np.stack(feats_list, out=_batch_buf[:n_cands])
            stacked_feats = _batch_buf[:n_cands]  # zero-copy slice view

            # OPT-A (i3): serial horizon predicts.
            # XGBoost.inplace_predict is internally multithreaded (OpenMP); on a
            # 2-core CPU, running 4 of them concurrently produces oversubscription,
            # cache thrashing and thermal throttling. Serial execution is faster
            # AND deterministic (results identical to threaded version).
            for h, m_group in _HORIZON_MODELS:
                for direction in ["long", "short"]:
                    m = m_group.get(direction)
                    if m is None: continue
                    p_raw      = m["clf"].inplace_predict(stacked_feats)
                    p_cal_arr  = m["calib"].predict(p_raw)
                    mu_hat_arr = m["reg"].inplace_predict(stacked_feats)
                    scale = horizon_atr_scale(h)
                    thr_override = 0.52 if h == 720 else 0.60
                    
                    for i, cand in enumerate(valid_cands):
                        tp_mult = regime_tp_mult(cand["hurst"]) * scale
                        sl_mult = SL_MULT * scale
                        tp_pct = tp_mult * cand["atr_val"] / cand["entry"]
                        sl_pct = sl_mult * cand["atr_val"] / cand["entry"]
                        sl_pct = min(sl_pct, MAX_SL_PCT)
                        
                        if tp_pct < MIN_TP_TO_COST_RATIO * ROUND_TRIP_COST:
                            filter_stats["tp_too_small"] += 1
                            continue
                            
                        p_cal = float(p_cal_arr[i])
                        if p_cal < thr_override:
                            filter_stats["p_below_threshold"] += 1
                            continue
                            
                        mu_hat = float(mu_hat_arr[i])
                        if direction == "short":
                            mu_hat = -mu_hat
                            
                        ev = p_cal * tp_pct - (1 - p_cal) * sl_pct - ROUND_TRIP_COST
                        if ev < MIN_EV_PCT:
                            filter_stats["ev_below_threshold"] += 1
                            continue

                        _h_scale = HORIZON_REGIME_EV_SCALE.get(cand["regime"], {}).get(h, 1.0)
                        _effective_ev = ev * _h_scale
                        
                        if direction == "long":
                            tp_price = cand["entry"] + tp_mult * cand["atr_val"]
                            sl_price = cand["entry"] * (1.0 - sl_pct)
                        else:
                            tp_price = cand["entry"] - tp_mult * cand["atr_val"]
                            sl_price = cand["entry"] * (1.0 + sl_pct)

                        candidates.append({
                            "sym": cand["sym"], "idx": cand["idx"], "entry": cand["entry"],
                            "horizon": h, "direction": direction, "p_up": p_cal, "mu_hat": mu_hat,
                            "tp_pct": tp_pct, "sl_pct": sl_pct,
                            "tp_price": tp_price,
                            "sl_price": sl_price,
                            "ev": ev, "effective_ev": _effective_ev,
                            "hurst": cand["hurst"], "regime": cand["regime"],
                        })
                        
                        filter_stats["candidates_passed"] += 1
        # Feature 3: Market Momentum Filter
        # Elevate the EV bar during BTC quiet periods to suppress low-conviction setups.
        _btc_mom = _btc_4h_mom[ci]
        _active_min_ev = MIN_EV_PCT * (BTC_QUIET_EV_MULTIPLIER if abs(_btc_mom) < BTC_QUIET_4H_THRESHOLD else 1.0)
        
        candidates = [c for c in candidates if c["ev"] >= _active_min_ev]

        candidates.sort(key=lambda c: -c["effective_ev"])  # Feature 2: sort by regime-weighted EV
        open_now = sum(1 for t in open_until_ts.values() if t > check_ts)
        budget = max(0, MAX_OPEN_POSITIONS_TOTAL - open_now)

        # Log when no candidates found (debug why trading stopped)
        if len(candidates) == 0 and ci < 100:
            d_ts = datetime.datetime.utcfromtimestamp(check_ts/1000)
            print(f"    NO CANDIDATES at {d_ts.strftime('%Y-%m-%d %H:%M')} (week {week_num+1}) - "
                  f"open_positions={open_now}", flush=True)

        # Portfolio Science: Correlation-Adjusted Sizing
        # If the model finds multiple trades at the exact same minute, they are likely
        # highly correlated to a macro momentum shift (Beta). We scale down the risk 
        # using the square root of the number of simultaneous trades to prevent over-exposure.
        num_simultaneous = len(candidates[:budget])
        correlation_discount = 1.0 / np.sqrt(num_simultaneous) if num_simultaneous > 0 else 1.0

        for c in candidates[:budget]:
            arr = feats[c["sym"]]; idx = c["idx"]

            # Fix 3: Restore hybrid sigma for sizing only.
            # This only fires ~94x per full run (negligible speed cost) but protects
            # against ATR underestimating true volatility on gap-spike coins like ORDI.
            window = max(min(c["horizon"], idx), 5)
            log_w = np.diff(np.log(np.maximum(arr["cl"][idx-window:idx+1], 1e-12)))
            realized_sigma = float(np.std(log_w) * np.sqrt(c["horizon"])) if len(log_w) > 1 else c["sl_pct"]
            atr_sigma = float(c["sl_pct"] / SL_MULT)
            sigma_pct = max(realized_sigma, atr_sigma, 1e-4)

            mu_pct = max(c["mu_hat"], 0.0)

            # Base position size
            position = mean_variance_size(
                capital, mu_pct, sigma_pct, c["p_up"], c["tp_pct"], c["sl_pct"]
            )
            # Fix 2: Per-asset position cap for known high-risk coins
            asset_scale = ASSET_MAX_POSITION_SCALE.get(c["sym"], 1.0)
            position *= asset_scale
            # Fix 1 (sizing side): also cap position via SL-implied risk
            # (sl_pct is already hard-capped at MAX_SL_PCT from candidate building)
            # Apply Correlation Discount
            position *= correlation_discount

            # --- DYNAMIC FUTURES LEVERAGE ---
            # Minimum 20x leverage. Scales up to 50x based on model confidence (p_up).
            confidence_surplus = max(0.0, c["p_up"] - 0.59)
            leverage = 20.0 + (confidence_surplus / 0.11) * 30.0 # At p_up=0.70+, lev=50x
            leverage = min(50.0, max(20.0, leverage))
            
            # The 'position' variable previously represented spot size. 
            # Now it represents margin. We cap margin to available capital.
            margin_required = min(position, capital)
            
            # The new trading position is the leveraged notional amount
            position = margin_required * leverage

            if position <= 0:
                continue

            end_idx = min(idx + c["horizon"], len(arr["cl"]) - 1)
            fwd_hi = arr["hi"][idx+1:end_idx+1]
            fwd_lo = arr["lo"][idx+1:end_idx+1]
            if c["direction"] == "long":
                tp_hits = np.where(fwd_hi >= c["tp_price"])[0]
                sl_hits = np.where(fwd_lo <= c["sl_price"])[0]
            else:
                tp_hits = np.where(fwd_lo <= c["tp_price"])[0]
                sl_hits = np.where(fwd_hi >= c["sl_price"])[0]
            INF = np.iinfo(np.int32).max
            tp_t = tp_hits[0] if len(tp_hits) else INF
            sl_t = sl_hits[0] if len(sl_hits) else INF
            if tp_t < sl_t:
                if c["horizon"] >= 240:
                    # ── PARTIAL SCALING LOGIC (TP1 + RUNNER) ──
                    runner_hi = fwd_hi[tp_t:]
                    runner_lo = fwd_lo[tp_t:]
                    
                    if c["direction"] == "long":
                        tp2_price = c["entry"] * (1.0 + c["tp_pct"] * 1.5)
                        be_sl_price = c["entry"] * (1.0 + ROUND_TRIP_COST)
                        hit_tp2 = np.where(runner_hi >= tp2_price)[0]
                        hit_be = np.where(runner_lo <= be_sl_price)[0]
                    else:
                        tp2_price = c["entry"] * (1.0 - c["tp_pct"] * 1.5)
                        be_sl_price = c["entry"] * (1.0 - ROUND_TRIP_COST)
                        hit_tp2 = np.where(runner_lo <= tp2_price)[0]
                        hit_be = np.where(runner_hi >= be_sl_price)[0]
                    
                    t2_t = hit_tp2[0] if len(hit_tp2) else INF
                    be_t = hit_be[0] if len(hit_be) else INF
                    
                    if t2_t == INF and be_t == INF:
                        runner_exit = float(arr["cl"][end_idx])
                        if c["direction"] == "long":
                            runner_gross = (runner_exit - c["entry"]) / c["entry"]
                        else:
                            runner_gross = (c["entry"] - runner_exit) / c["entry"]
                        reason = "partial_tp1_time_exit"
                        close_offset = len(fwd_hi) - 1
                    elif t2_t < be_t:
                        runner_gross = c["tp_pct"] * 1.5
                        reason = "partial_tp1_and_tp2"
                        close_offset = tp_t + t2_t
                    else:
                        runner_gross = ROUND_TRIP_COST # Stopped exactly at breakeven
                        reason = "partial_tp1_stopped_be"
                        close_offset = tp_t + be_t
                        
                    # Blended gross return: 50% at TP1, 50% at Runner Exit
                    gross = (0.5 * c["tp_pct"]) + (0.5 * runner_gross)
                    if c["direction"] == "long":
                        exit_price = c["entry"] * (1.0 + gross)
                    else:
                        exit_price = c["entry"] * (1.0 - gross)
                else:
                    exit_price = c["tp_price"]; reason = "take_profit"; gross = c["tp_pct"]
                    close_offset = tp_t
            elif sl_t < tp_t:
                exit_price = c["sl_price"]; reason = "stop_loss"; gross = -c["sl_pct"]
                close_offset = sl_t
            elif tp_t == sl_t and tp_t != INF:
                exit_price = c["sl_price"]; reason = "ambiguous_assume_sl"; gross = -c["sl_pct"]
                close_offset = sl_t
            else:
                exit_price = float(arr["cl"][end_idx]); reason = "time_exit"
                if c["direction"] == "long":
                    gross = (exit_price - c["entry"]) / c["entry"]
                else:
                    gross = (c["entry"] - exit_price) / c["entry"]
                close_offset = min(c["horizon"], len(fwd_hi)-1)

            actual_fwd_hi = fwd_hi[:close_offset+1]
            actual_fwd_lo = fwd_lo[:close_offset+1]
            if c["direction"] == "long":
                mfe_pct = float(np.max(actual_fwd_hi) - c["entry"]) / c["entry"] if len(actual_fwd_hi) else 0.0
                mae_pct = float(np.min(actual_fwd_lo) - c["entry"]) / c["entry"] if len(actual_fwd_lo) else 0.0
            else:
                mfe_pct = float(c["entry"] - np.min(actual_fwd_lo)) / c["entry"] if len(actual_fwd_lo) else 0.0
                mae_pct = float(c["entry"] - np.max(actual_fwd_hi)) / c["entry"] if len(actual_fwd_hi) else 0.0
            hold_mins = int(close_offset + 1)

            net = gross - ROUND_TRIP_COST
            pnl = position * net
            capital += pnl
            fees_paid += position * ROUND_TRIP_COST
            peak = max(peak, capital)
            dd = (peak - capital) / peak * 100
            max_dd = max(max_dd, dd)

            close_idx = min(idx + 1 + close_offset, len(arr["ts"]) - 1)
            close_ts = int(arr["ts"][close_idx])
            open_until_ts[c["sym"]] = close_ts

            d_open  = datetime.datetime.utcfromtimestamp(arr["ts"][idx]/1000).strftime('%Y-%m-%d %H:%M')
            d_close = datetime.datetime.utcfromtimestamp(close_ts/1000).strftime('%Y-%m-%d %H:%M')
            dir_str = c["direction"].upper()
            print(f"    [TRADE] {c['sym']:<10} {dir_str:<5} | Margin: ${margin_required:>7,.2f} ({leverage:.1f}x Lev) -> Size: ${position:>9,.2f} | Open: {d_open} -> Close: {d_close} | "
                  f"Result: {reason:<20} | PnL: ${pnl:>+8,.2f} | Cap: ${capital:>10,.2f} | Hold: {hold_mins}m", flush=True)

            # OPT-3: increment weekly counter in O(1)
            _week_trade_count += 1

            # Runner contribution: how much of the gross came from the runner leg
            if reason == "partial_tp1_and_tp2":
                _runner_gross_pct = round((0.5 * c["tp_pct"] * 1.5) * 100, 3)  # 50% of runner at TP2
            elif reason == "partial_tp1_time_exit":
                _runner_gross_pct = round((runner_gross - c["tp_pct"]) * 0.5 * 100, 3)
            elif reason == "partial_tp1_stopped_be":
                _runner_gross_pct = 0.0  # Stopped at BE, runner added nothing net
            else:
                _runner_gross_pct = 0.0  # Non-scaled trade

            all_trades.append({
                "open_ts": int(arr["ts"][idx]), "close_ts": close_ts,
                "sym": c["sym"], "horizon": c["horizon"], "direction": c["direction"],
                "regime": c["regime"], "p_up": round(c["p_up"], 4),
                "mu_hat_pct": round(c["mu_hat"]*100, 4),
                "ev_pct": round(c["ev"]*100, 4),
                "entry": round(c["entry"], 6), "exit": round(exit_price, 6),
                "reason": reason,
                "gross_pct": round(gross*100, 4),
                "net_pct":   round(net*100, 4),
                "pnl_usd":   round(pnl, 2),
                "margin":    round(margin_required, 2),
                "leverage":  round(leverage, 2),
                "notional_size": round(position, 2),
                "tp_pct":    round(c["tp_pct"]*100, 3),
                "sl_pct":    round(c["sl_pct"]*100, 3),
                "sigma_pct": round(sigma_pct*100, 3),
                "mfe_pct":   round(mfe_pct*100, 3),
                "mae_pct":   round(mae_pct*100, 3),
                "hold_mins": hold_mins,
                "runner_gross_pct": _runner_gross_pct,
                "_week": week_num + 1,
            })

            if (day_start_cap - capital)/max(day_start_cap,1e-9) >= DAILY_LOSS_HALT_PCT:
                day_halted = True; break
            if (week_start_cap - capital)/max(week_start_cap,1e-9) >= WEEKLY_LOSS_HALT_PCT:
                week_halted = True; break

        if (ci + 1) % 500 == 0:  # Speed: print every 500 iters not 50 (I/O flush is expensive)
            print(f"    [{(ci+1)/len(check_times)*100:.0f}%] "
                  f"{len(all_trades)} trades, capital ${capital:,.2f}", flush=True)
            # Log filter stats periodically
            total_checks = sum(filter_stats.values())
            print(f"      Filter stats (total {total_checks}): "
                  f"pos_open={filter_stats['position_open']} "
                  f"no_data={filter_stats['insufficient_data']} "
                  f"bad_feat={filter_stats['non_finite_features']} "
                  f"bad_atr={filter_stats['invalid_atr']} "
                  f"halt_reg={filter_stats['halt_regime']} "
                  f"tp_small={filter_stats['tp_too_small']} "
                  f"p_low={filter_stats['p_below_threshold']} "
                  f"ev_low={filter_stats['ev_below_threshold']} "
                  f"passed={filter_stats['candidates_passed']}", flush=True)

    week_num += 1
    wk_pnl = capital - week_start_cap
    n_wk = sum(1 for t in all_trades if t["_week"] == week_num)
    weekly.append({"week": week_num, "capital": round(capital,2),
                   "pnl": round(wk_pnl,2), "trades": n_wk})

    total_pnl = capital - STARTING_CAPITAL
    total_ret = total_pnl / STARTING_CAPITAL * 100
    winners = [t for t in all_trades if t["pnl_usd"] > 0]
    losers  = [t for t in all_trades if t["pnl_usd"] <= 0]

    print("\n" + "=" * 78)
    print("  FINAL REPORT (V5)")
    print("=" * 78)
    print(f"  Starting capital:    ${STARTING_CAPITAL:>10,.2f}")
    print(f"  Final capital:       ${capital:>10,.2f}")
    print(f"  Total P&L:           ${total_pnl:>+10,.2f}  ({total_ret:+.2f}%)")
    print(f"  Max drawdown:        {max_dd:>10.2f}%")
    print(f"  Total fees paid:     ${fees_paid:>10,.2f}")
    print(f"  Total trades:        {len(all_trades)}")
    if all_trades:
        wr = len(winners)/len(all_trades)*100
        print(f"  Win rate:            {wr:>10.1f}%")
        if winners:
            print(f"  Avg win:             ${np.mean([t['pnl_usd'] for t in winners]):>+10,.2f}")
        if losers:
            print(f"  Avg loss:            ${np.mean([t['pnl_usd'] for t in losers]):>+10,.2f}")
        if winners and losers:
            gw = sum(t["pnl_usd"] for t in winners)
            gl = abs(sum(t["pnl_usd"] for t in losers))
            print(f"  Profit factor:       {gw/max(gl,0.01):>10.2f}")
        print(f"  Avg predicted EV:    {np.mean([t['ev_pct'] for t in all_trades]):>+10.3f}% / trade")
        print(f"  Avg realised net:    {np.mean([t['net_pct'] for t in all_trades]):>+10.3f}% / trade")

    print("\n  HORIZON BREAKDOWN:")
    if all_trades:
        for h in HORIZONS:
            h_trades = [t for t in all_trades if t["horizon"] == h]
            if not h_trades: continue
            wr = sum(1 for t in h_trades if t["pnl_usd"] > 0) / len(h_trades)
            pnl = sum(t["pnl_usd"] for t in h_trades)
            avg_hold = sum(t["hold_mins"] for t in h_trades) / len(h_trades)
            avg_mfe = sum(t["mfe_pct"] for t in h_trades) / len(h_trades)
            avg_mae = sum(t["mae_pct"] for t in h_trades) / len(h_trades)
            print(f"    h={h:<4} {len(h_trades):>3} trades  WR {wr*100:>5.1f}%  PnL ${pnl:>+8,.2f}  "
                  f"AvgHold: {int(avg_hold):>3}m  AvgMFE: {avg_mfe:>+5.2f}%  AvgMAE: {avg_mae:>+5.2f}%")

    print("\n  SCALING ARCHITECTURE AUDIT:")
    SCALED_HORIZONS = [h for h in HORIZONS if h >= 240]
    if all_trades and SCALED_HORIZONS:
        print(f"  {'Horizon':<8} {'Eligible':>8} {'TP1 Hit':>8} {'TP2 Hit':>8} {'BE Stop':>8} {'TimExit':>8} {'Runner $':>10} {'Runner%PnL':>11}")
        print(f"  {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*10} {'-'*11}")
        for h in SCALED_HORIZONS:
            h_trades = [t for t in all_trades if t["horizon"] == h]
            if not h_trades: continue
            n = len(h_trades)
            # TP1 hit = any partial or full TP exit (trade reached TP1 price)
            tp1_hit = sum(1 for t in h_trades if t["reason"] in
                          ("partial_tp1_and_tp2", "partial_tp1_stopped_be", "partial_tp1_time_exit"))
            tp2_hit = sum(1 for t in h_trades if t["reason"] == "partial_tp1_and_tp2")
            be_stop = sum(1 for t in h_trades if t["reason"] == "partial_tp1_stopped_be")
            te_exit = sum(1 for t in h_trades if t["reason"] == "partial_tp1_time_exit")
            runner_pnl = sum(t["notional_size"] * t["runner_gross_pct"] / 100 for t in h_trades)
            total_pnl_h = sum(t["pnl_usd"] for t in h_trades)
            runner_pct = runner_pnl / max(abs(total_pnl_h), 0.01) * 100
            print(f"  h={h:<6} {n:>8} {tp1_hit:>7} ({tp1_hit/n*100:>4.0f}%)"
                  f" {tp2_hit:>7} ({tp2_hit/n*100:>4.0f}%)"
                  f" {be_stop:>7} ({be_stop/n*100:>4.0f}%)"
                  f" {te_exit:>7} ({te_exit/n*100:>4.0f}%)"
                  f" ${runner_pnl:>+9,.2f}  {runner_pct:>+9.1f}%")
        # Summary verdict
        all_scaled = [t for t in all_trades if t["horizon"] in SCALED_HORIZONS]
        if all_scaled:
            total_runner = sum(t["notional_size"] * t["runner_gross_pct"] / 100 for t in all_scaled)
            total_scaled_pnl = sum(t["pnl_usd"] for t in all_scaled)
            verdict = "ADDING VALUE" if total_runner > 0 else "NET DRAG"
            print(f"\n  Runner total: ${total_runner:>+9,.2f}  ({total_runner/max(abs(total_scaled_pnl),0.01)*100:>+.1f}% of scaled PnL)  [{verdict}]")
    else:
        print("  No scaled horizons active.")

    print("\n  REGIME ENCOUNTERS:")
    total_dp = sum(regime_counts.values()) or 1
    for r, c in sorted(regime_counts.items(), key=lambda x: -x[1]):
        print(f"    {r:<14} {c:>6}  ({c/total_dp*100:>5.1f}%)")

    print("\n  PER-COIN BREAKDOWN:")
    for sym in COINS:
        ct = [t for t in all_trades if t.get("sym", t.get("coin")) == sym]
        if not ct: continue
        cp = sum(t["pnl_usd"] for t in ct)
        cw = sum(1 for t in ct if t["pnl_usd"] > 0) / max(len(ct),1) * 100
        print(f"    {sym:<12} {len(ct):>4} trades  WR {cw:>5.1f}%  PnL ${cp:>+9,.2f}")

    print("\n  BY EXIT REASON:")
    for reason in ["take_profit", "stop_loss", "time_exit", "ambiguous_assume_sl"]:
        rt = [t for t in all_trades if t["reason"] == reason]
        if not rt: continue
        rp = sum(t["pnl_usd"] for t in rt)
        rw = sum(1 for t in rt if t["pnl_usd"] > 0)/len(rt)*100
        print(f"    {reason:<22} {len(rt):>4}  WR {rw:>5.1f}%  PnL ${rp:>+9,.2f}")

    print("\n  WEEKLY EQUITY CURVE:")
    bar_max = max((abs(w["pnl"]) for w in weekly), default=1)
    for wc in weekly:
        blen = max(1, int(abs(wc["pnl"]) / max(bar_max, 1) * 20))
        bar = ("+" * blen if wc["pnl"] >= 0 else "-" * blen) if wc["pnl"] != 0 else "."
        print(f"    week {wc['week']:>2}: {wc['trades']:>3} trades  "
              f"${wc['pnl']:>+8,.2f}  ${wc['capital']:>10,.2f}  {bar}")

    print("\n  FILTER STATS (why candidates rejected):")
    total_checks = sum(filter_stats.values())
    for key, val in filter_stats.items():
        print(f"    {key:<20} {val:>8}  ({val/max(total_checks,1)*100:>5.1f}%)")

    out_path = os.path.join(DATA_DIR, "simulation_v5_results_oos.json")
    with open(out_path, "w") as f:
        json.dump({
            "starting_capital": STARTING_CAPITAL,
            "final_capital": round(capital, 2),
            "total_pnl": round(total_pnl, 2),
            "total_return_pct": round(total_ret, 2),
            "max_dd_pct": round(max_dd, 2),
            "total_trades": len(all_trades),
            "fees_paid": round(fees_paid, 2),
            "win_rate": round(len(winners)/max(len(all_trades),1)*100, 1),
            "regime_counts": regime_counts,
            "weekly": weekly, "trades": all_trades,
        }, f, indent=2, default=str)
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
