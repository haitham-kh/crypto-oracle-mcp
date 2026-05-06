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
)
from features_v5 import (
    FEATURE_NAMES_V5, build_v5_full,
)
from regime_filter import classify_regime
import polars as pl

BINANCE_BASE = "https://api.binance.com"
DATA_DIR = os.path.join(os.path.dirname(__file__), "challenge_data")
PERP_DIR = os.path.join(os.path.dirname(__file__), "data", "perp")
COINS = ["DOTUSDT", "APTUSDT", "NEARUSDT", "SUIUSDT", "ATOMUSDT"]
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
    while current < end_ms:
        try:
            r = requests.get(f"{BINANCE_BASE}/api/v3/klines",
                             params={"symbol": symbol, "interval": "1m",
                                     "startTime": current, "endTime": end_ms,
                                     "limit": 1500}, timeout=15)
            r.raise_for_status(); data = r.json()
        except Exception as e:
            print(f"    retry: {e}"); time.sleep(2); continue
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
    p = os.path.join(DATA_DIR, f"{symbol}_challenge.json")
    if os.path.exists(p):
        with open(p) as f: data = json.load(f)
        print(f"  {symbol}: cached ({len(data):,} candles)")
        return data
    data = download_klines(symbol, start_ms, end_ms)
    with open(p, "w") as f: json.dump(data, f)
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

def build_sim_basket_state(coin_data):
    """Lightweight version of basket_alignment.py: build cross-sectional ranks
    and basket aggregates from the 5 sim coins on a unified 1m grid.
    """
    print("\nBuilding sim-time basket state (5-coin basket)...")
    syms = list(coin_data.keys())
    # build wide table of closes
    parts = []
    for s in syms:
        ts = np.array([c["timestamp"] for c in coin_data[s]], dtype=np.int64)
        cl = np.array([c["close"]     for c in coin_data[s]], dtype=np.float64)
        parts.append(pl.DataFrame({"timestamp": ts, s: cl}))
    wide = parts[0]
    for d in parts[1:]:
        wide = wide.join(d, on="timestamp", how="full", coalesce=True)
    wide = wide.sort("timestamp")
    n = wide.height

    # per-coin returns + RV
    log_cl = {}; r1 = {}; r4 = {}; rv = {}
    for s in syms:
        cl = wide[s].to_numpy().astype(np.float64)
        # forward-fill within coin
        last = np.nan
        for i in range(n):
            if not np.isnan(cl[i]): last = cl[i]
            elif not np.isnan(last): cl[i] = last
        lc = np.log(np.maximum(cl, 1e-12))
        log_cl[s] = lc
        a = np.zeros(n); a[60:]  = lc[60:]  - lc[:-60];  r1[s] = a
        b = np.zeros(n); b[240:] = lc[240:] - lc[:-240]; r4[s] = b
        m1 = np.zeros(n); m1[1:] = lc[1:] - lc[:-1]
        v = np.zeros(n)
        if n > 60:
            from numpy.lib.stride_tricks import sliding_window_view
            win = sliding_window_view(m1, 60)
            v[59:59+len(win)] = win.std(axis=1) * np.sqrt(60)
        rv[s] = v

    R1 = np.column_stack([r1[s] for s in syms])
    R4 = np.column_stack([r4[s] for s in syms])
    V1 = np.column_stack([rv[s] for s in syms])
    n_coins = (~np.isnan(R1)).sum(axis=1)
    with np.errstate(invalid="ignore", all="ignore"):
        med_r4 = np.nan_to_num(np.nanmedian(R4, axis=1), nan=0.0)
    # ranks
    def rrow(row):
        m = ~np.isnan(row); k = m.sum()
        if k <= 1: return np.full_like(row, np.nan)
        order = np.argsort(np.argsort(np.where(m, row, np.inf)))
        out = np.full_like(row, np.nan)
        out[m] = order[m] / max(k-1, 1)
        return out
    rk_r1 = np.full_like(R1, np.nan)
    rk_r4 = np.full_like(R4, np.nan)
    rk_v1 = np.full_like(V1, np.nan)
    for i in range(n):
        if n_coins[i] >= 2:
            rk_r1[i] = rrow(R1[i]); rk_r4[i] = rrow(R4[i]); rk_v1[i] = rrow(V1[i])

    ts = wide["timestamp"].to_numpy().astype(np.int64)
    rows = []
    for ci, s in enumerate(syms):
        df = pl.DataFrame({
            "timestamp": ts,
            "symbol": [s] * n,
            "ret_rank_1h":   rk_r1[:, ci],
            "ret_rank_4h":   rk_r4[:, ci],
            "rv_rank_1h":    rk_v1[:, ci],
            "coin_alpha_4h": R4[:, ci] - med_r4,
            "basket_mom_4h": med_r4,
            "n_coins":       n_coins.astype(np.float64),
        }).filter(pl.col("ret_rank_4h").is_not_null())
        rows.append(df)
    out = pl.concat(rows).sort(["symbol", "timestamp"])
    print(f"  basket state: {out.height:,} rows over {n} unique minutes")
    return out


# ── Per-coin feature precompute ─────────────────────────────────────────────

def _coin_onehot_block(symbol, n):
    block = np.zeros((n, len(COIN_ONEHOT_NAMES)), dtype=np.float64)
    if symbol in COIN_ONEHOT_NAMES:
        block[:, COIN_ONEHOT_NAMES.index(symbol)] = 1.0
    return block


def precompute_features(coin_data, btc_data, basket_state):
    print("\nPre-computing v5 features...")
    from train_ev_model_v2 import build_features_v2

    btc_ts = np.array([c["timestamp"] for c in btc_data], dtype=np.int64)
    btc_cl = np.array([c["close"] for c in btc_data], dtype=np.float64)
    out = {}
    for sym, candles in coin_data.items():
        t0 = time.time()
        ts = np.array([c["timestamp"] for c in candles], dtype=np.float64)
        hi = np.array([c["high"] for c in candles], dtype=np.float64)
        lo = np.array([c["low"] for c in candles], dtype=np.float64)
        cl = np.array([c["close"] for c in candles], dtype=np.float64)
        vo = np.array([c["volume"] for c in candles], dtype=np.float64)
        tbv = np.array([c.get("taker_buy_volume", 0) for c in candles], dtype=np.float64)
        ofi = 2 * tbv - vo
        btc_aligned = np.interp(ts, btc_ts.astype(np.float64), btc_cl)
        F_v2, atr = build_features_v2(cl, hi, lo, vo, ofi, ts, btc_aligned)
        F_feats = build_v5_full(sym, F_v2, cl, hi, lo, vo, ofi, tbv, ts, atr,
                                btc_aligned, PERP_DIR, basket_state)
        F_full = np.hstack([F_feats, _coin_onehot_block(sym, len(cl))])
        hurst = F_v2[:, 10].copy()
        out[sym] = {
            "ts":  np.array([c["timestamp"] for c in candles], dtype=np.int64),
            "hi": hi, "lo": lo, "cl": cl,
            "F":  F_full, "atr": atr, "hurst": hurst,
        }
        print(f"  {sym}: {len(cl):,} bars × {F_full.shape[1]} feats  ({time.time()-t0:.1f}s)")
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
        r = meta["horizon_results"].get(str(h))
        if r is None:
            continue
        clf = xgb.Booster(); clf.load_model(os.path.join(here, r["clf_path"]))
        reg = xgb.Booster(); reg.load_model(os.path.join(here, r["reg_path"]))
        with open(os.path.join(here, r["calib_path"]), "rb") as f:
            calib = pickle.load(f)
        models[h] = {"clf": clf, "reg": reg, "calib": calib,
                     "p_threshold": float(r["p_threshold"])}
        print(f"  h={h:>4}m  thr={r['p_threshold']:.2f}  "
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
    print("=" * 78)
    print("  ORACLE V5 SIMULATION — perp + cross-sectional + regime-gated")
    print(f"  Capital ${STARTING_CAPITAL:,.0f}  coins {', '.join(COINS)}")
    print(f"  Decision every {CHECK_EVERY_MINS//60}h  horizons {HORIZONS}m  "
          f"cost {ROUND_TRIP_COST*100:.2f}%")
    print("=" * 78)

    months = get_month_boundaries(6)
    overall_start = int(months[0][0].timestamp() * 1000)
    overall_end   = int(months[-1][1].timestamp() * 1000)

    print("\nLoading spot data...")
    coin_data = {sym: load_or_download(sym, overall_start, overall_end)
                 for sym in list(COINS)}
    btc_data = load_or_download("BTCUSDT", overall_start, overall_end)

    print("\nEnsuring perp data for sim coins...")
    ensure_perp_data(COINS)

    study_end_ms = int(months[3][0].timestamp() * 1000)
    trade_end_ms = int(months[5][1].timestamp() * 1000)
    print(f"\n  Study:  {months[0][0].strftime('%b %Y')} - {months[2][1].strftime('%b %Y')}")
    print(f"  Trade:  {months[3][0].strftime('%b %Y')} - {months[5][1].strftime('%b %Y')}")

    basket_state = build_sim_basket_state(coin_data)
    feats = precompute_features(coin_data, btc_data, basket_state)

    print("\nLoading v5 models...")
    meta, models = load_v5_models()
    if not models:
        print("ERROR: no models loaded. Run train_v5.py first."); return

    import xgboost as xgb
    feat_names = meta["feature_names"]

    check_ms = CHECK_EVERY_MINS * 60 * 1000
    check_times = list(range(study_end_ms, trade_end_ms, check_ms))
    print(f"\n  Decision points: {len(check_times)} × {len(COINS)} coins")

    capital = STARTING_CAPITAL
    peak = capital; max_dd = 0.0; fees_paid = 0.0
    all_trades = []
    open_until_ts = {sym: 0 for sym in COINS}

    DAY_MS = 24*60*60*1000; WEEK_MS = 7*DAY_MS
    day_start_ms = study_end_ms; day_start_cap = capital
    week_start_ms = study_end_ms; week_start_cap = capital
    week_num = 0; weekly = []
    day_halted = False; week_halted = False
    regime_counts = {}

    print("\n" + "=" * 78)
    print("  RUNNING")
    print("=" * 78)

    for ci, check_ts in enumerate(check_times):
        while check_ts >= day_start_ms + DAY_MS:
            day_start_ms += DAY_MS; day_start_cap = capital; day_halted = False
        while check_ts >= week_start_ms + WEEK_MS:
            week_num += 1
            wk_pnl = capital - week_start_cap
            n_wk = sum(1 for t in all_trades if t["_week"] == week_num)
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
        for sym in COINS:
            if open_until_ts.get(sym, 0) > check_ts:
                continue
            d = feats[sym]
            idx = int(np.searchsorted(d["ts"], check_ts, side="right") - 1)
            if idx < 1500 or idx + max(HORIZONS) >= len(d["ts"]):
                continue
            f_row = d["F"][idx:idx+1]
            if not np.all(np.isfinite(f_row)):
                continue
            atr_val = d["atr"][idx]
            if not np.isfinite(atr_val) or atr_val <= 0:
                continue
            entry = float(d["cl"][idx])
            hurst = float(d["hurst"][idx]) if np.isfinite(d["hurst"][idx]) else 0.5

            regime = classify_regime(
                hurst, float(f_row[0, IDX_TE]),
                float(f_row[0, IDX_RV1H]), float(f_row[0, IDX_RV_Z_1D]),
                float(f_row[0, IDX_VOL_BURST]), float(f_row[0, IDX_RANGE_POS]),
            )
            regime_counts[regime] = regime_counts.get(regime, 0) + 1
            if regime in HALT_REGIMES:
                continue

            dmat = xgb.DMatrix(f_row, feature_names=feat_names)
            best = None
            for h, m in models.items():
                scale = horizon_atr_scale(h)
                tp_mult = regime_tp_mult(hurst) * scale
                sl_mult = SL_MULT * scale
                tp_pct = tp_mult * atr_val / entry
                sl_pct = sl_mult * atr_val / entry
                if tp_pct < MIN_TP_TO_COST_RATIO * ROUND_TRIP_COST:
                    continue
                p_raw = float(m["clf"].predict(dmat)[0])
                p_cal = float(m["calib"].predict([p_raw])[0])
                if p_cal < m["p_threshold"]:
                    continue
                mu_hat = float(m["reg"].predict(dmat)[0])
                ev = p_cal * tp_pct - (1 - p_cal) * sl_pct - ROUND_TRIP_COST
                if ev < MIN_EV_PCT:
                    continue
                cand = {
                    "sym": sym, "idx": idx, "entry": entry,
                    "horizon": h, "p_up": p_cal, "mu_hat": mu_hat,
                    "tp_pct": tp_pct, "sl_pct": sl_pct,
                    "tp_price": entry + tp_mult * atr_val,
                    "sl_price": entry - sl_mult * atr_val,
                    "ev": ev, "hurst": hurst, "regime": regime,
                }
                if best is None or ev > best["ev"]:
                    best = cand
            if best is not None:
                candidates.append(best)

        candidates.sort(key=lambda c: -c["ev"])
        open_now = sum(1 for s, t in open_until_ts.items() if t > check_ts)
        budget = max(0, MAX_OPEN_POSITIONS_TOTAL - open_now)
        for c in candidates[:budget]:
            arr = feats[c["sym"]]; idx = c["idx"]
            window = max(min(c["horizon"], idx), 5)
            log_w = np.diff(np.log(np.maximum(arr["cl"][idx-window:idx+1], 1e-12)))
            sigma_pct = float(np.std(log_w) * np.sqrt(c["horizon"])) if len(log_w) > 1 else c["sl_pct"]
            sigma_pct = max(sigma_pct, 1e-4)
            mu_pct = max(c["mu_hat"], 0.0)
            position = mean_variance_size(
                capital, mu_pct, sigma_pct, c["p_up"], c["tp_pct"], c["sl_pct"]
            )
            if position <= 0:
                continue

            end_idx = min(idx + c["horizon"], len(arr["cl"]) - 1)
            fwd_hi = arr["hi"][idx+1:end_idx+1]
            fwd_lo = arr["lo"][idx+1:end_idx+1]
            tp_hits = np.where(fwd_hi >= c["tp_price"])[0]
            sl_hits = np.where(fwd_lo <= c["sl_price"])[0]
            INF = np.iinfo(np.int32).max
            tp_t = tp_hits[0] if len(tp_hits) else INF
            sl_t = sl_hits[0] if len(sl_hits) else INF
            if tp_t < sl_t:
                exit_price = c["tp_price"]; reason = "take_profit"; gross = c["tp_pct"]
            elif sl_t < tp_t:
                exit_price = c["sl_price"]; reason = "stop_loss"; gross = -c["sl_pct"]
            elif tp_t == sl_t and tp_t != INF:
                exit_price = c["sl_price"]; reason = "ambiguous_assume_sl"; gross = -c["sl_pct"]
            else:
                exit_price = float(arr["cl"][end_idx]); reason = "time_exit"
                gross = (exit_price - c["entry"]) / c["entry"]

            net = gross - ROUND_TRIP_COST
            pnl = position * net
            capital += pnl
            fees_paid += position * ROUND_TRIP_COST
            peak = max(peak, capital)
            dd = (peak - capital) / peak * 100
            max_dd = max(max_dd, dd)

            close_ts = int(arr["ts"][end_idx])
            open_until_ts[c["sym"]] = close_ts

            all_trades.append({
                "open_ts": int(arr["ts"][idx]), "close_ts": close_ts,
                "coin": c["sym"], "horizon": int(c["horizon"]),
                "regime": c["regime"], "p_up": round(c["p_up"], 4),
                "mu_hat_pct": round(c["mu_hat"]*100, 4),
                "ev_pct": round(c["ev"]*100, 4),
                "entry": round(c["entry"], 6), "exit": round(exit_price, 6),
                "reason": reason,
                "gross_pct": round(gross*100, 4),
                "net_pct":   round(net*100, 4),
                "pnl_usd":   round(pnl, 2),
                "size":      round(position, 2),
                "tp_pct":    round(c["tp_pct"]*100, 3),
                "sl_pct":    round(c["sl_pct"]*100, 3),
                "sigma_pct": round(sigma_pct*100, 3),
                "_week": week_num + 1,
            })

            if (day_start_cap - capital)/max(day_start_cap,1e-9) >= DAILY_LOSS_HALT_PCT:
                day_halted = True; break
            if (week_start_cap - capital)/max(week_start_cap,1e-9) >= WEEKLY_LOSS_HALT_PCT:
                week_halted = True; break

        if (ci + 1) % 50 == 0:
            print(f"    [{(ci+1)/len(check_times)*100:.0f}%] "
                  f"{len(all_trades)} trades, capital ${capital:,.2f}", flush=True)

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
    for h in HORIZONS:
        ts = [t for t in all_trades if t["horizon"] == h]
        if not ts: continue
        pnl_h = sum(t["pnl_usd"] for t in ts)
        wr_h = sum(1 for t in ts if t["pnl_usd"] > 0) / len(ts) * 100
        print(f"    h={h:<5} {len(ts):>4} trades  WR {wr_h:>5.1f}%  PnL ${pnl_h:>+9,.2f}")

    print("\n  REGIME ENCOUNTERS:")
    total_dp = sum(regime_counts.values()) or 1
    for r, c in sorted(regime_counts.items(), key=lambda x: -x[1]):
        print(f"    {r:<14} {c:>6}  ({c/total_dp*100:>5.1f}%)")

    print("\n  PER-COIN BREAKDOWN:")
    for sym in COINS:
        ct = [t for t in all_trades if t["coin"] == sym]
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

    out_path = os.path.join(DATA_DIR, "simulation_v5_results.json")
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
