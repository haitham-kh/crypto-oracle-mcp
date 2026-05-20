import os
import sys
import time
import math
import hmac
import hashlib
import json
import datetime
import requests
import urllib.parse
import numpy as np
import polars as pl
import xgboost as xgb
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(__file__))

from trading_config import (
    CHECK_EVERY_MINS, HORIZONS, ROUND_TRIP_COST,
    horizon_atr_scale, regime_tp_mult, SL_MULT,
    MIN_TP_TO_COST_RATIO, MIN_EV_PCT,
    RISK_PER_TRADE_PCT, MAX_POSITION_PCT, KELLY_FRACTION,
    RISK_AVERSION_TARGET, MAX_OPEN_POSITIONS_TOTAL,
    DAILY_LOSS_HALT_PCT, WEEKLY_LOSS_HALT_PCT,
    HALT_REGIMES, MAX_SL_PCT, ASSET_MAX_POSITION_SCALE,
    HORIZON_REGIME_EV_SCALE, BTC_QUIET_4H_THRESHOLD, BTC_QUIET_EV_MULTIPLIER
)
from features_v5 import FEATURE_NAMES_V5, build_v5_full
from features_v7 import FEATURE_NAMES_V7_EXTRA, build_v7_features
from regime_filter import classify_regime
from test_oos_btc_v5 import build_sim_basket_state, _as_compact_arrays, load_v5_models
from train_ev_model_v2 import build_features_v2

# API Credentials for Binance Futures Testnet/Demo
API_KEY = "DcOsggYt78jpdIeqxTxOjlBQyUrSy9dGGwexTNZQ7eRpAkTQOADDFIAoSOYaNH7H"
SECRET_KEY = "bOvAgsH86HOwgFNT4NK1G6gXnbeCqcfGVzol6VV4bYsFN3I48iEIH2lPZA6pkxz1"
DEMO_BASE = "https://demo-fapi.binance.com"
PROD_BASE = "https://fapi.binance.com"

# File to persist active trades
TRADES_FILE = os.path.join(os.path.dirname(__file__), "active_trades.json")

# Global cache to feed the monkey-patched _load_perp
LIVE_PERP_CACHE = {}

# Monkey patch features_v5._load_perp to load from our live cache
import features_v5
features_v5._load_perp = lambda sym, pdir: LIVE_PERP_CACHE.get(sym, (None, None))

# Re-calibrated original model parameters
MIN_EV_PCT_THRESHOLD = 0.0015      # Original V5 threshold: >= 0.15% expected net return
MIN_P_UP_THRESHOLD = 0.59          # Original V5 Long threshold: 0.59
MIN_P_DOWN_THRESHOLD = 0.61        # Original V5 Short threshold: 0.61
MAX_OPEN_POSITIONS = 100           # Allow up to 100 concurrent positions
ALLOCATED_MARGIN_PER_TRADE = 300.0  # $300 USDT margin per position

# ── Trading quality gates ────────────────────────────────────────────────────
# LOOSEN_MODE MUST stay False — loosened thresholds allow negative-EV trades.
# V5 test EV is already negative; trading below 0.59 threshold guarantees losses.
LOOSEN_MODE = False
ALLOW_HALT_REGIMES = False           # Do not trade in CHAOS / LOW_LIQUIDITY regimes

# V7 blending weights (user-specified: V7 is 70% of the decision)
V7_WEIGHT = 0.70
V5_WEIGHT = 0.30
# V7 micro-gate: require V7 micro-model agreement (probability > this)
V7_MICRO_GATE_THRESHOLD = 0.50
# Set to False before V7 models are trained; True after colab_v7_train.py has run.
V7_MODELS_AVAILABLE = False

COIN_ONEHOT_NAMES = [
    "AAVEUSDT", "ADAUSDT", "AVAXUSDT", "BNBUSDT", "BTCUSDT",
    "DOGEUSDT", "ETHUSDT", "LINKUSDT", "PEPEUSDT", "SHIBUSDT",
    "SOLUSDT", "UNIUSDT", "WIFUSDT", "XRPUSDT",
]
ALL_FEATURE_NAMES    = FEATURE_NAMES_V5 + [f"coin_is_{s}" for s in COIN_ONEHOT_NAMES]  # 94 features
ALL_FEATURE_NAMES_V7 = ALL_FEATURE_NAMES + FEATURE_NAMES_V7_EXTRA                     # 130+ features

# Global metrics for Dashboard
DASHBOARD_DATA = {
    "balance": 0.0,
    "available": 0.0,
    "offset": 0,
    "last_scan_time": "Never",
    "scan_results": [],
    "cycle": 0
}

# ─────────────────────────────────────────────────────────────────────────────
# V7 Model Loader
# ─────────────────────────────────────────────────────────────────────────────

def load_v7_models():
    """Load V7_full classifier + calibrator models for all horizons/directions.

    Returns a nested dict:
        models_v7[h][direction] = {"clf": XGBModel, "calib": IsotonicRegression,
                                   "p_threshold": float}
    Also returns v7_micro_models[h][direction] = {"clf", "calib", "p_threshold"}

    If any model file is missing, returns None (caller should set
    V7_MODELS_AVAILABLE = False and fall back to V5-only).
    """
    import pickle
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    meta_path = os.path.join(data_dir, "v7_meta.json")
    if not os.path.exists(meta_path):
        print("[V7] data/v7_meta.json not found — V7 disabled. Run colab_v7_train.py first.")
        return None, None

    with open(meta_path) as f:
        meta = json.load(f)

    horizons = meta.get("horizons", [60, 720])
    full_models  = {}
    micro_models = {}

    for h in horizons:
        full_models[h]  = {}
        micro_models[h] = {}
        for direction in ["long", "short"]:
            for tag, store in [("full", full_models), ("micro", micro_models)]:
                clf_path   = os.path.join(os.path.dirname(__file__),
                                          f"data/v7_{tag}_clf_{direction}_h{h}.json")
                calib_path = os.path.join(os.path.dirname(__file__),
                                          f"data/v7_{tag}_calib_{direction}_h{h}.pkl")
                if not os.path.exists(clf_path) or not os.path.exists(calib_path):
                    print(f"[V7] Missing: {clf_path} or {calib_path}")
                    return None, None

                clf = xgb.Booster()
                clf.load_model(clf_path)

                with open(calib_path, "rb") as fp:
                    calib = pickle.load(fp)

                # p_threshold from meta
                try:
                    p_thr = meta["horizon_results"][str(h)][direction][tag]["p_threshold"]
                except (KeyError, TypeError):
                    p_thr = 0.56  # safe default

                store[h][direction] = {"clf": clf, "calib": calib, "p_threshold": float(p_thr)}

    print("[V7] Models loaded successfully (full + micro).")
    return full_models, micro_models

# ── Binance Futures API Client ──────────────────────────────────────────────────

class BinanceFuturesClient:
    def __init__(self, api_key, api_secret, base_url):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({"X-MBX-APIKEY": self.api_key})
        self.time_offset = 0
        self.sync_time()

    def sync_time(self):
        try:
            r = requests.get(f"{self.base_url}/fapi/v1/time", timeout=10)
            r.raise_for_status()
            server_time = r.json()["serverTime"]
            local_time = int(time.time() * 1000)
            self.time_offset = server_time - local_time
            DASHBOARD_DATA["offset"] = self.time_offset
        except Exception as e:
            self.time_offset = 0

    def _sign(self, params):
        params["timestamp"] = int(time.time() * 1000) + self.time_offset
        query_string = urllib.parse.urlencode(params)
        signature = hmac.new(
            self.api_secret.encode("utf-8"),
            query_string.encode("utf-8"),
            hashlib.sha256
        ).hexdigest()
        params["signature"] = signature
        return params

    def request(self, method, path, params=None, signed=False):
        url = f"{self.base_url}{path}"
        if params is None:
            params = {}
        if signed:
            params = self._sign(params.copy())
            
        try:
            if method.upper() == "GET":
                r = self.session.get(url, params=params, timeout=15)
            elif method.upper() == "POST":
                r = self.session.post(url, data=params, timeout=15)
            elif method.upper() == "DELETE":
                r = self.session.delete(url, params=params, timeout=15)
            else:
                raise ValueError(f"Unsupported method: {method}")
                
            r.raise_for_status()
            return r.json()
        except requests.exceptions.HTTPError as he:
            print(f"[Client HTTPError] {method} {path} response body: {he.response.text}", flush=True)
            raise he
        except Exception as e:
            raise e

# ── Local Trades Persistence ──────────────────────────────────────────────────

def load_active_trades():
    if not os.path.exists(TRADES_FILE):
        return {}
    try:
        with open(TRADES_FILE, "r") as f:
            return json.load(f)
    except Exception as e:
        return {}

def save_active_trades(trades):
    try:
        with open(TRADES_FILE, "w") as f:
            json.dump(trades, f, indent=2)
    except Exception as e:
        pass

# ── Data Fetching Helpers ────────────────────────────────────────────────────────

def is_crypto_symbol(symbol):
    if symbol.startswith("XAU") or symbol.startswith("XAG") or symbol.startswith("XPT") or symbol.startswith("XPD"):
        return False
    if "INDEX" in symbol or symbol.startswith("USDC") or "USDC" in symbol:
        return False
    return True

# Predefined stable basket of high-liquidity, non-meme coins from the top 25
STABLE_BASKET_COINS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "ADAUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT", "LTCUSDT",
    "NEARUSDT", "UNIUSDT", "SUIUSDT", "AAVEUSDT", "APTUSDT",
    "FILUSDT", "ATOMUSDT", "OPUSDT", "BCHUSDT", "XLMUSDT",
    "FTMUSDT", "ICPUSDT", "ETCUSDT", "ALGOUSDT", "POLUSDT"
]

def fetch_ticker_top_30():
    """Return the predefined stable, non-meme coins basket."""
    return STABLE_BASKET_COINS

def fetch_klines(symbol, limit=1000):
    """Fetch recent klines from MEXC (bypassing Binance geo-block), with Binance fallback."""
    # Try MEXC first
    try:
        url = "https://api.mexc.com/api/v3/klines"
        params = {"symbol": symbol, "interval": "1m", "limit": limit}
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, list):
            candles = []
            for k in data:
                candles.append({
                    "timestamp": int(k[0]),
                    "open": float(k[1]), "high": float(k[2]),
                    "low": float(k[3]), "close": float(k[4]),
                    "volume": float(k[5]),
                    "taker_buy_volume": float(k[9]),
                })
            return candles
    except Exception as e:
        print(f"  [Warning] MEXC kline fetch failed for {symbol}: {e}")

    # Fallback to Binance Futures API
    url = f"{PROD_BASE}/fapi/v1/klines"
    params = {"symbol": symbol, "interval": "1m", "limit": limit}
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    candles = []
    for k in r.json():
        candles.append({
            "timestamp": int(k[0]),
            "open": float(k[1]), "high": float(k[2]),
            "low": float(k[3]), "close": float(k[4]),
            "volume": float(k[5]),
            "taker_buy_volume": float(k[9]),
        })
    return candles

def fetch_live_perp_dfs(symbol):
    """Fetch perp metrics from production with graceful fallback on fail."""
    funding_df = None
    metrics_df = None

    # 1. Funding rates
    try:
        funding_url = f"{PROD_BASE}/fapi/v1/fundingRate"
        r = requests.get(funding_url, params={"symbol": symbol, "limit": 100}, timeout=8)
        if r.status_code == 200:
            funding_data = r.json()
            funding_list = [{"ts_ms": int(item["fundingTime"]), "funding_rate": float(item["fundingRate"])} for item in funding_data]
            if funding_list:
                funding_df = pl.DataFrame(funding_list).sort("ts_ms")
    except Exception:
        pass

    # 2. Open interest
    oi_dict = {}
    try:
        oi_url = f"{PROD_BASE}/futures/data/openInterestHist"
        r = requests.get(oi_url, params={"symbol": symbol, "period": "5m", "limit": 500}, timeout=8)
        if r.status_code == 200:
            oi_dict = {item["timestamp"]: float(item["sumOpenInterest"]) for item in r.json()}
    except Exception:
        pass

    # 3. Top LSR
    lsr_top_dict = {}
    try:
        lsr_top_url = f"{PROD_BASE}/futures/data/topLongShortAccountRatio"
        r = requests.get(lsr_top_url, params={"symbol": symbol, "period": "5m", "limit": 500}, timeout=8)
        if r.status_code == 200:
            lsr_top_dict = {item["timestamp"]: float(item["longShortRatio"]) for item in r.json()}
    except Exception:
        pass

    # 4. Global LSR
    lsr_global_dict = {}
    try:
        lsr_global_url = f"{PROD_BASE}/futures/data/globalLongShortAccountRatio"
        r = requests.get(lsr_global_url, params={"symbol": symbol, "period": "5m", "limit": 500}, timeout=8)
        if r.status_code == 200:
            lsr_global_dict = {item["timestamp"]: float(item["longShortRatio"]) for item in r.json()}
    except Exception:
        pass

    # 5. Taker Ratio
    taker_dict = {}
    try:
        taker_url = f"{PROD_BASE}/futures/data/takerlongshortRatio"
        r = requests.get(taker_url, params={"symbol": symbol, "period": "5m", "limit": 500}, timeout=8)
        if r.status_code == 200:
            taker_dict = {item["timestamp"]: float(item["buySellRatio"]) for item in r.json()}
    except Exception:
        pass

    # Merge on timestamp
    all_timestamps = sorted(list(oi_dict.keys())) if oi_dict else []
    metrics_list = []
    for ts in all_timestamps:
        metrics_list.append({
            "ts_ms": int(ts),
            "oi": oi_dict.get(ts, 0.0),
            "lsr_top": lsr_top_dict.get(ts, 1.0),
            "lsr_global": lsr_global_dict.get(ts, 1.0),
            "taker_ratio": taker_dict.get(ts, 1.0)
        })
    if metrics_list:
        metrics_df = pl.DataFrame(metrics_list).sort("ts_ms")

    return funding_df, metrics_df

# ── Feature & Prediction Helpers ─────────────────────────────────────────────────

def predict_coin(symbol, candles, btc_candles, models, meta_feature_names, basket_state, active_min_ev,
                 v7_full_models=None, v7_micro_models=None):
    """Build features and predict direction for the last bar of the symbol."""
    n_c = len(candles)
    ts = np.fromiter((c["timestamp"] for c in candles), dtype=np.float64, count=n_c)
    hi = np.fromiter((c["high"] for c in candles), dtype=np.float32, count=n_c)
    lo = np.fromiter((c["low"] for c in candles), dtype=np.float32, count=n_c)
    cl = np.fromiter((c["close"] for c in candles), dtype=np.float32, count=n_c)
    vo = np.fromiter((c["volume"] for c in candles), dtype=np.float32, count=n_c)
    tbv = np.fromiter((c.get("taker_buy_volume", 0) for c in candles), dtype=np.float32, count=n_c)
    ts_int = np.fromiter((c["timestamp"] for c in candles), dtype=np.int64, count=n_c)

    # 1. Align BTC
    btc_ts = np.array([c["timestamp"] for c in btc_candles], dtype=np.float64)
    btc_cl = np.array([c["close"] for c in btc_candles], dtype=np.float64)
    btc_aligned = np.interp(ts, btc_ts, btc_cl)

    # 2. Build features
    ofi = (2.0 * tbv - vo).astype(np.float32)
    F_v2, atr = build_features_v2(cl, hi, lo, vo, ofi, ts, btc_aligned)
    
    # We pass None for perp_dir since features_v5._load_perp is monkey patched
    F_feats = build_v5_full(symbol, F_v2, cl, hi, lo, vo, ofi, tbv, ts_int, atr,
                            btc_aligned, None, basket_state)
    
    onehot = np.zeros((len(cl), len(COIN_ONEHOT_NAMES)), dtype=np.float32)
    if symbol in COIN_ONEHOT_NAMES:
        onehot[:, COIN_ONEHOT_NAMES.index(symbol)] = 1.0
        
    F_full = np.hstack([F_feats, onehot]).astype(np.float32)  # (n, 94)

    # ── V7 feature block ──────────────────────────────────────────────────────
    F_v7_block = None
    if V7_MODELS_AVAILABLE and v7_full_models is not None:
        try:
            # We need to extract funding_df for V7. We cached it in LIVE_PERP_CACHE.
            fund_ts, fund_rt = np.array([], dtype=np.int64), np.array([], dtype=np.float64)
            cached_perp = LIVE_PERP_CACHE.get(symbol)
            if cached_perp and cached_perp[0] is not None:
                funding_df = cached_perp[0]
                fund_ts = funding_df["ts_ms"].to_numpy().astype(np.int64)
                fund_rt = funding_df["funding_rate"].to_numpy().astype(np.float64)

            F_v7_block = build_v7_features(
                cl.astype(np.float64), hi.astype(np.float64), lo.astype(np.float64),
                vo.astype(np.float64), ofi.astype(np.float64), tbv.astype(np.float64),
                ts_int, atr.astype(np.float64), fund_ts, fund_rt
            )  # (n, V7_EXTRA)
            F_v7_full = np.hstack([F_full, F_v7_block]).astype(np.float32)  # (n, 130+)
        except Exception as e:
            print(f"  [V7] Feature build error for {symbol}: {e}")
            F_v7_block = None
            F_v7_full  = None
    else:
        F_v7_full = None

    # Extract latest row (V5 — always available)
    latest_row = F_full[-1:]
    current_price = float(cl[-1])
    latest_atr = float(atr[-1])

    # Extract regime features dynamically to avoid fragile index hardcoding
    latest_row_vec = F_full[-1]
    hurst = float(latest_row_vec[ALL_FEATURE_NAMES.index("hurst")])
    te = float(latest_row_vec[ALL_FEATURE_NAMES.index("trend_efficiency")])
    rv = float(latest_row_vec[ALL_FEATURE_NAMES.index("rv_1h")])
    rz = float(latest_row_vec[ALL_FEATURE_NAMES.index("rv_zscore_1d")])
    vb = float(latest_row_vec[ALL_FEATURE_NAMES.index("vol_burst_1h")])
    rp = float(latest_row_vec[ALL_FEATURE_NAMES.index("range_position")])
    
    # Compute regime
    regime = classify_regime(hurst, te, rv, rz, vb, rp)

    candidates = []
    horizon_data = {}

    # V7 latest row (130+ features) — used for blending if available
    latest_v7_full  = F_v7_full[-1:] if F_v7_full is not None else None
    latest_v7_only  = F_v7_block[-1:] if F_v7_block is not None else None

    for h in [60, 720]:
        horizon_data[h] = {}
        for direction in ["long", "short"]:
            m = models[h][direction]

            # ── V5 probability ────────────────────────────────────────────────
            p_v5_raw = m["clf"].inplace_predict(latest_row)
            p_v5_cal = float(m["calib"].predict(p_v5_raw)[0])
            mu_hat   = float(m["reg"].inplace_predict(latest_row)[0])

            # ── V7 blending (70/30) ───────────────────────────────────────────
            if (V7_MODELS_AVAILABLE
                    and v7_full_models is not None
                    and latest_v7_full is not None
                    and h in v7_full_models
                    and direction in v7_full_models[h]):
                try:
                    mv7 = v7_full_models[h][direction]
                    p_v7_raw = mv7["clf"].inplace_predict(latest_v7_full)
                    p_v7_cal = float(mv7["calib"].predict(p_v7_raw)[0])

                    # V7 micro gate — require V7 micro to agree
                    gate_pass = True
                    if (v7_micro_models is not None
                            and latest_v7_only is not None
                            and h in v7_micro_models
                            and direction in v7_micro_models[h]):
                        mv7m = v7_micro_models[h][direction]
                        p_micro_raw = mv7m["clf"].inplace_predict(latest_v7_only)
                        p_micro_cal = float(mv7m["calib"].predict(p_micro_raw)[0])
                        gate_pass = p_micro_cal >= V7_MICRO_GATE_THRESHOLD

                    if gate_pass:
                        # User-specified 70/30 blend: V7 has 70% weight
                        p_cal = V5_WEIGHT * p_v5_cal + V7_WEIGHT * p_v7_cal
                        # Effective threshold: weighted blend of both thresholds
                        thr_model = (V5_WEIGHT * float(m["p_threshold"])
                                     + V7_WEIGHT * float(mv7["p_threshold"]))
                    else:
                        # V7 micro gate failed — fall back to V5 alone
                        p_cal     = p_v5_cal
                        thr_model = float(m["p_threshold"])
                except Exception:
                    p_cal     = p_v5_cal
                    thr_model = float(m["p_threshold"])
            else:
                # V7 not available — use V5 only
                p_cal     = p_v5_cal
                thr_model = float(m["p_threshold"])

            scale = horizon_atr_scale(h)
            
            tp_mult = regime_tp_mult(hurst) * scale
            sl_mult = SL_MULT * scale
            tp_pct = tp_mult * latest_atr / current_price
            sl_pct = min(sl_mult * latest_atr / current_price, MAX_SL_PCT)
            
            if direction == "short":
                mu_hat = -mu_hat
                
            ev = p_cal * tp_pct - (1 - p_cal) * sl_pct - ROUND_TRIP_COST
            _h_scale = HORIZON_REGIME_EV_SCALE.get(regime, {}).get(h, 1.0)
            effective_ev = ev * _h_scale
            
            if direction == "long":
                tp_price = current_price + tp_mult * latest_atr
                sl_price = current_price * (1.0 - sl_pct)
            else:
                tp_price = current_price - tp_mult * latest_atr
                sl_price = current_price * (1.0 + sl_pct)
                
            candidate = {
                "horizon": h,
                "direction": direction,
                "p_cal": p_cal,
                "mu_hat": mu_hat,
                "tp_pct": tp_pct,
                "sl_pct": sl_pct,
                "tp_price": tp_price,
                "sl_price": sl_price,
                "ev": ev,
                "effective_ev": effective_ev,
                "thr_model": thr_model,
                "tp_mult": tp_mult,
                "sl_mult": sl_mult
            }
            candidates.append(candidate)
            horizon_data[h][direction] = candidate

    valid_candidates = []
    is_regime_ok = (regime not in HALT_REGIMES) or ALLOW_HALT_REGIMES
    if is_regime_ok:
        for c in candidates:
            min_tp = MIN_TP_TO_COST_RATIO * ROUND_TRIP_COST   # strict: no loosen
            tp_ok  = c["tp_pct"] >= min_tp
            p_ok   = c["p_cal"]  >= c["thr_model"]
            ev_ok  = c["ev"]     >= active_min_ev

            if tp_ok and p_ok and ev_ok:
                valid_candidates.append(c)

    selected_candidate = None
    if valid_candidates:
        valid_candidates.sort(key=lambda x: -x["effective_ev"])
        selected_candidate = valid_candidates[0]

    # Determine status reason for the Canditates Search Table
    status = "SKIP"
    if regime in HALT_REGIMES and not ALLOW_HALT_REGIMES:
        status = f"HALT ({regime})"
    else:
        if selected_candidate:
            v6_tag = " [V6]" if V6_MODELS_AVAILABLE else ""
            status = f"SIGNAL ({selected_candidate['direction'].upper()} {selected_candidate['horizon']}m){v6_tag}"
        else:
            candidates.sort(key=lambda x: -x["ev"])
            best_c = candidates[0]
            min_tp = MIN_TP_TO_COST_RATIO * ROUND_TRIP_COST
            tp_ok  = best_c["tp_pct"] >= min_tp
            p_ok   = best_c["p_cal"]  >= best_c["thr_model"]
            ev_ok  = best_c["ev"]     >= active_min_ev

            if not tp_ok:
                status = f"SKIP (TP_SMALL {best_c['tp_pct']*100:.2f}%)"
            elif not p_ok:
                status = f"SKIP (LOW_P {best_c['p_cal']*100:.1f}% < {best_c['thr_model']*100:.1f}%)"
            elif not ev_ok:
                status = f"SKIP (LOW_EV {best_c['ev']*100:+.3f}%)"
            else:
                status = "SKIP"

    # Maintain baseline compatibility for dashboard printing (Horizon 720)
    p_up_720 = horizon_data[720]["long"]["p_cal"]
    ev_long_720 = horizon_data[720]["long"]["ev"]
    p_down_720 = horizon_data[720]["short"]["p_cal"]
    ev_short_720 = horizon_data[720]["short"]["ev"]

    return {
        "p_up": p_up_720,
        "ev_long": ev_long_720,
        "p_down": p_down_720,
        "ev_short": ev_short_720,
        "hurst": hurst,
        "atr": latest_atr,
        "price": current_price,
        "regime": regime,
        "status": status,
        "selected_candidate": selected_candidate,
        "horizon_data": horizon_data
    }

# ── Account Cleanup & Position Monitoring ───────────────────────────────────────────────

def cleanup_account(client):
    # Cancel open orders
    try:
        open_orders = client.request("GET", "/fapi/v1/openOrders", signed=True)
        for order in open_orders:
            client.request("DELETE", "/fapi/v1/order", {"symbol": order["symbol"], "orderId": order["orderId"]}, signed=True)
    except Exception:
        pass

    # Close open positions
    try:
        positions = client.request("GET", "/fapi/v2/positionRisk", signed=True)
        for pos in positions:
            amt = float(pos["positionAmt"])
            if amt != 0:
                symbol = pos["symbol"]
                side = "SELL" if amt > 0 else "BUY"
                qty = abs(amt)
                client.request("POST", "/fapi/v1/order", {
                    "symbol": symbol,
                    "side": side,
                    "type": "MARKET",
                    "quantity": qty,
                    "reduceOnly": "true"
                }, signed=True)
    except Exception:
        pass
        
    save_active_trades({})

def get_exchange_positions(client):
    """Query currently open positions from Binance."""
    try:
        positions = client.request("GET", "/fapi/v2/positionRisk", signed=True)
        active = {}
        for pos in positions:
            amt = float(pos["positionAmt"])
            if amt != 0:
                active[pos["symbol"]] = {
                    "size": amt,
                    "entry_price": float(pos["entryPrice"]),
                    "mark_price": float(pos["markPrice"]),
                    "unrealized_pnl": float(pos["unRealizedProfit"])
                }
        return active
    except Exception:
        return {}

def monitor_and_close_positions(client):
    """Monitor active trades and trigger market order closes when TP/SL/time are hit."""
    active_trades = load_active_trades()
    if not active_trades:
        return False

    exchange_positions = get_exchange_positions(client)
    updated = False

    for symbol in list(active_trades.keys()):
        trade = active_trades[symbol]

        # 1. If position no longer exists on exchange, clean up local tracker
        if symbol not in exchange_positions:
            print(f"  [MONITOR] {symbol} — position closed externally, removing from tracker", flush=True)
            active_trades.pop(symbol)
            updated = True
            continue

        pos = exchange_positions[symbol]
        mark_price = pos["mark_price"]
        direction  = trade["direction"]
        tp_price   = trade["tp_price"]
        sl_price   = trade["sl_price"]
        qty        = trade["qty"]
        entry_time = trade["entry_time"]
        elapsed_mins = (time.time() - entry_time) / 60
        pnl = pos["unrealized_pnl"]

        trigger_close = False
        reason = ""

        if direction == "LONG":
            if mark_price >= tp_price:
                trigger_close = True; reason = "TAKE_PROFIT"
            elif mark_price <= sl_price:
                trigger_close = True; reason = "STOP_LOSS"
        else:  # SHORT
            if mark_price <= tp_price:
                trigger_close = True; reason = "TAKE_PROFIT"
            elif mark_price >= sl_price:
                trigger_close = True; reason = "STOP_LOSS"

        if elapsed_mins >= 720.0:
            trigger_close = True; reason = "TIMED_EXIT"

        if trigger_close:
            close_side = "SELL" if direction == "LONG" else "BUY"
            print(f"  [EXIT] {symbol} | {reason} | mark={mark_price:.4f} | PnL=${pnl:+.2f} | held={elapsed_mins:.0f}m", flush=True)
            try:
                client.request("POST", "/fapi/v1/order", {
                    "symbol": symbol,
                    "side": close_side,
                    "type": "MARKET",
                    "quantity": qty,
                    "reduceOnly": "true"
                }, signed=True)
                # Only remove from tracker if order succeeded
                active_trades.pop(symbol)
                updated = True
                print(f"  [EXIT] {symbol} closed OK ({reason})", flush=True)
            except Exception as e:
                print(f"  [EXIT ERROR] {symbol} close order failed: {e}", flush=True)

    if updated:
        save_active_trades(active_trades)
    return updated

def set_leverage(client, symbol, leverage=10):
    try:
        client.request("POST", "/fapi/v1/leverage", {
            "symbol": symbol,
            "leverage": leverage
        }, signed=True)
    except Exception:
        pass

# ── Dashboard & Monitoring UI ────────────────────────────────────────────────────

def render_dashboard(client, exchange_positions, active_trades):
    """Render the live trading dashboard. Reuses already-fetched position data."""
    balance, available = 0.0, 0.0
    try:
        res = client.request("GET", "/fapi/v2/account", signed=True)
        balance = float(res.get("totalWalletBalance", 0.0))
        available = float(res.get("availableBalance", 0.0))
    except Exception:
        pass
    
    # Clear console screen
    os.system('cls' if os.name == 'nt' else 'clear')
    
    print("=" * 110)
    print("                      CRYPTO ORACLE V5 HIGH-FREQUENCY DEMO ENGINE")
    if LOOSEN_MODE:
        print("                   [!!!] ACTIVE TESTING MODE — LOOSENED TRADE HURDLES ACTIVE [!!!]")
    print("=" * 110)
    print(f"  Cycle: {DASHBOARD_DATA['cycle']:04d}  |  Server Time Offset: {DASHBOARD_DATA['offset']:+d} ms  |  USDT Equity: ${balance:,.2f}  |  Available Margin: ${available:,.2f}")
    print(f"  Last Scan Completed: {DASHBOARD_DATA['last_scan_time']}  |  Open Positions: {len(exchange_positions)} / {MAX_OPEN_POSITIONS}")
    print("-" * 110)
    
    # 1. Active Positions table
    print(" ACTIVE POSITIONS IN PLAY:")
    print(" " + "-" * 106)
    print("  Symbol       Side    Size         Entry Price   Mark Price    Take Profit   Stop Loss     Unrealized P&L  Lev")
    print(" " + "-" * 106)
    if not exchange_positions:
        print("  < No Active Positions Open >")
    else:
        for symbol, pos in exchange_positions.items():
            trade = active_trades.get(symbol, {})
            tp_str = f"{trade.get('tp_price', 0.0):.4f}" if 'tp_price' in trade else "Unknown"
            sl_str = f"{trade.get('sl_price', 0.0):.4f}" if 'sl_price' in trade else "Unknown"
            side = trade.get("direction", "LONG")
            lev_str = f"{trade.get('leverage', 10)}x" if 'leverage' in trade else "10x"
            
            pnl = pos["unrealized_pnl"]
            pnl_str = f"${pnl:+.2f}"
            print(f"  {symbol:<12} {side:<7} {abs(pos['size']):<12.4f} {pos['entry_price']:<13.4f} {pos['mark_price']:<13.4f} {tp_str:<13} {sl_str:<13} {pnl_str:<15} {lev_str:<4}")
    print(" " + "-" * 106)
    print()

    # 2. Latest Scan Candidates table
    print(" LATEST CANDIDATE SEARCH & SIGNALS SCAN:")
    print(" " + "-" * 116)
    print("  Symbol       Price          Regime         P(up)     EV(long)    P(down)   EV(short)   Decision / Reason")
    print(" " + "-" * 116)
    if not DASHBOARD_DATA["scan_results"]:
        print("  < Scanning Candidates Pool... >")
    else:
        for r in DASHBOARD_DATA["scan_results"]:
            symbol = r["symbol"]
            if r.get("error"):
                print(f"  {symbol:<12} Error: {r['error'][:80]}")
                continue
                
            pred = r["pred"]
            p_up = pred["p_up"]
            ev_long = pred["ev_long"]
            p_down = pred["p_down"]
            ev_short = pred["ev_short"]
            price = pred["price"]
            regime = pred.get("regime", "UNKNOWN")
            
            status = r.get("status", "SKIP")
            if symbol in exchange_positions:
                status = "OPEN / HELD"
                
            print(f"  {symbol:<12} {price:<14.4f} {regime:<14} {p_up*100:>5.1f}%    {ev_long*100:>+7.3f}%    {p_down*100:>5.1f}%    {ev_short*100:>+7.3f}%    {status}")
    print(" " + "-" * 116)
    print()
    print("  [!] Press Ctrl+C in this Command Prompt to exit/stop the bot.")
    print("  [!] TO RESTART BOT CMD: python -u live_demo_engine.py")
    print("=" * 110)
    sys.stdout.flush()

# ── Main Execution Loop ──────────────────────────────────────────────────────────

def analyze_single_coin(symbol, candles, btc_candles, models, meta_feature_names, exchange_rules, basket_state, active_min_ev):
    """
    Worker task: fetch perp data (thread-safe — writes to local var
    then stores atomically), run ML prediction.
    """
    try:
        if symbol not in exchange_rules:
            return None
        rule = exchange_rules[symbol]
        if rule["status"] != "TRADING":
            return None

        # Fetch perp data here (in thread) but store result before predict_coin
        # to avoid race condition on the global LIVE_PERP_CACHE during prediction
        funding_df, metrics_df = fetch_live_perp_dfs(symbol)
        LIVE_PERP_CACHE[symbol] = (funding_df, metrics_df)  # atomic dict assignment

        pred = predict_coin(symbol, candles, btc_candles, models, meta_feature_names, basket_state, active_min_ev,
                            v7_full_models=v7_full_models, v7_micro_models=v7_micro_models)
        return {"symbol": symbol, "pred": pred, "rule": rule, "status": pred["status"]}
    except Exception as e:
        return {"symbol": symbol, "error": str(e)[:120], "status": "ERROR"}

def main():
    print("=" * 70, flush=True)
    print("  CRYPTO ORACLE V6 — LIVE DEMO ENGINE STARTING", flush=True)
    print("=" * 70, flush=True)

    # 1. Load V5 models
    print("[1] Loading V5 models...", flush=True)
    try:
        meta, models = load_v5_models()
        meta_feature_names = meta["feature_names"]
        print(f"    V5 models loaded. Feature count: {len(meta_feature_names)}", flush=True)
    except Exception as e:
        print(f"FATAL: Could not load V5 models: {e}", flush=True)
        return

    # 1b. Load V6 models (optional — falls back to V5-only if not trained yet)
    global V6_MODELS_AVAILABLE
    print("[1b] Attempting to load V6 models...", flush=True)
    v6_full_models, v6_micro_models = load_v6_models()
    if v6_full_models is not None:
        V6_MODELS_AVAILABLE = True
        print(f"     V6 ACTIVE — blend: V5×{V5_WEIGHT:.0%} + V6×{V6_WEIGHT:.0%}", flush=True)
    else:
        V6_MODELS_AVAILABLE = False
        v6_full_models = None
        v6_micro_models = None
        print("     V6 NOT available — running V5-only (train_v6.py first to enable V6)", flush=True)

    # 2. Init Binance Demo client
    print("[2] Connecting to Binance Demo API...", flush=True)
    client = BinanceFuturesClient(API_KEY, SECRET_KEY, DEMO_BASE)
    print(f"    Time offset: {client.time_offset:+d} ms", flush=True)

    # 3. Cleanup only if no persisted trades exist (avoid wiping live trades on restart)
    existing_trades = load_active_trades()
    if existing_trades:
        print(f"[3] Resuming — {len(existing_trades)} persisted trade(s) found, skipping cleanup.", flush=True)
    else:
        print("[3] No persisted trades — cleaning up stale positions...", flush=True)
        cleanup_account(client)

    # 4. Get exchange precision filters
    print("[4] Fetching exchange rules...", flush=True)
    exchange_rules = get_exchange_rules(client)
    print(f"    Loaded rules for {len(exchange_rules)} symbols.", flush=True)

    print(f"[5] Starting main loop. Exit monitor every 10s, scan every 60s.", flush=True)
    last_scan_time = 0

    while True:
        try:
            # Run position exit monitor every 10 seconds.
            # If a position was closed/exited, immediately refresh the dashboard UI!
            if monitor_and_close_positions(client):
                exchange_positions = get_exchange_positions(client)
                active_trades = load_active_trades()
                render_dashboard(client, exchange_positions, active_trades)
        except Exception as e:
            print(f"  [ERROR] Exit monitor exception: {e}", flush=True)
            
        now = time.time()
        if now - last_scan_time >= 60:
            last_scan_time = now
            DASHBOARD_DATA["cycle"] += 1
            DASHBOARD_DATA["last_scan_time"] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            try:
                # Sync client time to avoid timestamp drifting
                if DASHBOARD_DATA["cycle"] % 5 == 0:
                    client.sync_time()

                # Fallback to reload exchange rules if they are empty
                if not exchange_rules:
                    exchange_rules = get_exchange_rules(client)

                # Query active positions on exchange
                exchange_positions = get_exchange_positions(client)
                active_trades = load_active_trades()

                # Fetch candles for all basket coins in parallel to construct the true basket state
                basket_candles = {}
                def _fetch_helper(sym):
                    try:
                        return sym, fetch_klines(sym, 1000)
                    except Exception as e:
                        return sym, None

                with ThreadPoolExecutor(max_workers=12) as executor:
                    results = executor.map(_fetch_helper, STABLE_BASKET_COINS)
                    for sym, candles in results:
                        if candles is not None and len(candles) >= 1000:
                            basket_candles[sym] = candles

                # We need at least BTCUSDT to continue
                if "BTCUSDT" not in basket_candles:
                    print("  [Scan Cycle] BTCUSDT candles missing. Skipping cycle.", flush=True)
                    continue

                btc_candles = basket_candles["BTCUSDT"]

                # Calculate BTC 4-hour log return exactly like blind_backtest.py
                btc_4h_return = 0.0
                if len(btc_candles) >= 241:
                    btc_4h_return = math.log(btc_candles[-1]["close"] / btc_candles[-241]["close"])
                
                # Active min EV threshold conditioned on quiet BTC market
                is_btc_quiet = abs(btc_4h_return) < BTC_QUIET_4H_THRESHOLD
                if LOOSEN_MODE:
                    active_min_ev = LOOSENED_MIN_EV_THRESHOLD
                else:
                    active_min_ev = MIN_EV_PCT * (BTC_QUIET_EV_MULTIPLIER if is_btc_quiet else 1.0)

                # Build the true 25-coin cross-sectional basket state
                coin_data = {}
                for sym, candles in basket_candles.items():
                    n_c = len(candles)
                    ts_arr = np.fromiter((c["timestamp"] for c in candles), dtype=np.int64, count=n_c)
                    cl_arr = np.fromiter((c["close"] for c in candles), dtype=np.float64, count=n_c)
                    coin_data[sym] = (ts_arr, cl_arr)

                try:
                    basket_state = build_sim_basket_state(coin_data)
                except Exception as e:
                    print(f"  [Scan Cycle] Error building basket state: {e}", flush=True)
                    continue

                # Always scan candidates (coins not currently held)
                candidates_to_scan = [symbol for symbol in STABLE_BASKET_COINS if symbol not in exchange_positions]
                scan_results = []
                
                with ThreadPoolExecutor(max_workers=12) as executor:
                    futures = [
                        executor.submit(analyze_single_coin, symbol, basket_candles[symbol], btc_candles, models, meta_feature_names, exchange_rules, basket_state, active_min_ev)
                        for symbol in candidates_to_scan if symbol in basket_candles
                    ]
                    for fut in futures:
                        res = fut.result()
                        if res is not None:
                            scan_results.append(res)
                
                DASHBOARD_DATA["scan_results"] = scan_results

                # Execute Signals
                for r in scan_results:
                    if r.get("error") or len(exchange_positions) >= MAX_OPEN_POSITIONS_TOTAL:
                        continue
                        
                    symbol = r["symbol"]
                    pred = r["pred"]
                    rule = r["rule"]
                    
                    price = pred["price"]
                    atr = pred["atr"]
                    hurst = pred["hurst"]
                    selected = pred["selected_candidate"]

                    if selected is not None:
                        direction = selected["direction"].upper() # "LONG" or "SHORT"
                        side = "BUY" if direction == "LONG" else "SELL"
                        horizon = selected["horizon"]
                        
                        # 1. Conviction-scaled leverage exactly like blind_backtest.py
                        confidence_surplus = max(0.0, selected["p_cal"] - 0.59)
                        leverage = 20.0 + (confidence_surplus / 0.11) * 30.0
                        leverage = min(50.0, max(20.0, leverage))
                        
                        # Enforce leverage cap
                        MMR = 0.005 # 0.5% Maintenance Margin Rate
                        leverage_cap = 1.0 / (selected["sl_pct"] + MMR)
                        leverage = min(leverage, leverage_cap)
                        leverage = int(math.floor(leverage))
                        leverage = max(1, leverage)
                        
                        # 2. Place leverage configuration on Binance
                        set_leverage(client, symbol, leverage)

                        # 3. Calculate quantity
                        notional = ALLOCATED_MARGIN_PER_TRADE * leverage
                        raw_qty = notional / price
                        qty_rounded = round_quantity(raw_qty, rule["step_size"], rule["qty_precision"])
                        
                        if qty_rounded == 0:
                            continue
                            
                        try:
                            order_res = client.request("POST", "/fapi/v1/order", {
                                "symbol": symbol,
                                "side": side,
                                "type": "MARKET",
                                "quantity": qty_rounded
                            }, signed=True)
                            
                            fill_price = float(order_res.get("avgPrice", price))
                            if fill_price == 0:
                                fill_price = price

                            # 4. Calculate TP and SL prices using chosen candidate's prices
                            tp_diff = selected["tp_mult"] * atr
                            sl_diff = selected["sl_mult"] * atr
                            
                            if direction == "LONG":
                                tp_price = fill_price + tp_diff
                                sl_price = fill_price - sl_diff
                            else:
                                tp_price = fill_price - tp_diff
                                sl_price = fill_price + sl_diff

                            tp_rounded = round_price(tp_price, rule["tick_size"], rule["price_precision"])
                            sl_rounded = round_price(sl_price, rule["tick_size"], rule["price_precision"])

                            # Cap maximum SL distance to avoid extreme gaps
                            max_sl_dist = fill_price * MAX_SL_PCT
                            if direction == "LONG" and sl_rounded < (fill_price - max_sl_dist):
                                sl_rounded = round_price(fill_price - max_sl_dist, rule["tick_size"], rule["price_precision"])
                            elif direction == "SHORT" and sl_rounded > (fill_price + max_sl_dist):
                                sl_rounded = round_price(fill_price + max_sl_dist, rule["tick_size"], rule["price_precision"])

                            trade_record = {
                                "symbol": symbol,
                                "direction": direction,
                                "qty": qty_rounded,
                                "entry_price": fill_price,
                                "tp_price": tp_rounded,
                                "sl_price": sl_rounded,
                                "leverage": leverage,
                                "entry_time": time.time(),
                                "horizon": horizon
                            }
                            active_trades[symbol] = trade_record
                            save_active_trades(active_trades)
                            # Update local cache with all required fields for dashboard
                            exchange_positions[symbol] = {
                                "size": qty_rounded,
                                "entry_price": fill_price,
                                "mark_price": fill_price,
                                "unrealized_pnl": 0.0
                            }
                            print(f"  [TRADE] {direction} {symbol} | qty={qty_rounded} | entry={fill_price:.4f} | TP={tp_rounded:.4f} | SL={sl_rounded:.4f} | {leverage}x | horizon={horizon}m", flush=True)

                        except Exception as e:
                            print(f"  [ORDER ERROR] {symbol}: {e}", flush=True)

                # Draw updated dashboard — pass already-fetched data, no extra API calls
                render_dashboard(client, exchange_positions, active_trades)

            except Exception as e:
                print(f"  [ERROR] Scan cycle exception: {e}", flush=True)

        # Wait 10 seconds before next exit check
        time.sleep(10)

def get_exchange_rules(client):
    """Fetch and parse exchange rules from Binance Futures."""
    rules = {}
    try:
        info = client.request("GET", "/fapi/v1/exchangeInfo")
        for sym_info in info.get("symbols", []):
            symbol = sym_info["symbol"]
            status = sym_info["status"]
            qty_precision = int(sym_info["quantityPrecision"])
            price_precision = int(sym_info["pricePrecision"])
            
            step_size = 0.0
            tick_size = 0.0
            for f in sym_info.get("filters", []):
                if f["filterType"] == "LOT_SIZE":
                    step_size = float(f["stepSize"])
                elif f["filterType"] == "PRICE_FILTER":
                    tick_size = float(f["tickSize"])
                    
            rules[symbol] = {
                "status": status,
                "qty_precision": qty_precision,
                "price_precision": price_precision,
                "step_size": step_size,
                "tick_size": tick_size
            }
    except Exception:
        pass
    return rules

def round_quantity(qty, step_size, precision):
    if step_size > 0:
        qty = math.floor(qty / step_size) * step_size
    return round(qty, precision)

def round_price(price, tick_size, precision):
    if tick_size > 0:
        price = round(price / tick_size) * tick_size
    return round(price, precision)

if __name__ == "__main__":
    main()
