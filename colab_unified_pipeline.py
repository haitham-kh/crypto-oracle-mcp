"""
╔═════════════════════════════════════════════════════════════════════════════╗
║  CRYPTO ORACLE — UNIFIED COLAB PIPELINE (V5 + V6 FEATURES)                  ║
║                                                                             ║
║  A single self-contained script for Google Colab that:                      ║
║    1. Downloads 1m OHLCV & Perp Microstructure Data (24 months, 20 coins)   ║
║    2. Computes the complete 130-feature stack (V1–V6 + Coin One-Hot)        ║
║       V5 (94): OFI, CVD, Hurst, regime, VPVR, perp, cross-sectional         ║
║       V6 (36): Donchian, Anchored VWAP, Value Area, Sweeps, SMA, Fib, BB    ║
║    3. Generates Multi-Horizon Triple-Barrier Labels                         ║
║    4. Saves final clean training parquets to 'processed/'                   ║
║       Columns: timestamp_ms, X_0..X_129, y_60, net_60, … any_valid          ║
║                                                                             ║
║  No external dependencies or multiple files needed.                         ║
║  Optimized to run under 12GB RAM by processing coin-by-coin.                ║
╚═════════════════════════════════════════════════════════════════════════════╝
"""

# ── 0. Installs ──────────────────────────────────────────────────────────────
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "polars", "pyarrow", "requests"], check=False)

import os, time, datetime, json, math, gc, glob
import requests
import polars as pl
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
# Set to '/content/drive/MyDrive/crypto_oracle_data' if mounting Google Drive
ROOT = "/content/data"

PROD_BASE = "https://fapi.binance.com"
MONTHS_HISTORY = 24

COINS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "ADAUSDT", "AVAXUSDT", "DOGEUSDT", "LINKUSDT", "DOTUSDT",
    "LTCUSDT", "NEARUSDT", "UNIUSDT", "INJUSDT", "SUIUSDT",
    "AAVEUSDT", "APTUSDT", "FILUSDT", "ATOMUSDT", "OPUSDT",
]

# Labeling parameters
HORIZONS = [60, 720]
ROUND_TRIP_COST = 0.0022
SL_MULT = 1.2
MIN_TP_COST_R = 1.5
WARMUP_BARS = 1500
SAMPLE_EVERY = 30

COIN_ONEHOT_NAMES = [
    "AAVEUSDT", "ADAUSDT", "AVAXUSDT", "BNBUSDT", "BTCUSDT",
    "DOGEUSDT", "ETHUSDT", "LINKUSDT", "PEPEUSDT", "SHIBUSDT",
    "SOLUSDT", "UNIUSDT", "WIFUSDT", "XRPUSDT",
]

# Dirs setup
KLINES_DIR = os.path.join(ROOT, "raw", "klines")
PERP_DIR   = os.path.join(ROOT, "raw", "perp")
OUT_DIR    = os.path.join(ROOT, "processed")
os.makedirs(KLINES_DIR, exist_ok=True)
os.makedirs(PERP_DIR,   exist_ok=True)
os.makedirs(OUT_DIR,    exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# 2. DOWNLOAD ENGINE (PHASE 1)
# ─────────────────────────────────────────────────────────────────────────────

def _get(url, params, retries=5, backoff=2.0):
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=30)
            if r.status_code == 429:
                wait = backoff * (2 ** attempt)
                print(f"    [RateLimit] sleeping {wait:.0f}s ...")
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r.json()
        except Exception as e:
            if attempt == retries - 1:
                raise
            time.sleep(backoff * (2 ** attempt))

def _month_range(months_back):
    now = datetime.datetime.utcnow()
    end_y, end_m = now.year, now.month - 1
    if end_m == 0:
        end_y -= 1; end_m = 12
    result = []
    y, m = end_y, end_m
    for _ in range(months_back):
        result.append((y, m))
        m -= 1
        if m == 0:
            y -= 1; m = 12
    return list(reversed(result))

def _month_start_end_ms(year, month):
    start = datetime.datetime(year, month, 1, tzinfo=datetime.timezone.utc)
    if month == 12:
        end = datetime.datetime(year + 1, 1, 1, tzinfo=datetime.timezone.utc)
    else:
        end = datetime.datetime(year, month + 1, 1, tzinfo=datetime.timezone.utc)
    return int(start.timestamp() * 1000), int(end.timestamp() * 1000) - 1

import zipfile, io

def download_klines_month(symbol, year, month):
    url = f"https://data.binance.vision/data/futures/um/monthly/klines/{symbol}/1m/{symbol}-1m-{year:04d}-{month:02d}.zip"
    try:
        r = requests.get(url, timeout=30)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        
        z = zipfile.ZipFile(io.BytesIO(r.content))
        csv_name = z.namelist()[0]
        csv_bytes = z.read(csv_name)
        has_header = b"open_time" in csv_bytes[:200]
        
        df = pl.read_csv(io.BytesIO(csv_bytes), has_header=has_header)
        if not has_header:
            # Map default columns manually if file has no header row
            cols = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume", "ignore"]
            df.columns = cols[:len(df.columns)]
            
        df = df.select([
            pl.col("open_time").cast(pl.Int64).alias("timestamp"),
            pl.col("open").cast(pl.Float64),
            pl.col("high").cast(pl.Float64),
            pl.col("low").cast(pl.Float64),
            pl.col("close").cast(pl.Float64),
            pl.col("volume").cast(pl.Float64),
            pl.col("quote_volume").cast(pl.Float64),
            pl.col("count").cast(pl.Int64).alias("num_trades"),
            pl.col("taker_buy_volume").cast(pl.Float64)
        ])
        
        df = df.with_columns([
            pl.from_epoch(pl.col("timestamp"), time_unit="ms").alias("timestamp"),
            (2.0 * pl.col("taker_buy_volume") - pl.col("volume")).alias("ofi")
        ])
        return df
    except Exception as e:
        print(f"    Error downloading klines for {symbol} {year}-{month:02d}: {e}")
        return None

def download_all_klines():
    months = _month_range(MONTHS_HISTORY)
    total = len(COINS) * len(months)
    done = 0
    for symbol in COINS:
        print(f"[KLINES] {symbol}")
        for year, month in months:
            fname = os.path.join(KLINES_DIR, f"{symbol}_1m_{year:04d}-{month:02d}.parquet")
            done += 1
            if os.path.exists(fname):
                continue
            try:
                df = download_klines_month(symbol, year, month)
                if df is not None and len(df) > 0:
                    df.write_parquet(fname, compression="zstd", compression_level=3)
                    print(f"  [{done}/{total}] {year}-{month:02d} {len(df):,} bars OK")
                else:
                    print(f"  [{done}/{total}] {year}-{month:02d} Skipped (404 / No Data)")
            except Exception as e:
                print(f"  [{done}/{total}] {year}-{month:02d} ERROR: {e}")
            gc.collect()

def download_funding_month(symbol, year, month):
    url = f"https://data.binance.vision/data/futures/um/monthly/fundingRate/{symbol}/{symbol}-fundingRate-{year:04d}-{month:02d}.zip"
    try:
        r = requests.get(url, timeout=30)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        
        z = zipfile.ZipFile(io.BytesIO(r.content))
        csv_name = z.namelist()[0]
        csv_bytes = z.read(csv_name)
        has_header = b"calc_time" in csv_bytes[:200]
        
        df = pl.read_csv(io.BytesIO(csv_bytes), has_header=has_header)
        if not has_header:
            df.columns = ["calc_time", "funding_interval_hours", "last_funding_rate"][:len(df.columns)]
            
        df = df.select([
            pl.col("calc_time").cast(pl.Int64).alias("ts_ms"),
            pl.col("last_funding_rate").cast(pl.Float64).alias("funding_rate")
        ])
        return df
    except Exception as e:
        print(f"    Error downloading funding for {symbol} {year}-{month:02d}: {e}")
        return None

def download_funding(symbol):
    months = _month_range(MONTHS_HISTORY)
    dfs = []
    for year, month in months:
        df = download_funding_month(symbol, year, month)
        if df is not None:
            dfs.append(df)
    if not dfs:
        return None
    return pl.concat(dfs).sort("ts_ms").unique("ts_ms")

def _download_hist_endpoint(symbol, endpoint_path, value_key, rename_key, period="5m"):
    url = f"{PROD_BASE}{endpoint_path}"
    all_rows = []
    start_ms = _month_start_end_ms(*_month_range(MONTHS_HISTORY)[0])[0]
    now_ms   = int(time.time() * 1000)
    cur = start_ms
    
    # Track geo-block status
    is_blocked = False
    
    while cur < now_ms:
        params = {"symbol": symbol, "period": period, "limit": 500, "startTime": cur}
        try:
            r = requests.get(url, params=params, timeout=30)
            if r.status_code == 451:
                is_blocked = True
                break
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            print(f"    [{endpoint_path}] {symbol} error at {cur}: {e}")
            break
        if not data:
            break
        for item in data:
            ts = int(item.get("timestamp") or item.get("fundingTime", 0))
            val = float(item.get(value_key, 0.0))
            all_rows.append({"ts_ms": ts, rename_key: val})
        last_ts = int(data[-1].get("timestamp") or data[-1].get("fundingTime", 0))
        if last_ts <= cur:
            break
        cur = last_ts + 1
        time.sleep(0.08)
        
    if is_blocked:
        print(f"    [{endpoint_path}] Geoblocked (HTTP 451) for Google Colab. Skipping daily stats.")
        return None
    if not all_rows:
        return None
    return pl.DataFrame(all_rows).sort("ts_ms").unique("ts_ms")

def download_metrics(symbol):
    dfs = {}
    df_oi = _download_hist_endpoint(symbol, "/futures/data/openInterestHist", "sumOpenInterest", "oi")
    if df_oi is not None:
        dfs["oi"] = df_oi
    df_lsr_top = _download_hist_endpoint(symbol, "/futures/data/topLongShortAccountRatio", "longShortRatio", "lsr_top")
    if df_lsr_top is not None:
        dfs["lsr_top"] = df_lsr_top
    df_lsr_global = _download_hist_endpoint(symbol, "/futures/data/globalLongShortAccountRatio", "longShortRatio", "lsr_global")
    if df_lsr_global is not None:
        dfs["lsr_global"] = df_lsr_global
    df_taker = _download_hist_endpoint(symbol, "/futures/data/takerlongshortRatio", "buySellRatio", "taker_ratio")
    if df_taker is not None:
        dfs["taker"] = df_taker
        
    if not dfs:
        return None
        
    base = list(dfs.values())[0]
    for df in list(dfs.values())[1:]:
        base = base.join(df, on="ts_ms", how="outer_coalesce")
    for col in ["oi", "lsr_top", "lsr_global", "taker_ratio"]:
        if col not in base.columns:
            base = base.with_columns(pl.lit(None).cast(pl.Float64).alias(col))
    return base.sort("ts_ms").select(["ts_ms", "oi", "lsr_top", "lsr_global", "taker_ratio"])

def download_all_perp():
    for i, symbol in enumerate(COINS, 1):
        print(f"[PERP] {symbol} ({i}/{len(COINS)})")
        fund_path = os.path.join(PERP_DIR, f"{symbol}_funding.parquet")
        if not os.path.exists(fund_path):
            try:
                df_f = download_funding(symbol)
                if df_f is not None:
                    df_f.write_parquet(fund_path, compression="zstd")
                    print(f"  Funding downloaded -> {fund_path}")
            except Exception as e:
                print(f"  Funding ERROR: {e}")
        metr_path = os.path.join(PERP_DIR, f"{symbol}_metrics.parquet")
        if not os.path.exists(metr_path):
            try:
                df_m = download_metrics(symbol)
                if df_m is not None:
                    df_m.write_parquet(metr_path, compression="zstd")
                    print(f"  Metrics downloaded -> {metr_path}")
            except Exception as e:
                print(f"  Metrics ERROR: {e}")
        gc.collect()


# ─────────────────────────────────────────────────────────────────────────────
# 3. MATH & FEATURE PRIMITIVES (PHASE 2A)
# ─────────────────────────────────────────────────────────────────────────────

def _roll_mean(x, w):
    out = np.full(len(x), np.nan)
    cs  = np.cumsum(np.nan_to_num(x, nan=0.0))
    out[w-1:] = (cs[w-1:] - np.concatenate([[0], cs[:-w]])) / w
    return out

def _roll_std(x, w):
    m  = _roll_mean(x, w)
    m2 = _roll_mean(x**2, w)
    return np.sqrt(np.maximum(m2 - m**2, 0.0))

def _ema(x, period):
    out = np.full(len(x), np.nan)
    k   = 2.0 / (period + 1)
    if period > len(x): return out
    out[period-1] = np.nanmean(x[:period])
    for i in range(period, len(x)):
        out[i] = x[i]*k + out[i-1]*(1-k)
    return out

def _atr(high, low, close, p=14):
    n  = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    tr[1:] = np.maximum.reduce([high[1:]-low[1:],
                                 np.abs(high[1:]-close[:-1]),
                                 np.abs(low[1:]-close[:-1])])
    atr = np.full(n, np.nan)
    if p > n: return atr
    atr[p-1] = tr[:p].mean()
    k = (p-1)/p
    for i in range(p, n):
        atr[i] = atr[i-1]*k + tr[i]/p
    return atr

def _rsi(close, period=14):
    n   = len(close)
    out = np.full(n, np.nan)
    d   = np.diff(close)
    g   = np.where(d > 0, d, 0.0)
    l   = np.where(d < 0, -d, 0.0)
    if period > len(g): return out
    ag = np.full(len(g), np.nan); al = ag.copy()
    ag[period-1] = g[:period].mean()
    al[period-1] = l[:period].mean()
    for i in range(period, len(g)):
        ag[i] = (ag[i-1]*(period-1) + g[i]) / period
        al[i] = (al[i-1]*(period-1) + l[i]) / period
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = np.where(al > 0, ag/al, 100.0)
    out[1:] = 100.0 - 100.0/(1.0+rs)
    return out

def _hurst_scalar(prices):
    if len(prices) < 50: return 0.5
    rets = np.diff(np.log(np.maximum(prices, 1e-10)))
    n    = len(rets)
    lags, rs = [], []
    for lag in [max(10, n//8), max(10, n//4), max(10, n//2)]:
        if lag >= n: continue
        sub = rets[-lag:]
        dev = np.cumsum(sub - sub.mean())
        R   = dev.max() - dev.min()
        S   = sub.std(ddof=1)
        if S > 0 and R > 0:
            lags.append(np.log(lag)); rs.append(np.log(R/S))
    if len(rs) < 2: return 0.5
    return float(np.clip(np.polyfit(lags, rs, 1)[0], 0.01, 0.99))

def _hurst_rolling(close, w=500, stride=30):
    n   = len(close)
    out = np.full(n, 0.5)
    last = 0.5
    for i in range(w, n):
        if (i - w) % stride == 0:
            last = _hurst_scalar(close[i-w:i])
        out[i] = last
    return out

def _causal_zscore(x, window):
    m = _roll_mean(x, window); s = _roll_std(x, window)
    with np.errstate(divide='ignore', invalid='ignore'):
        z = np.where(s > 1e-12, (x - m)/s, 0.0)
    return np.nan_to_num(np.clip(z, -5, 5), nan=0.0)

def _rolling_log_return(close, w):
    n   = len(close); out = np.zeros(n)
    if n > w:
        out[w:] = np.log(np.maximum(close[w:], 1e-12) / np.maximum(close[:-w], 1e-12))
    return out

def _rolling_rv(close, w):
    n    = len(close); out = np.zeros(n)
    rets = np.zeros(n)
    rets[1:] = np.log(np.maximum(close[1:],1e-12)/np.maximum(close[:-1],1e-12))
    s = _roll_std(rets, w)
    out[:] = np.nan_to_num(s * math.sqrt(max(w,1)), nan=0.0)
    return out

def _rolling_beta(ar, br, w):
    n = len(ar); out = np.zeros(n)
    a = np.nan_to_num(ar, nan=0.0); b = np.nan_to_num(br, nan=0.0)
    ab  = _roll_mean(a*b, w); bb = _roll_mean(b*b, w)
    am  = _roll_mean(a,  w); bm = _roll_mean(b,  w)
    with np.errstate(divide='ignore', invalid='ignore'):
        cov   = ab - am*bm
        var_b = np.maximum(bb - bm**2, 1e-12)
        beta  = cov / var_b
    out[:] = np.nan_to_num(np.clip(beta, -3, 3), nan=0.0)
    return out

def _nr7(high, low):
    n = len(high); out = np.zeros(n)
    if n < 7: return out
    rng = high - low
    try:
        from numpy.lib.stride_tricks import sliding_window_view
        prev = sliding_window_view(rng[:-1], 7).min(axis=1)
        out[7:] = (rng[7:] <= prev[:n-7]).astype(np.float64)
    except Exception:
        for i in range(7, n):
            out[i] = 1.0 if rng[i] <= rng[i-7:i].min() else 0.0
    return out

def _inside_bar_streak(high, low):
    n = len(high); out = np.zeros(n)
    if n < 2: return out
    inside = np.zeros(n, dtype=bool)
    inside[1:] = (high[1:] <= high[:-1]) & (low[1:] >= low[:-1])
    cum = 0
    for i in range(n):
        cum = cum+1 if inside[i] else 0
        out[i] = min(cum, 10) / 10.0
    return out

def _vpvr_poc(volume, close, window=1440, stride=30, n_bins=24):
    n = len(close); poc_d = np.zeros(n); conc = np.zeros(n)
    if n < window: return poc_d, conc
    last_pd, last_c = 0.0, 0.0
    for i in range(window, n):
        if (i-window) % stride == 0:
            cl_w = close[i-window:i]; v_w = volume[i-window:i]
            lo, hi = float(cl_w.min()), float(cl_w.max())
            if hi > lo:
                edges = np.linspace(lo, hi, n_bins+1)
                bins  = np.clip(np.searchsorted(edges, cl_w, 'right')-1, 0, n_bins-1)
                bv    = np.bincount(bins, weights=v_w, minlength=n_bins)
                tot   = bv.sum()
                if tot > 0:
                    pi     = int(np.argmax(bv))
                    pp     = (edges[pi]+edges[pi+1])/2
                    last_pd = float(np.clip((close[i]-pp)/max(close[i],1e-12),-0.5,0.5))*2
                    last_c  = float(np.partition(bv,-3)[-3:].sum()/tot)
        poc_d[i] = last_pd; conc[i] = last_c
    return poc_d, conc

def build_features_v2(close, high, low, volume, ofi, timestamps_ms, btc_close=None):
    n   = len(close)
    F   = np.full((n, 25), np.nan)
    vs  = np.where(volume > 0, volume, 1.0)
    ofi_score = np.clip(ofi / vs, -1, 1)
    atr_arr   = _atr(high, low, close, 14)

    F[:, 0] = ofi_score
    F[5:, 1] = np.clip(ofi_score[5:]-ofi_score[:-5], -1, 1)
    F[:, 2]  = 0.0
    rng = np.maximum(high-low, 1e-10)
    F[:, 3]  = np.clip(np.abs(ofi)/np.maximum(rng*vs, 1e-10), 0, 1)

    cvd = np.cumsum(np.nan_to_num(ofi_score, nan=0.0))
    for w, c_col, d_col in [(15,4,7),(60,5,8),(240,6,9)]:
        cvd_ma = _roll_mean(cvd, w)
        delta  = np.zeros(n); delta[w:] = cvd[w:] - cvd[:-w]
        F[:, c_col] = np.clip(delta / np.maximum(np.abs(_roll_mean(cvd, w*4))+1e-9, 1e-9), -1, 1)
        F[:, d_col] = np.clip((cvd - cvd_ma) / (np.maximum(_roll_std(cvd, w*4), 1e-9)), -3, 3) / 3

    hurst = _hurst_rolling(close, 500, 30)
    F[:, 10] = hurst
    log_r  = np.zeros(n); log_r[1:] = np.log(np.maximum(close[1:],1e-12)/np.maximum(close[:-1],1e-12))
    path   = np.abs(log_r)
    for i in range(100, n):
        net = abs(math.log(max(close[i],1e-12)/max(close[i-100],1e-12)))
        gross = path[i-100:i].sum()
        F[i, 11] = net / max(gross, 1e-10)
    F[:, 12] = np.clip(_causal_zscore(log_r, 60), -1, 1)
    F[:, 13] = 0.0
    rv1h = _rolling_rv(close, 60); rv1d = _rolling_rv(close, 1440)
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(rv1d > 1e-9, rv1h/rv1d, 1.0)
    F[:, 14] = np.nan_to_num(np.clip(np.log(np.maximum(ratio,1e-6)),-2,2)/2, nan=0.0)
    F[:, 15] = 0.0

    money_flow = np.where(close > np.roll(close,1), ofi_score, -ofi_score)
    mf_ma = _roll_mean(money_flow, 60)
    F[:, 16] = np.clip((mf_ma+1)/2, 0, 1)
    F[:, 17] = np.clip((-mf_ma+1)/2, 0, 1)

    F[:, 18] = _causal_zscore(volume, 60)
    with np.errstate(divide='ignore', invalid='ignore'):
        atr_norm_ofi = np.where(atr_arr>0, ofi_score*np.minimum(atr_arr/np.maximum(np.abs(close),1e-10),1), 0.0)
    F[:, 19] = np.clip(atr_norm_ofi, -1, 1)

    secs = (timestamps_ms / 1000).astype(np.int64)
    hour = ((secs // 3600) % 24).astype(np.float64)
    dow  = (((secs // 86400)+4) % 7).astype(np.float64)
    F[:, 20] = np.sin(2*math.pi*hour/24)
    F[:, 21] = np.cos(2*math.pi*hour/24)
    F[:, 22] = np.sin(2*math.pi*dow/7)
    F[:, 23] = np.cos(2*math.pi*dow/7)

    if btc_close is not None and len(btc_close) == n:
        F[60:, 24] = np.clip(np.log(np.maximum(btc_close[60:],1e-12)/np.maximum(btc_close[:-60],1e-12))*100, -10, 10)/10
    else:
        F[:, 24] = 0.0

    return F, atr_arr

def build_v3_from_v2(F_v2, close, high, low, atr_arr):
    n      = len(close)
    F_kept = F_v2[:, [0,1,3,7,8,9,10,11,12,14,16,17,18,19,20,21,22,23,24]]

    P = np.full((n, 7), np.nan)
    ema200 = _ema(close, 200)
    with np.errstate(divide='ignore', invalid='ignore'):
        P[:, 0] = np.clip((close-ema200)/np.maximum(ema200,1e-10),-0.5,0.5)*2
    sma20  = _roll_mean(close, 20); std20 = _roll_std(close, 20)
    bb_u   = sma20+2*std20; bb_l = sma20-2*std20
    P[:, 1] = np.clip((close-bb_l)/np.maximum(bb_u-bb_l,1e-10),0,1)*2-1
    P[:, 2] = (_rsi(close, 14)-50)/50
    atr_rank = np.full(n, np.nan)
    for i in range(100, n):
        w = atr_arr[i-100:i+1]; valid = w[~np.isnan(w)]
        if len(valid): atr_rank[i] = np.mean(valid <= atr_arr[i])
    P[:, 3] = np.nan_to_num(atr_rank*2-1, nan=0.0)
    P[5:,  4] = np.clip((close[5:]-close[:-5])/np.maximum(close[:-5],1e-10)*100,-5,5)/5
    P[60:, 5] = np.clip((close[60:]-close[:-60])/np.maximum(close[:-60],1e-10)*100,-10,10)/10
    for i in range(20, n):
        hi20 = high[i-19:i+1].max(); lo20 = low[i-19:i+1].min(); rng = hi20-lo20
        if rng > 0: P[i, 6] = (close[i]-lo20)/rng*2-1

    F_partial = np.hstack([F_kept, P])
    I = np.zeros((n, 3))
    I[:, 0] = np.nan_to_num(F_partial[:,6]*F_partial[:,0], nan=0.0)
    I[:, 1] = np.nan_to_num(F_partial[:,9]*F_partial[:,2], nan=0.0)
    I[:, 2] = np.nan_to_num(F_partial[:,7]*F_partial[:,5], nan=0.0)
    return np.hstack([F_partial, I])


# ─────────────────────────────────────────────────────────────────────────────
# 4. ADVANCED V4/V5 FEATURE BUILDERS (PHASE 2B)
# ─────────────────────────────────────────────────────────────────────────────

def build_v4_extras(close, high, low, volume, ofi, taker_buy_volume, timestamps_ms, btc_close=None):
    n  = len(close)
    F  = np.zeros((n, 33), dtype=np.float32)
    lr = np.zeros(n)
    lr[1:] = np.log(np.maximum(close[1:],1e-12)/np.maximum(close[:-1],1e-12))

    for j, w in enumerate([5,15,60,240,1440]):
        F[:,j] = _causal_zscore(ofi, w)

    cvd = np.cumsum(np.nan_to_num(ofi, nan=0.0))
    for j, w in enumerate([60,240], start=5):
        d = np.zeros(n); d[w:] = cvd[w:]-cvd[:-w]
        F[:,j] = _causal_zscore(d, 1440)

    sv  = _roll_mean(volume, 60)*60; sb = _roll_mean(taker_buy_volume, 60)*60
    with np.errstate(divide='ignore', invalid='ignore'):
        rat = np.where(sv>1e-9, sb/sv, 0.5)
    F[:,7] = np.nan_to_num(np.clip((rat-0.5)*2,-1,1), nan=0.0)

    pd_, cn_ = _vpvr_poc(volume, close, 1440)
    F[:,8] = np.nan_to_num(pd_, nan=0.0)
    F[:,9] = np.nan_to_num(np.clip(cn_*2-1,-1,1), nan=0.0)

    F[:,10] = _causal_zscore(volume, 5)
    v1h = _roll_mean(volume,60); v1d = _roll_mean(volume,1440)
    with np.errstate(divide='ignore', invalid='ignore'):
        burst = np.where(v1d>1e-9, v1h/v1d, 1.0)
    F[:,11] = np.nan_to_num(np.clip(np.log(np.maximum(burst,1e-6)),-2,2)/2, nan=0.0)

    rv5   = _rolling_rv(close, 5)
    rv1h  = _rolling_rv(close, 60)
    rv4h  = _rolling_rv(close, 240)
    rv1d  = _rolling_rv(close, 1440)
    F[:,12] = np.nan_to_num(np.clip(rv5*100,0,5)/5, nan=0.0)
    F[:,13] = np.nan_to_num(np.clip(rv1h*100,0,5)/5, nan=0.0)
    with np.errstate(divide='ignore', invalid='ignore'):
        rvr = np.where(rv4h>1e-9, rv1h/rv4h, 1.0)
    F[:,14] = np.nan_to_num(np.clip(np.log(np.maximum(rvr,1e-6)),-1.5,1.5)/1.5, nan=0.0)
    F[:,15] = _causal_zscore(rv1h, 1440)
    vov = _roll_std(rv1h, 1440)
    F[:,16] = np.nan_to_num(np.clip(vov*100,0,2)/2, nan=0.0)

    F[:,17] = _nr7(high, low)
    F[:,18] = _inside_bar_streak(high, low)
    rng   = high - low
    rsh   = _roll_mean(rng,60); rlg = _roll_mean(rng,1440)
    with np.errstate(divide='ignore', invalid='ignore'):
        rr = np.where(rlg>1e-9, rsh/rlg, 1.0)
    F[:,19] = np.nan_to_num(np.clip(np.log(np.maximum(rr,1e-6)),-2,2)/2, nan=0.0)
    tr = np.maximum(rng, np.maximum(np.abs(high-np.roll(close,1)), np.abs(low-np.roll(close,1))))
    tr[0] = rng[0]
    atr_s = _roll_mean(tr,60); atr_l = _roll_mean(tr,1440)
    with np.errstate(divide='ignore', invalid='ignore'):
        ar = np.where(atr_l>1e-9, atr_s/atr_l, 1.0)
    F[:,20] = np.nan_to_num(np.clip(np.log(np.maximum(ar,1e-6)),-2,2)/2, nan=0.0)

    secs = (timestamps_ms/1000).astype(np.int64)
    hour = ((secs//3600)%24).astype(np.int32)
    dow  = (((secs//86400)+4)%7).astype(np.int32)
    F[:,21] = ((hour>=0)&(hour<8)).astype(np.float32)*2-1
    F[:,22] = ((hour>=7)&(hour<16)).astype(np.float32)*2-1
    F[:,23] = ((hour>=13)&(hour<22)).astype(np.float32)*2-1
    F[:,24] = ((dow==5)|(dow==6)).astype(np.float32)*2-1

    if btc_close is not None and len(btc_close)==n:
        blr = np.zeros(n)
        blr[1:] = np.log(np.maximum(btc_close[1:],1e-12)/np.maximum(btc_close[:-1],1e-12))
        brv1h = _rolling_rv(btc_close, 60)
        br1h  = np.zeros(n); br1h[60:]  = np.log(np.maximum(btc_close[60:],1e-12)/np.maximum(btc_close[:-60],1e-12))
        br4h  = np.zeros(n); br4h[240:] = np.log(np.maximum(btc_close[240:],1e-12)/np.maximum(btc_close[:-240],1e-12))
        F[:,25] = np.nan_to_num(np.clip(brv1h*100,0,3)/3, nan=0.0)
        F[:,26] = np.nan_to_num(np.clip(br1h*100,-10,10)/10, nan=0.0)
        F[:,27] = np.nan_to_num(np.clip(br4h*100,-15,15)/15, nan=0.0)
        F[:,28] = _rolling_beta(lr, blr, 1440)/3.0

    for j, w in enumerate([15,60,240,1440], start=29):
        F[:,j] = _causal_zscore(_rolling_log_return(close,w), 1440)

    return F

def build_v4_from_components(F_v2, close, high, low, volume, ofi, taker_buy_volume, timestamps_ms, atr_arr, btc_close=None):
    F_v3 = build_v3_from_v2(F_v2, close, high, low, atr_arr)
    F_ex = build_v4_extras(close, high, low, volume, ofi, taker_buy_volume, timestamps_ms, btc_close)
    return np.hstack([F_v3, F_ex]).astype(np.float32)

def _asof_align(target_ts, src_ts, src_vals):
    t   = np.asarray(target_ts, dtype=np.int64)
    st  = np.asarray(src_ts,    dtype=np.int64)
    sv  = np.asarray(src_vals,  dtype=np.float64)
    if len(st)==0: return np.full(len(t), np.nan)
    idx = np.searchsorted(st, t, side='right')-1
    return np.where(idx>=0, sv[np.maximum(idx,0)], np.nan)

def build_v5_perp_features(timestamps_ms, close, funding_df, metrics_df):
    n   = len(timestamps_ms)
    F   = np.zeros((n, 12), dtype=np.float32)

    if funding_df is not None and len(funding_df) > 0:
        fts  = funding_df["ts_ms"].to_numpy().astype(np.int64)
        fr   = funding_df["funding_rate"].to_numpy().astype(np.float64)
        f_aligned = _asof_align(timestamps_ms, fts, fr)
        F[:,0] = np.nan_to_num(np.clip(f_aligned*1000,-5,5), nan=0.0)
        F[:,1] = _causal_zscore(np.nan_to_num(f_aligned,nan=0.0), 30*3*8)
        f_prev = _asof_align(timestamps_ms - 480*60_000, fts, fr)
        F[:,2] = np.nan_to_num(np.clip((f_aligned-f_prev)*1000,-2,2), nan=0.0)
        F[:,3] = (np.abs(f_aligned) > 0.0005).astype(np.float32)

    if metrics_df is not None and len(metrics_df) > 0:
        mts = metrics_df["ts_ms"].to_numpy().astype(np.int64)
        if "oi" in metrics_df.columns:
            oi_raw = metrics_df["oi"].to_numpy().astype(np.float64)
            oi = _asof_align(timestamps_ms, mts, oi_raw)
            log_oi = np.log(np.maximum(oi, 1e-10))
            F[:,4] = _causal_zscore(np.nan_to_num(log_oi, nan=0.0), 7*1440)
            d1h  = np.zeros(n); d4h = np.zeros(n)
            d1h[60:]  = np.where(oi[:-60]>0,  (oi[60:]-oi[:-60])/oi[:-60],   0.0)
            d4h[240:] = np.where(oi[:-240]>0, (oi[240:]-oi[:-240])/oi[:-240], 0.0)
            F[:,5] = np.nan_to_num(np.clip(d1h,-0.5,0.5)*4, nan=0.0)
            F[:,6] = np.nan_to_num(np.clip(d4h,-0.5,0.5)*4, nan=0.0)
            F[:,7] = _causal_zscore(np.nan_to_num(oi, nan=0.0), 7*1440)
            pr4h = np.zeros(n)
            pr4h[240:] = np.sign(close[240:]-close[:-240])
            F[:,8] = np.nan_to_num(np.sign(d4h)*pr4h, nan=0.0)

        if "lsr_top" in metrics_df.columns and "lsr_global" in metrics_df.columns:
            lt = _asof_align(timestamps_ms, mts, metrics_df["lsr_top"].to_numpy().astype(np.float64))
            lg = _asof_align(timestamps_ms, mts, metrics_df["lsr_global"].to_numpy().astype(np.float64))
            diff = np.nan_to_num(lt-lg, nan=0.0)
            F[:,9]  = np.clip(diff,-2,2)/2
            F[:,10] = _causal_zscore(np.nan_to_num(lt, nan=0.0), 1440)

        if "taker_ratio" in metrics_df.columns:
            tr = _asof_align(timestamps_ms, mts, metrics_df["taker_ratio"].to_numpy().astype(np.float64))
            F[:,11] = _causal_zscore(np.nan_to_num(tr, nan=0.0), 1440)

    return F

def build_v5_cross_sectional(symbol, timestamps_ms, close, btc_close, basket_closes):
    n   = len(timestamps_ms)
    F   = np.zeros((n, 6), dtype=np.float32)
    aligned = {}
    for sym, (bts, bcl) in basket_closes.items():
        if sym == symbol: continue
        aligned[sym] = np.interp(timestamps_ms.astype(np.float64), bts.astype(np.float64), bcl.astype(np.float64))
    aligned[symbol] = close.astype(np.float64)
    syms = list(aligned.keys())
    if len(syms) < 2: return F

    def _coin_ret(cl, w):
        r = np.zeros(len(cl))
        r[w:] = np.log(np.maximum(cl[w:],1e-12)/np.maximum(cl[:-w],1e-12))
        return r

    ret1h  = np.vstack([_coin_ret(aligned[s], 60)  for s in syms])
    ret4h  = np.vstack([_coin_ret(aligned[s], 240) for s in syms])
    rv1h_  = np.vstack([_rolling_rv(aligned[s].astype(np.float64), 60) for s in syms])

    self_idx = syms.index(symbol)
    for t in range(60, n):
        r1 = ret1h[:, t]; r4 = ret4h[:, t]; rv = rv1h_[:, t]
        F[t, 0] = float(np.mean(r1 <= r1[self_idx]))
        F[t, 1] = float(np.mean(r4 <= r4[self_idx]))
        F[t, 2] = float(np.mean(rv <= rv[self_idx]))

    med4h = np.median(ret4h, axis=0)
    self4h = ret4h[self_idx]
    F[:,3] = np.nan_to_num(np.clip(self4h - med4h, -0.2, 0.2)/0.2, nan=0.0)
    F[:,4] = np.nan_to_num(_causal_zscore(med4h, 1440), nan=0.0)

    if btc_close is not None:
        btc4h = _coin_ret(btc_close.astype(np.float64), 240)
        F[:,5] = np.nan_to_num(np.clip(self4h-btc4h,-0.2,0.2)/0.2, nan=0.0)

    return F

def build_v5_full(symbol, F_v2, close, high, low, volume, ofi, taker_buy_volume, timestamps_ms, atr_arr, btc_close, funding_df, metrics_df, basket_closes):
    F_v4  = build_v4_from_components(F_v2, close, high, low, volume, ofi, taker_buy_volume, timestamps_ms, atr_arr, btc_close)
    F_perp = build_v5_perp_features(timestamps_ms, close, funding_df, metrics_df)
    F_xs   = build_v5_cross_sectional(symbol, timestamps_ms, close, btc_close, basket_closes)
    return np.hstack([F_v4, F_perp, F_xs]).astype(np.float32)

def coin_onehot(symbol, n):
    block = np.zeros((n, len(COIN_ONEHOT_NAMES)), dtype=np.float32)
    if symbol in COIN_ONEHOT_NAMES:
        block[:, COIN_ONEHOT_NAMES.index(symbol)] = 1.0
    return block


# ─────────────────────────────────────────────────────────────────────────────
# 5. LABELLING ENGINE (PHASE 2C)
# ─────────────────────────────────────────────────────────────────────────────

_ATR_SCALE = {60: math.sqrt(60/14), 240: math.sqrt(240/14), 720: math.sqrt(720/14)}

def horizon_atr_scale(h):
    return _ATR_SCALE.get(int(h), math.sqrt(h/14))

def regime_tp_mult(hurst):
    if hurst is None or not math.isfinite(hurst): return 2.0
    if hurst > 0.55: return 2.5
    if hurst < 0.45: return 1.0
    return 2.0

def label_multi_horizon(close, high, low, atr_arr, hurst_arr, sample_idx):
    n   = len(close); k = len(sample_idx)
    INF = np.iinfo(np.int32).max
    cost = ROUND_TRIP_COST
    out = {"any_valid": np.zeros(k, dtype=bool)}
    for h in HORIZONS:
        for key in [f"y_{h}", f"net_{h}", f"y_short_{h}", f"net_short_{h}", f"ret_{h}"]:
            out[key] = np.full(k, np.nan, dtype=np.float32)
    log_c = np.log(np.maximum(close, 1e-12))

    for kk, i in enumerate(sample_idx):
        atr = atr_arr[i]
        if not np.isfinite(atr) or atr <= 0: continue
        entry = close[i]
        if entry <= 0: continue
        h_val = float(hurst_arr[i]) if np.isfinite(hurst_arr[i]) else 0.5
        any_v = False

        for h in HORIZONS:
            end = min(i+h, n-1)
            if end <= i: continue
            scale = horizon_atr_scale(h)
            tp_mult = regime_tp_mult(h_val) * scale
            sl_mult_ = SL_MULT * scale
            tp_p = tp_mult * atr / entry
            sl_p = sl_mult_ * atr / entry
            out[f"ret_{h}"][kk] = float(log_c[end] - log_c[i])

            if tp_p < MIN_TP_COST_R * cost: continue
            any_v = True

            fhi = high[i+1:end+1]; flo = low[i+1:end+1]

            # Long
            tpl = entry + tp_mult*atr; sll = entry - sl_mult_*atr
            th = np.where(fhi >= tpl)[0]; sh = np.where(flo <= sll)[0]
            tt = th[0] if len(th) else INF; st = sh[0] if len(sh) else INF
            if tt < st: gross_l = tp_p
            elif st < tt: gross_l = -sl_p
            elif tt == st != INF: gross_l = -sl_p
            else: gross_l = (close[end]-entry)/entry
            nl = gross_l - cost
            out[f"y_{h}"][kk] = 1.0 if nl > 0 else 0.0
            out[f"net_{h}"][kk] = nl

            # Short
            tps = entry - tp_mult*atr; sls = entry + sl_mult_*atr
            th2 = np.where(flo <= tps)[0]; sh2 = np.where(fhi >= sls)[0]
            tt2 = th2[0] if len(th2) else INF; st2 = sh2[0] if len(sh2) else INF
            if tt2 < st2: gross_s = tp_p
            elif st2 < tt2: gross_s = -sl_p
            elif tt2 == st2 != INF: gross_s = -sl_p
            else: gross_s = (entry-close[end])/entry
            ns = gross_s - cost
            out[f"y_short_{h}"][kk] = 1.0 if ns > 0 else 0.0
            out[f"net_short_{h}"][kk] = ns

        out["any_valid"][kk] = any_v
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 5b. V6 FEATURE BUILDER (self-contained inline version)
# Implements the 36 V6 features: Donchian, Anchored VWAP, Volume Profile VAH,
# Liquidity Sweeps, SMA Trends, Order Flow extras, Fibonacci, Bollinger extras.
# ─────────────────────────────────────────────────────────────────────────────

def _sliding_max_v6(arr, w):
    n = len(arr); out = np.full(n, np.nan)
    try:
        from numpy.lib.stride_tricks import sliding_window_view
        out[w-1:] = sliding_window_view(arr, w).max(axis=1)
    except Exception:
        for i in range(w-1, n): out[i] = arr[i-w+1:i+1].max()
    return out

def _sliding_min_v6(arr, w):
    n = len(arr); out = np.full(n, np.nan)
    try:
        from numpy.lib.stride_tricks import sliding_window_view
        out[w-1:] = sliding_window_view(arr, w).min(axis=1)
    except Exception:
        for i in range(w-1, n): out[i] = arr[i-w+1:i+1].min()
    return out

def _avwap_v6(close, volume, ts_ms, period='day'):
    n = len(close); secs = (ts_ms / 1000).astype(np.int64)
    period_id = (secs // 86400 if period == 'day'
                 else (secs + 3*86400) // (7*86400) if period == 'week'
                 else secs // (30*86400))
    boundaries = np.zeros(n, dtype=bool); boundaries[0] = True
    boundaries[1:] = period_id[1:] != period_id[:-1]
    pv = close.astype(np.float64) * volume.astype(np.float64)
    cpv = np.cumsum(pv); cvol = np.cumsum(volume.astype(np.float64))
    apv = np.zeros(n, dtype=np.float64); avol = np.zeros(n, dtype=np.float64)
    for j, bi in enumerate(np.where(boundaries)[0]):
        end = np.where(boundaries)[0][j+1] if j+1 < boundaries.sum() else n
        p = cpv[bi-1] if bi > 0 else 0.0; v_ = cvol[bi-1] if bi > 0 else 0.0
        apv[bi:end] = p; avol[bi:end] = v_
    return ((cpv - apv) / np.maximum(cvol - avol, 1e-9)).astype(np.float32)

def _roll_mean_v6(x, w):
    out = np.full(len(x), np.nan)
    cs = np.cumsum(np.nan_to_num(x, nan=0.0))
    out[w-1:] = (cs[w-1:] - np.concatenate([[0], cs[:-w]])) / w
    return out

def _roll_std_v6(x, w):
    m = _roll_mean_v6(x, w); m2 = _roll_mean_v6(x**2, w)
    return np.sqrt(np.maximum(m2 - m**2, 0))

def _causal_z_v6(x, w):
    m = _roll_mean_v6(x, w); s = _roll_std_v6(x, w)
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(s > 0, (x - m) / s, 0.0)

def _vpvr_va_v6(volume, close, window=1440, stride=30, n_bins=48, va_pct=0.70):
    n = len(close)
    poc_d = np.zeros(n, np.float32); vah_d = poc_d.copy()
    val_d = poc_d.copy(); aw = poc_d.copy()
    lp = lv = lva = law = 0.0
    for i in range(window, n):
        poc_d[i]=lp; vah_d[i]=lv; val_d[i]=lva; aw[i]=law
        if (i-window) % stride != 0: continue
        clw=close[i-window:i]; vw=volume[i-window:i]
        lo=float(clw.min()); hi=float(clw.max()); cur=float(close[i])
        if hi<=lo or cur<=0: continue
        edges=np.linspace(lo,hi,n_bins+1)
        bins=np.clip(np.searchsorted(edges,clw,'right')-1,0,n_bins-1)
        bv=np.bincount(bins,weights=vw,minlength=n_bins).astype(np.float64)
        tot=bv.sum()
        if tot<=0: continue
        pi=int(np.argmax(bv)); pp=(edges[pi]+edges[pi+1])/2
        lp=float(np.clip((cur-pp)/cur*20,-1,1))
        tgt=tot*va_pct; vi=pi; vai=pi; cap=bv[pi]
        while cap<tgt:
            up=bv[vi+1] if vi+1<n_bins else 0.0
            dn=bv[vai-1] if vai-1>=0 else 0.0
            if up<=0 and dn<=0: break
            if up>=dn and vi+1<n_bins: vi+=1; cap+=up
            elif vai-1>=0: vai-=1; cap+=dn
            else:
                if vi+1<n_bins: vi+=1; cap+=up
                else: break
        vah_p=(edges[vi]+edges[vi+1])/2; val_p=(edges[vai]+edges[vai+1])/2
        lv=float(np.clip((cur-vah_p)/cur*20,-1,1))
        lva=float(np.clip((cur-val_p)/cur*20,-1,1))
        law=float(np.clip((vah_p-val_p)/max(cur,1e-10)*10,0,1))*2-1
        poc_d[i]=lp; vah_d[i]=lv; val_d[i]=lva; aw[i]=law
    return poc_d, vah_d, val_d, aw

def _build_v6_inline(close, high, low, volume, ofi, tbv, ts_ms, atr_arr):
    """Build all 36 V6 features inline (self-contained, no external imports).
    Returns float32 array of shape (n, 36). All features causal, clipped to [-1,1].
    """
    n = len(close)
    cl = close.astype(np.float64); hi = high.astype(np.float64); lo = low.astype(np.float64)
    vo = volume.astype(np.float64); tbv64 = tbv.astype(np.float64)
    ofi64 = ofi.astype(np.float64); atr64 = atr_arr.astype(np.float64)
    ts64 = ts_ms.astype(np.int64)

    F = np.zeros((n, 36), dtype=np.float32); c = 0

    # ── 1. Donchian (6) ──────────────────────────────────────────────────────
    don_u = _sliding_max_v6(hi, 20); don_l = _sliding_min_v6(lo, 20)
    don_r = np.maximum(don_u - don_l, 1e-10); don_m = (don_u + don_l) / 2
    F[:,c] = np.nan_to_num(np.clip((cl-don_l)/don_r*2-1,-1,1)); c+=1
    F[:,c] = np.nan_to_num(np.clip((cl-don_u)/np.maximum(cl,1e-10)*20,-1,1)); c+=1
    F[:,c] = np.nan_to_num(np.clip((don_l-cl)/np.maximum(cl,1e-10)*20,-1,1)); c+=1
    F[:,c] = np.nan_to_num(np.clip((cl-don_m)/np.maximum(don_m,1e-10)*10,-1,1)); c+=1
    F[:,c] = np.clip(_causal_z_v6(don_r,1440)/3,-1,1); c+=1
    bk = np.zeros(n)
    if n>20: bk[20:]=np.where(cl[20:]>don_u[19:-1],1.,np.where(cl[20:]<don_l[19:-1],-1.,0.))
    F[:,c] = bk; c+=1

    # ── 2. Anchored VWAP (3) ─────────────────────────────────────────────────
    for p in ['day','week','month']:
        vw = _avwap_v6(cl, vo, ts64, p)
        F[:,c] = np.nan_to_num(np.clip((cl-vw)/np.maximum(vw,1e-10)*20,-1,1)); c+=1

    # ── 3. Volume Profile VAH/VAL (4) ────────────────────────────────────────
    pd_, vhd_, vld_, aw_ = _vpvr_va_v6(vo, cl, 1440, 30, 48)
    F[:,c]=np.clip(pd_,-1,1); c+=1; F[:,c]=np.clip(vhd_,-1,1); c+=1
    F[:,c]=np.clip(vld_,-1,1); c+=1; F[:,c]=np.clip(aw_,-1,1); c+=1

    # ── 4. Liquidity Sweeps (5) ──────────────────────────────────────────────
    sb=np.zeros(n,np.float32); sr=np.zeros(n,np.float32); ss=np.zeros(n,np.float32)
    if n>21:
        try:
            from numpy.lib.stride_tricks import sliding_window_view
            kh=sliding_window_view(hi[:-1],20).max(axis=1)
            kl=sliding_window_view(lo[:-1],20).min(axis=1)
            s=21; ch2=hi[s:]; cl2=lo[s:]; cc2=cl[s:]; at2=np.maximum(atr64[s:],1e-10)
            kh2=kh[:len(ch2)]; kl2=kl[:len(cl2)]
            bm=(cl2<kl2)&(cc2>kl2); rm=(ch2>kh2)&(cc2<kh2)
            sb[s:]=bm.astype(np.float32); sr[s:]=rm.astype(np.float32)
            ss[s:]=np.clip(np.where(rm,(ch2-kh2)/at2,0)+np.where(bm,(kl2-cl2)/at2,0),0,3)/3
        except Exception:
            for i in range(21,n):
                kh_=hi[i-20:i].max(); kl_=lo[i-20:i].min(); at_=max(atr64[i],1e-10)
                if hi[i]>kh_ and cl[i]<kh_: sr[i]=1.; ss[i]=min((hi[i]-kh_)/at_,1.)
                elif lo[i]<kl_ and cl[i]>kl_: sb[i]=1.; ss[i]=min((kl_-lo[i])/at_,1.)
    F[:,c]=sb*2-1; c+=1; F[:,c]=sr*2-1; c+=1
    F[:,c]=np.clip(_causal_z_v6(_roll_mean_v6(sb,60)*60,1440)/3,-1,1); c+=1
    F[:,c]=np.clip(_causal_z_v6(_roll_mean_v6(sr,60)*60,1440)/3,-1,1); c+=1
    F[:,c]=(ss*2-1).astype(np.float32); c+=1

    # ── 5. SMA Trends (6) ────────────────────────────────────────────────────
    s20=_roll_mean_v6(cl,20); s50=_roll_mean_v6(cl,50); s200=_roll_mean_v6(cl,200)
    F[:,c]=np.nan_to_num(np.clip((cl-s20)/np.maximum(s20,1e-10)*20,-1,1)); c+=1
    F[:,c]=np.nan_to_num(np.clip((cl-s50)/np.maximum(s50,1e-10)*10,-1,1)); c+=1
    F[:,c]=np.nan_to_num(np.clip((cl-s200)/np.maximum(s200,1e-10)*5,-1,1)); c+=1
    a20=cl>s20; a50=cl>s50; g=s20>s50
    F[:,c]=np.nan_to_num(np.where(g&a50,1.,np.where(~g&~a50,-1.,0.))); c+=1
    F[:,c]=np.nan_to_num((s50>s200).astype(np.float32)*2-1); c+=1
    F[:,c]=np.nan_to_num(np.where((cl>s20)&(s20>s50)&(s50>s200),1.,np.where((cl<s20)&(s20<s50)&(s50<s200),-1.,0.))); c+=1

    # ── 6. Order Flow Extras (4) ──────────────────────────────────────────────
    ts_ = np.maximum(vo - tbv64, 0.)
    avg60 = _roll_mean_v6(vo, 60); thr_ = 2.*avg60
    lb = np.where(tbv64>thr_, tbv64, 0.); ls = np.where(ts_>thr_, ts_, 0.)
    lsum = _roll_mean_v6(lb+ls,60)*60; lbsum = _roll_mean_v6(lb,60)*60
    with np.errstate(divide='ignore',invalid='ignore'):
        lbr = np.where(lsum>1e-9, lbsum/lsum, 0.5)
    F[:,c]=np.nan_to_num(np.clip((lbr-0.5)*2,-1,1)); c+=1
    cvd=np.cumsum(np.nan_to_num(ofi64))
    d15=np.zeros(n); d15[15:]=cvd[15:]-cvd[:-15]
    d60=np.zeros(n); d60[60:]=cvd[60:]-cvd[:-60]
    F[:,c]=np.clip(_causal_z_v6(d15-d60/4.,1440)/3,-1,1); c+=1
    tbr=tbv64/np.maximum(vo,1e-9)
    F[:,c]=np.clip(_causal_z_v6(tbr,240)/3,-1,1); c+=1
    tbs30=np.zeros(n)
    if n>30: tbs30[30:]=tbr[:-30]
    F[:,c]=np.clip((tbr-tbs30)*5,-1,1); c+=1

    # ── 7. Fibonacci (5) ──────────────────────────────────────────────────────
    shi=_sliding_max_v6(hi,100); slo=_sliding_min_v6(lo,100)
    sr_=np.maximum(shi-slo,1e-10)
    for ratio in [0.382,0.500,0.618,0.786]:
        lv_=shi-ratio*sr_
        F[:,c]=np.nan_to_num(np.clip((cl-lv_)/np.maximum(cl,1e-10)*10,-1,1)); c+=1
    fib_lvls=np.stack([shi-r*sr_ for r in [0.382,0.5,0.618,0.786]],axis=1)
    mfd=np.nanmin(np.abs(cl[:,None]-fib_lvls)/np.maximum(cl[:,None],1e-10),axis=1)
    F[:,c]=np.nan_to_num(np.clip(1.-mfd*20,-1,1)); c+=1

    # ── 8. Bollinger Extras (3) ───────────────────────────────────────────────
    bw=4.*_roll_std_v6(cl,20)
    bwz=_causal_z_v6(bw,100)
    F[:,c]=np.clip(bwz/3,-1,1); c+=1
    F[:,c]=(bwz<-0.84).astype(np.float32)*2-1; c+=1
    F[:,c]=(bwz>0.84).astype(np.float32)*2-1; c+=1

    assert c == 36, f"V6 inline: expected 36 features, built {c}"
    return F


# ─────────────────────────────────────────────────────────────────────────────
# 6. PIPELINE ORCHESTRATION
# ─────────────────────────────────────────────────────────────────────────────

def load_coin_klines(symbol):
    files = sorted(glob.glob(os.path.join(KLINES_DIR, f"{symbol}_1m_*.parquet")))
    if not files: return None
    dfs = []
    for f in files:
        try: dfs.append(pl.read_parquet(f))
        except Exception: pass
    if not dfs: return None
    df = pl.concat(dfs)
    if df["timestamp"].dtype == pl.Datetime or df["timestamp"].dtype == pl.Date:
        df = df.with_columns(pl.col("timestamp").cast(pl.Datetime("ms")).dt.cast_time_unit("ms").cast(pl.Int64).alias("ts_ms"))
    else:
        df = df.with_columns(pl.col("timestamp").alias("ts_ms"))
    return df.sort("ts_ms")

def load_perp(symbol):
    fp = os.path.join(PERP_DIR, f"{symbol}_funding.parquet")
    mp = os.path.join(PERP_DIR, f"{symbol}_metrics.parquet")
    fd = pl.read_parquet(fp).sort("ts_ms") if os.path.exists(fp) else None
    md = pl.read_parquet(mp).sort("ts_ms") if os.path.exists(mp) else None
    return fd, md

def build_basket_cache():
    print("Building basket close cache...")
    cache = {}
    for sym in COINS:
        df = load_coin_klines(sym)
        if df is None: continue
        ts = df["ts_ms"].to_numpy().astype(np.int64)
        cl = df["close"].to_numpy().astype(np.float64)
        cache[sym] = (ts, cl)
        print(f"  {sym}: {len(ts):,} bars cached")
    print()
    return cache

def process_coin(symbol, basket_cache):
    print(f"\n==================== Processing {symbol} ====================")
    t0 = time.time()
    out_path = os.path.join(OUT_DIR, f"{symbol}_training_data.parquet")
    if os.path.exists(out_path):
        print(f"  SKIP — {out_path} already exists")
        return

    df = load_coin_klines(symbol)
    if df is None:
        print("  SKIP — no klines found"); return
    n = len(df)
    print(f"  Klines: {n:,} bars")

    close  = df["close"].to_numpy().astype(np.float64)
    high   = df["high"].to_numpy().astype(np.float64)
    low    = df["low"].to_numpy().astype(np.float64)
    volume = df["volume"].to_numpy().astype(np.float64)
    ts_ms  = df["ts_ms"].to_numpy().astype(np.float64)
    ts_int = ts_ms.astype(np.int64)

    tbv = df["taker_buy_volume"].to_numpy().astype(np.float64) if "taker_buy_volume" in df.columns else volume * 0.5
    ofi = (2.0 * tbv - volume).astype(np.float64)
    del df; gc.collect()

    btc_close = None
    if symbol != "BTCUSDT" and "BTCUSDT" in basket_cache:
        btc_ts, btc_cl = basket_cache["BTCUSDT"]
        btc_close = np.interp(ts_ms, btc_ts.astype(np.float64), btc_cl)

    funding_df, metrics_df = load_perp(symbol)

    print("  Computing V2 & V5 Features...")
    F_v2, atr_arr = build_features_v2(close, high, low, volume, ofi, ts_ms, btc_close)
    F_v5 = build_v5_full(symbol, F_v2, close, high, low, volume, ofi, tbv, ts_ms, atr_arr, btc_close, funding_df, metrics_df, basket_cache)

    print("  Computing V6 Features (Donchian, AVWAP, VP, Sweeps, SMA, OF, Fib, BB)...")
    F_v6 = _build_v6_inline(close, high, low, volume, ofi, tbv, ts_ms, atr_arr)

    # Re-extract hurst from first pass
    hurst_raw = F_v2[:, 10].copy()
    del F_v2; gc.collect()

    oh = coin_onehot(symbol, n)
    X = np.hstack([F_v5, oh, F_v6]).astype(np.float32)   # (n, 94 + 36) = (n, 130)
    del F_v5, oh, F_v6; gc.collect()

    max_h = max(HORIZONS)
    idx = np.arange(WARMUP_BARS, n - max_h, SAMPLE_EVERY)
    X_samp = X[idx]
    valid = ~np.any(np.isnan(X_samp), axis=1)
    idx_v = idx[valid]; X_v = X_samp[valid]
    del X_samp; gc.collect()

    if len(idx_v) == 0:
        print("  SKIP — no valid samples"); return

    print(f"  Labelling {len(idx_v):,} samples...")
    labels = label_multi_horizon(close, high, low, atr_arr, hurst_raw, idx_v)
    keep = labels["any_valid"]

    if keep.sum() == 0:
        print("  SKIP — all labels rejected"); return

    X_out  = X_v[keep]
    ts_out = ts_int[idx_v[keep]]

    feat_cols = {f"X_{i}": X_out[:, i] for i in range(X_out.shape[1])}
    label_cols = {}
    for h in HORIZONS:
        for k in [f"y_{h}", f"net_{h}", f"y_short_{h}", f"net_short_{h}", f"ret_{h}"]:
            label_cols[k] = labels[k][keep]
    label_cols["any_valid"] = labels["any_valid"][keep].astype(np.float32)

    all_cols = {"timestamp_ms": ts_out, **feat_cols, **label_cols}
    out_df = pl.DataFrame({k: v.tolist() for k, v in all_cols.items()})
    out_df.write_parquet(out_path, compression="zstd", compression_level=3)
    print(f"  SUCCESS Saved {out_df.shape} -> {out_path} in {(time.time()-t0)/60:.1f}m")

    del X, X_v, X_out, close, high, low, volume, ofi, tbv, atr_arr, hurst_raw, labels, out_df, all_cols
    gc.collect()


# ─────────────────────────────────────────────────────────────────────────────
# 7. MAIN RUNNER
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    t_start = time.time()
    print("==================================================================")
    print("             UNIFIED CRYPTO ORACLE V5 DATA PIPELINE")
    print("==================================================================")
    print(f"Start Time (UTC) : {datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Data Directory   : {ROOT}")
    print()

    # Step 1: Download All Raw Data
    print("━" * 50)
    print(" STEP 1: Downloading 1m OHLCV Klines")
    print("━" * 50)
    download_all_klines()

    print("\n" + "━" * 50)
    print(" STEP 2: Downloading Perp Microstructure Data")
    print("━" * 50)
    download_all_perp()

    # Step 2: Build Cache and Compute Features
    print("\n" + "━" * 50)
    print(" STEP 3: Building Cache & Computing Features + Labels")
    print("━" * 50)
    basket_cache = build_basket_cache()

    for i, symbol in enumerate(COINS, 1):
        print(f"[{i}/{len(COINS)}] Starting...")
        try:
            process_coin(symbol, basket_cache)
        except Exception as e:
            import traceback
            print(f"  CRITICAL ERROR processing {symbol}: {e}")
            traceback.print_exc()
        gc.collect()

    # Final Validation Report
    print("\n" + "━" * 50)
    print(" PIPELINE PROCESS COMPLETE SUMMARY")
    print("━" * 50)
    total_samples = 0
    for symbol in COINS:
        fp = os.path.join(OUT_DIR, f"{symbol}_training_data.parquet")
        if os.path.exists(fp):
            try:
                cnt = pl.scan_parquet(fp).select(pl.count()).collect().item()
                total_samples += cnt
                print(f"  {symbol:<14} : {cnt:>8,} samples ready")
            except Exception:
                print(f"  {symbol:<14} : [Error reading output]")
        else:
            print(f"  {symbol:<14} : MISSING")

    print()
    print(f"  Total Processed Samples : {total_samples:,}")
    print(f"  Total Run Time          : {(time.time()-t_start)/60:.1f} minutes")
    print("==================================================================")
