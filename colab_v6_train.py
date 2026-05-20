"""
╔═══════════════════════════════════════════════════════════════════╗
║  CRYPTO ORACLE V6 — GOOGLE COLAB TRAINING SCRIPT                 ║
║                                                                   ║
║  USAGE — paste this into a Colab cell and run:                   ║
║                                                                   ║
║    !git clone https://github.com/haitham-kh/crypto-oracle-mcp    ║
║    %cd crypto-oracle-mcp                                         ║
║    !python colab_v6_train.py                                      ║
║                                                                   ║
║  That's it. The script:                                          ║
║    1. Reads training data from training_data/ (already in repo)  ║
║    2. Downloads 24m Binance klines for V6 features (free)        ║
║    3. Trains V6_full + V6_micro on GPU                           ║
║    4. Saves models to MyDrive/crypto_oracle/models_v6/           ║
╚═══════════════════════════════════════════════════════════════════╝
"""

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 0 — CONFIG  (edit these if needed)
# ─────────────────────────────────────────────────────────────────────────────

# Where to read training parquets from:
#   Priority 1: training_data/ inside the cloned repo (already in GitHub — no upload needed!)
#   Priority 2: Google Drive processed folder (fallback if running without git clone)
#   Priority 3: /content/data/processed (last resort local)
REPO_TRAINING_DIR   = "/content/crypto-oracle-mcp/training_data"  # after git clone
DRIVE_PROCESSED_DIR = "/content/drive/MyDrive/crypto_oracle/processed"
DRIVE_MODELS_DIR    = "/content/drive/MyDrive/crypto_oracle/models_v6"
LOCAL_DATA_DIR      = "/content/data"
MONTHS_HISTORY      = 24        # months of 1m klines to download
SKIP_MICRO_MODEL    = False     # set True to save ~40% time (micro gate disabled)
HORIZONS            = [60, 720] # 1h and 12h horizons
ROUND_TRIP_COST     = 0.0022    # 0.22% round-trip (fees + slippage)
MIN_EV_PCT          = 0.0010    # 0.10% minimum expected net return threshold
MIN_P_UP_DEFAULT    = 0.56      # default probability threshold if search fails

COINS = [
    "AAVEUSDT", "ADAUSDT",  "APTUSDT",  "ATOMUSDT", "AVAXUSDT",
    "BNBUSDT",  "BTCUSDT",  "DOGEUSDT", "DOTUSDT",  "ETHUSDT",
    "FILUSDT",  "INJUSDT",  "LINKUSDT", "LTCUSDT",  "NEARUSDT",
    "OPUSDT",   "SOLUSDT",  "SUIUSDT",  "UNIUSDT",  "XRPUSDT",
]

COIN_ONEHOT_NAMES = [
    "AAVEUSDT", "ADAUSDT", "AVAXUSDT", "BNBUSDT",  "BTCUSDT",
    "DOGEUSDT", "ETHUSDT", "LINKUSDT", "PEPEUSDT", "SHIBUSDT",
    "SOLUSDT",  "UNIUSDT", "WIFUSDT",  "XRPUSDT",
]

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — INSTALLS
# ─────────────────────────────────────────────────────────────────────────────

import subprocess, sys
print("Installing dependencies...")
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "polars", "pyarrow", "xgboost", "scikit-learn",
                "requests", "scipy"], check=False)
print("Done.\n")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — IMPORTS
# ─────────────────────────────────────────────────────────────────────────────

import os, time, datetime, json, math, gc, glob, pickle
import requests
import numpy as np
import polars as pl
import xgboost as xgb

os.makedirs(LOCAL_DATA_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — GOOGLE DRIVE MOUNT
# ─────────────────────────────────────────────────────────────────────────────

DRIVE_AVAILABLE = False
try:
    from google.colab import drive
    drive.mount("/content/drive", force_remount=False)
    DRIVE_AVAILABLE = os.path.exists("/content/drive/MyDrive")
    print(f"Drive mounted: {DRIVE_AVAILABLE}")
except Exception:
    print("Drive mount skipped (not in Colab or already mounted).")

# Only create Drive dirs if Drive is actually mounted
if DRIVE_AVAILABLE:
    os.makedirs(DRIVE_PROCESSED_DIR, exist_ok=True)
    os.makedirs(DRIVE_MODELS_DIR,    exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — KLINE DOWNLOADER
#   Strategy (in order of priority):
#     1. data.binance.vision  — Binance's public AWS S3 bucket.
#                               Monthly + daily ZIP files. No API key.
#                               NOT geo-blocked (it's static S3, not the API).
#                               Blazing fast: one ZIP per month vs 1000s of calls.
#     2. MEXC public API      — Binance-compatible format, not geo-restricted.
#                               Used only for bars not yet on vision (today's data).
# ─────────────────────────────────────────────────────────────────────────────

import io, zipfile, csv as _csv

VISION_BASE = "https://data.binance.vision"
MEXC_BASE   = "https://api.mexc.com"
REQUEST_DELAY = 0.05   # seconds between MEXC calls (vision has no rate limit)


# Threshold: timestamps below this are in ms; at/above this are in microseconds.
# Binance Vision switched from ms to us timestamps on 2025-01-01.
_MS_THRESHOLD = 2_000_000_000_000   # 2 trillion ms ≈ year 2033 — safe cutoff


def _to_ms(ts_raw):
    """Normalise a Vision timestamp to milliseconds regardless of ms/us format."""
    ts = int(ts_raw)
    return ts // 1_000 if ts > _MS_THRESHOLD else ts


def _parse_vision_csv(raw_bytes):
    """Parse a Binance Vision CSV (bytes) into a list of bar dicts.
    Handles both the old ms format (pre-2025) and the new us format (2025+).
    """
    rows = []
    text = raw_bytes.decode("utf-8", errors="replace")
    reader = _csv.reader(text.splitlines())
    for line in reader:
        if len(line) < 10:
            continue
        try:
            rows.append({
                "timestamp_ms":     _to_ms(line[0]),   # normalised to ms
                "open":             float(line[1]),
                "high":             float(line[2]),
                "low":              float(line[3]),
                "close":            float(line[4]),
                "volume":           float(line[5]),
                "taker_buy_volume": float(line[9]),
            })
        except (ValueError, IndexError):
            continue
    return rows


def _dl_vision_zip(url, label=""):
    """Download one vision ZIP and return parsed bars, or None on 404/failure."""
    try:
        r = requests.get(url, timeout=90)
        if r.status_code == 404:
            return None   # file doesn't exist
        if r.status_code != 200:
            print(f"    vision {label}: HTTP {r.status_code}", flush=True)
            return None
        with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
            csv_name = next(n for n in zf.namelist() if n.endswith(".csv"))
            with zf.open(csv_name) as f:
                return _parse_vision_csv(f.read())
    except Exception as e:
        print(f"    vision {label}: {e}", flush=True)
        return None


def _fetch_mexc_page(symbol, start_ms, limit=1000):
    """Fetch one page from MEXC public klines (Binance-compatible). Returns list of bar dicts."""
    for attempt in range(4):
        try:
            r = requests.get(
                f"{MEXC_BASE}/api/v3/klines",
                params={"symbol": symbol, "interval": "1m",
                        "startTime": start_ms, "limit": limit},
                timeout=25,
            )
            if r.status_code == 429:
                time.sleep(30)
                continue
            if r.status_code != 200:
                time.sleep(2 ** attempt)
                continue
            data = r.json()
            if isinstance(data, dict):
                print(f"    MEXC error: {data}", flush=True)
                time.sleep(2 ** attempt)
                continue
            return [{
                "timestamp_ms":     int(row[0]),   # MEXC is always ms
                "open":             float(row[1]),
                "high":             float(row[2]),
                "low":              float(row[3]),
                "close":            float(row[4]),
                "volume":           float(row[5]),
                "taker_buy_volume": float(row[9]),
            } for row in data]
        except Exception as e:
            time.sleep(2 ** attempt)
    return []


def _test_connectivity():
    """Test both data sources. Returns (vision_ok, mexc_ok)."""
    vision_ok = False
    mexc_ok   = False

    print("  Testing data.binance.vision (S3)...", flush=True)
    try:
        r = requests.head(
            f"{VISION_BASE}/data/spot/monthly/klines/BTCUSDT/1m/BTCUSDT-1m-2024-01.zip",
            timeout=15,
        )
        if r.status_code in (200, 301, 302):
            print("  data.binance.vision: OK", flush=True)
            vision_ok = True
        else:
            print(f"  data.binance.vision: HTTP {r.status_code}", flush=True)
    except Exception as e:
        print(f"  data.binance.vision: {e}", flush=True)

    print("  Testing MEXC API...", flush=True)
    try:
        r = requests.get(
            f"{MEXC_BASE}/api/v3/klines",
            params={"symbol": "BTCUSDT", "interval": "1m", "limit": 2},
            timeout=10,
        )
        if r.status_code == 200 and isinstance(r.json(), list):
            print("  MEXC API: OK", flush=True)
            mexc_ok = True
        else:
            print(f"  MEXC API: HTTP {r.status_code}", flush=True)
    except Exception as e:
        print(f"  MEXC API: {e}", flush=True)

    return vision_ok, mexc_ok


def download_klines(symbol, start_ms, end_ms):
    """
    Download 1m bars for [start_ms, end_ms).

    Strategy:
      1. Monthly ZIPs from data.binance.vision  — fast, no geo-block.
         NOTE: monthly ZIPs for 2025+ are often empty/delayed on vision;
         the daily fallback below fills those gaps automatically.
      2. Daily ZIPs from data.binance.vision    — covers ALL missing days
         (not just the current month). This is the main data source for 2025+.
      3. MEXC API                               — fills any bars still missing
         (e.g. today's bars not yet uploaded to vision).

    Timestamps: Vision switched from ms to microseconds on 2025-01-01.
    _to_ms() normalises both formats to milliseconds before any comparison.
    """
    all_rows = []
    start_dt = datetime.datetime.utcfromtimestamp(start_ms / 1000)
    today    = datetime.datetime.utcnow()

    # ── 1. Monthly ZIPs (2024 data is reliably here) ──────────────────────────
    yr, mo = start_dt.year, start_dt.month
    while (yr, mo) < (today.year, today.month):
        fname = f"{symbol}-1m-{yr}-{mo:02d}.zip"
        url   = f"{VISION_BASE}/data/spot/monthly/klines/{symbol}/1m/{fname}"
        rows  = _dl_vision_zip(url, fname)
        if rows:
            rows = [r for r in rows if start_ms <= r["timestamp_ms"] < end_ms]
            if rows:
                all_rows.extend(rows)
                print(f"    {symbol}: monthly {yr}-{mo:02d} → {len(rows):,} bars", flush=True)
            else:
                # File exists but all timestamps filtered out — skip to daily
                print(f"    {symbol}: monthly {yr}-{mo:02d} empty after filter", flush=True)
        # None = 404 (not uploaded yet) — daily fallback will cover this
        mo += 1
        if mo > 12:
            yr += 1; mo = 1

    # ── 2. Daily ZIPs — covers ALL missing days since start ───────────────────
    # Figure out which days we still need (any gap or everything from 2025-01)
    covered_ms = set(r["timestamp_ms"] for r in all_rows)
    last_covered = (max(covered_ms) if covered_ms else start_ms - 60_000)
    next_needed_ms = last_covered + 60_000

    if next_needed_ms < end_ms:
        cur_day = datetime.datetime.utcfromtimestamp(next_needed_ms / 1000).replace(
            hour=0, minute=0, second=0, microsecond=0)
        daily_added = 0
        while cur_day.date() < today.date():
            fname = f"{symbol}-1m-{cur_day.year}-{cur_day.month:02d}-{cur_day.day:02d}.zip"
            url   = f"{VISION_BASE}/data/spot/daily/klines/{symbol}/1m/{fname}"
            rows  = _dl_vision_zip(url, fname)
            if rows:
                rows = [r for r in rows
                        if r["timestamp_ms"] not in covered_ms
                        and start_ms <= r["timestamp_ms"] < end_ms]
                if rows:
                    all_rows.extend(rows)
                    covered_ms.update(r["timestamp_ms"] for r in rows)
                    daily_added += len(rows)
            cur_day += datetime.timedelta(days=1)
        if daily_added:
            first_d = datetime.datetime.utcfromtimestamp(next_needed_ms/1000).strftime("%Y-%m-%d")
            print(f"    {symbol}: daily ZIPs → +{daily_added:,} bars (from {first_d})", flush=True)

    # ── 3. MEXC API — today's bars not yet on vision ─────────────────────────
    all_rows.sort(key=lambda r: r["timestamp_ms"])
    last_ms = (all_rows[-1]["timestamp_ms"] + 60_000) if all_rows else start_ms
    if last_ms < end_ms:
        print(f"    {symbol}: MEXC gap-fill "
              f"({datetime.datetime.utcfromtimestamp(last_ms/1000).strftime('%Y-%m-%d %H:%M')} → now)...",
              flush=True)
        cur = last_ms
        while cur < end_ms:
            page = _fetch_mexc_page(symbol, cur)
            if not page:
                break
            valid = [r for r in page if cur <= r["timestamp_ms"] < end_ms]
            if not valid:
                break
            all_rows.extend(valid)
            cur = page[-1]["timestamp_ms"] + 60_000
            time.sleep(REQUEST_DELAY)

    # De-duplicate and sort
    seen = set()
    unique = []
    for r in all_rows:
        if r["timestamp_ms"] not in seen:
            seen.add(r["timestamp_ms"])
            unique.append(r)
    unique.sort(key=lambda r: r["timestamp_ms"])
    return unique


def download_all_coins(coins, months, out_dir):
    """Download all coins sequentially using vision + MEXC."""
    end_dt        = datetime.datetime.utcnow().replace(second=0, microsecond=0)
    start_dt      = end_dt - datetime.timedelta(days=months * 30)
    start_ms      = int(start_dt.timestamp() * 1000)
    end_ms        = int(end_dt.timestamp() * 1000)
    expected_bars = (end_ms - start_ms) // 60_000

    print(f"\n{'='*65}")
    print(f"  DOWNLOADING {len(coins)} coins  |  {months}m  |  ~{expected_bars:,} bars/coin")
    print(f"  Period: {start_dt.date()} -> {end_dt.date()}")
    print(f"  Source: data.binance.vision (S3) + MEXC fallback")
    print(f"{'='*65}\n")

    os.makedirs(out_dir, exist_ok=True)
    ohlcv_paths = {}

    for sym in coins:
        out_path = os.path.join(out_dir, f"{sym}_1m_ohlcv.parquet")

        # Skip already-complete downloads
        if os.path.exists(out_path):
            try:
                df_check = pl.read_parquet(out_path)
                if len(df_check) >= expected_bars * 0.90:
                    sz = os.path.getsize(out_path) / 1e6
                    print(f"  {sym}: SKIP — cached ({len(df_check):,} bars, {sz:.0f}MB)")
                    ohlcv_paths[sym] = out_path
                    continue
                else:
                    print(f"  {sym}: incomplete ({len(df_check):,}/{expected_bars:,}) — re-downloading")
            except Exception:
                pass

        t0 = time.time()
        print(f"  {sym}: downloading...", flush=True)
        candles = download_klines(sym, start_ms, end_ms)

        if not candles:
            print(f"  {sym}: FAILED — no bars returned. Skipping.")
            continue

        df = pl.DataFrame(candles).sort("timestamp_ms")
        df = df.with_columns((2.0 * pl.col("taker_buy_volume") - pl.col("volume")).alias("ofi"))
        df.write_parquet(out_path)
        elapsed = (time.time() - t0) / 60
        sz = os.path.getsize(out_path) / 1e6
        cov = len(candles) / max(expected_bars, 1) * 100
        print(f"  {sym}: done  {len(candles):,} bars  {sz:.0f}MB  coverage={cov:.0f}%  ({elapsed:.1f} min)", flush=True)
        ohlcv_paths[sym] = out_path

    print(f"\nDownload complete: {len(ohlcv_paths)}/{len(coins)} coins ready.")
    return ohlcv_paths

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — INLINE V6 FEATURE BUILDER (self-contained)
# ─────────────────────────────────────────────────────────────────────────────

N_V6_EXTRA = 36

FEATURE_NAMES_V6_EXTRA = [
    "don_position","don_upper_dist","don_lower_dist","don_mid_dist",
    "don_width_z","don_breakout",
    "avwap_session_dist","avwap_weekly_dist","avwap_monthly_dist",
    "vp_poc_dist","vp_vah_dist","vp_val_dist","vp_area_width",
    "sweep_bull_flag","sweep_bear_flag","sweep_bull_count_z",
    "sweep_bear_count_z","sweep_strength",
    "sma20_dist","sma50_dist","sma200_dist",
    "sma20_50_signal","sma50_200_signal","sma_alignment",
    "large_trade_buy_ratio","delta_acceleration","of_imbalance_z",
    "taker_pressure_shift",
    "fib_382_dist","fib_500_dist","fib_618_dist","fib_786_dist",
    "fib_confluence",
    "bb_width_z","bb_squeeze_flag","bb_expansion_flag",
]

def _sw_max(arr, w):
    n = len(arr); out = np.full(n, np.nan)
    from numpy.lib.stride_tricks import sliding_window_view
    out[w-1:] = sliding_window_view(arr, w).max(axis=1)
    return out

def _sw_min(arr, w):
    n = len(arr); out = np.full(n, np.nan)
    from numpy.lib.stride_tricks import sliding_window_view
    out[w-1:] = sliding_window_view(arr, w).min(axis=1)
    return out

def _avwap(close, volume, ts_ms, period='day'):
    n = len(close); secs = (ts_ms / 1000).astype(np.int64)
    pid = (secs // 86400 if period=='day'
           else (secs+3*86400)//(7*86400) if period=='week'
           else secs//(30*86400))
    bd = np.zeros(n, bool); bd[0] = True; bd[1:] = pid[1:] != pid[:-1]
    pv = close * volume; cpv = np.cumsum(pv); cvol = np.cumsum(volume)
    apv = np.zeros(n); avol = np.zeros(n)
    bis = np.where(bd)[0]
    for j, bi in enumerate(bis):
        end = bis[j+1] if j+1 < len(bis) else n
        p = cpv[bi-1] if bi>0 else 0.0; v = cvol[bi-1] if bi>0 else 0.0
        apv[bi:end]=p; avol[bi:end]=v
    return ((cpv-apv)/np.maximum(cvol-avol, 1e-9)).astype(np.float32)

def _rm(x, w):
    out = np.full(len(x), np.nan)
    cs = np.cumsum(np.nan_to_num(x))
    out[w-1:] = (cs[w-1:] - np.concatenate([[0], cs[:-w]])) / w
    return out

def _rs(x, w):
    m = _rm(x, w); m2 = _rm(x**2, w)
    return np.sqrt(np.maximum(m2 - m**2, 0))

def _rz(x, w):
    m = _rm(x, w); s = _rs(x, w)
    return np.where(s > 0, (x - m) / s, 0.0)

def _vpvr(volume, close, window=1440, stride=30, n_bins=48, va_pct=0.70):
    n = len(close)
    poc_d = np.zeros(n, np.float32); vah_d = poc_d.copy()
    val_d = poc_d.copy(); aw = poc_d.copy()
    lp = lv = lva = law = 0.0
    for i in range(window, n):
        poc_d[i]=lp; vah_d[i]=lv; val_d[i]=lva; aw[i]=law
        if (i-window)%stride!=0: continue
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
            up=bv[vi+1] if vi+1<n_bins else 0.; dn=bv[vai-1] if vai-1>=0 else 0.
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

def _atr14(high, low, close):
    n = len(close)
    tr = np.maximum(high - low, np.maximum(
        np.abs(high - np.roll(close, 1)),
        np.abs(low  - np.roll(close, 1))
    ))
    tr[0] = high[0] - low[0]
    atr = np.full(n, np.nan)
    atr[13] = tr[:14].mean()
    alpha = 1.0 / 14.0
    for i in range(14, n):
        atr[i] = atr[i-1] * (1 - alpha) + tr[i] * alpha
    return atr

def build_v6_features(close, high, low, volume, ofi, tbv, ts_ms, atr_arr):
    """36 V6 features. Returns float32 (n, 36). All causal, clipped to [-1,1]."""
    n = len(close)
    cl=close.astype(np.float64); hi=high.astype(np.float64); lo=low.astype(np.float64)
    vo=volume.astype(np.float64); tbv64=tbv.astype(np.float64)
    ofi64=ofi.astype(np.float64); atr64=np.asarray(atr_arr, np.float64)
    ts64=ts_ms.astype(np.int64)
    F=np.zeros((n,36),np.float32); c=0

    # 1. Donchian (6)
    du=_sw_max(hi,20); dl=_sw_min(lo,20); dr=np.maximum(du-dl,1e-10); dm=(du+dl)/2
    F[:,c]=np.nan_to_num(np.clip((cl-dl)/dr*2-1,-1,1)); c+=1
    F[:,c]=np.nan_to_num(np.clip((cl-du)/np.maximum(cl,1e-10)*20,-1,1)); c+=1
    F[:,c]=np.nan_to_num(np.clip((dl-cl)/np.maximum(cl,1e-10)*20,-1,1)); c+=1
    F[:,c]=np.nan_to_num(np.clip((cl-dm)/np.maximum(dm,1e-10)*10,-1,1)); c+=1
    F[:,c]=np.clip(_rz(dr,1440)/3,-1,1); c+=1
    bk=np.zeros(n)
    if n>20: bk[20:]=np.where(cl[20:]>du[19:-1],1.,np.where(cl[20:]<dl[19:-1],-1.,0.))
    F[:,c]=bk; c+=1

    # 2. Anchored VWAP (3)
    for p in ['day','week','month']:
        vw=_avwap(cl,vo,ts64,p)
        F[:,c]=np.nan_to_num(np.clip((cl-vw)/np.maximum(vw,1e-10)*20,-1,1)); c+=1

    # 3. Volume Profile (4)
    pd_,vhd_,vld_,aw_=_vpvr(vo,cl,1440,30,48)
    F[:,c]=np.clip(pd_,-1,1); c+=1; F[:,c]=np.clip(vhd_,-1,1); c+=1
    F[:,c]=np.clip(vld_,-1,1); c+=1; F[:,c]=np.clip(aw_,-1,1); c+=1

    # 4. Liquidity Sweeps (5)
    sb=np.zeros(n,np.float32); sr=np.zeros(n,np.float32); ss=np.zeros(n,np.float32)
    if n>21:
        try:
            from numpy.lib.stride_tricks import sliding_window_view
            kh=sliding_window_view(hi[:-1],20).max(axis=1)
            kl=sliding_window_view(lo[:-1],20).min(axis=1)
            s=21; at2=np.maximum(atr64[s:],1e-10)
            kh2=kh[:n-s]; kl2=kl[:n-s]
            bm=(lo[s:]<kl2)&(cl[s:]>kl2); rm=(hi[s:]>kh2)&(cl[s:]<kh2)
            sb[s:]=bm.astype(np.float32); sr[s:]=rm.astype(np.float32)
            ss[s:]=np.clip(np.where(rm,(hi[s:]-kh2)/at2,0)+np.where(bm,(kl2-lo[s:])/at2,0),0,3)/3
        except Exception:
            for i in range(21,n):
                kh_=hi[i-20:i].max(); kl_=lo[i-20:i].min(); at_=max(atr64[i],1e-10)
                if hi[i]>kh_ and cl[i]<kh_: sr[i]=1.; ss[i]=min((hi[i]-kh_)/at_,1.)
                elif lo[i]<kl_ and cl[i]>kl_: sb[i]=1.; ss[i]=min((kl_-lo[i])/at_,1.)
    F[:,c]=sb*2-1; c+=1; F[:,c]=sr*2-1; c+=1
    F[:,c]=np.clip(_rz(_rm(sb,60)*60,1440)/3,-1,1); c+=1
    F[:,c]=np.clip(_rz(_rm(sr,60)*60,1440)/3,-1,1); c+=1
    F[:,c]=(ss*2-1).astype(np.float32); c+=1

    # 5. SMA Trends (6)
    s20=_rm(cl,20); s50=_rm(cl,50); s200=_rm(cl,200)
    F[:,c]=np.nan_to_num(np.clip((cl-s20)/np.maximum(s20,1e-10)*20,-1,1)); c+=1
    F[:,c]=np.nan_to_num(np.clip((cl-s50)/np.maximum(s50,1e-10)*10,-1,1)); c+=1
    F[:,c]=np.nan_to_num(np.clip((cl-s200)/np.maximum(s200,1e-10)*5,-1,1)); c+=1
    g=s20>s50; a50=cl>s50
    F[:,c]=np.nan_to_num(np.where(g&a50,1.,np.where(~g&~a50,-1.,0.))); c+=1
    F[:,c]=np.nan_to_num((s50>s200).astype(np.float32)*2-1); c+=1
    F[:,c]=np.nan_to_num(np.where((cl>s20)&(s20>s50)&(s50>s200),1.,np.where((cl<s20)&(s20<s50)&(s50<s200),-1.,0.))); c+=1

    # 6. Order Flow Extras (4)
    ts_=np.maximum(vo-tbv64,0.); avg60=_rm(vo,60); thr_=2.*avg60
    lb=np.where(tbv64>thr_,tbv64,0.); ls=np.where(ts_>thr_,ts_,0.)
    lsum=_rm(lb+ls,60)*60; lbsum=_rm(lb,60)*60
    with np.errstate(divide='ignore',invalid='ignore'):
        lbr=np.where(lsum>1e-9,lbsum/lsum,0.5)
    F[:,c]=np.nan_to_num(np.clip((lbr-0.5)*2,-1,1)); c+=1
    cvd=np.cumsum(np.nan_to_num(ofi64))
    d15=np.zeros(n); d15[15:]=cvd[15:]-cvd[:-15]
    d60=np.zeros(n); d60[60:]=cvd[60:]-cvd[:-60]
    F[:,c]=np.clip(_rz(d15-d60/4.,1440)/3,-1,1); c+=1
    tbr=tbv64/np.maximum(vo,1e-9)
    F[:,c]=np.clip(_rz(tbr,240)/3,-1,1); c+=1
    tbs30=np.zeros(n)
    if n>30: tbs30[30:]=tbr[:-30]
    F[:,c]=np.clip((tbr-tbs30)*5,-1,1); c+=1

    # 7. Fibonacci (5)
    shi=_sw_max(hi,100); slo=_sw_min(lo,100); sr_=np.maximum(shi-slo,1e-10)
    for ratio in [0.382,0.500,0.618,0.786]:
        lv_=shi-ratio*sr_
        F[:,c]=np.nan_to_num(np.clip((cl-lv_)/np.maximum(cl,1e-10)*10,-1,1)); c+=1
    fib_lvls=np.stack([shi-r*sr_ for r in [0.382,0.5,0.618,0.786]],axis=1)
    fib_safe=np.where(np.isnan(fib_lvls),cl[:,None],fib_lvls)
    mfd=np.min(np.abs(cl[:,None]-fib_safe)/np.maximum(cl[:,None],1e-10),axis=1)
    F[:,c]=np.nan_to_num(np.clip(1.-mfd*20,-1,1)); c+=1

    # 8. Bollinger Extras (3)
    bw=4.*_rs(cl,20); bwz=_rz(bw,100)
    F[:,c]=np.clip(bwz/3,-1,1); c+=1
    F[:,c]=(bwz<-0.84).astype(np.float32)*2-1; c+=1
    F[:,c]=(bwz>0.84).astype(np.float32)*2-1; c+=1

    assert c==36, f"V6 count error: {c}"
    return F

def v6_signal_strength(F):
    don_bk=np.abs(F[:,5]); don_pos=np.abs(F[:,0])
    avwap_s=np.abs(F[:,6]); avwap_w=np.abs(F[:,7])
    sweep=np.maximum((F[:,13]+1)/2,(F[:,14]+1)/2)
    sma_a=np.abs(F[:,23])
    return np.clip(0.25*don_bk+0.20*don_pos+0.15*avwap_s+0.10*avwap_w+0.15*sweep+0.15*sma_a,0,1).astype(np.float32)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — DATA LOADING  (parquet -> X_v5 + labels)
# ─────────────────────────────────────────────────────────────────────────────

N_V5_FULL = 94  # 80 V5 features + 14 coin one-hot

def load_training_bundle(symbol, processed_dir, ohlcv_dir):
    """Load X_v5 + labels from training_data.parquet, add V6 from ohlcv parquet."""
    tp = os.path.join(processed_dir, f"{symbol}_training_data.parquet")
    op = os.path.join(ohlcv_dir,     f"{symbol}_1m_ohlcv.parquet")

    if not os.path.exists(tp):
        print(f"  [{symbol}] SKIP — no training_data.parquet")
        return None
    if not os.path.exists(op):
        print(f"  [{symbol}] SKIP — no _1m_ohlcv.parquet  (download failed?)")
        return None

    df = pl.read_parquet(tp)
    x_cols = sorted([c for c in df.columns if c.startswith("X_")],
                    key=lambda c: int(c.split("_")[1]))
    if len(x_cols) < N_V5_FULL:
        print(f"  [{symbol}] SKIP — only {len(x_cols)} X cols")
        return None

    X_v5 = df.select(x_cols[:N_V5_FULL]).to_numpy().astype(np.float32)
    ts   = df["timestamp_ms"].to_numpy().astype(np.int64)
    n    = len(df)

    label_data = {}
    for h in HORIZONS:
        for k in [f"y_{h}", f"net_{h}", f"y_short_{h}", f"net_short_{h}", f"ret_{h}"]:
            label_data[k] = (df[k].to_numpy().astype(np.float32)
                             if k in df.columns
                             else np.full(n, np.nan, np.float32))
    any_v = df["any_valid"].to_numpy().astype(bool) if "any_valid" in df.columns else np.ones(n, bool)

    print(f"  [{symbol}] {n:,} samples  |  Computing V6 features...", flush=True)
    odf = pl.read_parquet(op).sort("timestamp_ms")
    cl_f  = odf["close"].to_numpy().astype(np.float64)
    hi_f  = odf["high"].to_numpy().astype(np.float64)
    lo_f  = odf["low"].to_numpy().astype(np.float64)
    vo_f  = odf["volume"].to_numpy().astype(np.float64)
    tbv_f = odf["taker_buy_volume"].to_numpy().astype(np.float64)
    ofi_f = (odf["ofi"].to_numpy().astype(np.float64)
             if "ofi" in odf.columns else 2*tbv_f - vo_f)
    ts_f  = odf["timestamp_ms"].to_numpy().astype(np.int64)
    atr_f = _atr14(hi_f, lo_f, cl_f)

    F_v6_full = build_v6_features(cl_f, hi_f, lo_f, vo_f, ofi_f, tbv_f, ts_f, atr_f)

    pos = np.searchsorted(ts_f, ts, side="left")
    pos = np.clip(pos, 0, len(ts_f) - 1)
    ok  = np.abs(ts_f[pos] - ts) <= 5 * 60_000
    F_v6_samp = np.zeros((n, N_V6_EXTRA), np.float32)
    F_v6_samp[ok] = F_v6_full[pos[ok]]
    print(f"  [{symbol}] V6 match: {ok.sum()}/{n} ({ok.mean()*100:.1f}%)")

    X_v6   = np.hstack([X_v5, F_v6_samp]).astype(np.float32)
    str_v6 = v6_signal_strength(F_v6_samp)

    return {
        "symbol":     symbol,
        "X_v6_full":  X_v6,
        "X_v6_only":  F_v6_samp,
        "v6_strength":str_v6,
        "ts":         ts.astype(np.float64),
        "labels":     label_data,
        "n":          n,
    }

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — XGB TRAINING
# ─────────────────────────────────────────────────────────────────────────────

XGB_FULL_PARAMS = {
    "objective": "binary:logistic", "eval_metric": ["logloss","auc"],
    "max_depth": 7, "learning_rate": 0.030,
    "subsample": 0.80, "colsample_bytree": 0.65, "colsample_bylevel": 0.80,
    "min_child_weight": 80, "reg_alpha": 0.20, "reg_lambda": 1.5,
    "tree_method": "hist", "device": "cuda", "seed": 42, "verbosity": 1,
}
XGB_MICRO_PARAMS = {
    "objective": "binary:logistic", "eval_metric": ["logloss","auc"],
    "max_depth": 5, "learning_rate": 0.035,
    "subsample": 0.80, "colsample_bytree": 0.80,
    "min_child_weight": 60, "reg_alpha": 0.10, "reg_lambda": 1.0,
    "tree_method": "hist", "device": "cuda", "seed": 44, "verbosity": 0,
}
XGB_REG_PARAMS = {
    "objective": "reg:squarederror", "eval_metric": "rmse",
    "max_depth": 7, "learning_rate": 0.030,
    "subsample": 0.80, "colsample_bytree": 0.65, "colsample_bylevel": 0.80,
    "min_child_weight": 80, "reg_alpha": 0.20, "reg_lambda": 1.5,
    "tree_method": "hist", "device": "cuda", "seed": 43, "verbosity": 0,
}

def _check_gpu():
    try:
        xgb.train({"tree_method":"hist","device":"cuda","verbosity":0,"objective":"binary:logistic"},
                  xgb.DMatrix(np.random.randn(100,4), label=np.random.randint(0,2,100)),
                  num_boost_round=1)
        print("  GPU detected — using CUDA acceleration")
        return True
    except Exception:
        print("  No GPU — switching to CPU (tree_method=hist)")
        for d in [XGB_FULL_PARAMS, XGB_MICRO_PARAMS, XGB_REG_PARAMS]:
            d.pop("device", None)
        return False

def recency_weights(ts, anchor_ts, half_life_months=12):
    age_m = np.maximum((anchor_ts - ts) / (30.44 * 24 * 3600 * 1000), 0)
    lam = np.log(2) / half_life_months
    w = np.exp(-lam * age_m)
    return w / w.mean()

def pick_threshold(p_cal, net_pct, min_trades=100):
    best = {"threshold": MIN_P_UP_DEFAULT, "ev_pct": -1e9, "n": 0, "win_rate": 0.0, "profit_factor": 0.0}
    for thr in np.round(np.arange(0.50, 0.81, 0.01), 2):
        m = p_cal >= thr; n = int(m.sum())
        if n < min_trades: continue
        nets = net_pct[m]; ev = float(np.nanmean(nets)); wr = float((nets>0).mean())
        pf = float(nets[nets>0].sum() / max(-nets[nets<=0].sum(), 1e-9))
        if ev > best["ev_pct"]:
            best = {"threshold":float(thr),"ev_pct":ev*100,"n":n,"win_rate":wr*100,"profit_factor":pf}
    return best

def train_horizon(h, direction, X_tr, ts_tr, y_tr, net_tr, X_va, y_va, net_va,
                  X_te, y_te, net_te, ret_tr, ret_va, ret_te, v6_str_tr,
                  feat_names, anchor_ts, model_tag, models_dir):
    from sklearn.isotonic import IsotonicRegression
    print(f"\n  -- V6_{model_tag} | h={h}m | {direction.upper()} ---", flush=True)

    m_tr = ~np.isnan(y_tr); m_va = ~np.isnan(y_va); m_te = ~np.isnan(y_te)
    n_tr, n_va, n_te = m_tr.sum(), m_va.sum(), m_te.sum()
    print(f"    train={n_tr:,}  val={n_va:,}  test={n_te:,}", flush=True)
    if n_tr < 3000 or n_va < 300:
        print("    SKIPPED — insufficient data"); return None

    w_base = recency_weights(ts_tr[m_tr], anchor_ts)
    if model_tag == "full":
        boost = 1.0 + 2.33 * v6_str_tr[m_tr]
        sample_w = w_base * boost; sample_w /= sample_w.mean()
    else:
        sample_w = w_base

    params = XGB_FULL_PARAMS if model_tag == "full" else XGB_MICRO_PARAMS
    dtr = xgb.DMatrix(X_tr[m_tr], label=y_tr[m_tr], weight=sample_w, feature_names=feat_names)
    dva = xgb.DMatrix(X_va[m_va], label=y_va[m_va], feature_names=feat_names)
    dte = xgb.DMatrix(X_te[m_te], label=y_te[m_te], feature_names=feat_names)

    clf = xgb.train(params, dtr, num_boost_round=800,
                    evals=[(dtr,"train"),(dva,"val")],
                    early_stopping_rounds=40, verbose_eval=100)

    p_va_raw = clf.predict(dva)
    calib = IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds="clip")
    calib.fit(p_va_raw, y_va[m_va])
    p_va_cal = calib.predict(p_va_raw)
    p_te_cal = calib.predict(clf.predict(dte))

    best = pick_threshold(p_va_cal, net_va[m_va], min_trades=max(100, n_va//1000))
    print(f"    thr={best['threshold']:.2f}  val_EV={best['ev_pct']:+.3f}%  WR={best['win_rate']:.1f}%  PF={best['profit_factor']:.2f}", flush=True)

    m_t = p_te_cal >= best["threshold"]
    if m_t.sum() > 0:
        nets_t = net_te[m_te][m_t]
        ev_t = float(nets_t.mean())*100; wr_t = float((nets_t>0).mean())*100
        pf_t = float(nets_t[nets_t>0].sum()/max(-nets_t[nets_t<=0].sum(),1e-9))
        print(f"    OOS test: n={int(m_t.sum())}  EV={ev_t:+.3f}%  WR={wr_t:.1f}%  PF={pf_t:.2f}", flush=True)
    else:
        ev_t = wr_t = pf_t = 0.0
        print("    OOS test: no trades passed threshold")

    clf_path   = os.path.join(models_dir, f"v6_{model_tag}_clf_{direction}_h{h}.json")
    calib_path = os.path.join(models_dir, f"v6_{model_tag}_calib_{direction}_h{h}.pkl")
    clf.save_model(clf_path)
    with open(calib_path, "wb") as fp: pickle.dump(calib, fp)

    rank_ic = 0.0
    if direction == "long" and model_tag == "full":
        rm_tr = ~np.isnan(ret_tr); rm_va = ~np.isnan(ret_va)
        if rm_tr.sum() > 1000:
            wr = recency_weights(ts_tr[rm_tr], anchor_ts)
            dtr_r = xgb.DMatrix(X_tr[rm_tr], label=ret_tr[rm_tr], weight=wr, feature_names=feat_names)
            dva_r = xgb.DMatrix(X_va[rm_va], label=ret_va[rm_va], feature_names=feat_names)
            reg = xgb.train(XGB_REG_PARAMS, dtr_r, num_boost_round=800,
                            evals=[(dtr_r,"train"),(dva_r,"val")],
                            early_stopping_rounds=40, verbose_eval=False)
            reg_path = os.path.join(models_dir, f"v6_full_reg_h{h}.json")
            reg.save_model(reg_path)
            rm_te = ~np.isnan(ret_te)
            if rm_te.sum() > 0:
                from scipy.stats import spearmanr
                rank_ic, _ = spearmanr(reg.predict(xgb.DMatrix(X_te[rm_te], feature_names=feat_names)), ret_te[rm_te])
            print(f"    Regressor rank IC: {rank_ic:+.3f}")

    return {
        "horizon": int(h), "direction": direction, "model_tag": model_tag,
        "p_threshold": best["threshold"], "val_ev_pct": best["ev_pct"],
        "val_win_rate_pct": best["win_rate"], "val_profit_factor": best["profit_factor"],
        "val_n": best["n"], "test_ev_pct": ev_t, "test_win_rate_pct": wr_t,
        "test_profit_factor": pf_t, "test_n": int(m_t.sum()),
        "regressor_rank_ic": float(rank_ic),
    }

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 — MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print("\n" + "="*70)
    print("  CRYPTO ORACLE V6 — COLAB TRAINING PIPELINE")
    print(f"  {len(COINS)} coins  |  horizons={HORIZONS}  |  ROUND_TRIP={ROUND_TRIP_COST*100:.2f}%")
    print("="*70)

    _check_gpu()

    # Step 1: Check connectivity + Download OHLCV
    vision_ok, mexc_ok = _test_connectivity()
    if not vision_ok and not mexc_ok:
        print("\n[ERROR] Both data.binance.vision and MEXC API are unreachable.")
        print("  This is unusual — check your Colab internet connection.")
        return
    if not vision_ok:
        print("  [WARN] data.binance.vision unreachable — will use MEXC only (slower)", flush=True)
    ohlcv_dir   = os.path.join(LOCAL_DATA_DIR, "ohlcv")
    ohlcv_paths = download_all_coins(COINS, MONTHS_HISTORY, ohlcv_dir)

    # Step 2: Find training parquets (repo > Drive > local)
    processed_dir = None
    for candidate in [REPO_TRAINING_DIR, DRIVE_PROCESSED_DIR,
                      os.path.join(LOCAL_DATA_DIR, "processed")]:
        if os.path.exists(candidate) and glob.glob(os.path.join(candidate, "*_training_data.parquet")):
            processed_dir = candidate
            break

    if processed_dir is None:
        print("\n[ERROR] No training_data.parquet files found.")
        print(f"  Checked: {REPO_TRAINING_DIR}")
        print(f"  Checked: {DRIVE_PROCESSED_DIR}")
        print("  Make sure you ran:  !git clone https://github.com/haitham-kh/crypto-oracle-mcp.git")
        return

    tp_files = glob.glob(os.path.join(processed_dir, "*_training_data.parquet"))
    print(f"\n[DATA] {processed_dir}  ({len(tp_files)} parquets)")

    # Step 3: Load bundles
    bundles = []
    for sym in COINS:
        b = load_training_bundle(sym, processed_dir, ohlcv_dir)
        if b is not None:
            bundles.append(b)
        gc.collect()

    if not bundles:
        print("[ERROR] No bundles loaded."); return

    # Step 4: Stack all coins
    X_full  = np.vstack([b["X_v6_full"]  for b in bundles])
    X_micro = np.vstack([b["X_v6_only"]  for b in bundles])
    v6_str  = np.concatenate([b["v6_strength"] for b in bundles])
    ts_all  = np.concatenate([b["ts"] for b in bundles])
    per_h   = {}
    for h in HORIZONS:
        for k in [f"y_{h}",f"net_{h}",f"y_short_{h}",f"net_short_{h}",f"ret_{h}"]:
            per_h[k] = np.concatenate([b["labels"].get(k, np.full(b["n"],np.nan,np.float32)) for b in bundles])

    order = np.argsort(ts_all, kind="mergesort")
    X_full=X_full[order]; X_micro=X_micro[order]; v6_str=v6_str[order]; ts_all=ts_all[order]
    for k in list(per_h.keys()): per_h[k]=per_h[k][order]

    N = len(X_full)
    tr_end = int(N*0.70); va_end = int(N*0.85)
    sl_tr=slice(0,tr_end); sl_va=slice(tr_end,va_end); sl_te=slice(va_end,N)

    def _fmt(ms): return datetime.datetime.utcfromtimestamp(ms/1000).strftime("%Y-%m-%d")
    symbols_used = [b["symbol"] for b in bundles]
    print(f"\nTotal: {N:,} samples x {X_full.shape[1]} features")
    print(f"  train : {tr_end:,}  {_fmt(ts_all[0])} -> {_fmt(ts_all[tr_end-1])}")
    print(f"  val   : {va_end-tr_end:,}  {_fmt(ts_all[tr_end])} -> {_fmt(ts_all[va_end-1])}")
    print(f"  test  : {N-va_end:,}  {_fmt(ts_all[va_end])} -> {_fmt(ts_all[-1])}")

    anchor_ts   = float(ts_all.max())
    full_names  = [f"X{i}" for i in range(X_full.shape[1])]
    micro_names = [f"V{i}" for i in range(N_V6_EXTRA)]

    # Step 5: Save models
    global DRIVE_MODELS_DIR
    if DRIVE_AVAILABLE:
        os.makedirs(DRIVE_MODELS_DIR, exist_ok=True)
    else:
        # Save locally if Drive not mounted
        DRIVE_MODELS_DIR = DRIVE_MODELS_DIR.replace("/content/drive/MyDrive", LOCAL_DATA_DIR)
        os.makedirs(DRIVE_MODELS_DIR, exist_ok=True)
    horizon_results = {}

    for h in HORIZONS:
        horizon_results[str(h)] = {}
        ret_arr = per_h[f"ret_{h}"]
        for direction in ["long", "short"]:
            y_arr   = per_h[f"y_{h}"]   if direction=="long" else per_h[f"y_short_{h}"]
            net_arr = per_h[f"net_{h}"] if direction=="long" else per_h[f"net_short_{h}"]
            direction_results = {}

            r_full = train_horizon(
                h, direction,
                X_full[sl_tr], ts_all[sl_tr], y_arr[sl_tr], net_arr[sl_tr],
                X_full[sl_va],                y_arr[sl_va], net_arr[sl_va],
                X_full[sl_te],                y_arr[sl_te], net_arr[sl_te],
                ret_arr[sl_tr], ret_arr[sl_va], ret_arr[sl_te],
                v6_str[sl_tr], full_names, anchor_ts, "full", DRIVE_MODELS_DIR
            )
            if r_full: direction_results["full"] = r_full

            if not SKIP_MICRO_MODEL:
                r_micro = train_horizon(
                    h, direction,
                    X_micro[sl_tr], ts_all[sl_tr], y_arr[sl_tr], net_arr[sl_tr],
                    X_micro[sl_va],                y_arr[sl_va], net_arr[sl_va],
                    X_micro[sl_te],                y_arr[sl_te], net_arr[sl_te],
                    ret_arr[sl_tr], ret_arr[sl_va], ret_arr[sl_te],
                    v6_str[sl_tr], micro_names, anchor_ts, "micro", DRIVE_MODELS_DIR
                )
                if r_micro: direction_results["micro"] = r_micro

            horizon_results[str(h)][direction] = direction_results

    # Step 6: Summary
    print("\n" + "="*80)
    print("  TRAINING SUMMARY")
    print("="*80)
    hdr = f"  {'model':<18} {'h':>4} {'dir':>6} {'val_EV%':>9} {'val_WR%':>9} {'val_PF':>7} {'test_EV%':>10} {'test_WR%':>10} {'test_PF':>8}"
    print(hdr)
    for h_str, dirs in horizon_results.items():
        for direction, models in dirs.items():
            for tag, r in models.items():
                print(f"  {'V6_'+tag:<18} {h_str:>4} {direction:>6} "
                      f"{r['val_ev_pct']:>+9.3f} {r['val_win_rate_pct']:>9.1f} "
                      f"{r['val_profit_factor']:>7.2f} {r['test_ev_pct']:>+10.3f} "
                      f"{r['test_win_rate_pct']:>10.1f} {r['test_profit_factor']:>8.2f}")

    meta = {
        "model_type": "xgboost_v6_full_plus_micro",
        "n_features_v6_full": X_full.shape[1],
        "n_features_v6_micro": N_V6_EXTRA,
        "n_features_v5_full": N_V5_FULL,
        "v6_extra_feature_names": FEATURE_NAMES_V6_EXTRA,
        "horizons": HORIZONS, "training_coins": symbols_used,
        "coin_onehot_order": COIN_ONEHOT_NAMES,
        "total_samples": int(N),
        "split": {"train":int(tr_end),"val":int(va_end-tr_end),"test":int(N-va_end)},
        "round_trip_cost": ROUND_TRIP_COST, "min_ev_pct": MIN_EV_PCT,
        "blending": {"v5_weight":0.30,"v6_weight":0.70},
        "horizon_results": horizon_results,
    }
    meta_path = os.path.join(DRIVE_MODELS_DIR, "v6_meta.json")
    with open(meta_path, "w") as fp: json.dump(meta, fp, indent=2)
    print(f"\n[OK] Models saved -> {DRIVE_MODELS_DIR}")
    print(f"     Total time  -> {(time.time()-t0)/60:.1f} min")
    print()
    print("NEXT STEPS:")
    if DRIVE_AVAILABLE:
        print("  1. Download MyDrive/crypto_oracle/models_v6/ from Google Drive")
    else:
        print("  1. Download the models_v6 folder from the Colab file explorer on the left")
    print("  2. Copy all files into:  crypto-oracle-mcp/data/")
    print("  3. Run:  python live_demo_engine.py")

if __name__ == "__main__":
    main()

# Auto-run when pasted directly into a Colab cell
try:
    get_ipython  # only exists in Jupyter/Colab
    main()
except NameError:
    pass
