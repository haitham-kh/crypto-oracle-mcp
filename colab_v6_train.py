"""
╔═══════════════════════════════════════════════════════════════════╗
║  CRYPTO ORACLE V6 — GOOGLE COLAB TRAINING SCRIPT                 ║
║                                                                   ║
║  Paste this entire file into a single Colab cell and run it.     ║
║  Or upload as colab_v6_train.py and run:  !python colab_v6_train.py  ║
║                                                                   ║
║  What it does (fully automatic):                                  ║
║    1. Installs dependencies                                       ║
║    2. Mounts Google Drive (optional — for saving results)         ║
║    3. Downloads 24m of 1m Binance Futures klines for 20 coins    ║
║    4. Loads your existing training_data.parquets from Drive       ║
║    5. Computes all 36 V6 features inline (self-contained)        ║
║    6. Trains V6_full (130 feat) + V6_micro (36 feat) with XGBoost║
║    7. Saves model files to Drive → copy back to your PC          ║
║                                                                   ║
║  SETUP (do once before running):                                  ║
║    - Upload your E-drive parquets to Google Drive at:            ║
║      MyDrive/crypto_oracle/processed/                            ║
║    - The script will auto-detect and use them                    ║
╚═══════════════════════════════════════════════════════════════════╝
"""

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 0 — CONFIG  (edit these if needed)
# ─────────────────────────────────────────────────────────────────────────────

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

import os, time, datetime, json, math, gc, glob, pickle, threading
import requests
import numpy as np
import polars as pl
import xgboost as xgb
from concurrent.futures import ThreadPoolExecutor, as_completed

os.makedirs(LOCAL_DATA_DIR, exist_ok=True)
os.makedirs(DRIVE_MODELS_DIR, exist_ok=True)

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

os.makedirs(DRIVE_PROCESSED_DIR, exist_ok=True)
os.makedirs(DRIVE_MODELS_DIR,    exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — BINANCE KLINE DOWNLOADER (parallel, fast)
# ─────────────────────────────────────────────────────────────────────────────

BINANCE_BASE = "https://fapi.binance.com"

def _download_chunk(symbol, start_ms, end_ms, limit=1000):
    for attempt in range(6):
        try:
            r = requests.get(f"{BINANCE_BASE}/fapi/v1/klines",
                             params={"symbol": symbol, "interval": "1m",
                                     "startTime": start_ms, "endTime": end_ms,
                                     "limit": limit},
                             timeout=30)
            if r.status_code == 429:
                time.sleep(30); continue
            r.raise_for_status()
            return r.json()
        except Exception:
            time.sleep(2 ** attempt)
    return []

def download_klines(symbol, start_ms, end_ms, n_workers=8):
    """Parallel download: splits time range into chunks and fetches concurrently."""
    # Split into ~500-bar chunks for parallel fetch
    chunk_ms = 1000 * 60_000  # 1000 minutes per chunk
    ranges = []
    cur = start_ms
    while cur < end_ms:
        nxt = min(cur + chunk_ms, end_ms)
        ranges.append((cur, nxt))
        cur = nxt

    all_rows = {}  # ts -> row, dedup by timestamp
    lock = threading.Lock()

    def fetch(rng):
        data = _download_chunk(symbol, rng[0], rng[1])
        with lock:
            for row in data:
                ts = int(row[0])
                if ts not in all_rows:
                    all_rows[ts] = row

    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futs = [ex.submit(fetch, r) for r in ranges]
        for i, f in enumerate(as_completed(futs)):
            f.result()
            if (i + 1) % 50 == 0:
                print(f"    {symbol}: {i+1}/{len(ranges)} chunks done", flush=True)

    sorted_rows = sorted(all_rows.values(), key=lambda r: int(r[0]))
    candles = [{
        "timestamp_ms":      int(r[0]),
        "open":              float(r[1]),
        "high":              float(r[2]),
        "low":               float(r[3]),
        "close":             float(r[4]),
        "volume":            float(r[5]),
        "taker_buy_volume":  float(r[9]),
    } for r in sorted_rows]
    return candles


def download_all_coins(coins, months, out_dir, n_coin_workers=4):
    """Download all coins in parallel (4 coins at a time)."""
    end_dt   = datetime.datetime.utcnow().replace(minute=0, second=0, microsecond=0)
    start_dt = end_dt - datetime.timedelta(days=months * 30)
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms   = int(end_dt.timestamp() * 1000)
    expected_bars = (end_ms - start_ms) // 60_000

    print(f"\n{'='*65}")
    print(f"  DOWNLOADING {len(coins)} coins  |  {months}m  |  ~{expected_bars:,} bars/coin")
    print(f"  Period: {start_dt.date()} → {end_dt.date()}")
    print(f"{'='*65}\n")

    os.makedirs(out_dir, exist_ok=True)
    ohlcv_paths = {}

    def process_coin(sym):
        out_path = os.path.join(out_dir, f"{sym}_1m_ohlcv.parquet")
        if os.path.exists(out_path):
            sz = os.path.getsize(out_path) / 1e6
            df_check = pl.read_parquet(out_path)
            if len(df_check) > expected_bars * 0.90:
                print(f"  {sym}: SKIP (already {len(df_check):,} bars, {sz:.0f}MB)")
                return sym, out_path
            print(f"  {sym}: incomplete ({len(df_check):,}/{expected_bars:,} bars), re-downloading")

        t0 = time.time()
        print(f"  {sym}: downloading...", flush=True)
        candles = download_klines(sym, start_ms, end_ms, n_workers=6)
        if not candles:
            print(f"  {sym}: ERROR — no data")
            return sym, None

        df = pl.DataFrame(candles).sort("timestamp_ms")
        df = df.with_columns((2.0 * pl.col("taker_buy_volume") - pl.col("volume")).alias("ofi"))
        df.write_parquet(out_path)
        sz = os.path.getsize(out_path) / 1e6
        elapsed = time.time() - t0
        print(f"  {sym}: {len(candles):,} bars → {sz:.0f}MB  ({elapsed:.0f}s)", flush=True)
        return sym, out_path

    # Sequential to avoid Binance rate limits — Colab is fast enough
    for sym in coins:
        sym, path = process_coin(sym)
        if path:
            ohlcv_paths[sym] = path

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
    """ATR-14 using true range."""
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
# SECTION 6 — DATA LOADING  (parquet → X_v5 + labels)
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

    # Load training parquet (sampled rows)
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

    # Load OHLCV and compute V6 features on full 1m series
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

    # Align by timestamp (±5 min tolerance)
    pos = np.searchsorted(ts_f, ts, side="left")
    pos = np.clip(pos, 0, len(ts_f) - 1)
    ok  = np.abs(ts_f[pos] - ts) <= 5 * 60_000
    F_v6_samp = np.zeros((n, N_V6_EXTRA), np.float32)
    F_v6_samp[ok] = F_v6_full[pos[ok]]
    print(f"  [{symbol}] V6 match: {ok.sum()}/{n} ({ok.mean()*100:.1f}%)")

    X_v6  = np.hstack([X_v5, F_v6_samp]).astype(np.float32)   # (n, 130)
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
    print(f"\n  ── V6_{model_tag} | h={h}m | {direction.upper()} ────────────────────────", flush=True)

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

    # Save
    clf_path   = os.path.join(models_dir, f"v6_{model_tag}_clf_{direction}_h{h}.json")
    calib_path = os.path.join(models_dir, f"v6_{model_tag}_calib_{direction}_h{h}.pkl")
    clf.save_model(clf_path)
    with open(calib_path, "wb") as fp: pickle.dump(calib, fp)

    # Regressor (long + full only)
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

    # ── Check GPU ──────────────────────────────────────────────────────────
    _check_gpu()

    # ── Step 1: Download OHLCV ─────────────────────────────────────────────
    ohlcv_dir = os.path.join(LOCAL_DATA_DIR, "ohlcv")
    ohlcv_paths = download_all_coins(COINS, MONTHS_HISTORY, ohlcv_dir)

    # ── Step 2: Load training bundles ─────────────────────────────────────
    processed_dir = DRIVE_PROCESSED_DIR
    if not os.path.exists(processed_dir) or not glob.glob(os.path.join(processed_dir, "*_training_data.parquet")):
        # Fallback: look in local data dir
        alt = os.path.join(LOCAL_DATA_DIR, "processed")
        if glob.glob(os.path.join(alt, "*_training_data.parquet")):
            processed_dir = alt
        else:
            print(f"\n[ERROR] No training_data.parquet files found in:")
            print(f"  {DRIVE_PROCESSED_DIR}")
            print(f"  {alt}")
            print("\nUpload your E-drive parquets to Google Drive at:")
            print(f"  MyDrive/crypto_oracle/processed/")
            return

    print(f"\n[DATA] Processed dir: {processed_dir}")
    tp_files = glob.glob(os.path.join(processed_dir, "*_training_data.parquet"))
    print(f"  Found {len(tp_files)} training_data.parquet files")

    bundles = []
    for sym in COINS:
        b = load_training_bundle(sym, processed_dir, ohlcv_dir)
        if b is not None:
            bundles.append(b)
        gc.collect()

    if not bundles:
        print("[ERROR] No bundles loaded — check paths above."); return

    # ── Step 3: Stack ─────────────────────────────────────────────────────
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
    print(f"\nTotal: {N:,} samples × {X_full.shape[1]} features")
    print(f"  train : {tr_end:,}  {_fmt(ts_all[0])} → {_fmt(ts_all[tr_end-1])}")
    print(f"  val   : {va_end-tr_end:,}  {_fmt(ts_all[tr_end])} → {_fmt(ts_all[va_end-1])}")
    print(f"  test  : {N-va_end:,}  {_fmt(ts_all[va_end])} → {_fmt(ts_all[-1])}")

    anchor_ts = float(ts_all.max())
    full_names  = [f"X{i}" for i in range(X_full.shape[1])]
    micro_names = [f"V{i}" for i in range(N_V6_EXTRA)]

    # ── Step 4: Train models ──────────────────────────────────────────────
    os.makedirs(DRIVE_MODELS_DIR, exist_ok=True)
    horizon_results = {}

    for h in HORIZONS:
        horizon_results[str(h)] = {}
        ret_arr = per_h[f"ret_{h}"]
        for direction in ["long", "short"]:
            y_arr   = per_h[f"y_{h}"]   if direction=="long" else per_h[f"y_short_{h}"]
            net_arr = per_h[f"net_{h}"] if direction=="long" else per_h[f"net_short_{h}"]
            direction_results = {}

            # V6_full (130 features)
            r_full = train_horizon(
                h, direction,
                X_full[sl_tr], ts_all[sl_tr], y_arr[sl_tr], net_arr[sl_tr],
                X_full[sl_va],                y_arr[sl_va], net_arr[sl_va],
                X_full[sl_te],                y_arr[sl_te], net_arr[sl_te],
                ret_arr[sl_tr], ret_arr[sl_va], ret_arr[sl_te],
                v6_str[sl_tr], full_names, anchor_ts, "full", DRIVE_MODELS_DIR
            )
            if r_full: direction_results["full"] = r_full

            # V6_micro (36 features)
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

    # ── Step 5: Summary & Meta ───────────────────────────────────────────
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
    print(f"\n[OK] Meta saved → {meta_path}")
    print(f"     Models dir  → {DRIVE_MODELS_DIR}")
    print(f"     Total time  → {(time.time()-t0)/60:.1f} min")
    print()
    print("NEXT STEPS:")
    print("  1. Download everything from Google Drive:")
    print(f"     {DRIVE_MODELS_DIR}/")
    print("  2. Copy ALL files to your PC:")
    print("     c:\\\\Users\\\\skuna\\\\cryptogame\\\\crypto-oracle-mcp\\\\data\\\\")
    print("  3. Restart the live engine:")
    print("     python live_demo_engine.py")
    print("     (it will auto-detect V6 and activate 70/30 blending)")

if __name__ == "__main__":
    main()

# ─────────────────────────────────────────────────────────────────────────────
# Run automatically when pasted into a Colab cell
# ─────────────────────────────────────────────────────────────────────────────
try:
    get_ipython  # only exists inside Jupyter/Colab
    main()
except NameError:
    pass  # running as a script, main() called via __main__ above
