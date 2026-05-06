"""
download_perp_data.py - Pull historical perp microstructure data.

Fetches from data.binance.vision (Binance's public S3 mirror) for every
coin × every month covered by your spot history:

  • Monthly funding-rate archives:
      data/futures/um/monthly/fundingRate/{SYM}/{SYM}-fundingRate-{YYYY-MM}.zip
    Contains 8-hour funding-rate snapshots (calc_time, last_funding_rate,
    funding_interval_hours).

  • Daily futures-metrics archives (5-minute resolution):
      data/futures/um/daily/metrics/{SYM}/{SYM}-metrics-{YYYY-MM-DD}.zip
    Contains: sum_open_interest, sum_open_interest_value, top-trader and
    global long/short ratios, taker buy/sell volume ratio.

For each symbol we concatenate everything we got into:

    data/perp/{SYMBOL}_funding.parquet     (8h cadence, sorted)
    data/perp/{SYMBOL}_metrics.parquet     (5min cadence, sorted)

Resume-safe: already-existing parquets are not re-downloaded, and per-month
download cache files are skipped if present.

Usage:
    python download_perp_data.py
"""
from __future__ import annotations
import os, io, sys, time, zipfile
import datetime as dt
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import polars as pl

HERE = os.path.dirname(os.path.abspath(__file__))
PERP_DIR  = os.path.join(HERE, "data", "perp")
CACHE_DIR = os.path.join(HERE, "data", "perp_cache")
os.makedirs(PERP_DIR, exist_ok=True); os.makedirs(CACHE_DIR, exist_ok=True)

BASE = "https://data.binance.vision/data/futures/um"
COINS = [
    "AAVEUSDT", "ADAUSDT", "AVAXUSDT", "BNBUSDT", "BTCUSDT",
    "DOGEUSDT", "ETHUSDT", "LINKUSDT", "PEPEUSDT", "SHIBUSDT",
    "SOLUSDT", "UNIUSDT", "WIFUSDT", "XRPUSDT",
]

START_DATE = dt.date(2023, 1, 1)
END_DATE   = dt.date.today().replace(day=1)   # current month not yet archived

MAX_WORKERS = 12      # threads for parallel downloads
TIMEOUT     = 30


# ── HTTP helpers ────────────────────────────────────────────────────────────

def _get_zip_csv(url, cache_path):
    """Download `url` (zipped CSV) → return bytes of the first CSV inside.
    Caches the zip file at cache_path. Returns None on 404 / failure."""
    if os.path.exists(cache_path) and os.path.getsize(cache_path) > 0:
        with open(cache_path, "rb") as f:
            data = f.read()
    else:
        for attempt in range(3):
            try:
                r = requests.get(url, timeout=TIMEOUT)
                if r.status_code == 404:
                    # mark as missing with empty file so we don't retry
                    open(cache_path, "wb").close()
                    return None
                r.raise_for_status()
                data = r.content
                with open(cache_path, "wb") as f:
                    f.write(data)
                break
            except Exception as e:
                if attempt == 2:
                    print(f"      !! gave up on {url}: {e}")
                    return None
                time.sleep(1.0 + attempt)
        else:
            return None
    if not data:
        return None
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            name = [n for n in zf.namelist() if n.endswith(".csv")][0]
            return zf.read(name)
    except Exception as e:
        print(f"      !! bad zip {url}: {e}")
        return None


# ── Funding rate (monthly) ──────────────────────────────────────────────────

def funding_url(sym, year, month):
    return f"{BASE}/monthly/fundingRate/{sym}/{sym}-fundingRate-{year:04d}-{month:02d}.zip"


def fetch_funding_month(sym, year, month):
    fname = f"{sym}-fundingRate-{year:04d}-{month:02d}.zip"
    cache = os.path.join(CACHE_DIR, "funding", sym, fname)
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    csv_bytes = _get_zip_csv(funding_url(sym, year, month), cache)
    if csv_bytes is None or len(csv_bytes) < 30:
        return None
    try:
        df = pl.read_csv(io.BytesIO(csv_bytes), has_header=True, infer_schema_length=200)
    except Exception:
        return None
    cols = {c.lower(): c for c in df.columns}
    # Schema (typical): calc_time, funding_interval_hours, last_funding_rate
    ct  = cols.get("calc_time")
    rt  = cols.get("last_funding_rate") or cols.get("funding_rate")
    if ct is None or rt is None:
        return None
    return df.select([
        pl.col(ct).cast(pl.Int64).alias("ts_ms"),
        pl.col(rt).cast(pl.Float64).alias("funding_rate"),
    ])


# ── Daily metrics (OI + LSR + taker ratio @ 5min) ───────────────────────────

def metrics_url(sym, date):
    return (f"{BASE}/daily/metrics/{sym}/"
            f"{sym}-metrics-{date.isoformat()}.zip")


def fetch_metrics_day(sym, date):
    fname = f"{sym}-metrics-{date.isoformat()}.zip"
    cache = os.path.join(CACHE_DIR, "metrics", sym, fname)
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    csv_bytes = _get_zip_csv(metrics_url(sym, date), cache)
    if csv_bytes is None or len(csv_bytes) < 30:
        return None
    try:
        df = pl.read_csv(io.BytesIO(csv_bytes), has_header=True, infer_schema_length=200)
    except Exception:
        return None
    # Schema: create_time, symbol, sum_open_interest, sum_open_interest_value,
    #         count_toptrader_long_short_ratio, sum_toptrader_long_short_ratio,
    #         count_long_short_ratio, sum_taker_long_short_vol_ratio
    cm = {c.lower(): c for c in df.columns}
    ct = cm.get("create_time")
    if ct is None:
        return None

    def pick(*names):
        for n in names:
            if n in cm: return cm[n]
        return None

    oi   = pick("sum_open_interest")
    oi_v = pick("sum_open_interest_value")
    lsr_top = pick("sum_toptrader_long_short_ratio")
    lsr_glb = pick("sum_long_short_ratio", "count_long_short_ratio")
    taker   = pick("sum_taker_long_short_vol_ratio")
    if oi is None:
        return None
    parts = [
        # create_time can be ISO string OR ms int; coerce robustly.
        pl.col(ct).cast(pl.Utf8).str.strptime(pl.Datetime, strict=False)
            .dt.timestamp("ms").alias("ts_ms"),
        pl.col(oi).cast(pl.Float64, strict=False).alias("oi"),
    ]
    if oi_v: parts.append(pl.col(oi_v).cast(pl.Float64, strict=False).alias("oi_value"))
    if lsr_top: parts.append(pl.col(lsr_top).cast(pl.Float64, strict=False).alias("lsr_top"))
    if lsr_glb: parts.append(pl.col(lsr_glb).cast(pl.Float64, strict=False).alias("lsr_global"))
    if taker:   parts.append(pl.col(taker).cast(pl.Float64, strict=False).alias("taker_ratio"))
    out = df.select(parts).filter(pl.col("ts_ms").is_not_null())
    return out


# ── Driver ──────────────────────────────────────────────────────────────────

def months_between(start, end):
    y, m = start.year, start.month
    while (y, m) < (end.year, end.month):
        yield y, m
        m += 1
        if m > 12: m = 1; y += 1


def days_between(start, end):
    d = start
    one = dt.timedelta(days=1)
    while d < end:
        yield d
        d += one


def process_symbol(sym):
    funding_path = os.path.join(PERP_DIR, f"{sym}_funding.parquet")
    metrics_path = os.path.join(PERP_DIR, f"{sym}_metrics.parquet")
    have_f = os.path.exists(funding_path)
    have_m = os.path.exists(metrics_path)
    if have_f and have_m:
        print(f"[{sym}] already have funding + metrics parquets, skipping.")
        return

    # Funding (monthly)
    if not have_f:
        print(f"[{sym}] downloading funding rates...")
        months = list(months_between(START_DATE, END_DATE))
        funds = []
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futs = {ex.submit(fetch_funding_month, sym, y, m): (y, m) for y, m in months}
            done = 0; total = len(futs)
            for fut in as_completed(futs):
                y, m = futs[fut]
                try:
                    df = fut.result()
                    if df is not None and df.height > 0:
                        funds.append(df)
                except Exception as e:
                    print(f"      !! {sym} funding {y}-{m}: {e}")
                done += 1
                if done % 10 == 0:
                    print(f"    {sym} funding {done}/{total}", flush=True)
        if funds:
            full = pl.concat(funds).unique(subset=["ts_ms"]).sort("ts_ms")
            full.write_parquet(funding_path)
            print(f"  -> {funding_path}  ({full.height:,} rows)")
        else:
            print(f"  !! no funding data for {sym}")

    # Metrics (daily, 5min snapshots inside each)
    if not have_m:
        print(f"[{sym}] downloading metrics (OI/LSR/taker @ 5min)...")
        days = list(days_between(START_DATE, END_DATE))
        mets = []
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futs = {ex.submit(fetch_metrics_day, sym, d): d for d in days}
            done = 0; total = len(futs); ok = 0
            for fut in as_completed(futs):
                d = futs[fut]
                try:
                    df = fut.result()
                    if df is not None and df.height > 0:
                        mets.append(df); ok += 1
                except Exception as e:
                    print(f"      !! {sym} metrics {d}: {e}")
                done += 1
                if done % 100 == 0:
                    print(f"    {sym} metrics {done}/{total}  (got {ok})", flush=True)
        if mets:
            full = pl.concat(mets, how="diagonal").unique(subset=["ts_ms"]).sort("ts_ms")
            full.write_parquet(metrics_path)
            print(f"  -> {metrics_path}  ({full.height:,} rows)")
        else:
            print(f"  !! no metrics data for {sym}")


def main():
    print("=" * 78)
    print("  PERP DATA DOWNLOADER  (data.binance.vision)")
    print(f"  Coverage: {START_DATE} → {END_DATE}")
    print(f"  Coins:   {len(COINS)}   |   workers: {MAX_WORKERS}")
    print(f"  Output:  {PERP_DIR}")
    print("=" * 78)
    t0 = time.time()
    for sym in COINS:
        process_symbol(sym)
    print(f"\nDone in {(time.time()-t0)/60:.1f} min.")


if __name__ == "__main__":
    main()
