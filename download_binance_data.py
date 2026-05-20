"""
download_binance_data.py — Parallel downloader for Binance Vision trade data.

Downloads monthly trade ZIPs, extracts them, processes each CSV into a compact
1-minute-bar Parquet via data_utils.process_tick_data_in_chunks(), then deletes
the raw CSV and ZIP to reclaim disk space.

Architecture:
  - 3 concurrent worker threads (download+process in each).
  - Each worker: download ZIP → extract → aggregate to parquet → delete CSV.
  - The aggregation is fully streaming (hash-based group_by) so each worker
    uses <2 GB RAM even for 10 GB CSVs.  3 workers ≈ 6 GB peak, safe on 16 GB.
  - 1 MB download chunks for maximum throughput.
  - Resume logic: if the output parquet already exists, the job is skipped.
"""
import os
import sys
import time
import requests
import zipfile
import shutil
import gc
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from data_utils import process_tick_data_in_chunks

# ── Storage paths ──────────────────────────────────────────────────────────────
STORAGE_DIR   = r"E:\training data for quant"
RAW_TICKS_DIR = os.path.join(STORAGE_DIR, "raw_ticks")
PROCESSED_DIR = os.path.join(STORAGE_DIR, "processed_features")

os.makedirs(RAW_TICKS_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

# ── Constants ──────────────────────────────────────────────────────────────────
BASE_URL             = "https://data.binance.vision/data/spot/monthly"
MIN_FREE_SPACE_BYTES = 20 * 1024 * 1024 * 1024   # 20 GB safety buffer
DOWNLOAD_CHUNK_SIZE  = 1 * 1024 * 1024            # 1 MB per read (was 8 KB)
MAX_WORKERS          = 3                           # concurrent download+process

# ── Thread-safe progress counter ──────────────────────────────────────────────
_lock    = threading.Lock()
_done    = 0
_total   = 0
_failed  = []

def _progress(symbol, date_str, status):
    global _done
    with _lock:
        _done += 1
        pct = _done / _total * 100 if _total else 0
        print(f"[{_done}/{_total}  {pct:5.1f}%]  {symbol} {date_str} — {status}")
        sys.stdout.flush()


def check_storage():
    """Returns True if E: has enough free space."""
    _, _, free = shutil.disk_usage("E:\\")
    return free >= MIN_FREE_SPACE_BYTES


# ── Single job: download → extract → process → cleanup ───────────────────────
def process_one_month(symbol: str, date_str: str) -> bool:
    """
    Complete pipeline for one (symbol, month).
    Returns True on success or skip, False on failure.
    """
    parquet_path = os.path.join(PROCESSED_DIR, f"{symbol}_1m_features_{date_str}.parquet")

    # Resume: already done?
    if os.path.exists(parquet_path):
        _progress(symbol, date_str, "SKIP (parquet exists)")
        return True

    if not check_storage():
        _progress(symbol, date_str, "ABORT (disk full)")
        return False

    zip_filename = f"{symbol}-trades-{date_str}.zip"
    csv_filename = f"{symbol}-trades-{date_str}.csv"
    url          = f"{BASE_URL}/trades/{symbol}/{zip_filename}"
    zip_path     = os.path.join(RAW_TICKS_DIR, zip_filename)
    csv_path     = os.path.join(RAW_TICKS_DIR, csv_filename)

    try:
        # ── 1. Download ZIP (streamed to disk, 1 MB chunks) ──────────────
        resp = requests.get(url, stream=True, timeout=300)
        if resp.status_code == 404:
            _progress(symbol, date_str, "SKIP (404 — not on Binance yet)")
            return True
        resp.raise_for_status()

        with open(zip_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=DOWNLOAD_CHUNK_SIZE):
                f.write(chunk)

        # ── 2. Extract ZIP ───────────────────────────────────────────────
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(RAW_TICKS_DIR)

        # Delete ZIP immediately — we only need the CSV now
        os.remove(zip_path)

        # ── 3. Process CSV → Parquet (streaming, <2 GB RAM) ─────────────
        process_tick_data_in_chunks(symbol, date_str)

        # ── 4. Delete the massive CSV ────────────────────────────────────
        if os.path.exists(csv_path):
            os.remove(csv_path)

        gc.collect()
        _progress(symbol, date_str, "DONE ✓")
        return True

    except Exception as e:
        # Clean up partial files on failure
        for p in (zip_path, csv_path):
            if os.path.exists(p):
                try: os.remove(p)
                except OSError: pass
        with _lock:
            _failed.append((symbol, date_str, str(e)))
        _progress(symbol, date_str, f"FAIL: {e}")
        return False


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Binance Data Downloader — targeting {STORAGE_DIR}")
    print(f"Workers: {MAX_WORKERS}  |  Chunk size: {DOWNLOAD_CHUNK_SIZE // 1024} KB\n")

    TARGET_COINS = ["BTCUSDT", "ETHUSDT"]

    SCHEDULE = {
        2023: list(range(1, 13)),
        2024: list(range(1, 13)),
        2025: list(range(1, 13)),
        2026: list(range(1, 5)),
    }

    # Build flat job list: [(symbol, "YYYY-MM"), ...]
    jobs = []
    for coin in TARGET_COINS:
        for year, months in SCHEDULE.items():
            for m in months:
                jobs.append((coin, f"{year}-{m:02d}"))

    _total = len(jobs)
    print(f"Total jobs: {_total}  (already-done parquets will be skipped instantly)\n")

    t0 = time.time()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(process_one_month, sym, ds): (sym, ds)
                   for sym, ds in jobs}

        for fut in as_completed(futures):
            # Exceptions inside the worker are caught there; this is just
            # a safety net so one crash doesn't kill the pool.
            try:
                fut.result()
            except Exception as exc:
                sym, ds = futures[fut]
                print(f"  ** Unhandled error for {sym} {ds}: {exc}")

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"Finished in {elapsed/60:.1f} minutes.")
    print(f"Succeeded: {_done - len(_failed)}  |  Failed: {len(_failed)}")
    if _failed:
        print("\nFailed jobs:")
        for sym, ds, err in _failed:
            print(f"  {sym} {ds}: {err}")
    print(f"{'='*60}")
