"""
download_v6_ohlcv.py
=====================
Downloads 24 months of 1m OHLCV klines from Binance public API for all 20 coins
and saves them as {symbol}_1m_ohlcv.parquet in the processed_features directory.

These are then used by train_v6.py to compute the 36 V6 features on-the-fly
and join them to the existing X_0..X_93 training_data parquets.

Usage:
    python download_v6_ohlcv.py
    python download_v6_ohlcv.py --months 12          # faster, less history
    python download_v6_ohlcv.py --coin BTCUSDT        # single coin test
"""
import os, sys, time, datetime, argparse
import requests
import numpy as np
import polars as pl

# ── Config ────────────────────────────────────────────────────────────────────
BASE = "https://fapi.binance.com"
PROCESSED_DIR = r"E:\training data for quant\processed_features"
MONTHS = 24

COINS = [
    "AAVEUSDT", "ADAUSDT", "APTUSDT", "ATOMUSDT", "AVAXUSDT",
    "BNBUSDT",  "BTCUSDT", "DOGEUSDT", "DOTUSDT",  "ETHUSDT",
    "FILUSDT",  "INJUSDT", "LINKUSDT", "LTCUSDT",  "NEARUSDT",
    "OPUSDT",   "SOLUSDT", "SUIUSDT",  "UNIUSDT",  "XRPUSDT",
]

def download_klines(symbol, start_ms, end_ms):
    candles = []
    current = start_ms
    batch = 1000  # Binance max per request

    while current < end_ms:
        for attempt in range(5):
            try:
                r = requests.get(
                    f"{BASE}/fapi/v1/klines",
                    params={"symbol": symbol, "interval": "1m",
                            "startTime": current, "endTime": end_ms,
                            "limit": batch},
                    timeout=30
                )
                if r.status_code == 429:
                    print(f"    Rate limited — waiting 30s...")
                    time.sleep(30)
                    continue
                r.raise_for_status()
                data = r.json()
                break
            except Exception as e:
                if attempt == 4:
                    print(f"    FATAL after 5 tries: {e}")
                    return candles
                time.sleep(2 ** attempt)

        if not data:
            break

        for row in data:
            candles.append({
                "timestamp_ms": int(row[0]),
                "open":         float(row[1]),
                "high":         float(row[2]),
                "low":          float(row[3]),
                "close":        float(row[4]),
                "volume":       float(row[5]),
                "taker_buy_volume": float(row[9]),
            })

        current = int(data[-1][0]) + 60_000
        if len(data) < batch:
            break

        time.sleep(0.05)  # gentle rate limiting

    return candles


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--months", type=int, default=MONTHS)
    parser.add_argument("--coin",   type=str, default=None, help="Single coin to download")
    parser.add_argument("--dir",    type=str, default=PROCESSED_DIR)
    args = parser.parse_args()

    os.makedirs(args.dir, exist_ok=True)

    end_dt   = datetime.datetime.utcnow().replace(minute=0, second=0, microsecond=0)
    start_dt = end_dt - datetime.timedelta(days=args.months * 30)
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms   = int(end_dt.timestamp() * 1000)

    coins = [args.coin] if args.coin else COINS

    print("=" * 65)
    print("  CRYPTO ORACLE V6 — OHLCV DOWNLOADER")
    print(f"  Period : {start_dt.date()} → {end_dt.date()}  ({args.months} months)")
    print(f"  Coins  : {len(coins)}")
    print(f"  Output : {args.dir}")
    print("=" * 65)

    for i, symbol in enumerate(coins, 1):
        out_path = os.path.join(args.dir, f"{symbol}_1m_ohlcv.parquet")

        # Skip if already downloaded and recent
        if os.path.exists(out_path):
            sz = os.path.getsize(out_path) / 1e6
            mtime = datetime.datetime.fromtimestamp(os.path.getmtime(out_path))
            age_h = (datetime.datetime.utcnow() - mtime).total_seconds() / 3600
            if age_h < 6:
                print(f"[{i}/{len(coins)}] {symbol} — SKIP (already downloaded {sz:.0f}MB, {age_h:.0f}h ago)")
                continue

        print(f"\n[{i}/{len(coins)}] {symbol} — downloading {args.months}m of 1m klines...")
        t0 = time.time()
        candles = download_klines(symbol, start_ms, end_ms)

        if not candles:
            print(f"  ERROR: no data returned")
            continue

        df = pl.DataFrame(candles).sort("timestamp_ms")
        # Add ofi column (used by V6 features)
        df = df.with_columns(
            (2.0 * pl.col("taker_buy_volume") - pl.col("volume")).alias("ofi")
        )
        df.write_parquet(out_path)
        elapsed = time.time() - t0
        sz = os.path.getsize(out_path) / 1e6
        print(f"  Done: {len(candles):,} bars → {out_path} ({sz:.0f} MB) in {elapsed:.0f}s")

    print("\n==> All done. Now run: python train_v6.py")


if __name__ == "__main__":
    main()
