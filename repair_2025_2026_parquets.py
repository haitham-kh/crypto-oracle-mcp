"""
repair_2025_2026_parquets.py
============================
The 2025+ Parquet files were saved with raw tick-level data instead of
1-minute bars. Root cause: Binance Vision changed trade timestamp resolution
from milliseconds to microseconds for 2025+ data. The `pl.from_epoch(...,
time_unit="ms")` call in data_utils.py treated microseconds as milliseconds,
so `group_by_dynamic(every="1m")` saw each tick as being 1,000x further
apart than it was — meaning no two ticks fell in the same 1-minute window,
producing one output row per tick instead of per minute.

This script:
1. Reads each broken Parquet (which contains raw tick rows).
2. Reinterprets the timestamp integers as microseconds (the correct unit).
3. Converts to proper Datetime and re-aggregates into 1-minute OHLCV bars.
4. Overwrites the bad Parquet with the correct one IN-PLACE.

Only touches files from 2025 onwards. 2023 and 2024 files are left alone.
"""

import os
import glob
import polars as pl

PROCESSED_DIR = r"E:\training data for quant\processed_features"

def repair_file(path: str) -> None:
    filename = os.path.basename(path)
    print(f"\n[REPAIRING] {filename}")

    # --- 1. Load the broken file ---
    df = pl.read_parquet(path)
    original_rows = len(df)

    # --- 2. Validate: is this actually tick data? ---
    # Tick-data fingerprint: majority of rows have open == close == high == low
    tick_pct = df.select(
        ((pl.col("open") == pl.col("close")) &
         (pl.col("high") == pl.col("low")) &
         (pl.col("open") == pl.col("high"))).alias("is_tick")
    ).to_series().mean()

    if tick_pct < 0.50:
        print(f"  SKIP: {filename} — only {tick_pct*100:.1f}% tick-fingerprint rows, looks like proper bars already.")
        return

    print(f"  Tick fingerprint: {tick_pct*100:.1f}% of rows are single-price rows (confirmed raw ticks)")
    print(f"  Original row count: {original_rows:,}  |  Expected ~44,640 after repair")

    # --- 3. Reinterpret timestamp as MICROSECONDS then cast to ms Datetime ---
    # The raw integer in the Parquet is a Unix microsecond epoch (not ms).
    # Cast to Int64 first, divide by 1000 to get milliseconds, then to Datetime(ms).
    df_fixed = df.with_columns([
        (pl.col("timestamp").cast(pl.Int64) // 1000)
            .cast(pl.Datetime(time_unit="ms"))
            .alias("timestamp")
    ])

    # --- 4. Re-aggregate into 1-minute bars ---
    # The "open" column holds the individual trade price (since it was a tick).
    # We use it as our price source.  OFI was already computed per-tick correctly.
    minute_bars = (
        df_fixed
        .sort("timestamp")
        .group_by_dynamic("timestamp", every="1m")
        .agg([
            pl.col("open").first().alias("open"),
            pl.col("high").max().alias("high"),
            pl.col("low").min().alias("low"),
            pl.col("close").last().alias("close"),
            pl.col("volume").sum().alias("volume"),
            pl.col("ofi").sum().alias("ofi"),
        ])
        .sort("timestamp")
    )

    repaired_rows = len(minute_bars)
    print(f"  Repaired row count: {repaired_rows:,}  |  Compression ratio: {original_rows/max(repaired_rows,1):.0f}x")

    # Sanity check: repaired row count should be within ±5% of 44640 (days*24*60)
    # For months with different day counts (28-31 days) → 40320 to 44640 range
    # For 2026-04 (30 days) → ~43200
    if repaired_rows < 38000 or repaired_rows > 45000:
        print(f"  WARNING: Repaired row count {repaired_rows:,} is outside expected 38000-45000 range.")
        print(f"  This may indicate the month has sparse trading or a different issue.")
        print(f"  Proceeding anyway — manually verify this file.")

    # --- 5. Overwrite in-place ---
    minute_bars.write_parquet(path)
    print(f"  DONE: Saved repaired file -> {path}")


def main():
    # Only repair 2025 and 2026 files
    patterns = [
        os.path.join(PROCESSED_DIR, "*_1m_features_2025-*.parquet"),
        os.path.join(PROCESSED_DIR, "*_1m_features_2026-*.parquet"),
    ]

    all_files = []
    for pat in patterns:
        all_files.extend(sorted(glob.glob(pat)))

    print(f"Found {len(all_files)} files to repair (2025 + 2026).")
    print("2023 and 2024 files are untouched.\n")

    ok = 0
    skipped = 0
    errors = []

    for path in all_files:
        try:
            repair_file(path)
            ok += 1
        except Exception as e:
            print(f"  ERROR on {os.path.basename(path)}: {e}")
            errors.append((path, str(e)))

    print(f"\n{'='*60}")
    print(f"Repair complete: {ok} repaired, {skipped} skipped, {len(errors)} errors.")
    if errors:
        print("Errors:")
        for p, e in errors:
            print(f"  {os.path.basename(p)}: {e}")


if __name__ == "__main__":
    main()
