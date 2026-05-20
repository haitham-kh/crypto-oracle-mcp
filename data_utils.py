import pandas as pd
import numpy as np

def apply_triple_barrier(
    price_series: pd.Series,
    events: pd.DatetimeIndex,
    target_volatility: pd.Series,
    pt_multiplier: float = 1.5,
    sl_multiplier: float = 1.0,
    vertical_barrier_minutes: int = 240,
) -> pd.DataFrame:
    """
    Marcos Lopez de Prado's Triple Barrier Method.
    
    Args:
        price_series: High-resolution price data (e.g., 1-minute bars or tick data).
        events: The timestamps where feature snapshots were taken (your entry times).
        target_volatility: ATR or volatility measure (in percentage/absolute terms) at each event.
        pt_multiplier: Profit-taking multiplier (Upper Barrier).
        sl_multiplier: Stop-loss multiplier (Lower Barrier).
        vertical_barrier_minutes: Time limit for the trade (Vertical Barrier).
        
    Returns:
        DataFrame with columns: ['entry_time', 'exit_time', 'label', 'return']
        Label = 1 if Upper Barrier hit first (Success)
        Label = 0 if Lower or Vertical Barrier hit first (Failure/Stop)
    """
    out = pd.DataFrame(index=events, columns=['exit_time', 'label', 'return'])
    
    for entry_time in events:
        if entry_time not in price_series.index:
            continue
            
        entry_price = price_series.loc[entry_time]
        vol = target_volatility.loc[entry_time]
        
        # Define the barriers
        upper_barrier = entry_price * (1 + (vol * pt_multiplier))
        lower_barrier = entry_price * (1 - (vol * sl_multiplier))
        vertical_barrier_time = entry_time + pd.Timedelta(minutes=vertical_barrier_minutes)
        
        # Slice the future path up to the vertical barrier
        path = price_series.loc[entry_time:vertical_barrier_time]
        if path.empty:
            continue
            
        # Find first touch of barriers
        hit_upper = path[path >= upper_barrier].index.min()
        hit_lower = path[path <= lower_barrier].index.min()
        
        # Determine which barrier was hit first
        first_touch = pd.NaT
        label = 0 # Default to failure
        
        if pd.notna(hit_upper) and pd.notna(hit_lower):
            if hit_upper < hit_lower:
                first_touch = hit_upper
                label = 1
            else:
                first_touch = hit_lower
                label = 0
        elif pd.notna(hit_upper):
            first_touch = hit_upper
            label = 1
        elif pd.notna(hit_lower):
            first_touch = hit_lower
            label = 0
        else:
            # Vertical barrier hit
            first_touch = path.index[-1]
            label = 0
            
        exit_price = price_series.loc[first_touch]
        ret = (exit_price - entry_price) / entry_price
        
        out.loc[entry_time, 'exit_time'] = first_touch
        out.loc[entry_time, 'label'] = label
        out.loc[entry_time, 'return'] = ret
        
    return out

# --- DATA GATHERING & STORAGE CONFIGURATION ---
import os
import polars as pl

# Explicitly set the storage directory to the E: drive to prevent C: drive OOM/storage crashes
STORAGE_DIR = r"E:\training data for quant"

def setup_storage():
    """Ensure the E: drive storage directories exist before downloading data."""
    os.makedirs(os.path.join(STORAGE_DIR, "raw_ticks"), exist_ok=True)
    os.makedirs(os.path.join(STORAGE_DIR, "processed_features"), exist_ok=True)
    print(f"Storage directories verified at {STORAGE_DIR}")

# --- TICK-TO-MINUTE AGGREGATION ---
# Uses hash-based group_by on an integer minute-key so Polars never needs to
# sort the full CSV. Sorting a 9+ GB file requires holding it all in RAM and
# breaks streaming — this approach stays well under 2 GB regardless of CSV size.
def process_tick_data_in_chunks(symbol: str, date_str: str):
    setup_storage()

    raw_csv_path = os.path.join(STORAGE_DIR, "raw_ticks", f"{symbol}-trades-{date_str}.csv")
    output_parquet = os.path.join(STORAGE_DIR, "processed_features", f"{symbol}_1m_features_{date_str}.parquet")

    if not os.path.exists(raw_csv_path):
        print(f"Waiting for data: {raw_csv_path} does not exist yet.")
        return

    # Binance Vision CSVs have no header row.
    binance_columns = ["id", "price", "qty", "quote_qty", "time_ms", "is_buyer_maker", "is_best_match"]

    df = pl.scan_csv(raw_csv_path, has_header=False, new_columns=binance_columns)

    # Detect time unit: Binance Vision switched from milliseconds to microseconds
    # in newer files. Sample the first value to decide: ms timestamps are 13 digits,
    # us timestamps are 16 digits.
    sample = pl.scan_csv(raw_csv_path, has_header=False,
                         new_columns=binance_columns, n_rows=1).collect()
    raw_ts = int(sample["time_ms"][0])
    # 16-digit value = microseconds (introduced ~2025); 13-digit = milliseconds
    if raw_ts > 1_000_000_000_000_000:
        us_per_minute = 60_000_000
        time_unit = "us"
    else:
        us_per_minute = 60_000
        time_unit = "ms"

    # Add an integer minute-bucket key by flooring to the nearest minute.
    # Pure arithmetic — no sort, fully streaming.
    df_bucketed = df.with_columns([
        pl.col("id").cast(pl.Int64),
        pl.col("price").cast(pl.Float64),
        pl.col("qty").cast(pl.Float64),
        pl.col("is_buyer_maker").cast(pl.Boolean),
        ((pl.col("time_ms").cast(pl.Int64) // us_per_minute) * us_per_minute).alias("minute_key"),
    ])

    # Hash-based group_by: fully streaming, zero global-sort overhead.
    # open/close use sort_by("id") — trade IDs are sequential integers so
    # min-id = first trade, max-id = last trade in the minute.
    minute_bars = (
        df_bucketed.group_by("minute_key")
        .agg([
            pl.col("price").sort_by("id").first().alias("open"),
            pl.col("price").max().alias("high"),
            pl.col("price").min().alias("low"),
            pl.col("price").sort_by("id").last().alias("close"),
            pl.col("qty").sum().alias("volume"),
            pl.when(~pl.col("is_buyer_maker")).then(pl.col("qty"))
              .otherwise(-pl.col("qty")).sum().alias("ofi"),
        ])
        .collect(streaming=True)
    )

    # Sort the tiny result (~44k rows/month) and convert bucket key to datetime.
    minute_bars = (
        minute_bars
        .sort("minute_key")
        .with_columns(
            pl.from_epoch(pl.col("minute_key"), time_unit=time_unit)
              .dt.cast_time_unit("ms")
              .alias("timestamp")
        )
        .drop("minute_key")
    )

    minute_bars.write_parquet(output_parquet)
    print(f"SUCCESS: {symbol} {date_str} -> {minute_bars.height:,} 1m bars  [{output_parquet}]")
    return minute_bars
