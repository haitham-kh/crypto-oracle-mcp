import os
import requests
import zipfile
import io
import shutil
import glob
import gc
from concurrent.futures import ThreadPoolExecutor
from data_utils import process_tick_data_in_chunks

# Target the E: drive
STORAGE_DIR = r"E:\training data for quant"
RAW_TICKS_DIR = os.path.join(STORAGE_DIR, "raw_ticks")
RAW_KLINES_DIR = os.path.join(STORAGE_DIR, "raw_klines")
PROCESSED_DIR = os.path.join(STORAGE_DIR, "processed_features")

os.makedirs(RAW_TICKS_DIR, exist_ok=True)
os.makedirs(RAW_KLINES_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Safety threshold: 20 GB in bytes
MIN_FREE_SPACE_BYTES = 20 * 1024 * 1024 * 1024  

# Binance Vision URLs (100% Free, No API Keys needed)
BASE_URL = "https://data.binance.vision/data/spot/monthly"

def check_storage():
    """Returns True if there is enough free space, False otherwise."""
    total, used, free = shutil.disk_usage("E:\\")
    if free < MIN_FREE_SPACE_BYTES:
        free_gb = free / (1024**3)
        print(f"\nCRITICAL: Storage space critically low ({free_gb:.2f} GB free).")
        print("Halting downloads to protect the 20GB safety buffer on your E: drive.")
        return False
    return True

def download_extract_and_process(url: str, symbol: str, date_str: str, is_trades: bool):
    """Downloads, extracts, processes to Parquet, and DELETES the massive CSV."""
    if not check_storage():
        return False
        
    # --- RESUME LOGIC ---
    # Check if the final Parquet file already exists. If it does, skip the download entirely!
    if is_trades:
        expected_parquet = os.path.join(PROCESSED_DIR, f"{symbol}_1m_features_{date_str}.parquet")
        if os.path.exists(expected_parquet):
            print(f"Skipping {symbol} {date_str} - Parquet already exists!")
            return True

    filename = url.split("/")[-1]
    extract_to = RAW_TICKS_DIR if is_trades else RAW_KLINES_DIR
    
    print(f"Downloading {filename}...")
    try:
        response = requests.get(url, stream=True)
        if response.status_code == 404:
            print(f"Not found (data might not exist for this month): {url}")
            return True 
            
        response.raise_for_status()
        
        # 1. Extract the ZIP
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            z.extractall(extract_to)
            print(f"Extracted to {extract_to}")
            
        # 2. Process and Delete (Only for Trades/Ticks)
        if is_trades:
            print(f"Processing massive CSV into Parquet...")
            try:
                process_tick_data_in_chunks(symbol, date_str)
                
                # Delete the massive CSV immediately to save space!
                csv_path = os.path.join(extract_to, f"{symbol}-trades-{date_str}.csv")
                if os.path.exists(csv_path):
                    os.remove(csv_path)
                    print(f"Deleted massive raw CSV: {csv_path}")
                
                # Force Garbage Collection to clear RAM
                gc.collect()
            except Exception as e:
                print(f"Error processing/deleting {filename}: {e}")
                
        return True
            
    except Exception as e:
        print(f"Error downloading {filename}: {e}")
        return False

def get_historical_data(symbol: str, year: int, months: list[int], download_trades: bool = True, download_klines: bool = True):
    """
    Downloads monthly historical data from Binance for a specific coin.
    """
    symbol = symbol.upper()
    
    for month in months:
        if not check_storage():
            break
            
        month_str = f"{month:02d}"
        date_str = f"{year}-{month_str}"
        
        # 1. Download Trades (Tick Data for OFI) -> Now processes and deletes CSV!
        if download_trades:
            trades_url = f"{BASE_URL}/trades/{symbol}/{symbol}-trades-{date_str}.zip"
            success = download_extract_and_process(trades_url, symbol, date_str, is_trades=True)
            if not success and not check_storage(): break
            
        # 2. Download Klines (1h Candles for EV Model)
        if download_klines:
            klines_url = f"{BASE_URL}/klines/{symbol}/1h/{symbol}-1h-{year}-{month_str}.zip"
            download_extract_and_process(klines_url, symbol, date_str, is_trades=False)

if __name__ == "__main__":
    print(f"Starting Binance Data Downloader targeting {STORAGE_DIR}")
    
    # --- CONFIGURATION ---
    # Optimal Basket of 20 Coins: Covers Majors, L1s, DeFi, AI, and Memes (including OG)
    TARGET_COINS = [
        "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT",           # The Majors
        "DOGEUSDT", "PEPEUSDT", "SHIBUSDT", "WIFUSDT",        # Memecoin Sector
        "LINKUSDT", "UNIUSDT", "AAVEUSDT",                    # DeFi & Oracles
        "ADAUSDT", "AVAXUSDT", "XRPUSDT", "TONUSDT",          # Layer 1 Alternatives
        "FETUSDT", "RNDRUSDT",                                # AI / Compute
        "OGUSDT", "GALAUSDT",                                 # Gaming / Fan / Culture
        "XMRUSDT"                                             # Privacy
    ]
    
    # Define the schedule: 2023, 2024, 2025, and Jan-Apr of 2026 
    # This gives the model a massive 40-month dataset spanning bear, chop, and bull regimes!
    DOWNLOAD_SCHEDULE = {
        2023: list(range(1, 13)),  # Months 1 through 12
        2024: list(range(1, 13)),  # Months 1 through 12
        2025: list(range(1, 13)),  # Months 1 through 12
        2026: list(range(1, 5))    # Months 1 through 4 (Jan to Apr)
    }
    
    for coin in TARGET_COINS:
        if not check_storage():
            break
        print(f"\n{'='*50}\nSTARTING DOWNLOADS FOR {coin}\n{'='*50}")
        for year, months in DOWNLOAD_SCHEDULE.items():
            if not check_storage():
                break
            print(f"\nQueueing downloads for {coin} ({year}) - Months: {months}")
            get_historical_data(
                symbol=coin,
                year=year,
                months=months,
                download_trades=True,
                download_klines=True
            )
            
    print("\nScript execution finished. Check your E: drive!")
