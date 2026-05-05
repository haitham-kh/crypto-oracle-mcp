from __future__ import annotations
"""
CryptoOracle MCP — MEXC REST Client
Handles: spot ticker, klines, depth, trades
"""

import os
import hmac
import hashlib
import time
from typing import Optional

from utils import fetch_json, safe_float, safe_int

BASE_URL = "https://api.mexc.com"


def _api_key() -> str:
    return os.getenv("MEXC_API_KEY", "")


def _api_secret() -> str:
    return os.getenv("MEXC_API_SECRET", "")


def _headers() -> dict:
    key = _api_key()
    if key:
        return {"X-MEXC-APIKEY": key}
    return {}


# ------------------------------------------------------------------
# Spot — Ticker price
# ------------------------------------------------------------------

async def get_ticker_price(symbol: str) -> dict:
    """GET /api/v3/ticker/price"""
    url = f"{BASE_URL}/api/v3/ticker/price"
    data = await fetch_json(url, source="mexc", params={"symbol": f"{symbol}USDT"})
    return {
        "price": safe_float(data.get("price")),
        "symbol": data.get("symbol", f"{symbol}USDT"),
    }


async def get_ticker_24h(symbol: str) -> dict:
    """GET /api/v3/ticker/24hr"""
    url = f"{BASE_URL}/api/v3/ticker/24hr"
    data = await fetch_json(url, source="mexc", params={"symbol": f"{symbol}USDT"})
    return {
        "price": safe_float(data.get("lastPrice")),
        "price_change_pct": safe_float(data.get("priceChangePercent")),
        "high_24h": safe_float(data.get("highPrice")),
        "low_24h": safe_float(data.get("lowPrice")),
        "volume_24h": safe_float(data.get("volume")),
        "quote_volume_24h": safe_float(data.get("quoteVolume")),
        "count": safe_int(data.get("count")),
    }


# ------------------------------------------------------------------
# Spot — Klines (OHLCV)
# ------------------------------------------------------------------

async def get_klines(
    symbol: str,
    interval: str = "1h",
    limit: int = 500,
) -> list[dict]:
    """
    GET /api/v3/klines
    MEXC uses same kline format as Binance.
    """
    # MEXC interval mapping (slightly different from Binance)
    interval_map = {
        "1m": "1m", "5m": "5m", "15m": "15m",
        "1h": "60m", "4h": "4h", "1d": "1d", "1w": "1W",
    }
    mexc_interval = interval_map.get(interval, interval)

    url = f"{BASE_URL}/api/v3/klines"
    params = {
        "symbol": f"{symbol}USDT",
        "interval": mexc_interval,
        "limit": min(limit, 1000),
    }
    raw = await fetch_json(url, source="mexc", params=params)
    candles = []
    for k in raw:
        candles.append({
            "open_time": int(k[0]),
            "open": safe_float(k[1]),
            "high": safe_float(k[2]),
            "low": safe_float(k[3]),
            "close": safe_float(k[4]),
            "volume": safe_float(k[5]),
            "close_time": int(k[6]),
            "quote_volume": safe_float(k[7]),
            "num_trades": safe_int(k[8]) if len(k) > 8 else 0,
            "taker_buy_volume": safe_float(k[9]) if len(k) > 9 else 0,
            "taker_sell_volume": safe_float(k[5]) - safe_float(k[9]) if len(k) > 9 else 0,
        })
    return candles


# ------------------------------------------------------------------
# Spot — Order book depth
# ------------------------------------------------------------------

async def get_depth(symbol: str, limit: int = 100) -> dict:
    """GET /api/v3/depth"""
    url = f"{BASE_URL}/api/v3/depth"
    params = {"symbol": f"{symbol}USDT", "limit": min(limit, 5000)}
    data = await fetch_json(url, source="mexc", params=params)
    bids = [{"price": safe_float(b[0]), "qty": safe_float(b[1])} for b in data.get("bids", [])]
    asks = [{"price": safe_float(a[0]), "qty": safe_float(a[1])} for a in data.get("asks", [])]
    return {"bids": bids, "asks": asks}


# ------------------------------------------------------------------
# Spot — Recent trades
# ------------------------------------------------------------------

async def get_recent_trades(symbol: str, limit: int = 500) -> list[dict]:
    """GET /api/v3/trades"""
    url = f"{BASE_URL}/api/v3/trades"
    params = {"symbol": f"{symbol}USDT", "limit": min(limit, 1000)}
    raw = await fetch_json(url, source="mexc", params=params)
    trades = []
    for t in raw:
        trades.append({
            "id": t.get("id"),
            "price": safe_float(t.get("price")),
            "qty": safe_float(t.get("qty")),
            "quote_qty": safe_float(t.get("quoteQty")),
            "time": t.get("time"),
            "is_buyer_maker": t.get("isBuyerMaker", False),
        })
    return trades
