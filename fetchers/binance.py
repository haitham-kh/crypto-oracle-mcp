from __future__ import annotations
"""
CryptoOracle MCP — Binance REST Client
Handles: spot ticker, klines, depth, trades, futures funding/OI
"""

import os
import hmac
import hashlib
import time
from typing import Optional, Literal

from utils import fetch_json, safe_float, safe_int

SPOT_BASE = "https://api.binance.com"
FUTURES_BASE = "https://fapi.binance.com"


def _api_key() -> str:
    return os.getenv("BINANCE_API_KEY", "")


def _api_secret() -> str:
    return os.getenv("BINANCE_API_SECRET", "")


def _headers() -> dict:
    key = _api_key()
    if key:
        return {"X-MBX-APIKEY": key}
    return {}


# ------------------------------------------------------------------
# Spot — Ticker price
# ------------------------------------------------------------------

async def get_ticker_price(symbol: str) -> dict:
    """GET /api/v3/ticker/price  —  single symbol."""
    url = f"{SPOT_BASE}/api/v3/ticker/price"
    data = await fetch_json(url, source="binance", params={"symbol": f"{symbol}USDT"})
    return {
        "price": safe_float(data.get("price")),
        "symbol": data.get("symbol", f"{symbol}USDT"),
    }


async def get_ticker_24h(symbol: str) -> dict:
    """GET /api/v3/ticker/24hr  —  24h stats."""
    url = f"{SPOT_BASE}/api/v3/ticker/24hr"
    data = await fetch_json(url, source="binance", params={"symbol": f"{symbol}USDT"})
    return {
        "price": safe_float(data.get("lastPrice")),
        "price_change_pct": safe_float(data.get("priceChangePercent")),
        "high_24h": safe_float(data.get("highPrice")),
        "low_24h": safe_float(data.get("lowPrice")),
        "volume_24h": safe_float(data.get("volume")),
        "quote_volume_24h": safe_float(data.get("quoteVolume")),
        "weighted_avg_price": safe_float(data.get("weightedAvgPrice")),
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
    Returns list of candle dicts.
    """
    url = f"{SPOT_BASE}/api/v3/klines"
    params = {
        "symbol": f"{symbol}USDT",
        "interval": interval,
        "limit": min(limit, 1000),
    }
    raw = await fetch_json(url, source="binance", params=params)
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
            "num_trades": safe_int(k[8]),
            "taker_buy_volume": safe_float(k[9]),
            "taker_sell_volume": safe_float(k[5]) - safe_float(k[9]),  # total vol - taker buy
        })
    return candles


# ------------------------------------------------------------------
# Spot — Order book depth
# ------------------------------------------------------------------

async def get_depth(symbol: str, limit: int = 100) -> dict:
    """GET /api/v3/depth"""
    url = f"{SPOT_BASE}/api/v3/depth"
    params = {"symbol": f"{symbol}USDT", "limit": min(limit, 5000)}
    data = await fetch_json(url, source="binance", params=params)
    bids = [{"price": safe_float(b[0]), "qty": safe_float(b[1])} for b in data.get("bids", [])]
    asks = [{"price": safe_float(a[0]), "qty": safe_float(a[1])} for a in data.get("asks", [])]
    return {"bids": bids, "asks": asks}


# ------------------------------------------------------------------
# Spot — Recent trades
# ------------------------------------------------------------------

async def get_recent_trades(symbol: str, limit: int = 500) -> list[dict]:
    """GET /api/v3/trades"""
    url = f"{SPOT_BASE}/api/v3/trades"
    params = {"symbol": f"{symbol}USDT", "limit": min(limit, 1000)}
    raw = await fetch_json(url, source="binance", params=params)
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


# ------------------------------------------------------------------
# Futures — Funding rate
# ------------------------------------------------------------------

async def get_funding_rate(symbol: str, limit: int = 8) -> list[dict]:
    """GET /fapi/v1/fundingRate  —  last N funding periods."""
    url = f"{FUTURES_BASE}/fapi/v1/fundingRate"
    params = {"symbol": f"{symbol}USDT", "limit": limit}
    raw = await fetch_json(url, source="binance", params=params)
    return [
        {
            "symbol": r.get("symbol"),
            "funding_rate": safe_float(r.get("fundingRate")),
            "funding_time": r.get("fundingTime"),
            "mark_price": safe_float(r.get("markPrice")),
        }
        for r in raw
    ]


# ------------------------------------------------------------------
# Futures — Open interest
# ------------------------------------------------------------------

async def get_open_interest(symbol: str) -> dict:
    """GET /fapi/v1/openInterest"""
    url = f"{FUTURES_BASE}/fapi/v1/openInterest"
    params = {"symbol": f"{symbol}USDT"}
    data = await fetch_json(url, source="binance", params=params)
    return {
        "symbol": data.get("symbol"),
        "open_interest": safe_float(data.get("openInterest")),
        "time": data.get("time"),
    }


# ------------------------------------------------------------------
# Futures — Long/Short ratio (top traders)
# ------------------------------------------------------------------

async def get_top_long_short_ratio(symbol: str, period: str = "1h", limit: int = 1) -> dict:
    """GET /futures/data/topLongShortAccountRatio"""
    url = f"{FUTURES_BASE}/futures/data/topLongShortAccountRatio"
    params = {"symbol": f"{symbol}USDT", "period": period, "limit": limit}
    raw = await fetch_json(url, source="binance", params=params)
    if raw:
        latest = raw[0]
        return {
            "long_short_ratio": safe_float(latest.get("longShortRatio")),
            "long_account": safe_float(latest.get("longAccount")),
            "short_account": safe_float(latest.get("shortAccount")),
            "timestamp": latest.get("timestamp"),
        }
    return {"long_short_ratio": None, "long_account": None, "short_account": None, "timestamp": None}


# ------------------------------------------------------------------
# Futures — Global Long/Short ratio
# ------------------------------------------------------------------

async def get_global_long_short_ratio(symbol: str, period: str = "1h", limit: int = 1) -> dict:
    """GET /futures/data/globalLongShortAccountRatio"""
    url = f"{FUTURES_BASE}/futures/data/globalLongShortAccountRatio"
    params = {"symbol": f"{symbol}USDT", "period": period, "limit": limit}
    raw = await fetch_json(url, source="binance", params=params)
    if raw:
        latest = raw[0]
        return {
            "long_short_ratio": safe_float(latest.get("longShortRatio")),
            "long_account": safe_float(latest.get("longAccount")),
            "short_account": safe_float(latest.get("shortAccount")),
            "timestamp": latest.get("timestamp"),
        }
    return {"long_short_ratio": None, "long_account": None, "short_account": None, "timestamp": None}


# ------------------------------------------------------------------
# Futures — Open Interest History (for OI delta tracking)
# ------------------------------------------------------------------

async def get_oi_history(symbol: str, period: str = "5m", limit: int = 288) -> list[dict]:
    """GET /futures/data/openInterestHist — OI over time for delta analysis.
    period=5m, limit=288 gives 24h of data.
    """
    url = f"{FUTURES_BASE}/futures/data/openInterestHist"
    params = {"symbol": f"{symbol}USDT", "period": period, "limit": limit}
    raw = await fetch_json(url, source="binance", params=params)
    return [
        {
            "timestamp": r.get("timestamp"),
            "sum_open_interest": safe_float(r.get("sumOpenInterest")),
            "sum_open_interest_value": safe_float(r.get("sumOpenInterestValue")),
        }
        for r in raw
    ]


# ------------------------------------------------------------------
# Futures — Taker Buy/Sell Volume Ratio
# ------------------------------------------------------------------

async def get_taker_volume_ratio(symbol: str, period: str = "5m", limit: int = 48) -> list[dict]:
    """GET /futures/data/takerlongshortRatio — taker buy vs sell ratio."""
    url = f"{FUTURES_BASE}/futures/data/takerlongshortRatio"
    params = {"symbol": f"{symbol}USDT", "period": period, "limit": limit}
    raw = await fetch_json(url, source="binance", params=params)
    return [
        {
            "buy_sell_ratio": safe_float(r.get("buySellRatio")),
            "buy_vol": safe_float(r.get("buyVol")),
            "sell_vol": safe_float(r.get("sellVol")),
            "timestamp": r.get("timestamp"),
        }
        for r in raw
    ]
