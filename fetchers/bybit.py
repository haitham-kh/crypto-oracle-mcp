from __future__ import annotations
"""
CryptoOracle MCP — Bybit REST Client
Handles: Cross-exchange funding rates for funding spread analysis
"""

from utils import fetch_json, safe_float

BASE_URL = "https://api.bybit.com/v5"


async def get_funding_rate(symbol: str, limit: int = 8) -> list[dict]:
    """GET /v5/market/funding/history — last N funding periods."""
    url = f"{BASE_URL}/market/funding/history"
    params = {"category": "linear", "symbol": f"{symbol}USDT", "limit": limit}
    try:
        data = await fetch_json(url, source="default", params=params)
        result = data.get("result", {}).get("list", [])
        return [
            {
                "symbol": r.get("symbol"),
                "funding_rate": safe_float(r.get("fundingRate")),
                "funding_time": r.get("fundingRateTimestamp"),
            }
            for r in result
        ]
    except Exception:
        return []


async def get_tickers(symbol: str) -> dict:
    """GET /v5/market/tickers — price and OI from Bybit."""
    url = f"{BASE_URL}/market/tickers"
    params = {"category": "linear", "symbol": f"{symbol}USDT"}
    try:
        data = await fetch_json(url, source="default", params=params)
        items = data.get("result", {}).get("list", [])
        if items:
            t = items[0]
            return {
                "price": safe_float(t.get("lastPrice")),
                "open_interest": safe_float(t.get("openInterest")),
                "open_interest_value": safe_float(t.get("openInterestValue")),
                "funding_rate": safe_float(t.get("fundingRate")),
                "next_funding_time": t.get("nextFundingTime"),
            }
        return {}
    except Exception:
        return {}
