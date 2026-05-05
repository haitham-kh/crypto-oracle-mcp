from __future__ import annotations
"""
CryptoOracle MCP — Coinglass REST Client
Handles: Liquidation heatmap data, aggregated OI
"""

import os
from utils import fetch_json, safe_float

BASE_URL = "https://open-api.coinglass.com/public/v2"


def _headers() -> dict:
    key = os.getenv("COINGLASS_API_KEY", "")
    return {"coinglassSecret": key, "accept": "application/json"} if key else {}


def _has_key() -> bool:
    return bool(os.getenv("COINGLASS_API_KEY", ""))


async def get_liquidation_map(symbol: str) -> dict:
    """Fetch liquidation heatmap / aggregated liquidation data."""
    if not _has_key():
        return {"source": "unavailable", "note": "COINGLASS_API_KEY not set"}
    url = f"{BASE_URL}/liquidation_info"
    params = {"symbol": symbol.upper(), "time_type": 2}  # 24h
    try:
        data = await fetch_json(url, source="default", params=params, headers=_headers())
        result = data.get("data", {})
        return {
            "source": "coinglass",
            "symbol": symbol.upper(),
            "long_liquidations_24h_usd": safe_float(result.get("longLiqUsd")),
            "short_liquidations_24h_usd": safe_float(result.get("shortLiqUsd")),
            "total_liquidations_24h_usd": safe_float(result.get("totalLiqUsd")),
            "long_liq_count": result.get("longLiqCount"),
            "short_liq_count": result.get("shortLiqCount"),
            "liquidation_ratio": round(
                safe_float(result.get("longLiqUsd")) /
                max(safe_float(result.get("shortLiqUsd")), 1), 2
            ),
        }
    except Exception as e:
        return {"source": "coinglass", "error": str(e)}


async def get_liquidation_levels(symbol: str) -> dict:
    """Estimated liquidation levels / clusters."""
    if not _has_key():
        return {"source": "unavailable", "note": "COINGLASS_API_KEY not set"}
    url = f"{BASE_URL}/liquidation_order"
    params = {"symbol": symbol.upper(), "time_type": 4}  # 7d
    try:
        data = await fetch_json(url, source="default", params=params, headers=_headers())
        prices = data.get("data", {}).get("priceList", [])
        liq_long = data.get("data", {}).get("longList", [])
        liq_short = data.get("data", {}).get("shortList", [])
        # Find clusters: top 5 levels with highest liquidation volume
        long_clusters = []
        short_clusters = []
        if prices and liq_long:
            combined = list(zip(prices, liq_long))
            combined.sort(key=lambda x: abs(safe_float(x[1])), reverse=True)
            long_clusters = [{"price": safe_float(p), "volume_usd": safe_float(v)}
                           for p, v in combined[:5]]
        if prices and liq_short:
            combined = list(zip(prices, liq_short))
            combined.sort(key=lambda x: abs(safe_float(x[1])), reverse=True)
            short_clusters = [{"price": safe_float(p), "volume_usd": safe_float(v)}
                            for p, v in combined[:5]]
        return {
            "source": "coinglass",
            "long_liquidation_clusters": long_clusters,
            "short_liquidation_clusters": short_clusters,
        }
    except Exception as e:
        return {"source": "coinglass", "error": str(e)}


async def get_aggregated_oi(symbol: str) -> dict:
    """Aggregated open interest across all exchanges."""
    if not _has_key():
        return {"source": "unavailable"}
    url = f"{BASE_URL}/open_interest"
    params = {"symbol": symbol.upper(), "time_type": 0}
    try:
        data = await fetch_json(url, source="default", params=params, headers=_headers())
        result = data.get("data", [])
        if isinstance(result, list) and result:
            latest = result[0] if isinstance(result[0], dict) else {}
            return {
                "source": "coinglass",
                "aggregated_oi_usd": safe_float(latest.get("openInterest")),
                "oi_change_24h_pct": safe_float(latest.get("h24Change")),
            }
        return {"source": "coinglass", "aggregated_oi_usd": None}
    except Exception as e:
        return {"source": "coinglass", "error": str(e)}
