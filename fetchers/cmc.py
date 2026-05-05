from __future__ import annotations
"""
CryptoOracle MCP — CoinMarketCap REST Client
Handles: coin map/search, global metrics, quotes
"""

import os
from typing import Optional

from utils import fetch_json, safe_float, safe_int

BASE_URL = "https://pro-api.coinmarketcap.com"


def _headers() -> dict:
    key = os.getenv("CMC_API_KEY", "")
    return {
        "X-CMC_PRO_API_KEY": key,
        "Accept": "application/json",
    }


# ------------------------------------------------------------------
# Cryptocurrency map — resolve names to CMC IDs
# ------------------------------------------------------------------

async def search_coins(query: str) -> list[dict]:
    """
    GET /v1/cryptocurrency/map
    Resolve symbol/name to CMC canonical IDs.
    """
    url = f"{BASE_URL}/v1/cryptocurrency/map"
    params = {"symbol": query.upper(), "limit": 10}
    try:
        data = await fetch_json(url, source="cmc", params=params, headers=_headers())
        coins = []
        for c in data.get("data", []):
            coins.append({
                "id": c.get("id"),
                "name": c.get("name"),
                "symbol": c.get("symbol"),
                "slug": c.get("slug"),
                "rank": c.get("rank"),
                "is_active": c.get("is_active"),
                "platform": c.get("platform"),
                "first_historical_data": c.get("first_historical_data"),
            })
        return coins
    except Exception:
        # Fallback: try listing endpoint
        return await _search_by_listing(query)


async def _search_by_listing(query: str) -> list[dict]:
    """Fallback search using listings endpoint."""
    url = f"{BASE_URL}/v1/cryptocurrency/listings/latest"
    params = {"start": 1, "limit": 200, "convert": "USD"}
    data = await fetch_json(url, source="cmc", params=params, headers=_headers())
    coins = []
    query_upper = query.upper()
    for c in data.get("data", []):
        if query_upper in c.get("symbol", "").upper() or query_upper in c.get("name", "").upper():
            coins.append({
                "id": c.get("id"),
                "name": c.get("name"),
                "symbol": c.get("symbol"),
                "slug": c.get("slug"),
                "rank": c.get("cmc_rank"),
            })
    return coins


# ------------------------------------------------------------------
# Quotes (latest)
# ------------------------------------------------------------------

async def get_quotes(cmc_id: int) -> dict:
    """
    GET /v2/cryptocurrency/quotes/latest
    Full price + market data for a coin by CMC ID.
    """
    url = f"{BASE_URL}/v2/cryptocurrency/quotes/latest"
    params = {"id": str(cmc_id), "convert": "USD"}
    data = await fetch_json(url, source="cmc", params=params, headers=_headers())

    coin_data = data.get("data", {}).get(str(cmc_id), {})
    quote = coin_data.get("quote", {}).get("USD", {})

    return {
        "id": coin_data.get("id"),
        "name": coin_data.get("name"),
        "symbol": coin_data.get("symbol"),
        "slug": coin_data.get("slug"),
        "cmc_rank": coin_data.get("cmc_rank"),
        "circulating_supply": safe_float(coin_data.get("circulating_supply")),
        "total_supply": safe_float(coin_data.get("total_supply")),
        "max_supply": safe_float(coin_data.get("max_supply")),
        "tags": coin_data.get("tags", []),
        "price_usd": safe_float(quote.get("price")),
        "volume_24h": safe_float(quote.get("volume_24h")),
        "volume_change_24h": safe_float(quote.get("volume_change_24h")),
        "pct_change_1h": safe_float(quote.get("percent_change_1h")),
        "pct_change_24h": safe_float(quote.get("percent_change_24h")),
        "pct_change_7d": safe_float(quote.get("percent_change_7d")),
        "pct_change_30d": safe_float(quote.get("percent_change_30d")),
        "market_cap": safe_float(quote.get("market_cap")),
        "fully_diluted_market_cap": safe_float(quote.get("fully_diluted_market_cap")),
        "market_cap_dominance": safe_float(quote.get("market_cap_dominance")),
    }


# ------------------------------------------------------------------
# Global metrics
# ------------------------------------------------------------------

async def get_global_metrics() -> dict:
    """GET /v1/global-metrics/quotes/latest — global crypto market overview."""
    url = f"{BASE_URL}/v1/global-metrics/quotes/latest"
    params = {"convert": "USD"}
    data = await fetch_json(url, source="cmc", params=params, headers=_headers())
    d = data.get("data", {})
    quote = d.get("quote", {}).get("USD", {})

    return {
        "active_cryptocurrencies": safe_int(d.get("active_cryptocurrencies")),
        "active_exchanges": safe_int(d.get("active_exchanges")),
        "btc_dominance": safe_float(d.get("btc_dominance")),
        "eth_dominance": safe_float(d.get("eth_dominance")),
        "defi_volume_24h": safe_float(d.get("defi_volume_24h")),
        "defi_market_cap": safe_float(d.get("defi_market_cap")),
        "total_market_cap": safe_float(quote.get("total_market_cap")),
        "total_volume_24h": safe_float(quote.get("total_volume_24h")),
        "altcoin_market_cap": safe_float(quote.get("altcoin_market_cap")),
        "altcoin_volume_24h": safe_float(quote.get("altcoin_volume_24h")),
    }
