from __future__ import annotations
"""
CryptoOracle MCP — CoinGecko REST Client
Handles: search, metadata, price, tickers, global, market chart
"""

import os
from typing import Optional

from utils import fetch_json, safe_float, safe_int

BASE_URL = "https://api.coingecko.com/api/v3"


def _headers() -> dict:
    key = os.getenv("COINGECKO_API_KEY", "")
    if key:
        return {"x-cg-demo-api-key": key, "accept": "application/json"}
    return {"accept": "application/json"}


# ------------------------------------------------------------------
# Search
# ------------------------------------------------------------------

async def search_coins(query: str) -> list[dict]:
    """GET /search — search coins by query string."""
    url = f"{BASE_URL}/search"
    data = await fetch_json(url, source="coingecko", params={"query": query}, headers=_headers())
    coins = []
    for c in data.get("coins", []):
        coins.append({
            "id": c.get("id"),
            "name": c.get("name"),
            "symbol": c.get("symbol"),
            "market_cap_rank": c.get("market_cap_rank"),
            "thumb": c.get("thumb"),
            "large": c.get("large"),
        })
    return coins


# ------------------------------------------------------------------
# Coin detail / metadata
# ------------------------------------------------------------------

async def get_coin_detail(coingecko_id: str) -> dict:
    """
    GET /coins/{id}
    Full coin metadata including community, developer, links.
    """
    url = f"{BASE_URL}/coins/{coingecko_id}"
    params = {
        "localization": "false",
        "tickers": "false",
        "market_data": "true",
        "community_data": "true",
        "developer_data": "true",
        "sparkline": "false",
    }
    data = await fetch_json(url, source="coingecko", params=params, headers=_headers())

    links = data.get("links", {})
    dev = data.get("developer_data", {})
    community = data.get("community_data", {})
    market = data.get("market_data", {})

    return {
        "id": data.get("id"),
        "symbol": data.get("symbol"),
        "name": data.get("name"),
        "description": (data.get("description", {}).get("en") or "")[:500],
        "categories": data.get("categories", []),
        "hashing_algorithm": data.get("hashing_algorithm"),
        "genesis_date": data.get("genesis_date"),
        "sentiment_votes_up_percentage": safe_float(data.get("sentiment_votes_up_percentage")),
        "market_cap_rank": data.get("market_cap_rank"),
        "links": {
            "homepage": links.get("homepage", [None])[0],
            "twitter": f"https://twitter.com/{links.get('twitter_screen_name')}" if links.get("twitter_screen_name") else None,
            "reddit": links.get("subreddit_url"),
            "telegram": links.get("telegram_channel_identifier"),
            "github": links.get("repos_url", {}).get("github", [None])[0] if links.get("repos_url") else None,
            "whitepaper": data.get("links", {}).get("whitepaper"),
        },
        "contract_addresses": data.get("platforms", {}),
        "dev_data": {
            "stars": safe_int(dev.get("stars")),
            "forks": safe_int(dev.get("forks")),
            "subscribers": safe_int(dev.get("subscribers")),
            "total_issues": safe_int(dev.get("total_issues")),
            "closed_issues": safe_int(dev.get("closed_issues")),
            "pull_requests_merged": safe_int(dev.get("pull_requests_merged")),
            "commit_count_4_weeks": safe_int(dev.get("commit_count_4_weeks")),
        },
        "community": {
            "twitter_followers": safe_int(community.get("twitter_followers")),
            "reddit_subscribers": safe_int(community.get("reddit_subscribers")),
            "reddit_accounts_active_48h": safe_int(community.get("reddit_accounts_active_48h")),
            "telegram_channel_user_count": safe_int(community.get("telegram_channel_user_count")),
        },
        "market_data": {
            "current_price_usd": safe_float(market.get("current_price", {}).get("usd")),
            "market_cap_usd": safe_float(market.get("market_cap", {}).get("usd")),
            "fully_diluted_valuation": safe_float(market.get("fully_diluted_valuation", {}).get("usd")),
            "total_volume_usd": safe_float(market.get("total_volume", {}).get("usd")),
            "circulating_supply": safe_float(market.get("circulating_supply")),
            "total_supply": safe_float(market.get("total_supply")),
            "max_supply": safe_float(market.get("max_supply")),
            "ath_usd": safe_float(market.get("ath", {}).get("usd")),
            "ath_change_pct": safe_float(market.get("ath_change_percentage", {}).get("usd")),
            "atl_usd": safe_float(market.get("atl", {}).get("usd")),
            "price_change_1h_pct": safe_float(market.get("price_change_percentage_1h_in_currency", {}).get("usd")),
            "price_change_24h_pct": safe_float(market.get("price_change_percentage_24h")),
            "price_change_7d_pct": safe_float(market.get("price_change_percentage_7d")),
            "price_change_30d_pct": safe_float(market.get("price_change_percentage_30d")),
        },
    }


# ------------------------------------------------------------------
# Simple price
# ------------------------------------------------------------------

async def get_simple_price(coingecko_id: str) -> dict:
    """GET /simple/price — quick price lookup."""
    url = f"{BASE_URL}/simple/price"
    params = {
        "ids": coingecko_id,
        "vs_currencies": "usd",
        "include_24hr_change": "true",
        "include_24hr_vol": "true",
        "include_market_cap": "true",
    }
    data = await fetch_json(url, source="coingecko", params=params, headers=_headers())
    coin_data = data.get(coingecko_id, {})
    return {
        "price_usd": safe_float(coin_data.get("usd")),
        "change_24h_pct": safe_float(coin_data.get("usd_24h_change")),
        "volume_24h": safe_float(coin_data.get("usd_24h_vol")),
        "market_cap": safe_float(coin_data.get("usd_market_cap")),
    }


# ------------------------------------------------------------------
# Market chart (historical prices for correlation)
# ------------------------------------------------------------------

async def get_market_chart(coingecko_id: str, days: int = 30) -> dict:
    """GET /coins/{id}/market_chart  —  daily prices for correlation."""
    url = f"{BASE_URL}/coins/{coingecko_id}/market_chart"
    params = {"vs_currency": "usd", "days": days, "interval": "daily"}
    data = await fetch_json(url, source="coingecko", params=params, headers=_headers())
    return {
        "prices": data.get("prices", []),       # [[timestamp, price], ...]
        "volumes": data.get("total_volumes", []),
        "market_caps": data.get("market_caps", []),
    }


# ------------------------------------------------------------------
# Tickers (exchange listings)
# ------------------------------------------------------------------

async def get_coin_tickers(coingecko_id: str) -> list[dict]:
    """GET /coins/{id}/tickers — exchange listing data."""
    url = f"{BASE_URL}/coins/{coingecko_id}/tickers"
    params = {"include_exchange_logo": "false", "depth": "false"}
    data = await fetch_json(url, source="coingecko", params=params, headers=_headers())
    tickers = []
    for t in data.get("tickers", []):
        tickers.append({
            "exchange": t.get("market", {}).get("name"),
            "exchange_id": t.get("market", {}).get("identifier"),
            "pair": f"{t.get('base')}/{t.get('target')}",
            "price_usd": safe_float(t.get("converted_last", {}).get("usd")),
            "volume_24h_usd": safe_float(t.get("converted_volume", {}).get("usd")),
            "trust_score": t.get("trust_score"),
            "bid_ask_spread_pct": safe_float(t.get("bid_ask_spread_percentage")),
            "last_traded_at": t.get("last_traded_at"),
        })
    return tickers


# ------------------------------------------------------------------
# Global market data
# ------------------------------------------------------------------

async def get_global_data() -> dict:
    """GET /global — total market cap, BTC dominance, etc."""
    url = f"{BASE_URL}/global"
    data = await fetch_json(url, source="coingecko", headers=_headers())
    g = data.get("data", {})
    return {
        "active_cryptocurrencies": safe_int(g.get("active_cryptocurrencies")),
        "total_market_cap_usd": safe_float(g.get("total_market_cap", {}).get("usd")),
        "total_volume_24h_usd": safe_float(g.get("total_volume", {}).get("usd")),
        "btc_dominance_pct": safe_float(g.get("market_cap_percentage", {}).get("btc")),
        "eth_dominance_pct": safe_float(g.get("market_cap_percentage", {}).get("eth")),
        "market_cap_change_24h_pct": safe_float(g.get("market_cap_change_percentage_24h_usd")),
    }


# ------------------------------------------------------------------
# Trending coins
# ------------------------------------------------------------------

async def get_trending() -> list[dict]:
    """GET /search/trending — trending coins on CoinGecko."""
    url = f"{BASE_URL}/search/trending"
    data = await fetch_json(url, source="coingecko", headers=_headers())
    trending = []
    for item in data.get("coins", []):
        c = item.get("item", {})
        trending.append({
            "id": c.get("id"),
            "name": c.get("name"),
            "symbol": c.get("symbol"),
            "market_cap_rank": c.get("market_cap_rank"),
            "score": c.get("score"),
        })
    return trending
