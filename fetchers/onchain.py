from __future__ import annotations
"""
CryptoOracle MCP — On-Chain & Explorer Clients
Handles: Glassnode metrics, Etherscan/BSCScan/Solscan top holders,
         Fear & Greed index
"""

import os
from typing import Optional, Literal

from utils import fetch_json, safe_float, safe_int

# ------------------------------------------------------------------
# Fear & Greed Index (FREE — no API key needed)
# ------------------------------------------------------------------

async def get_fear_greed(limit: int = 30) -> dict:
    """
    https://api.alternative.me/fng/?limit=N
    Returns current + historical fear/greed index values.
    """
    url = "https://api.alternative.me/fng/"
    data = await fetch_json(url, source="default", params={"limit": limit, "format": "json"})
    entries = data.get("data", [])
    if not entries:
        return {"current_value": None, "current_label": None}

    current = entries[0]
    yesterday = entries[1] if len(entries) > 1 else {}
    week_ago = entries[7] if len(entries) > 7 else {}

    values = [int(e.get("value", 0)) for e in entries if e.get("value")]
    avg_30d = sum(values) / len(values) if values else 0

    # Determine trend
    recent_5 = values[:5] if len(values) >= 5 else values
    if len(recent_5) >= 2:
        if recent_5[0] > recent_5[-1] + 5:
            trend = "rising"
        elif recent_5[0] < recent_5[-1] - 5:
            trend = "falling"
        else:
            trend = "stable"
    else:
        trend = "unknown"

    current_val = int(current.get("value", 0))
    if current_val < 20:
        interpretation = "Extreme Fear — historically a good buying zone for quality assets"
    elif current_val < 40:
        interpretation = "Fear — market is cautious, potential opportunity for contrarians"
    elif current_val < 60:
        interpretation = "Neutral — no strong sentiment bias"
    elif current_val < 80:
        interpretation = "Greed — market is optimistic, exercise caution"
    else:
        interpretation = "Extreme Greed — elevated risk, historically precedes corrections"

    return {
        "current_value": current_val,
        "current_label": current.get("value_classification", ""),
        "yesterday_value": int(yesterday.get("value", 0)),
        "last_week_value": int(week_ago.get("value", 0)),
        "avg_30d": round(avg_30d, 1),
        "historical_trend": trend,
        "interpretation": interpretation,
        "historical": [
            {"value": int(e.get("value", 0)), "label": e.get("value_classification", ""), "timestamp": e.get("timestamp")}
            for e in entries[:30]
        ],
    }


# ------------------------------------------------------------------
# Glassnode (optional — requires API key)
# ------------------------------------------------------------------

GLASSNODE_BASE = "https://api.glassnode.com/v1/metrics"


def _glassnode_key() -> str:
    return os.getenv("GLASSNODE_API_KEY", "")


async def get_glassnode_metric(asset: str, metric_path: str, resolution: str = "24h") -> Optional[list]:
    """
    Generic Glassnode metric fetcher.
    metric_path example: "addresses/active_count"
    """
    key = _glassnode_key()
    if not key:
        return None

    url = f"{GLASSNODE_BASE}/{metric_path}"
    params = {
        "a": asset.upper(),
        "s": "30d",
        "i": resolution,
        "api_key": key,
    }
    try:
        data = await fetch_json(url, source="glassnode", params=params)
        return data
    except Exception:
        return None


async def get_onchain_summary(symbol: str) -> dict:
    """Aggregate several Glassnode metrics into a summary."""
    key = _glassnode_key()
    if not key:
        return {
            "source": "unavailable",
            "note": "Glassnode API key not configured. Using CoinGecko dev data as proxy.",
            "active_addresses_24h": None,
            "transaction_count_24h": None,
            "nvt_ratio": None,
            "exchange_reserve": None,
            "hash_rate": None,
        }

    results = {}

    # Active addresses
    active = await get_glassnode_metric(symbol, "addresses/active_count")
    if active and len(active) > 0:
        results["active_addresses_24h"] = active[-1].get("v")
    else:
        results["active_addresses_24h"] = None

    # Transaction count
    txn = await get_glassnode_metric(symbol, "transactions/count")
    if txn and len(txn) > 0:
        results["transaction_count_24h"] = txn[-1].get("v")
    else:
        results["transaction_count_24h"] = None

    # NVT ratio
    nvt = await get_glassnode_metric(symbol, "indicators/nvt")
    if nvt and len(nvt) > 0:
        results["nvt_ratio"] = nvt[-1].get("v")
    else:
        results["nvt_ratio"] = None

    # Exchange reserve / net flows
    flows = await get_glassnode_metric(symbol, "distribution/exchange_net_position_change")
    if flows and len(flows) > 0:
        results["exchange_net_flow"] = flows[-1].get("v")
    else:
        results["exchange_net_flow"] = None

    # Hash rate (PoW only)
    hr = await get_glassnode_metric(symbol, "mining/hash_rate_mean")
    if hr and len(hr) > 0:
        results["hash_rate"] = hr[-1].get("v")
    else:
        results["hash_rate"] = None

    results["source"] = "glassnode"
    return results


# ------------------------------------------------------------------
# Block Explorer — Top Holders
# ------------------------------------------------------------------

EXPLORER_APIS = {
    "eth": {
        "base_url": "https://api.etherscan.io/api",
        "key_env": "ETHERSCAN_API_KEY",
    },
    "bsc": {
        "base_url": "https://api.bscscan.com/api",
        "key_env": "BSCSCAN_API_KEY",
    },
    "polygon": {
        "base_url": "https://api.polygonscan.com/api",
        "key_env": "POLYGONSCAN_API_KEY",
    },
    "arb": {
        "base_url": "https://api.arbiscan.io/api",
        "key_env": "ARBISCAN_API_KEY",
    },
}


async def get_top_token_holders(
    contract_address: str,
    chain: Literal["eth", "bsc", "sol", "polygon", "arb"] = "eth",
) -> dict:
    """
    Get top token holders for ERC-20 / BEP-20 tokens.
    For Solana, uses a different approach.
    """
    if chain == "sol":
        return await _get_solana_top_holders(contract_address)

    explorer = EXPLORER_APIS.get(chain)
    if not explorer:
        return {"error": f"Unsupported chain: {chain}", "top_wallets": []}

    api_key = os.getenv(explorer["key_env"], "")
    if not api_key:
        return {
            "error": f"No API key for {chain} explorer",
            "note": "Set the appropriate explorer API key in .env",
            "top_wallets": [],
        }

    # Etherscan-like API for token holder list
    # Note: Free tier may not support this endpoint — graceful fallback
    url = explorer["base_url"]
    params = {
        "module": "token",
        "action": "tokenholderlist",
        "contractaddress": contract_address,
        "page": 1,
        "offset": 20,
        "apikey": api_key,
    }

    try:
        data = await fetch_json(url, source="default", params=params)
        holders = data.get("result", [])
        if isinstance(holders, str):
            # API returned error string
            return {
                "error": holders,
                "note": "Token holder list may require paid API plan",
                "top_wallets": [],
            }

        top_wallets = []
        for i, h in enumerate(holders[:20], 1):
            top_wallets.append({
                "rank": i,
                "address": h.get("TokenHolderAddress", ""),
                "quantity": safe_float(h.get("TokenHolderQuantity", 0)),
            })

        return {
            "chain": chain,
            "contract": contract_address,
            "top_wallets": top_wallets,
            "source": f"{chain}scan",
        }
    except Exception as e:
        return {
            "error": str(e),
            "top_wallets": [],
        }


async def _get_solana_top_holders(mint_address: str) -> dict:
    """
    Get top holders for a Solana SPL token.
    Uses the public Solana RPC or Solscan API.
    """
    # Solscan public API
    url = f"https://public-api.solscan.io/token/holders"
    params = {"tokenAddress": mint_address, "limit": 20, "offset": 0}

    try:
        data = await fetch_json(url, source="default", params=params)
        holders_data = data.get("data", []) if isinstance(data, dict) else data
        top_wallets = []
        for i, h in enumerate(holders_data[:20], 1):
            top_wallets.append({
                "rank": i,
                "address": h.get("owner", h.get("address", "")),
                "amount": safe_float(h.get("amount", 0)),
                "decimals": h.get("decimals", 0),
            })
        return {
            "chain": "sol",
            "contract": mint_address,
            "top_wallets": top_wallets,
            "source": "solscan",
        }
    except Exception as e:
        return {
            "error": str(e),
            "top_wallets": [],
        }


# ------------------------------------------------------------------
# Stablecoin Flows — DefiLlama (FREE, no API key)
# ------------------------------------------------------------------

async def get_stablecoin_flows() -> dict:
    """Track stablecoin supply changes — macro fuel gauge for crypto market.
    Rising stablecoin supply = dry powder for buying.
    Falling = less fuel for rallies.
    """
    url = "https://stablecoins.llama.fi/stablecoins?includePrices=true"
    try:
        data = await fetch_json(url, source="default")
        stables = data.get("peggedAssets", [])
        # Focus on top stablecoins
        target_symbols = {"USDT", "USDC", "DAI", "BUSD", "TUSD", "FDUSD"}
        total_mcap = 0
        total_mcap_7d_ago = 0
        breakdown = []
        for s in stables:
            sym = s.get("symbol", "")
            if sym not in target_symbols:
                continue
            chains = s.get("chainCirculating", {})
            current = sum(
                safe_float((v or {}).get("current", {}).get("peggedUSD"))
                for v in chains.values()
            )
            # 7d change from circulating history if available
            circ = s.get("circulatingPrevDay", {}).get("peggedUSD")
            prev_7d = safe_float(circ) if circ else current
            total_mcap += current
            total_mcap_7d_ago += prev_7d
            breakdown.append({
                "symbol": sym,
                "name": s.get("name"),
                "market_cap": round(current, 0),
            })
        pct_change = ((total_mcap - total_mcap_7d_ago) / total_mcap_7d_ago * 100
                      if total_mcap_7d_ago > 0 else 0)
        return {
            "source": "defillama",
            "total_stablecoin_mcap": round(total_mcap, 0),
            "mcap_change_pct": round(pct_change, 2),
            "fuel_gauge": "expanding" if pct_change > 0.5 else ("contracting" if pct_change < -0.5 else "stable"),
            "interpretation": (
                "Stablecoin supply growing — dry powder increasing, bullish macro"
                if pct_change > 0.5 else
                "Stablecoin supply shrinking — less fuel for rallies, cautious macro"
                if pct_change < -0.5 else
                "Stablecoin supply stable — neutral macro signal"
            ),
            "top_stables": sorted(breakdown, key=lambda x: x["market_cap"], reverse=True)[:5],
        }
    except Exception as e:
        return {"source": "defillama", "error": str(e)}


# ------------------------------------------------------------------
# Token Unlocks — TokenUnlocks API
# ------------------------------------------------------------------

async def get_token_unlocks(symbol: str) -> dict:
    """Fetch upcoming token unlock events — the #1 supply-side risk for altcoins."""
    url = f"https://token.unlocks.app/api/v1/token/{symbol.lower()}"
    try:
        data = await fetch_json(url, source="default")
        if isinstance(data, dict) and data.get("error"):
            # Try CryptoRank as fallback
            return await _get_cryptorank_vesting(symbol)
        events = data.get("unlockEvents", data.get("events", []))
        upcoming = []
        for ev in events[:10]:
            upcoming.append({
                "date": ev.get("date") or ev.get("unlockDate"),
                "amount": safe_float(ev.get("amount") or ev.get("tokenAmount")),
                "value_usd": safe_float(ev.get("valueUsd")),
                "pct_of_supply": safe_float(ev.get("pctOfSupply") or ev.get("percentOfCirculating")),
                "type": ev.get("type", "cliff"),
                "description": ev.get("description", ""),
            })
        return {
            "source": "tokenunlocks",
            "symbol": symbol.upper(),
            "upcoming_unlocks": upcoming,
            "next_unlock": upcoming[0] if upcoming else None,
            "total_locked_pct": safe_float(data.get("totalLockedPct")),
        }
    except Exception:
        return await _get_cryptorank_vesting(symbol)


async def _get_cryptorank_vesting(symbol: str) -> dict:
    """Fallback: CryptoRank vesting schedule."""
    url = f"https://api.cryptorank.io/v1/coins/{symbol.lower()}/vesting"
    try:
        data = await fetch_json(url, source="default")
        vesting = data.get("data", {})
        events = vesting.get("events", [])
        upcoming = []
        for ev in events[:5]:
            upcoming.append({
                "date": ev.get("date"),
                "pct_of_supply": safe_float(ev.get("percentOfTotal")),
                "type": ev.get("type", "vesting"),
            })
        return {
            "source": "cryptorank",
            "symbol": symbol.upper(),
            "upcoming_unlocks": upcoming,
            "next_unlock": upcoming[0] if upcoming else None,
        }
    except Exception as e:
        return {
            "source": "none",
            "symbol": symbol.upper(),
            "error": str(e),
            "upcoming_unlocks": [],
            "note": "Token unlock data unavailable from all sources.",
        }
