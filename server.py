from __future__ import annotations
"""
CryptoOracle MCP Server — Main Entry Point
Lightweight JSON-RPC MCP server compatible with Python 3.9+.
Registers all 28 tools and communicates via STDIO.
Includes 5 new quant engine tools (Phase 10 — EV-based spot trading).
"""
import sys
import os
import json
import asyncio
import logging
from typing import Dict, Any, List, Optional

sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

from tools_core import (
    tool_search_coin, tool_get_coin_metadata, tool_get_spot_price,
    tool_get_ohlcv_history, tool_get_order_book_depth, tool_get_recent_trades,
    tool_compute_technical_indicators, tool_detect_chart_patterns,
    tool_compute_support_resistance, tool_get_fear_greed_index,
    tool_get_market_sentiment, tool_get_funding_rates, tool_get_open_interest,
    tool_get_whale_activity, tool_get_onchain_metrics,
    tool_get_oi_delta, tool_get_cross_exchange_funding, tool_get_liquidation_data,
    tool_get_stablecoin_flows, tool_get_token_unlocks,
    # Quant engine tools (Phase 10)
    tool_spot_flow_analysis, tool_accumulation_distribution,
    tool_regime_classification, tool_volatility_sustainability,
    tool_quant_ev_brief,
)
from tools_advanced import (
    tool_get_global_market_context, tool_get_correlations,
    tool_get_exchange_listings, tool_get_upcoming_events,
    tool_get_top_holders, tool_calculate_risk_metrics,
    tool_compute_entry_zones, tool_full_coin_intelligence_report,
    tool_oracle_intelligence_brief,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
                    stream=sys.stderr)
logger = logging.getLogger("crypto-oracle-mcp")

# ── Tool Registry ────────────────────────────────────────────────

TOOLS = {
    "search_coin": {
        "handler": tool_search_coin,
        "description": "Resolve any coin name/symbol into canonical IDs across all exchanges. Always call this first.",
        "inputSchema": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Coin name, symbol, or partial string"}},
            "required": ["query"],
        },
    },
    "get_coin_metadata": {
        "handler": tool_get_coin_metadata,
        "description": "Full project metadata — team, tokenomics, social links, contract addresses, dev activity.",
        "inputSchema": {
            "type": "object",
            "properties": {"coingecko_id": {"type": "string"}},
            "required": ["coingecko_id"],
        },
    },
    "get_spot_price": {
        "handler": tool_get_spot_price,
        "description": "Real-time price from Binance, MEXC, CoinGecko simultaneously. Spread > 0.5% = flag.",
        "inputSchema": {
            "type": "object",
            "properties": {"symbol": {"type": "string"}, "coingecko_id": {"type": "string"}},
            "required": ["symbol", "coingecko_id"],
        },
    },
    "get_ohlcv_history": {
        "handler": tool_get_ohlcv_history,
        "description": "Full candlestick history. Intervals: 1m,5m,15m,1h,4h,1d,1w. Max limit: 1000.",
        "inputSchema": {
            "type": "object",
            "properties": {"symbol": {"type": "string"}, "interval": {"type": "string", "default": "1h"},
                           "limit": {"type": "integer", "default": 500}},
            "required": ["symbol"],
        },
    },
    "get_order_book_depth": {
        "handler": tool_get_order_book_depth,
        "description": "Order book snapshot with bid/ask walls and imbalance ratio.",
        "inputSchema": {
            "type": "object",
            "properties": {"symbol": {"type": "string"}, "depth": {"type": "integer", "default": 50}},
            "required": ["symbol"],
        },
    },
    "get_recent_trades": {
        "handler": tool_get_recent_trades,
        "description": "Last N trades with whale detection (>10x avg marked as large).",
        "inputSchema": {
            "type": "object",
            "properties": {"symbol": {"type": "string"}, "limit": {"type": "integer", "default": 200}},
            "required": ["symbol"],
        },
    },
    "compute_technical_indicators": {
        "handler": tool_compute_technical_indicators,
        "description": "THE MOST IMPORTANT TOOL. 25+ indicators across multiple timeframes (RSI, MACD, BB, Ichimoku, etc).",
        "inputSchema": {
            "type": "object",
            "properties": {"symbol": {"type": "string"},
                           "timeframes": {"type": "array", "items": {"type": "string"}, "default": ["15m","1h","4h","1d"]}},
            "required": ["symbol"],
        },
    },
    "detect_chart_patterns": {
        "handler": tool_detect_chart_patterns,
        "description": "Chart pattern recognition: H&S, Double Top/Bottom, Triangles, Flags, Wedges.",
        "inputSchema": {
            "type": "object",
            "properties": {"symbol": {"type": "string"}, "lookback_candles": {"type": "integer", "default": 100}},
            "required": ["symbol"],
        },
    },
    "compute_support_resistance": {
        "handler": tool_compute_support_resistance,
        "description": "Key S/R levels using zigzag pivot + swing clustering.",
        "inputSchema": {
            "type": "object",
            "properties": {"symbol": {"type": "string"}, "timeframe": {"type": "string", "default": "4h"},
                           "lookback": {"type": "integer", "default": 200}},
            "required": ["symbol"],
        },
    },
    "get_fear_greed_index": {
        "handler": tool_get_fear_greed_index,
        "description": "Crypto Fear & Greed Index. <20 = buy zone, >80 = elevated risk.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    "get_market_sentiment": {
        "handler": tool_get_market_sentiment,
        "description": "Coin-specific sentiment: Twitter, Reddit, Telegram, trending rank.",
        "inputSchema": {
            "type": "object",
            "properties": {"coingecko_id": {"type": "string"}},
            "required": ["coingecko_id"],
        },
    },
    "get_funding_rates": {
        "handler": tool_get_funding_rates,
        "description": "Futures funding rate. High positive > 0.1% predicts spot pullback.",
        "inputSchema": {
            "type": "object",
            "properties": {"symbol": {"type": "string"}},
            "required": ["symbol"],
        },
    },
    "get_open_interest": {
        "handler": tool_get_open_interest,
        "description": "Futures open interest + long/short ratios.",
        "inputSchema": {
            "type": "object",
            "properties": {"symbol": {"type": "string"}},
            "required": ["symbol"],
        },
    },
    "get_whale_activity": {
        "handler": tool_get_whale_activity,
        "description": "Whale accumulation/distribution signals. Net negative flow = bullish.",
        "inputSchema": {
            "type": "object",
            "properties": {"symbol": {"type": "string"}, "coingecko_id": {"type": "string"}},
            "required": ["symbol", "coingecko_id"],
        },
    },
    "get_onchain_metrics": {
        "handler": tool_get_onchain_metrics,
        "description": "On-chain health: active addresses, NVT, exchange reserve, hash rate.",
        "inputSchema": {
            "type": "object",
            "properties": {"coingecko_id": {"type": "string"}, "symbol": {"type": "string"}},
            "required": ["coingecko_id", "symbol"],
        },
    },
    "get_oi_delta": {
        "handler": tool_get_oi_delta,
        "description": "OI delta tracking — identifies if new longs/shorts are entering or closing.",
        "inputSchema": {
            "type": "object",
            "properties": {"symbol": {"type": "string"}},
            "required": ["symbol"],
        },
    },
    "get_cross_exchange_funding": {
        "handler": tool_get_cross_exchange_funding,
        "description": "Funding rate spread between Binance and Bybit. Divergence = asymmetry.",
        "inputSchema": {
            "type": "object",
            "properties": {"symbol": {"type": "string"}},
            "required": ["symbol"],
        },
    },
    "get_liquidation_data": {
        "handler": tool_get_liquidation_data,
        "description": "Coinglass liquidation heatmap and leverage clusters.",
        "inputSchema": {
            "type": "object",
            "properties": {"symbol": {"type": "string"}},
            "required": ["symbol"],
        },
    },
    "get_stablecoin_flows": {
        "handler": tool_get_stablecoin_flows,
        "description": "Macro stablecoin supply from DefiLlama. Tracks systemic dry powder.",
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
    "get_token_unlocks": {
        "handler": tool_get_token_unlocks,
        "description": "Upcoming token unlocks and vesting cliffs. #1 supply-side risk.",
        "inputSchema": {
            "type": "object",
            "properties": {"symbol": {"type": "string"}},
            "required": ["symbol"],
        },
    },
    "get_global_market_context": {
        "handler": tool_get_global_market_context,
        "description": "Macro context: BTC dominance, total market cap, altcoin season.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    "get_correlations": {
        "handler": tool_get_correlations,
        "description": "Coin correlation with BTC/ETH. >0.8 = follows BTC, <0.4 = independent.",
        "inputSchema": {
            "type": "object",
            "properties": {"symbol": {"type": "string"}, "lookback_days": {"type": "integer", "default": 30}},
            "required": ["symbol"],
        },
    },
    "get_exchange_listings": {
        "handler": tool_get_exchange_listings,
        "description": "Exchange listings with volume and trust score.",
        "inputSchema": {
            "type": "object",
            "properties": {"coingecko_id": {"type": "string"}},
            "required": ["coingecko_id"],
        },
    },
    "get_upcoming_events": {
        "handler": tool_get_upcoming_events,
        "description": "Scheduled events: unlocks, mainnet, halvings.",
        "inputSchema": {
            "type": "object",
            "properties": {"coingecko_id": {"type": "string"}},
            "required": ["coingecko_id"],
        },
    },
    "get_top_holders": {
        "handler": tool_get_top_holders,
        "description": "Wallet concentration for ERC-20/BEP-20/SPL tokens.",
        "inputSchema": {
            "type": "object",
            "properties": {"contract_address": {"type": "string"}, "chain": {"type": "string", "default": "eth"}},
            "required": ["contract_address"],
        },
    },
    "calculate_risk_metrics": {
        "handler": tool_calculate_risk_metrics,
        "description": "Risk profile: Sharpe, Sortino, max drawdown, VaR, beta vs BTC.",
        "inputSchema": {
            "type": "object",
            "properties": {"symbol": {"type": "string"}, "lookback_days": {"type": "integer", "default": 30}},
            "required": ["symbol"],
        },
    },
    "compute_entry_zones": {
        "handler": tool_compute_entry_zones,
        "description": "Entry zones, SL (ATR method), TP1/TP2/TP3 with R:R ratios.",
        "inputSchema": {
            "type": "object",
            "properties": {"symbol": {"type": "string"}, "current_price": {"type": "number"},
                           "direction": {"type": "string", "enum": ["long", "short"], "default": "long"},
                           "risk_tolerance": {"type": "string", "enum": ["conservative","moderate","aggressive"], "default": "moderate"}},
            "required": ["symbol", "current_price"],
        },
    },
    "full_coin_intelligence_report": {
        "handler": tool_full_coin_intelligence_report,
        "description": "MASTER DATA TOOL — runs complete pipeline: price, TA, patterns, S/R, derivatives, whales, risk, entry zones. Also attaches ORACLE engine analysis (Phases 1-9).",
        "inputSchema": {
            "type": "object",
            "properties": {"query": {"type": "string"}, "exchanges": {"type": "array", "items": {"type": "string"}}},
            "required": ["query"],
        },
    },
    "oracle_intelligence_brief": {
        "handler": tool_oracle_intelligence_brief,
        "description": "ORACLE — fused Wolf/Insider/Quant intelligence brief. Executes Phases 1-9 (regime, MTF, DPI+WFI, patterns, macro, composite, Bayesian scenarios + EV, Kelly-sized trade plan, risk audit) and returns a Markdown brief matching the ORACLE spec.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Coin name, symbol, or id"},
                "exchanges": {"type": "array", "items": {"type": "string"}},
                "format": {"type": "string", "enum": ["markdown", "json", "both"], "default": "both"},
            },
            "required": ["query"],
        },
    },
    # ── Quant Engine Tools (Phase 10) ────────────────────────────────
    "spot_flow_analysis": {
        "handler": tool_spot_flow_analysis,
        "description": "QUANT: Real spot Order Flow Imbalance (OFI), absorption detection, iceberg orders, liquidity gaps. No derivatives. All signals from spot execution data only.",
        "inputSchema": {
            "type": "object",
            "properties": {"symbol": {"type": "string", "description": "e.g. BTC, ETH"}},
            "required": ["symbol"],
        },
    },
    "accumulation_distribution": {
        "handler": tool_accumulation_distribution,
        "description": "QUANT: Multi-timescale CVD (15m/1h/4h) accumulation/distribution analysis. Detects stealth buying/selling via price-CVD divergence. Builds volume profile (VPVR proxy).",
        "inputSchema": {
            "type": "object",
            "properties": {"symbol": {"type": "string"}},
            "required": ["symbol"],
        },
    },
    "regime_classification": {
        "handler": tool_regime_classification,
        "description": "QUANT: Data-driven market regime: TRENDING_UP/DOWN, RANGING, EXPANSION, LOW_LIQUIDITY. Uses Hurst exponent, autocorrelation, trend efficiency. Determines which signal class dominates.",
        "inputSchema": {
            "type": "object",
            "properties": {"symbol": {"type": "string"}},
            "required": ["symbol"],
        },
    },
    "volatility_sustainability": {
        "handler": tool_volatility_sustainability,
        "description": "QUANT: Trend sustainability analysis. Realized vol, trend persistence score, breakout quality, P(continuation) vs P(mean reversion). Note: probabilities are uncalibrated priors.",
        "inputSchema": {
            "type": "object",
            "properties": {"symbol": {"type": "string"}},
            "required": ["symbol"],
        },
    },
    "quant_ev_brief": {
        "handler": tool_quant_ev_brief,
        "description": "QUANT MASTER TOOL: Full EV-based analysis. Runs spot flow, accumulation, regime, volatility, failure detection and logistic regression EV model. Outputs P(up), P(down), EV%, top 5 features, failure modes. MODEL IS UNCALIBRATED until backtested.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string"},
                "coingecko_id": {"type": "string", "description": "Optional CoinGecko ID"},
            },
            "required": ["symbol"],
        },
    },
}


# ── MCP Prompts Registry ────────────────────────────────────────

PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "prompts")

PROMPTS = {
    "analysis_oracle": {
        "description": "ORACLE — fused Wolf/Insider/Quant system prompt. Full 9-phase quantitative trading intelligence spec.",
        "file": "analysis_oracle.md",
        "arguments": [],
    },
    "datacollector": {
        "description": "DataCollector — orchestrates MCP tools to feed ORACLE with complete market intelligence.",
        "file": "datacollector.md",
        "arguments": [],
    },
}


def _load_prompt(name: str) -> str:
    meta = PROMPTS.get(name)
    if not meta:
        raise KeyError(f"Unknown prompt: {name}")
    path = os.path.join(PROMPTS_DIR, meta["file"])
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# ── MCP JSON-RPC STDIO Transport ────────────────────────────────

SERVER_INFO = {
    "name": "CryptoOracle",
    "version": "2.0.0-oracle",
}

PROTOCOL_VERSION = "2024-11-05"


async def handle_request(request: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Handle a single JSON-RPC request."""
    method = request.get("method", "")
    req_id = request.get("id")
    params = request.get("params", {})

    if method == "initialize":
        return _jsonrpc_response(req_id, {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {
                "tools": {"listChanged": False},
                "prompts": {"listChanged": False},
            },
            "serverInfo": SERVER_INFO,
        })

    elif method == "notifications/initialized":
        return None  # No response for notifications

    elif method == "tools/list":
        tools_list = []
        for name, t in TOOLS.items():
            tools_list.append({
                "name": name,
                "description": t["description"],
                "inputSchema": t["inputSchema"],
            })
        return _jsonrpc_response(req_id, {"tools": tools_list})

    elif method == "tools/call":
        tool_name = params.get("name", "")
        tool_args = params.get("arguments", {})

        if tool_name not in TOOLS:
            return _jsonrpc_error(req_id, -32602, f"Unknown tool: {tool_name}")

        try:
            handler = TOOLS[tool_name]["handler"]
            result = await handler(**tool_args)
            return _jsonrpc_response(req_id, {
                "content": [{"type": "text", "text": json.dumps(result, default=str)}],
                "isError": False,
            })
        except Exception as e:
            logger.error(f"Tool {tool_name} failed: {e}")
            return _jsonrpc_response(req_id, {
                "content": [{"type": "text", "text": json.dumps({"error": str(e)})}],
                "isError": True,
            })

    elif method == "prompts/list":
        prompts_list = [{
            "name": n,
            "description": p["description"],
            "arguments": p.get("arguments", []),
        } for n, p in PROMPTS.items()]
        return _jsonrpc_response(req_id, {"prompts": prompts_list})

    elif method == "prompts/get":
        name = params.get("name", "")
        if name not in PROMPTS:
            return _jsonrpc_error(req_id, -32602, f"Unknown prompt: {name}")
        try:
            text = _load_prompt(name)
            return _jsonrpc_response(req_id, {
                "description": PROMPTS[name]["description"],
                "messages": [{
                    "role": "user",
                    "content": {"type": "text", "text": text},
                }],
            })
        except Exception as e:
            return _jsonrpc_error(req_id, -32603, f"Failed to load prompt: {e}")

    elif method == "ping":
        return _jsonrpc_response(req_id, {})

    else:
        if req_id is not None:
            return _jsonrpc_error(req_id, -32601, f"Method not found: {method}")
        return None


def _jsonrpc_response(req_id: Any, result: Any) -> Dict:
    return {"jsonrpc": "2.0", "id": req_id, "result": result}


def _jsonrpc_error(req_id: Any, code: int, message: str) -> Dict:
    return {"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}}


async def main():
    """Run the MCP server over STDIO using newline-delimited JSON-RPC.

    MCP's STDIO transport is line-delimited (one JSON-RPC object per line).
    We read stdin in a worker thread because asyncio's connect_read_pipe()
    does not support sys.stdin on Windows.
    """
    logger.info("CryptoOracle MCP Server starting (STDIO mode)...")

    # Force binary, unbuffered streams so framing is byte-exact.
    stdin = sys.stdin.buffer
    stdout = sys.stdout.buffer
    loop = asyncio.get_event_loop()

    def _read_line() -> bytes:
        return stdin.readline()

    def _write_line(data: bytes) -> None:
        stdout.write(data)
        stdout.flush()

    while True:
        try:
            line = await loop.run_in_executor(None, _read_line)
            if not line:  # EOF — client closed the pipe
                break
            line = line.strip()
            if not line:
                continue

            try:
                request = json.loads(line.decode("utf-8"))
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error: {e} | line={line[:200]!r}")
                continue

            response = await handle_request(request)
            if response is not None:
                payload = (json.dumps(response) + "\n").encode("utf-8")
                await loop.run_in_executor(None, _write_line, payload)

        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.exception(f"Server loop error: {e}")
            # Don't break — keep serving subsequent requests.
            continue

    logger.info("CryptoOracle MCP Server stopped.")


if __name__ == "__main__":
    if sys.platform == "win32":
        # ProactorEventLoop is fine here since we no longer use connect_read_pipe.
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
