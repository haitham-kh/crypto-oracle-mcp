from __future__ import annotations
"""
CryptoOracle MCP — Tool implementations (Categories 1-4)
"""
import asyncio
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from utils import (build_response, build_error_response, utc_now_iso, utc_now_ts,
                   safe_float, safe_int, spread_pct, pct_change, cache,
                   CACHE_TTL_PRICE, CACHE_TTL_METADATA, CACHE_TTL_OHLCV,
                   CACHE_TTL_ORDERBOOK, CACHE_TTL_SENTIMENT, CACHE_TTL_GLOBAL,
                   CACHE_TTL_FEAR_GREED)
from fetchers import binance, mexc, coingecko, cmc, onchain
from indicators import compute_all_indicators
from patterns import detect_chart_patterns as _detect_patterns, compute_support_resistance as _compute_sr


# ── CATEGORY 1: Coin Discovery ───────────────────────────────────

async def tool_search_coin(query: str) -> dict:
    """Always call this first to resolve canonical coin IDs across all exchanges."""
    sources_used, sources_failed = [], []
    cg_results, cmc_results = [], []

    try:
        cg_results = await coingecko.search_coins(query)
        sources_used.append("coingecko")
    except Exception as e:
        sources_failed.append("coingecko")

    try:
        cmc_results = await cmc.search_coins(query)
        sources_used.append("cmc")
    except Exception:
        sources_failed.append("cmc")

    if not cg_results and not cmc_results:
        return build_error_response("search_coin", query, "No results from any source")

    # Prefer exact symbol or id match over fuzzy relevance ranking
    q = (query or "").strip().lower()

    def _pick(results: list) -> dict:
        if not results:
            return {}
        for r in results:
            if (r.get("symbol") or "").lower() == q:
                return r
        for r in results:
            if (r.get("id") or "").lower() == q:
                return r
        for r in results:
            if (r.get("name") or "").lower() == q:
                return r
        return results[0]

    best_cg = _pick(cg_results)
    best_cmc = _pick(cmc_results)
    symbol = (best_cg.get("symbol") or best_cmc.get("symbol") or query).upper()

    data = {
        "symbol": symbol,
        "name": best_cg.get("name") or best_cmc.get("name"),
        "coingecko_id": best_cg.get("id"),
        "cmc_id": best_cmc.get("id"),
        "cmc_slug": best_cmc.get("slug"),
        "binance_symbol": symbol,
        "mexc_symbol": symbol,
        "market_cap_rank": best_cg.get("market_cap_rank") or best_cmc.get("rank"),
        "all_coingecko_results": cg_results[:5],
        "all_cmc_results": cmc_results[:5],
    }

    if best_cg.get("id"):
        try:
            detail = await coingecko.get_coin_detail(best_cg["id"])
            md = detail.get("market_data", {})
            data.update({
                "description": detail.get("description", "")[:300],
                "category": (detail.get("categories") or [None])[0],
                "tags": detail.get("categories", []),
                "website": (detail.get("links", {}) or {}).get("homepage"),
                "whitepaper_url": (detail.get("links", {}) or {}).get("whitepaper"),
                "launch_date": detail.get("genesis_date"),
                "max_supply": md.get("max_supply"),
                "circulating_supply": md.get("circulating_supply"),
                "total_supply": md.get("total_supply"),
            })
        except Exception:
            pass

    return build_response("search_coin", symbol, data, sources_used, sources_failed)


async def tool_get_coin_metadata(coingecko_id: str) -> dict:
    """Full project metadata — team, tokenomics, social links, contract addresses."""
    cached = cache.get("metadata", {"id": coingecko_id})
    if cached:
        return cached
    try:
        detail = await coingecko.get_coin_detail(coingecko_id)
        resp = build_response("get_coin_metadata", detail.get("symbol", ""), detail, ["coingecko"])
        cache.set("metadata", {"id": coingecko_id}, resp, CACHE_TTL_METADATA)
        return resp
    except Exception as e:
        return build_error_response("get_coin_metadata", coingecko_id, str(e))


# ── CATEGORY 2: Price Data ───────────────────────────────────────

async def tool_get_spot_price(symbol: str, coingecko_id: str) -> dict:
    """Real-time price from multiple exchanges for cross-validation.
    Price spread > 0.5% indicates arbitrage opportunity or data anomaly."""
    sources_used, sources_failed = [], []
    prices = {}

    async def _binance():
        try:
            r = await binance.get_ticker_price(symbol)
            prices["binance"] = r["price"]
            sources_used.append("binance")
        except Exception:
            sources_failed.append("binance")

    async def _mexc():
        try:
            r = await mexc.get_ticker_price(symbol)
            prices["mexc"] = r["price"]
            sources_used.append("mexc")
        except Exception:
            sources_failed.append("mexc")

    async def _cg():
        try:
            r = await coingecko.get_simple_price(coingecko_id)
            prices["coingecko"] = r["price_usd"]
            sources_used.append("coingecko")
        except Exception:
            sources_failed.append("coingecko")

    await asyncio.gather(_binance(), _mexc(), _cg())

    valid = [v for v in prices.values() if v and v > 0]
    avg = sum(valid) / len(valid) if valid else 0
    sp = spread_pct(valid)

    data = {
        "binance_price": prices.get("binance"),
        "mexc_price": prices.get("mexc"),
        "coingecko_price": prices.get("coingecko"),
        "average_price": round(avg, 8) if avg else None,
        "price_spread_pct": round(sp, 4),
        "spread_flag": sp > 0.5,
        "timestamp": utc_now_iso(),
    }
    return build_response("get_spot_price", symbol, data, sources_used, sources_failed)


async def tool_get_ohlcv_history(symbol: str, interval: str = "1h", limit: int = 500) -> dict:
    """Full candlestick history for technical analysis. Fetches from Binance (primary) and MEXC (validation)."""
    sources_used, sources_failed = [], []
    binance_candles, mexc_candles = [], []

    try:
        binance_candles = await binance.get_klines(symbol, interval, limit)
        sources_used.append("binance")
    except Exception:
        sources_failed.append("binance")

    try:
        mexc_candles = await mexc.get_klines(symbol, interval, limit)
        sources_used.append("mexc")
    except Exception:
        sources_failed.append("mexc")

    primary = binance_candles or mexc_candles
    if not primary:
        return build_error_response("get_ohlcv_history", symbol, "No candle data from any exchange")

    data = {
        "interval": interval,
        "candle_count": len(primary),
        "candles": primary,
        "source": "binance" if binance_candles else "mexc",
        "validation_source_candle_count": len(mexc_candles) if binance_candles else len(binance_candles),
    }
    return build_response("get_ohlcv_history", symbol, data, sources_used, sources_failed)


async def tool_get_order_book_depth(symbol: str, depth: int = 50) -> dict:
    """Order book snapshot for entry/exit price impact analysis."""
    sources_used, sources_failed = [], []
    book = None

    try:
        book = await binance.get_depth(symbol, depth)
        sources_used.append("binance")
    except Exception:
        sources_failed.append("binance")
        try:
            book = await mexc.get_depth(symbol, depth)
            sources_used.append("mexc")
        except Exception:
            sources_failed.append("mexc")

    if not book:
        return build_error_response("get_order_book_depth", symbol, "No order book data")

    bids = book["bids"][:depth]
    asks = book["asks"][:depth]

    # Compute analytics
    cum_bid = 0
    bids_enriched = []
    for b in bids[:10]:
        cum_bid += b["price"] * b["qty"]
        bids_enriched.append({**b, "cumulative_usdt": round(cum_bid, 2)})

    cum_ask = 0
    asks_enriched = []
    for a in asks[:10]:
        cum_ask += a["price"] * a["qty"]
        asks_enriched.append({**a, "cumulative_usdt": round(cum_ask, 2)})

    total_bid = sum(b["price"] * b["qty"] for b in bids)
    total_ask = sum(a["price"] * a["qty"] for a in asks)
    imbalance = round(total_bid / total_ask, 4) if total_ask > 0 else 0

    # Detect walls (orders >= 2x average)
    avg_bid_size = total_bid / len(bids) if bids else 0
    avg_ask_size = total_ask / len(asks) if asks else 0
    bid_wall = next((b["price"] for b in bids if b["price"] * b["qty"] >= 2 * avg_bid_size), None)
    ask_wall = next((a["price"] for a in asks if a["price"] * a["qty"] >= 2 * avg_ask_size), None)

    spread = 0
    if bids and asks:
        spread = round(((asks[0]["price"] - bids[0]["price"]) / bids[0]["price"]) * 100, 6)

    data = {
        "bids_top10": bids_enriched,
        "asks_top10": asks_enriched,
        "bid_ask_spread_pct": spread,
        "bid_wall_at": bid_wall,
        "ask_wall_at": ask_wall,
        "total_bid_usdt_depth": round(total_bid, 2),
        "total_ask_usdt_depth": round(total_ask, 2),
        "imbalance_ratio": imbalance,
    }
    return build_response("get_order_book_depth", symbol, data, sources_used, sources_failed)


async def tool_get_recent_trades(symbol: str, limit: int = 200) -> dict:
    """Last N trades — reveals momentum, acceleration, whale activity.
    Trades > 10x average size are marked as large/whale trades."""
    sources_used, sources_failed = [], []
    trades = []

    try:
        trades = await binance.get_recent_trades(symbol, limit)
        sources_used.append("binance")
    except Exception:
        sources_failed.append("binance")
        try:
            trades = await mexc.get_recent_trades(symbol, limit)
            sources_used.append("mexc")
        except Exception:
            sources_failed.append("mexc")

    if not trades:
        return build_error_response("get_recent_trades", symbol, "No trade data")

    sizes = [t["price"] * t["qty"] for t in trades]
    avg_size = sum(sizes) / len(sizes) if sizes else 0
    buy_vol = sum(s for s, t in zip(sizes, trades) if not t["is_buyer_maker"])
    sell_vol = sum(s for s, t in zip(sizes, trades) if t["is_buyer_maker"])
    total_vol = buy_vol + sell_vol

    large_trades = [
        {**t, "usdt_value": round(t["price"] * t["qty"], 2)}
        for t, s in zip(trades, sizes) if s > avg_size * 10
    ]

    # Acceleration: compare first half vs second half volume
    mid = len(sizes) // 2
    first_half = sum(sizes[:mid]) if mid > 0 else 0
    second_half = sum(sizes[mid:]) if mid > 0 else 0
    accel = round(second_half / first_half, 2) if first_half > 0 else 1.0

    data = {
        "trades_sample": trades[:20],
        "large_trades": large_trades[:10],
        "buy_volume_pct": round(buy_vol / total_vol * 100, 2) if total_vol else 0,
        "sell_volume_pct": round(sell_vol / total_vol * 100, 2) if total_vol else 0,
        "avg_trade_size_usdt": round(avg_size, 2),
        "largest_trade_usdt": round(max(sizes), 2) if sizes else 0,
        "trade_acceleration_score": accel,
    }
    return build_response("get_recent_trades", symbol, data, sources_used, sources_failed)


# ── CATEGORY 3: Technical Indicators ─────────────────────────────

async def tool_compute_technical_indicators(symbol: str, timeframes: list = None) -> dict:
    """The most important tool. Computes full suite of 25+ TA indicators across multiple timeframes.
    Always call with all 4 timeframes for complete MTF analysis. Higher timeframe bias overrides lower."""
    if timeframes is None:
        timeframes = ["15m", "1h", "4h", "1d", "1w"]

    results = {}
    sources_used, sources_failed = [], []

    for tf in timeframes:
        candles = None
        used = None
        try:
            candles = await binance.get_klines(symbol, tf, 200)
            if candles:
                used = "binance"
        except Exception:
            sources_failed.append(f"binance_{tf}")
        if not candles:
            try:
                candles = await mexc.get_klines(symbol, tf, 200)
                if candles:
                    used = "mexc"
            except Exception:
                sources_failed.append(f"mexc_{tf}")
        if not candles:
            results[tf] = {"error": "no candle data from binance or mexc", "timeframe": tf}
            continue
        try:
            indicators = compute_all_indicators(candles)
            indicators["timeframe"] = tf
            results[tf] = indicators
            if used and used not in sources_used:
                sources_used.append(used)
        except Exception as e:
            results[tf] = {"error": str(e), "timeframe": tf}

    return build_response("compute_technical_indicators", symbol, results, sources_used, sources_failed)


async def _klines_with_fallback(symbol: str, interval: str, limit: int):
    """Try Binance first, fall back to MEXC. Returns (candles, source)."""
    try:
        candles = await binance.get_klines(symbol, interval, limit)
        if candles:
            return candles, "binance"
    except Exception:
        pass
    try:
        candles = await mexc.get_klines(symbol, interval, limit)
        if candles:
            return candles, "mexc"
    except Exception:
        pass
    return None, None


async def tool_detect_chart_patterns(symbol: str, lookback_candles: int = 100) -> dict:
    """Structural chart pattern recognition on recent price history."""
    try:
        candles, src = await _klines_with_fallback(symbol, "4h", lookback_candles)
        if not candles:
            return build_error_response("detect_chart_patterns", symbol, "no candle data")
        result = _detect_patterns(candles, lookback_candles)
        return build_response("detect_chart_patterns", symbol, result, [src])
    except Exception as e:
        return build_error_response("detect_chart_patterns", symbol, str(e))


async def tool_compute_support_resistance(symbol: str, timeframe: str = "4h", lookback: int = 200) -> dict:
    """Key price levels for entry, TP, and SL placement."""
    try:
        candles, src = await _klines_with_fallback(symbol, timeframe, lookback)
        if not candles:
            return build_error_response("compute_support_resistance", symbol, "no candle data")
        result = _compute_sr(candles, lookback)
        return build_response("compute_support_resistance", symbol, result, [src])
    except Exception as e:
        return build_error_response("compute_support_resistance", symbol, str(e))


# ── CATEGORY 4: Sentiment & On-Chain ─────────────────────────────

async def tool_get_fear_greed_index() -> dict:
    """Crypto Fear & Greed Index. Extreme Fear <20 = good buy zone; Extreme Greed >80 = elevated risk."""
    cached = cache.get("fear_greed", {})
    if cached:
        return cached
    try:
        data = await onchain.get_fear_greed(30)
        resp = build_response("get_fear_greed_index", "MARKET", data, ["alternative.me"])
        cache.set("fear_greed", {}, resp, CACHE_TTL_FEAR_GREED)
        return resp
    except Exception as e:
        return build_error_response("get_fear_greed_index", "MARKET", str(e))


async def tool_get_market_sentiment(coingecko_id: str) -> dict:
    """Coin-specific sentiment across social channels."""
    try:
        detail = await coingecko.get_coin_detail(coingecko_id)
        community = detail.get("community", {})
        trending = await coingecko.get_trending()
        trend_rank = None
        for i, t in enumerate(trending):
            if t.get("id") == coingecko_id:
                trend_rank = i + 1
                break

        data = {
            "twitter_followers": community.get("twitter_followers"),
            "reddit_subscribers": community.get("reddit_subscribers"),
            "reddit_active_48h": community.get("reddit_accounts_active_48h"),
            "telegram_member_count": community.get("telegram_channel_user_count"),
            "sentiment_votes_up_pct": detail.get("sentiment_votes_up_percentage"),
            "trending_rank_coingecko": trend_rank,
        }
        return build_response("get_market_sentiment", detail.get("symbol", ""), data, ["coingecko"])
    except Exception as e:
        return build_error_response("get_market_sentiment", coingecko_id, str(e))


async def tool_get_funding_rates(symbol: str) -> dict:
    """Perpetual futures funding rate. High positive (>0.1%) predicts spot pullback.
    Very negative predicts bounce."""
    try:
        rates = await binance.get_funding_rate(symbol, limit=21)
        if not rates:
            return build_error_response("get_funding_rates", symbol, "No funding data")

        current = rates[-1]["funding_rate"]
        avg_24h = sum(r["funding_rate"] for r in rates[-3:]) / len(rates[-3:]) if len(rates) >= 3 else current
        avg_7d = sum(r["funding_rate"] for r in rates) / len(rates)

        sentiment = "longs_paying" if current > 0 else "shorts_paying"
        extreme = abs(current) > 0.001  # 0.1%

        data = {
            "current_funding_rate": round(current * 100, 6),
            "avg_24h_funding": round(avg_24h * 100, 6),
            "avg_7d_funding": round(avg_7d * 100, 6),
            "funding_sentiment": sentiment,
            "extreme_funding_flag": extreme,
            "historical_funding": [{"rate_pct": round(r["funding_rate"] * 100, 6), "time": r["funding_time"]} for r in rates],
        }
        return build_response("get_funding_rates", symbol, data, ["binance"])
    except Exception as e:
        return build_error_response("get_funding_rates", symbol, str(e))


async def tool_get_open_interest(symbol: str) -> dict:
    """Total open interest in futures. Rising OI + rising price = strong trend."""
    try:
        oi = await binance.get_open_interest(symbol)
        ls = await binance.get_top_long_short_ratio(symbol)
        gls = await binance.get_global_long_short_ratio(symbol)

        data = {
            "open_interest_usdt": oi.get("open_interest"),
            "long_short_ratio": ls.get("long_short_ratio"),
            "top_trader_long_short_ratio": gls.get("long_short_ratio"),
            "long_account_pct": ls.get("long_account"),
            "short_account_pct": ls.get("short_account"),
        }
        return build_response("get_open_interest", symbol, data, ["binance"])
    except Exception as e:
        return build_error_response("get_open_interest", symbol, str(e))


async def tool_get_whale_activity(symbol: str, coingecko_id: str) -> dict:
    """Large holder movements. Net negative exchange flow = bullish accumulation."""
    sources_used, sources_failed = [], []
    data = {}

    # Recent large trades from Binance
    try:
        trades = await binance.get_recent_trades(symbol, 500)
        sizes = [t["price"] * t["qty"] for t in trades]
        avg = sum(sizes) / len(sizes) if sizes else 0
        whales = [t for t, s in zip(trades, sizes) if s > avg * 10]

        buy_whales = [w for w in whales if not w["is_buyer_maker"]]
        sell_whales = [w for w in whales if w["is_buyer_maker"]]

        data["whale_trades_24h"] = [
            {"size_usdt": round(w["price"] * w["qty"], 2),
             "direction": "buy" if not w["is_buyer_maker"] else "sell",
             "exchange": "binance"} for w in whales[:10]
        ]
        data["whale_buy_count"] = len(buy_whales)
        data["whale_sell_count"] = len(sell_whales)
        sources_used.append("binance")
    except Exception:
        sources_failed.append("binance")
        data["whale_trades_24h"] = []

    # On-chain flows from Glassnode
    try:
        oc = await onchain.get_onchain_summary(symbol)
        net_flow = oc.get("exchange_net_flow")
        if net_flow is not None:
            data["exchange_net_flow_24h"] = net_flow
            if net_flow < 0:
                data["net_flow_interpretation"] = "accumulation"
            elif net_flow > 0:
                data["net_flow_interpretation"] = "distribution"
            else:
                data["net_flow_interpretation"] = "neutral"
            sources_used.append("glassnode")
        else:
            data["exchange_net_flow_24h"] = None
            data["net_flow_interpretation"] = "data_unavailable"
    except Exception:
        data["exchange_net_flow_24h"] = None
        data["net_flow_interpretation"] = "data_unavailable"

    return build_response("get_whale_activity", symbol, data, sources_used, sources_failed)


async def tool_get_onchain_metrics(coingecko_id: str, symbol: str) -> dict:
    """On-chain health metrics. Uses Glassnode if available, else CoinGecko dev data as proxy."""
    try:
        oc = await onchain.get_onchain_summary(symbol)
        if oc.get("source") == "unavailable":
            detail = await coingecko.get_coin_detail(coingecko_id)
            dev = detail.get("dev_data", {})
            oc = {
                "source": "coingecko_proxy",
                "note": "Approximate — using CoinGecko developer activity as proxy",
                "commit_count_4_weeks": dev.get("commit_count_4_weeks"),
                "github_stars": dev.get("stars"),
                "github_forks": dev.get("forks"),
                "total_issues": dev.get("total_issues"),
                "network_growth_score": min(100, (dev.get("commit_count_4_weeks") or 0) * 2),
            }
        return build_response("get_onchain_metrics", symbol, oc, [oc.get("source", "unknown")])
    except Exception as e:
        return build_error_response("get_onchain_metrics", symbol, str(e))


import feedparser

async def tool_get_recent_news(symbol: str) -> dict:
    """Get recent news headlines for the coin to catch narratives and catalysts."""
    try:
        url = f"https://cryptopanic.com/news/rss/search/?q={symbol}"
        feed = feedparser.parse(url)
        headlines = []
        for entry in feed.entries[:5]:
            headlines.append({"title": entry.title, "link": entry.link, "published": entry.published})
        return build_response("get_recent_news", symbol, {"news": headlines}, ["cryptopanic"])
    except Exception as e:
        return build_error_response("get_recent_news", symbol, str(e))


# ── NEW INSTITUTIONAL-GRADE TOOLS ─────────────────────────────────

async def tool_get_oi_delta(symbol: str) -> dict:
    """OI delta analysis: correlates OI changes with price changes over 24h.
    Rising OI + rising price = strong trend (new longs entering).
    Rising OI + falling price = new shorts (building squeeze pressure).
    Falling OI + rising price = shorts closing (weak rally).
    Falling OI + falling price = longs liquidating (capitulation).
    """
    try:
        oi_hist = await binance.get_oi_history(symbol, "5m", 288)
        if not oi_hist or len(oi_hist) < 10:
            return build_response("get_oi_delta", symbol,
                                  {"available": False, "note": "Insufficient OI history"},
                                  ["binance"])
        # Get price data for same period
        klines = await binance.get_klines(symbol, "5m", 288)

        first_oi = safe_float(oi_hist[0].get("sum_open_interest_value"))
        last_oi = safe_float(oi_hist[-1].get("sum_open_interest_value"))
        first_price = safe_float(klines[0].get("close")) if klines else None
        last_price = safe_float(klines[-1].get("close")) if klines else None

        oi_change_pct = ((last_oi - first_oi) / first_oi * 100) if first_oi else 0
        price_change_pct = ((last_price - first_price) / first_price * 100) if first_price else 0

        # Determine OI-price correlation regime
        if oi_change_pct > 2 and price_change_pct > 0.5:
            regime = "new_longs_entering"
            interpretation = "Strong trend — new money flowing in long"
        elif oi_change_pct > 2 and price_change_pct < -0.5:
            regime = "new_shorts_entering"
            interpretation = "Building short pressure — potential squeeze setup"
        elif oi_change_pct < -2 and price_change_pct > 0.5:
            regime = "shorts_closing"
            interpretation = "Weak rally — driven by short covering, not conviction"
        elif oi_change_pct < -2 and price_change_pct < -0.5:
            regime = "longs_liquidating"
            interpretation = "Capitulation — forced long exits, watch for bottom"
        else:
            regime = "stable"
            interpretation = "OI stable — no significant positioning changes"

        # 1h OI velocity (most recent 12 data points = 1h)
        recent = oi_hist[-12:]
        oi_vals = [safe_float(r.get("sum_open_interest_value")) for r in recent]
        oi_velocity = ((oi_vals[-1] - oi_vals[0]) / oi_vals[0] * 100) if oi_vals[0] else 0

        data = {
            "available": True,
            "oi_24h_start": first_oi,
            "oi_24h_end": last_oi,
            "oi_change_24h_pct": round(oi_change_pct, 2),
            "price_change_24h_pct": round(price_change_pct, 2),
            "oi_price_regime": regime,
            "interpretation": interpretation,
            "oi_velocity_1h_pct": round(oi_velocity, 3),
            "data_points": len(oi_hist),
        }
        return build_response("get_oi_delta", symbol, data, ["binance"])
    except Exception as e:
        return build_error_response("get_oi_delta", symbol, str(e))


async def tool_get_cross_exchange_funding(symbol: str) -> dict:
    """Cross-exchange funding rate comparison.
    Divergence between exchanges reveals positioning asymmetry.
    """
    from fetchers import bybit
    sources_used, sources_failed = [], []
    binance_fr, bybit_fr = None, None

    try:
        b_data = await binance.get_funding_rate(symbol, 1)
        if b_data:
            binance_fr = safe_float(b_data[0].get("funding_rate"))
            sources_used.append("binance")
    except Exception:
        sources_failed.append("binance")

    try:
        by_data = await bybit.get_funding_rate(symbol, 1)
        if by_data:
            bybit_fr = safe_float(by_data[0].get("funding_rate"))
            sources_used.append("bybit")
    except Exception:
        sources_failed.append("bybit")

    spread = None
    if binance_fr is not None and bybit_fr is not None:
        spread = round(abs(binance_fr - bybit_fr) * 100, 4)

    data = {
        "binance_funding_rate": round(binance_fr * 100, 4) if binance_fr else None,
        "bybit_funding_rate": round(bybit_fr * 100, 4) if bybit_fr else None,
        "funding_spread_pct": spread,
        "arbitrage_signal": (
            "significant_divergence" if spread and spread > 0.02 else
            "minor_divergence" if spread and spread > 0.005 else
            "aligned"
        ) if spread is not None else "insufficient_data",
    }
    return build_response("get_cross_exchange_funding", symbol, data, sources_used, sources_failed)


async def tool_get_liquidation_data(symbol: str) -> dict:
    """Liquidation heatmap and cluster data — price magnets in derivatives markets."""
    from fetchers import coinglass
    try:
        liq_info = await coinglass.get_liquidation_map(symbol)
        liq_levels = await coinglass.get_liquidation_levels(symbol)
        data = {**liq_info, **liq_levels}
        return build_response("get_liquidation_data", symbol, data,
                              [data.get("source", "coinglass")])
    except Exception as e:
        return build_error_response("get_liquidation_data", symbol, str(e))


async def tool_get_stablecoin_flows() -> dict:
    """Macro fuel gauge: track stablecoin supply changes across the market."""
    try:
        data = await onchain.get_stablecoin_flows()
        return build_response("get_stablecoin_flows", "MARKET", data,
                              [data.get("source", "defillama")])
    except Exception as e:
        return build_error_response("get_stablecoin_flows", "MARKET", str(e))


async def tool_get_token_unlocks(symbol: str) -> dict:
    """Upcoming token unlock events — #1 systematic supply-side risk for altcoins."""
    try:
        data = await onchain.get_token_unlocks(symbol)
        return build_response("get_token_unlocks", symbol, data,
                              [data.get("source", "unknown")])
    except Exception as e:
        return build_error_response("get_token_unlocks", symbol, str(e))


# ── QUANT ENGINE TOOLS ────────────────────────────────────────────
# These tools expose the quant analysis modules as first-class MCP tools.
# All signals are derived from real spot market flows only.

async def tool_spot_flow_analysis(symbol: str) -> dict:
    """
    Spot Order Flow Analysis — OFI, absorption, iceberg detection, liquidity gaps.

    Computes:
    - Order Flow Imbalance (aggressive buys vs sells / total)
    - Absorption index (high volume + no price movement → accumulation/distribution)
    - Iceberg order detection (repeated fills at same price level)
    - Liquidity gap map (fragile zones in order book)
    - Net buying pressure score (-100 to +100)

    All signals are derived from spot execution data only. No derivatives.
    """
    try:
        from spot_flow_engine import analyze_spot_flow

        # Fetch real-time data
        trades = await binance.get_recent_trades(symbol, 500)
        candles = await binance.get_klines(symbol, "1h", 100)
        ob_raw = await binance.get_depth(symbol, 50)

        # Get current price
        price_data = await binance.get_ticker_price(symbol)
        current_price = float(price_data.get("price", 0))

        if not current_price:
            return build_error_response("spot_flow_analysis", symbol, "Could not get current price")

        result = analyze_spot_flow(
            trades=trades,
            candles=candles,
            order_book=ob_raw,
            current_price=current_price,
        )

        return build_response("spot_flow_analysis", symbol, result, ["binance"])
    except ImportError:
        return build_error_response("spot_flow_analysis", symbol,
                                    "spot_flow_engine module not found")
    except Exception as e:
        return build_error_response("spot_flow_analysis", symbol, str(e))


async def tool_accumulation_distribution(symbol: str) -> dict:
    """
    Multi-timescale Accumulation / Distribution Analysis.

    Computes CVD (Cumulative Volume Delta) at 15m, 1h, 4h independently.
    Detects stealth accumulation and distribution via price-CVD divergence.
    Builds a volume profile (VPVR proxy) to find high-volume price nodes.

    Outputs:
    - Accumulation probability score (0-100)
    - Distribution probability score (0-100)
    - Dominant signal (accumulation/distribution/neutral)
    - Volume profile (point of control, value area, HVN/LVN)
    """
    try:
        from accumulation_engine import analyze_accumulation

        candles_15m, candles_1h, candles_4h = await asyncio.gather(
            binance.get_klines(symbol, "15m", 200),
            binance.get_klines(symbol, "1h", 200),
            binance.get_klines(symbol, "4h", 200),
        )

        price_data = await binance.get_ticker_price(symbol)
        current_price = float(price_data.get("price", 0))

        result = analyze_accumulation(
            candles_15m=candles_15m,
            candles_1h=candles_1h,
            candles_4h=candles_4h,
            current_price=current_price,
        )

        return build_response("accumulation_distribution", symbol, result, ["binance"])
    except ImportError:
        return build_error_response("accumulation_distribution", symbol,
                                    "accumulation_engine module not found")
    except Exception as e:
        return build_error_response("accumulation_distribution", symbol, str(e))


async def tool_regime_classification(symbol: str) -> dict:
    """
    Data-Driven Market Regime Classification.

    Uses Hurst exponent, return autocorrelation, trend efficiency ratio,
    and volatility regime to classify the current market state.

    Regimes: TRENDING_UP, TRENDING_DOWN, RANGING, EXPANSION, LOW_LIQUIDITY

    Each regime comes with recommended signal weights and strategy bias
    (trend-follow vs mean-reversion vs wait).
    """
    try:
        from regime_classifier import classify_regime

        candles_4h, candles_1d = await asyncio.gather(
            binance.get_klines(symbol, "4h", 200),
            binance.get_klines(symbol, "1d", 90),
        )

        result = classify_regime(candles_4h, candles_1d)

        return build_response("regime_classification", symbol, result, ["binance"])
    except ImportError:
        return build_error_response("regime_classification", symbol,
                                    "regime_classifier module not found")
    except Exception as e:
        return build_error_response("regime_classification", symbol, str(e))


async def tool_volatility_sustainability(symbol: str) -> dict:
    """
    Volatility & Trend Sustainability Analysis.

    Computes:
    - Realized volatility (rolling 5/20/60 period)
    - Volatility state (expanding/contracting/stable)
    - Trend persistence score (Hurst + efficiency ratio + consecutive closes)
    - Breakout quality assessment (volume confirmation)
    - P(trend continuation) vs P(mean reversion)

    NOTE: Probabilities are heuristic base rates pending empirical calibration.
    """
    try:
        from regime_classifier import classify_regime
        from volatility_engine import analyze_volatility_sustainability

        candles_4h = await binance.get_klines(symbol, "4h", 200)
        candles_1d = await binance.get_klines(symbol, "1d", 90)

        regime = classify_regime(candles_4h, candles_1d)

        result = analyze_volatility_sustainability(
            candles=candles_4h,
            regime=regime.get("regime", "UNKNOWN"),
        )
        result["regime"] = regime.get("regime")

        return build_response("volatility_sustainability", symbol, result, ["binance"])
    except ImportError:
        return build_error_response("volatility_sustainability", symbol,
                                    "volatility_engine module not found")
    except Exception as e:
        return build_error_response("volatility_sustainability", symbol, str(e))


async def tool_quant_ev_brief(symbol: str, coingecko_id: str = "") -> dict:
    """
    QUANT EV BRIEF — Full quantitative expected value analysis.

    Runs all quant modules (spot flow, accumulation, regime, volatility,
    failure detection) and outputs a complete EV-based trading signal.

    Output format:
    {
      signal: BUY | SELL | NEUTRAL | STRONG_BUY | STRONG_SELL,
      p_up_pct: float (0-100),
      p_down_pct: float (0-100),
      ev_net_pct: float (expected value after fees & slippage),
      is_positive_ev: bool,
      top_5_features: [{ feature, value, contribution, direction, description }],
      market_regime: str,
      confidence_after_failure_filter: float,
      recommended_action: str,
      failure_modes: [str],
      model_calibrated: bool,
      calibration_warning: str | null,
    }

    IMPORTANT: The EV model is UNCALIBRATED until backtested with historical data.
    Use feature contributions for directional guidance until calibration is complete.
    """
    try:
        from spot_flow_engine import analyze_spot_flow
        from accumulation_engine import analyze_accumulation
        from regime_classifier import classify_regime
        from volatility_engine import analyze_volatility_sustainability
        from failure_detector import detect_all_failure_modes
        from ev_model import predict_ev

        # Fetch all needed data in parallel
        results = await asyncio.gather(
            binance.get_recent_trades(symbol, 500),
            binance.get_klines(symbol, "15m", 200),
            binance.get_klines(symbol, "1h", 200),
            binance.get_klines(symbol, "4h", 200),
            binance.get_klines(symbol, "1d", 90),
            binance.get_depth(symbol, 50),
            binance.get_ticker_price(symbol),
            return_exceptions=True,
        )

        trades, c15m, c1h, c4h, c1d, ob_raw, price_data = results
        if isinstance(trades, Exception): trades = []
        if isinstance(c15m, Exception): c15m = []
        if isinstance(c1h, Exception): c1h = []
        if isinstance(c4h, Exception): c4h = []
        if isinstance(c1d, Exception): c1d = []
        if isinstance(ob_raw, Exception): ob_raw = {}
        if isinstance(price_data, Exception): price_data = {}

        current_price = safe_float(price_data.get("price") if isinstance(price_data, dict) else 0)

        primary = c4h or c1h or []

        spot_flow = analyze_spot_flow(trades, primary, ob_raw, current_price) if trades and current_price else {}
        accum = analyze_accumulation(c15m, c1h, c4h, current_price) if current_price else {}
        regime = classify_regime(c4h, c1d)
        vol = analyze_volatility_sustainability(
            primary,
            regime.get("regime", "UNKNOWN"),
            (spot_flow.get("net_buying_pressure") or {}).get("net_buying_pressure_score", 0),
            (accum.get("scores") or {}).get("accumulation_probability", 50),
        ) if primary else {}

        # Key levels for failure detection
        sup_l = []; res_l = []
        # We don't have S/R here without running full pipeline — skip for this lightweight tool
        failure_modes = detect_all_failure_modes(
            candles=primary,
            key_levels=[],
            ofi_score=(spot_flow.get("net_buying_pressure") or {}).get("net_buying_pressure_score", 0),
            regime=regime.get("regime", "UNKNOWN"),
            mtf_scores={},
            trades=trades or None,
        )

        # ATR-based targets
        import numpy as np
        if primary and current_price:
            highs = [float(c.get("high", 0)) for c in primary[-14:]]
            lows = [float(c.get("low", 0)) for c in primary[-14:]]
            closes_prev = [float(c.get("close", 0)) for c in primary[-15:-1]]
            tr_vals = [max(h - l, abs(h - cp), abs(l - cp))
                       for h, l, cp in zip(highs, lows, closes_prev)]
            atr = float(np.mean(tr_vals)) if tr_vals else current_price * 0.02
        else:
            atr = (current_price or 1.0) * 0.02

        target_up = (atr / (current_price or 1.0)) * 100 * 1.5
        target_down = (atr / (current_price or 1.0)) * 100 * 1.0

        macro_data = {"fear_greed_value": 50.0}  # Default when no FG data in this lightweight call

        ev_pred = predict_ev(
            ofi=spot_flow.get("ofi") or {},
            absorption=spot_flow.get("absorption") or {},
            accum=accum,
            regime=regime,
            vol=vol,
            macro=macro_data,
            target_up_pct=max(0.5, target_up),
            target_down_pct=max(0.3, target_down),
        )

        conf_mult = failure_modes.get("confidence_multiplier", 1.0)
        ev_pred["confidence_after_failure_filter"] = round(
            ev_pred.get("model_confidence", 0) * conf_mult, 4
        )

        output = {
            "signal": ev_pred.get("signal", "NEUTRAL"),
            "p_up_pct": ev_pred.get("p_up_pct"),
            "p_down_pct": ev_pred.get("p_down_pct"),
            "ev_net_pct": ev_pred.get("ev_net_pct"),
            "ev_gross_pct": ev_pred.get("ev_gross_pct"),
            "is_positive_ev": ev_pred.get("is_positive_ev"),
            "top_5_features": ev_pred.get("top_5_features") or [],
            "market_regime": regime.get("regime"),
            "regime_confidence": regime.get("confidence"),
            "confidence_after_failure_filter": ev_pred.get("confidence_after_failure_filter"),
            "recommended_action": failure_modes.get("recommended_action"),
            "failure_modes_detected": [f.get("type") for f in failure_modes.get("failure_modes") or []],
            "failure_mode_summary": failure_modes.get("summary"),
            "accumulation_score": (accum.get("scores") or {}).get("accumulation_probability"),
            "distribution_score": (accum.get("scores") or {}).get("distribution_probability"),
            "ofi_score": (spot_flow.get("net_buying_pressure") or {}).get("net_buying_pressure_score"),
            "vol_state": (vol.get("realized_vol") or {}).get("vol_state"),
            "persistence_score": (vol.get("trend_persistence") or {}).get("persistence_score"),
            "model_calibrated": ev_pred.get("model_calibrated", False),
            "calibration_warning": ev_pred.get("calibration_warning"),
        }

        return build_response("quant_ev_brief", symbol, output, ["binance"])

    except ImportError as e:
        return build_error_response("quant_ev_brief", symbol,
                                    f"Quant module not available: {e}. Run: pip install scipy")
    except Exception as e:
        return build_error_response("quant_ev_brief", symbol, str(e))

