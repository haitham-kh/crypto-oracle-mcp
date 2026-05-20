from __future__ import annotations
from charting import generate_chart
"""
CryptoOracle MCP — Tool implementations (Categories 5-7)
Market Structure, Risk Metrics, and Master Aggregation tool.
"""
import asyncio
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from utils import (build_response, build_error_response, utc_now_iso,
                   safe_float, cache, CACHE_TTL_GLOBAL)
from fetchers import binance, mexc, coingecko, cmc, onchain


# ── CATEGORY 5: Market Structure & Context ───────────────────────

async def tool_get_global_market_context() -> dict:
    """Macro crypto market context — BTC dominance, total market cap, altcoin season."""
    cached = cache.get("global_context", {})
    if cached:
        return cached
    sources_used, sources_failed = [], []
    cg_data, cmc_data = {}, {}

    try:
        cg_data = await coingecko.get_global_data()
        sources_used.append("coingecko")
    except Exception:
        sources_failed.append("coingecko")

    try:
        cmc_data = await cmc.get_global_metrics()
        sources_used.append("cmc")
    except Exception:
        sources_failed.append("cmc")

    data = {
        "total_market_cap_usd": cg_data.get("total_market_cap_usd") or cmc_data.get("total_market_cap"),
        "total_volume_24h": cg_data.get("total_volume_24h_usd") or cmc_data.get("total_volume_24h"),
        "btc_dominance_pct": cg_data.get("btc_dominance_pct") or cmc_data.get("btc_dominance"),
        "eth_dominance_pct": cg_data.get("eth_dominance_pct") or cmc_data.get("eth_dominance"),
        "altcoin_market_cap": cmc_data.get("altcoin_market_cap"),
        "defi_volume": cmc_data.get("defi_volume_24h"),
        "active_coins": cg_data.get("active_cryptocurrencies") or cmc_data.get("active_cryptocurrencies"),
        "market_cap_change_24h_pct": cg_data.get("market_cap_change_24h_pct"),
    }
    resp = build_response("get_global_market_context", "MARKET", data, sources_used, sources_failed)
    cache.set("global_context", {}, resp, CACHE_TTL_GLOBAL)
    return resp


async def tool_get_correlations(symbol: str, lookback_days: int = 30) -> dict:
    """Coin correlation with BTC and ETH. High correlation means coin follows BTC."""
    try:
        coin_chart = await coingecko.get_market_chart(symbol, lookback_days)
        btc_chart = await coingecko.get_market_chart("bitcoin", lookback_days)
        eth_chart = await coingecko.get_market_chart("ethereum", lookback_days)

        def _daily_returns(prices_list):
            p = [x[1] for x in prices_list if x[1]]
            arr = np.array(p)
            return np.diff(arr) / arr[:-1] if len(arr) > 1 else np.array([])

        coin_ret = _daily_returns(coin_chart.get("prices", []))
        btc_ret = _daily_returns(btc_chart.get("prices", []))
        eth_ret = _daily_returns(eth_chart.get("prices", []))

        min_len = min(len(coin_ret), len(btc_ret), len(eth_ret))
        if min_len < 5:
            return build_error_response("get_correlations", symbol, "Insufficient data for correlation")

        coin_ret = coin_ret[-min_len:]
        btc_ret = btc_ret[-min_len:]
        eth_ret = eth_ret[-min_len:]

        btc_corr_full = float(np.corrcoef(coin_ret, btc_ret)[0, 1])
        eth_corr_full = float(np.corrcoef(coin_ret, eth_ret)[0, 1])

        # 7-day correlation
        min7 = min(7, min_len)
        btc_corr_7d = float(np.corrcoef(coin_ret[-min7:], btc_ret[-min7:])[0, 1])
        eth_corr_7d = float(np.corrcoef(coin_ret[-min7:], eth_ret[-min7:])[0, 1])

        independence = round(1 - abs(btc_corr_full), 2)
        if abs(btc_corr_full) > 0.8:
            interp = f"Highly correlated with BTC ({btc_corr_full:.2f}). Price largely follows BTC."
        elif abs(btc_corr_full) > 0.5:
            interp = f"Moderately correlated with BTC ({btc_corr_full:.2f}). Partially independent."
        else:
            interp = f"Low BTC correlation ({btc_corr_full:.2f}). Coin has independent price action."

        data = {
            "btc_correlation_30d": round(btc_corr_full, 4),
            "eth_correlation_30d": round(eth_corr_full, 4),
            "btc_correlation_7d": round(btc_corr_7d, 4),
            "eth_correlation_7d": round(eth_corr_7d, 4),
            "independence_score": independence,
            "interpretation": interp,
        }
        return build_response("get_correlations", symbol, data, ["coingecko"])
    except Exception as e:
        return build_error_response("get_correlations", symbol, str(e))


async def tool_get_exchange_listings(coingecko_id: str) -> dict:
    """Which exchanges list the coin, with liquidity per exchange."""
    try:
        tickers = await coingecko.get_coin_tickers(coingecko_id)
        total_liq = sum(t.get("volume_24h_usd", 0) for t in tickers)
        primary = tickers[0]["exchange"] if tickers else None

        listings = [{
            "exchange": t["exchange"],
            "trading_pair": t["pair"],
            "volume_24h_usdt": t.get("volume_24h_usd"),
            "trust_score": t.get("trust_score"),
        } for t in tickers[:20]]

        data = {
            "listings": listings,
            "total_liquidity_usdt": round(total_liq, 2),
            "primary_exchange": primary,
            "low_liquidity_flag": total_liq < 100000,
            "exchange_count": len(tickers),
        }
        return build_response("get_exchange_listings", coingecko_id, data, ["coingecko"])
    except Exception as e:
        return build_error_response("get_exchange_listings", coingecko_id, str(e))


async def tool_get_upcoming_events(coingecko_id: str, symbol: str = "") -> dict:
    """Scheduled events that can move price — token unlocks, vesting cliffs.
    Uses TokenUnlocks + CryptoRank APIs for real data.
    """
    try:
        sym = symbol or coingecko_id
        unlock_data = await onchain.get_token_unlocks(sym)
        unlocks = unlock_data.get("upcoming_unlocks", [])

        # Compute days until next unlock
        import datetime
        days_until = None
        next_event = None
        for u in unlocks:
            date_str = u.get("date")
            if date_str:
                try:
                    dt = datetime.datetime.fromisoformat(date_str.replace("Z", "+00:00"))
                    delta = (dt - datetime.datetime.now(datetime.timezone.utc)).days
                    if delta >= 0:
                        if days_until is None or delta < days_until:
                            days_until = delta
                            next_event = u
                except (ValueError, TypeError):
                    pass

        data = {
            "events": unlocks[:5],
            "next_major_event": next_event,
            "days_until_next_event": days_until,
            "source": unlock_data.get("source", "none"),
            "total_locked_pct": unlock_data.get("total_locked_pct"),
            "supply_pressure_warning": (
                days_until is not None and days_until <= 14 and
                (next_event or {}).get("pct_of_supply", 0) and
                safe_float((next_event or {}).get("pct_of_supply")) > 1.0
            ),
        }
        return build_response("get_upcoming_events", coingecko_id, data,
                              [unlock_data.get("source", "none")])
    except Exception as e:
        return build_error_response("get_upcoming_events", coingecko_id, str(e))


async def tool_get_top_holders(contract_address: str, chain: str = "eth") -> dict:
    """Wallet concentration — high whale concentration = higher manipulation risk."""
    try:
        result = await onchain.get_top_token_holders(contract_address, chain)
        wallets = result.get("top_wallets", [])

        total_pct_10 = sum(w.get("pct", 0) for w in wallets[:10])
        total_pct_20 = sum(w.get("pct", 0) for w in wallets[:20])

        if total_pct_10 > 50:
            risk = "high"
        elif total_pct_10 > 30:
            risk = "medium"
        else:
            risk = "low"

        data = {
            "top10_holders_pct": round(total_pct_10, 2),
            "top20_holders_pct": round(total_pct_20, 2),
            "top_wallets": wallets[:20],
            "concentration_risk": risk,
            "chain": chain,
        }
        return build_response("get_top_holders", contract_address, data, [result.get("source", chain)])
    except Exception as e:
        return build_error_response("get_top_holders", contract_address, str(e))


# ── CATEGORY 6: Risk Metrics ────────────────────────────────────

async def tool_calculate_risk_metrics(symbol: str, lookback_days: int = 30) -> dict:
    """Statistical risk profile: Sharpe, Sortino, max drawdown, VaR, beta vs BTC."""
    try:
        candles = await binance.get_klines(symbol, "1d", max(lookback_days, 90))
        btc_candles = await binance.get_klines("BTC", "1d", max(lookback_days, 90))

        closes = np.array([c["close"] for c in candles], dtype=float)
        btc_closes = np.array([c["close"] for c in btc_candles], dtype=float)

        if len(closes) < 10:
            return build_error_response("calculate_risk_metrics", symbol, "Insufficient data")

        returns = np.diff(closes) / closes[:-1]
        btc_returns = np.diff(btc_closes) / btc_closes[:-1]

        # Trim to same length
        min_len = min(len(returns), len(btc_returns))
        returns = returns[-min_len:]
        btc_returns = btc_returns[-min_len:]

        # Use last N days
        ret_30 = returns[-lookback_days:] if len(returns) >= lookback_days else returns
        ret_90 = returns[-90:] if len(returns) >= 90 else returns
        btc_30 = btc_returns[-lookback_days:] if len(btc_returns) >= lookback_days else btc_returns

        avg_ret = float(np.mean(ret_30))
        std_ret = float(np.std(ret_30))

        # Sharpe (annualized, 0% risk-free)
        sharpe = round((avg_ret / std_ret) * np.sqrt(365), 4) if std_ret > 0 else 0

        # Sortino (downside deviation only)
        downside = ret_30[ret_30 < 0]
        downside_std = float(np.std(downside)) if len(downside) > 0 else std_ret
        sortino = round((avg_ret / downside_std) * np.sqrt(365), 4) if downside_std > 0 else 0

        # Max drawdown
        def _max_dd(prices):
            peak = prices[0]
            max_dd = 0
            for p in prices:
                if p > peak:
                    peak = p
                dd = (peak - p) / peak
                if dd > max_dd:
                    max_dd = dd
            return round(max_dd * 100, 2)

        dd_30 = _max_dd(closes[-lookback_days:]) if len(closes) >= lookback_days else _max_dd(closes)
        dd_90 = _max_dd(closes[-90:]) if len(closes) >= 90 else _max_dd(closes)

        # VaR 95%
        var_95 = round(float(np.percentile(ret_30, 5)) * 100, 4)

        # Avg daily volatility
        avg_vol = round(std_ret * 100, 4)

        # Beta vs BTC
        cov = np.cov(ret_30[-min(len(ret_30), len(btc_30)):], btc_30[-min(len(ret_30), len(btc_30)):])[0][1]
        btc_var = np.var(btc_30[-min(len(ret_30), len(btc_30)):])
        beta = round(float(cov / btc_var), 4) if btc_var > 0 else 1.0

        # Calmar ratio
        calmar = round((avg_ret * 365) / (dd_30 / 100), 4) if dd_30 > 0 else 0

        # Risk category
        if avg_vol < 2:
            risk_cat = "low"
        elif avg_vol < 5:
            risk_cat = "medium"
        elif avg_vol < 10:
            risk_cat = "high"
        else:
            risk_cat = "extreme"

        data = {
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "max_drawdown_30d_pct": dd_30,
            "max_drawdown_90d_pct": dd_90,
            "daily_var_95_pct": var_95,
            "avg_daily_volatility_pct": avg_vol,
            "beta_vs_btc": beta,
            "relative_strength_30d_pct": round(float(np.sum(ret_30) - np.sum(btc_30)) * 100, 4) if len(ret_30) > 0 and len(btc_30) > 0 else 0,
            "calmar_ratio": calmar,
            "risk_category": risk_cat,
        }
        return build_response("calculate_risk_metrics", symbol, data, ["binance"])
    except Exception as e:
        return build_error_response("calculate_risk_metrics", symbol, str(e))


async def tool_compute_entry_zones(symbol: str, current_price: float, direction: str = "long", risk_tolerance: str = "moderate") -> dict:
    """Calculate optimal entry zones, stop loss, and take profit levels using ATR methodology."""
    try:
        candles = await binance.get_klines(symbol, "4h", 200)
        if not candles:
            candles = await mexc.get_klines(symbol, "4h", 200)

        from indicators import compute_all_indicators
        from patterns import compute_support_resistance
        ind = compute_all_indicators(candles)
        sr = compute_support_resistance(candles)

        atr = ind.get("atr") or (current_price * 0.02)
        vwap = ind.get("vwap") or current_price

        # ATR multipliers by risk tolerance
        sl_mult = {"conservative": 1.5, "moderate": 2.0, "aggressive": 2.5}.get(risk_tolerance, 2.0)

        if direction == "long":
            nearest_support = sr.get("nearest_support") or (current_price - atr)
            entry_ideal = min(vwap, current_price)
            entry_lower = entry_ideal - atr * 0.5
            entry_upper = entry_ideal + atr * 0.3
            sl = entry_ideal - (atr * sl_mult)
            sl = min(sl, nearest_support * 0.995)  # Just below support
            risk_per_unit = entry_ideal - sl
        else:
            nearest_resistance = sr.get("nearest_resistance") or (current_price + atr)
            entry_ideal = max(vwap, current_price)
            entry_lower = entry_ideal - atr * 0.3
            entry_upper = entry_ideal + atr * 0.5
            sl = entry_ideal + (atr * sl_mult)
            sl = max(sl, nearest_resistance * 1.005)
            risk_per_unit = sl - entry_ideal

        if risk_per_unit <= 0:
            risk_per_unit = atr

        # Take profit levels
        tp1 = entry_ideal + risk_per_unit * 1.5 if direction == "long" else entry_ideal - risk_per_unit * 1.5
        tp2 = entry_ideal + risk_per_unit * 2.5 if direction == "long" else entry_ideal - risk_per_unit * 2.5
        tp3 = entry_ideal + risk_per_unit * 4.0 if direction == "long" else entry_ideal - risk_per_unit * 4.0

        pct_from_entry = abs(risk_per_unit / entry_ideal) * 100

        data = {
            "direction": direction,
            "risk_tolerance": risk_tolerance,
            "entry_zone": {
                "ideal": round(entry_ideal, 8),
                "lower_bound": round(entry_lower, 8),
                "upper_bound": round(entry_upper, 8),
            },
            "stop_loss": {
                "price": round(sl, 8),
                "pct_from_entry": round(pct_from_entry, 2),
                "invalidation_reason": f"{'Below' if direction == 'long' else 'Above'} key structural level + {sl_mult}x ATR",
            },
            "tp1": {"price": round(tp1, 8), "rr_ratio": "1:1.5", "exit_pct": "40%"},
            "tp2": {"price": round(tp2, 8), "rr_ratio": "1:2.5", "exit_pct": "35%"},
            "tp3": {"price": round(tp3, 8), "rr_ratio": "1:4.0", "exit_pct": "25%"},
            "atr_used": round(atr, 8),
            "position_size_pct_of_capital_recommended": 2 if risk_tolerance != "conservative" else 1,
            "max_loss_per_trade_usdt_example": round(10000 * 0.02 if risk_tolerance != "conservative" else 10000 * 0.01, 2),
        }
        return build_response("compute_entry_zones", symbol, data, ["binance"])
    except Exception as e:
        return build_error_response("compute_entry_zones", symbol, str(e))


# ── CATEGORY 7: Master Aggregation ──────────────────────────────

async def tool_full_coin_intelligence_report(query: str, exchanges: list = None) -> dict:
    """THE MASTER TOOL — runs the complete data collection pipeline and returns a structured intelligence brief.
    Timeout per sub-call: 10s. Total: ~90s. Failed sections marked as null with reason."""
    if exchanges is None:
        exchanges = ["binance", "mexc"]

    from tools_core import (tool_search_coin, tool_get_coin_metadata, tool_get_recent_news, tool_get_spot_price,
                            tool_get_ohlcv_history, tool_compute_technical_indicators,
                            tool_detect_chart_patterns, tool_compute_support_resistance,
                            tool_get_order_book_depth, tool_get_recent_trades,
                            tool_get_fear_greed_index, tool_get_market_sentiment,
                            tool_get_funding_rates, tool_get_open_interest,
                            tool_get_whale_activity, tool_get_onchain_metrics)

    report = {}
    failed_tools = []

    async def _safe(name, coro, timeout: float = 45):
        try:
            return await asyncio.wait_for(coro, timeout=timeout)
        except Exception as e:
            failed_tools.append({"tool": name, "error": str(e)})
            return None

    # Step 1: Resolve coin
    search = await _safe("search_coin", tool_search_coin(query))
    if not search or not search.get("success"):
        return build_error_response("full_coin_intelligence_report", query, "Could not resolve coin")

    sd = search["data"]
    symbol = sd.get("symbol", query.upper())
    cg_id = sd.get("coingecko_id", query.lower())
    report["coin_profile"] = sd

    # Step 2: Parallel baseline
    meta_t = _safe("get_coin_metadata", tool_get_coin_metadata(cg_id))
    price_t = _safe("get_spot_price", tool_get_spot_price(symbol, cg_id))
    fg_t = _safe("get_fear_greed_index", tool_get_fear_greed_index())
    global_t = _safe("get_global_market_context", tool_get_global_market_context())

    meta, price, fg, global_ctx = await asyncio.gather(meta_t, price_t, fg_t, global_t)
    report["metadata"] = meta.get("data") if meta else None
    report["current_price"] = price.get("data") if price else None
    report["macro_context"] = {
        "fear_greed": fg.get("data") if fg else None,
        "global_market": global_ctx.get("data") if global_ctx else None,
    }

    # Step 3: Technical analysis
    ta_t = _safe("compute_technical_indicators", tool_compute_technical_indicators(symbol))
    patterns_t = _safe("detect_chart_patterns", tool_detect_chart_patterns(symbol))
    sr_t = _safe("compute_support_resistance", tool_compute_support_resistance(symbol))

    ta_res, patterns, sr = await asyncio.gather(ta_t, patterns_t, sr_t)
    report["technical_analysis"] = ta_res.get("data") if ta_res else None
    report["chart_patterns"] = patterns.get("data") if patterns else None
    sr_data = sr.get("data") if sr else None
    report["support_resistance"] = sr_data
    
    # Generate Multimodal Chart
    try:
        from fetchers import binance
        candles = await binance.get_klines(symbol, "4h", 200)
        if candles and sr_data:
            chart_path = generate_chart(candles, sr_data, symbol)
            report["chart_image_path"] = chart_path
    except Exception as e:
        report["chart_image_path"] = None
        failed_tools.append({"tool": "chart_generation", "error": str(e)})

    # Step 2: Price history (Parallel)
    t_15m = _safe("ohlcv_15m", tool_get_ohlcv_history(symbol, "15m", 200))
    t_1h = _safe("ohlcv_1h", tool_get_ohlcv_history(symbol, "1h", 200))
    t_4h = _safe("ohlcv_4h", tool_get_ohlcv_history(symbol, "4h", 200))
    t_1d = _safe("ohlcv_1d", tool_get_ohlcv_history(symbol, "1d", 200))
    t_1m = _safe("ohlcv_1m", tool_get_ohlcv_history(symbol, "1m", 1500))  # For V5 Quant Engine
    
    r_15m, r_1h, r_4h, r_1d, r_1m = await asyncio.gather(t_15m, t_1h, t_4h, t_1d, t_1m)
    report["technical_analysis"] = {
        "15m": r_15m.get("data") if r_15m else None,
        "1h": r_1h.get("data") if r_1h else None,
        "4h": r_4h.get("data") if r_4h else None,
        "1d": r_1d.get("data") if r_1d else None,
        "1m": r_1m.get("data") if r_1m else None,
    }

    # Step 4: Microstructure + derivatives
    ob_t = _safe("get_order_book_depth", tool_get_order_book_depth(symbol))
    rt_t = _safe("get_recent_trades", tool_get_recent_trades(symbol))
    fr_t = _safe("get_funding_rates", tool_get_funding_rates(symbol))
    oi_t = _safe("get_open_interest", tool_get_open_interest(symbol))

    ob, rt, fr, oi = await asyncio.gather(ob_t, rt_t, fr_t, oi_t)
    report["order_book"] = ob.get("data") if ob else None
    report["trade_flow"] = rt.get("data") if rt else None
    report["derivatives"] = {
        "funding_rates": fr.get("data") if fr else None,
        "open_interest": oi.get("data") if oi else None,
    }

    # Step 5: Sentiment + risk
    whale_t = _safe("get_whale_activity", tool_get_whale_activity(symbol, cg_id))
    sent_t = _safe("get_market_sentiment", tool_get_market_sentiment(cg_id))
    news_t = _safe("get_recent_news", tool_get_recent_news(symbol))
    corr_t = _safe("get_correlations", tool_get_correlations(cg_id, 30))
    risk_t = _safe("calculate_risk_metrics", tool_calculate_risk_metrics(symbol, 30))

    whale, sent, corr, risk, news = await asyncio.gather(whale_t, sent_t, corr_t, risk_t, news_t)
    report["whale_and_onchain"] = whale.get("data") if whale else None
    report["sentiment"] = sent.get("data") if sent else None
    report["news"] = news.get("data") if news else None
    report["correlations"] = corr.get("data") if corr else None
    report["risk_metrics"] = risk.get("data") if risk else None

    # Step 6: NEW — Institutional-grade data (OI delta, liquidations, funding spread, stablecoins, unlocks)
    from tools_core import (tool_get_oi_delta, tool_get_cross_exchange_funding,
                            tool_get_liquidation_data, tool_get_stablecoin_flows,
                            tool_get_token_unlocks)

    oi_delta_t = _safe("get_oi_delta", tool_get_oi_delta(symbol))
    xfunding_t = _safe("get_cross_exchange_funding", tool_get_cross_exchange_funding(symbol))
    liq_t = _safe("get_liquidation_data", tool_get_liquidation_data(symbol))
    stable_t = _safe("get_stablecoin_flows", tool_get_stablecoin_flows())
    unlock_t = _safe("get_token_unlocks", tool_get_token_unlocks(symbol))

    oi_delta, xfunding, liq, stable, unlocks = await asyncio.gather(
        oi_delta_t, xfunding_t, liq_t, stable_t, unlock_t)
    report["oi_delta"] = oi_delta.get("data") if oi_delta else None
    report["cross_exchange_funding"] = xfunding.get("data") if xfunding else None
    report["liquidation_data"] = liq.get("data") if liq else None
    report["stablecoin_flows"] = stable.get("data") if stable else None
    report["token_unlocks"] = unlocks.get("data") if unlocks else None

    # Step 7: Entry zones
    avg_price = None
    if report.get("current_price") and report["current_price"].get("average_price"):
        avg_price = report["current_price"]["average_price"]

    if avg_price:
        long_t = _safe("compute_entry_zones_long", tool_compute_entry_zones(symbol, avg_price, "long", "moderate"))
        short_t = _safe("compute_entry_zones_short", tool_compute_entry_zones(symbol, avg_price, "short", "moderate"))
        long_zones, short_zones = await asyncio.gather(long_t, short_t)
        report["entry_zones"] = {
            "long": long_zones.get("data") if long_zones else None,
            "short": short_zones.get("data") if short_zones else None,
        }

    # Collection metadata
    total_sections = 20  # increased for 1m klines
    successful = total_sections - len(failed_tools)
    report["collection_metadata"] = {
        "coin": sd.get("name", query),
        "symbol": symbol,
        "collection_timestamp": utc_now_iso(),
        "data_completeness_pct": round((successful / total_sections) * 100),
        "failed_tools": failed_tools,
    }

    # Attach ORACLE engine analysis (Phases 1-9) to the master report
    try:
        from oracle_engine import analyze as _oracle_analyze
        report["oracle_analysis"] = _oracle_analyze(report)
    except Exception as e:
        report["oracle_analysis"] = {"error": f"oracle_engine failed: {e}"}

    return build_response("full_coin_intelligence_report", symbol, report, ["binance", "coingecko", "cmc"], [f["tool"] for f in failed_tools])


# ── ORACLE Intelligence Brief (final wrapper) ────────────────────

async def tool_oracle_intelligence_brief(query: str, exchanges: list = None, format: str = "both") -> dict:
    """ORACLE — fused Wolf/Insider/Quant analysis brief.

    Runs the full intelligence pipeline and executes Phases 1-9 of the ORACLE
    spec (regime detection, MTF signal engine, smart-money DPI+WFI, pattern
    confidence, macro adjustment, composite score, Bayesian scenarios,
    trade plan with Kelly sizing, risk audit).

    Returns the structured analysis plus a fully formatted Markdown brief
    that matches prompts/analysis_oracle.md.

    format: "markdown" | "json" | "both" (default)
    """
    from oracle_engine import analyze as _oracle_analyze

    full = await tool_full_coin_intelligence_report(query, exchanges)
    if not full.get("success"):
        return full

    report = full.get("data") or {}
    analysis = report.get("oracle_analysis") or _oracle_analyze(report)

    payload: dict = {
        "symbol": analysis.get("symbol"),
        "name": analysis.get("name"),
        "signal": analysis.get("composite", {}).get("signal"),
        "final_score": analysis.get("composite", {}).get("final_score"),
        "regime": analysis.get("regime", {}).get("regime"),
        "ev_24h_pct": analysis.get("scenarios", {}).get("ev_24h_pct"),
        "data_completeness_pct": analysis.get("data_completeness_pct"),
    }
    if format in ("json", "both"):
        payload["analysis"] = {k: v for k, v in analysis.items() if k != "markdown"}
    if format in ("markdown", "both"):
        payload["brief_markdown"] = analysis.get("markdown")

    return build_response("oracle_intelligence_brief",
                          analysis.get("symbol") or query,
                          payload,
                          (full.get("data_quality") or {}).get("sources_used") or ["binance", "coingecko", "cmc"],
                          (full.get("data_quality") or {}).get("sources_failed") or [])
