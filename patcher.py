import os

# 1. Update tools_core.py
with open("tools_core.py", "r", encoding="utf-8") as f:
    core_code = f.read()

# Add get_recent_news
news_tool = """
import feedparser

async def tool_get_recent_news(symbol: str) -> dict:
    \"\"\"Get recent news headlines for the coin to catch narratives and catalysts.\"\"\"
    try:
        url = f"https://cryptopanic.com/news/rss/search/?q={symbol}"
        feed = feedparser.parse(url)
        headlines = []
        for entry in feed.entries[:5]:
            headlines.append({"title": entry.title, "link": entry.link, "published": entry.published})
        return build_response("get_recent_news", symbol, {"news": headlines}, ["cryptopanic"])
    except Exception as e:
        return build_error_response("get_recent_news", symbol, str(e))
"""
if "tool_get_recent_news" not in core_code:
    core_code += "\n" + news_tool

# Fix timeframes
core_code = core_code.replace('["15m", "1h", "4h", "1d"]', '["15m", "1h", "4h", "1d", "1w"]')

# Fix funding rates
old_funding = """        rates = await binance.get_funding_rate(symbol, limit=8)
        if not rates:
            return build_error_response("get_funding_rates", symbol, "No funding data")

        current = rates[-1]["funding_rate"]
        avg_24h = sum(r["funding_rate"] for r in rates) / len(rates)"""

new_funding = """        rates = await binance.get_funding_rate(symbol, limit=21)
        if not rates:
            return build_error_response("get_funding_rates", symbol, "No funding data")

        current = rates[-1]["funding_rate"]
        avg_24h = sum(r["funding_rate"] for r in rates[-3:]) / len(rates[-3:]) if len(rates) >= 3 else current
        avg_7d = sum(r["funding_rate"] for r in rates) / len(rates)"""

core_code = core_code.replace(old_funding, new_funding)
core_code = core_code.replace('"avg_24h_funding": round(avg_24h * 100, 6),', '"avg_24h_funding": round(avg_24h * 100, 6),\n            "avg_7d_funding": round(avg_7d * 100, 6),')

with open("tools_core.py", "w", encoding="utf-8") as f:
    f.write(core_code)

print("Updated tools_core.py")

# 2. Update patterns.py (VPVR)
with open("patterns.py", "r", encoding="utf-8") as f:
    pat_code = f.read()

vpvr_code = """
def compute_vpvr(df: pd.DataFrame, bins: int = 50) -> dict:
    min_p = df['low'].min()
    max_p = df['high'].max()
    if max_p == min_p: return {"poc": min_p, "vpvr_levels": []}
    bin_size = (max_p - min_p) / bins
    
    vpvr = np.zeros(bins)
    price_levels = np.linspace(min_p, max_p, bins)
    
    for _, row in df.iterrows():
        low_bin = max(0, int((row['low'] - min_p) / bin_size))
        high_bin = min(bins - 1, int((row['high'] - min_p) / bin_size))
        vol = row.get('volume', 0)
        if high_bin >= low_bin:
            v_per_bin = vol / max(1, (high_bin - low_bin + 1))
            for b in range(low_bin, high_bin + 1):
                if b < bins: vpvr[b] += v_per_bin
                
    poc_idx = np.argmax(vpvr)
    poc_price = price_levels[poc_idx]
    
    # Find local maxima for S/R
    peaks = []
    mean_v = np.mean(vpvr)
    for i in range(1, bins - 1):
        if vpvr[i] > vpvr[i-1] and vpvr[i] > vpvr[i+1] and vpvr[i] > mean_v:
            peaks.append({"level": round(float(price_levels[i]), 8), "volume": float(vpvr[i])})
            
    peaks.sort(key=lambda x: x["volume"], reverse=True)
    return {"poc": float(poc_price), "vpvr_levels": peaks[:10]}
"""

if "def compute_vpvr" not in pat_code:
    pat_code = pat_code.replace("def compute_support_resistance(", vpvr_code + "\ndef compute_support_resistance(")

    # Inject VPVR into compute_support_resistance
    old_ret = "    return {\n        \"strong_supports\": supports[:5],"
    new_ret = "    vpvr_data = compute_vpvr(df)\n    return {\n        \"vpvr_poc\": vpvr_data['poc'],\n        \"vpvr_levels\": vpvr_data['vpvr_levels'],\n        \"strong_supports\": supports[:5],"
    pat_code = pat_code.replace(old_ret, new_ret)

with open("patterns.py", "w", encoding="utf-8") as f:
    f.write(pat_code)

print("Updated patterns.py")

# 3. Update tools_advanced.py (Relative Strength, News, Master Report)
with open("tools_advanced.py", "r", encoding="utf-8") as f:
    adv_code = f.read()

# Add tool_get_recent_news import
old_import = "from tools_core import (tool_search_coin, tool_get_coin_metadata"
new_import = "from tools_core import (tool_search_coin, tool_get_coin_metadata, tool_get_recent_news"
adv_code = adv_code.replace(old_import, new_import)

# Call news tool
old_news = 'sent_t = _safe("get_market_sentiment", tool_get_market_sentiment(cg_id))'
new_news = old_news + '\n    news_t = _safe("get_recent_news", tool_get_recent_news(symbol))'
adv_code = adv_code.replace(old_news, new_news)

old_gather = "whale, sent, corr, risk = await asyncio.gather(whale_t, sent_t, corr_t, risk_t)"
new_gather = "whale, sent, corr, risk, news = await asyncio.gather(whale_t, sent_t, corr_t, risk_t, news_t)"
adv_code = adv_code.replace(old_gather, new_gather)

adv_code = adv_code.replace('report["sentiment"] = sent.get("data") if sent else None', 
                            'report["sentiment"] = sent.get("data") if sent else None\n    report["news"] = news.get("data") if news else None')

# Add Relative Strength to calculate_risk_metrics
old_risk = '"beta_vs_btc": beta,'
new_risk = '"beta_vs_btc": beta,\n            "relative_strength_30d_pct": round(float(np.sum(ret_30) - np.sum(btc_30)) * 100, 4) if len(ret_30) > 0 and len(btc_30) > 0 else 0,'
adv_code = adv_code.replace(old_risk, new_risk)

with open("tools_advanced.py", "w", encoding="utf-8") as f:
    f.write(adv_code)

print("Updated tools_advanced.py")
