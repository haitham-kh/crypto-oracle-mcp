import os

with open("tools_advanced.py", "r", encoding="utf-8") as f:
    adv_code = f.read()

# Add chart generation
if "from charting import generate_chart" not in adv_code:
    adv_code = "from charting import generate_chart\n" + adv_code

old_sr = """    ta_res, patterns, sr = await asyncio.gather(ta_t, patterns_t, sr_t)
    report["technical_analysis"] = ta_res.get("data") if ta_res else None
    report["chart_patterns"] = patterns.get("data") if patterns else None
    report["support_resistance"] = sr.get("data") if sr else None"""

new_sr = """    ta_res, patterns, sr = await asyncio.gather(ta_t, patterns_t, sr_t)
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
        failed_tools.append({"tool": "chart_generation", "error": str(e)})"""

adv_code = adv_code.replace(old_sr, new_sr)

with open("tools_advanced.py", "w", encoding="utf-8") as f:
    f.write(adv_code)

print("Injected Charting into tools_advanced.py")
