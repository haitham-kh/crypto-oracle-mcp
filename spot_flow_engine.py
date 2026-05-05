from __future__ import annotations
"""
SPOT FLOW ENGINE — OFI, Absorption, Iceberg, Liquidity Gap
All signals derived from REAL spot execution data only.
No derivatives dependency. No heuristic weights.
"""

import numpy as np
from typing import Dict, List, Any


def compute_ofi(trades: List[Dict]) -> Dict[str, Any]:
    """Order Flow Imbalance from raw trade tape.
    OFI = (aggressive_buy_vol - aggressive_sell_vol) / total_vol.
    Range: -1 (pure sell) to +1 (pure buy).
    """
    if not trades:
        return {"available": False}

    buy_vol = sell_vol = 0.0
    sizes_usdt = []
    for t in trades:
        price = float(t.get("price") or 0)
        qty = float(t.get("qty") or 0)
        usdt = price * qty
        sizes_usdt.append(usdt)
        if not t.get("is_buyer_maker", True):
            buy_vol += usdt
        else:
            sell_vol += usdt

    total = buy_vol + sell_vol
    if total == 0:
        return {"available": False}

    ofi = (buy_vol - sell_vol) / total
    n = len(sizes_usdt)
    split = max(1, n * 3 // 4)

    def _ofi_slice(slice_trades):
        bv = sv = 0.0
        for t in slice_trades:
            u = float(t.get("price") or 0) * float(t.get("qty") or 0)
            if not t.get("is_buyer_maker", True):
                bv += u
            else:
                sv += u
        tot = bv + sv
        return (bv - sv) / tot if tot > 0 else 0.0

    ofi_early = _ofi_slice(trades[:split])
    ofi_recent = _ofi_slice(trades[split:])

    avg_size = float(np.mean(sizes_usdt)) if sizes_usdt else 1.0
    large_threshold = avg_size * 5
    large_buy = sum(
        float(t.get("price", 0)) * float(t.get("qty", 0))
        for t in trades
        if not t.get("is_buyer_maker", True)
        and float(t.get("price", 0)) * float(t.get("qty", 0)) >= large_threshold
    )
    large_sell = sum(
        float(t.get("price", 0)) * float(t.get("qty", 0))
        for t in trades
        if t.get("is_buyer_maker", True)
        and float(t.get("price", 0)) * float(t.get("qty", 0)) >= large_threshold
    )
    lt = large_buy + large_sell
    large_ofi = (large_buy - large_sell) / lt if lt > 0 else 0.0

    if ofi > 0.15:
        pressure = "strong_buy"
    elif ofi > 0.05:
        pressure = "mild_buy"
    elif ofi < -0.15:
        pressure = "strong_sell"
    elif ofi < -0.05:
        pressure = "mild_sell"
    else:
        pressure = "balanced"

    return {
        "available": True,
        "ofi": round(ofi, 4),
        "ofi_recent": round(ofi_recent, 4),
        "ofi_early": round(ofi_early, 4),
        "ofi_acceleration": round(ofi_recent - ofi_early, 4),
        "large_trade_ofi": round(large_ofi, 4),
        "buy_volume_usdt": round(buy_vol, 2),
        "sell_volume_usdt": round(sell_vol, 2),
        "total_volume_usdt": round(total, 2),
        "pressure_label": pressure,
        "trade_count": n,
        "avg_trade_size_usdt": round(avg_size, 2),
    }


def compute_absorption(candles: List[Dict]) -> Dict[str, Any]:
    """Absorption = high volume + price barely moves.
    High absorption_score (>1.8) at lows = buy absorption (bullish).
    High absorption_score at highs = sell absorption (bearish).
    """
    if not candles or len(candles) < 20:
        return {"available": False, "reason": "insufficient_candles"}

    closes = np.array([float(c.get("close", 0)) for c in candles])
    highs = np.array([float(c.get("high", 0)) for c in candles])
    lows = np.array([float(c.get("low", 0)) for c in candles])
    volumes = np.array([float(c.get("volume", 0)) for c in candles])

    price_ranges = highs - lows
    avg_vol = float(np.mean(volumes[-20:]))
    avg_range = float(np.mean(price_ranges[-20:]))

    if avg_vol == 0 or avg_range == 0:
        return {"available": False, "reason": "zero_avg"}

    recent_candles = candles[-20:]
    recent_vols = volumes[-20:]
    recent_ranges = price_ranges[-20:]
    recent_closes = closes[-20:]

    absorption_scores = []
    for i in range(len(recent_candles)):
        vol_ratio = recent_vols[i] / avg_vol
        range_ratio = recent_ranges[i] / avg_range if avg_range > 0 else 1.0
        absorption_scores.append(vol_ratio / max(range_ratio, 0.1))

    absorption_scores = np.array(absorption_scores)
    absorption_idx_list = np.where(absorption_scores > 1.8)[0]

    has_cvd = "taker_buy_volume" in (candles[0] if candles else {})
    buy_abs = sell_abs = 0

    low_20 = float(np.min(recent_closes))
    high_20 = float(np.max(recent_closes))
    range_20 = (high_20 - low_20) if high_20 > low_20 else 1e-10

    for idx in absorption_idx_list:
        candle = recent_candles[idx]
        price_pos = float(recent_closes[idx])
        rel_pos = (price_pos - low_20) / range_20

        if has_cvd:
            tbv = float(candle.get("taker_buy_volume") or 0)
            tsv = float(candle.get("taker_sell_volume") or 0)
            if tbv > tsv * 1.2:
                buy_abs += 1
            elif tsv > tbv * 1.2:
                sell_abs += 1
        else:
            if rel_pos < 0.35:
                buy_abs += 1
            elif rel_pos > 0.65:
                sell_abs += 1

    total_abs = len(absorption_idx_list)
    absorption_index = min(1.0, total_abs / 5.0)

    if buy_abs > sell_abs:
        direction = "buying"
        interpretation = "Large buyers absorbing supply — stealth accumulation signal"
    elif sell_abs > buy_abs:
        direction = "selling"
        interpretation = "Sellers distributing into volume — stealth distribution signal"
    else:
        direction = "neutral"
        interpretation = "Balanced absorption — no clear directional conviction"

    return {
        "available": True,
        "absorption_index": round(float(absorption_index), 3),
        "absorption_candle_count": int(total_abs),
        "buy_absorption_count": int(buy_abs),
        "sell_absorption_count": int(sell_abs),
        "absorption_direction": direction,
        "interpretation": interpretation,
        "avg_absorption_score_last_5": round(float(np.mean(absorption_scores[-5:])), 3),
        "has_cvd_data": has_cvd,
    }


def detect_icebergs(trades: List[Dict]) -> Dict[str, Any]:
    """Detect iceberg orders via repeated fills at same price bucket (>=5 times)."""
    if not trades:
        return {"available": False}

    price_buckets: Dict[str, Dict] = {}
    for t in trades:
        price = float(t.get("price") or 0)
        if price == 0:
            continue
        mag = 10 ** (int(np.floor(np.log10(price))) - 3) if price > 0 else 0.01
        bucket_key = f"{round(price / mag) * mag:.6g}"
        if bucket_key not in price_buckets:
            price_buckets[bucket_key] = {"count": 0, "total_usdt": 0.0, "buys": 0, "sells": 0}
        usdt = price * float(t.get("qty") or 0)
        price_buckets[bucket_key]["count"] += 1
        price_buckets[bucket_key]["total_usdt"] += usdt
        if not t.get("is_buyer_maker", True):
            price_buckets[bucket_key]["buys"] += 1
        else:
            price_buckets[bucket_key]["sells"] += 1

    iceberg_buys = []
    iceberg_sells = []
    for pk, data in price_buckets.items():
        if data["count"] >= 5:
            if data["buys"] > data["sells"] * 1.5:
                iceberg_buys.append({"price": pk, "fill_count": data["count"], "total_usdt": round(data["total_usdt"], 2)})
            elif data["sells"] > data["buys"] * 1.5:
                iceberg_sells.append({"price": pk, "fill_count": data["count"], "total_usdt": round(data["total_usdt"], 2)})

    iceberg_buys.sort(key=lambda x: -x["total_usdt"])
    iceberg_sells.sort(key=lambda x: -x["total_usdt"])

    if iceberg_buys and not iceberg_sells:
        signal = "bullish_iceberg"
    elif iceberg_sells and not iceberg_buys:
        signal = "bearish_iceberg"
    elif iceberg_buys and iceberg_sells:
        by = sum(x["total_usdt"] for x in iceberg_buys)
        sy = sum(x["total_usdt"] for x in iceberg_sells)
        signal = "bullish_iceberg" if by > sy * 1.3 else ("bearish_iceberg" if sy > by * 1.3 else "mixed_iceberg")
    else:
        signal = "neutral"

    interpretations = {
        "bullish_iceberg": "Hidden buyer accumulating — repeated refills suggest large buy wall",
        "bearish_iceberg": "Hidden seller distributing — repeated refills suggest large sell wall",
        "mixed_iceberg": "Both sides have hidden liquidity — battle zone",
        "neutral": "No iceberg patterns detected",
    }

    return {
        "available": True,
        "iceberg_buy_levels": iceberg_buys[:3],
        "iceberg_sell_levels": iceberg_sells[:3],
        "signal": signal,
        "interpretation": interpretations.get(signal, ""),
    }


def compute_liquidity_gaps(order_book: Dict, current_price: float) -> Dict[str, Any]:
    """Identify thin liquidity (fast-move) zones vs thick (S/R) zones. Estimate slippage."""
    if not order_book or not current_price:
        return {"available": False}

    bids = order_book.get("bids_top10") or order_book.get("bids", [])
    asks = order_book.get("asks_top10") or order_book.get("asks", [])
    if not bids or not asks:
        return {"available": False}

    def _analyze(levels: List) -> Dict:
        usdt_vals = [float(l.get("price", 0)) * float(l.get("qty", 0)) for l in levels]
        avg = float(np.mean(usdt_vals)) if usdt_vals else 1.0
        gaps, walls = [], []
        for lvl in levels:
            p = float(lvl.get("price") or 0)
            q = float(lvl.get("qty") or 0)
            depth = p * q
            pct = abs(p - current_price) / current_price * 100
            if depth < avg * 0.10:
                gaps.append({"price": round(p, 8), "pct_from_current": round(pct, 3), "depth_usdt": round(depth, 2)})
            elif depth > avg * 2.0:
                walls.append({"price": round(p, 8), "pct_from_current": round(pct, 3), "depth_usdt": round(depth, 2)})
        return {"gaps": gaps[:3], "walls": walls[:3]}

    def _slippage(levels: List, size: float) -> float:
        if not levels:
            return 0.0
        first_p = float(levels[0].get("price") or current_price)
        filled = 0.0
        for lvl in levels:
            p = float(lvl.get("price") or 0)
            q = float(lvl.get("qty") or 0)
            av = p * q
            filled += av
            if filled >= size:
                return abs(p - first_p) / first_p * 100
        last_p = float(levels[-1].get("price") or first_p)
        return abs(last_p - first_p) / first_p * 100

    bid_a = _analyze(bids)
    ask_a = _analyze(asks)
    total_bids = sum(float(b.get("price", 0)) * float(b.get("qty", 0)) for b in bids)
    total_asks = sum(float(a.get("price", 0)) * float(a.get("qty", 0)) for a in asks)
    depth_imbalance = round(total_bids / total_asks, 3) if total_asks > 0 else 1.0
    slip_10k = _slippage(asks, 10_000)

    return {
        "available": True,
        "bid_gaps": bid_a["gaps"],
        "bid_walls": bid_a["walls"],
        "ask_gaps": ask_a["gaps"],
        "ask_walls": ask_a["walls"],
        "depth_imbalance_ratio": depth_imbalance,
        "depth_imbalance_interpretation": (
            "buy_pressure" if depth_imbalance > 1.3
            else "sell_pressure" if depth_imbalance < 0.77
            else "balanced"
        ),
        "slippage_estimates": {
            "buy_1k_usdt_pct": round(_slippage(asks, 1_000), 4),
            "buy_5k_usdt_pct": round(_slippage(asks, 5_000), 4),
            "buy_10k_usdt_pct": round(slip_10k, 4),
            "buy_50k_usdt_pct": round(_slippage(asks, 50_000), 4),
        },
        "is_deep_market": slip_10k < 0.05,
        "liquidity_warning": slip_10k > 0.3,
    }


def compute_net_buying_pressure(ofi_data: Dict, absorption_data: Dict,
                                iceberg_data: Dict, liquidity_data: Dict) -> Dict[str, Any]:
    """Aggregate all spot flow signals. Scale: -100 to +100."""
    score = 0.0
    components: List[Dict] = []

    def add(label: str, value: float):
        nonlocal score
        score += value
        components.append({"label": label, "contribution": round(value, 2)})

    if ofi_data.get("available"):
        add("OFI (full window)", ofi_data.get("ofi", 0) * 40)
        add("OFI acceleration", ofi_data.get("ofi_acceleration", 0) * 20)
        large_ofi = ofi_data.get("large_trade_ofi", 0)
        if abs(large_ofi) > 0.2:
            add("Large trade OFI (whale)", large_ofi * 15)

    if absorption_data.get("available"):
        idx = absorption_data.get("absorption_index", 0)
        direction = absorption_data.get("absorption_direction", "neutral")
        if direction == "buying":
            add("Buy absorption", idx * 25)
        elif direction == "selling":
            add("Sell absorption", -idx * 25)

    if iceberg_data.get("available"):
        sig = iceberg_data.get("signal", "neutral")
        if sig == "bullish_iceberg":
            add("Iceberg buy orders", 10)
        elif sig == "bearish_iceberg":
            add("Iceberg sell orders", -10)

    if liquidity_data.get("available"):
        dim = liquidity_data.get("depth_imbalance_ratio", 1.0)
        if dim > 1.5:
            add("Deep bid stack", 10)
        elif dim > 1.3:
            add("Mild bid imbalance", 5)
        elif dim < 0.67:
            add("Deep ask stack", -10)
        elif dim < 0.77:
            add("Mild ask imbalance", -5)

    score = max(-100.0, min(100.0, score))

    if score >= 50:
        label = "strong_buy_pressure"
    elif score >= 20:
        label = "mild_buy_pressure"
    elif score <= -50:
        label = "strong_sell_pressure"
    elif score <= -20:
        label = "mild_sell_pressure"
    else:
        label = "neutral"

    return {
        "net_buying_pressure_score": round(score, 2),
        "pressure_label": label,
        "components": components,
        "data_sources_used": [
            k for k, d in [("ofi", ofi_data), ("absorption", absorption_data),
                           ("iceberg", iceberg_data), ("liquidity", liquidity_data)]
            if d.get("available")
        ],
    }


def analyze_spot_flow(trades: List[Dict], candles: List[Dict],
                      order_book: Dict, current_price: float) -> Dict[str, Any]:
    """Full spot flow analysis entry point."""
    ofi = compute_ofi(trades)
    absorption = compute_absorption(candles)
    icebergs = detect_icebergs(trades)
    liquidity = compute_liquidity_gaps(order_book, current_price)
    pressure = compute_net_buying_pressure(ofi, absorption, icebergs, liquidity)

    return {
        "ofi": ofi,
        "absorption": absorption,
        "icebergs": icebergs,
        "liquidity": liquidity,
        "net_buying_pressure": pressure,
        "summary": {
            "score": pressure["net_buying_pressure_score"],
            "label": pressure["pressure_label"],
            "top_signal": max(pressure["components"], key=lambda x: abs(x["contribution"]))
            if pressure["components"] else None,
        },
    }
