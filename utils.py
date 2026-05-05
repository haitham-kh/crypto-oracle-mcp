from __future__ import annotations
"""
CryptoOracle MCP — Utilities Module
Retry logic, rate limiting, response formatting, caching
"""

import time
import asyncio
import hashlib
import json
import logging
from datetime import datetime, timezone
from functools import wraps
from typing import Any, Dict, Optional

import aiohttp

logger = logging.getLogger("crypto-oracle-mcp")

# ---------------------------------------------------------------------------
# In-memory cache with TTL
# ---------------------------------------------------------------------------

class TTLCache:
    """Simple in-memory cache with per-key TTL (seconds)."""

    def __init__(self):
        self._store: Dict[str, Dict[str, Any]] = {}

    def _make_key(self, namespace: str, params: dict) -> str:
        raw = f"{namespace}:{json.dumps(params, sort_keys=True)}"
        return hashlib.md5(raw.encode()).hexdigest()

    def get(self, namespace: str, params: dict) -> Optional[Any]:
        key = self._make_key(namespace, params)
        entry = self._store.get(key)
        if entry is None:
            return None
        if time.time() > entry["expires_at"]:
            del self._store[key]
            return None
        return entry["value"]

    def set(self, namespace: str, params: dict, value: Any, ttl_seconds: int):
        key = self._make_key(namespace, params)
        self._store[key] = {
            "value": value,
            "expires_at": time.time() + ttl_seconds,
        }

    def invalidate(self, namespace: str, params: dict):
        key = self._make_key(namespace, params)
        self._store.pop(key, None)

    def clear(self):
        self._store.clear()


# Global cache instance
cache = TTLCache()

# Cache TTL constants (seconds)
CACHE_TTL_PRICE = 30          # 30 seconds for live price
CACHE_TTL_OHLCV = 60          # 1 minute for candle data
CACHE_TTL_METADATA = 3600     # 1 hour for coin metadata
CACHE_TTL_ORDERBOOK = 10      # 10 seconds for order book
CACHE_TTL_SENTIMENT = 300     # 5 minutes for sentiment
CACHE_TTL_GLOBAL = 120        # 2 minutes for global market data
CACHE_TTL_FEAR_GREED = 600    # 10 minutes for fear & greed

# ---------------------------------------------------------------------------
# Rate Limiter
# ---------------------------------------------------------------------------

class RateLimiter:
    """Token-bucket rate limiter for API calls."""

    def __init__(self, max_calls: int, period_seconds: float):
        self.max_calls = max_calls
        self.period = period_seconds
        self._timestamps: list[float] = []
        self._lock = asyncio.Lock()

    async def acquire(self):
        async with self._lock:
            now = time.time()
            # Remove timestamps outside the window
            self._timestamps = [t for t in self._timestamps if now - t < self.period]
            if len(self._timestamps) >= self.max_calls:
                wait_time = self.period - (now - self._timestamps[0])
                if wait_time > 0:
                    logger.debug(f"Rate limited — waiting {wait_time:.2f}s")
                    await asyncio.sleep(wait_time)
            self._timestamps.append(time.time())


# Per-exchange rate limiters
rate_limiters = {
    "binance": RateLimiter(max_calls=1100, period_seconds=60),
    "mexc": RateLimiter(max_calls=450, period_seconds=60),
    "coingecko": RateLimiter(max_calls=25, period_seconds=60),
    "cmc": RateLimiter(max_calls=25, period_seconds=60),
    "glassnode": RateLimiter(max_calls=10, period_seconds=60),
    "default": RateLimiter(max_calls=30, period_seconds=60),
}


def get_rate_limiter(source: str) -> RateLimiter:
    return rate_limiters.get(source, rate_limiters["default"])

# ---------------------------------------------------------------------------
# Async HTTP helpers with retry + exponential backoff
# ---------------------------------------------------------------------------

async def fetch_json(
    url: str,
    *,
    source: str = "default",
    params: Optional[dict] = None,
    headers: Optional[dict] = None,
    retries: int = 3,
    backoff_base: float = 1.0,
    timeout_seconds: float = 10.0,
) -> dict:
    """
    Fetch JSON from a URL with:
      - Rate limiting per source
      - Retries with exponential backoff
      - Timeout
    Returns parsed JSON dict, or raises after all retries fail.
    """
    limiter = get_rate_limiter(source)
    last_exc = None

    for attempt in range(1, retries + 1):
        try:
            await limiter.acquire()
            timeout = aiohttp.ClientTimeout(total=timeout_seconds)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url, params=params, headers=headers) as resp:
                    if resp.status == 429:
                        wait = backoff_base * (2 ** (attempt - 1))
                        logger.warning(f"[{source}] 429 rate limited, retry in {wait:.1f}s")
                        await asyncio.sleep(wait)
                        continue
                    # Permanent client errors (symbol not listed, bad params, unauthorized,
                    # not found): do NOT retry — they will never succeed and waste the
                    # MCP transport's timeout budget. Fail fast so _safe() wrappers can
                    # mark the section unavailable and move on.
                    if 400 <= resp.status < 500 and resp.status != 429:
                        body_preview = ""
                        try:
                            body_preview = (await resp.text())[:200]
                        except Exception:
                            pass
                        logger.info(f"[{source}] {resp.status} (no retry) for {url} — {body_preview}")
                        raise aiohttp.ClientResponseError(
                            resp.request_info, resp.history,
                            status=resp.status,
                            message=f"{resp.status} {resp.reason} (permanent)",
                            headers=resp.headers,
                        )
                    resp.raise_for_status()
                    return await resp.json()
        except aiohttp.ClientResponseError as exc:
            # Permanent 4xx — surface immediately, do not retry.
            if 400 <= (exc.status or 0) < 500 and exc.status != 429:
                raise
            last_exc = exc
            if attempt < retries:
                wait = backoff_base * (2 ** (attempt - 1))
                logger.warning(f"[{source}] attempt {attempt} failed: {exc}, retry in {wait:.1f}s")
                await asyncio.sleep(wait)
            else:
                logger.error(f"[{source}] all {retries} attempts failed for {url}: {exc}")
        except Exception as exc:
            last_exc = exc
            if attempt < retries:
                wait = backoff_base * (2 ** (attempt - 1))
                logger.warning(f"[{source}] attempt {attempt} failed: {exc}, retry in {wait:.1f}s")
                await asyncio.sleep(wait)
            else:
                logger.error(f"[{source}] all {retries} attempts failed for {url}: {exc}")

    raise last_exc  # type: ignore[misc]

# ---------------------------------------------------------------------------
# Standard response builder
# ---------------------------------------------------------------------------

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def utc_now_ts() -> int:
    return int(datetime.now(timezone.utc).timestamp())


def build_response(
    tool: str,
    symbol: str,
    data: dict,
    sources_used: list[str] | None = None,
    sources_failed: list[str] | None = None,
    success: bool = True,
) -> dict:
    """Wrap tool output in standard envelope."""
    confidence = "high"
    if sources_failed:
        if len(sources_failed) >= len(sources_used or []):
            confidence = "low"
        else:
            confidence = "medium"

    return {
        "success": success,
        "tool": tool,
        "symbol": symbol,
        "timestamp": utc_now_ts(),
        "data": data,
        "data_quality": {
            "sources_used": sources_used or [],
            "sources_failed": sources_failed or [],
            "confidence": confidence,
        },
    }


def build_error_response(tool: str, symbol: str, error_msg: str, retry_after: int = 60) -> dict:
    return {
        "success": False,
        "tool": tool,
        "symbol": symbol,
        "timestamp": utc_now_ts(),
        "data": None,
        "error": error_msg,
        "retry_after": retry_after,
        "data_quality": {
            "sources_used": [],
            "sources_failed": ["all"],
            "confidence": "none",
        },
    }

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def safe_float(val, default=0.0) -> float:
    """Convert to float safely."""
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def safe_int(val, default=0) -> int:
    try:
        return int(val)
    except (TypeError, ValueError):
        return default


def pct_change(old: float, new: float) -> float:
    """Percentage change from old to new."""
    if old == 0:
        return 0.0
    return ((new - old) / abs(old)) * 100.0


def spread_pct(prices: list[float]) -> float:
    """Price spread as percentage of average."""
    valid = [p for p in prices if p and p > 0]
    if len(valid) < 2:
        return 0.0
    avg = sum(valid) / len(valid)
    return ((max(valid) - min(valid)) / avg) * 100.0


# ---------------------------------------------------------------------------
# State Persistence — Signal History per Coin
# ---------------------------------------------------------------------------

import os

STATE_DIR = os.path.join(os.path.dirname(__file__), ".oracle_state")


class SignalStateStore:
    """JSON-based persistence for tracking signal history per coin.
    
    Stores the last N signals for each coin to enable:
    - Signal flip detection (BUY→SELL transitions)
    - Repeated support/resistance test tracking
    - Historical accuracy measurement
    """

    def __init__(self, max_entries_per_coin: int = 50):
        self._max = max_entries_per_coin
        os.makedirs(STATE_DIR, exist_ok=True)

    def _path(self, symbol: str) -> str:
        return os.path.join(STATE_DIR, f"{symbol.upper()}_history.json")

    def load(self, symbol: str) -> list:
        path = self._path(symbol)
        if not os.path.exists(path):
            return []
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception:
            return []

    def append(self, symbol: str, entry: dict):
        """Append a signal entry: {timestamp, signal, score, price, regime, ...}"""
        history = self.load(symbol)
        entry["timestamp"] = utc_now_iso()
        history.append(entry)
        # Keep only last N entries
        history = history[-self._max:]
        try:
            with open(self._path(symbol), "w") as f:
                json.dump(history, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save signal state for {symbol}: {e}")

    def get_last_signal(self, symbol: str) -> Optional[dict]:
        history = self.load(symbol)
        return history[-1] if history else None

    def detect_signal_flip(self, symbol: str, current_signal: str) -> Optional[str]:
        """Returns the previous signal if it flipped, else None."""
        last = self.get_last_signal(symbol)
        if not last:
            return None
        prev = last.get("signal", "")
        bullish = {"STRONG_BUY", "BUY", "WEAK_BUY"}
        bearish = {"STRONG_SELL", "SELL", "WEAK_SELL"}
        if (prev in bullish and current_signal in bearish) or \
           (prev in bearish and current_signal in bullish):
            return prev
        return None


# Singleton instance
signal_store = SignalStateStore()


# ---------------------------------------------------------------------------
# Paper Trade Logger — Build Dataset for Calibration
# ---------------------------------------------------------------------------

PAPER_TRADE_FILE = os.path.join(STATE_DIR, "paper_trades.json")


class PaperTradeLogger:
    """Logs every signal + price for later backtesting calibration.
    
    Each entry records:
    - timestamp, symbol, signal, composite_score, price_at_signal
    - 24h_forward_price (filled in later by a reconciliation job)
    - actual_return_pct (computed once 24h_forward_price is known)
    
    This is THE dataset needed to calibrate probability estimates.
    """

    def __init__(self):
        os.makedirs(STATE_DIR, exist_ok=True)

    def log_signal(self, symbol: str, signal: str, score: float,
                   price: float, regime: str, conviction: float):
        entry = {
            "timestamp": utc_now_iso(),
            "symbol": symbol.upper(),
            "signal": signal,
            "composite_score": round(score, 2),
            "price_at_signal": price,
            "regime": regime,
            "conviction": round(conviction, 1),
            "24h_forward_price": None,
            "actual_return_pct": None,
            "calibrated": False,
        }
        try:
            trades = []
            if os.path.exists(PAPER_TRADE_FILE):
                with open(PAPER_TRADE_FILE, "r") as f:
                    trades = json.load(f)
            trades.append(entry)
            # Keep last 2000 entries
            trades = trades[-2000:]
            with open(PAPER_TRADE_FILE, "w") as f:
                json.dump(trades, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to log paper trade: {e}")

    def reconcile(self, symbol: str, current_price: float):
        """Fill in 24h_forward_price for entries that are now 24h+ old."""
        if not os.path.exists(PAPER_TRADE_FILE):
            return
        try:
            with open(PAPER_TRADE_FILE, "r") as f:
                trades = json.load(f)
            now = datetime.now(timezone.utc)
            modified = False
            for t in trades:
                if (t.get("symbol") == symbol.upper() and
                    not t.get("calibrated") and t.get("price_at_signal")):
                    ts = t.get("timestamp", "")
                    try:
                        entry_time = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                        hours_elapsed = (now - entry_time).total_seconds() / 3600
                        if hours_elapsed >= 24:
                            t["24h_forward_price"] = current_price
                            t["actual_return_pct"] = round(
                                (current_price - t["price_at_signal"]) / t["price_at_signal"] * 100, 4)
                            t["calibrated"] = True
                            modified = True
                    except (ValueError, TypeError):
                        pass
            if modified:
                with open(PAPER_TRADE_FILE, "w") as f:
                    json.dump(trades, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to reconcile paper trades: {e}")

    def get_calibration_stats(self) -> dict:
        """Compute empirical win rates by score bucket for calibration."""
        if not os.path.exists(PAPER_TRADE_FILE):
            return {"entries": 0, "calibrated": 0, "buckets": {}}
        try:
            with open(PAPER_TRADE_FILE, "r") as f:
                trades = json.load(f)
            calibrated = [t for t in trades if t.get("calibrated")]
            if not calibrated:
                return {"entries": len(trades), "calibrated": 0, "buckets": {}}
            # Bucket by score deciles
            buckets = {}
            for t in calibrated:
                score = t.get("composite_score", 0)
                bucket = int(score // 10) * 10
                key = f"{bucket:+d}_to_{bucket+10:+d}"
                if key not in buckets:
                    buckets[key] = {"count": 0, "wins": 0, "avg_return": 0}
                buckets[key]["count"] += 1
                ret = t.get("actual_return_pct", 0)
                buckets[key]["avg_return"] += ret
                if (score > 0 and ret > 0) or (score < 0 and ret < 0):
                    buckets[key]["wins"] += 1
            for k, v in buckets.items():
                v["win_rate_pct"] = round(v["wins"] / v["count"] * 100, 1) if v["count"] else 0
                v["avg_return"] = round(v["avg_return"] / v["count"], 3) if v["count"] else 0
            return {
                "entries": len(trades),
                "calibrated": len(calibrated),
                "buckets": dict(sorted(buckets.items())),
            }
        except Exception:
            return {"entries": 0, "calibrated": 0, "buckets": {}}


# Singleton instances
paper_logger = PaperTradeLogger()
