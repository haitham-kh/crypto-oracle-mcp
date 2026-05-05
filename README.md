# CryptoOracle MCP — ORACLE Edition

> Spot-trading intelligence engine that fuses **Wolf** (microstructure) + **Insider** (smart-money) + **Quant** (multi-timeframe technicals) into a single deterministic 9-phase analysis pipeline. Exposed over the Model Context Protocol (MCP) as **24 tools + 2 prompts**.

---

## 🤖 FOR AI AGENTS — How to Run a Coin Analysis

**This is the section the LLM reads when the user asks "analyze X" or "what do you think about coin Y".**

### The single rule

> **When the user asks about any coin, call `oracle_intelligence_brief` with the coin's symbol. That is the entire job. Do not chain other tools first.**

`oracle_intelligence_brief` is the master tool. It internally:

1. Resolves the coin (`search_coin` with exact-symbol/id matching),
2. Runs the full 14-section data pipeline in parallel (price, TA on 4 timeframes, patterns, S/R, order book, trade flow, derivatives, whales, on-chain, sentiment, correlations, risk, macro),
3. Executes ORACLE Phases 1–9 deterministically,
4. Returns a structured JSON block **and** a fully-formatted Markdown brief.

### Canonical invocation

```jsonc
// MCP tools/call request
{
  "name": "oracle_intelligence_brief",
  "arguments": {
    "query": "ARIA",          // symbol, name, or coingecko id
    "format": "both"           // "markdown" | "json" | "both"
  }
}
```

### Choosing the `query` value (in order of preference)

1. **Exact ticker symbol** — `BTC`, `ETH`, `ARIA`, `PEPE`. Best resolution path.
2. **CoinGecko id** — `bitcoin`, `aria-ai`, `pepe`. Use this when the user pastes a CoinGecko or CoinMarketCap URL — extract the slug from `coingecko.com/en/coins/<slug>` or `coinmarketcap.com/currencies/<slug>/`.
3. **Project name** — `Aria.AI`, `Solana`. Last resort; fuzzy-matches.

The resolver prefers an **exact symbol/id/name match** over fuzzy ranking, so `ARIA` resolves to `aria-ai` even when a higher-ranked unrelated CoinGecko entry exists.

### Choosing `format`

| Value | When to use |
|---|---|
| `"markdown"` | The user just wants a readable answer. Pull `data.brief_markdown` and paste verbatim. **Default for chat answers.** |
| `"json"` | You will post-process the numbers (charts, tables, follow-up reasoning). |
| `"both"` | You want to do both — the response carries `data.brief_markdown` and `data.analysis`. |

### Response shape

```jsonc
{
  "success": true,
  "tool": "oracle_intelligence_brief",
  "symbol": "ARIA",
  "timestamp": 1777981530,
  "data": {
    "symbol": "ARIA",
    "name": "Aria.AI",
    "signal": "NEUTRAL",            // STRONG_BUY|BUY|WEAK_BUY|NEUTRAL|WEAK_SELL|SELL|STRONG_SELL
    "final_score": -7.7,             // -100..+100, regime-multiplier applied
    "regime": "VOLATILE",            // TRENDING|RANGING|VOLATILE|CHAOTIC
    "ev_24h_pct": -0.48,             // probability-weighted 24h % move
    "data_completeness_pct": 100,
    "analysis": { /* full Phase 1-9 structured output */ },
    "brief_markdown": "# ORACLE INTELLIGENCE BRIEF — Aria.AI (ARIA)\n..."
  },
  "data_quality": {
    "sources_used": ["binance", "coingecko", "cmc", "mexc"],
    "sources_failed": [],
    "confidence": "high"
  }
}
```

### How the AI should present the answer

1. **If `format` was `markdown` or `both`:** display `data.brief_markdown` verbatim. It already follows the 11-section ORACLE template (Wolf read → Insider read → Quant MTF table → Composite scorecard → Key levels → Trade plan → Price prediction → Risk audit → Verdict).
2. **Then** add a one-line natural-language summary on top, e.g. *"Aria.AI is in a VOLATILE regime with a Bollinger squeeze on 4H. Score −7.7/100 (NEUTRAL). No edge — stand aside until either $0.0570 or $0.0601 breaks on volume."*
3. **Never** invent numbers, signal labels, or trade levels. If a field is `null` or `n/a`, say "n/a" — do not fill it from your prior knowledge.
4. **Never** say "buy" or "sell" if `signal` is `NEUTRAL`. The trade plan is intentionally blank for NEUTRAL.

### Edge cases the AI must handle

| Situation | What you'll see | What to do |
|---|---|---|
| Coin truly does not exist on any source | `success: false`, `error: "Could not resolve coin"` | Tell the user and offer to retry with a different spelling or contract address. |
| Coin not on Binance (only on MEXC) | `data.price_sources.binance: null`, but `data_completeness_pct ≥ 90` | Normal — the engine falls back to MEXC automatically. Use the brief as-is. |
| `data_completeness_pct < 60` | `risk_audit.data_completeness.status: "PARTIAL"` | Surface the warning prominently. Tell the user the read is reduced-confidence and list missing sections from `data.missing_sections`. |
| Regime is `CHAOTIC` | volatility too high for standard signals | The engine already applies a 0.5× multiplier. Emphasize "wait for vol to compress" in your wrapper sentence. |
| User asks for a follow-up like "what's the SL?" | All numbers are in `data.analysis.trade_plan` | Read from the structured payload, do not re-compute. |
| User asks "should I buy?" | | Quote the signal label and EV24H verbatim. Do not give personalized financial advice. |

### When to use the lower-level tools

Only when the user explicitly asks for a single piece of data (e.g. "what's BTC funding right now"). For any *analysis* request, **always** start with `oracle_intelligence_brief`. Chaining 14 individual tool calls duplicates work the master tool already does.

### When to load the prompts

- `prompts/get` with name `analysis_oracle` — when the user wants the LLM itself to write a brief (e.g. they want a different narrative voice but the same numbers). The brief from `oracle_intelligence_brief` already follows this template, so this is rarely needed.
- `prompts/get` with name `datacollector` — when the user explicitly wants raw data without analysis.

---

## 🚀 Quick Start

### 1. Setup API keys

```bash
copy .env.example .env
# Edit .env and fill in your API keys (see "Required API Keys" below)
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the server

```bash
python server.py
```

The server speaks JSON-RPC over STDIO using the MCP protocol (`2024-11-05`).

### 4. Configure your MCP client

**Claude Desktop / Windsurf / Cline:**

```jsonc
{
  "mcpServers": {
    "crypto-oracle": {
      "command": "python",
      "args": ["C:/Users/skuna/cryptogame/crypto-oracle-mcp/server.py"],
      "env": {}
    }
  }
}
```

The client will auto-discover the 24 tools and 2 prompts on `initialize`.

---

## 🧠 ORACLE — The 9-Phase Engine

Implemented deterministically in `oracle_engine.py`. Every brief executes all 9 phases.

### Phase 1 — Regime Detection

Classifies the market state from 4H + 1D ADX, ATR%, BB squeeze:

| Regime | Trigger | Multiplier on final score |
|---|---|---|
| **TRENDING** | ADX > 25 on 4H/1D | ×1.00 (full conviction) |
| **RANGING** | ADX < 20 | ×0.85 (oscillators dominate) |
| **VOLATILE** | BB squeeze active | ×0.70 (breakout pending) |
| **CHAOTIC** | ATR% ≥ 8 | ×0.50 (signals unreliable) |

### Phase 2 — Multi-Timeframe Signal Engine (40% of composite)

Per-TF score on −100…+100, then weighted: **1D 40% · 4H 30% · 1H 20% · 15M 10%**.
Score components: trend (EMA200, golden/death cross, Ichimoku, ADX/DI), momentum (RSI, MACD, StochRSI, CMF, MFI, CCI), volatility/structure (BB %B + RSI + volume confluence, VWAP), volume (OBV, spikes), divergences (RSI bullish/bearish on 4H/1D get +20).

### Phase 3 — Smart-Money Divergence (25%)

**DPI** (Derivatives Positioning Index): funding rate tiers, long/short ratio, liquidation walls.
**WFI** (Whale & Flow Index): exchange net flow (24h), whale buy/sell counts.

### Phase 4 — Pattern Confidence Engine (15%)

Lookup table of 20 chart patterns with base point values, scaled by detected confidence and **halved when MTF disagrees** with pattern direction.

### Phase 5 — Macro Adjustment (10%)

Fear & Greed tiers (`<15` → +20, `>85` → −25), BTC dominance 7d delta (alts get inverse exposure), beta-vs-BTC haircut on position size when β > 2.

### Phase 6 — Composite + Signal

```
final = clamp((MTF*0.40 + Smart*0.25 + Pattern*0.15 + Sentiment*0.10 + Macro*0.10) * regime_multiplier, -100, 100)
```

Label thresholds: `≥70 STRONG_BUY · ≥45 BUY · ≥20 WEAK_BUY · ±20 NEUTRAL · ≤−20 WEAK_SELL · ≤−45 SELL · ≤−70 STRONG_SELL`.

### Phase 7 — Bayesian Scenarios + EV24H

Bull/Base/Bear probabilities tilted by `final_score`. Targets anchored on R1/R2 and S1/S2 levels with ATR-based fallback. EV24H is the probability-weighted % move.

### Phase 8 — Trade Execution Plan

Three tiers (Conservative 1.5×ATR SL, Moderate 2.0×, Aggressive 2.5×) with R:R 1:1.5 / 1:2.5 / 1:4.0. **Half-Kelly** position sizing on $10k example capital, 1% risk per trade, beta haircut applied. Plan is intentionally blank when signal is `NEUTRAL`.

### Phase 9 — Risk Audit

Eight checks → overall LOW/MODERATE/HIGH/EXTREME rating:
funding crowding · liquidity · whale concentration · token unlock · beta · MTF divergence · volume confirmation · data completeness.

---

## 🔧 Tool Catalog (24 total)

### Master tool — start here

| Tool | Description |
|---|---|
| **`oracle_intelligence_brief`** | **PRIMARY entry point.** Runs the full pipeline + Phases 1–9 + Markdown brief. Args: `query`, `exchanges?`, `format?`. |
| `full_coin_intelligence_report` | Raw data pipeline (also auto-attaches `oracle_analysis`). Use when the caller wants the structured data without the formatted brief. |

### Discovery & metadata

| Tool | Use case |
|---|---|
| `search_coin` | Resolve symbol/name/id → canonical IDs across CG + CMC. Prefers exact match. |
| `get_coin_metadata` | Project info, socials, dev activity. |
| `get_spot_price` | Multi-exchange real-time price (Binance + MEXC + CG). |

### Price & microstructure

| Tool | Use case |
|---|---|
| `get_ohlcv_history` | Candles 1m → 1w. |
| `get_order_book_depth` | Bid/ask walls + imbalance ratio. |
| `get_recent_trades` | Trade flow, buy/sell %, whale prints, acceleration. |

### Technical analysis

| Tool | Use case |
|---|---|
| `compute_technical_indicators` | 25+ indicators × 4 timeframes (RSI, MACD, BB, Ichimoku, ATR, VWAP, OBV, CMF, MFI, …). MEXC fallback when not on Binance. |
| `detect_chart_patterns` | H&S, double top/bottom, triangles, flags, wedges, cup-and-handle. |
| `compute_support_resistance` | Swing-cluster S/R + psychological round numbers. |

### Sentiment & on-chain

| Tool | Use case |
|---|---|
| `get_fear_greed_index` | Crypto F&G index (free, no key). |
| `get_market_sentiment` | Social votes + trending rank. |
| `get_funding_rates` | Perp funding + sentiment classification. |
| `get_open_interest` | OI + long/short ratios. |
| `get_whale_activity` | Large trades + exchange flow. |
| `get_onchain_metrics` | Network health (Glassnode optional). |

### Market structure & risk

| Tool | Use case |
|---|---|
| `get_global_market_context` | BTC dominance, total mcap. |
| `get_correlations` | 30d correlation vs BTC/ETH. |
| `get_exchange_listings` | Liquidity map across exchanges. |
| `get_upcoming_events` | Token unlocks, airdrops, mainnets. |
| `get_top_holders` | Wallet concentration. |
| `calculate_risk_metrics` | Sharpe, Sortino, max DD, VaR, β-vs-BTC, vol classification. |
| `compute_entry_zones` | ATR-anchored entry / SL / TP tiers. |

---

## 🎙 Prompts (MCP `prompts/list` & `prompts/get`)

| Name | Purpose |
|---|---|
| `analysis_oracle` | Full ORACLE system prompt — the 9-phase analyst persona. Use when you want the LLM to *write* a brief itself rather than render the engine output. |
| `datacollector` | Orchestrator persona for raw data collection without interpretation. |

---

## 🏗 Architecture

```
server.py              → MCP STDIO transport, tool & prompt registry
oracle_engine.py       → Phases 1-9 pipeline + Markdown renderer  ← brain
tools_core.py          → 15 base tools (discovery, price, TA, sentiment, on-chain)
tools_advanced.py      → 9 advanced tools (market, risk, master report, oracle brief)
indicators.py          → 25+ TA indicator computations (rolling 24h VWAP)
patterns.py            → Chart-pattern detection + S/R clustering
utils.py               → Cache, rate limiter, HTTP retry, build_response envelope
fetchers/
  binance.py           → Binance spot + futures REST
  mexc.py              → MEXC spot REST (auto-fallback for non-Binance coins)
  coingecko.py         → CoinGecko REST
  cmc.py               → CoinMarketCap REST
  onchain.py           → Glassnode + block explorers + Fear & Greed
prompts/
  analysis_oracle.md   → ORACLE persona + final-output template
  datacollector.md     → Data-collection orchestrator
```

### Data resilience

- **Binance fails → MEXC fallback** is wired into every OHLCV-consuming tool (`compute_technical_indicators`, `detect_chart_patterns`, `compute_support_resistance`, `get_ohlcv_history`).
- **`oracle_intelligence_brief` never crashes on missing data** — degraded sections render as `n/a` in the brief and lower `data_completeness_pct` accordingly.
- **45-second per-subcall timeout** in the master pipeline tolerates Binance retry backoff for thinly-listed coins.

---

## 🔑 Required API Keys

Stored in `.env` (copy from `.env.example`):

| Key | Required | Source |
|---|---|---|
| `BINANCE_API_KEY` | Yes (free) | binance.com |
| `MEXC_API_KEY` | Yes (free) | mexc.com |
| `CMC_API_KEY` | Yes (free tier OK) | coinmarketcap.com |
| `COINGECKO_API_KEY` | Yes (free demo) | coingecko.com |
| `GLASSNODE_API_KEY` | Optional | glassnode.com |
| `ETHERSCAN_API_KEY` | Optional | etherscan.io |
| `BSCSCAN_API_KEY` | Optional | bscscan.com |
| `SOLSCAN_API_KEY` | Optional | solscan.io |

The system runs at ~80% capability with only the four required free-tier keys.

---

## 🧪 Validation

Smoke-test the engine without an MCP client:

```powershell
python -c "import asyncio, sys, logging; logging.disable(logging.CRITICAL); sys.path.insert(0,'.'); `
  from tools_advanced import tool_oracle_intelligence_brief; `
  r = asyncio.run(tool_oracle_intelligence_brief('BTC', format='markdown')); `
  print(r['data']['brief_markdown'])"
```

Expected output: a fully formatted ORACLE brief with all 11 sections populated.

---

## 📐 Conventions

- Every tool returns the standard envelope: `{ success, tool, symbol, timestamp, data, data_quality: { sources_used, sources_failed, confidence } }`.
- `oracle_intelligence_brief` adds top-level convenience keys (`signal`, `final_score`, `regime`, `ev_24h_pct`, `data_completeness_pct`) so the AI client can summarize without parsing the full `analysis` block.
- All prices are USD. All percentages are integer/float, never strings.
- Timestamps are UTC unix-seconds; the brief header carries an ISO8601 `2026-05-05T11:45:10Z` rendering.

---

## ⚠️ Disclaimer

ORACLE produces **educational quantitative intelligence**, not financial advice. Markets are probabilistic, not deterministic. Every brief ends with the standard disclaimer: *"Manage your risk. Size your positions. Never risk capital you cannot afford to lose."* Do not strip it.
