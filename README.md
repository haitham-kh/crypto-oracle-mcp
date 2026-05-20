# 🔮 Crypto Oracle MCP — Quantitative Spot Trading Intelligence

A production-grade quantitative trading system using machine learning to generate edge in crypto spot/perp markets. Built on **XGBoost** with a 130-feature stack combining microstructure signals, technical analysis, and on-chain metrics.

---

## ⚡ Quick Start — Train V6 Models on Google Colab

**Everything runs in one command. No local setup needed.**

### Step 1 — Upload training data to Google Drive

Create this folder in your Google Drive:
```
MyDrive/crypto_oracle/processed/
```
Upload your 20 `*_training_data.parquet` files there.

### Step 2 — Open Colab with GPU

[→ Open Google Colab](https://colab.research.google.com) · **Runtime → Change runtime type → T4 GPU**

### Step 3 — Clone the repo and run

Paste this into a Colab cell and run it:

```python
!git clone https://github.com/haitham-kh/crypto-oracle-mcp.git
%cd crypto-oracle-mcp
!python colab_v6_train.py
```

**That's it.** The script automatically:
- ✅ Installs all dependencies
- ✅ Mounts your Google Drive
- ✅ Downloads 24 months of Binance 1m klines for all 20 coins (free, no API key)
- ✅ Computes 36 V6 technical features per coin
- ✅ Trains V6_full (130 features) + V6_micro (36 features) on GPU
- ✅ Saves trained models to `MyDrive/crypto_oracle/models_v6/`

### Step 4 — Copy models to your machine

Download `MyDrive/crypto_oracle/models_v6/` from Drive and copy all files into:
```
crypto-oracle-mcp/data/
```

### Step 5 — Run the live engine

```bash
python live_demo_engine.py
```
On startup you'll see: `[V6] models loaded — blend: V5×30% + V6×70%`

---

## 🏗 Architecture

### V6 Model Stack (130 features)

| Group | Features | Tier | What it captures |
|---|---|---|---|
| **Donchian Channels** | 6 | S+ | Structural range, breakouts |
| **Anchored VWAP** | 3 | S | Institutional fair value (day/week/month) |
| **Volume Profile VAH/VAL** | 4 | S | Auction-market value area (70% of volume) |
| **Liquidity Sweeps** | 5 | S | Stop-hunt detection via wick rejection |
| **Order Flow (OFI/CVD)** | 4 | A | Large-trade delta, taker pressure |
| **SMA Stack** | 6 | A- | 20/50/200 trend alignment |
| **Fibonacci Levels** | 5 | B | 38.2 / 50 / 61.8 / 78.6 confluence |
| **Bollinger Extras** | 3 | B | Squeeze / expansion regime |
| **V5 Base** | 94 | — | Hurst, Perp microstructure, VPVR, cross-sectional |

### Inference Blending

```
p_final = 0.30 × p_v5  +  0.70 × p_v6_full
```
Trade fires only if both `p_final ≥ threshold` AND `p_v6_micro ≥ 0.50` (confirmation gate).

### Training Pipeline

```
Colab: colab_v6_train.py
  ↓ downloads 1m klines (Binance Futures API, free)
  ↓ computes 36 V6 features inline
  ↓ joins with pre-computed V5 features (your parquets)
  ↓ trains XGBoost on GPU with recency + V6-strength weighting
  ↓ calibrates probabilities (Isotonic Regression)
  ↓ auto-selects threshold via EV maximisation
  ↓ saves to Google Drive
```

---

## 📁 File Structure

```
crypto-oracle-mcp/
│
├── colab_v6_train.py          ← THE MAIN COLAB SCRIPT (start here)
│
├── features_v6.py             ← 36 V6 feature definitions
├── features_v5.py             ← V5 base features (94)
├── features_v4.py             ← V4 base features
├── train_v6.py                ← Local training (if you prefer CPU)
├── train_v5.py                ← V5 training pipeline
│
├── live_demo_engine.py        ← Live trading engine (paper/live)
├── blind_backtest.py          ← Out-of-sample backtest
│
├── trading_config.py          ← Risk parameters, thresholds
├── regime_filter.py           ← Market regime classification
├── labels_multi_horizon.py    ← Triple-barrier labelling
│
├── fetchers/
│   ├── binance.py
│   ├── mexc.py
│   └── coingecko.py
│
├── data/                      ← Model weights (gitignored, from Colab)
│   ├── v6_full_clf_*.json
│   ├── v6_micro_clf_*.json
│   ├── v6_*_calib_*.pkl
│   ├── v6_meta.json
│   └── v5_clf_*.json          ← V5 models (fallback)
│
└── .env                       ← API keys (gitignored, never committed)
```

---

## ⚙️ Local Setup (optional — for live engine only)

```bash
git clone https://github.com/haitham-kh/crypto-oracle-mcp.git
cd crypto-oracle-mcp
pip install -r requirements.txt
cp .env.example .env
# Fill in your API keys in .env
```

### Required API keys (`.env`)

```env
BINANCE_API_KEY=your_key
BINANCE_API_SECRET=your_secret
MEXC_API_KEY=your_key
MEXC_API_SECRET=your_secret
CMC_API_KEY=your_key
COINGECKO_API_KEY=your_key
```

---

## 📊 Expected Performance (V6 target)

| Metric | V5 Baseline | V6 Target |
|---|---|---|
| Test EV per trade | ~+0.05% | >+0.15% |
| Win rate | ~51% | >53% |
| Profit factor | ~1.15 | >1.40 |
| Trades/day | ~8–15 | ~6–12 (higher selectivity) |

---

## 🔄 Workflow

```
Train (Colab)          Live Engine              Backtest
─────────────    →    ──────────────────    →   ─────────────────
colab_v6_train.py     live_demo_engine.py       blind_backtest.py
  GPU, ~25 min          V5×30% + V6×70%           OOS 6 months
  saves to Drive        auto-detects V6            prints PF, EV
```

---

## 🛡 Risk Controls

- **LOOSEN_MODE = False** — strict signal quality required
- **Position sizing** — Kelly + volatility-scaled, max 10% capital
- **Stop loss** — ATR-based, capped at 2.5%
- **Regime filter** — blocks trading in CHAOS / LOW_LIQUIDITY regimes
- **Micro-gate** — V6_micro must confirm before taking any trade

---

*Built with Python, XGBoost, Polars, and Binance/MEXC APIs.*
