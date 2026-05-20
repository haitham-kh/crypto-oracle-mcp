"""
trading_config.py - Single source of truth for the v4 trading architecture.

v4 changes vs v3:
  • Multi-horizon trades (60m / 240m / 720m). The model picks the horizon
    with the highest expected value per coin, per decision.
  • Mean-variance position sizing: size scales with expected return AND
    inversely with predicted variance (vol-targeting).
  • Regime gating: trading halts in CHAOS regime regardless of signal.
  • Fee model includes slippage; live executor must respect ROUND_TRIP_COST.

EVERY consumer (labeler, trainer, simulator, live executor) imports from
here. Change a value, retrain.
"""
from __future__ import annotations
import math

# ── Execution costs ─────────────────────────────────────────────────────────
# IMPORTANT: ROUND_TRIP_COST must match what the v5 model was trained on.
# Training meta says: ev_model_v5_meta.json -> round_trip_cost: 0.0022 (22 bps).
# The model's labels and EV thresholds were calibrated against this cost.
# Lowering it in the sim creates a permissive bias the model wasn't trained for.
FEE_PCT_PER_SIDE       = 0.0005    # Binance USD-M Futures taker fee
SLIPPAGE_PCT_PER_SIDE  = 0.0006    # 6 bps slippage/side -> matches training-time 22 bps round-trip
ROUND_TRIP_COST        = 2 * (FEE_PCT_PER_SIDE + SLIPPAGE_PCT_PER_SIDE)   # = 0.0022

# ── Multi-horizon trade definitions ─────────────────────────────────────────
HORIZONS = [60, 720]          # Dropped 240m due to structural underperformance
ATR_HORIZON_SCALE_MAP = {
    60:  math.sqrt(60.0 / 14.0),     # ≈ 2.07
    240: math.sqrt(240.0 / 14.0),    # ≈ 4.14
    720: math.sqrt(720.0 / 14.0),    # ≈ 7.17
}

# Regime-conditioned TP multipliers (× ATR_HORIZON_SCALE × ATR14).
# DYNAMIC TP LOGIC: TP2 extension for partial scaling.
# TP_MULT_TREND at 2.5 keeps TP1 reachable while still extending beyond the old 2.0.
# TP2 in the sim is set to 1.5× TP1 distance — realistic extension into MFE space.
TP_MULT_TREND   = 2.5   # Hurst > 0.55 → extended but reachable target
TP_MULT_NEUTRAL = 2.0
TP_MULT_RANGE   = 1.0   # Hurst < 0.45 → tight TP
SL_MULT         = 1.2

# Reject any setup whose TP is not at least this multiple of round-trip cost.
MIN_TP_TO_COST_RATIO = 1.5

# ── Decision rule ───────────────────────────────────────────────────────────
MIN_P_UP_DEFAULT = 0.55
# Aligned to model meta: ev_model_v5_meta.json -> min_ev_pct_required_for_trade = 0.0015.
MIN_EV_PCT       = 0.0015         # require ≥ 0.15% expected net edge (training-time threshold)

# ── Decision cadence ────────────────────────────────────────────────────────
CHECK_EVERY_MINS = 10             # re-evaluate every 10 minutes

# ── Mean-variance sizing ────────────────────────────────────────────────────
# Oracle actively chooses leverage based on EV/Variance, up to the MAX_POSITION_PCT cap.
RISK_AVERSION_TARGET = 0.050      # Sizing scalar
KELLY_FRACTION       = 0.40       # Aggressive Kelly
RISK_PER_TRADE_PCT   = 0.03       # 3.0% max risk at stop loss
MAX_POSITION_PCT     = 3.0        # ORACLE LEVERAGE: Allow up to 300% per trade (3x native leverage)
MAX_OPEN_POSITIONS_TOTAL    = 5
MAX_OPEN_POSITIONS_PER_COIN = 1

# ── Loss containment ─────────────────────────────────────────────────────────
# Absolute SL cap: no trade can risk more than this % regardless of ATR scale.
# Prevents single blowups on high-vol coins at long horizons (720m ATR × 7.17).
MAX_SL_PCT = 0.08   # Hard cap: SL never more than 8% from entry

# Per-asset position scale: coins with pathological loss profiles get smaller
# sizing regardless of model conviction. Identified from per-coin backtest data.
ASSET_MAX_POSITION_SCALE = {
    "ORDIUSDT": 0.40,   # Inscription token: gap-spike risk, cap at 40% of normal
    "SEIUSDT":  0.50,   # 33% WR in backtest — low conviction, keep small
    "TIAUSDT":  0.65,   # 75% WR but outsized losses when wrong
}

# ── Risk halts ──────────────────────────────────────────────────────────────
DAILY_LOSS_HALT_PCT  = 0.04
WEEKLY_LOSS_HALT_PCT = 0.08

# ── Regime gating ───────────────────────────────────────────────────────────
# Regimes the model is allowed to trade. CHAOS = high realized vol + low
# trend efficiency + low autocorr → no edge, do nothing.
TRADE_REGIMES = {"TREND_UP", "TREND_DOWN", "RANGE", "EXPANSION"}
HALT_REGIMES  = {"CHAOS", "LOW_LIQUIDITY"}

# ── Data hygiene ────────────────────────────────────────────────────────────
WARMUP_BARS  = 1500     # need long history for 1-day rolling features
SAMPLE_EVERY = 30       # one training sample every 30 minutes (was 10);
                        # reduces auto-correlation between samples

# ── Dynamic Horizon Selection ───────────────────────────────────────────────
# Multiplier applied to raw EV when ranking candidates for execution.
# Encodes which horizons have empirical edge in each regime.
# Values > 1.0 = boost (prefer this horizon in this regime)
# Values < 1.0 = suppress (deprioritize this horizon in this regime)
HORIZON_REGIME_EV_SCALE = {
    "TREND_UP":   {60: 0.80, 240: 0.65, 720: 1.35},  # Trend favours long-horizon runners
    "TREND_DOWN": {60: 0.80, 240: 0.65, 720: 1.35},
    "EXPANSION":  {60: 1.35, 240: 1.00, 720: 0.70},  # Expansion favours fast momentum captures
    "RANGE":      {60: 1.10, 240: 0.80, 720: 0.85},  # Range: slight 60m bias, suppress long horizons
}

# ── Market Momentum Filter ───────────────────────────────────────────────
# When BTC 4-hour return is below this threshold, market is in a quiet/compressed
# state. We raise the minimum EV bar by BTC_QUIET_EV_MULTIPLIER to avoid
# deploying capital into low-activity regimes.
BTC_QUIET_4H_THRESHOLD = 0.005   # 0.5% BTC 4h move threshold (quiet below, active above)
BTC_QUIET_EV_MULTIPLIER = 1.6    # Raise MIN_EV_PCT by 60% during quiet markets

# ── Helpers ─────────────────────────────────────────────────────────────────

def regime_tp_mult(hurst):
    """TP multiplier (×ATR14) given Hurst exponent. Used by labeler & sim."""
    if hurst is None or hurst != hurst:
        return TP_MULT_NEUTRAL
    if hurst > 0.55:
        return TP_MULT_TREND
    if hurst < 0.45:
        return TP_MULT_RANGE
    return TP_MULT_NEUTRAL


def horizon_atr_scale(horizon_bars):
    return ATR_HORIZON_SCALE_MAP.get(int(horizon_bars),
                                     math.sqrt(horizon_bars / 14.0))


def barrier_pcts_for_horizon(atr_value, entry_price, hurst, horizon_bars):
    """Return (tp_pct, sl_pct) as fractions of entry, scaled to horizon."""
    scale = horizon_atr_scale(horizon_bars)
    tp_mult = regime_tp_mult(hurst) * scale
    sl_mult = SL_MULT * scale
    tp_pct = tp_mult * atr_value / max(entry_price, 1e-12)
    sl_pct = sl_mult * atr_value / max(entry_price, 1e-12)
    return tp_pct, sl_pct
