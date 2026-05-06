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
FEE_PCT_PER_SIDE       = 0.001     # Binance spot taker (no BNB discount)
SLIPPAGE_PCT_PER_SIDE  = 0.0001    # 1 bp slippage assumption per side
ROUND_TRIP_COST        = 2 * (FEE_PCT_PER_SIDE + SLIPPAGE_PCT_PER_SIDE)

# ── Multi-horizon trade definitions ─────────────────────────────────────────
# A trade can be opened with one of these holding horizons (in 1-minute bars).
# At decision time, the model produces (p_up_h, expected_return_h) for each
# horizon and the best one is selected by realised EV. ATR_HORIZON_SCALE_MAP
# expands the 14-bar ATR to the holding horizon (sqrt(h/14)).
HORIZONS = [60, 240, 720]
ATR_HORIZON_SCALE_MAP = {
    60:  math.sqrt(60.0 / 14.0),     # ≈ 2.07
    240: math.sqrt(240.0 / 14.0),    # ≈ 4.14
    720: math.sqrt(720.0 / 14.0),    # ≈ 7.17
}

# Regime-conditioned TP multipliers (× ATR_HORIZON_SCALE × ATR14).
TP_MULT_TREND   = 2.0   # Hurst > 0.55 → let winners run
TP_MULT_NEUTRAL = 1.5
TP_MULT_RANGE   = 1.0   # Hurst < 0.45 → tight TP
SL_MULT         = 1.0

# Reject any setup whose TP is not at least this multiple of round-trip cost.
MIN_TP_TO_COST_RATIO = 2.0

# ── Decision rule ───────────────────────────────────────────────────────────
# Trade only when BOTH:
#   p_up >= chosen_threshold (per-horizon, validation-tuned)  AND
#   EV_pct >= MIN_EV_PCT (after fees)
# AND regime != CHAOS (see regime_filter.py).
MIN_P_UP_DEFAULT = 0.55
MIN_EV_PCT       = 0.0015         # require ≥ 0.15% expected net edge

# ── Decision cadence ────────────────────────────────────────────────────────
CHECK_EVERY_MINS = 240            # re-evaluate every 4 hours

# ── Mean-variance sizing ────────────────────────────────────────────────────
# size_$ = clip( risk_aversion_target * mu / sigma^2, 0, max_position_$ )
# where mu = expected_net_return_pct, sigma = predicted_volatility_pct.
# Then layered with the classic risk-per-trade and Kelly caps.
RISK_AVERSION_TARGET = 0.0007     # tunes aggressiveness; ~0.05% target vol/trade
KELLY_FRACTION       = 0.25       # quarter-Kelly cap on top
RISK_PER_TRADE_PCT   = 0.01       # never lose more than 1% on a single trade at SL
MAX_POSITION_PCT     = 0.20       # never deploy more than 20% on one position
MAX_OPEN_POSITIONS_TOTAL    = 3
MAX_OPEN_POSITIONS_PER_COIN = 1

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
