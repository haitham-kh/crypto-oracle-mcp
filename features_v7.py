"""
features_v7.py — V7 Feature Engine (True Liquidation Sweeps)
=============================================================
Adds contextual liquidation sweep features using real Perpetual Funding Rates.
V7 total: 130 (V6) + 6 (V7-extras) = 136 features.
"""
import numpy as np
from features_v6 import FEATURE_NAMES_V6, build_v6_features, _roll_std

FEATURE_NAMES_V7_EXTRA = [
    "funding_rate_current",       # Forward-filled most recent funding rate
    "funding_rate_z_30d",         # Rolling 30-day z-score of funding rate
    "true_bull_liquidation_sweep", # sweep_bull_flag * max(0, fund_z)
    "true_bear_liquidation_sweep", # sweep_bear_flag * max(0, -fund_z)
    "liquidation_exhaustion",      # volume burst * abs(fund_z)
    "funding_extreme_flag"         # +1 if |funding_rate_z| > 2, 0 otherwise
]

FEATURE_NAMES_V7 = FEATURE_NAMES_V6 + FEATURE_NAMES_V7_EXTRA
N_V7_EXTRA = len(FEATURE_NAMES_V7_EXTRA)

def _forward_fill(ts_ms, event_ts, event_vals):
    """
    Given a high-frequency time array `ts_ms` and a low-frequency series 
    (`event_ts`, `event_vals`), forward-fill the low-frequency values to align with `ts_ms`.
    """
    if len(event_ts) == 0:
        return np.zeros(len(ts_ms), dtype=np.float32)
    # searchsorted finds the insertion point. We want the value *before* or *at* the current ts.
    idx = np.searchsorted(event_ts, ts_ms, side='right') - 1
    idx = np.clip(idx, 0, len(event_vals) - 1)
    # For timestamps before the very first event, use 0.0
    out = event_vals[idx].astype(np.float32)
    out[ts_ms < event_ts[0]] = 0.0
    return out

def build_v7_features(close, high, low, volume, ofi, tbv, ts_ms, atr_arr, funding_ts, funding_rates):
    """
    Build V6 features + V7 contextual liquidation features.
    funding_ts and funding_rates are 1D numpy arrays (from CSV or API).
    """
    # 1. Build base V6
    F_v6 = build_v6_features(close, high, low, volume, ofi, tbv, ts_ms, atr_arr)
    
    # 2. Align funding rates to 1m bars
    fund_current = _forward_fill(ts_ms, funding_ts, funding_rates)
    
    # 3. Rolling 30-day z-score of funding rate (43,200 mins in 30 days)
    window_30d = 43200
    n = len(fund_current)
    
    # Fast rolling mean via cumulative sum
    cum_fund = np.zeros(n + 1, dtype=np.float64)
    cum_fund[1:] = np.cumsum(fund_current)
    
    fund_mean = np.zeros(n, dtype=np.float32)
    if n > window_30d:
        fund_mean[window_30d:] = (cum_fund[window_30d+1:] - cum_fund[:-window_30d-1]) / window_30d
    for i in range(1, min(window_30d, n)):
        fund_mean[i] = cum_fund[i+1] / i
        
    fund_std = _roll_std(fund_current, window_30d)
    fund_std = np.where(fund_std < 1e-8, 1.0, fund_std)
    
    fund_z = (fund_current - fund_mean) / fund_std
    
    # 4. Extract V6 sweep flags and volume
    bull_idx = FEATURE_NAMES_V6.index("sweep_bull_flag")
    bear_idx = FEATURE_NAMES_V6.index("sweep_bear_flag")
    
    sweep_bull = F_v6[:, bull_idx]
    sweep_bear = F_v6[:, bear_idx]
    
    # True Liquidation Logic:
    # A Bull Sweep (down then up) traps shorts. It's most powerful when crowd is SHORT (funding < 0).
    # Wait, earlier I wrote:
    # Crowd is LONG -> Funding > 0. A sudden drop triggers LONG liquidations (Bull Sweep).
    # Yes! A drop that pierces a low and bounces back is stopping out LONGs. 
    # Therefore, true_bull_liq = sweep_bull * max(0, fund_z)
    true_bull_liq = sweep_bull * np.clip(fund_z, 0, None)
    
    # A Bear Sweep (up then down) triggers SHORT liquidations. 
    # Most powerful when crowd is SHORT (Funding < 0).
    true_bear_liq = sweep_bear * np.clip(-fund_z, 0, None)
    
    # Volume exhaustion = high volume bar at an extreme funding level
    vol_mean = np.zeros(n, dtype=np.float32)
    cum_vol = np.zeros(n + 1, dtype=np.float64)
    cum_vol[1:] = np.cumsum(volume)
    if n > 60:
        vol_mean[60:] = (cum_vol[61:] - cum_vol[:-60]) / 60
    for i in range(1, min(60, n)):
        vol_mean[i] = cum_vol[i+1] / i
    vol_mean = np.where(vol_mean < 1e-8, 1.0, vol_mean)
    vol_spike = np.clip((volume / vol_mean) - 1.0, 0, 5) / 5.0
    
    liq_exhaustion = vol_spike * np.abs(fund_z)
    
    fund_extreme = np.where(np.abs(fund_z) > 2.0, 1.0, 0.0)
    
    F_v7_extra = np.column_stack([
        np.clip(fund_current * 1000, -1, 1), # scaled
        np.clip(fund_z / 5.0, -1, 1),        # scaled
        np.clip(true_bull_liq, 0, 1),
        np.clip(true_bear_liq, 0, 1),
        np.clip(liq_exhaustion / 10.0, 0, 1),
        fund_extreme
    ]).astype(np.float32)
    
    return np.hstack([F_v6, F_v7_extra])
