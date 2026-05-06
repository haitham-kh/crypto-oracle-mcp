"""
features_v5.py - V5 feature engine.

Sits on top of v4 (62 features) and adds:

  Perp microstructure (12)
  ──────────────────────────────────
   funding_rate_current     signed scaled funding rate
   funding_rate_z_30d       30-day z-score of funding
   funding_change_8h        change vs previous funding event
   funding_extreme          1 if |funding| > 0.05 % else 0
   oi_log                   log(open interest, native units), z-scored
   oi_change_1h             pct change over last hour
   oi_change_4h             pct change over last 4h
   oi_z_7d                  7-day z-score of OI
   oi_price_divergence_4h   sign(ΔOI) * sign(Δprice) - i.e. squeeze setup
   lsr_top_minus_global     top-trader LSR  -  global LSR  (smart money skew)
   lsr_top_z_1d             1-day z of top-trader LSR
   taker_ratio_z_1d         1-day z of perp taker buy/sell ratio

  Cross-sectional (6)
  ──────────────────────────────────
   ret_rank_1h              this coin's 1h-return rank in basket  ∈ [0,1]
   ret_rank_4h              same for 4h
   rv_rank_1h               1h realized vol rank
   coin_alpha_4h            this coin's 4h ret - basket median 4h ret
   basket_mom_4h            basket median 4h return (z-scored)
   btc_alpha_4h             this coin's 4h ret - BTC's 4h ret

V5 total: 62 (v4) + 18 (v5-extras) = 80 features.

`build_v5_features` is a convenience function that combines v4 features
with v5 extras given the per-coin perp DataFrames + the basket-state
table aligned on the spot timestamp grid.
"""
from __future__ import annotations
import os, glob
import numpy as np
import polars as pl

from features_v4 import (
    FEATURE_NAMES_V4, build_v4_from_components,
    _causal_zscore, _roll_mean, _roll_std,
)

FEATURE_NAMES_V5_EXTRA = [
    # Perp microstructure (12)
    "funding_rate_current", "funding_rate_z_30d", "funding_change_8h", "funding_extreme",
    "oi_log_z", "oi_change_1h", "oi_change_4h", "oi_z_7d",
    "oi_price_divergence_4h",
    "lsr_top_minus_global", "lsr_top_z_1d", "taker_ratio_z_1d",
    # Cross-sectional (6)
    "ret_rank_1h", "ret_rank_4h", "rv_rank_1h",
    "coin_alpha_4h", "basket_mom_4h", "btc_alpha_4h",
]
FEATURE_NAMES_V5 = FEATURE_NAMES_V4 + FEATURE_NAMES_V5_EXTRA


# ── Perp data loaders ───────────────────────────────────────────────────────

def _load_perp(symbol, perp_dir):
    """Return (funding_df, metrics_df) sorted by ts_ms; either may be None."""
    f_path = os.path.join(perp_dir, f"{symbol}_funding.parquet")
    m_path = os.path.join(perp_dir, f"{symbol}_metrics.parquet")
    funding = pl.read_parquet(f_path) if os.path.exists(f_path) else None
    metrics = pl.read_parquet(m_path) if os.path.exists(m_path) else None
    if funding is not None: funding = funding.sort("ts_ms")
    if metrics is not None: metrics = metrics.sort("ts_ms")
    return funding, metrics


def _asof_align(target_ts_ms, source_ts_ms, source_values):
    """Forward-fill align: for each target ts, find the latest source ts ≤ it
    and return its value. NaN where no source data yet exists."""
    target = np.asarray(target_ts_ms, dtype=np.int64)
    src_ts = np.asarray(source_ts_ms, dtype=np.int64)
    src_v  = np.asarray(source_values, dtype=np.float64)
    if len(src_ts) == 0:
        return np.full(len(target), np.nan)
    idx = np.searchsorted(src_ts, target, side="right") - 1
    out = np.where(idx >= 0, src_v[np.clip(idx, 0, len(src_v)-1)], np.nan)
    return out


# ── Perp feature builder ────────────────────────────────────────────────────

def build_v5_perp_features(symbol, target_ts_ms, close, perp_dir):
    """Compute the 12 perp features for `symbol` at the given 1m timestamps.
    Returns (n, 12) array. Missing perp data → all-zeros (model treats neutrally).
    """
    n = len(target_ts_ms)
    F = np.zeros((n, 12), dtype=np.float64)

    funding, metrics = _load_perp(symbol, perp_dir)

    # ── Funding (8h cadence) ────────────────────────────────────────────────
    if funding is not None and funding.height > 0:
        f_ts = funding["ts_ms"].to_numpy()
        f_rate = funding["funding_rate"].to_numpy()
        cur = _asof_align(target_ts_ms, f_ts, f_rate)
        cur = np.nan_to_num(cur, nan=0.0)
        F[:, 0] = np.clip(cur * 1000, -2.0, 2.0) / 2.0    # scaled, ~ [-1,1]

        # 30-day rolling z-score on the irregular funding series, then align
        if len(f_rate) >= 90:
            mu = np.zeros_like(f_rate); sd = np.ones_like(f_rate)
            window = 90  # ≈ 30 days at 8h cadence
            for i in range(len(f_rate)):
                lo = max(0, i - window + 1)
                w = f_rate[lo:i+1]
                mu[i] = w.mean(); sd[i] = max(w.std(), 1e-6)
            f_z = (f_rate - mu) / sd
            zfull = _asof_align(target_ts_ms, f_ts, f_z)
            F[:, 1] = np.nan_to_num(np.clip(zfull, -3, 3) / 3.0, nan=0.0)

        # change vs previous funding event
        f_diff = np.zeros_like(f_rate); f_diff[1:] = f_rate[1:] - f_rate[:-1]
        cd = _asof_align(target_ts_ms, f_ts, f_diff)
        F[:, 2] = np.nan_to_num(np.clip(cd * 1000, -2, 2) / 2.0, nan=0.0)

        # extreme funding flag
        F[:, 3] = (np.abs(cur) > 0.0005).astype(np.float64) * 2 - 1  # ±1

    # ── Metrics: OI / LSR / taker ratio (5min cadence) ──────────────────────
    if metrics is not None and metrics.height > 0:
        m_ts = metrics["ts_ms"].to_numpy()
        oi   = metrics["oi"].to_numpy() if "oi" in metrics.columns else None
        if oi is None:
            return F
        # log + 1d z-score of log
        oi_log = np.log(np.maximum(oi, 1e-9))
        # 1d z (288 5-min bars)
        win_1d = 288
        if len(oi_log) > win_1d:
            mu = _roll_mean(oi_log, win_1d); sd = _roll_std(oi_log, win_1d)
            with np.errstate(divide="ignore", invalid="ignore"):
                z = np.where(sd > 1e-9, (oi_log - mu) / sd, 0.0)
        else:
            z = np.zeros_like(oi_log)
        oi_z_aligned = _asof_align(target_ts_ms, m_ts, z)
        F[:, 4] = np.nan_to_num(np.clip(oi_z_aligned, -3, 3) / 3.0, nan=0.0)

        # OI pct change over 1h (12 bars) and 4h (48 bars)
        def pct_change(arr, k):
            out = np.zeros_like(arr)
            if len(arr) > k:
                with np.errstate(divide="ignore", invalid="ignore"):
                    out[k:] = np.where(arr[:-k] > 0, (arr[k:] - arr[:-k]) / arr[:-k], 0.0)
            return out
        d1h = pct_change(oi, 12); d4h = pct_change(oi, 48)
        F[:, 5] = np.nan_to_num(np.clip(_asof_align(target_ts_ms, m_ts, d1h)*5, -1, 1), nan=0.0)
        F[:, 6] = np.nan_to_num(np.clip(_asof_align(target_ts_ms, m_ts, d4h)*5, -1, 1), nan=0.0)

        # 7d z of OI (2016 5-min bars)
        win_7d = 2016
        if len(oi_log) > win_7d:
            mu7 = _roll_mean(oi_log, win_7d); sd7 = _roll_std(oi_log, win_7d)
            with np.errstate(divide="ignore", invalid="ignore"):
                z7 = np.where(sd7 > 1e-9, (oi_log - mu7) / sd7, 0.0)
            oi_z7_aligned = _asof_align(target_ts_ms, m_ts, z7)
            F[:, 7] = np.nan_to_num(np.clip(oi_z7_aligned, -3, 3) / 3.0, nan=0.0)

        # OI / price divergence over 4h
        # signal = sign(d4h_oi) * sign(d4h_price); +1 trend, -1 squeeze setup
        log_close = np.log(np.maximum(close, 1e-12))
        dprice_4h = np.zeros_like(log_close)
        if len(log_close) > 240:
            dprice_4h[240:] = log_close[240:] - log_close[:-240]
        d4h_aligned = _asof_align(target_ts_ms, m_ts, d4h)
        d4h_aligned = np.nan_to_num(d4h_aligned, nan=0.0)
        F[:, 8] = np.sign(d4h_aligned) * np.sign(dprice_4h)

        # LSR top - global
        if "lsr_top" in metrics.columns and "lsr_global" in metrics.columns:
            lt = metrics["lsr_top"].to_numpy(); lg = metrics["lsr_global"].to_numpy()
            spread = lt - lg
            sa = _asof_align(target_ts_ms, m_ts, spread)
            F[:, 9] = np.nan_to_num(np.clip(sa / 2.0, -1, 1), nan=0.0)
            # 1d z of top LSR
            if len(lt) > win_1d:
                mu = _roll_mean(lt, win_1d); sd = _roll_std(lt, win_1d)
                with np.errstate(divide="ignore", invalid="ignore"):
                    zlt = np.where(sd > 1e-9, (lt - mu)/sd, 0.0)
                F[:, 10] = np.nan_to_num(
                    np.clip(_asof_align(target_ts_ms, m_ts, zlt), -3, 3) / 3.0, nan=0.0)

        # taker ratio z-score
        if "taker_ratio" in metrics.columns:
            tr = metrics["taker_ratio"].to_numpy()
            if len(tr) > win_1d:
                mu = _roll_mean(tr, win_1d); sd = _roll_std(tr, win_1d)
                with np.errstate(divide="ignore", invalid="ignore"):
                    ztr = np.where(sd > 1e-9, (tr - mu)/sd, 0.0)
                F[:, 11] = np.nan_to_num(
                    np.clip(_asof_align(target_ts_ms, m_ts, ztr), -3, 3) / 3.0, nan=0.0)

    return F


# ── Cross-sectional feature builder ─────────────────────────────────────────

def build_v5_cross_sectional(symbol, target_ts_ms, close, basket_state):
    """Compute the 6 cross-sectional features for one coin at given timestamps.

    `basket_state` is the parquet from basket_alignment.py (long-form).
    """
    n = len(target_ts_ms)
    F = np.zeros((n, 6), dtype=np.float64)
    if basket_state is None:
        return F
    sub = basket_state.filter(pl.col("symbol") == symbol).sort("timestamp")
    if sub.height == 0:
        return F
    src_ts = sub["timestamp"].to_numpy().astype(np.int64)

    def _af(col):
        if col not in sub.columns: return np.zeros(n)
        v = sub[col].to_numpy().astype(np.float64)
        a = _asof_align(target_ts_ms, src_ts, v)
        return np.nan_to_num(a, nan=0.0)

    # ranks already in [0,1]; remap to [-1,1]
    F[:, 0] = np.clip(_af("ret_rank_1h") * 2 - 1, -1, 1)
    F[:, 1] = np.clip(_af("ret_rank_4h") * 2 - 1, -1, 1)
    F[:, 2] = np.clip(_af("rv_rank_1h")  * 2 - 1, -1, 1)
    F[:, 3] = np.clip(_af("coin_alpha_4h") * 50, -2, 2) / 2.0
    F[:, 4] = np.clip(_af("basket_mom_4h") * 50, -2, 2) / 2.0

    # btc_alpha_4h: this coin 4h ret minus BTC 4h ret  (computed inline)
    btc_state = basket_state.filter(pl.col("symbol") == "BTCUSDT").sort("timestamp")
    if btc_state.height > 0 and "coin_alpha_4h" in btc_state.columns:
        # BTC's 4h return ≈ basket_mom_4h for BTC + coin_alpha_4h for BTC
        btc_total = (btc_state["basket_mom_4h"].to_numpy()
                     + btc_state["coin_alpha_4h"].to_numpy())
        btc_ts = btc_state["timestamp"].to_numpy().astype(np.int64)
        btc_4h = np.nan_to_num(_asof_align(target_ts_ms, btc_ts, btc_total), nan=0.0)
        # this coin's 4h log-return
        log_close = np.log(np.maximum(close, 1e-12))
        my_4h = np.zeros(n)
        if len(log_close) > 240:
            my_4h[240:] = log_close[240:] - log_close[:-240]
        F[:, 5] = np.clip((my_4h - btc_4h) * 50, -2, 2) / 2.0
    return F


# ── Top-level builder ───────────────────────────────────────────────────────

def build_v5_full(symbol, F_v2, close, high, low, volume, ofi,
                  taker_buy_volume, timestamps_ms, atr_arr,
                  btc_close, perp_dir, basket_state):
    """Build the full v5 (n, 80) feature matrix for one coin."""
    F_v4 = build_v4_from_components(F_v2, close, high, low, volume, ofi,
                                    taker_buy_volume, timestamps_ms, atr_arr,
                                    btc_close=btc_close)               # (n, 62)
    F_perp = build_v5_perp_features(symbol, timestamps_ms, close, perp_dir)  # (n, 12)
    F_xs   = build_v5_cross_sectional(symbol, timestamps_ms, close, basket_state)  # (n, 6)
    return np.hstack([F_v4, F_perp, F_xs])
