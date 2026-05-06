"""
basket_alignment.py - Pre-compute basket-wide cross-sectional state.

For every minute-bar timestamp present anywhere in the basket of 14 coins
we record:

    basket_median_ret_1h     - median 1h log-return across the basket
    basket_median_ret_4h     - median 4h log-return across the basket
    basket_rv_1h             - median 1h realized vol across the basket
    n_coins                  - how many of the 14 had data this minute

Per-coin cross-sectional features are then derived at feature-build time as:

    ret_rank_1h   = (this coin's 1h ret rank among basket) / (n - 1)   ∈ [0,1]
    ret_rank_4h   = same for 4h
    rv_rank_1h    = same for 1h realised vol
    coin_alpha_4h = this coin's 4h ret  -  basket_median_ret_4h
    basket_mom_4h = z-scored basket median 4h return
    btc_alpha_4h  = this coin's 4h ret  -  BTC's 4h ret  (decoupling signal)

Saved to data/basket/basket_state.parquet.

Usage:
    python basket_alignment.py
"""
from __future__ import annotations
import os, sys, glob, time
import numpy as np
import polars as pl

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from train_ev_model_v2 import PROCESSED_DIR
BASKET_DIR = os.path.join(HERE, "data", "basket")
os.makedirs(BASKET_DIR, exist_ok=True)
OUT_PATH = os.path.join(BASKET_DIR, "basket_state.parquet")


def load_coin_minute_close(symbol):
    pattern = os.path.join(PROCESSED_DIR, f"{symbol}_1m_features_*.parquet")
    files = sorted(glob.glob(pattern))
    if not files:
        return None
    dfs = []
    for f in files:
        try:
            dfs.append(pl.read_parquet(f).select(["timestamp", "close"]))
        except Exception:
            pass
    if not dfs:
        return None
    return pl.concat(dfs).unique(subset=["timestamp"]).sort("timestamp")


def main():
    t0 = time.time()
    files = glob.glob(os.path.join(PROCESSED_DIR, "*.parquet"))
    symbols = sorted({os.path.basename(f).split("_1m_")[0] for f in files})
    print(f"Coins: {len(symbols)}  ({', '.join(symbols)})")

    # Step 1: load each coin's (ts, close) and reduce to a wide table.
    print("\nLoading per-coin closes...")
    parts = []
    for s in symbols:
        df = load_coin_minute_close(s)
        if df is None:
            print(f"  {s}: no data, skipping"); continue
        df = df.rename({"close": s})
        parts.append(df)
        print(f"  {s}: {df.height:,} bars")

    print("\nJoining basket on timestamp...")
    wide = parts[0]
    for d in parts[1:]:
        wide = wide.join(d, on="timestamp", how="full", coalesce=True)
    wide = wide.sort("timestamp")
    print(f"  joined: {wide.height:,} unique minute-bars")

    closes = {s: wide[s].to_numpy() for s in symbols}
    ts_ms = wide["timestamp"].to_numpy().astype(np.int64)

    # Step 2: per-coin log returns at 60 and 240 bar lags (1h / 4h).
    print("\nComputing per-coin returns (1h / 4h) and realized vol (1h)...")
    rets_1h = {}
    rets_4h = {}
    rv_1h   = {}
    n = wide.height
    for s in symbols:
        cl = closes[s].astype(np.float64)
        good = ~np.isnan(cl)
        # forward-fill closes within this coin so log-return shifts are well defined
        last = np.nan
        for i in range(n):
            if not np.isnan(cl[i]): last = cl[i]
            elif not np.isnan(last): cl[i] = last
        log_cl = np.log(np.maximum(cl, 1e-12))
        r1 = np.zeros(n); r1[60:]  = log_cl[60:]  - log_cl[:-60]
        r4 = np.zeros(n); r4[240:] = log_cl[240:] - log_cl[:-240]
        # 1h realised vol = std of 1m log returns × sqrt(60)
        m1 = np.zeros(n); m1[1:] = log_cl[1:] - log_cl[:-1]
        rv = np.zeros(n)
        if n > 60:
            from numpy.lib.stride_tricks import sliding_window_view
            win = sliding_window_view(m1, 60)
            rv_part = win.std(axis=1) * np.sqrt(60)
            rv[60-1:60-1+len(rv_part)] = rv_part
        # null out portions where we never had data
        invalid = ~good
        # Roll the invalid mask forward to capture freshly-missing windows.
        r1[~good] = np.nan; r4[~good] = np.nan; rv[~good] = np.nan
        rets_1h[s] = r1; rets_4h[s] = r4; rv_1h[s] = rv

    R1 = np.column_stack([rets_1h[s] for s in symbols])
    R4 = np.column_stack([rets_4h[s] for s in symbols])
    V1 = np.column_stack([rv_1h[s]   for s in symbols])

    # Step 3: basket aggregates (nan-aware median + count-of-non-nan).
    print("Computing basket medians...")
    n_coins = (~np.isnan(R1)).sum(axis=1)
    with np.errstate(invalid="ignore", all="ignore"):
        basket_med_r1 = np.nanmedian(R1, axis=1)
        basket_med_r4 = np.nanmedian(R4, axis=1)
        basket_med_rv = np.nanmedian(V1, axis=1)
    basket_med_r1 = np.nan_to_num(basket_med_r1, nan=0.0)
    basket_med_r4 = np.nan_to_num(basket_med_r4, nan=0.0)
    basket_med_rv = np.nan_to_num(basket_med_rv, nan=0.0)

    # Step 4: per-coin ranks at each timestamp (in [0,1]).
    print("Computing per-coin ranks...")
    rank_r1 = np.full_like(R1, np.nan)
    rank_r4 = np.full_like(R4, np.nan)
    rank_v1 = np.full_like(V1, np.nan)

    def _rank_row(row):
        m = ~np.isnan(row)
        k = m.sum()
        if k <= 1: return np.full_like(row, np.nan)
        order = np.argsort(np.argsort(np.where(m, row, np.inf)))
        out = np.full_like(row, np.nan)
        out[m] = order[m] / max(k - 1, 1)
        return out

    # vectorised row-by-row (still O(n*m log m); m=14 so cheap).
    for i in range(n):
        if n_coins[i] >= 3:
            rank_r1[i] = _rank_row(R1[i])
            rank_r4[i] = _rank_row(R4[i])
            rank_v1[i] = _rank_row(V1[i])

    # Step 5: persist a single basket-state parquet keyed by (timestamp, symbol).
    print("Building long-form basket-state table...")
    rows = []
    for ci, sym in enumerate(symbols):
        df = pl.DataFrame({
            "timestamp": ts_ms,
            "symbol": [sym] * n,
            "ret_rank_1h":   rank_r1[:, ci],
            "ret_rank_4h":   rank_r4[:, ci],
            "rv_rank_1h":    rank_v1[:, ci],
            "coin_alpha_4h": R4[:, ci] - basket_med_r4,
            "basket_mom_4h": basket_med_r4,
            "n_coins":       n_coins.astype(np.float64),
        })
        # filter to rows where we actually had this coin
        df = df.filter(pl.col("ret_rank_4h").is_not_null())
        rows.append(df)

    full = pl.concat(rows).sort(["symbol", "timestamp"])
    full.write_parquet(OUT_PATH)
    print(f"\n[OK] basket state -> {OUT_PATH}")
    print(f"  rows: {full.height:,}   coins: {len(symbols)}")
    print(f"  wall: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
