"""
oos_audit.py — Post-hoc audit of the v5 simulation results.

Answers four questions:
  1. What was the actual training window (train / val / test split timestamps)?
  2. Which trades from `simulation_v5_results.json` are truly out-of-sample
     (i.e. on a coin the model never saw OR after the train+val cutoff)?
  3. What is per-regime PnL (not just regime counts)?
  4. De-cluster check: total PnL with the top-5 best weeks removed.

This script reads ONLY the `timestamp` column of each parquet so it runs in
seconds, not the minutes/hours that a full training-pipeline rebuild would
take.
"""
from __future__ import annotations
import os
import sys
import json
import datetime as dt
from collections import defaultdict

import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(__file__))
from trading_config import WARMUP_BARS, SAMPLE_EVERY, HORIZONS  # noqa: E402

PROCESSED_DIR = r"E:\training data for quant\processed_features"
META_PATH     = os.path.join(os.path.dirname(__file__), "data", "ev_model_v5_meta.json")
DEFAULT_RESULTS = "simulation_v5_results.json"


def _resolve_results_path(arg: str | None) -> str:
    """Accept either a bare filename (resolved under challenge_data/) or a full path."""
    if not arg:
        arg = DEFAULT_RESULTS
    if os.path.isabs(arg) or os.path.sep in arg or "/" in arg:
        return arg
    return os.path.join(os.path.dirname(__file__), "challenge_data", arg)


def _audit_out_path(results_path: str) -> str:
    base = os.path.basename(results_path)
    stem = base[:-5] if base.endswith(".json") else base
    # simulation_v5_results -> simulation_v5_audit
    out = stem.replace("results", "audit") if "results" in stem else stem + "_audit"
    return os.path.join(os.path.dirname(__file__), "challenge_data", out + ".json")


def _fmt(ms: int) -> str:
    return dt.datetime.utcfromtimestamp(ms / 1000).strftime("%Y-%m-%d %H:%M")


def collect_training_timestamps(training_coins, horizons):
    """Re-derive the timestamp array train_v5.py would have produced, by reading
    only the `timestamp` column of each coin's monthly parquets and applying the
    same WARMUP_BARS / SAMPLE_EVERY / max(horizons) trim.

    This is an approximation: we skip the per-sample NaN-feature filter and
    label validity filter (those would require recomputing v5 features). The
    resulting timestamps are a superset, so the 70/85 percentiles will be
    *very close* but not byte-identical to what train_v5.py used.
    """
    import glob
    max_h = max(horizons)
    all_ts = []

    for sym in training_coins:
        pattern = os.path.join(PROCESSED_DIR, f"{sym}_1m_features_*.parquet")
        files = sorted(glob.glob(pattern))
        if not files:
            print(f"  WARNING: no parquets for {sym}")
            continue
        # Read only the timestamp column from each month, concat, sort.
        ts_pieces = []
        for f in files:
            df = pl.scan_parquet(f).select("timestamp").collect()
            ts_pieces.append(df["timestamp"].to_numpy().astype(np.int64))
        ts_coin = np.concatenate(ts_pieces)
        # The training pipeline sorts within-coin then concatenates; we replicate.
        ts_coin.sort()
        n = ts_coin.size
        if n <= WARMUP_BARS + max_h:
            continue
        idx = np.arange(WARMUP_BARS, n - max_h, SAMPLE_EVERY, dtype=np.int64)
        all_ts.append(ts_coin[idx])
        print(f"  {sym}: {n:,} bars → {idx.size:,} candidate samples")

    ts_all = np.concatenate(all_ts)
    ts_all.sort()
    return ts_all


def derive_split(ts_all):
    n = ts_all.size
    train_end = int(n * 0.70)
    val_end   = int(n * 0.85)
    return {
        "n_samples":      n,
        "train_start_ms": int(ts_all[0]),
        "train_end_ms":   int(ts_all[train_end - 1]),
        "val_start_ms":   int(ts_all[train_end]),
        "val_end_ms":     int(ts_all[val_end - 1]),
        "test_start_ms":  int(ts_all[val_end]),
        "test_end_ms":    int(ts_all[-1]),
    }


def classify_trade(t, split, training_coins):
    """Return one of: untrained_coin, train, val, test, post_test."""
    if t["coin"] not in training_coins:
        return "untrained_coin"
    o = t["open_ts"]
    if o <= split["train_end_ms"]:
        return "train"
    if o <= split["val_end_ms"]:
        return "val"
    if o <= split["test_end_ms"]:
        return "test"
    return "post_test"


def pct(x, base):
    return 100.0 * x / base if base else 0.0


def main():
    results_path = _resolve_results_path(sys.argv[1] if len(sys.argv) > 1 else None)
    print("=" * 78)
    print(f"  OOS AUDIT — {os.path.basename(results_path)}")
    print("=" * 78)

    # ── 1. Determine training split ─────────────────────────────────────────
    meta = json.load(open(META_PATH))
    training_coins = set(meta["training_coins"])
    print(f"\n[1] Training coins ({len(training_coins)}): "
          f"{', '.join(sorted(training_coins))}")
    print(f"\n[1] Reconstructing training sample timestamps "
          f"(WARMUP={WARMUP_BARS}, SAMPLE_EVERY={SAMPLE_EVERY}, max_h={max(HORIZONS)})...")
    ts_all = collect_training_timestamps(sorted(training_coins), HORIZONS)
    split = derive_split(ts_all)
    print(f"\n  Reconstructed n={split['n_samples']:,} samples "
          f"(meta says {meta['total_samples']:,}; diff = filtered-out NaN/label rows)")
    print(f"  TRAIN window:  {_fmt(split['train_start_ms'])}  →  {_fmt(split['train_end_ms'])}")
    print(f"  VAL   window:  {_fmt(split['val_start_ms'])}    →  {_fmt(split['val_end_ms'])}")
    print(f"  TEST  window:  {_fmt(split['test_start_ms'])}   →  {_fmt(split['test_end_ms'])}")

    # ── 2. Slice the sim trades by training-status ──────────────────────────
    if not os.path.exists(results_path):
        print(f"\nERROR: {results_path} not found.")
        return
    res = json.load(open(results_path))
    trades = res["trades"]
    n_total = len(trades)
    starting_cap = res["starting_capital"]
    print(f"\n[2] Total sim trades: {n_total}  "
          f"final cap: ${res['final_capital']:,.2f}  "
          f"raw return: {res['total_return_pct']:+.2f}%")

    buckets = defaultdict(list)
    for t in trades:
        buckets[classify_trade(t, split, training_coins)].append(t)

    print(f"\n  Bucket breakdown:")
    print(f"    {'bucket':<18} {'n':>5} {'PnL($)':>12} {'WR%':>6} "
          f"{'avg net%':>10} {'first':<12} {'last':<12}")
    rows = []
    for b in ("untrained_coin", "train", "val", "test", "post_test"):
        tl = buckets[b]
        if not tl:
            print(f"    {b:<18} {0:>5} {0:>12.2f}  --      --        --           --")
            continue
        pnl = sum(t["pnl_usd"] for t in tl)
        wr  = pct(sum(1 for t in tl if t["pnl_usd"] > 0), len(tl))
        avg = np.mean([t["net_pct"] for t in tl])
        first = _fmt(min(t["open_ts"] for t in tl))[:10]
        last  = _fmt(max(t["open_ts"] for t in tl))[:10]
        print(f"    {b:<18} {len(tl):>5} {pnl:>+12,.2f} {wr:>5.1f}% {avg:>+9.3f}% "
              f"{first:<12} {last:<12}")
        rows.append((b, len(tl), pnl, wr))

    # True-OOS = untrained_coin ∪ test ∪ post_test (NEVER in train+val)
    oos_trades = buckets["untrained_coin"] + buckets["test"] + buckets["post_test"]
    is_trades  = buckets["train"] + buckets["val"]
    oos_pnl = sum(t["pnl_usd"] for t in oos_trades)
    is_pnl  = sum(t["pnl_usd"] for t in is_trades)
    print(f"\n  → IN-SAMPLE (train+val):  {len(is_trades):>4} trades  "
          f"${is_pnl:>+10,.2f}  ({pct(is_pnl, starting_cap):+.2f}% of starting cap)")
    print(f"  → TRUE OOS (untrained_coin ∪ test ∪ post_test): "
          f"{len(oos_trades):>4} trades  ${oos_pnl:>+10,.2f}  "
          f"({pct(oos_pnl, starting_cap):+.2f}% of starting cap)")

    # Untrained-coin-only PnL (still TIME-overlap with training, but coin-OOS)
    uc_pnl = sum(t["pnl_usd"] for t in buckets["untrained_coin"])
    print(f"\n  Of which untrained_coin (time-overlap kept, coin-OOS):")
    print(f"    {len(buckets['untrained_coin']):>4} trades  ${uc_pnl:>+10,.2f}  "
          f"({pct(uc_pnl, starting_cap):+.2f}% of starting cap)")

    # ── 3. Per-regime PnL ───────────────────────────────────────────────────
    print(f"\n[3] Per-regime PnL (vs raw regime-count distribution)")
    reg_counts = res.get("regime_counts", {})
    total_dp = sum(reg_counts.values()) or 1
    print(f"  {'regime':<14} {'n_trades':>9} {'PnL($)':>12} {'WR%':>6} "
          f"{'avg net%':>10} {'decision%':>10}")
    by_regime = defaultdict(list)
    for t in trades:
        by_regime[t["regime"]].append(t)
    for r in sorted(by_regime, key=lambda x: -sum(t["pnl_usd"] for t in by_regime[x])):
        tl = by_regime[r]
        pnl = sum(t["pnl_usd"] for t in tl)
        wr  = pct(sum(1 for t in tl if t["pnl_usd"] > 0), len(tl))
        avg = np.mean([t["net_pct"] for t in tl])
        dp_share = pct(reg_counts.get(r, 0), total_dp)
        print(f"  {r:<14} {len(tl):>9} {pnl:>+12,.2f} {wr:>5.1f}% "
              f"{avg:>+9.3f}% {dp_share:>9.1f}%")

    # ── 4. De-cluster check: remove top-5 best weeks ────────────────────────
    print(f"\n[4] De-cluster check (top-5 weeks removed)")
    weekly = res.get("weekly", [])
    if not weekly:
        print("  No weekly data available.")
        return
    sorted_wk = sorted(weekly, key=lambda w: -w["pnl"])
    top5 = sorted_wk[:5]
    top5_keys = {w["week"] for w in top5}
    kept = [w for w in weekly if w["week"] not in top5_keys]
    top5_pnl = sum(w["pnl"] for w in top5)
    kept_pnl = sum(w["pnl"] for w in kept)
    total_pnl = sum(w["pnl"] for w in weekly)

    print(f"  Top-5 weeks (chronological order shown):")
    for w in sorted(top5, key=lambda x: x["week"]):
        print(f"    week {w['week']:>2}  {w['trades']:>3} trades  "
              f"${w['pnl']:>+9,.2f}")
    print(f"\n  Full total      PnL: ${total_pnl:>+10,.2f}  "
          f"({pct(total_pnl, starting_cap):+.2f}% of $100k)")
    print(f"  Top-5 contribution:  ${top5_pnl:>+10,.2f}  "
          f"({pct(top5_pnl, total_pnl):+.1f}% of total PnL)")
    print(f"  WITHOUT top-5:   PnL: ${kept_pnl:>+10,.2f}  "
          f"({pct(kept_pnl, starting_cap):+.2f}% return on $100k base)")

    n_kept_wks = len(kept)
    annualised = (kept_pnl / starting_cap) * (52.0 / max(n_kept_wks, 1)) * 100
    print(f"  Annualised (kept weeks only): {annualised:+.1f}% / yr  "
          f"over {n_kept_wks} weeks")

    # Save audit JSON
    out_path = _audit_out_path(results_path)
    with open(out_path, "w") as f:
        json.dump({
            "split_dates": {k: _fmt(v) if k.endswith("_ms") else v
                            for k, v in split.items()},
            "split_ms":    split,
            "training_coins": sorted(training_coins),
            "buckets": {
                b: {"n": len(buckets[b]),
                    "pnl_usd": round(sum(t["pnl_usd"] for t in buckets[b]), 2)}
                for b in ("untrained_coin", "train", "val", "test", "post_test")
            },
            "in_sample":  {"n": len(is_trades),  "pnl_usd": round(is_pnl,  2)},
            "true_oos":   {"n": len(oos_trades), "pnl_usd": round(oos_pnl, 2)},
            "per_regime": {
                r: {"n": len(by_regime[r]),
                    "pnl_usd": round(sum(t["pnl_usd"] for t in by_regime[r]), 2),
                    "win_rate_pct":
                        round(pct(sum(1 for t in by_regime[r] if t["pnl_usd"] > 0),
                                  len(by_regime[r])), 1),
                    "decision_share_pct": round(pct(reg_counts.get(r, 0), total_dp), 2)}
                for r in by_regime
            },
            "top5_weeks": top5,
            "de_cluster": {
                "top5_pnl":   round(top5_pnl, 2),
                "kept_pnl":   round(kept_pnl, 2),
                "kept_return_pct": round(pct(kept_pnl, starting_cap), 2),
                "annualised_kept_pct": round(annualised, 1),
            },
        }, f, indent=2, default=str)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
