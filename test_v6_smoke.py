"""Quick smoke test for features_v6.py"""
import sys
sys.path.insert(0, ".")

from features_v6 import (FEATURE_NAMES_V6, FEATURE_NAMES_V6_EXTRA,
                          N_V6_EXTRA, build_v6_features, compute_v6_signal_strength)
print(f"features_v6 import OK  N_V6_EXTRA={N_V6_EXTRA}, total={len(FEATURE_NAMES_V6)}")

import numpy as np
rng = np.random.default_rng(0)
n = 1500
cl = 100 + np.cumsum(rng.standard_normal(n) * 0.1)
hi = cl + rng.uniform(0.0, 0.5, n)
lo = cl - rng.uniform(0.0, 0.5, n)
vo = rng.uniform(100, 1000, n)
tbv = vo * rng.uniform(0.3, 0.7, n)
ofi = 2 * tbv - vo
ts = (1_700_000_000_000 + np.arange(n) * 60_000).astype(np.int64)
atr = np.abs(hi - lo) * 0.7

F = build_v6_features(cl, hi, lo, vo, ofi, tbv, ts, atr)
print(f"build_v6_features shape: {F.shape}  expected ({n},{N_V6_EXTRA})")
assert F.shape == (n, N_V6_EXTRA), "Shape mismatch!"

warmup = 1440
nan_cols = [i for i in range(N_V6_EXTRA) if np.isnan(F[warmup:, i]).any()]
print(f"  NaN cols post-warmup: {nan_cols if nan_cols else 'none'}")

any_inf = np.isinf(F).any()
print(f"  Inf present: {any_inf}")

fmin = float(F[warmup:].min())
fmax = float(F[warmup:].max())
print(f"  Range post-warmup: [{fmin:.3f}, {fmax:.3f}]  (should be in [-1,1])")
assert fmin >= -1.01 and fmax <= 1.01, f"Values out of [-1,1]: min={fmin} max={fmax}"

strength = compute_v6_signal_strength(F)
print(f"  Signal strength: min={strength.min():.3f}  max={strength.max():.3f}  mean={strength.mean():.3f}")
assert 0.0 <= strength.min() and strength.max() <= 1.01, "Strength out of range"

print()
print("==> ALL CHECKS PASSED")
