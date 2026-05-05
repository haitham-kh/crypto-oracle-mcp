from __future__ import annotations
"""
EV MODEL — True Expected Value Prediction System
================================================
Replaces heuristic composite scoring with a data-driven EV model.

Architecture:
  Stage 1: Feature vector construction from all flow/regime/volatility signals
  Stage 2: Logistic regression for P(up) — interpretable, fast, no overfitting
  Stage 3: EV computation from calibrated probabilities + empirical gain/loss
  Stage 4: Walk-forward validation framework

The model uses pre-fitted coefficients from historical data when available.
When no trained model exists (cold start), it falls back to a calibrated
heuristic model — clearly labeled as UNCALIBRATED.

CRITICAL NOTE: The probabilities from this model are only as good as the
training data. Until backtested, treat outputs as directional guidance only.
Always use the feature_validator to measure IC before trusting any feature.
"""

import numpy as np
import json
import os
from typing import Dict, List, Any, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE VECTOR CONSTRUCTION
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_NAMES = [
    "ofi_score",          # -1 to +1, primary spot flow signal
    "ofi_acceleration",   # -1 to +1, flow momentum
    "large_trade_ofi",    # -1 to +1, whale flow
    "absorption_index",   # 0 to 1, with sign (positive = buy abs)
    "cvd_trend_15m",      # -1 rising, 0 flat, +1 falling (inverted for signal)
    "cvd_trend_1h",
    "cvd_trend_4h",
    "cvd_divergence_15m", # -1 distribution, 0 none, +1 accumulation
    "cvd_divergence_1h",
    "cvd_divergence_4h",
    "hurst",              # 0 to 1 (0.5 = random walk)
    "trend_efficiency",   # 0 to 1
    "persistence_score",  # 0 to 100, normalized to 0-1
    "depth_imbalance",    # ratio, normalized (>1 = buy pressure)
    "vol_state",          # -1 contracting, 0 stable, +1 expanding
    "fear_greed_norm",    # 0 to 1 (0 = extreme fear, 1 = extreme greed)
    "accum_probability",  # 0 to 1
    "distrib_probability", # 0 to 1
]


def build_feature_vector(
    ofi: Dict,
    absorption: Dict,
    accum: Dict,
    regime: Dict,
    vol: Dict,
    macro: Dict,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Build a normalized feature vector from all analysis modules.

    All features are scaled to approximately [-1, +1] for logistic regression.
    Returns both the array and a named dict for interpretability.
    """
    def _safe(d: Dict, *keys, default=0.0) -> float:
        cur = d
        for k in keys:
            if not isinstance(cur, dict):
                return default
            cur = cur.get(k)
            if cur is None:
                return default
        try:
            return float(cur)
        except (TypeError, ValueError):
            return default

    # OFI signals (already -1..+1 range as ratios, scale ×1)
    ofi_score = _safe(ofi, "ofi", default=0.0)
    ofi_accel = _safe(ofi, "ofi_acceleration", default=0.0)
    large_ofi = _safe(ofi, "large_trade_ofi", default=0.0)

    # Absorption (signed: positive = buy absorption)
    abs_dir = absorption.get("absorption_direction", "neutral") if absorption.get("available") else "neutral"
    abs_idx = _safe(absorption, "absorption_index", default=0.0)
    abs_signed = abs_idx if abs_dir == "buying" else (-abs_idx if abs_dir == "selling" else 0.0)

    # CVD trends: convert to -1/0/+1
    cvd = accum.get("cvd") or {}

    def _cvd_trend_score(tf_cvd: Dict) -> float:
        trend = tf_cvd.get("cvd_trend", "flat") if tf_cvd.get("available") else "flat"
        return 1.0 if trend == "rising" else (-1.0 if trend == "falling" else 0.0)

    def _cvd_div_score(tf_div: Dict) -> float:
        div = tf_div.get("divergence", "none")
        conf = tf_div.get("confidence", 0) / 100.0
        if div == "bullish_accumulation":
            return conf
        if div == "bearish_distribution":
            return -conf
        return 0.0

    cvd_trend_15m = _cvd_trend_score(cvd.get("15m") or {})
    cvd_trend_1h = _cvd_trend_score(cvd.get("1h") or {})
    cvd_trend_4h = _cvd_trend_score(cvd.get("4h") or {})

    div = accum.get("divergence") or {}
    cvd_div_15m = _cvd_div_score(div.get("15m") or {})
    cvd_div_1h = _cvd_div_score(div.get("1h") or {})
    cvd_div_4h = _cvd_div_score(div.get("4h") or {})

    # Regime metrics
    all_metrics = regime.get("all_metrics") or {}
    hurst = _safe(all_metrics, "hurst_4h", default=0.5)
    hurst_norm = (hurst - 0.5) * 2  # -1 (mean revert) to +1 (trending)

    er = _safe(all_metrics, "trend_efficiency_ratio", default=0.5)
    er_norm = (er - 0.5) * 2  # normalize

    # Volatility persistence
    persist = vol.get("trend_persistence") or {}
    persist_score = _safe(persist, "persistence_score", default=50.0) / 100.0  # 0-1
    persist_norm = persist_score * 2 - 1  # -1 to +1

    # Depth imbalance
    liq = ofi.get("_liquidity") or {}  # passed through if available
    depth_imb = _safe(liq, "depth_imbalance_ratio", default=1.0)
    depth_norm = np.clip((depth_imb - 1.0) * 2, -1.0, 1.0)  # 0.5→-1, 1.0→0, 1.5→+1

    # Volatility state
    rv = vol.get("realized_vol") or {}
    vol_state_str = rv.get("vol_state", "stable") if rv.get("available") else "stable"
    vol_state_val = 1.0 if vol_state_str == "expanding" else (-1.0 if vol_state_str == "contracting" else 0.0)

    # Macro / Fear & Greed
    fg = _safe(macro, "fear_greed_value", default=50.0)
    fg_norm = fg / 100.0 * 2 - 1  # -1 (extreme fear) to +1 (extreme greed)

    # Accumulation scores
    scores = accum.get("scores") or {}
    accum_prob = _safe(scores, "accumulation_probability", default=50.0) / 100.0
    distrib_prob = _safe(scores, "distribution_probability", default=50.0) / 100.0

    feature_dict = {
        "ofi_score": ofi_score,
        "ofi_acceleration": ofi_accel,
        "large_trade_ofi": large_ofi,
        "absorption_index": abs_signed,
        "cvd_trend_15m": cvd_trend_15m,
        "cvd_trend_1h": cvd_trend_1h,
        "cvd_trend_4h": cvd_trend_4h,
        "cvd_divergence_15m": cvd_div_15m,
        "cvd_divergence_1h": cvd_div_1h,
        "cvd_divergence_4h": cvd_div_4h,
        "hurst": hurst_norm,
        "trend_efficiency": er_norm,
        "persistence_score": persist_norm,
        "depth_imbalance": depth_norm,
        "vol_state": vol_state_val,
        "fear_greed_norm": fg_norm,
        "accum_probability": accum_prob * 2 - 1,
        "distrib_probability": -(distrib_prob * 2 - 1),
    }

    vector = np.array([feature_dict[k] for k in FEATURE_NAMES], dtype=float)
    return vector, feature_dict


# ─────────────────────────────────────────────────────────────────────────────
# LOGISTIC REGRESSION PREDICTOR
# ─────────────────────────────────────────────────────────────────────────────

# Default (uncalibrated) weights — these encode our PRIOR beliefs about
# which signals are directionally relevant for spot price movement.
# They MUST be replaced by actual fitted coefficients after backtesting.
UNCALIBRATED_WEIGHTS = {
    "ofi_score": 0.50,          # Primary spot flow — highest prior weight
    "ofi_acceleration": 0.30,   # Flow momentum
    "large_trade_ofi": 0.40,    # Whale signal
    "absorption_index": 0.45,   # Accumulation/distribution
    "cvd_trend_15m": 0.10,      # Short-term CVD
    "cvd_trend_1h": 0.25,       # Medium-term CVD
    "cvd_trend_4h": 0.40,       # Long-term CVD (most signal)
    "cvd_divergence_15m": 0.15,
    "cvd_divergence_1h": 0.35,
    "cvd_divergence_4h": 0.55,  # Strongest divergence signal
    "hurst": 0.20,              # Trend persistence context
    "trend_efficiency": 0.15,
    "persistence_score": 0.10,
    "depth_imbalance": 0.20,    # Order book structure
    "vol_state": 0.05,          # Volatility regime is context, not signal
    "fear_greed_norm": 0.05,    # Macro context — smallest weight
    "accum_probability": 0.35,
    "distrib_probability": 0.35,
}


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -10, 10)))


class LogisticEVModel:
    """
    Logistic regression model for P(spot price up) prediction.

    Can run in two modes:
    1. CALIBRATED: uses fitted weights from backtested data
    2. UNCALIBRATED: uses prior weights from domain knowledge

    Always outputs a calibration_warning when in uncalibrated mode.
    """

    def __init__(self, weights_path: Optional[str] = None):
        self.weights = dict(UNCALIBRATED_WEIGHTS)
        self.intercept = 0.0
        self.calibrated = False
        self.training_samples = 0

        if weights_path and os.path.exists(weights_path):
            self.load_weights(weights_path)

    def load_weights(self, path: str):
        try:
            with open(path) as f:
                data = json.load(f)
            self.weights = data.get("weights", self.weights)
            self.intercept = data.get("intercept", 0.0)
            self.calibrated = data.get("calibrated", False)
            self.training_samples = data.get("training_samples", 0)
        except Exception:
            pass

    def save_weights(self, path: str, metadata: Dict = None):
        data = {
            "weights": self.weights,
            "intercept": self.intercept,
            "calibrated": self.calibrated,
            "training_samples": self.training_samples,
            "metadata": metadata or {},
        }
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass

    def predict(self, feature_dict: Dict[str, float]) -> Dict[str, Any]:
        """
        Predict P(up) from feature vector.

        Returns P(up), P(down), confidence score, and feature contributions.
        """
        # Compute weighted sum
        logit = self.intercept
        contributions: List[Dict] = []

        for feat, weight in self.weights.items():
            value = feature_dict.get(feat, 0.0)
            contribution = weight * value
            logit += contribution
            contributions.append({
                "feature": feat,
                "value": round(value, 4),
                "weight": round(weight, 4),
                "contribution": round(contribution, 4),
            })

        p_up = sigmoid(logit)
        p_down = 1.0 - p_up

        # Sort contributions by absolute value for top features
        contributions.sort(key=lambda x: -abs(x["contribution"]))

        # Confidence: how far from 0.5 (threshold)
        confidence = abs(p_up - 0.5) * 2  # 0 = no edge, 1 = max edge

        return {
            "p_up": round(float(p_up), 4),
            "p_down": round(float(p_down), 4),
            "p_up_pct": round(float(p_up * 100), 1),
            "p_down_pct": round(float(p_down * 100), 1),
            "raw_logit": round(float(logit), 4),
            "confidence": round(float(confidence), 4),
            "top_features": contributions[:5],
            "all_features": contributions,
            "model_calibrated": self.calibrated,
            "training_samples": self.training_samples,
            "calibration_warning": (
                None if self.calibrated else
                "⚠️ UNCALIBRATED MODEL — probabilities are based on domain priors, "
                "not empirical backtest. Do not use for sizing until calibrated."
            ),
        }

    def fit(self, X: np.ndarray, y: np.ndarray, learning_rate: float = 0.01,
            n_epochs: int = 100, l2_lambda: float = 0.01) -> Dict[str, Any]:
        """
        Gradient descent logistic regression fit with L2 regularization.

        X: (n_samples, n_features) — each row is a feature vector
        y: (n_samples,) — binary labels: 1 = price went up, 0 = price went down

        Uses walk-forward split: first 80% train, last 20% test.
        """
        if len(X) < 50:
            return {"error": "Insufficient samples for training (need >= 50)"}

        n_features = len(FEATURE_NAMES)
        if X.shape[1] != n_features:
            return {"error": f"Expected {n_features} features, got {X.shape[1]}"}

        # Walk-forward split
        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        # Initialize weights
        w = np.zeros(n_features)
        b = 0.0

        train_losses = []
        for epoch in range(n_epochs):
            # Forward pass
            logits = X_train @ w + b
            p = 1 / (1 + np.exp(-np.clip(logits, -10, 10)))

            # Binary cross-entropy loss + L2
            eps = 1e-10
            loss = -np.mean(y_train * np.log(p + eps) + (1 - y_train) * np.log(1 - p + eps))
            loss += l2_lambda * np.sum(w ** 2)
            train_losses.append(float(loss))

            # Gradients
            error = p - y_train
            dw = X_train.T @ error / len(y_train) + 2 * l2_lambda * w
            db = np.mean(error)

            w -= learning_rate * dw
            b -= learning_rate * db

        # Evaluate on test set (out-of-sample)
        logits_test = X_test @ w + b
        p_test = 1 / (1 + np.exp(-np.clip(logits_test, -10, 10)))
        preds_test = (p_test >= 0.5).astype(int)
        accuracy = float(np.mean(preds_test == y_test))

        # Information Coefficient (Spearman correlation with outcomes)
        from scipy.stats import spearmanr
        ic, _ = spearmanr(p_test, y_test)

        # Update model
        self.weights = {k: float(w[i]) for i, k in enumerate(FEATURE_NAMES)}
        self.intercept = float(b)
        self.calibrated = True
        self.training_samples = len(X)

        return {
            "success": True,
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "out_of_sample_accuracy": round(accuracy, 4),
            "information_coefficient": round(float(ic), 4),
            "final_loss": round(float(train_losses[-1]), 6),
            "message": (
                f"Model fitted. OOS accuracy: {accuracy:.1%}, IC: {ic:.3f}. "
                f"IC > 0.05 is considered useful in quant research."
            ),
        }


# ─────────────────────────────────────────────────────────────────────────────
# EV COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────

def compute_ev(
    p_up: float,
    p_down: float,
    target_up_pct: float,
    target_down_pct: float,
    fee_pct: float = 0.10,
    slippage_pct: float = 0.10,
) -> Dict[str, Any]:
    """
    True Expected Value computation.

    EV = P(up) × avg_gain% - P(down) × avg_loss% - round_trip_costs%

    Where gains and losses are price targets based on structural levels (ATR/S/R),
    NOT arbitrary multiples.

    Args:
        p_up, p_down: calibrated probabilities from logistic model
        target_up_pct: expected gain if trade goes right (% from entry)
        target_down_pct: expected loss if trade goes wrong (% from entry, positive)
        fee_pct: taker fee (0.10% = 0.001)
        slippage_pct: estimated slippage on entry + exit (0.10% = 0.001)
    """
    round_trip = (fee_pct + slippage_pct) * 2 / 100  # Convert to decimal

    ev_gross = p_up * (target_up_pct / 100) - p_down * (target_down_pct / 100)
    ev_net = ev_gross - round_trip

    breakeven_p_up = (target_down_pct / 100 + round_trip) / (
        (target_up_pct + target_down_pct) / 100 + round_trip
    )

    return {
        "ev_gross_pct": round(ev_gross * 100, 3),
        "ev_net_pct": round(ev_net * 100, 3),
        "round_trip_cost_pct": round(round_trip * 100, 3),
        "breakeven_p_up": round(breakeven_p_up, 4),
        "is_positive_ev": ev_net > 0,
        "ev_interpretation": (
            f"EV = {ev_net * 100:+.3f}% per trade. "
            + ("Positive expectancy — favorable edge." if ev_net > 0 else
               "Negative expectancy — do not trade.")
        ),
        "p_up_required_to_breakeven": round(breakeven_p_up * 100, 1),
    }


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE IMPORTANCE RANKING
# ─────────────────────────────────────────────────────────────────────────────

def rank_features(model_output: Dict) -> List[Dict]:
    """
    Rank top 5 features by absolute contribution.
    Used in the final output report.
    """
    all_feats = model_output.get("all_features") or []
    ranked = sorted(all_feats, key=lambda x: -abs(x.get("contribution", 0)))

    output = []
    for f in ranked[:5]:
        direction = "bullish" if f["contribution"] > 0 else "bearish"
        output.append({
            "feature": f["feature"],
            "value": f["value"],
            "contribution": f["contribution"],
            "direction": direction,
            "description": _FEATURE_DESCRIPTIONS.get(f["feature"], f["feature"]),
        })
    return output


_FEATURE_DESCRIPTIONS = {
    "ofi_score": "Order Flow Imbalance (aggressive buys vs sells)",
    "ofi_acceleration": "OFI momentum (is flow accelerating?)",
    "large_trade_ofi": "Whale/large trade flow direction",
    "absorption_index": "Volume absorbed without price move (accumulation/distribution)",
    "cvd_trend_15m": "15-minute Cumulative Volume Delta trend",
    "cvd_trend_1h": "1-hour Cumulative Volume Delta trend",
    "cvd_trend_4h": "4-hour Cumulative Volume Delta trend",
    "cvd_divergence_15m": "CVD vs price divergence (15m)",
    "cvd_divergence_1h": "CVD vs price divergence (1h)",
    "cvd_divergence_4h": "CVD vs price divergence (4h) — stealth accum/distrib",
    "hurst": "Hurst exponent (trend persistence measure)",
    "trend_efficiency": "Trend efficiency ratio (signal-to-noise)",
    "persistence_score": "Multi-factor trend persistence",
    "depth_imbalance": "Order book depth imbalance (bid vs ask)",
    "vol_state": "Volatility regime (expanding/contracting)",
    "fear_greed_norm": "Fear & Greed macro context",
    "accum_probability": "Accumulation probability score",
    "distrib_probability": "Distribution probability score",
}


# ─────────────────────────────────────────────────────────────────────────────
# SINGLETON MODEL INSTANCE
# ─────────────────────────────────────────────────────────────────────────────

_MODEL_WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "data", "ev_model_weights.json")
_model_instance: Optional[LogisticEVModel] = None


def get_model() -> LogisticEVModel:
    global _model_instance
    if _model_instance is None:
        _model_instance = LogisticEVModel(weights_path=_MODEL_WEIGHTS_PATH)
    return _model_instance


def predict_ev(
    ofi: Dict,
    absorption: Dict,
    accum: Dict,
    regime: Dict,
    vol: Dict,
    macro: Dict,
    target_up_pct: float = 3.0,
    target_down_pct: float = 2.0,
    fee_pct: float = 0.10,
    slippage_pct: float = 0.10,
) -> Dict[str, Any]:
    """
    Full EV prediction pipeline.

    Args:
        All analysis module outputs
        target_up/down_pct: ATR-based or S/R-based price targets

    Returns:
        Complete EV model output with top features, probabilities, EV
    """
    model = get_model()
    vector, feature_dict = build_feature_vector(ofi, absorption, accum, regime, vol, macro)
    prediction = model.predict(feature_dict)

    ev = compute_ev(
        prediction["p_up"],
        prediction["p_down"],
        target_up_pct,
        target_down_pct,
        fee_pct,
        slippage_pct,
    )

    top_features = rank_features(prediction)

    regime_name = regime.get("regime", "UNKNOWN")
    signal = (
        "BUY" if (prediction["p_up"] > 0.60 and ev["is_positive_ev"]) else
        "STRONG_BUY" if (prediction["p_up"] > 0.70 and ev["is_positive_ev"]) else
        "SELL" if (prediction["p_down"] > 0.60 and ev["is_positive_ev"]) else
        "STRONG_SELL" if (prediction["p_down"] > 0.70 and ev["is_positive_ev"]) else
        "NEUTRAL"
    )

    return {
        "signal": signal,
        "p_up": prediction["p_up"],
        "p_down": prediction["p_down"],
        "p_up_pct": prediction["p_up_pct"],
        "p_down_pct": prediction["p_down_pct"],
        "model_confidence": prediction["confidence"],
        "ev_net_pct": ev["ev_net_pct"],
        "ev_gross_pct": ev["ev_gross_pct"],
        "is_positive_ev": ev["is_positive_ev"],
        "round_trip_cost_pct": ev["round_trip_cost_pct"],
        "breakeven_p_up": ev["breakeven_p_up"],
        "top_5_features": top_features,
        "regime": regime_name,
        "model_calibrated": prediction["model_calibrated"],
        "calibration_warning": prediction["calibration_warning"],
        "ev_interpretation": ev["ev_interpretation"],
        "raw_logit": prediction["raw_logit"],
    }
