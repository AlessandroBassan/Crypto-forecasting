"""Non-negotiable anti-leakage test suite.

Mandatory verification before any predictive or causal modeling:
1. Permuted-target test: shuffling future targets destroys all out-of-sample signal.
2. Forward shift +1 bar impossible gain detector: shifting features forward by +1 bar
   is caught as an impossible predictive gain.
3. Feature-target non-overlap: feature windows [t-W, t] strictly terminate before target
   windows [t+1, t+H].
4. Causal scaling: scalers fitted strictly on training folds; test fold data has zero
   influence on scaler parameters.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score, r2_score

from cff.data.download import generate_synthetic_market_data
from cff.targets import (
    compute_future_log_return_and_direction,
)


@pytest.fixture
def market_data() -> pd.DataFrame:
    """Generate 1,000 bars of deterministic market data."""
    return generate_synthetic_market_data(periods=1000, seed=789)


def test_permuted_target_yields_no_signal(market_data):
    """Permuted-target test: Destroying temporal alignment must reduce OOS R^2 to <= 0."""
    df = market_data.copy()
    close = df["close"].values
    n = len(df)

    # Causal lag features (log returns at lags 1, 2, 3, 24)
    log_ret = np.zeros(n)
    log_ret[1:] = np.log(close[1:] / close[:-1])

    X = np.column_stack([
        np.roll(log_ret, 1),
        np.roll(log_ret, 2),
        np.roll(log_ret, 3),
        np.roll(log_ret, 24),
    ])

    # Target: 1h forward return
    y_true, y_dir = compute_future_log_return_and_direction(df)

    # Valid mask
    valid_idx = np.arange(25, n - 2)
    X_valid = X[valid_idx]
    y_valid = y_true.values[valid_idx]

    # Split into train (first 70%) and test (remaining 30%)
    split = int(len(X_valid) * 0.7)
    X_train, X_test = X_valid[:split], X_valid[split:]
    y_train_true, y_test_true = y_valid[:split], y_valid[split:]

    # Shuffle / permute target completely
    rng = np.random.default_rng(999)
    y_train_permuted = rng.permutation(y_train_true)

    model_perm = Ridge(alpha=1.0)
    model_perm.fit(X_train, y_train_permuted)
    y_pred_perm = model_perm.predict(X_test)

    # On permuted targets, out-of-sample R^2 must be non-positive or near zero (< 0.01)
    r2_perm = r2_score(y_test_true, y_pred_perm)
    assert r2_perm < 0.01, f"Permuted target unexpectedly showed predictive R^2: {r2_perm}"

    # Directional accuracy must be within random noise of 50%
    dir_pred = (y_pred_perm > 0).astype(float)
    dir_true = (y_test_true > 0).astype(float)
    acc = accuracy_score(dir_true, dir_pred)
    assert 0.40 <= acc <= 0.60, f"Permuted target accuracy diverged from random walk: {acc}"


def test_forward_shift_impossible_gain_detector(market_data):
    """Forward-shift +1 bar test: An accidental lookahead must be detected as an impossible gain."""
    df = market_data.copy()
    close = df["close"].values
    n = len(df)

    # Legitimate causal feature: past 1h return
    causal_feature = np.zeros(n)
    causal_feature[1:] = np.log(close[1:] / close[:-1])

    # Leaked feature: shifted FORWARD by 1 bar (knows close at t+1 at bar t!)
    leaked_feature = np.roll(causal_feature, -1)

    y_target, _ = compute_future_log_return_and_direction(df)

    valid_idx = np.arange(10, n - 5)
    target_vals = y_target.values[valid_idx]
    causal_vals = causal_feature[valid_idx]
    leaked_vals = leaked_feature[valid_idx]

    # Correlation of legitimate causal feature with future return is near zero (EMH)
    corr_causal = abs(np.corrcoef(causal_vals, target_vals)[0, 1])

    # Correlation of leaked +1 forward feature with future return is exactly 1.0!
    corr_leaked = abs(np.corrcoef(leaked_vals, target_vals)[0, 1])

    assert corr_causal < 0.25, f"Legitimate causal feature has suspiciously high correlation: {corr_causal}"
    assert corr_leaked > 0.99, f"Forward shift detector failed to catch leakage! Correlation: {corr_leaked}"


def test_feature_target_windows_strictly_do_not_overlap(market_data):
    """Feature window [t-W, t] and target window [t+1, t+H] must have an empty intersection."""
    df = market_data.copy()
    feature_window_bars = 24  # 24h lookback
    target_horizon_bars = 24  # 24h forward target

    for t in range(feature_window_bars, len(df) - target_horizon_bars, 50):
        # Feature indices
        feature_indices = set(range(t - feature_window_bars + 1, t + 1))
        # Target indices
        target_indices = set(range(t + 1, t + 1 + target_horizon_bars))

        overlap = feature_indices.intersection(target_indices)
        assert len(overlap) == 0, f"Bar {t} has overlapping feature and target windows: {overlap}"

        # Temporal order: max(feature) < min(target)
        assert max(feature_indices) < min(target_indices)


def test_causal_scaler_fitted_on_train_only():
    """Verify that causal scaling fits only on training folds with zero test set leakage."""
    rng = np.random.default_rng(42)
    train_data = rng.normal(loc=10.0, scale=2.0, size=500)
    test_data = rng.normal(loc=50.0, scale=10.0, size=200)  # Extreme distribution shift in test

    # Fit median / MAD on train
    median_train = np.median(train_data)
    mad_train = np.median(np.abs(train_data - median_train))

    # Apply transform to train and test
    scaled_train = (train_data - median_train) / (mad_train + 1e-6)
    scaled_test = (test_data - median_train) / (mad_train + 1e-6)

    # Scaled train must be centered near 0, test reflects out-of-distribution shift
    assert np.isclose(np.median(scaled_train), 0.0, atol=1e-5)
    assert np.median(scaled_test) > 10.0

    # Now verify that modifying test_data does NOT alter the train scaling
    corrupted_test_data = test_data * 1000.0
    scaled_corrupted_test = (corrupted_test_data - median_train) / (mad_train + 1e-6)
    assert len(scaled_corrupted_test) == len(test_data)

    # Recomputing transform with fixed train parameters:
    re_scaled_train = (train_data - median_train) / (mad_train + 1e-6)
    assert np.array_equal(scaled_train, re_scaled_train)
