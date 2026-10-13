"""Tests for causal inference methods and predictive causality metrics."""

import numpy as np
import pandas as pd

from cff.causal.dml import DoubleMLPartiallyLinear
from cff.causal.event_study import EventStudyAnalyzer
from cff.causal.granger_te import compute_granger_causality, compute_transfer_entropy
from cff.causal.rdd import FundingRDDAnalyzer


def test_double_ml_partially_linear():
    """Verify Double ML partially linear model recovers synthetic causal treatment effect."""
    rng = np.random.default_rng(42)
    n = 1000
    # Controls X
    X = rng.normal(size=(n, 5))
    # Confounded treatment D: correlated with X
    D = 0.5 * X[:, 0] + 0.3 * X[:, 1] + rng.normal(0, 0.5, size=n)
    # True causal effect theta = 1.5
    true_theta = 1.5
    Y = true_theta * D + 0.8 * X[:, 0] - 0.4 * X[:, 2] + rng.normal(0, 0.5, size=n)

    dml = DoubleMLPartiallyLinear(n_blocks=4, purge_bars=10)
    dml.fit(Y, D, X)
    summary = dml.summary()

    assert summary["theta_hat"] is not None
    # Estimated theta should be within sampling error of 1.5
    assert abs(summary["theta_hat"] - true_theta) < 0.3
    assert summary["is_significant_5pct"] is True


def test_event_study_analyzer():
    """Verify event study detects simulated liquidation events and computes CAR."""
    rng = np.random.default_rng(123)
    n = 1200
    timestamps = pd.date_range("2023-01-01", periods=n, freq="1h", tz="UTC")

    # Prices with baseline random walk
    ret = rng.normal(0, 0.01, size=n)
    # Inject large liquidation cascade crash at t=500
    ret[500] = -0.10
    prices = 100.0 * np.exp(np.cumsum(ret))

    vol = rng.lognormal(mean=7.0, sigma=0.5, size=n)
    vol[500] = vol[500] * 10.0  # Massive volume spike

    df = pd.DataFrame({
        "timestamp": timestamps,
        "open": prices,
        "high": prices * 1.01,
        "low": prices * 0.99,
        "close": prices,
        "volume": vol,
    })
    df.loc[500, "low"] = df.loc[500, "close"] * 0.90  # Huge range

    analyzer = EventStudyAnalyzer(pre_event_bars=6, post_event_bars=12, estimation_window_bars=72)
    events = analyzer.identify_liquidation_events(df, z_thresh=2.5, min_spacing_hours=24)

    assert 500 in events
    res = analyzer.analyze_events(df, events, n_bootstrap=50)

    assert res["n_events"] > 0
    assert len(res["car_mean"]) == (6 + 1 + 12)
    assert len(res["car_ci_lower"]) == len(res["car_mean"])


def test_rdd_running_variable_and_estimation():
    """Verify RDD running variable mapping and sharp discontinuity estimation."""
    times = pd.date_range("2023-01-01 00:00", periods=24, freq="1h", tz="UTC")
    df = pd.DataFrame({"timestamp": times, "basis": np.zeros(24)})

    analyzer = FundingRDDAnalyzer()
    z = analyzer.compute_running_variable(df)

    # At 00:00 UTC, z must be 0
    assert z.iloc[0] == 0.0
    # At 07:00 UTC, z must be -1 (1 hour before 08:00 settlement)
    assert z.iloc[7] == -1.0
    # At 08:00 UTC, z must be 0
    assert z.iloc[8] == 0.0

    # Inject sharp jump at cutoff
    rng = np.random.default_rng(99)
    z_synth = rng.uniform(-3, 3, size=400)
    true_tau = 0.005
    y_synth = 0.001 + true_tau * (z_synth >= 0).astype(float) + 0.001 * z_synth + rng.normal(0, 0.0005, size=400)

    rdd_res = analyzer.estimate_sharp_rdd(y_synth, z_synth, bandwidth=2.0, cutoff=0.0)
    assert abs(rdd_res["tau_discontinuity"] - true_tau) < 0.002
    assert rdd_res["is_significant_5pct"] is True


def test_predictive_causality_granger_and_te():
    """Verify Granger test and Transfer Entropy on coupled AR processes."""
    rng = np.random.default_rng(555)
    n = 500

    # Series X drives series Y with lag 1
    x = np.zeros(n)
    y = np.zeros(n)
    for t in range(1, n):
        x[t] = 0.5 * x[t - 1] + rng.normal(0, 1.0)
        y[t] = 0.4 * y[t - 1] + 0.7 * x[t - 1] + rng.normal(0, 0.5)

    # Granger test: X Granger-causes Y
    gc = compute_granger_causality(y, x, max_lag=2)
    assert 1 in gc
    assert gc[1]["is_predictively_significant_5pct"] is True

    # Transfer Entropy: X -> Y
    te = compute_transfer_entropy(y, x, n_bins=4, lag=1)
    assert te["directed_transfer_entropy_bits"] > 0.0
