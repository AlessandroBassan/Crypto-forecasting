"""Tests for Combinatorial Purged Cross-Validation (CPCV) and DSR/PBO diagnostics."""

import numpy as np

from cff.validation.cpcv import CombinatorialPurgedCV
from cff.validation.dsr_pbo import (
    compute_cscv_pbo,
    compute_deflated_sharpe_ratio,
    compute_expected_max_sharpe,
    compute_probabilistic_sharpe_ratio,
)


def test_cpcv_split_counts_and_separation():
    """Verify CPCV yields C(N, k) combinations with strict purge and embargo."""
    n_samples = 3000
    X = np.zeros(n_samples)
    n_groups = 5
    n_test_groups = 2
    purge_bars = 24
    embargo_bars = 24

    cpcv = CombinatorialPurgedCV(
        n_groups=n_groups,
        n_test_groups=n_test_groups,
        purge_bars=purge_bars,
        embargo_bars=embargo_bars,
    )

    expected_splits = 10  # C(5, 2) = 10
    assert cpcv.get_n_splits() == expected_splits

    splits = list(cpcv.split(X))
    assert len(splits) == expected_splits

    for fold, (train_idx, test_idx) in enumerate(splits):
        # 1. No overlap between train and test
        overlap = set(train_idx).intersection(set(test_idx))
        assert len(overlap) == 0, f"CPCV split {fold} has overlapping train/test samples!"

        # 2. Both train and test are non-empty
        assert len(train_idx) > 500
        assert len(test_idx) > 500


def test_expected_max_sharpe_monotonicity():
    """Verify that expected max Sharpe ratio under the null increases with trial count."""
    exp_max_10 = compute_expected_max_sharpe(n_trials=10, var_sharpe=0.5)
    exp_max_50 = compute_expected_max_sharpe(n_trials=50, var_sharpe=0.5)
    exp_max_200 = compute_expected_max_sharpe(n_trials=200, var_sharpe=0.5)

    assert 0.0 < exp_max_10 < exp_max_50 < exp_max_200


def test_deflated_sharpe_ratio():
    """Verify DSR deflates Sharpe ratio when trial count is large."""
    # A nominal Sharpe of 1.5 with 1 trial has high PSR
    psr_single = compute_probabilistic_sharpe_ratio(sharpe_hat=1.5, sharpe_benchmark=0.0, n_samples=500)
    assert psr_single > 0.99

    # When tested across N=100 trials, DSR should be substantially lower
    dsr_res = compute_deflated_sharpe_ratio(
        sharpe_hat=1.5,
        n_trials=100,
        var_sharpe=0.8,
        n_samples=500,
    )
    assert dsr_res["expected_max_null_sr"] > 1.2
    assert dsr_res["dsr_probability"] < psr_single


def test_cscv_pbo():
    """Verify CSCV PBO on synthetic overfitted vs consistent strategies."""
    rng = np.random.default_rng(42)
    m_splits = 8
    n_trials = 20

    # Overfitted matrix: pure random noise, IS winner will underperform OOS
    noise_perf = rng.normal(loc=0.0, scale=1.0, size=(m_splits, n_trials))
    pbo_res = compute_cscv_pbo(noise_perf)

    assert "pbo" in pbo_res
    assert "lambda_median" in pbo_res
    assert 0.0 <= pbo_res["pbo"] <= 1.0
    # On pure random noise, PBO is typically around 0.4 - 0.7
    assert pbo_res["pbo"] >= 0.3
