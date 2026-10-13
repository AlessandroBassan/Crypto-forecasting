"""Tests verifying purged walk-forward cross-validation invariants."""

import numpy as np

from cff.validation.purged_wf import PurgedWalkForwardCV


def test_purged_walk_forward_separation():
    """Verify that train and test indices have at least purge_bars separation."""
    n_samples = 3000
    X = np.zeros(n_samples)
    purge_bars = 24
    embargo_bars = 24

    cv = PurgedWalkForwardCV(
        n_splits=4,
        purge_bars=purge_bars,
        embargo_bars=embargo_bars,
        min_train_bars=500,
        test_size_bars=400,
    )

    splits = list(cv.split(X))
    assert len(splits) >= 3

    for fold, (train_idx, test_idx) in enumerate(splits):
        # 1. No overlapping indices between train and test
        overlap = set(train_idx).intersection(set(test_idx))
        assert len(overlap) == 0, f"Fold {fold} has overlapping train/test indices!"

        # 2. Minimum purge separation
        last_train = train_idx[-1]
        first_test = test_idx[0]
        gap = first_test - last_train

        assert gap > purge_bars, (
            f"Fold {fold} violates purge requirement: gap={gap} <= purge_bars={purge_bars}"
        )

        # 3. Test indices must be contiguous
        assert np.array_equal(test_idx, np.arange(test_idx[0], test_idx[-1] + 1))


def test_purged_walk_forward_expanding_vs_rolling():
    """Verify expanding vs rolling behavior."""
    n_samples = 4000
    X = np.zeros(n_samples)

    # Expanding: train_idx always starts at 0
    cv_expanding = PurgedWalkForwardCV(n_splits=3, min_train_bars=800, test_size_bars=500, rolling=False)
    for train_idx, _ in cv_expanding.split(X):
        assert train_idx[0] == 0

    # Rolling: train_idx start shifts forward
    cv_rolling = PurgedWalkForwardCV(n_splits=3, min_train_bars=800, test_size_bars=500, rolling=True)
    starts = [train_idx[0] for train_idx, _ in cv_rolling.split(X)]
    assert len(starts) > 1
    # Check that subsequent folds shift start forward
    assert starts[-1] > starts[0]
