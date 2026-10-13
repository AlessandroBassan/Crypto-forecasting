"""Tests for NVIDIA Kumo-TS foundation model wrappers and multi-task loss."""

import numpy as np
import pytest

from cff.models.multitask_loss import (
    MultiTaskLossEvaluator,
    bce_direction_numpy,
    mse_log_vol_numpy,
    student_t_nll_numpy,
    sum_pinball_loss_numpy,
)
from cff.models.tesseract import KumoAnomalyWrapper, KumoForecastWrapper


@pytest.fixture
def mock_dataset():
    """Generate synthetic representations and multi-task targets."""
    rng = np.random.default_rng(42)
    n = 300
    X = rng.normal(size=(n, 12))
    y_log_vol = -3.5 + 0.3 * X[:, 0] + rng.normal(0, 0.1, size=n)
    y_dir = (X[:, 1] > 0).astype(float)
    r_24h = rng.normal(loc=0.0, scale=0.03, size=n)
    return X, y_log_vol, y_dir, r_24h


def test_multitask_loss_components():
    """Verify Student-t NLL, MSE, BCE, and Pinball loss calculations."""
    r_true = np.array([0.01, -0.02, 0.005, 0.03])
    mu = np.zeros(4)
    sigma = np.full(4, 0.02)
    df = np.full(4, 5.0)

    nll = student_t_nll_numpy(r_true, mu, sigma, df)
    assert np.isfinite(nll)

    y_log_vol = np.array([-3.5, -4.0, -3.2, -3.8])
    pred_log_vol = np.array([-3.4, -3.9, -3.3, -3.7])
    mse_val = mse_log_vol_numpy(y_log_vol, pred_log_vol)
    assert 0.0 < mse_val < 0.05

    y_dir = np.array([1, 0, 1, 0])
    p_dir = np.array([0.8, 0.2, 0.7, 0.3])
    bce = bce_direction_numpy(y_dir, p_dir)
    assert 0.0 < bce < 0.5

    quantiles = {0.05: np.full(4, -0.04), 0.5: np.zeros(4), 0.95: np.full(4, 0.04)}
    p_loss = sum_pinball_loss_numpy(r_true, quantiles)
    assert p_loss > 0.0

    evaluator = MultiTaskLossEvaluator()
    res = evaluator.compute(
        r_true, mu, sigma, df,
        y_log_vol, pred_log_vol,
        y_dir, p_dir,
        r_true, quantiles,
    )
    assert "total_loss" in res
    assert np.isfinite(res["total_loss"])


def test_kumo_forecast_zero_shot(mock_dataset):
    """Verify zero-shot inference with and without DARR."""
    X, y_log_vol, _, _ = mock_dataset
    wrapper_zs = KumoForecastWrapper(mode="zero_shot", darr=False)
    wrapper_zs.fit(X[:200], y_log_vol[:200], np.zeros(200), np.zeros(200))

    preds = wrapper_zs.predict(X[200:], trailing_gk_log_vol=y_log_vol[200:])
    assert "pred_log_vol" in preds
    assert "pred_var" in preds
    assert "pred_dir_prob" in preds
    assert "quantile_preds" in preds
    assert len(preds["pred_log_vol"]) == 100
    assert np.all(preds["pred_var"] > 0.0)

    # Test DARR mode
    wrapper_darr = KumoForecastWrapper(mode="zero_shot", darr=True)
    preds_darr = wrapper_darr.predict(X[200:], trailing_gk_log_vol=y_log_vol[200:])
    assert len(preds_darr["pred_log_vol"]) == 100


def test_kumo_forecast_head_finetune(mock_dataset):
    """Verify head-only fine-tuning produces valid multi-task outputs."""
    X, y_log_vol, y_dir, r_24h = mock_dataset
    wrapper = KumoForecastWrapper(mode="head_finetune", taus=(0.05, 0.5, 0.95))
    wrapper.fit(X[:200], y_log_vol[:200], y_dir[:200], r_24h[:200])

    preds = wrapper.predict(X[200:])
    assert len(preds["pred_log_vol"]) == 100
    assert np.all((preds["pred_dir_prob"] >= 0.0) & (preds["pred_dir_prob"] <= 1.0))
    assert 0.05 in preds["quantile_preds"]
    assert 0.95 in preds["quantile_preds"]


def test_kumo_anomaly_wrapper():
    """Verify diffusion anomaly scoring and detector evaluation."""
    rng = np.random.default_rng(123)
    n = 200
    range_pct = rng.uniform(0.01, 0.05, size=n)
    vol = rng.lognormal(mean=7.0, sigma=0.5, size=n)
    flow = rng.uniform(-0.3, 0.3, size=n)

    # Introduce extreme anomalies
    range_pct[10] = 0.25
    vol[10] = vol[10] * 10.0

    gt = np.zeros(n)
    gt[10] = 1.0
    gt[25] = 1.0

    detector = KumoAnomalyWrapper()
    scores = detector.score_anomalies(range_pct, vol, flow)
    assert len(scores) == n
    assert scores[10] > np.median(scores)  # Spike detected

    metrics = detector.evaluate_detector(scores, gt)
    assert "roc_auc" in metrics
    assert "pr_auc" in metrics
    assert metrics["roc_auc"] > 0.5
