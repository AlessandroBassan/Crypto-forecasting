"""Tests for classical econometric baselines, ML models, and volatility metrics."""

import numpy as np
import pytest

from cff.metrics.prob import brier_score, directional_accuracy, log_loss_score
from cff.metrics.quantile import pinball_loss
from cff.metrics.vol import (
    compute_diebold_mariano,
    compute_mincer_zarnowitz,
    qlike_loss,
)
from cff.models.baselines import (
    GARCHModel,
    GJRGARCHModel,
    HARRVModel,
    LightGBMDirModel,
    LightGBMVolModel,
)


@pytest.fixture
def sample_vol_data():
    """Generate synthetic realized and forecast variances."""
    rng = np.random.default_rng(42)
    true_var = rng.uniform(0.0001, 0.001, size=200)
    # Add unbiased noise to create an imperfect forecast
    pred_var = true_var * rng.uniform(0.8, 1.2, size=200)
    return true_var, pred_var


def test_qlike_loss_properties(sample_vol_data):
    """Verify QLIKE loss properties: minimum at perfect prediction."""
    true_var, pred_var = sample_vol_data

    # Perfect forecast must have QLIKE = 0
    qlike_perfect = qlike_loss(true_var, true_var)
    assert np.isclose(qlike_perfect, 0.0, atol=1e-8)

    # Imperfect forecast must have QLIKE > 0
    qlike_noisy = qlike_loss(true_var, pred_var)
    assert qlike_noisy > 0.0


def test_mincer_zarnowitz_regression(sample_vol_data):
    """Verify Mincer-Zarnowitz regression statistics."""
    true_var, pred_var = sample_vol_data
    mz = compute_mincer_zarnowitz(true_var, pred_var, hac_lags=5)

    assert "alpha" in mz
    assert "beta" in mz
    assert "r_squared" in mz
    assert "wald_stat" in mz
    assert 0.0 <= mz["r_squared"] <= 1.0


def test_diebold_mariano():
    """Verify Diebold-Mariano test detecting a clearly superior model."""
    rng = np.random.default_rng(123)
    loss_good = rng.exponential(scale=1.0, size=200)
    loss_bad = loss_good + 0.5 + rng.normal(0.0, 0.2, size=200)  # Systematically worse with dispersion

    dm_stat, p_val = compute_diebold_mariano(loss_good, loss_bad, hac_lags=5)
    # loss_good - loss_bad is negative -> dm_stat should be strongly negative
    assert dm_stat < -2.0
    assert p_val < 0.05


def test_probability_metrics():
    """Verify directional accuracy, Brier score, and log loss."""
    y_true = np.array([1, 0, 1, 1, 0])
    p_pred = np.array([0.9, 0.1, 0.8, 0.7, 0.2])

    acc = directional_accuracy(y_true * 2 - 1, p_pred - 0.5)
    assert acc == 1.0

    brier = brier_score(y_true, p_pred)
    assert 0.0 <= brier <= 0.05

    ll = log_loss_score(y_true, p_pred)
    assert ll < 0.3


def test_pinball_loss():
    """Verify pinball loss for quantiles."""
    y = np.array([1.0, 2.0, 3.0])
    q = np.array([1.0, 2.0, 3.0])
    # Exact prediction: pinball loss = 0
    assert pinball_loss(y, q, tau=0.5) == 0.0

    # Over-predicting
    q_high = np.array([2.0, 3.0, 4.0])
    p_loss = pinball_loss(y, q_high, tau=0.9)
    assert p_loss > 0.0


def test_har_rv_model():
    """Verify HAR-RV model fit and predict."""
    rng = np.random.default_rng(99)
    rv_day = rng.uniform(0.0001, 0.0005, size=150)
    rv_week = rng.uniform(0.0002, 0.0006, size=150)
    rv_month = rng.uniform(0.0002, 0.0007, size=150)
    target = 0.5 * rv_day + 0.3 * rv_week + 0.2 * rv_month + rng.normal(0, 0.00005, size=150)

    model = HARRVModel(windows=(24, 168, 720), hac_lags=5)
    model.fit(rv_day, rv_week, rv_month, target)
    preds = model.predict(rv_day[:10], rv_week[:10], rv_month[:10])

    assert len(preds) == 10
    assert np.all(preds > 0.0)


def test_garch_models():
    """Verify GARCH and GJR-GARCH fit and predict on returns."""
    rng = np.random.default_rng(777)
    returns = rng.standard_t(df=5, size=300) * 0.01

    garch = GARCHModel(p=1, q=1, dist="t")
    garch.fit(returns)
    var_24h = garch.predict_24h_variance(horizon=24)
    assert var_24h > 0.0

    gjr = GJRGARCHModel(p=1, o=1, q=1, dist="t")
    gjr.fit(returns)
    var_gjr = gjr.predict_24h_variance(horizon=24)
    assert var_gjr > 0.0


def test_lightgbm_models():
    """Verify LightGBM volatility regressor and direction classifier."""
    rng = np.random.default_rng(888)
    X = rng.normal(size=(200, 10))
    y_log_vol = rng.normal(loc=-4.0, scale=0.5, size=200)
    y_dir = rng.integers(0, 2, size=200)

    vol_model = LightGBMVolModel(n_estimators=20, num_leaves=7)
    vol_model.fit(X, y_log_vol)
    pred_vol = vol_model.predict_log_vol(X[:5])
    assert len(pred_vol) == 5

    dir_model = LightGBMDirModel(n_estimators=20, num_leaves=7)
    dir_model.fit(X, y_dir)
    pred_probs = dir_model.predict_proba(X[:5])
    assert len(pred_probs) == 5
    assert np.all((pred_probs >= 0.0) & (pred_probs <= 1.0))
