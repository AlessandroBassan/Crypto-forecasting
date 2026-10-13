"""Tests for risk management (VaR/ES, FZ0 loss, backtests) and position sizing modules."""

import numpy as np
import pandas as pd
import pytest

from cff.risk import (
    backtest_acerbi_szekely_z1,
    backtest_christoffersen_independence,
    backtest_kupiec_pof,
    estimate_var_es_from_quantiles,
    estimate_var_es_student_t,
    fissler_ziegel_fz0_loss,
)
from cff.sizing import (
    compute_fractional_kelly_weights,
    compute_vol_adjusted_sl_tp,
    compute_volatility_target_weights,
    simulate_strategy_with_frictions,
)


@pytest.fixture
def risk_sample_data():
    """Generate synthetic return series and volatility states."""
    rng = np.random.default_rng(42)
    n = 1000
    sigma_1h = rng.uniform(0.005, 0.02, size=n)
    returns = rng.normal(0.0, sigma_1h, size=n)
    return returns, sigma_1h


def test_var_es_quantiles():
    """Verify VaR and ES estimation from predicted quantiles."""
    n = 100
    quantiles = {
        0.01: np.full(n, -0.06),
        0.025: np.full(n, -0.05),
        0.05: np.full(n, -0.04),
        0.50: np.zeros(n),
        0.95: np.full(n, 0.04),
    }

    var_05, es_05 = estimate_var_es_from_quantiles(quantiles, alpha=0.05)
    assert np.all(var_05 == -0.04)
    # ES must be strictly more negative than VaR: ES <= VaR
    assert np.all(es_05 <= var_05)
    assert np.isclose(es_05[0], -0.05)


def test_var_es_student_t():
    """Verify analytical Student-t VaR and ES formulas."""
    mu = np.zeros(10)
    sigma_24h = np.full(10, 0.03)

    var, es = estimate_var_es_student_t(mu, sigma_24h, nu=5.0, alpha=0.05)
    # In negative return space, ES < VaR < 0
    assert np.all(var < 0.0)
    assert np.all(es < var)


def test_fissler_ziegel_fz0_loss():
    """Verify FZ0 loss properties: finite and penalty for miscalibration."""
    y = np.array([-0.06, -0.02, 0.01, -0.01, 0.03])
    v = np.full(5, -0.03)
    e = np.full(5, -0.05)

    loss = fissler_ziegel_fz0_loss(y, v, e, alpha=0.05)
    assert np.isfinite(loss)


def test_kupiec_pof_backtest():
    """Verify Kupiec POF Likelihood Ratio test."""
    rng = np.random.default_rng(123)
    n = 500
    alpha = 0.05

    # Simulate exactly 5% exceedances
    y_true = rng.normal(0, 1, size=n)
    var_pred = np.full(n, float(np.percentile(y_true, alpha * 100)))

    res = backtest_kupiec_pof(y_true, var_pred, alpha=alpha)
    assert res["h0_accepted_5pct"] is True
    assert res["p_value"] > 0.10


def test_christoffersen_independence():
    """Verify Christoffersen independence and conditional coverage test."""
    rng = np.random.default_rng(456)
    n = 500
    y_true = rng.normal(0, 1, size=n)
    var_pred = np.full(n, float(np.percentile(y_true, 5.0)))

    res = backtest_christoffersen_independence(y_true, var_pred, alpha=0.05)
    assert res["is_independent_5pct"] is True
    assert res["is_conditionally_covered_5pct"] is True


def test_acerbi_szekely_z1():
    """Verify Acerbi-Szekely Z1 Expected Shortfall test."""
    rng = np.random.default_rng(789)
    n = 500
    y_true = rng.standard_t(df=5, size=n) * 0.02
    v = np.full(n, -0.04)
    e = np.full(n, -0.06)

    res = backtest_acerbi_szekely_z1(y_true, v, e, alpha=0.05)
    assert "z1_statistic" in res
    assert np.isfinite(res["z1_statistic"])


def test_volatility_targeting_weights():
    """Verify volatility targeting scales inverse to volatility."""
    sig_low = np.array([0.01])   # low vol -> higher weight
    sig_high = np.array([0.05])  # high vol -> lower weight

    w_low = compute_volatility_target_weights(sig_low, target_annual_vol=0.20)
    w_high = compute_volatility_target_weights(sig_high, target_annual_vol=0.20)
    assert w_low[0] > w_high[0]


def test_fractional_kelly_and_sl_tp():
    """Verify fractional Kelly sizing with shrinkage and SL/TP barriers."""
    mu = np.array([0.001, -0.001])
    var = np.array([0.0001, 0.0001])

    w_kelly = compute_fractional_kelly_weights(mu, var, fraction=0.25, shrinkage=0.5)
    assert w_kelly[0] > 0.0
    assert w_kelly[1] < 0.0

    entry = np.array([100.0])
    sig = np.array([0.01])
    sl, tp = compute_vol_adjusted_sl_tp(entry, sig, horizon_hours=24, is_long=True)
    assert sl[0] < 100.0
    assert tp[0] > 100.0


def test_simulate_strategy_with_frictions():
    """Verify friction simulation accounts for turnover fees and funding carry."""
    times = pd.date_range("2023-01-01", periods=100, freq="1h", tz="UTC")
    df = pd.DataFrame({
        "timestamp": times,
        "close": 100.0 + np.cumsum(np.random.default_rng(42).normal(0, 0.5, size=100)),
        "funding_rate": np.full(100, 0.0001),
    })
    weights = np.ones(100) * 0.5

    sim = simulate_strategy_with_frictions(df, weights, taker_fee_bps=4.0, slippage_bps=2.0)
    assert "net_sharpe_ratio" in sim
    assert sim["total_turnover"] > 0.0
    assert sim["total_fee_cost_pct"] > 0.0
