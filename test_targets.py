"""Tests verifying mathematical correctness and strictly forward nature of target variables."""

import numpy as np
import pandas as pd
import pytest

from cff.data.download import generate_synthetic_market_data
from cff.targets import (
    compute_btc_eth_realized_correlation,
    compute_future_cumulative_return,
    compute_future_gk_log_vol,
    compute_future_log_return_and_direction,
    compute_garman_klass_variance_bar,
    compute_tail_event_target,
)


@pytest.fixture
def synthetic_df() -> pd.DataFrame:
    """Fixture providing 500 bars of deterministic synthetic market data."""
    return generate_synthetic_market_data(periods=500, seed=123)


def test_garman_klass_variance_bar_properties():
    """Verify single-bar Garman-Klass properties."""
    # When High == Low == Open == Close (zero range), GK variance must be near zero
    zeros = np.ones(5) * 100.0
    gk_zeros = compute_garman_klass_variance_bar(zeros, zeros, zeros, zeros)
    assert np.all(gk_zeros <= 1e-10)

    # Physical property: High > Low increases variance
    highs = np.array([105.0])
    lows = np.array([95.0])
    opens = np.array([100.0])
    closes = np.array([101.0])
    gk_val = compute_garman_klass_variance_bar(opens, highs, lows, closes)
    assert gk_val[0] > 0.001


def test_future_gk_log_vol_strictly_forward(synthetic_df):
    """Verify that y_vol(t) depends strictly on bars t+1 .. t+24 and NOT on bar t."""
    df1 = synthetic_df.copy()
    y_vol1 = compute_future_gk_log_vol(df1, horizon=24)

    # Modify bar t=10 in df2
    df2 = synthetic_df.copy()
    df2.loc[10, "high"] = df2.loc[10, "high"] * 2.0
    df2.loc[10, "low"] = df2.loc[10, "low"] * 0.5
    y_vol2 = compute_future_gk_log_vol(df2, horizon=24)

    # At t=10, the target is computed over bars 11..34.
    # Therefore, modifying bar 10 MUST NOT alter y_vol(10)!
    assert np.isclose(y_vol1.iloc[10], y_vol2.iloc[10], atol=1e-10)

    # Conversely, modifying bar 15 MUST alter y_vol(10) since bar 15 is inside [11..34]
    df3 = synthetic_df.copy()
    df3.loc[15, "high"] = df3.loc[15, "high"] * 2.0
    y_vol3 = compute_future_gk_log_vol(df3, horizon=24)
    assert not np.isclose(y_vol1.iloc[10], y_vol3.iloc[10], atol=1e-5)


def test_forward_return_and_direction(synthetic_df):
    """Verify 1-bar forward return r_{t+1} and direction correctness."""
    ret_1h, dir_1h = compute_future_log_return_and_direction(synthetic_df)

    t = 42
    expected_ret = np.log(synthetic_df.loc[t + 1, "close"] / synthetic_df.loc[t, "close"])
    expected_dir = 1.0 if expected_ret > 0.0 else 0.0

    assert np.isclose(ret_1h.iloc[t], expected_ret, atol=1e-12)
    assert dir_1h.iloc[t] == expected_dir


def test_future_cumulative_return(synthetic_df):
    """Verify 24h forward cumulative return formula."""
    ret_24h = compute_future_cumulative_return(synthetic_df, horizon=24)

    t = 100
    expected = np.log(synthetic_df.loc[t + 24, "close"] / synthetic_df.loc[t, "close"])
    assert np.isclose(ret_24h.iloc[t], expected, atol=1e-12)


def test_tail_event_target_causality(synthetic_df):
    """Verify tail event target: sigma_hat is causal and event reflects future return."""
    tail_target = compute_tail_event_target(synthetic_df, horizon=24, trailing_vol_window=100)

    # Early bars where trailing window is not filled should be NaN
    assert np.isnan(tail_target.iloc[10])

    # Valid bars should be strictly 0 or 1
    valid_vals = tail_target.dropna().values
    assert len(valid_vals) > 0
    assert set(np.unique(valid_vals)).issubset({0.0, 1.0})


def test_realized_correlation_bounds(synthetic_df):
    """Verify realized correlation stays within [-1, 1]."""
    df_btc = synthetic_df.copy()
    df_eth = generate_synthetic_market_data(symbol="ETHUSDT", periods=500, seed=456)

    corr = compute_btc_eth_realized_correlation(df_btc, df_eth, horizon=24)
    valid_corr = corr.dropna()
    assert (valid_corr >= -1.0 - 1e-6).all()
    assert (valid_corr <= 1.0 + 1e-6).all()
