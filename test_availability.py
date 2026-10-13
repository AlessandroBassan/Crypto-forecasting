"""Tests verifying point-in-time availability and strict backward alignment."""

import pandas as pd
import pytest

from cff.data.availability import (
    LookaheadLeakageError,
    align_point_in_time,
    tag_availability,
    verify_point_in_time_integrity,
)


def test_tag_availability_bar_open_time():
    """Verify that bar open time tags availability at the close of the bar."""
    times = pd.date_range("2023-01-01 00:00", periods=5, freq="1h", tz="UTC")
    df = pd.DataFrame({"timestamp": times, "close": [10, 11, 12, 13, 14]})

    tagged = tag_availability(df, time_col="timestamp", is_bar_open_time=True, bar_duration_seconds=3600)

    # 00:00 bar open is only available at 01:00
    assert tagged.loc[0, "avail_ts"] == pd.Timestamp("2023-01-01 01:00", tz="UTC")
    assert tagged.loc[4, "avail_ts"] == pd.Timestamp("2023-01-01 05:00", tz="UTC")


def test_align_point_in_time_backward_only():
    """Verify that merge_asof backwards strictly pairs past information."""
    base_times = pd.date_range("2023-01-01 00:00", periods=24, freq="1h", tz="UTC")
    base_df = pd.DataFrame({"decision_ts": base_times})

    # Alternative data published every 8 hours: 00:00, 08:00, 16:00
    alt_times = pd.date_range("2023-01-01 00:00", periods=4, freq="8h", tz="UTC")
    alt_df = pd.DataFrame({
        "avail_ts": alt_times,
        "funding_rate": [0.0001, 0.0002, 0.0003, 0.0004],
    })

    merged = align_point_in_time(
        base_df,
        alt_df,
        base_time_col="decision_ts",
        alt_avail_col="avail_ts",
    )

    # At 07:00 UTC, the latest available funding rate is 00:00 (0.0001)
    bar_7 = merged[merged["decision_ts"] == pd.Timestamp("2023-01-01 07:00", tz="UTC")].iloc[0]
    assert bar_7["funding_rate"] == 0.0001
    assert bar_7["avail_ts"] == pd.Timestamp("2023-01-01 00:00", tz="UTC")

    # At 08:00 UTC, the newly settled funding rate (0.0002) becomes available
    bar_8 = merged[merged["decision_ts"] == pd.Timestamp("2023-01-01 08:00", tz="UTC")].iloc[0]
    assert bar_8["funding_rate"] == 0.0002
    assert bar_8["avail_ts"] == pd.Timestamp("2023-01-01 08:00", tz="UTC")


def test_leakage_detection_raises_error():
    """Verify that if a feature timestamp exceeds decision time, LookaheadLeakageError is raised."""
    base_times = pd.date_range("2023-01-01 00:00", periods=3, freq="1h", tz="UTC")

    # Intentionally corrupt alternative data with a future availability timestamp
    corrupted_df = pd.DataFrame({
        "decision_ts": base_times,
        "avail_ts": [
            pd.Timestamp("2023-01-01 00:00", tz="UTC"),
            pd.Timestamp("2023-01-01 02:00", tz="UTC"),  # Future leak at index 1!
            pd.Timestamp("2023-01-01 02:00", tz="UTC"),
        ],
    })

    with pytest.raises(LookaheadLeakageError):
        verify_point_in_time_integrity(
            corrupted_df,
            base_time_col="decision_ts",
            alt_avail_col="avail_ts",
        )
