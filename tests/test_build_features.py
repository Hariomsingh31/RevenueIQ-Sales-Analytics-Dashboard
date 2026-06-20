"""
tests/test_build_features.py
-------------------------------
Unit tests for src/features/build_features.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import pytest

from src.features.build_features import (
    build_daily_revenue_series, add_lag_features, add_rolling_features,
    add_calendar_features, build_forecasting_features, build_product_features,
    build_customer_like_features, add_order_level_anomaly_features
)


@pytest.fixture
def sample_df():
    dates = pd.date_range("2024-01-01", periods=20, freq="D")
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "Order ID": [f"ORD{i}" for i in range(20)],
        "Date": dates,
        "Category": ["Electronics", "Books"] * 10,
        "Product Name": ["Mouse", "Notebook"] * 10,
        "Quantity": rng.integers(1, 5, 20),
        "Unit Price": [500.0] * 10 + [50.0] * 10,
        "Total Price": rng.integers(50, 2000, 20).astype(float),
        "Region": ["Asia", "Europe"] * 10,
        "Payment Method": ["UPI"] * 20,
    })


def test_build_daily_revenue_series_has_expected_columns(sample_df):
    ts = build_daily_revenue_series(sample_df)
    assert set(["Date", "Revenue", "Units", "Orders"]).issubset(ts.columns)
    assert len(ts) == 20  # one row per day, no gaps in this fixture


def test_add_lag_features_creates_correct_columns(sample_df):
    ts = build_daily_revenue_series(sample_df)
    ts_lagged = add_lag_features(ts, lags=(1, 3))
    assert "Revenue_lag_1" in ts_lagged.columns
    assert "Revenue_lag_3" in ts_lagged.columns
    # lag_1 at row i should equal Revenue at row i-1
    assert ts_lagged["Revenue_lag_1"].iloc[1] == ts_lagged["Revenue"].iloc[0]


def test_add_rolling_features_no_lookahead_leakage(sample_df):
    ts = build_daily_revenue_series(sample_df)
    ts_roll = add_rolling_features(ts, windows=(3,))
    # rolling mean at row i must only use data shifted by 1 (no current-day leakage)
    # so the first non-null value should appear no earlier than index 3
    first_valid_idx = ts_roll["Revenue_roll_mean_3"].first_valid_index()
    assert first_valid_idx >= 3


def test_add_calendar_features_adds_expected_columns(sample_df):
    ts = build_daily_revenue_series(sample_df)
    ts_cal = add_calendar_features(ts)
    for col in ["DayOfWeek", "Month", "Day", "WeekOfYear", "IsWeekend"]:
        assert col in ts_cal.columns


def test_build_forecasting_features_pipeline_runs(sample_df):
    ts = build_forecasting_features(sample_df)
    assert "Revenue" in ts.columns
    assert "Revenue_lag_1" in ts.columns
    assert "Revenue_roll_mean_7" in ts.columns


def test_build_product_features_aggregates_correctly(sample_df):
    pf = build_product_features(sample_df)
    assert set(pf["Product Name"].unique()) == {"Mouse", "Notebook"}
    assert "TotalRevenue" in pf.columns
    assert "RecencyDays" in pf.columns
    # No negative recency
    assert (pf["RecencyDays"] >= 0).all()


def test_build_customer_like_features_groups_by_region(sample_df):
    cf = build_customer_like_features(sample_df, group_col="Region")
    assert set(cf["Region"].unique()) == {"Asia", "Europe"}
    assert "AvgOrderValue" in cf.columns


def test_add_order_level_anomaly_features_zscore_present(sample_df):
    af = add_order_level_anomaly_features(sample_df)
    assert "PriceZScore" in af.columns
    assert af["PriceZScore"].notna().all()
