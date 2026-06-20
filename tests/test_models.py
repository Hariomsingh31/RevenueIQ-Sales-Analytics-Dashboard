"""
tests/test_models.py
----------------------
Unit tests for the forecasting, segmentation, and anomaly detection models.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import pytest

from src.features.build_features import build_forecasting_features
from src.models.forecast_model import SalesForecaster
from src.models.segmentation_model import SegmentationModel, run_segmentation
from src.models.anomaly_model import AnomalyDetector


@pytest.fixture
def sample_df():
    dates = pd.date_range("2024-01-01", periods=120, freq="D")
    rng = np.random.default_rng(42)
    categories = ["Electronics", "Books", "Clothing"]
    products = {
        "Electronics": ["Mouse", "Keyboard"],
        "Books": ["Notebook"],
        "Clothing": ["T-Shirt", "Jacket"],
    }
    rows = []
    for i, date in enumerate(dates):
        for _ in range(rng.integers(2, 5)):
            cat = rng.choice(categories)
            prod = rng.choice(products[cat])
            qty = int(rng.integers(1, 6))
            price = float(rng.uniform(100, 2000))
            rows.append({
                "Order ID": f"ORD{len(rows)}",
                "Date": date,
                "Category": cat,
                "Product Name": prod,
                "Quantity": qty,
                "Unit Price": price,
                "Total Price": qty * price,
                "Region": rng.choice(["Asia", "Europe", "North America"]),
                "Payment Method": rng.choice(["UPI", "Credit Card"]),
            })
    return pd.DataFrame(rows)


class TestSalesForecaster:
    def test_fit_and_predict_shapes(self, sample_df):
        ts = build_forecasting_features(sample_df)
        forecaster = SalesForecaster()
        forecaster.fit(ts)
        assert forecaster.is_fitted

    def test_forecast_future_returns_correct_horizon(self, sample_df):
        ts = build_forecasting_features(sample_df)
        forecaster = SalesForecaster()
        forecaster.fit(ts)
        future = forecaster.forecast_future(ts, horizon=10)
        assert len(future) == 10
        assert "Date" in future.columns and "Revenue" in future.columns

    def test_forecast_values_are_non_negative(self, sample_df):
        ts = build_forecasting_features(sample_df)
        forecaster = SalesForecaster()
        forecaster.fit(ts)
        future = forecaster.forecast_future(ts, horizon=15)
        assert (future["Revenue"] >= 0).all()

    def test_evaluate_returns_expected_metric_keys(self, sample_df):
        ts = build_forecasting_features(sample_df)
        forecaster = SalesForecaster()
        metrics = forecaster.evaluate(ts, n_splits=2)
        assert set(["MAE", "MAPE", "R2", "n_splits"]).issubset(metrics.keys())

    def test_feature_importances_sum_or_shape(self, sample_df):
        ts = build_forecasting_features(sample_df)
        forecaster = SalesForecaster()
        forecaster.fit(ts)
        importances = forecaster.feature_importances()
        assert len(importances) == len(forecaster.feature_cols)
        assert "importance" in importances.columns


class TestSegmentationModel:
    def test_run_segmentation_product_level(self, sample_df):
        model, result = run_segmentation(sample_df, level="product")
        assert "Segment" in result.columns
        assert "Cluster" in result.columns
        assert model.k >= 2

    def test_run_segmentation_region_level(self, sample_df):
        model, result = run_segmentation(sample_df, level="region")
        assert "Segment" in result.columns
        assert len(result) == sample_df["Region"].nunique()

    def test_segment_ordering_is_monotonic_in_revenue(self, sample_df):
        model, result = run_segmentation(sample_df, level="product", k=3)
        means = result.groupby("SegmentRank")["TotalRevenue"].mean().sort_index()
        assert (means.diff().dropna() >= 0).all()

    def test_invalid_level_raises(self, sample_df):
        with pytest.raises(ValueError):
            run_segmentation(sample_df, level="customer")


class TestAnomalyDetector:
    def test_fit_predict_flags_some_anomalies(self, sample_df):
        detector = AnomalyDetector(contamination=0.05)
        result = detector.fit_predict(sample_df)
        assert "IsAnomaly" in result.columns
        assert "AnomalyScore" in result.columns
        assert result["IsAnomaly"].sum() > 0

    def test_anomaly_rate_roughly_matches_contamination(self, sample_df):
        detector = AnomalyDetector(contamination=0.05)
        result = detector.fit_predict(sample_df)
        rate = result["IsAnomaly"].mean()
        assert 0.0 < rate < 0.15  # loose bound, IsolationForest contamination is approximate

    def test_predict_after_fit_returns_scores(self, sample_df):
        detector = AnomalyDetector(contamination=0.05)
        detector.fit_predict(sample_df)
        result = detector.predict(sample_df.head(20))
        assert len(result) == 20
        assert "AnomalyScore" in result.columns
